/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#include "NNLayers.h"
#include "frontends/standard/StandardLayer.h"
#include "global.h"
#include "units/standard/SysArray.h"
#include "units/standard/VectorUnit.h"
#include <frontends/standard/StandardArch.h>

#include <functional>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <unordered_set>

using namespace frontend::standard;

JobList createSAJobs(int m, int k, int n, int num_jobs, int n_cores = 1) {
  JobList jobs;
  
  // Tensor parallelism: split N dimension across cores
  int core_n = n / n_cores;
  
  // Per-core task counters for independent scheduling
  static std::vector<int> core_task_counters(n_cores, 0);
  for (int core = 0; core < n_cores; ++core) {
    for (int job = 0; job < num_jobs; ++job) {
      auto sys_job = new SystolicArray::SysArrayJob(m, k, core_n);
      sys_job->core_id = core;  // Assign to specific core for parallel execution
      sys_job->task_idx = core_task_counters[core]++;  // Sequential task ID per core
      jobs.push_back(sys_job);
    }
  }
  
  return jobs;
}

static const std::vector<std::pair<VectorUnit::VPUPhase, int>> softmax_phases =
    {{VectorUnit::VPUPhase::BROADCAST, 1}, {VectorUnit::VPUPhase::REDUCE, 1}, {VectorUnit::VPUPhase::BROADCAST, 1}};

namespace {
  struct CoreChunk {
    int core = 0;
    int par_dim = 0;
  };

  int elems_to_bytes(int elems) {
    return std::max(elems, 0) * data_type_width * batch_size;
  }

  std::vector<CoreChunk> split_parallel_to_cores(int total_parallel, int n_cores) {
    std::vector<CoreChunk> chunks;
    int cores = std::max(n_cores, 1);
    int base = total_parallel / cores;
    int rem = total_parallel % cores;
    for (int core = 0; core < cores; ++core) {
      int par = base + (core < rem ? 1 : 0);
      if (par > 0) {
        chunks.push_back({core, par});
      }
    }
    return chunks;
  }

  SystolicArray::CgraJob *make_cgra_job(int core_id,
                                        int lin_dim,
                                        int par_dim,
                                        SystolicArray::CgraOp op,
                                        int read_elems,
                                        int write_elems,
                                        bool is_prebuffered,
                                        const ArchConfig &a_config,
                                        int op_latency_mul = 1,
                                        int ii_mul = 1,
                                        int pipe_mul = 1) {
    auto *job = new SystolicArray::CgraJob(
        lin_dim,
        par_dim,
        op,
        std::max(op_latency_mul, 1),
        std::max(a_config.cgra_default_ii * std::max(ii_mul, 1), 1),
        std::max(a_config.cgra_default_pipeline * std::max(pipe_mul, 1), 1),
        is_prebuffered ? 0 : elems_to_bytes(read_elems),
        elems_to_bytes(write_elems),
        is_prebuffered);
    job->core_id = core_id;
    return job;
  }

  JobPair make_activation_jobs(const ArchConfig &a_config, int lin_dim, int par_dim, bool is_prebuffered) {
    if (!a_config.reuse_sa_for_vu) {
      auto *job = new VectorUnit::VecUnitJob(lin_dim, par_dim, is_prebuffered, {{VectorUnit::VPUPhase::BROADCAST, 1}});
      return {{job}, {job}};
    }

    JobList heads;
    JobList tails;
    for (const auto &chunk: split_parallel_to_cores(par_dim, a_config.n_cores)) {
      auto *job = make_cgra_job(chunk.core,
                                lin_dim,
                                chunk.par_dim,
                                SystolicArray::CGRA_FMA,
                                lin_dim * chunk.par_dim,
                                lin_dim * chunk.par_dim,
                                is_prebuffered,
                                a_config,
                                1,
                                1,
                                1);
      heads.push_back(job);
      tails.push_back(job);
    }
    return {heads, tails};
  }

  JobPair make_softmax_jobs(const ArchConfig &a_config, int M, int heads, bool is_prebuffered) {
    int spl = 1;
    int Mp = M * heads;
    if (heads * M * M * data_type_width * batch_size > buffer_size_bytes || Mp > 1024) {
      spl = std::max(div_ru(heads * M * M * data_type_width * batch_size, buffer_size_bytes), div_ru(Mp, 1024));
      if (spl > Mp) {
        std::cerr << "Can't split this enough to fit inside buffer." << std::endl;
        throw std::exception();
      }
      Mp /= spl;
    }
    std::cout << "Splitting by " << spl << std::endl;

    if (!a_config.reuse_sa_for_vu) {
      int n_jobs = div_ru(div_ru(M * heads, Mp), n_vpus);
      JobList softmax_layer;
      for (int i = 0; i < n_jobs; ++i) {
        softmax_layer.push_back(new VectorUnit::VecUnitJob(M, Mp, is_prebuffered, softmax_phases));
      }
      return {softmax_layer, softmax_layer};
    }

    int n_jobs = std::max(div_ru(M * heads, Mp), 1);
    JobList all_heads;
    JobList all_tails;
    for (int job_idx = 0; job_idx < n_jobs; ++job_idx) {
      for (const auto &chunk: split_parallel_to_cores(Mp, a_config.n_cores)) {
        int elems = M * chunk.par_dim;
        auto *max_reduce = make_cgra_job(chunk.core,
                                         M,
                                         chunk.par_dim,
                                         SystolicArray::CGRA_REDUCE,
                                         elems,
                                         chunk.par_dim,
                                         is_prebuffered,
                                         a_config,
                                         1,
                                         1,
                                         1);
        auto *exp_approx = make_cgra_job(chunk.core,
                                         M,
                                         chunk.par_dim,
                                         SystolicArray::CGRA_EXP_APPROX,
                                         elems + chunk.par_dim,
                                         elems,
                                         false,
                                         a_config,
                                         2,
                                         2,
                                         2);
        auto *sum_reduce = make_cgra_job(chunk.core,
                                         M,
                                         chunk.par_dim,
                                         SystolicArray::CGRA_REDUCE,
                                         elems,
                                         chunk.par_dim,
                                         false,
                                         a_config,
                                         1,
                                         1,
                                         1);
        auto *normalize = make_cgra_job(chunk.core,
                                        M,
                                        chunk.par_dim,
                                        SystolicArray::CGRA_NORMALIZE,
                                        elems + chunk.par_dim,
                                        elems,
                                        false,
                                        a_config,
                                        2,
                                        1,
                                        2);
        max_reduce->add_child(exp_approx);
        exp_approx->add_child(sum_reduce);
        sum_reduce->add_child(normalize);
        all_heads.push_back(max_reduce);
        all_tails.push_back(normalize);
      }
    }
    return {all_heads, all_tails};
  }

  JobPair make_layernorm_jobs(const ArchConfig &a_config, int lin_dim, int par_dim) {
    if (!a_config.reuse_sa_for_vu) {
      JobList jl;
      int par_acc = par_dim;
      int dec_amt = std::max(buffer_size_bytes / data_type_width / lin_dim, 1);
      while (par_acc > 0) {
        jl.push_back(new VectorUnit::VecUnitJob(lin_dim, std::min(dec_amt, par_acc), false,
                                                {{VectorUnit::VPUPhase::REDUCE, 1}, {VectorUnit::VPUPhase::REDUCE, 4}, {VectorUnit::VPUPhase::BROADCAST, 1}}));
        par_acc -= dec_amt;
      }
      return {jl, jl};
    }

    JobList all_heads;
    JobList all_tails;
    int par_acc = par_dim;
    int dec_amt = std::max(buffer_size_bytes / data_type_width / lin_dim, 1);
    while (par_acc > 0) {
      int par_chunk = std::min(dec_amt, par_acc);
      for (const auto &chunk: split_parallel_to_cores(par_chunk, a_config.n_cores)) {
        int elems = lin_dim * chunk.par_dim;
        auto *mean_reduce = make_cgra_job(chunk.core,
                                          lin_dim,
                                          chunk.par_dim,
                                          SystolicArray::CGRA_REDUCE,
                                          elems,
                                          chunk.par_dim,
                                          false,
                                          a_config,
                                          1,
                                          1,
                                          1);
        auto *var_reduce = make_cgra_job(chunk.core,
                                         lin_dim,
                                         chunk.par_dim,
                                         SystolicArray::CGRA_REDUCE,
                                         elems + chunk.par_dim,
                                         chunk.par_dim,
                                         false,
                                         a_config,
                                         2,
                                         1,
                                         2);
        auto *normalize = make_cgra_job(chunk.core,
                                        lin_dim,
                                        chunk.par_dim,
                                        SystolicArray::CGRA_NORMALIZE,
                                        elems + chunk.par_dim,
                                        elems,
                                        false,
                                        a_config,
                                        2,
                                        1,
                                        2);
        mean_reduce->add_child(var_reduce);
        var_reduce->add_child(normalize);
        all_heads.push_back(mean_reduce);
        all_tails.push_back(normalize);
      }
      par_acc -= par_chunk;
    }
    return {all_heads, all_tails};
  }

  std::string make_layer_tag(const LayerConfig &cfg, int idx) {
    std::string tag = std::to_string(idx) + "_" + cfg.layer_type;
    for (const auto &d: cfg.dimensions) {
      tag += "x" + std::to_string(d);
    }
    return tag;
  }

  void annotate_layer_jobs(const JobPair &jp, int layer_idx, const std::string &layer_tag) {
    std::vector<Job *> stack;
    std::unordered_set<Job *> visited;
    for (auto *j: jp.first) {
      if (j != nullptr) {
        stack.push_back(j);
      }
    }
    for (auto *j: jp.second) {
      if (j != nullptr) {
        stack.push_back(j);
      }
    }
    while (!stack.empty()) {
      Job *cur = stack.back();
      stack.pop_back();
      if (cur == nullptr || visited.find(cur) != visited.end()) {
        continue;
      }
      visited.insert(cur);
      cur->layer_idx = layer_idx;
      cur->layer_name = layer_tag;
      for (auto *child: cur->children) {
        if (child != nullptr && visited.find(child) == visited.end()) {
          stack.push_back(child);
        }
      }
    }
  }
}// namespace

using JobCreate_f = std::function<JobPair(const ArchConfig &, const LayerConfig &)>;

JobPair Matmul(const ArchConfig &a_config, const LayerConfig &l_config) {
  int M, K, N;
  if (l_config.dimensions.size() == 3) {
    M = l_config.dimensions[0];
    K = l_config.dimensions[1];
    N = l_config.dimensions[2];
  } else if (l_config.dimensions.size() == 4) {
    M = l_config.dimensions[1];
    K = l_config.dimensions[2];
    N = l_config.dimensions[3] * l_config.dimensions[0];
  } else {
    std::cerr << "MM Not expecting " << l_config.dimensions.size() << " dimensions..." << std::endl;
    throw std::exception();
  }

  if (a_config.ws) {
    JobList jl;
    
    // WS: Split N first, then check buffer constraints per core
    int core_n = N / a_config.n_cores;
    std::cout << "WS N-splitting: " << N << " output channels across " << a_config.n_cores << " cores" << std::endl;
    
    static std::vector<int> core_task_counters(a_config.n_cores, 0);
    for (int core = 0; core < a_config.n_cores; ++core) {
      int required_buff_sz_per_core = (M * core_n + M * std::min(K, a_config.sa_sz_allo)) * batch_size * data_type_width;
      bool core_is_bufferable = required_buff_sz_per_core <= buffer_size_bytes;
      
      if (core_is_bufferable) {
        auto job = new SystolicArray::SysArrayJob(M, K, core_n);
        job->core_id = core;
        job->task_idx = core_task_counters[core]++;
        jl.push_back(job);
        std::cout << "  Core " << core << ": " << core_n << " out dim - bufferable" << std::endl;
      } else {
        // Fallback: sequential jobs if buffer too small
        std::cout << "  Core " << core << ": " << core_n << " out dim - not bufferable, using sequential execution" << std::endl;
        float new_N_f = (float) buffer_size_bytes / float(data_type_width * M * batch_size);
        int N_per_job = std::max(1, (int) std::floor(new_N_f));
        int num_sequential_jobs = (core_n + N_per_job - 1) / N_per_job;
        
        for (int i = 0; i < num_sequential_jobs; ++i) {
          int remaining_N = core_n - i * N_per_job;
          int current_N = std::min(N_per_job, remaining_N);
          auto job = new SystolicArray::SysArrayJob(M, K, current_N);
          job->core_id = core;
          job->task_idx = core_task_counters[core]++;
          jl.push_back(job);
        }
      }
    }
    return {jl, jl};
  } else {
    // OS: Create sequential jobs per core
    int num_jobs = std::max(1, M / a_config.sa_sz_allo);
    JobList matmul_layers = createSAJobs(a_config.sa_sz_allo,
                                         K,
                                         N, num_jobs, a_config.n_cores);
    return {matmul_layers, matmul_layers};
  }
}


JobPair Conv(const ArchConfig &a_config, const LayerConfig &l_config) {
  int M, K, N;
  
  // Conv layer parameters: batch, input_channels, input_height, input_width, output_channels, kernel_size, stride, padding
  if (l_config.dimensions.size() < 5) {
    std::cerr << "Conv expecting at least 5 dimensions: batch, input_channels, input_height, input_width, output_channels" << std::endl;
    throw std::exception();
  }
  
  int batch = l_config.dimensions[0];
  int input_channels = l_config.dimensions[1]; 
  int input_height = l_config.dimensions[2];
  int input_width = l_config.dimensions[3];
  int output_channels = l_config.dimensions[4];
  
  // Default values if not specified
  int kernel_size = (l_config.dimensions.size() > 5) ? l_config.dimensions[5] : 3;
  int stride = (l_config.dimensions.size() > 6) ? l_config.dimensions[6] : 1;
  int padding = (l_config.dimensions.size() > 7) ? l_config.dimensions[7] : 1;
  
  
  // Calculate output dimensions after convolution
  int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
  int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
  
  // Convert to GEMM dimensions (im2col approach):
  // M = batch * output_height * output_width (number of output spatial positions)
  // K = input_channels * kernel_size * kernel_size (input channels * kernel area) 
  // N = output_channels (number of filters)
  M = batch * output_height * output_width;
  K = input_channels * kernel_size * kernel_size;
  N = output_channels;
  
  std::cout << "Conv2GEMM: batch=" << batch << ", in_ch=" << input_channels << ", in_h=" << input_height << ", in_w=" << input_width << std::endl;
  std::cout << "           out_ch=" << output_channels << ", kernel=" << kernel_size << ", stride=" << stride << ", padding=" << padding << std::endl;
  std::cout << "           out_h=" << output_height << ", out_w=" << output_width << std::endl;
  std::cout << "           GEMM dimensions: M=" << M << ", K=" << K << ", N=" << N << std::endl;

  if (a_config.ws) {
    JobList jl;
    
  // First split N by number of cores (assume remainder is 0)
    int core_n = N / a_config.n_cores;
    std::cout << "Conv WS N-splitting: " << N << " output channels across " << a_config.n_cores << " cores" << std::endl;
    
    static std::vector<int> core_task_counters(a_config.n_cores, 0);  // Per-core task counters
    for (int core = 0; core < a_config.n_cores; ++core) {
      // Check if this core's portion fits in buffer
      int required_buff_sz_per_core = (M * core_n + M * std::min(K, a_config.sa_sz_allo)) * batch_size * data_type_width;
      bool core_is_bufferable = required_buff_sz_per_core <= buffer_size_bytes;
      
      if (core_is_bufferable) {
        auto job = new SystolicArray::SysArrayJob(M, K, core_n);
        job->core_id = core;  // Assign core ID for parallel scheduling
        job->task_idx = core_task_counters[core]++;  // Sequential task ID per core (0,1,2,3...)
        jl.push_back(job);
        std::cout << "  Core " << core << ": " << core_n << " out dim - bufferable (core_id=" << core << ")" << std::endl;
      } else {
        // Sequential execution for this core's portion
        std::cout << "  Core " << core << ": " << core_n << " out dim - not bufferable, using sequential execution" << std::endl;
        float new_N_f = (float) buffer_size_bytes / float(data_type_width * M * batch_size);
        int N_per_job = std::max(1, (int) std::floor(new_N_f));
        int num_sequential_jobs = (core_n + N_per_job - 1) / N_per_job;
        
        for (int i = 0; i < num_sequential_jobs; ++i) {
          int remaining_N = core_n - i * N_per_job;
          int current_N = std::min(N_per_job, remaining_N);
          auto job = new SystolicArray::SysArrayJob(M, K, current_N);
          job->core_id = core;
          job->task_idx = core_task_counters[core]++;
          jl.push_back(job);
        }
      }
    }
    return {jl, jl};
  } else {
    // OS: Create sequential jobs per core  
    int num_jobs = std::max(1, M / a_config.sa_sz_allo);
    JobList matmul_layers = createSAJobs(a_config.sa_sz_allo,
                                         K,
                                         N, num_jobs, a_config.n_cores);
    return {matmul_layers, matmul_layers};
  }
}


JobPair MatmulAct(const ArchConfig &a_config, const LayerConfig &l_config) {
  // Parse matmul dimensions from layer config
  int M, K, N;
  if (l_config.dimensions.size() == 3) {
    M = l_config.dimensions[0];
    K = l_config.dimensions[1];
    N = l_config.dimensions[2];
  } else if (l_config.dimensions.size() == 4) {
    M = l_config.dimensions[1];
    K = l_config.dimensions[2];
    N = l_config.dimensions[3] * l_config.dimensions[0];
  } else {
    std::cerr << "MA Not expecting " << l_config.dimensions.size() << " dimensions..." << std::endl;
    throw std::exception();
  }

  if (a_config.ws) {
    int num_jobs = std::max(1, int(std::ceil(float(K) / a_config.sa_sz_allo)));

    JobList matmul_layers = createSAJobs(M,
                                         a_config.sa_sz_allo,
                                         N, num_jobs);
    JobPair act_layer = make_activation_jobs(a_config, 1, M * K, true);

    connectJobLists(matmul_layers, act_layer.first);

    return {matmul_layers, act_layer.second};
  } else {
    int num_jobs = std::max(1, M / a_config.sa_sz_allo);
    JobList matmul_layers = createSAJobs(a_config.sa_sz_allo,
                                         K,
                                         N, num_jobs);
    JobPair act_layer = make_activation_jobs(a_config, 1, M * K, true);

    connectJobLists(matmul_layers, act_layer.first);

    return {matmul_layers, act_layer.second};
  }
}

JobPair ActMatmul(const ArchConfig &a_config, const LayerConfig &l_config) {
  // Parse matmul dimensions from layer config
  int M, K, N;
  if (l_config.dimensions.size() == 3) {
    M = l_config.dimensions[0];
    K = l_config.dimensions[1];
    N = l_config.dimensions[2];
  } else if (l_config.dimensions.size() == 4) {
    M = l_config.dimensions[1];
    K = l_config.dimensions[2];
    N = l_config.dimensions[3] * l_config.dimensions[0];
  } else {
    std::cerr << "AM Not expecting " << l_config.dimensions.size() << " dimensions..." << std::endl;
    throw std::exception();
  }

  // Create activation layer job
  JobPair act_layer = make_activation_jobs(a_config, 1, M * K, true);

  if (a_config.ws) {
    int num_jobs = std::max(1, K / a_config.sa_sz_allo);
    JobList matmul_layers = createSAJobs(M,
                                         a_config.sa_sz_allo,
                                         N, num_jobs);
    connectJobLists(act_layer.second, matmul_layers);

    return {act_layer.first, matmul_layers};
  } else {
    int num_jobs = std::max(1, M / a_config.sa_sz_allo);
    JobList matmul_layers = createSAJobs(a_config.sa_sz_allo,
                                         K,
                                         N, num_jobs);
    connectJobLists(act_layer.second, matmul_layers);

    return {act_layer.first, matmul_layers};
  }
}

JobPair LayerNorm(const ArchConfig &a_config, const LayerConfig &l_config) {
  // Parse dimensions for layernorm operation
  int lin_dim, par_dim = 1;
  switch (l_config.dimensions.size()) {
    case 1:
      lin_dim = l_config.dimensions[0];
      break;
    case 2:
      par_dim = l_config.dimensions[0];
      lin_dim = l_config.dimensions[1];
      break;
    case 3:
      par_dim = l_config.dimensions[0] * l_config.dimensions[1];
      lin_dim = l_config.dimensions[2] / l_config.dimensions[0];
      if (l_config.dimensions[2] % l_config.dimensions[0] != 0) {
        std::cerr << "linear dimension is not divisible by group size in layernorm..." << std::endl;
        throw std::exception();
      }
      break;
    default:
      std::cout << "Unexpected number of dimensions received in LayerNorm" << std::endl;
      throw std::exception();
  }
  return make_layernorm_jobs(a_config, lin_dim, par_dim);
}

JobPair Activation(const ArchConfig &a_config, const LayerConfig &l_config) {
  int sz = 1;
  for (const auto &dim: l_config.dimensions) sz *= dim;

  return make_activation_jobs(a_config, 1, sz, false);
}

JobPair Softmax(const ArchConfig &a_config, const LayerConfig &l_config) {
  int M;
  int heads = 1;
  if (l_config.dimensions.size() == 1) {
    M = l_config.dimensions[0];
  } else if (l_config.dimensions.size() == 2) {
    heads = l_config.dimensions[0];
    M = l_config.dimensions[1];
  } else {
    std::cerr << "SM Not expecting " << l_config.dimensions.size() << " dimensions..." << std::endl;
    throw std::exception();
  }

  return make_softmax_jobs(a_config, M, heads, false);
}

JobPair SelfAttention(const ArchConfig &a_config, const LayerConfig &l_config) {
  // Parse self-attention dimensions (batch, seq_len, hidden_dim, n_heads)
  int M, K, N;
  int n_heads = 1;
  if (l_config.dimensions.size() == 3) {
    M = l_config.dimensions[0];
    K = l_config.dimensions[1];
    N = l_config.dimensions[2];
  } else if (l_config.dimensions.size() == 4) {
    n_heads = l_config.dimensions[0];
    M = l_config.dimensions[1];
    K = l_config.dimensions[2];
    N = l_config.dimensions[3];
  } else {
    std::cerr << "SA Not expecting " << l_config.dimensions.size() << " dimensions..." << std::endl;
    throw std::exception();
  }

  if (a_config.ws) {
    // Create K, Q, V projection matrices
    auto K_proj = Matmul(a_config, LayerConfig("Matmul", {M, K, N}));
    auto Q_proj = Matmul(a_config, LayerConfig("Matmul", {M, K, N}));
    auto V_proj = Matmul(a_config, LayerConfig("Matmul", {M, K, N}));
    
    // Attention mechanism: Q*K^T and softmax(QK^T)*V
    auto Dot1 = Matmul(a_config, LayerConfig("Matmul", {n_heads, M, N / n_heads, M}));
    auto Dot2 = Matmul(a_config, LayerConfig("Matmul", {n_heads, M, M, N / n_heads}));
    auto O_proj = Matmul(a_config, LayerConfig("Matmul", {n_heads, M, N / n_heads, K}));
    auto softmax_layer = make_softmax_jobs(a_config, M, n_heads, false);

    // Chain operations: K->Q->V->QK^T->softmax->attention*V->output
    connectJobs(K_proj, Q_proj);
    connectJobs(Q_proj, V_proj);
    connectJobs(V_proj, Dot1);
    connectJobs(Dot1, softmax_layer);
    connectJobs(softmax_layer, Dot2);
    connectJobs(Dot2, O_proj);
    return {K_proj.first, O_proj.second};
  } else {
    int num_jobs = std::max(1, M / a_config.sa_sz_allo);
    JobList K_proj = createSAJobs(a_config.sa_sz_allo,
                                  K,
                                  N, num_jobs);
    JobList Q_proj = createSAJobs(a_config.sa_sz_allo,
                                  K,
                                  N, num_jobs);
    JobList V_proj = createSAJobs(a_config.sa_sz_allo,
                                  K,
                                  N, num_jobs);
    JobList Dot1 = createSAJobs(a_config.sa_sz_allo,
                                K,// / m_config.n_heads,
                                M, num_jobs);
    JobList Dot2 = createSAJobs(a_config.sa_sz_allo,
                                M,
                                N, num_jobs);/// m_config.n_heads
    JobList O_proj = createSAJobs(a_config.sa_sz_allo,
                                  K,
                                  N, num_jobs);

    JobPair softmax_layer = make_softmax_jobs(a_config, M, 1, true);
    connectJobLists(K_proj, Q_proj);
    connectJobLists(Q_proj, Dot1);
    connectJobLists(Dot1, softmax_layer.first);
    connectJobLists(softmax_layer.second, V_proj);
    connectJobLists(V_proj, Dot2);
    connectJobLists(Dot2, O_proj);


    return {K_proj, O_proj};
  }
}

JobPair MultiHeadSelfAttention(const ArchConfig &a_config, const LayerConfig &l_config) {
  JobList all_heads, all_tails;

  int N = ceil(static_cast<double>(n_heads) / a_config.n_cores);
  std::vector<JobPair> head_jobs;
  for (int i = 0; i < N; ++i) {
    head_jobs.push_back(SelfAttention(a_config, l_config));
  }
  for (int i = 0; i < N - 1; ++i) {
    connectJobLists(head_jobs[i].second, head_jobs[i + 1].first);
  }
  return {head_jobs.front().first, head_jobs.back().second};
}


JobCreate_f getLayerLambda(const std::string &layer_type) {
  if (layer_type == "Matmul")
    return Matmul;
  if (layer_type == "Conv")
    return Conv;
  if (layer_type == "MatmulAct")
    return MatmulAct;
  if (layer_type == "Softmax")
    return Softmax;
  if (layer_type == "Activation" || layer_type == "ReLU")
    return Activation;
  // BatchNorm is treated as a vector activation-style pass in this frontend.
  if (layer_type == "BatchNorm")
    return Activation;
  if (layer_type == "LayerNorm")
    return LayerNorm;
  if (layer_type == "SelfAttention")
    return SelfAttention;
  if (layer_type == "MultiHeadSelfAttention")
    return MultiHeadSelfAttention;
  throw std::runtime_error("Unknown layer type: " + layer_type);
}

ArchConfig frontend::standard::arch_config;

std::vector<JobPair> StandardLayer::make_layers(const std::vector<LayerConfig> &layer_configs) const {
  std::vector<JobPair> model_heads;
  JobList jp;
  for (int m = 0; m < model_parallelism; ++m) {
    std::vector<JobPair> lists;
    int layer_idx = 0;
    for (const auto &layer_config: layer_configs) {
      auto layer_f = getLayerLambda(layer_config.layer_type);
      JobPair layer_jobs = layer_f(arch_config, layer_config);
      int global_layer_idx = m * static_cast<int>(layer_configs.size()) + layer_idx;
      annotate_layer_jobs(layer_jobs, global_layer_idx, make_layer_tag(layer_config, global_layer_idx));
      lists.push_back(layer_jobs);
      layer_idx++;
    }
    std::cout << "list size: " << lists.size() << std::endl;
    for (int i = 1; i < lists.size(); ++i) {
      connectJobLists(lists[i - 1].second, lists[i].first);
      std::cout << "connect " << i << " to " << (i - 1) << std::endl;
    }
    if (m == 0 || do_par) {
      model_heads.push_back(lists[0]);
      jp = lists.back().second;
    } else {
      connectJobLists(jp, lists[0].first);
      jp = lists[0].second;
    }
  }

  return model_heads;
}
