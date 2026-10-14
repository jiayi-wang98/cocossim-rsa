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
#include <climits>
#include <cstdint>

using namespace frontend::standard;

// Split n into at most `parts` near-equal positive pieces (the first n % parts pieces get one more).
static std::vector<int> split_even(int n, int parts) {
  std::vector<int> out;
  parts = std::max(1, std::min(parts, n));
  for (int i = 0; i < parts; ++i) out.push_back(n / parts + (i < n % parts ? 1 : 0));
  return out;
}

JobList createSAJobs(int m, int k, int n, int num_jobs, int n_cores) {
  JobList jobs;
  // Tensor parallelism: split N dimension across cores
  auto core_ns = split_even(n, n_cores);
  for (int core = 0; core < (int) core_ns.size(); ++core) {
    for (int job = 0; job < num_jobs; ++job) {
      auto sys_job = new SystolicArray::SysArrayJob(m, k, core_ns[core]);
      sys_job->core_id = core;  // Assign to specific core for parallel execution
      jobs.push_back(sys_job);
    }
  }
  return jobs;
}

// Vector work of `par` independent rows of length `lin`, split across the VUs (rows per job capped at
// max_par, e.g. by the on-chip buffer). Jobs carry no core id, so any idle VU picks them up.
static JobList createVUJobs(int lin, int par, bool is_prebuffered,
                            const std::vector<std::pair<VectorUnit::VPUPhase, int>> &phases,
                            int n_cores, int max_par = INT32_MAX) {
  JobList jobs;
  int per_job = std::max(1, std::min(max_par, (par + n_cores - 1) / n_cores));
  for (int done = 0; done < par; done += per_job) {
    jobs.push_back(new VectorUnit::VecUnitJob(lin, std::min(per_job, par - done), is_prebuffered, phases));
  }
  return jobs;
}

// TPUv3-fitted WS GEMM partitioning (-gemm_sched 1). These are the scheduling rules of the published COCOSSim
// TPUv3 validation, which simulated one MXU doing one share of the work; here every share runs on its own core.
//  - output and one input slice fit the buffer, no forced split: one job on whichever SA is free (the TPU often
//    leaves the other MXUs idle on such GEMMs); a Conv with K > 2048 is instead split 4 ways along K
//  - otherwise N is split over split_by cores (all cores if it splits into at least n_cores / 2 buffer-sized
//    chunks, else one core per chunk), each core walking its share in buffer-sized chunks
//  - force_split overrides split_by (used by SelfAttention, as in the validated model)
static JobList tpu_gemm_jobs(const ArchConfig &a_config, int M, int K, int N, int force_split, bool is_conv) {
  const int sz = a_config.sa_sz_allo, ncore = std::max(1, a_config.n_cores);
  int64_t need = (int64_t(M) * N + int64_t(M) * std::min(K, sz)) * batch_size * data_type_width;
  JobList jl;
  if (need <= buffer_size_bytes && force_split == -1) {
    int ksplit = (is_conv && K > 2048) ? std::min(4, ncore) : 1;
    for (int c = 0; c < ksplit; ++c) {
      auto *job = new SystolicArray::SysArrayJob(M, K / ksplit, N);
      job->core_id = ksplit > 1 ? c : -1;
      jl.push_back(job);
    }
    return jl;
  }
  int n_max = std::max(1, (int) std::floor(double(buffer_size_bytes) / double(data_type_width * M * batch_size)) - sz);
  int splitability = div_ru(N, n_max);
  int split_by = force_split != -1 ? force_split : (splitability >= ncore / 2 ? ncore : splitability);
  auto parts = split_even(N, std::max(1, std::min(split_by, ncore)));
  for (int c = 0; c < (int) parts.size(); ++c) {
    for (int done = 0; done < parts[c]; done += n_max) {
      auto *job = new SystolicArray::SysArrayJob(M, K, std::min(n_max, parts[c] - done));
      job->core_id = c;
      jl.push_back(job);
    }
  }
  return jl;
}

static const std::vector<std::pair<VectorUnit::VPUPhase, int>> softmax_phases =
    {{VectorUnit::VPUPhase::BROADCAST, 1}, {VectorUnit::VPUPhase::REDUCE, 1}, {VectorUnit::VPUPhase::BROADCAST, 1}};

using JobCreate_f = std::function<JobPair(const ArchConfig &, const LayerConfig &)>;

JobPair MatmulSplit(const ArchConfig &a_config, const LayerConfig &l_config, int force_split);
JobPair Matmul(const ArchConfig &a_config, const LayerConfig &l_config) { return MatmulSplit(a_config, l_config, -1); }

JobPair MatmulSplit(const ArchConfig &a_config, const LayerConfig &l_config, int force_split) {
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

  if (a_config.ws && a_config.gemm_sched == 1) {
    JobList jl = tpu_gemm_jobs(a_config, M, K, N, force_split, false);
    return {jl, jl};
  }
  if (a_config.ws) {
    JobList jl;
    
    // WS: Split N first, then check buffer constraints per core
    auto core_ns = split_even(N, a_config.n_cores);
    std::cout << "WS N-splitting: " << N << " output channels across " << core_ns.size() << " cores" << std::endl;

    for (int core = 0; core < (int) core_ns.size(); ++core) {
      int core_n = core_ns[core];
      int required_buff_sz_per_core = (M * core_n + M * std::min(K, a_config.sa_sz_allo)) * batch_size * data_type_width;
      bool core_is_bufferable = required_buff_sz_per_core <= buffer_size_bytes;
      
      if (core_is_bufferable) {
        auto job = new SystolicArray::SysArrayJob(M, K, core_n);
        job->core_id = core;
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
  // A 3-dim Conv is already in im2col GEMM form (M K N), as in the TPU validation workloads.
  if (l_config.dimensions.size() == 3) {
    M = l_config.dimensions[0];
    K = l_config.dimensions[1];
    N = l_config.dimensions[2];
    if (a_config.ws && a_config.gemm_sched == 1) {
      JobList jl = tpu_gemm_jobs(a_config, M, K, N, -1, true);
      return {jl, jl};
    }
    return Matmul(a_config, l_config);
  }
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

  if (a_config.ws && a_config.gemm_sched == 1) {
    JobList jl = tpu_gemm_jobs(a_config, M, K, N, -1, true);
    return {jl, jl};
  }
  if (a_config.ws) {
    JobList jl;
    
    // First split N by number of cores
    auto core_ns = split_even(N, a_config.n_cores);
    std::cout << "Conv WS N-splitting: " << N << " output channels across " << core_ns.size() << " cores" << std::endl;

    for (int core = 0; core < (int) core_ns.size(); ++core) {
      int core_n = core_ns[core];
      // Check if this core's portion fits in buffer
      int required_buff_sz_per_core = (M * core_n + M * std::min(K, a_config.sa_sz_allo)) * batch_size * data_type_width;
      bool core_is_bufferable = required_buff_sz_per_core <= buffer_size_bytes;
      
      if (core_is_bufferable) {
        auto job = new SystolicArray::SysArrayJob(M, K, core_n);
        job->core_id = core;  // Assign core ID for parallel scheduling
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
                                         N, num_jobs, a_config.n_cores);
    JobList act_layer = createVUJobs(1, M * N, true, {{VectorUnit::VPUPhase::BROADCAST, 1}}, a_config.n_cores);

    connectJobLists(matmul_layers, act_layer);

    return {matmul_layers, act_layer};
  } else {
    int num_jobs = std::max(1, M / a_config.sa_sz_allo);
    JobList matmul_layers = createSAJobs(a_config.sa_sz_allo,
                                         K,
                                         N, num_jobs, a_config.n_cores);
    JobList act_layer = createVUJobs(1, M * N, true, {{VectorUnit::VPUPhase::BROADCAST, 1}}, a_config.n_cores);

    connectJobLists(matmul_layers, act_layer);

    return {matmul_layers, act_layer};
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
  JobList act_layer = createVUJobs(1, M * K, true, {{VectorUnit::VPUPhase::BROADCAST, 1}}, a_config.n_cores);

  if (a_config.ws) {
    int num_jobs = std::max(1, K / a_config.sa_sz_allo);
    JobList matmul_layers = createSAJobs(M,
                                         a_config.sa_sz_allo,
                                         N, num_jobs, a_config.n_cores);
    connectJobLists(act_layer, matmul_layers);

    return {act_layer, matmul_layers};
  } else {
    int num_jobs = std::max(1, M / a_config.sa_sz_allo);
    JobList matmul_layers = createSAJobs(a_config.sa_sz_allo,
                                         K,
                                         N, num_jobs, a_config.n_cores);
    connectJobLists(act_layer, matmul_layers);

    return {act_layer, matmul_layers};
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
  // Create jobs with buffer size constraints
  int dec_amt = std::max(1, buffer_size_bytes / data_type_width / lin_dim);
  JobList jl = createVUJobs(lin_dim, par_dim, false,
                            {{VectorUnit::VPUPhase::REDUCE, 1}, {VectorUnit::VPUPhase::REDUCE, 4}, {VectorUnit::VPUPhase::BROADCAST, 1}},
                            a_config.gemm_sched == 1 ? 1 : a_config.n_cores, dec_amt);
  return {jl, jl};
}

JobPair Activation(const ArchConfig &a_config, const LayerConfig &l_config) {
  int sz = 1;
  for (const auto &dim: l_config.dimensions) sz *= dim;

  JobList jl = createVUJobs(1, sz, false, {{VectorUnit::VPUPhase::BROADCAST, 1}}, a_config.gemm_sched == 1 ? 1 : a_config.n_cores);
  return {jl, jl};
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

  // All rows are simulated. (This used to keep only 1/n_vpus of the jobs, a leftover of the old
  // quarter-chip shortcut where one MXU and one VPU each simulated a quarter of the work; the
  // matmul lowering no longer takes that shortcut.) Rows per job: the job's scores must fit in
  // the buffer and at most 1024 rows; the jobs are spread over the VUs.
  int rows = M * heads;
  int64_t row_bytes = int64_t(M) * data_type_width * batch_size;
  int max_rows = int(std::max<int64_t>(1, std::min<int64_t>(1024, buffer_size_bytes / row_bytes)));
  JobList softmax_layer = createVUJobs(M, rows, false, softmax_phases, a_config.gemm_sched == 1 ? 1 : a_config.n_cores, max_rows);
  std::cout << "Softmax: " << rows << " rows in " << softmax_layer.size() << " jobs" << std::endl;
  return {softmax_layer, softmax_layer};
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
    // forced splits are those of the validated TPUv3 model; they only apply with -gemm_sched 1
    int fs_proj = N > 320 ? 4 : -1, fs_dot = 4, fs_out = N > 320 ? 4 : 1;
    auto K_proj = MatmulSplit(a_config, LayerConfig("Matmul", {M, K, N}), fs_proj);
    auto Q_proj = MatmulSplit(a_config, LayerConfig("Matmul", {M, K, N}), fs_proj);
    auto V_proj = MatmulSplit(a_config, LayerConfig("Matmul", {M, K, N}), fs_proj);
    
    // Attention mechanism: Q*K^T and softmax(QK^T)*V
    auto Dot1 = MatmulSplit(a_config, LayerConfig("Matmul", {n_heads, M, N / n_heads, M}), fs_dot);
    auto Dot2 = MatmulSplit(a_config, LayerConfig("Matmul", {n_heads, M, M, N / n_heads}), fs_dot);
    auto O_proj = MatmulSplit(a_config, LayerConfig("Matmul", {n_heads, M, N / n_heads, K}), fs_out);
    auto softmax_layer = Softmax(a_config, LayerConfig("Softmax", {n_heads, M}));

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
                                  N, num_jobs, a_config.n_cores);
    JobList Q_proj = createSAJobs(a_config.sa_sz_allo,
                                  K,
                                  N, num_jobs, a_config.n_cores);
    JobList V_proj = createSAJobs(a_config.sa_sz_allo,
                                  K,
                                  N, num_jobs, a_config.n_cores);
    JobList Dot1 = createSAJobs(a_config.sa_sz_allo,
                                K,// / m_config.n_heads,
                                M, num_jobs, a_config.n_cores);
    JobList Dot2 = createSAJobs(a_config.sa_sz_allo,
                                M,
                                N, num_jobs, a_config.n_cores);/// m_config.n_heads
    JobList O_proj = createSAJobs(a_config.sa_sz_allo,
                                  K,
                                  N, num_jobs, a_config.n_cores);

    JobList softmax_layer = {new VectorUnit::VecUnitJob(M, M, true,
                                                        {{VectorUnit::VPUPhase::REDUCE, 1}, {VectorUnit::VPUPhase::REDUCE, 1}, {VectorUnit::VPUPhase::BROADCAST, 1}})};
    connectJobLists(K_proj, Q_proj);
    connectJobLists(Q_proj, Dot1);
    connectJobLists(Dot1, softmax_layer);
    connectJobLists(softmax_layer, V_proj);
    connectJobLists(V_proj, Dot2);
    connectJobLists(Dot2, O_proj);


    return {K_proj, O_proj};
  }
}

// FlashAttention-style fused attention core (projections are separate layers):
//   FlashAttention B H N D [block_q=1024] [block_kv=auto] [window=2] [causal=0]
// Each (batch, head, q-block) item is pinned to one core and walks the KV sequence in chunks:
//   S = Q K^T (SA) -> online softmax on S (VU: max, exp, sum) -> rescale O (VU, chunks after the first)
//   -> O += P V (SA), and finally O / l (VU), which writes O to DRAM.
// Scores, P and the running O never leave the buffer; Q is read once per item, K and V once per item.
// At most `window` score chunks per item and `window` items per core are in flight, so one item's softmax
// overlaps another item's GEMMs on the same core (FlashAttention-3 style ping-pong).
// block_kv=auto sizes a score chunk to buffer / (window * cores).
JobPair FlashAttention(const ArchConfig &a_config, const LayerConfig &l_config) {
  const auto &d = l_config.dimensions;
  if (d.size() < 4) {
    std::cerr << "FlashAttention expects B H N D [block_q] [block_kv] [window] [causal]" << std::endl;
    throw std::exception();
  }
  int B = d[0], H = d[1], N = d[2], D = d[3];
  int ncore = std::max(1, a_config.n_cores);
  int Bq = (d.size() > 4 && d[4] > 0) ? std::min(d[4], N) : std::min(1024, N);
  int W = (d.size() > 6 && d[6] > 0) ? d[6] : 2;
  bool causal = d.size() > 7 && d[7] != 0;
  int Bkv;
  if (d.size() > 5 && d[5] > 0) {
    Bkv = std::min(d[5], N);
  } else {
    int64_t budget = buffer_size_bytes / (int64_t(W) * ncore);
    Bkv = int(std::max<int64_t>(128, budget / (int64_t(Bq) * data_type_width * batch_size) / 128 * 128));
    Bkv = std::min(Bkv, N);
  }
  const std::vector<std::pair<VectorUnit::VPUPhase, int>> one_pass = {{VectorUnit::VPUPhase::BROADCAST, 1}};
  JobList heads, tails;
  std::vector<Job *> item_done;
  int qblocks = div_ru(N, Bq), item = 0;
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      for (int qb = 0; qb < qblocks; ++qb, ++item) {
        int core = item % ncore;
        int r0 = qb * Bq, q = std::min(Bq, N - r0);
        int kv_end = causal ? std::min(N, r0 + q) : N;
        Job *first = nullptr, *prev_qk = nullptr, *prev_sm = nullptr, *prev_pv = nullptr;
        std::vector<Job *> pvs;
        for (int c0 = 0, j = 0; c0 < kv_end; c0 += Bkv, ++j) {
          int kv = std::min(Bkv, kv_end - c0);
          auto *qk = new SystolicArray::SysArrayJob(q, D, kv);  // Q rows stream, K^T chunk is stationary
          qk->act_on_chip = j > 0;
          qk->out_on_chip = true;
          qk->core_id = core;
          auto *sm = new VectorUnit::VecUnitJob(kv, q, true, softmax_phases);
          sm->out_on_chip = true;
          sm->core_id = ncore + core;  // the VU of the same core
          qk->add_child(sm);
          if (prev_sm) prev_sm->add_child(sm);  // running max and sum
          Job *before_pv = sm;
          if (j > 0) {  // O *= exp(m_old - m_new)
            auto *rs = new VectorUnit::VecUnitJob(D, q, true, one_pass);
            rs->out_on_chip = true;
            rs->core_id = ncore + core;
            sm->add_child(rs);
            before_pv = rs;
          }
          auto *pv = new SystolicArray::SysArrayJob(q, kv, D);  // P rows stream from the buffer, V chunk is stationary
          pv->act_on_chip = true;
          pv->out_on_chip = true;
          pv->core_id = core;
          before_pv->add_child(pv);
          if (prev_pv) prev_pv->add_child(pv);
          if (prev_qk) prev_qk->add_child(qk);
          if (j >= W) pvs[j - W]->add_child(qk);  // at most W score chunks of this item in the buffer
          if (!first) first = qk;
          prev_qk = qk;
          prev_sm = sm;
          prev_pv = pv;
          pvs.push_back(pv);
        }
        auto *norm = new VectorUnit::VecUnitJob(D, q, true, one_pass);  // O / l, written to DRAM
        norm->core_id = ncore + core;
        prev_pv->add_child(norm);
        tails.push_back(norm);
        if (item >= W * ncore) {
          item_done[item - W * ncore]->add_child(first);  // at most W items in flight per core
        } else {
          heads.push_back(first);
        }
        item_done.push_back(norm);
      }
    }
  }
  std::cout << "FlashAttention: " << item << " items (batch x head x q-block), block_q " << Bq << ", block_kv " << Bkv
            << ", window " << W << (causal ? ", causal" : "") << std::endl;
  return {heads, tails};
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
  if (layer_type == "Activation")
    return Activation;
  if (layer_type == "LayerNorm")
    return LayerNorm;
  if (layer_type == "SelfAttention")
    return SelfAttention;
  if (layer_type == "MultiHeadSelfAttention")
    return MultiHeadSelfAttention;
  if (layer_type == "FlashAttention")
    return FlashAttention;
  throw std::runtime_error("Unknown layer type: " + layer_type);
}

ArchConfig frontend::standard::arch_config;

std::vector<JobPair> StandardLayer::make_layers(const std::vector<LayerConfig> &layer_configs) const {
  std::vector<JobPair> model_heads;
  JobList jp;
  for (int m = 0; m < model_parallelism; ++m) {
    std::vector<JobPair> lists;
    for (const auto &layer_config: layer_configs) {
      auto layer_f = getLayerLambda(layer_config.layer_type);
      lists.push_back(layer_f(arch_config, layer_config));
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
