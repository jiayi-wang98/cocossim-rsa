/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#include "frontends/Frontend.h"
#include "frontends/standard/StandardLayer.h"
#include "frontends/standard/StandardParser.h"

#include "memory.h"
#include <chrono>

std::string layer_file;
std::string ofile;

using namespace frontend::standard;

using MyArchParser = StandardParser;
using MyLayerParser = StandardLayer;

int main(int argc, char **argv) {
  MyArchParser archParser(argc, argv);
  archParser.parse_early_runtime_overrides();
  MyLayerParser layerParser;

  auto t1 = std::chrono::high_resolution_clock::now();
  mem::setup();

#ifdef VCD
  vcd = fopen("out.vcd", "w");
#endif

  Arch *arch = archParser.make_arch();
  arch->init_waveforms();

  std::string line;
  std::vector<LayerConfig> layer_configs;

  std::ifstream layer_stream(layer_file);
  if (!layer_stream.is_open()) {
    throw std::runtime_error("Error: Could not open layer configuration file: " + layer_file);
  }

  // Parse layer configuration file (format: layer_name dim1 dim2 ... dim8)
  int n_layers = 0;
  while (std::getline(layer_stream, line)) {
    auto content_start = line.find_first_not_of(" \t\r\n");
    if (content_start == std::string::npos || line[content_start] == '#') {
      continue;
    }

    std::string parse_line = line.substr(content_start);
    auto comment_pos = parse_line.find('#');
    if (comment_pos != std::string::npos) {
      parse_line = parse_line.substr(0, comment_pos);
    }

    auto content_end = parse_line.find_last_not_of(" \t\r\n");
    if (content_end == std::string::npos) {
      continue;
    }
    parse_line.resize(content_end + 1);

    const char *nt_buff = parse_line.c_str();
    LayerConfig l_config;
    char nm[64] = {0};
    std::cout << "processing " << parse_line << std::endl;
    
    // Parse up to 8 dimensions per layer
    std::vector<int> dims(8);
    int successes = sscanf(nt_buff, "%63s %d %d %d %d %d %d %d %d",
                           nm,
                           &dims[0], &dims[1], &dims[2], &dims[3],
                           &dims[4], &dims[5], &dims[6], &dims[7]);
    if (successes < 2) {
      std::cerr << "failed with '" << nt_buff << "'" << std::endl;
      throw std::exception();
    }
    
    l_config.layer_type = std::string(nm);
    int n_dims = successes - 1;
    dims.resize(n_dims);
    l_config.dimensions = dims;
    layer_configs.push_back(l_config);
    n_layers++;
  }
  layer_stream.close();

  // Setup multi-period simulation with time-based job enqueuing
  TimeBasedEnqueue time_enqueues;
  uint64_t t = 0;
  auto dt = 30000000;  // Time between periods
  std::vector<Job *> period_jobs[periods];
  
  std::cout << "Period: " << periods << std::endl;
  for (int i = 0; i < periods; ++i) {
    // Create jobs for each thread in this period
    for (int j = 0; j < n_threads; ++j) {
      auto network = layerParser.make_layers(layer_configs);
      for (auto &layer: network) {
        period_jobs[i].insert(period_jobs[i].end(), layer.first.begin(), layer.first.end());
      }
      alloc_task_idx++;
    }
    time_enqueues.enqueue_at(t, &period_jobs[i]);
    t += dt;

    std::cout << "Jobs for Period " << i << ":" << std::endl;
    for (auto *job: period_jobs[i]) {
      job->printDetails();
    }
  }

  jobs_to_dot(period_jobs[0]);

  mem::mem_sys->ResetStats();

  auto res = arch->get_cycles(time_enqueues);
  FILE *f = fopen(ofile.c_str(), "w");
  for (int p = 0; p < periods; ++p) {
    fprintf(f, "Cycles %llu\n", res[p].cycles);
    for (int i = 0; i < arch->states.size(); ++i) {
      fprintf(f, "%s %f\n", arch->states[i]->get_ty_string().c_str(), res[p].pct_active[i]);
    }
  }
  fprintf(f, "CGRA_MODE_SWITCH_CYCLES %llu\n", static_cast<unsigned long long>(arch->cgra_mode_switch_cycles));
  fprintf(f, "LAYER_BREAKDOWN_BEGIN\n");
  for (const auto &pr: arch->layer_runtime_stats) {
    const int layer_idx = pr.first;
    const auto &ls = pr.second;
    uint64_t first = ls.first_cycle;
    uint64_t last = ls.last_cycle;
    if (ls.active_cycles == 0) {
      first = 0;
      last = 0;
    }
    const uint64_t window_cycles = ls.active_cycles > 0 ? (last - first + 1) : 0;
    fprintf(f,
            "LAYER %d %s ACTIVE %llu WINDOW %llu FIRST %llu LAST %llu SA %llu VU %llu CGRA_CFG %llu CGRA_STREAM %llu CGRA_WRITE %llu\n",
            layer_idx,
            ls.layer_name.c_str(),
            static_cast<unsigned long long>(ls.active_cycles),
            static_cast<unsigned long long>(window_cycles),
            static_cast<unsigned long long>(first),
            static_cast<unsigned long long>(last),
            static_cast<unsigned long long>(ls.sa_cycles),
            static_cast<unsigned long long>(ls.vu_cycles),
            static_cast<unsigned long long>(ls.cgra_config_cycles),
            static_cast<unsigned long long>(ls.cgra_stream_cycles),
            static_cast<unsigned long long>(ls.cgra_write_cycles));
  }
  fprintf(f, "LAYER_BREAKDOWN_END\n");


  uint64_t last_cycles = res[periods - 1].cycles;
  auto lc = (double) last_cycles;
  auto expected_c = (double)res[0].cycles;
  double ratio = lc / expected_c;
  printf("Drain Ratio: %f\n", ratio);

  fclose(f);
  mem::mem_sys->PrintEpochStats();
#ifdef VCD
  fclose(vcd);
#endif
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "Simulation took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms" << std::endl;
}
