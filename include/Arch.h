/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#ifndef PROSE_COMPILER_ARCH_H
#define PROSE_COMPILER_ARCH_H

#include "EnqueueStructures.h"
#include "RuntimeStats_t.h"
#include "global.h"
#include <unordered_map>
#include <map>
#include <vector>

extern std::unordered_map<int, int> state_updates;

struct LayerRuntimeStat {
  std::string layer_name;
  uint64_t first_cycle = 0xFFFFFFFFFFFFFFFFull;
  uint64_t last_cycle = 0;
  uint64_t active_cycles = 0;
  uint64_t sa_cycles = 0;
  uint64_t vu_cycles = 0;
  uint64_t cgra_config_cycles = 0;
  uint64_t cgra_stream_cycles = 0;
  uint64_t cgra_write_cycles = 0;
};

struct Arch {
  std::vector<State *> states;
  std::map<int, LayerRuntimeStat> layer_runtime_stats;
  uint64_t cgra_mode_switch_cycles = 0;

  State *have_idle_type(int ty);


  int total_frontier = 0;
  int *n_idle_units = nullptr;
  int total_idle = 0;

  Arch() = default;

  void init_waveforms();

  RuntimeStats_t *get_cycles(TimeBasedEnqueue &time_enqueues);

  private:
  bool have_inited = false;
};

#endif//PROSE_COMPILER_ARCH_H
