/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#ifndef PERF_MODEL_STANDARD_ARCH_H
#define PERF_MODEL_STANDARD_ARCH_H

#include "Arch.h"

namespace frontend::standard {
  struct ArchConfig {
    int n_cores = -1;
    int sa_sz_allo = -1;
    int vu_sz_allo = -1;
    bool ws = false;
    bool ws_double_buffer = false;
    bool reuse_sa_for_vu = false;
    int cgra_cluster = 4;
    float cgra_cfg_ratio = 1.0f;
    int cgra_default_ii = 1;
    int cgra_default_pipeline = 8;
    ArchConfig(int n_cores,
               int sa_sz_allo,
               int vu_sz_allo,
               bool ws,
               bool ws_double_buffer,
               bool reuse_sa_for_vu,
               int cgra_cluster,
               float cgra_cfg_ratio,
               int cgra_default_ii,
               int cgra_default_pipeline)
        : n_cores(n_cores),
          sa_sz_allo(sa_sz_allo),
          vu_sz_allo(vu_sz_allo),
          ws(ws),
          ws_double_buffer(ws_double_buffer),
          reuse_sa_for_vu(reuse_sa_for_vu),
          cgra_cluster(cgra_cluster),
          cgra_cfg_ratio(cgra_cfg_ratio),
          cgra_default_ii(cgra_default_ii),
          cgra_default_pipeline(cgra_default_pipeline) {}
    ArchConfig() = default;
  };

  extern ArchConfig arch_config;

  struct StandardArch: Arch {
    StandardArch();
  };

  extern StandardArch *arch;
}


#endif//PERF_MODEL_STANDARD_ARCH_H
