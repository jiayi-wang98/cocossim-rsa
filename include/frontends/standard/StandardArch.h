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
    // How WS GEMMs and vector layers are partitioned over the cores:
    //   0: split N evenly over all cores; vector rows spread over all VUs
    //   1: the TPUv3-fitted policy of the published COCOSSim validation (see tpu_gemm_jobs), with every
    //      partition simulated on its own core
    int gemm_sched = 0;
    ArchConfig(int n_cores, int sa_sz_allo, int vu_sz_allo, bool ws, int gemm_sched = 0)
        : n_cores(n_cores), sa_sz_allo(sa_sz_allo), vu_sz_allo(vu_sz_allo), ws(ws), gemm_sched(gemm_sched) {}
    ArchConfig() = default;
  };

  extern ArchConfig arch_config;

  struct StandardArch: Arch {
    StandardArch();
  };

  extern StandardArch *arch;
}


#endif//PERF_MODEL_STANDARD_ARCH_H
