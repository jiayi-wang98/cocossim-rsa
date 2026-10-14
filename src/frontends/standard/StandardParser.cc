/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#include "frontends/standard/StandardParser.h"

using namespace frontend::standard;

Arch* StandardParser::make_arch() {
  int cores = 1;
  int sa_sz = 128;
  int vu_sz = 128;
  int ws = 0;
  int gemm_sched = 0;
  parse_args({{"-c", &cores},
              {"-sa_sz", &sa_sz},
              {"-vu_sz", &vu_sz},
              {"-ws", &ws},
              {"-gemm_sched", &gemm_sched}},
             "-c       number of cores\n"
             "-sa_sz   size of the systolic array\n"
             "-sz_vu   size of the vector unit\n"
             "-ws      weight stationary (1) or output stationary (0)\n"
             "-gemm_sched  0: split GEMMs evenly over cores (default); 1: TPUv3-fitted partitioning (WS only)");
  arch_config = ArchConfig(cores, sa_sz, vu_sz, ws, gemm_sched);
  return new StandardArch;
}
