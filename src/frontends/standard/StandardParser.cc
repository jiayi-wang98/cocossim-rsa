/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#include "frontends/standard/StandardParser.h"
#include <algorithm>

using namespace frontend::standard;

Arch* StandardParser::make_arch() {
  int cores;
  int sa_sz;
  int vu_sz;
  int ws;
  int ws_dbuf = 0;
  int reuse_sa_for_vu = 0;
  int cgra_cluster = 4;
  int cgra_default_ii = 1;
  int cgra_default_pipeline = 8;
  float cgra_cfg_ratio = 1.0f;
  parse_args({{"-c", &cores},
              {"-sa_sz", &sa_sz},
              {"-vu_sz", &vu_sz},
              {"-ws", &ws},
              {"-ws_dbuf", &ws_dbuf},
              {"-reuse_sa_for_vu", &reuse_sa_for_vu},
              {"-cgra_cluster", &cgra_cluster},
              {"-cgra_ii", &cgra_default_ii},
              {"-cgra_pipe", &cgra_default_pipeline}},
             "-c       number of cores\n"
             "-sa_sz   size of the systolic array\n"
             "-sz_vu   size of the vector unit\n"
             "-ws      weight stationary (1) or output stationary (0)\n"
             "-ws_dbuf enable WS double-buffer overlap mode (0|1)\n"
             "-reuse_sa_for_vu route vector-style ops to SA CGRA mode (0|1)\n"
             "-cgra_cluster PE merge factor for CGRA ALUs (default: 4)\n"
             "-cgra_cfg_ratio config overhead ratio of sa_sz (default: 1.0)\n"
             "-cgra_ii default CGRA initiation interval (default: 1)\n"
             "-cgra_pipe default CGRA pipeline depth (default: 8)",
             {{"-cgra_cfg_ratio", &cgra_cfg_ratio}});
  cgra_cluster = std::max(cgra_cluster, 1);
  cgra_default_ii = std::max(cgra_default_ii, 1);
  cgra_default_pipeline = std::max(cgra_default_pipeline, 1);
  cgra_cfg_ratio = std::max(cgra_cfg_ratio, 0.0f);
  arch_config = ArchConfig(cores,
                           sa_sz,
                           vu_sz,
                           ws != 0,
                           ws_dbuf != 0,
                           reuse_sa_for_vu != 0,
                           cgra_cluster,
                           cgra_cfg_ratio,
                           cgra_default_ii,
                           cgra_default_pipeline);
  return new StandardArch;
}
