/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#ifndef PERF_MODEL_ARCHPARSER_H
#define PERF_MODEL_ARCHPARSER_H
#include "Arch.h"
#include "global.h"
#include <cstring>

extern std::string layer_file;
extern std::string ofile;

struct ArchParser {
  virtual Arch *make_arch() {
    throw std::runtime_error("make_arch() not implemented for the given frontend.");
  }

  static std::string dram_profile_to_cfg(const std::string &profile) {
    if (profile == "tpuv3") {
      return "../dramsim3/configs/HBM2_TPUv3_900GBps_proxy.ini";
    }
    if (profile == "tpuv4") {
      return "../dramsim3/configs/HBM2_TPUv4_1228GBps_proxy.ini";
    }
    if (profile == "hbm2_default" || profile == "default") {
      return "../dramsim3/configs/HBM2_8Gb_x128.ini";
    }
    throw std::runtime_error(
        "Unknown value for -dram_profile: '" + profile +
        "'. Expected one of: tpuv3, tpuv4, hbm2_default");
  }

  void parse_early_runtime_overrides() const {
    for (int i = 1; i < argc; ++i) {
      if (strcmp(argv[i], "-dram_cfg") == 0) {
        dram_config_path = read_flag_value(i, "-dram_cfg");
      } else if (strcmp(argv[i], "-dram_profile") == 0) {
        dram_config_path = dram_profile_to_cfg(read_flag_value(i, "-dram_profile"));
      }
    }
  }

  void parse_args(const std::vector<std::pair<std::string, int *>> &input_prs,
                  const std::string &help_str = "",
                  const std::vector<std::pair<std::string, float *>> &float_prs = {}) const {
    for (int i = 1; i < argc; ++i) {
      if (strcmp(argv[i], "-i") == 0) {
        layer_file = read_flag_value(i, "-i");
      } else if (strcmp(argv[i], "-o") == 0) {
        ofile = read_flag_value(i, "-o");
      } else if (strcmp(argv[i], "-f") == 0) {
        freq_sa = std::stof(read_flag_value(i, "-f"));
      } else if (strcmp(argv[i], "-dram_cfg") == 0) {
        dram_config_path = read_flag_value(i, "-dram_cfg");
      } else if (strcmp(argv[i], "-dram_profile") == 0) {
        dram_config_path = dram_profile_to_cfg(read_flag_value(i, "-dram_profile"));
      } else if (strcmp(argv[i], "-h") == 0) {
        std::cerr << "Global Options:\n"
                     "-i <file>     layer input file\n"
                     "-o <file>     output statistic file\n"
                     "-f <float>    frequency (GHz)\n"
                     "-dram_profile DRAM preset (tpuv3|tpuv4|hbm2_default)\n"
                     "-dram_cfg     explicit DRAMSim3 .ini path\n";
        if (help_str != "") {
          std::cerr << "Arch Specific Options:\n"
                    << help_str << std::endl;
        }
        exit(0);
      } else {
        bool found = false;
        for (auto &pr: input_prs) {
          if (strcmp(argv[i], pr.first.c_str()) == 0) {
            *pr.second = std::stoi(read_flag_value(i, pr.first.c_str()));
            found = true;
            break;
          }
        }
        if (not found) {
          for (auto &fpr: float_prs) {
            if (strcmp(argv[i], fpr.first.c_str()) == 0) {
              *fpr.second = std::stof(read_flag_value(i, fpr.first.c_str()));
              found = true;
              break;
            }
          }
        }
        if (not found) {
          throw std::runtime_error("Failed to parse passed flag: '" + std::string(argv[i]) + "'\n");
        }
      }
    }
  }

  int argc;
  char **argv;
  ArchParser(int argc, char **argv) : argc(argc), argv(argv) {}
  ArchParser() = delete;

 private:
  const char *read_flag_value(int &i, const char *flag) const {
    if (i + 1 >= argc) {
      throw std::runtime_error("Missing value for flag '" + std::string(flag) + "'");
    }
    return argv[++i];
  }
};
#endif//PERF_MODEL_ARCHPARSER_H
