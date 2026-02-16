/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#ifndef PERF_MODEL_SYSARRAY_H
#define PERF_MODEL_SYSARRAY_H
#include "State.h"
#include "frontends/standard/StandardUnits.h"

namespace SystolicArray {
  enum CgraOp {
    CGRA_FMA = 0,
    CGRA_LOGIC,
    CGRA_COMPARE,
    CGRA_INT16_OP,
    CGRA_CONVERT,
    CGRA_REDUCE,
    CGRA_EXP_APPROX,
    CGRA_NORMALIZE
  };

  struct SysArrayJob : Job {

    int get_type() const override {
      return SYSTOLIC_ARRAY_IDX;
    }
    int M, K, N;

    SysArrayJob(int m, int k, int n);

    [[nodiscard]] std::string get_job_dims_string() const override;
  };

  struct CgraJob : Job {
    int linearized_dimension;
    int parallel_dimension;
    CgraOp op;
    int op_latency;
    int initiation_interval;
    int pipeline_depth;
    int read_bytes;
    int write_bytes;
    bool is_prebuffered;

    CgraJob(int linearized_dimension,
            int parallel_dimension,
            CgraOp op,
            int op_latency,
            int initiation_interval,
            int pipeline_depth,
            int read_bytes,
            int write_bytes,
            bool is_prebuffered);

    int get_type() const override {
      return SYSTOLIC_ARRAY_IDX;
    }

    [[nodiscard]] std::string get_job_dims_string() const override;
  };

  enum ExState {
    idle = 0,
    prefetch,//only WS
    read,
    shift,
    write,
    cgra_config,
    cgra_stream
  };

  struct SysArrayState : State {
public:
    explicit SysArrayState(int sz,
                           bool ws,
                           bool ws_double_buffer = false,
                           bool reuse_sa_for_vu = false,
                           int cgra_cluster = 4,
                           float cgra_cfg_ratio = 1.0f,
                           int cgra_default_ii = 1,
                           int cgra_default_pipeline = 8);
    bool ws = false;// Flag for weight-stationary systolic array
    bool ws_double_buffer = false;// WS mode with idealized overlap
    bool reuse_sa_for_vu = false;
    int cgra_cluster = 4;
    float cgra_cfg_ratio = 1.0f;
    int cgra_default_ii = 1;
    int cgra_default_pipeline = 8;

    void init_row_loop(bool new_row);
    void init() override;
    int sz;
    ExState state = idle;

    bool increment(const std::function<void(Job *)> &enqueue_job,
                   int &total_idle,
                   int *n_idle_units) override;

    void set_state(int st) override {
      state = (ExState) st;
    };

    int get_state() override {
      return state;
    }
    

    int get_ty_idx() override {
      return SYSTOLIC_ARRAY_IDX;
    }
    std::string get_ty_string() override {
      return SYSTOLIC_ARRAY_STRING;
    }

private:
    int beats_per_wb;
  };

};// namespace SystolicArray


#endif//PERF_MODEL_SYSARRAY_H
