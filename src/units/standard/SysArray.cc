/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#include "units/standard/SysArray.h"
#include "State.h"
#include "frontends/standard/StandardArch.h"
#include <cstdint>
#include <algorithm>
#include <cmath>
#include "global.h"


using namespace frontend::standard;

bool SystolicArray::SysArrayState::increment(const std::function<void(Job *)> &enqueue_job, int &total_idle, int *n_idle_units) {
  auto *cj = dynamic_cast<CgraJob *>(j);
  enqueue_reads();
  enqueue_writes();
  if (!process_stage()) {
    return state != SystolicArray::ExState::idle;
  }

  if (cj != nullptr) {
    switch (state) {
      case cgra_config: {
        int merge_factor = std::max(cgra_cluster, 1);
        int cgra_side = std::max(sz / merge_factor, 1);
        int n_cgra_alus = std::max(cgra_side * cgra_side, 1);
        int work_items = std::max(1, cj->linearized_dimension * cj->parallel_dimension);
        int stream_quanta = std::max(div_ru(work_items, n_cgra_alus), 1);
        int ii = std::max(cj->initiation_interval, std::max(cgra_default_ii, 1));
        int op_latency = std::max(cj->op_latency, 1);
        int pipe_depth = std::max(cj->pipeline_depth, std::max(cgra_default_pipeline, 1));
        int stream_cycles = pipe_depth + std::max(stream_quanta - 1, 0) * ii * op_latency;
        state_transfer(cgra_stream, 0, 0, std::max(stream_cycles, 1));
      } break;
      case cgra_stream:
        state_transfer(write, 0, std::max(cj->write_bytes, 0), 0);
        break;
      case write:
        state_transfer(idle, 0, 0, 0);
        TO_IDLE_CLEANUP();
        break;
      case idle:
        break;
      default:
        std::cerr << "Unexpected systolic state while processing CGRA job..." << std::endl;
        throw std::exception();
    }
    return state != SystolicArray::ExState::idle;
  }

  auto *sj = (SysArrayJob *) j;
  auto is_last_tile = [&](int row, int col) {
    return row == loop_row_tiles && col == loop_cols_tiles;
  };
  auto advance_tile = [&](int &row, int &col) -> bool {
    if (is_last_tile(row, col)) {
      return false;
    }
    if (col == loop_cols_tiles) {
      col = 1;
      row++;
    } else {
      col++;
    }
    return true;
  };
  auto preload_bytes_for_tile = [&](int row, int col) {
    int weight_preload = std::min(sz, sj->K) * std::min(sz, sj->N) * data_type_width;
    int activation_preload = 0;
    if (col == 1 && row > 1) {
      activation_preload = std::min(sz, sj->K) * sj->M * batch_size * data_type_width;
    }
    return weight_preload + activation_preload;
  };

  if (ws) {  // Weight Stationary mode
    switch (state) {
        case prefetch:  // Load weights into systolic array
          {
            int next_read_preload = 0;
            if (ws_double_buffer) {
              int nxt_r = row_i;
              int nxt_c = col_i;
              if (advance_tile(nxt_r, nxt_c)) {
                // During read(row_i,col_i), preload the next tile.
                next_read_preload = preload_bytes_for_tile(nxt_r, nxt_c);
              }
            }
          state_transfer(read,
                         next_read_preload,
                         0,
                         sj->M * std::max(systolic_fpu_latency, batch_size));
          }
          break;
        case read:  // Read input activations
          if (ws_double_buffer) {
            if (is_last_tile(row_i, col_i)) {
              // Keep explicit tail flush for the final tile.
              state_transfer(shift, 0, 0, sz * std::max(systolic_fpu_latency, batch_size));
            } else {
              // Idealized overlap: steady-state tiles skip explicit shift.
              state_transfer(write, 0, 0, 0);
            }
          } else {
            state_transfer(shift,
                           std::min(sz, sj->K) * std::min(sz, sj->N) * data_type_width,
                           0,
                           sz * std::max(systolic_fpu_latency, batch_size));
          }
          break;
          
        case shift: {  // Compute phase: shift data through systolic array
          if (ws_double_buffer) {
            // In double-buffer mode, shift is only used for final-tile tail flush.
            int amt_to_write = is_last_tile(row_i, col_i)
                                   ? sj->M * sj->N * data_type_width * batch_size
                                   : 0;
            state_transfer(write, 0, amt_to_write, 0);
            break;
          }

          int amt_to_write = 0;
          int amt_to_read = 0;
          int n_cycles = 0;
          int activation_preload = 0;
          
          // Check if we're at the end of tile computation
          if (col_i == loop_cols_tiles) {
            if (row_i == loop_row_tiles) {
              amt_to_write = sj->M * sj->N * data_type_width * batch_size;
            } else {
              activation_preload = std::min(sz, sj->K) * sj->M * batch_size * data_type_width;
            }
          }
          amt_to_read = activation_preload;
          n_cycles = 0;
          state_transfer(write, amt_to_read, amt_to_write, n_cycles);
        } break;
        case write: {  // Write output data to memory
          int rd_cycles = sj->M * std::max(systolic_fpu_latency, batch_size);
          if (is_last_tile(row_i, col_i)) {
            // Job completed
            state_transfer(idle, 0, 0, 0);
            TO_IDLE_CLEANUP();
          } else {
            int next_row = row_i;
            int next_col = col_i;
            advance_tile(next_row, next_col);
            if (next_col == 1) {
              j->addr = j->addr_hold;
            }

            int overlap_preload = 0;
            if (ws_double_buffer) {
              int nxt2_row = next_row;
              int nxt2_col = next_col;
              if (advance_tile(nxt2_row, nxt2_col)) {
                // During read(next_row,next_col), preload one tile ahead.
                overlap_preload = preload_bytes_for_tile(nxt2_row, nxt2_col);
              }
            }
            state_transfer(read, overlap_preload, 0, rd_cycles);
            row_i = next_row;
            col_i = next_col;
          }
        } break;
        case idle:
          break;
    }
  } else {  // Output Stationary mode
    switch (state) {
        case read:  // Read weights and activations
          state_transfer(shift, 0, 0, sz * std::min(systolic_fpu_latency, batch_size));
          break;
        case shift:  // Compute and accumulate outputs
          state_transfer(write, 0, beats_per_wb, 0);
          break;
        case write:  // Write partial sums back to memory
          if (col_i == loop_cols_tiles) {
            if (row_i == loop_row_tiles) {
              // Job completed
              state_transfer(SystolicArray::idle, 0, 0, 0);
              TO_IDLE_CLEANUP();
            } else {
              // Move to next row tile
              init_row_loop(true);
              j->addr = j->addr_hold;
              UPDATE_STATE(SystolicArray::read);
              if (is_idle_from_memory) {
                UPDATE_IDLEMEM(false);
              }
              col_i = 1;
              row_i++;
            }
          } else {
            init_row_loop(false);
            UPDATE_STATE(SystolicArray::read);
            if (is_idle_from_memory) {
              UPDATE_IDLEMEM(false);
            }
            col_i++;
          }
          break;
        case idle:
          break;
        default:
          std::cerr << "Caught in unexpected state in Output Stationary Systolic Array..." << std::endl;
          throw std::exception();
    }
  }
  return state != SystolicArray::ExState::idle;
}

void SystolicArray::SysArrayState::init_row_loop(bool new_row) {
  auto sj = (SysArrayJob *) j;

  int n_read_bytes = 0;
  int n_read_beats = 0;
  if (ws) {
    throw std::exception();
  } else {
    min_stage_cycles = sj->K * systolic_fpu_latency;
    if (new_row) {
      n_read_bytes = std::min(sz, sj->M) * sj->K * (batch_size + (j->batched_weights ? batch_size : 1)) * data_type_width;
    } else {
      n_read_bytes = std::min(sz, sj->M) * sj->K * (j->batched_weights ? batch_size : 1) * data_type_width;
    }
    n_read_beats = std::max(n_read_bytes / bytes_per_tx, 1);
  }
  mem_read_left = mem_read_left_unqueued = n_read_beats;
}

void SystolicArray::SysArrayState::init() {
  auto *cj = dynamic_cast<CgraJob *>(j);
  if (cj != nullptr) {
    UPDATE_STATE(SystolicArray::cgra_config);
    int cfg_cycles = std::max(1, int(std::ceil(cgra_cfg_ratio * float(sz))));
    int read_amt = cj->is_prebuffered ? 0 : std::max(cj->read_bytes, 0);
    state_transfer(SystolicArray::cgra_config, read_amt, 0, cfg_cycles);
    loop_row_tiles = 1;
    loop_cols_tiles = 1;
    row_i = 1;
    col_i = 1;
    return;
  }

  auto sj = (SysArrayJob *) j;
  if (j->is_done) {
    std::cerr << "ERROR" << std::endl;
  }
  if (ws) {
    UPDATE_STATE(SystolicArray::prefetch);
    loop_cols_tiles = div_ru(sj->N, sz);
    loop_row_tiles = div_ru(sj->K, sz);
    int sys_array_preload = std::min(sz, sj->N) * std::min(sz, sj->K) * data_type_width;
    int activation_preload = std::min(sz, sj->K) * sj->M * data_type_width;
    state_transfer(SystolicArray::prefetch, activation_preload + sys_array_preload, 0, sz);
    row_i = 1;
    col_i = 1;
  } else {
    UPDATE_STATE(SystolicArray::read);
    min_stage_cycles = sj->K * std::max(systolic_fpu_latency, batch_size);
    int n_read_bytes = std::min(sz, sj->M) * sj->K * (batch_size + (j->batched_weights ? batch_size : 1)) * data_type_width;
    int n_read_beats = n_read_bytes / bytes_per_tx;
    mem_read_left = mem_read_left_unqueued = n_read_beats;

    loop_cols_tiles = std::max(sj->N / sz, 1);
    loop_row_tiles = std::max(sj->M / sz, 1);

    row_i = 1;
    col_i = 1;
  }
  if (loop_row_tiles == 0)
    throw std::runtime_error("loop_row_tiles == 0");
  if (loop_cols_tiles == 0)
    throw std::runtime_error("loop_cols_tiles == 0");
}

SystolicArray::SysArrayState::SysArrayState(int sz,
                                            bool ws,
                                            bool ws_double_buffer,
                                            bool reuse_sa_for_vu,
                                            int cgra_cluster,
                                            float cgra_cfg_ratio,
                                            int cgra_default_ii,
                                            int cgra_default_pipeline)
    : State(1),
      sz(sz),
      ws(ws),
      ws_double_buffer(ws_double_buffer),
      reuse_sa_for_vu(reuse_sa_for_vu),
      cgra_cluster(std::max(cgra_cluster, 1)),
      cgra_cfg_ratio(std::max(cgra_cfg_ratio, 0.0f)),
      cgra_default_ii(std::max(cgra_default_ii, 1)),
      cgra_default_pipeline(std::max(cgra_default_pipeline, 1)),
      state(SystolicArray::idle) {
  beats_per_wb = std::max((sz * sz * data_type_width * batch_size) / bytes_per_tx, 1);
}

SystolicArray::SysArrayJob::SysArrayJob(int m, int k, int n)
    : Job(m * m * n * data_type_width * batch_size * 2 + n * m * data_type_width * batch_size), M(m), K(k), N(n) {}


std::string SystolicArray::SysArrayJob::get_job_dims_string() const {
  return std::to_string(M) + " x " + std::to_string(K) + " x " + std::to_string(N);
}

SystolicArray::CgraJob::CgraJob(int linearized_dimension,
                                int parallel_dimension,
                                SystolicArray::CgraOp op,
                                int op_latency,
                                int initiation_interval,
                                int pipeline_depth,
                                int read_bytes,
                                int write_bytes,
                                bool is_prebuffered)
    : Job(std::max(read_bytes + write_bytes, 1)),
      linearized_dimension(linearized_dimension),
      parallel_dimension(parallel_dimension),
      op(op),
      op_latency(std::max(op_latency, 1)),
      initiation_interval(std::max(initiation_interval, 1)),
      pipeline_depth(std::max(pipeline_depth, 1)),
      read_bytes(std::max(read_bytes, 0)),
      write_bytes(std::max(write_bytes, 0)),
      is_prebuffered(is_prebuffered) {}

std::string SystolicArray::CgraJob::get_job_dims_string() const {
  return "CGRA op=" + std::to_string(op) +
         " dims=" + std::to_string(parallel_dimension) +
         " x " + std::to_string(linearized_dimension);
}
