#pragma once
#include "simulator.hpp"

namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());

  Matrix *key_prefix = nullptr;
  constexpr size_t kSramPrefixRounds = 22;

  for (size_t i = 0; i < keys.size(); ++i) {
    Matrix *query = rater.GetNextQuery();

    // Values stay as individual SRAM rows.  Small K prefixes also remain in
    // SRAM; larger prefixes use HBM so rebuilding them cannot set the SRAM
    // high-water mark.
    gpu_sim.MoveMatrixToSharedMem(values[i]);
    if (i < kSramPrefixRounds) {
      gpu_sim.MoveMatrixToSharedMem(keys[i]);
    }

    Position key_position = i < kSramPrefixRounds ? kInSharedMemory
                                                   : kInGpuHbm;

    if (i == 0) {
      key_prefix = keys[i];
      gpu_sim.Transpose(key_prefix, key_position);
    } else {
      gpu_sim.Transpose(keys[i], key_position);
      Matrix *next_key_prefix =
          matrix_memory_allocator.Allocate("key_prefix_" + std::to_string(i));
      gpu_sim.Concat(key_prefix, keys[i], next_key_prefix, 1,
                     key_position);
      gpu_sim.ReleaseMatrix(key_prefix);
      gpu_sim.ReleaseMatrix(keys[i]);
      key_prefix = next_key_prefix;
    }
    if (key_position == kInGpuHbm) {
      gpu_sim.MoveMatrixToSharedMem(key_prefix);
    }

    // Q K^T is the sum of outer products of matching Q/K columns.  This keeps
    // each score's floating-point accumulation order unchanged while avoiding
    // the simulator's costly multiplication of two large operand buffers.
    const size_t feature_count = query->GetColumnNum();
    std::vector<Matrix *> query_columns(feature_count, nullptr);
    query_columns[0] = matrix_memory_allocator.Allocate(
        "query_column_" + std::to_string(i) + "_0");
    gpu_sim.GetColumn(query, 0, query_columns[0], kInGpuHbm);
    gpu_sim.MoveMatrixToSharedMem(query_columns[0]);

    Matrix *scores = nullptr;
    for (size_t feature = 0; feature < feature_count; ++feature) {
      if (feature + 1 < feature_count) {
        query_columns[feature + 1] = matrix_memory_allocator.Allocate(
            "query_column_" + std::to_string(i) + "_" +
            std::to_string(feature + 1));
        gpu_sim.GetColumn(query, feature + 1, query_columns[feature + 1],
                          kInGpuHbm);
        gpu_sim.MoveMatrixToSharedMem(query_columns[feature + 1]);
      }

      Matrix *key_row = matrix_memory_allocator.Allocate(
          "key_row_" + std::to_string(i) + "_" +
          std::to_string(feature));
      Matrix *outer_product = matrix_memory_allocator.Allocate(
          "outer_product_" + std::to_string(i) + "_" +
          std::to_string(feature));
      gpu_sim.GetRow(key_prefix, feature, key_row, kInSharedMemory);
      gpu_sim.MatMul(query_columns[feature], key_row, outer_product);
      gpu_sim.ReleaseMatrix(query_columns[feature]);
      gpu_sim.ReleaseMatrix(key_row);

      if (scores == nullptr) {
        scores = outer_product;
      } else {
        Matrix *next_scores = matrix_memory_allocator.Allocate(
            "scores_" + std::to_string(i) + "_" +
            std::to_string(feature));
        gpu_sim.MatAdd(scores, outer_product, next_scores);
        gpu_sim.ReleaseMatrix(scores);
        gpu_sim.ReleaseMatrix(outer_product);
        scores = next_scores;
      }
    }
    gpu_sim.ReleaseMatrix(query);

    Matrix *exp_scores =
        matrix_memory_allocator.Allocate("exp_scores_" + std::to_string(i));
    gpu_sim.MatExp(scores, exp_scores);
    gpu_sim.ReleaseMatrix(scores);

    std::vector<Matrix *> weight_rows;
    weight_rows.reserve(i + 1);
    for (size_t row = 0; row <= i; ++row) {
      Matrix *exp_row = matrix_memory_allocator.Allocate(
          "exp_row_" + std::to_string(i) + "_" + std::to_string(row));
      Matrix *row_sum = matrix_memory_allocator.Allocate(
          "row_sum_" + std::to_string(i) + "_" + std::to_string(row));
      Matrix *normalized_row = matrix_memory_allocator.Allocate(
          "normalized_row_" + std::to_string(i) + "_" +
          std::to_string(row));
      gpu_sim.GetRow(exp_scores, row, exp_row, kInSharedMemory);
      gpu_sim.Sum(exp_row, row_sum);
      gpu_sim.MatDiv(exp_row, row_sum, normalized_row);
      gpu_sim.ReleaseMatrix(exp_row);
      gpu_sim.ReleaseMatrix(row_sum);
      weight_rows.push_back(normalized_row);
    }
    gpu_sim.ReleaseMatrix(exp_scores);

    // Finish all score calculations before moving K back to HBM.  Output can
    // then be formed with only the value rows resident in SRAM.
    gpu_sim.Run(false, &matrix_memory_allocator);
    if (i + 1 >= kSramPrefixRounds) {
      gpu_sim.MoveMatrixToGpuHbm(key_prefix);
      gpu_sim.Run(false, &matrix_memory_allocator);
    }

    // Scaling each value row by one scalar weight is algebraically identical
    // to weights * V.  Under this simulator's size-product MatMul cost it is
    // much cheaper than multiplying by the complete value matrix.
    std::vector<Matrix *> answer_rows;
    answer_rows.reserve(i + 1);
    for (size_t row = 0; row <= i; ++row) {
      Matrix *answer_row = nullptr;
      for (size_t column = 0; column <= i; ++column) {
        Matrix *weight = matrix_memory_allocator.Allocate(
            "weight_" + std::to_string(i) + "_" + std::to_string(row) +
            "_" + std::to_string(column));
        Matrix *contribution = matrix_memory_allocator.Allocate(
            "contribution_" + std::to_string(i) + "_" +
            std::to_string(row) + "_" + std::to_string(column));
        gpu_sim.GetColumn(weight_rows[row], column, weight, kInSharedMemory);
        gpu_sim.MatMul(weight, values[column], contribution);
        gpu_sim.ReleaseMatrix(weight);

        if (answer_row == nullptr) {
          answer_row = contribution;
        } else {
          Matrix *next_answer = matrix_memory_allocator.Allocate(
              "answer_sum_" + std::to_string(i) + "_" +
              std::to_string(row) + "_" + std::to_string(column));
          gpu_sim.MatAdd(answer_row, contribution, next_answer);
          gpu_sim.ReleaseMatrix(answer_row);
          gpu_sim.ReleaseMatrix(contribution);
          answer_row = next_answer;
        }
      }
      gpu_sim.ReleaseMatrix(weight_rows[row]);
      gpu_sim.MoveMatrixToGpuHbm(answer_row);
      answer_rows.push_back(answer_row);
    }

    // A balanced merge keeps HBM concatenation work O(n log n), instead of
    // repeatedly copying an ever-growing prefix in an O(n^2) chain.
    std::vector<Matrix *> answer_blocks = answer_rows;
    size_t merge_level = 0;
    while (answer_blocks.size() > 1) {
      std::vector<Matrix *> next_blocks;
      next_blocks.reserve((answer_blocks.size() + 1) / 2);
      for (size_t block = 0; block < answer_blocks.size(); block += 2) {
        if (block + 1 == answer_blocks.size()) {
          next_blocks.push_back(answer_blocks[block]);
          continue;
        }
        Matrix *merged = matrix_memory_allocator.Allocate(
            "answer_" + std::to_string(i) + "_" +
            std::to_string(merge_level) + "_" + std::to_string(block / 2));
        gpu_sim.Concat(answer_blocks[block], answer_blocks[block + 1], merged,
                       0, kInGpuHbm);
        gpu_sim.ReleaseMatrix(answer_blocks[block]);
        gpu_sim.ReleaseMatrix(answer_blocks[block + 1]);
        next_blocks.push_back(merged);
      }
      answer_blocks = std::move(next_blocks);
      ++merge_level;
    }
    Matrix *answer = answer_blocks[0];

    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*answer);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
