#pragma once
#include "simulator.hpp"

namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());

  Matrix *key_prefix = nullptr;
  Matrix *value_prefix = nullptr;

  for (size_t i = 0; i < keys.size(); ++i) {
    Matrix *query = rater.GetNextQuery();

    // Keep the growing key/value prefixes in SRAM.  Moving only the new rows
    // avoids retransferring the complete prefix in every round.
    gpu_sim.MoveMatrixToSharedMem(keys[i]);
    gpu_sim.MoveMatrixToSharedMem(values[i]);
    gpu_sim.MoveMatrixToSharedMem(query);

    if (i == 0) {
      key_prefix = keys[i];
      value_prefix = values[i];
      gpu_sim.Transpose(key_prefix, kInSharedMemory);
    } else {
      gpu_sim.Transpose(keys[i], kInSharedMemory);
      Matrix *next_key_prefix =
          matrix_memory_allocator.Allocate("key_prefix_" + std::to_string(i));
      gpu_sim.Concat(key_prefix, keys[i], next_key_prefix, 1,
                     kInSharedMemory);
      gpu_sim.ReleaseMatrix(key_prefix);
      gpu_sim.ReleaseMatrix(keys[i]);
      key_prefix = next_key_prefix;

      Matrix *next_value_prefix = matrix_memory_allocator.Allocate(
          "value_prefix_" + std::to_string(i));
      gpu_sim.Concat(value_prefix, values[i], next_value_prefix, 0,
                     kInSharedMemory);
      gpu_sim.ReleaseMatrix(value_prefix);
      gpu_sim.ReleaseMatrix(values[i]);
      value_prefix = next_value_prefix;
    }

    // scores = Q K^T.  key_prefix is kept transposed, which also lets each
    // subsequent key be appended as one inexpensive column.
    Matrix *scores =
        matrix_memory_allocator.Allocate("scores_" + std::to_string(i));
    gpu_sim.MatMul(query, key_prefix, scores);
    gpu_sim.ReleaseMatrix(query);

    Matrix *exp_scores =
        matrix_memory_allocator.Allocate("exp_scores_" + std::to_string(i));
    gpu_sim.MatExp(scores, exp_scores);
    gpu_sim.ReleaseMatrix(scores);

    // The simulator exposes only a whole-matrix Sum, so normalize each row
    // separately and concatenate the normalized rows back together.
    Matrix *weights = nullptr;
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

      if (weights == nullptr) {
        weights = normalized_row;
      } else {
        Matrix *next_weights = matrix_memory_allocator.Allocate(
            "weights_" + std::to_string(i) + "_" + std::to_string(row));
        gpu_sim.Concat(weights, normalized_row, next_weights, 0,
                       kInSharedMemory);
        gpu_sim.ReleaseMatrix(weights);
        gpu_sim.ReleaseMatrix(normalized_row);
        weights = next_weights;
      }
    }
    gpu_sim.ReleaseMatrix(exp_scores);

    Matrix *answer =
        matrix_memory_allocator.Allocate("answer_" + std::to_string(i));
    gpu_sim.MatMul(weights, value_prefix, answer);
    gpu_sim.ReleaseMatrix(weights);
    gpu_sim.MoveMatrixToGpuHbm(answer);

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
