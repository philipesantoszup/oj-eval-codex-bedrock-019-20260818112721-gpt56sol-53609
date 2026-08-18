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

    // Keep the growing key prefix in SRAM.  The value prefix remains in HBM
    // between rounds so it does not inflate SRAM while the key is rebuilt.
    gpu_sim.MoveMatrixToSharedMem(keys[i]);

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
                     kInGpuHbm);
      gpu_sim.ReleaseMatrix(value_prefix);
      gpu_sim.ReleaseMatrix(values[i]);
      value_prefix = next_value_prefix;
    }

    // Compute Q K^T one row at a time.  Keeping Q in HBM and fetching the next
    // row before multiplying the current one pipelines its transfer with the
    // expensive dot products and avoids storing all of Q in SRAM.
    std::vector<Matrix *> query_rows(i + 1, nullptr);
    query_rows[0] = matrix_memory_allocator.Allocate(
        "query_row_" + std::to_string(i) + "_0");
    gpu_sim.GetRow(query, 0, query_rows[0], kInGpuHbm);
    gpu_sim.MoveMatrixToSharedMem(query_rows[0]);
    // Queue V behind the first query row, allowing its transfer to overlap
    // that row's substantially longer dot product.
    gpu_sim.MoveMatrixToSharedMem(value_prefix);

    std::vector<Matrix *> answer_rows;
    answer_rows.reserve(i + 1);
    for (size_t row = 0; row <= i; ++row) {
      if (row < i) {
        query_rows[row + 1] = matrix_memory_allocator.Allocate(
            "query_row_" + std::to_string(i) + "_" +
            std::to_string(row + 1));
        gpu_sim.GetRow(query, row + 1, query_rows[row + 1], kInGpuHbm);
        gpu_sim.MoveMatrixToSharedMem(query_rows[row + 1]);
      }

      Matrix *score_row = matrix_memory_allocator.Allocate(
          "score_row_" + std::to_string(i) + "_" + std::to_string(row));
      Matrix *exp_row = matrix_memory_allocator.Allocate(
          "exp_row_" + std::to_string(i) + "_" + std::to_string(row));
      Matrix *row_sum = matrix_memory_allocator.Allocate(
          "row_sum_" + std::to_string(i) + "_" + std::to_string(row));
      Matrix *normalized_row = matrix_memory_allocator.Allocate(
          "normalized_row_" + std::to_string(i) + "_" +
          std::to_string(row));

      gpu_sim.MatMul(query_rows[row], key_prefix, score_row);
      gpu_sim.ReleaseMatrix(query_rows[row]);
      gpu_sim.MatExp(score_row, exp_row);
      gpu_sim.ReleaseMatrix(score_row);
      gpu_sim.Sum(exp_row, row_sum);
      gpu_sim.MatDiv(exp_row, row_sum, normalized_row);
      gpu_sim.ReleaseMatrix(exp_row);
      gpu_sim.ReleaseMatrix(row_sum);

      Matrix *answer_row = matrix_memory_allocator.Allocate(
          "answer_row_" + std::to_string(i) + "_" + std::to_string(row));
      gpu_sim.MatMul(normalized_row, value_prefix, answer_row);
      gpu_sim.ReleaseMatrix(normalized_row);
      gpu_sim.MoveMatrixToGpuHbm(answer_row);
      answer_rows.push_back(answer_row);
    }
    gpu_sim.ReleaseMatrix(query);
    gpu_sim.MoveMatrixToGpuHbm(value_prefix);

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
