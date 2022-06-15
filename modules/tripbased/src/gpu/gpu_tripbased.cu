#include "motis/tripbased/gpu/gpu_tripbased.h"

#include <cstdio>

namespace motis::tripbased {

#define CUDA_COPY(target, source, size, copy_type)                         \
  if(copy_type == cudaMemcpyHostToDevice) {                                      \
    cudaMalloc((void**) &target, size);                          \
  }                                                                             \
  cudaMemcpy((void**) &target, source, size, copy_type);

__global__ void hello_world() {
  printf("helloWorld!");
}

void search_fwd_gpu(unsigned const max_transfers,
                    gpu_device_ptrs const ptrs) {
  // TODO(sarah): just testing for now
  hello_world<<<1, 1>>>();
}

gpu_device_ptrs allocate_and_copy_on_device(
    std::vector<std::vector<gpu_dest_arrival>> dest_arrs,
    gpu_fws_multimap_arrival_times arrival_times,
    uint16_t total_earliest_arrival,
    uint16_t* line_stop_count,
    std::size_t line_stop_count_size,
    gpu_nested_fws_multimap_transfers transfers) {

  gpu_device_ptrs ptrs;

  std::size_t dest_arrivals_size = dest_arrs.size();

  std::vector<std::size_t> dest_arrivals_index;
  dest_arrivals_index.resize(dest_arrivals_size + 1);
  dest_arrivals_index.emplace_back(0);

  for(auto dest_arr : dest_arrs) {
    CUDA_COPY(ptrs.dest_arrivals_device_[dest_arrivals_index.back()],
              dest_arr.data(),
              dest_arr.size() * sizeof(gpu_dest_arrival),
              cudaMemcpyHostToDevice)
    dest_arrivals_index.emplace_back(dest_arrivals_index.back() + dest_arr.size());
  }
  CUDA_COPY(ptrs.dest_arrivals_index_device_,
            dest_arrivals_index.data(),
            dest_arrivals_index.size() * sizeof(std::size_t),
            cudaMemcpyHostToDevice)
  CUDA_COPY(ptrs.dest_arrivals_size_device_,
            &dest_arrivals_size,
            sizeof(std::size_t),
            cudaMemcpyHostToDevice)

  std::size_t size_arrival_times = (1 + arrival_times.index_size_)
                               * sizeof(std::size_t)
                           + arrival_times.index_[arrival_times.index_size_ - 1]
                                 * sizeof(uint16_t);
  CUDA_COPY(ptrs.arrival_times_device_,
            &arrival_times,
            size_arrival_times,
            cudaMemcpyHostToDevice)

  CUDA_COPY(ptrs.total_earliest_arrival_device_,
            &total_earliest_arrival,
            sizeof(uint16_t),
            cudaMemcpyHostToDevice)

  CUDA_COPY(ptrs.line_stop_count_device_,
            line_stop_count,
            line_stop_count_size * sizeof(uint16_t),
            cudaMemcpyHostToDevice)
  CUDA_COPY(ptrs.line_stop_count_size_device_,
            &line_stop_count_size,
            sizeof(std::size_t),
            cudaMemcpyHostToDevice)

  std::size_t size_transfers = (2 + transfers.base_index_size_ + transfers.index_size_)
                                   * sizeof(std::size_t)
                               + transfers.index_[transfers.index_size_ - 1]
                                     * sizeof(gpu_tb_transfer);
  CUDA_COPY(ptrs.transfers_device,
            &transfers,
            size_transfers,
            cudaMemcpyHostToDevice)

  return ptrs;
}

void free_on_device(gpu_device_ptrs ptrs) {
  // TODO(sarah)
  cudaFree(ptrs.dest_arrivals_device_);
  cudaFree(ptrs.dest_arrivals_index_device_);
  cudaFree(ptrs.dest_arrivals_size_device_);
  cudaFree(ptrs.arrival_times_device_);
  cudaFree(ptrs.total_earliest_arrival_device_);
  cudaFree(ptrs.line_stop_count_device_);
  cudaFree(ptrs.line_stop_count_size_device_);
  cudaFree(ptrs.transfers_device);
}

} // namespace motis::tripbased