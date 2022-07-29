#include "motis/tripbased/gpu/gpu_tripbased.h"

#include <cstdio>

namespace motis::tripbased {

#define CUDA_ALLOC(target, size) \
  cudaMalloc((void**) &(target), size);

#define CUDA_COPY(target, source, size, copy_type) \
  cudaMemcpy(target, source, size, copy_type);

#define CUDA_ALLOC_COPY(target, source, size)   \
  CUDA_ALLOC(target, size)                      \
  CUDA_COPY(target, source, size, cudaMemcpyHostToDevice)

__global__ void check_dominated(gpu_device_pointers pointers) {
  unsigned idx1 = threadIdx.x;
  unsigned idx2 = threadIdx.y;
  if (idx1 <= idx2 || idx1 >= *pointers.result_set_size_device_) {
    return;
  }
  gpu_tb_journey journey1 = pointers.result_set_device_[idx1];
  gpu_tb_journey journey2 = pointers.result_set_device_[idx2];
  if (journey1.transfers_ == journey2.transfers_) {
    if (journey1.arrival_time_ > journey2.arrival_time_) {
      pointers.is_dominated_device_[idx1] = 1;
    } else {
      pointers.is_dominated_device_[idx2] = 1;
    }
  } else if (journey1.transfers_ > journey2.transfers_) {
    if(journey1.arrival_time_ > journey2.arrival_time_) {
      pointers.is_dominated_device_[idx1] = 1;
    }
  } else { // journey2.transfers_ < journey2.transfers_
    if (journey1.arrival_time_ < journey2.arrival_time_) {
      pointers.is_dominated_device_[idx2] = 1;
    }
  }
}

__device__ void destination_reached(gpu_device_pointers pointers,
                                    gpu_queue_entry queue_entry,
                                    unsigned entry_num,
                                    gpu_dest_arrival dest_arrival,
                                    unsigned transfers) {
  // TODO(sarah): max_travel_time limit check?
  unsigned set_entry_index = atomicAdd(pointers.result_set_size_device_, 1U);
  uint16_t arrival_time
      = pointers.arrival_times_device_.data_[pointers.arrival_times_device_
                                                  .index_[queue_entry.trip_]
                                              + dest_arrival.stop_index_]
        + dest_arrival.fp_duration_;
  pointers.result_set_device_[set_entry_index]
      = gpu_tb_journey{*pointers.start_time_device_,
                       arrival_time,
                       transfers,
                       transfers + 1,
                       dest_arrival.fp_to_station_id_,
                       dest_arrival,
                       entry_num};
  //std::printf("1\n");
}

__device__ void enqueue(gpu_device_pointers const pointers,
                        uint32_t trip,
                        uint16_t stop_idx,
                        unsigned transfers,
                        std::size_t prev_trip_seg) {
  auto const old_first_reachable = pointers.first_reachable_stop_device_[trip];
  if (stop_idx < old_first_reachable) {
    unsigned queue_entry_index = atomicAdd(&pointers.queue_sizes_device_[transfers+1], 1U);
    pointers.queue_device_[pointers.queue_index_device_[transfers + 1] + queue_entry_index]
        = gpu_queue_entry{trip, stop_idx, old_first_reachable, prev_trip_seg};
    auto const line = pointers.trip_to_line_device_[trip];
    for (uint32_t t = trip;
         t < *pointers.trip_count_device_ && pointers.trip_to_line_device_[t] == line;
         ++t) {
      if (stop_idx < pointers.first_reachable_stop_device_[t]) {
        pointers.first_reachable_stop_device_[t] = stop_idx;
      }
    }
  }
}

__global__ void search(gpu_device_pointers const pointers,
                       unsigned const transfers,
                       unsigned const max_transfers) {
  unsigned trip_seg = blockIdx.x * 32 + threadIdx.x;
  unsigned long idx = trip_seg + pointers.queue_index_device_[transfers];
  if (trip_seg >= pointers.queue_sizes_device_[transfers]) {
    return;
  }
  auto& entry = pointers.queue_device_[idx];
  auto const line = pointers.trip_to_line_device_[entry.trip_];

  auto const dest_arrs_size = pointers.dest_arrivals_index_device_[line + 1]
                                  - pointers.dest_arrivals_index_device_[line];
  if (dest_arrs_size > 0) {
    for (auto i = pointers.dest_arrivals_index_device_[line];
         i < pointers.dest_arrivals_index_device_[line + 1]; ++i) {
      if (entry.from_stop_index_ < pointers.dest_arrivals_device_[i].stop_index_) {
        destination_reached(pointers,
                            entry,
                            trip_seg,
                            pointers.dest_arrivals_device_[i],
                            transfers);
      }
    }
  }

  if (transfers + 1 > max_transfers) {
    return;
  }
  auto next_stop_arrival_times = pointers.arrival_times_device_
                                     .data_[pointers.arrival_times_device_
                                                .index_[entry.trip_]
                                            + entry.from_stop_index_ + 1];

  if (next_stop_arrival_times < *pointers.total_earliest_arrival_device_) {
    auto const stop_count =
        std::min(entry.to_stop_index_,
                 static_cast<stop_idx_t>(pointers.line_stop_count_device_[line] - 1));
    for (auto i = entry.from_stop_index_ + 1; i <= stop_count; ++i) {
      auto start_idx = pointers.transfers_device
                           .index_[pointers.transfers_device
                                       .base_index_[entry.trip_] + i];
      auto end_idx = pointers.transfers_device
                         .index_[pointers.transfers_device
                                     .base_index_[entry.trip_] + i + 1];
      for (auto j = start_idx; j < end_idx; ++j) {
        enqueue(pointers,
                pointers.transfers_device.data_[j].to_trip,
                pointers.transfers_device.data_[j].to_stop_idx,
                transfers,
                trip_seg);
      }
    }
  }
}

gpu_search_results search_fwd_gpu(unsigned const max_transfers,
                    gpu_device_pointers const pointers) {
  std::vector<unsigned> queue_sizes(max_transfers + 1);
  // TODO(sarah): <= or < ?
  for (auto transfers = 0U; transfers <= max_transfers; ++transfers) {
    CUDA_COPY(&queue_sizes[transfers],
              &pointers.queue_sizes_device_[transfers],
              sizeof(std::size_t),
              cudaMemcpyDeviceToHost)
    // TODO(sarah)
    unsigned thread_num = queue_sizes[transfers];
    unsigned block_num = 1;
    if(thread_num > 32) {
      block_num = (thread_num + 31) / 32;
      thread_num = 32;
    }
    search<<<block_num, thread_num>>>
        (pointers, transfers, max_transfers);
    cudaDeviceSynchronize();
  }


  std::vector<gpu_queue_entry> test;
  test.resize(queue_sizes[0]);
  cudaMemcpy(test.data(),
            pointers.queue_device_,
            queue_sizes[0] * sizeof(gpu_queue_entry),
            cudaMemcpyDeviceToHost);

  gpu_search_results results;
  std::vector<std::size_t> queue_index = {0, 3750000, 7500000, 11250000,
                                          15000000, 18750000, 22500000, 26250000};

  results.gpu_final_queues_.resize(max_transfers + 1);
  for (auto transfers = 0U; transfers <= max_transfers; ++ transfers) {
    results.gpu_final_queues_[transfers].resize(queue_sizes[transfers]);
    CUDA_COPY(results.gpu_final_queues_[transfers].data(),
              pointers.queue_device_ + queue_index[transfers],
              queue_sizes[transfers] * sizeof(gpu_queue_entry),
              cudaMemcpyDeviceToHost)
  }

  unsigned result_set_size;
  CUDA_COPY(&result_set_size,
            pointers.result_set_size_device_,
            sizeof(unsigned),
            cudaMemcpyDeviceToHost)
  std::cout << result_set_size << std::endl;

  unsigned thread_num = result_set_size;
  unsigned block_num = 1;
  if(thread_num > 32) {
    block_num = (thread_num + 31) / 32;
    thread_num = 32;
  }
  check_dominated<<<block_num, thread_num>>>(pointers);
  cudaDeviceSynchronize();

  results.gpu_result_journeys_.resize(result_set_size);
  CUDA_COPY(results.gpu_result_journeys_.data(),
            pointers.result_set_device_,
            result_set_size * sizeof(gpu_tb_journey),
            cudaMemcpyDeviceToHost)
  std::cout << results.gpu_result_journeys_.size() << std::endl;

  results.gpu_is_dominated_.resize(result_set_size);
  CUDA_COPY(results.gpu_is_dominated_.data(),
            pointers.is_dominated_device_,
            result_set_size * sizeof(uint8_t),
            cudaMemcpyDeviceToHost)
  std::cout << results.gpu_is_dominated_.size() << std::endl;

  return results;
}

gpu_device_pointers allocate_and_copy_on_device(
    std::vector<std::vector<gpu_dest_arrival>> dest_arrs,
    gpu_fws_multimap_arrival_times arrival_times,
    uint16_t total_earliest_arrival,
    uint16_t* line_stop_count,
    std::size_t line_stop_count_size,
    gpu_nested_fws_multimap_transfers transfers,
    uint32_t* trip_to_line,
    std::size_t trip_to_line_size,
    uint16_t start_time,
    uint16_t* first_reachable_stop,
    std::size_t first_reachable_stop_size,
    uint64_t trip_count,
    gpu_queue_entry* initial_queue,
    std::size_t initial_queue_size,
    unsigned max_transfers) {

  gpu_device_pointers pointers;

  std::size_t dest_arrivals_size = dest_arrs.size();

  std::vector<std::size_t> dest_arrivals_index;
  //dest_arrivals_index.resize(dest_arrivals_size + 1);
  dest_arrivals_index.emplace_back(0);

  std::size_t dest_arrs_size = 0;
  for (auto dest_arr : dest_arrs) {
    dest_arrs_size += dest_arr.size();
    dest_arrivals_index.emplace_back(dest_arrs_size);
  }
  CUDA_ALLOC(pointers.dest_arrivals_device_,
             dest_arrs_size * sizeof(gpu_dest_arrival))
  for (auto i = 0; i < dest_arrivals_size; ++i) {
    std::vector<gpu_dest_arrival> dest_arr = dest_arrs[i];
    CUDA_COPY(&pointers.dest_arrivals_device_[dest_arrivals_index[i]],
              dest_arr.data(),
              dest_arr.size() * sizeof(gpu_dest_arrival),
              cudaMemcpyHostToDevice)
  }
  CUDA_ALLOC_COPY(pointers.dest_arrivals_index_device_,
                  dest_arrivals_index.data(),
                  dest_arrivals_index.size() * sizeof(std::size_t))

  std::vector<std::size_t> test2;
  test2.resize(dest_arrivals_index.size());
  CUDA_COPY(test2.data(), pointers.dest_arrivals_index_device_,
            dest_arrivals_index.size() * sizeof(std::size_t), cudaMemcpyDeviceToHost)
  for(auto i = 0; i < test2.size() - 1; ++i) {
    if (test2[i + 1] > test2[i]) {
      //std::printf("test");
      std::cout << i << std::endl;
    }
  }

  CUDA_ALLOC_COPY(pointers.arrival_times_device_.data_,
            arrival_times.data_,
            arrival_times.index_[*arrival_times.index_size_ - 1] * sizeof(uint16_t))
  CUDA_ALLOC_COPY(pointers.arrival_times_device_.index_,
            arrival_times.index_,
            *arrival_times.index_size_ * sizeof(std::size_t))
  CUDA_ALLOC_COPY(pointers.arrival_times_device_.index_size_,
            &arrival_times.index_size_,
            sizeof(std::size_t))

  CUDA_ALLOC_COPY(pointers.total_earliest_arrival_device_,
                  &total_earliest_arrival,
                  sizeof(uint16_t))

  CUDA_ALLOC_COPY(pointers.line_stop_count_device_,
                  line_stop_count,
                  line_stop_count_size * sizeof(uint16_t))

  CUDA_ALLOC_COPY(pointers.transfers_device.data_,
            transfers.data_,
            transfers.index_[*transfers.index_size_ - 1] * sizeof(gpu_tb_transfer))
  CUDA_ALLOC_COPY(pointers.transfers_device.base_index_,
            transfers.base_index_,
            *transfers.base_index_size_ * sizeof(std::size_t))
  CUDA_ALLOC_COPY(pointers.transfers_device.index_,
            transfers.index_,
            *transfers.index_size_ * sizeof(std::size_t))
  CUDA_ALLOC_COPY(pointers.transfers_device.base_index_size_,
            &transfers.base_index_size_,
            sizeof(std::size_t))
  CUDA_ALLOC_COPY(pointers.transfers_device.index_size_,
            &transfers.index_size_,
            sizeof(std::size_t))

  CUDA_ALLOC_COPY(pointers.trip_to_line_device_,
                  trip_to_line,
                  trip_to_line_size * sizeof(uint32_t))

  CUDA_ALLOC_COPY(pointers.start_time_device_,
                  &start_time,
                  sizeof(uint16_t))

  CUDA_ALLOC_COPY(pointers.first_reachable_stop_device_,
                  first_reachable_stop,
                  first_reachable_stop_size * sizeof(uint16_t))

  CUDA_ALLOC_COPY(pointers.trip_count_device_,
                  &trip_count,
                  sizeof(uint64_t))

  std::size_t queue_size = 30000000 * sizeof(gpu_queue_entry);
  CUDA_ALLOC(pointers.queue_device_,
             queue_size)
  CUDA_COPY(pointers.queue_device_,
            initial_queue,
            initial_queue_size * sizeof(gpu_queue_entry),
            cudaMemcpyHostToDevice)

  std::vector<gpu_queue_entry> test;
  test.resize(initial_queue_size);
  cudaMemcpy(test.data(), pointers.queue_device_,
             initial_queue_size * sizeof(gpu_queue_entry), cudaMemcpyDeviceToHost);

  std::vector<std::size_t> queue_index = {0, 3750000, 7500000, 11250000,
                                          15000000, 18750000, 22500000, 26250000};
  CUDA_ALLOC_COPY(pointers.queue_index_device_,
                  queue_index.data(),
                  queue_index.size() * sizeof(std::size_t))

  std::vector<unsigned> used_queue_sizes(max_transfers + 1);
  used_queue_sizes[0] = initial_queue_size;
  CUDA_ALLOC(pointers.queue_sizes_device_,
             used_queue_sizes.size() * sizeof(unsigned));
  CUDA_COPY(pointers.queue_sizes_device_,
            used_queue_sizes.data(),
            used_queue_sizes.size() * sizeof(unsigned),
            cudaMemcpyHostToDevice);

  // TODO(sarah): size enough?
  std::size_t result_set_alloc_num = 1000;
  CUDA_ALLOC(pointers.result_set_device_,
             result_set_alloc_num * sizeof(gpu_tb_journey))
  unsigned zero = 0;
  CUDA_ALLOC_COPY(pointers.result_set_size_device_,
                  &zero,
                  sizeof(unsigned))

  // TODO(sarah): same size as result set above (-> same question)
  std::vector<uint8_t> is_dominated_init(result_set_alloc_num);
  CUDA_ALLOC_COPY(pointers.is_dominated_device_,
                  is_dominated_init.data(),
                  result_set_alloc_num * sizeof(uint8_t))

  return pointers;
}

void free_on_device(gpu_device_pointers pointers) {
  cudaFree(pointers.dest_arrivals_device_);
  cudaFree(pointers.dest_arrivals_index_device_);
  cudaFree(pointers.arrival_times_device_.data_);
  cudaFree(pointers.arrival_times_device_.index_);
  cudaFree(pointers.arrival_times_device_.index_size_);
  cudaFree(pointers.total_earliest_arrival_device_);
  cudaFree(pointers.line_stop_count_device_);
  cudaFree(pointers.transfers_device.data_);
  cudaFree(pointers.transfers_device.base_index_);
  cudaFree(pointers.transfers_device.base_index_size_);
  cudaFree(pointers.transfers_device.index_);
  cudaFree(pointers.transfers_device.index_size_);
  cudaFree(pointers.trip_to_line_device_);
  cudaFree(pointers.start_time_device_);
  cudaFree(pointers.first_reachable_stop_device_);
  cudaFree(pointers.trip_count_device_);
  cudaFree(pointers.queue_device_);
  cudaFree(pointers.queue_index_device_);
  cudaFree(pointers.queue_sizes_device_);
  cudaFree(pointers.result_set_device_);
  cudaFree(pointers.result_set_size_device_);
  cudaFree(pointers.is_dominated_device_);
}

} // namespace motis::tripbased