#include "motis/tripbased/gpu/gpu_tripbased.h"

#include <cstdio>

namespace motis::tripbased {

#define cucheck_dev(call)                                    \
  {                                                          \
    cudaError_t cucheck_err = (call);                        \
    if (cucheck_err != cudaSuccess) {                        \
      const char* err_str = cudaGetErrorString(cucheck_err); \
      printf("%s (%d): %s\n", __FILE__, __LINE__, err_str);  \
    }                                                        \
  }

#define cuda_check() \
  { cucheck_dev(cudaGetLastError()); }

#define CUDA_ALLOC(target, size) \
  cudaMalloc((void**) &(target), size); \
  cuda_check()

#define CUDA_COPY(target, source, size, copy_type) \
  cudaMemcpy(target, source, size, copy_type);     \
  cuda_check()

#define CUDA_ALLOC_COPY(target, source, size)   \
  CUDA_ALLOC(target, size)                      \
  CUDA_COPY(target, source, size, cudaMemcpyHostToDevice)

#define CUDA_FREE(target) \
  cudaFree(target);       \
  cuda_check()

__global__ void check_dominated(gpu_device_pointers pointers) {
  unsigned idx1 = blockIdx.x * 32 + threadIdx.x;
  unsigned idx2 = blockIdx.y * 32 + threadIdx.y;
  if (idx1 >= idx2 || idx2 >= *pointers.result_set_size_device_) {
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
    if(journey1.arrival_time_ >= journey2.arrival_time_) {
      pointers.is_dominated_device_[idx1] = 1;
    }
  } else { // journey2.transfers_ < journey2.transfers_
    if (journey1.arrival_time_ <= journey2.arrival_time_) {
      pointers.is_dominated_device_[idx2] = 1;
    }
  }
}

__device__ void destination_reached(gpu_device_pointers pointers,
                                    gpu_queue_entry queue_entry,
                                    unsigned entry_num,
                                    gpu_dest_arrival dest_arrival,
                                    unsigned transfers,
                                    uint16_t start_time) {
  unsigned set_entry_index = atomicAdd(pointers.result_set_size_device_, 1U);
  uint16_t arrival_time
      = pointers.arrival_times_device_.data_[pointers.arrival_times_device_
                                                  .index_[queue_entry.trip_]
                                              + dest_arrival.stop_index_]
        + dest_arrival.fp_duration_;
  if (arrival_time < *pointers.total_earliest_arrival_device_) {
    atomicMin(pointers.total_earliest_arrival_device_, arrival_time);
  }
  pointers.result_set_device_[set_entry_index]
      = gpu_tb_journey{start_time,
                       arrival_time,
                       transfers,
                       transfers + 1,
                       dest_arrival.fp_to_station_id_,
                       dest_arrival,
                       entry_num};
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
        = gpu_queue_entry{trip, stop_idx, (uint16_t)old_first_reachable, prev_trip_seg};
    auto const line = pointers.trip_to_line_device_[trip];
    for (uint32_t t = trip;
         t < *pointers.trip_count_device_ && pointers.trip_to_line_device_[t] == line;
         ++t) {
      if (stop_idx < pointers.first_reachable_stop_device_[t]) {
        atomicMin(&pointers.first_reachable_stop_device_[t], stop_idx);
      }
    }
  }
}

__global__ void search(gpu_device_pointers const pointers,
                       gpu_device_query_pointers query_pointers,
                       unsigned const transfers,
                       unsigned const max_transfers,
                       uint16_t start_time) {
  unsigned trip_seg = blockIdx.x * 32 + threadIdx.x;
  unsigned long idx = trip_seg + pointers.queue_index_device_[transfers];
  if (trip_seg >= pointers.queue_sizes_device_[transfers]) {
    return;
  }

  auto entry = pointers.queue_device_[idx];
  auto const line = pointers.trip_to_line_device_[entry.trip_];

  auto const dest_arrs_size = query_pointers.dest_arrivals_index_device_[line + 1]
                                  - query_pointers.dest_arrivals_index_device_[line];
  if (dest_arrs_size > 0) {
    for (auto i = query_pointers.dest_arrivals_index_device_[line];
         i < query_pointers.dest_arrivals_index_device_[line + 1]; ++i) {
      if (entry.from_stop_index_ < query_pointers.dest_arrivals_device_[i].stop_index_) {
        destination_reached(pointers,
                            entry,
                            trip_seg,
                            query_pointers.dest_arrivals_device_[i],
                            transfers,
                            start_time);
      }
    }
  }

  if (transfers+1 >= max_transfers) {
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

gpu_search_results search_fwd_gpu(gpu_device_pointers const& pointers,
                                  gpu_device_query_pointers const& query_pointers,
                                  uint16_t start_time,
                                  std::vector<std::size_t> const& queue_index) {
  cuda_check()

  gpu_search_results results;
  results.gpu_final_queues_.resize(MAX_TRANSFERS);

  std::vector<unsigned> queue_sizes(MAX_TRANSFERS);
  for (auto transfers = 0U; transfers < MAX_TRANSFERS; ++transfers) {
    CUDA_COPY(&queue_sizes[transfers],
              &pointers.queue_sizes_device_[transfers],
              sizeof(unsigned),
              cudaMemcpyDeviceToHost)

    if (queue_sizes[transfers] == 0) {
      break;
    }

    unsigned thread_num = queue_sizes[transfers];
    unsigned block_num = 1;
    if(thread_num > 32) {
      block_num = (thread_num + 31) / 32;
      thread_num = 32;
    }
    search<<<block_num, thread_num>>>
        (pointers, query_pointers, transfers, MAX_TRANSFERS, start_time);
    cudaDeviceSynchronize();

    cuda_check()
  }

  for (std::size_t transfers = 0; transfers < MAX_TRANSFERS; ++transfers) {
    if (queue_sizes[transfers] == 0) {
      break;
    }

    results.gpu_final_queues_[transfers].resize(queue_sizes[transfers]);
    CUDA_COPY(results.gpu_final_queues_[transfers].data(),
              &pointers.queue_device_[queue_index[transfers]],
              queue_sizes[transfers] * sizeof(gpu_queue_entry),
              cudaMemcpyDeviceToHost)
  }

  unsigned result_set_size;
  CUDA_COPY(&result_set_size,
            pointers.result_set_size_device_,
            sizeof(unsigned),
            cudaMemcpyDeviceToHost)

  if (result_set_size == 0) {
    free_query_on_device(query_pointers);
    return results;
  }

  unsigned thread_num = result_set_size;
  unsigned block_num = 1;
  if(thread_num > 32) {
    block_num = (thread_num + 31) / 32;
    thread_num = 32;
  }
  dim3 block_dim(thread_num, thread_num, 1);
  dim3 grid_dim(block_num, block_num, 1);
  check_dominated<<<grid_dim, block_dim>>>(pointers);
  cudaDeviceSynchronize();
  cuda_check()

  results.gpu_result_journeys_.resize(result_set_size);
  CUDA_COPY(results.gpu_result_journeys_.data(),
            pointers.result_set_device_,
            result_set_size * sizeof(gpu_tb_journey),
            cudaMemcpyDeviceToHost)

  results.gpu_is_dominated_.resize(result_set_size);
  CUDA_COPY(results.gpu_is_dominated_.data(),
            pointers.is_dominated_device_,
            result_set_size * sizeof(uint8_t),
            cudaMemcpyDeviceToHost)

  cuda_check()

  free_query_on_device(query_pointers);

  return results;
}

gpu_device_pointers allocate_and_copy_on_device(
    gpu_fws_multimap_arrival_times const& arrival_times,
    uint16_t* line_stop_count,
    std::size_t const& line_stop_count_size,
    gpu_nested_fws_multimap_transfers const& transfers,
    uint32_t* trip_to_line,
    std::size_t const& trip_to_line_size,
    uint64_t const& trip_count,
    std::size_t const& queue_size,
    std::vector<std::size_t> const& queue_index,
    std::size_t const& result_set_alloc_num) {

  gpu_device_pointers pointers;

  CUDA_ALLOC_COPY(pointers.arrival_times_device_.data_,
            arrival_times.data_,
            arrival_times.index_[*arrival_times.index_size_ - 1] * sizeof(uint16_t))
  CUDA_ALLOC_COPY(pointers.arrival_times_device_.index_,
            arrival_times.index_,
            *arrival_times.index_size_ * sizeof(std::size_t))
  CUDA_ALLOC_COPY(pointers.arrival_times_device_.index_size_,
            &arrival_times.index_size_,
            sizeof(std::size_t))

  CUDA_ALLOC(pointers.total_earliest_arrival_device_,
             sizeof(int))

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

  CUDA_ALLOC(pointers.first_reachable_stop_device_,
             trip_count * sizeof(int))

  CUDA_ALLOC_COPY(pointers.trip_count_device_,
                  &trip_count,
                  sizeof(uint64_t))

  CUDA_ALLOC(pointers.queue_device_,
             queue_size * sizeof(gpu_queue_entry))
  CUDA_ALLOC_COPY(pointers.queue_index_device_,
                  queue_index.data(),
                  queue_index.size() * sizeof(std::size_t))

  CUDA_ALLOC(pointers.queue_sizes_device_,
             (MAX_TRANSFERS) * sizeof(unsigned));

  CUDA_ALLOC(pointers.result_set_device_,
             result_set_alloc_num * sizeof(gpu_tb_journey))
  CUDA_ALLOC(pointers.result_set_size_device_,
             sizeof(unsigned))

  CUDA_ALLOC(pointers.is_dominated_device_,
             result_set_alloc_num * sizeof(uint8_t))

  return pointers;
}

gpu_device_query_pointers allocate_and_copy_on_device_query(
    gpu_device_pointers const& pointers,
    std::vector<std::vector<gpu_dest_arrival>> const& dest_arrs,
    std::vector<gpu_queue_entry> const& initial_queue,
    uint64_t const& trip_count,
    std::size_t const& result_set_alloc_num) {

  gpu_device_query_pointers query_pointers;

  std::size_t dest_arrivals_size = dest_arrs.size();
  std::vector<std::size_t> dest_arrivals_index;
  dest_arrivals_index.emplace_back(0);
  std::size_t dest_arrs_size = 0;
  for (auto const& dest_arr : dest_arrs) {
    dest_arrs_size += dest_arr.size();
    dest_arrivals_index.emplace_back(dest_arrs_size);
  }
  CUDA_ALLOC(query_pointers.dest_arrivals_device_,
             dest_arrs_size * sizeof(gpu_dest_arrival))
  for (auto i = 0; i < dest_arrivals_size; ++i) {
    std::vector<gpu_dest_arrival> dest_arr = dest_arrs[i];
    CUDA_COPY(&query_pointers.dest_arrivals_device_[dest_arrivals_index[i]],
              dest_arr.data(),
              dest_arr.size() * sizeof(gpu_dest_arrival),
              cudaMemcpyHostToDevice)
  }
  CUDA_ALLOC_COPY(query_pointers.dest_arrivals_index_device_,
                  dest_arrivals_index.data(),
                  dest_arrivals_index.size() * sizeof(std::size_t))

  std::vector<unsigned> used_queue_sizes(MAX_TRANSFERS, 0);
  used_queue_sizes[0] = initial_queue.size();
  CUDA_COPY(pointers.queue_sizes_device_,
            used_queue_sizes.data(),
            used_queue_sizes.size() * sizeof(unsigned),
            cudaMemcpyHostToDevice);
  CUDA_COPY(pointers.queue_device_,
            initial_queue.data(),
            initial_queue.size() * sizeof(gpu_queue_entry),
            cudaMemcpyHostToDevice)

  std::vector<int> const first_reachable_init(trip_count,
                                        std::numeric_limits<uint16_t>::max());
  CUDA_COPY(pointers.first_reachable_stop_device_,
            first_reachable_init.data(),
            trip_count * sizeof(int),
            cudaMemcpyHostToDevice)

  int const total_earliest_init = std::numeric_limits<uint16_t>::max();
  CUDA_COPY(pointers.total_earliest_arrival_device_,
            &total_earliest_init,
            sizeof(int),
            cudaMemcpyHostToDevice)

  unsigned zero = 0;
  CUDA_COPY(pointers.result_set_size_device_,
            &zero,
            sizeof(unsigned),
            cudaMemcpyHostToDevice)

  std::vector<uint8_t> const is_dominated_init(result_set_alloc_num);
  CUDA_COPY(pointers.is_dominated_device_,
            is_dominated_init.data(),
            result_set_alloc_num * sizeof(uint8_t),
            cudaMemcpyHostToDevice)

  return query_pointers;
}

void free_query_on_device(gpu_device_query_pointers query_pointers) {
  CUDA_FREE(query_pointers.dest_arrivals_device_)
  CUDA_FREE(query_pointers.dest_arrivals_index_device_)
}

} // namespace motis::tripbased