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
  // TODO(sarah): max_travel_time limit check?
  unsigned set_entry_index = atomicAdd(pointers.result_set_size_device_, 1U);
  /*if (set_entry_index > 40000) {
    printf("test_dest\n");
  }*/
  uint16_t arrival_time
      = pointers.arrival_times_device_.data_[pointers.arrival_times_device_
                                                  .index_[queue_entry.trip_]
                                              + dest_arrival.stop_index_]
        + dest_arrival.fp_duration_;
  // TODO(sarah): following 3 lines only for version 2
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
  //printf("test_enq\n");
  // TODO(sarah): following line (maybe) only for Version 2
  auto const old_first_reachable = pointers.first_reachable_stop_device_[trip];
  // TODO(sarah): following line (maybe) only for Version 1
  /*auto const first_reachable = pointers.line_stop_count_device_
                                   [pointers.trip_to_line_device_[trip]];*/
  //printf("test_enq_2\n");
  // TODO(sarah): following if statement only for Version 2
  if (stop_idx < old_first_reachable) {
    unsigned queue_entry_index = atomicAdd(&pointers.queue_sizes_device_[transfers+1], 1U);
    assert(queue_entry_index < 3750000);
    /*if (queue_entry_index > 1000000) {
      printf("entry: %d\n", queue_entry_index);
    }*/
    // TODO(sarah): following ifelse (maybe) only for Version 1
    /*if (transfers % 2 == 0) {
      pointers.queue_1_device_[queue_entry_index]
          = gpu_queue_entry{trip, stop_idx, (uint16_t)old_first_reachable, prev_trip_seg};
    } else {
      pointers.queue_0_device_[queue_entry_index]
          = gpu_queue_entry{trip, stop_idx, (uint16_t)old_first_reachable, prev_trip_seg};
    }*/
    // TODO(sarah): following several lines (maybe) only for Version 2
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
  // TODO(sarah): following line (maybe) only for Version 2
  unsigned long idx = trip_seg + pointers.queue_index_device_[transfers];
  if (trip_seg >= pointers.queue_sizes_device_[transfers]) {
    return;
  }

  //printf("test_search %d\n", trip_seg);
  // TODO(sarah): following line (maybe) only for Version 2
  auto entry = pointers.queue_device_[idx];
  // TODO(sarah): following line and ifelse (maybe) only for Version 1
  // TODO(sarah): wip -> figure out how to do, cause could not work if nothing to ref to.
  //  Is ref even needed?
  //gpu_queue_entry entry;
  /*if (transfers % 2 == 0) {
    entry = pointers.queue_0_device_[trip_seg];
  } else {
    entry = pointers.queue_1_device_[trip_seg];
  }*/
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

  // TODO(sarah): following if statement only for Version 2
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
                                  gpu_device_pointers const pointers,
                                  gpu_device_query_pointers const query_pointers,
                                  uint16_t const start_time) {
  cuda_check()

  gpu_search_results results;
  results.gpu_final_queues_.resize(max_transfers);

  //std::cout << "testing1" << std::endl;

  std::vector<unsigned> queue_sizes(max_transfers);
  // TODO(sarah): <= or < ?
  for (auto transfers = 0U; transfers < max_transfers; ++transfers) {
    //if(transfers < max_transfers) {
      //std::cout << "loop " << transfers << std::endl;
      CUDA_COPY(&queue_sizes[transfers],
                &pointers.queue_sizes_device_[transfers],
                sizeof(unsigned),
                cudaMemcpyDeviceToHost)
      //std::cout << queue_sizes[transfers] << std::endl;

      /*if (transfers < max_transfers) {
        unsigned testsize;
        CUDA_COPY(&testsize,
                  &pointers.queue_sizes_device_[transfers+1],
                  sizeof(unsigned),
                  cudaMemcpyDeviceToHost)
        std::cout << testsize << std::endl;
      }*/
    //}

    // TODO(sarah): following 2 lines and ifelse (maybe) only for Version 1
    /*results.gpu_final_queues_[transfers].resize(queue_sizes[transfers]);
    if (transfers % 2 == 0) {
      //std::cout << "test 0" << std::endl;
      CUDA_COPY(results.gpu_final_queues_[transfers].data(),
                pointers.queue_0_device_,
                queue_sizes[transfers] * sizeof(gpu_queue_entry),
                cudaMemcpyDeviceToHost)
      if (transfers > 0) {
        cudaFree(pointers.queue_1_device_);
      }
    } else {
      //std::cout << "test 1" << std::endl;
      CUDA_COPY(results.gpu_final_queues_[transfers].data(),
                pointers.queue_1_device_,
                queue_sizes[transfers] * sizeof(gpu_queue_entry),
                cudaMemcpyDeviceToHost)
      cudaFree(pointers.queue_0_device_);
    }*/

    // TODO(sarah): wip for Version 1
    //if (transfers < max_transfers) {
      /*std::size_t next_queue_alloc = queue_sizes[transfers] * 100;
      //std::cout << "next queue: " << next_queue_alloc << std::endl;
      if (transfers % 2 == 0) {
        CUDA_ALLOC(pointers.queue_1_device_,
                   next_queue_alloc * sizeof(gpu_queue_entry))
      } else {
        CUDA_ALLOC(pointers.queue_0_device_,
                   next_queue_alloc * sizeof(gpu_queue_entry))
      }*/
    //}

    // TODO(sarah) testing
    /*std::vector<std::size_t> queue_index = {0, 3750000, 7500000, 11250000,
                                            15000000, 18750000, 22500000, 26250000};
    std::vector<gpu_queue_entry> test;
    test.resize(queue_sizes[transfers]);
    CUDA_COPY(test.data(),
              &pointers.queue_device_[queue_index[transfers]],
              queue_sizes[transfers] * sizeof(gpu_queue_entry),
              cudaMemcpyDeviceToHost)*/

    if (queue_sizes[transfers] == 0) {
      break;
    }
    unsigned thread_num = queue_sizes[transfers];
    unsigned block_num = 1;
    if(thread_num > 32) {
      block_num = (thread_num + 31) / 32;
      thread_num = 32;
    }
    /*std::cout << queue_sizes[transfers] << " ; " <<
        block_num << " , " << thread_num << std::endl;*/
    search<<<block_num, thread_num>>>
        (pointers, query_pointers, transfers, max_transfers, start_time);
    cudaDeviceSynchronize();

    cuda_check()
  }

  // TODO(sarah): testing
  /*std::vector<gpu_queue_entry> test;
  test.resize(queue_sizes[0]);
  cudaMemcpy(test.data(),
            pointers.queue_device_,
            queue_sizes[0] * sizeof(gpu_queue_entry),
            cudaMemcpyDeviceToHost);*/

  // TODO(sarah): following 2 lines only (maybe) for Version 2
  std::vector<std::size_t> queue_index = {0, 3750000, 7500000, 11250000,
                                          15000000, 18750000, 22500000, 26250000};;

  // TODO(sarah): following lines only (maybe) for Version 2
  for (std::size_t transfers = 0; transfers < max_transfers; ++transfers) {
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
  //std::cout << block_num << " , " << thread_num << std::endl;
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
  //cudaDeviceSynchronize();
  // TODO(sarah): ^ necessary?
  cuda_check()

  free_query_on_device(query_pointers);

  return results;
}

// TODO(sarah): cleanup unnecessary params? Maybe still needed for Version 1?
gpu_device_pointers allocate_and_copy_on_device(
    gpu_fws_multimap_arrival_times arrival_times,
    uint16_t* line_stop_count,
    std::size_t line_stop_count_size,
    gpu_nested_fws_multimap_transfers transfers,
    uint32_t* trip_to_line,
    std::size_t trip_to_line_size,
    uint64_t trip_count) {

  cuda_check()

  gpu_device_pointers pointers;

  /*std::size_t dest_arrivals_size = dest_arrs.size();

  std::vector<std::size_t> dest_arrivals_index;
  dest_arrivals_index.emplace_back(0);

  std::size_t dest_arrs_size = 0;
  for (auto dest_arr : dest_arrs) {
    dest_arrs_size += dest_arr.size();
    dest_arrivals_index.emplace_back(dest_arrs_size);
  }
  cuda_check();
  CUDA_ALLOC(pointers.dest_arrivals_device_,
             dest_arrs_size * sizeof(gpu_dest_arrival))
  cudaDeviceSynchronize();
  for (auto i = 0; i < dest_arrivals_size; ++i) {
    std::vector<gpu_dest_arrival> dest_arr = dest_arrs[i];
    CUDA_COPY(&pointers.dest_arrivals_device_[dest_arrivals_index[i]],
              dest_arr.data(),
              dest_arr.size() * sizeof(gpu_dest_arrival),
              cudaMemcpyHostToDevice)
  }
  CUDA_ALLOC_COPY(pointers.dest_arrivals_index_device_,
                  dest_arrivals_index.data(),
                  dest_arrivals_index.size() * sizeof(std::size_t))*/

  CUDA_ALLOC_COPY(pointers.arrival_times_device_.data_,
            arrival_times.data_,
            arrival_times.index_[*arrival_times.index_size_ - 1] * sizeof(uint16_t))
  CUDA_ALLOC_COPY(pointers.arrival_times_device_.index_,
            arrival_times.index_,
            *arrival_times.index_size_ * sizeof(std::size_t))
  CUDA_ALLOC_COPY(pointers.arrival_times_device_.index_size_,
            &arrival_times.index_size_,
            sizeof(std::size_t))

  // TODO(sarah): following 4 lines only for Version 2
  /*int total_earliest_init = std::numeric_limits<uint16_t>::max();
  CUDA_ALLOC_COPY(pointers.total_earliest_arrival_device_,
                  &total_earliest_init,
                  sizeof(int))*/
  /*CUDA_ALLOC_COPY(pointers.total_earliest_arrival_device_,
                  &total_earliest_arrival,
                  sizeof(uint16_t))*/
  CUDA_ALLOC(pointers.total_earliest_arrival_device_,
             sizeof(int))

  CUDA_ALLOC_COPY(pointers.line_stop_count_device_,
                  line_stop_count,
                  line_stop_count_size * sizeof(uint16_t))
  auto test5 = *transfers.index_size_;
  auto test6 = *transfers.base_index_size_;
  auto test7 = transfers.index_[*transfers.index_size_ - 1];

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

  /*CUDA_ALLOC_COPY(pointers.start_time_device_,
                  &start_time,
                  sizeof(uint16_t))*/

  // TODO(sarah): following 5 lines only for Version 2
  /*std::vector<int> first_reachable_init(first_reachable_stop_size,
                                        std::numeric_limits<uint16_t>::max());
  CUDA_ALLOC_COPY(pointers.first_reachable_stop_device_,
                  first_reachable_init.data(),
                  first_reachable_stop_size * sizeof(int))*/
  /*CUDA_ALLOC_COPY(pointers.first_reachable_stop_device_,
                  first_reachable_stop,
                  first_reachable_stop_size * sizeof(uint16_t))*/
  CUDA_ALLOC(pointers.first_reachable_stop_device_,
             trip_count * sizeof(int))

  CUDA_ALLOC_COPY(pointers.trip_count_device_,
                  &trip_count,
                  sizeof(uint64_t))

  // TODO(sarah): following 3 lines only (maybe) for Version 2
  std::size_t queue_size = 30000000 * sizeof(gpu_queue_entry);
  CUDA_ALLOC(pointers.queue_device_,
             queue_size)
  // TODO(sarah): following line needed?
  //cudaDeviceSynchronize();
  // TODO(sarah): following 4 lines only (maybe) for Version 2
  /*CUDA_COPY(pointers.queue_device_,
            initial_queue.data(),
            initial_queue.size() * sizeof(gpu_queue_entry),
            cudaMemcpyHostToDevice)*/
  /*CUDA_ALLOC_COPY(pointers.queue_0_device_,
                  initial_queue.data(),
                  initial_queue.size() * sizeof(gpu_queue_entry))*/

  // TODO(sarah): following 5 lines only (maybe) for Version 2
  std::vector<std::size_t> queue_index = {0, 3750000, 7500000, 11250000,
                                          15000000, 18750000, 22500000, 26250000};
  CUDA_ALLOC_COPY(pointers.queue_index_device_,
                  queue_index.data(),
                  queue_index.size() * sizeof(std::size_t))

  // TODO(sarah): WIP!!! (see notes) maybe for now version above for queues?
  /*std::vector<unsigned> used_queue_sizes(max_transfers, 0);
  used_queue_sizes[0] = initial_queue.size();*/
  CUDA_ALLOC(pointers.queue_sizes_device_,
             (MAX_TRANSFERS) * sizeof(unsigned));
  // TODO(sarah): following line needed?
  /*cudaDeviceSynchronize();
  CUDA_COPY(pointers.queue_sizes_device_,
            used_queue_sizes.data(),
            used_queue_sizes.size() * sizeof(unsigned),
            cudaMemcpyHostToDevice);*/

  // TODO(sarah) testing
  /*std::vector<gpu_queue_entry> test;
  test.resize(initial_queue.size());
  CUDA_COPY(test.data(),
            pointers.queue_device_,
            initial_queue.size() * sizeof(gpu_queue_entry),
            cudaMemcpyDeviceToHost)*/

  // TODO(sarah): size enough?
  std::size_t result_set_alloc_num = 50000;
  CUDA_ALLOC(pointers.result_set_device_,
             result_set_alloc_num * sizeof(gpu_tb_journey))
  /*unsigned zero = 0;
  CUDA_ALLOC_COPY(pointers.result_set_size_device_,
                  &zero,
                  sizeof(unsigned))*/
  CUDA_ALLOC(pointers.result_set_size_device_,
             sizeof(unsigned))

  // TODO(sarah): same size as result set above (-> same question)
  /*std::vector<uint8_t> is_dominated_init(result_set_alloc_num);
  CUDA_ALLOC_COPY(pointers.is_dominated_device_,
                  is_dominated_init.data(),
                  result_set_alloc_num * sizeof(uint8_t))*/
  CUDA_ALLOC(pointers.is_dominated_device_,
             result_set_alloc_num * sizeof(uint8_t))

  return pointers;
}

gpu_device_query_pointers allocate_and_copy_on_device_query(
    gpu_device_pointers pointers,
    std::vector<std::vector<gpu_dest_arrival>> dest_arrs,
    std::vector<gpu_queue_entry> initial_queue,
    uint64_t trip_count) {

  gpu_device_query_pointers query_pointers;

  std::size_t dest_arrivals_size = dest_arrs.size();
  std::vector<std::size_t> dest_arrivals_index;
  dest_arrivals_index.emplace_back(0);
  std::size_t dest_arrs_size = 0;
  for (auto dest_arr : dest_arrs) {
    dest_arrs_size += dest_arr.size();
    dest_arrivals_index.emplace_back(dest_arrs_size);
  }
  CUDA_ALLOC(query_pointers.dest_arrivals_device_,
             dest_arrs_size * sizeof(gpu_dest_arrival))
  //cudaDeviceSynchronize();
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

  std::vector<int> first_reachable_init(trip_count,
                                        std::numeric_limits<uint16_t>::max());
  CUDA_COPY(pointers.first_reachable_stop_device_,
            first_reachable_init.data(),
            trip_count * sizeof(int),
            cudaMemcpyHostToDevice)

  int total_earliest_init = std::numeric_limits<uint16_t>::max();
  CUDA_COPY(pointers.total_earliest_arrival_device_,
            &total_earliest_init,
            sizeof(int),
            cudaMemcpyHostToDevice)

  unsigned zero = 0;
  CUDA_COPY(pointers.result_set_size_device_,
            &zero,
            sizeof(unsigned),
            cudaMemcpyHostToDevice)

  std::size_t result_set_alloc_num = 50000;
  std::vector<uint8_t> is_dominated_init(result_set_alloc_num);
  CUDA_COPY(pointers.is_dominated_device_,
            is_dominated_init.data(),
            result_set_alloc_num * sizeof(uint8_t),
            cudaMemcpyHostToDevice)

  return query_pointers;
}

/*void free_on_device(gpu_device_pointers pointers) {
  cuda_check();
  /*cudaFree(pointers.dest_arrivals_device_);
  cudaFree(pointers.dest_arrivals_index_device_);
  cudaFree(pointers.arrival_times_device_.data_);
  cudaFree(pointers.arrival_times_device_.index_);
  cudaFree(pointers.arrival_times_device_.index_size_);*/
  /*CUDA_FREE(pointers.total_earliest_arrival_device_);
  CUDA_FREE(pointers.line_stop_count_device_);
  CUDA_FREE(pointers.transfers_device.data_);
  CUDA_FREE(pointers.transfers_device.base_index_);
  CUDA_FREE(pointers.transfers_device.base_index_size_);
  CUDA_FREE(pointers.transfers_device.index_);
  CUDA_FREE(pointers.transfers_device.index_size_);
  CUDA_FREE(pointers.trip_to_line_device_);
  //cudaFree(pointers.start_time_device_);
  CUDA_FREE(pointers.first_reachable_stop_device_);
  CUDA_FREE(pointers.trip_count_device_);
  // TODO(sarah): following line (maybe) only for Version 2
  CUDA_FREE(pointers.queue_device_);
  CUDA_FREE(pointers.queue_index_device_);
  // TODO(sarah): following 2 lines (maybe) only for Version 1
  //CUDA_FREE(pointers.queue_0_device_);
  //CUDA_FREE(pointers.queue_1_device_);
  CUDA_FREE(pointers.queue_sizes_device_);
  CUDA_FREE(pointers.result_set_device_);
  CUDA_FREE(pointers.result_set_size_device_);
  CUDA_FREE(pointers.is_dominated_device_);
}*/

/*gpu_device_query_pointers::~gpu_device_query_pointers() {
  CUDA_FREE(dest_arrivals_device_)
  CUDA_FREE(dest_arrivals_index_device_)
}*/

void free_query_on_device(gpu_device_query_pointers query_pointers) {
  CUDA_FREE(query_pointers.dest_arrivals_device_)
  CUDA_FREE(query_pointers.dest_arrivals_index_device_)
}

} // namespace motis::tripbased