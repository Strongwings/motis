#pragma once

#include <cstdint>
#include <array>
#include <queue>
#include <vector>

#include "motis/tripbased/tb_search_common.h"

namespace motis::tripbased {

struct gpu_dest_arrival {
  uint32_t line_id_;
  uint16_t stop_index_;
  uint32_t fp_from_station_id_, fp_to_station_id_;
  unsigned fp_duration_;
};

struct gpu_fws_multimap_arrival_times {
  uint16_t* data_;
  std::size_t* index_;
  std::size_t* index_size_;
};

struct gpu_tb_transfer {
  uint32_t to_trip;
  uint16_t to_stop_idx;
};

struct gpu_nested_fws_multimap_transfers {
  gpu_tb_transfer* data_;
  std::size_t* base_index_;
  std::size_t* base_index_size_;
  std::size_t* index_;
  std::size_t* index_size_;
};

struct gpu_queue_entry {
  uint32_t trip_;
  uint16_t from_stop_index_, to_stop_index_;
  std::size_t previous_trip_segment_;
};

struct gpu_tb_journey {
  uint16_t start_time_, arrival_time_;
  unsigned transfers_, transports_;
  uint32_t destination_station_;
  gpu_dest_arrival destination_arrival_;
  unsigned final_queue_entry_;
};

struct gpu_device_pointers {
  gpu_fws_multimap_arrival_times arrival_times_device_;
  int* total_earliest_arrival_device_;
  uint16_t* line_stop_count_device_;
  gpu_nested_fws_multimap_transfers transfers_device;
  uint32_t* trip_to_line_device_;
  int* first_reachable_stop_device_;
  uint64_t* trip_count_device_;
  gpu_queue_entry* queue_device_;
  std::size_t* queue_index_device_;
  unsigned* queue_sizes_device_;
  gpu_tb_journey* result_set_device_;
  unsigned* result_set_size_device_;
  uint8_t* is_dominated_device_;
};

struct gpu_device_query_pointers {
  gpu_dest_arrival* dest_arrivals_device_;
  std::size_t* dest_arrivals_index_device_;
};



struct gpu_search_results {
  std::vector<std::vector<gpu_queue_entry>> gpu_final_queues_;
  std::vector<gpu_tb_journey> gpu_result_journeys_;
  std::vector<uint8_t> gpu_is_dominated_;
};

gpu_search_results search_fwd_gpu(gpu_device_pointers const pointers,
                                  gpu_device_query_pointers const query_pointers,
                                  uint16_t const start_time,
                                  std::vector<std::size_t> queue_index);

gpu_device_pointers allocate_and_copy_on_device(
    gpu_fws_multimap_arrival_times arrival_times,
    uint16_t* line_stop_count,
    std::size_t line_stop_count_size,
    gpu_nested_fws_multimap_transfers transfers,
    uint32_t* trip_to_line,
    std::size_t trip_to_line_size,
    uint64_t trip_count,
    std::size_t queue_size,
    std::vector<std::size_t> queue_index,
    std::size_t result_set_alloc_num);

gpu_device_query_pointers allocate_and_copy_on_device_query(
    gpu_device_pointers pointers,
    std::vector<std::vector<gpu_dest_arrival>> dest_arrs,
    std::vector<gpu_queue_entry> initial_queue,
    uint64_t trip_count,
    std::size_t result_set_alloc_num);

void free_query_on_device(gpu_device_query_pointers query_pointers);

} // namespace motis::tripbased