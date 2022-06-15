#pragma once

#include <queue>
#include <vector>
#include <array>
#include <cstdint>

namespace motis::tripbased {

struct gpu_dest_arrival {
  uint32_t line_id_;
  uint16_t stop_;
  uint32_t fp_from_station_id_, fp_to_station_id_;
  unsigned fp_duration_;
};

struct gpu_map_entry_arrival_times {
  uint16_t* data_;
  std::size_t start_index_, end_index_;
};

struct gpu_fws_multimap_arrival_times {
  uint16_t* data_;
  std::size_t* index_;
  std::size_t index_size_;

  inline gpu_map_entry_arrival_times at(std::size_t idx) {
    return {data_, index_[idx] - index_[idx + 1]};
  }
};

struct gpu_tb_transfer {
  uint32_t to_trip;
  uint16_t to_stop_idx;
};

struct gpu_map_entry_transfers {
  gpu_tb_transfer* data_;
  std::size_t start_index_, end_index_;
};

struct gpu_nested_fws_multimap_transfers {
  gpu_tb_transfer* data_;
  std::size_t* base_index_;
  std::size_t base_index_size_;
  std::size_t* index_;
  std::size_t index_size_;

  inline gpu_map_entry_transfers at(std::size_t outer_index,
                                    std::size_t inner_index) {
    auto const start_idx = index_[base_index_[outer_index] + inner_index];
    auto const end_idx = index_[base_index_[outer_index] + inner_index + 1];
    return {data_, start_idx, end_idx};
  }
};

struct gpu_device_ptrs {
  gpu_dest_arrival* dest_arrivals_device_;
  std::size_t* dest_arrivals_index_device_;
  std::size_t* dest_arrivals_size_device_;
  gpu_fws_multimap_arrival_times* arrival_times_device_;
  uint16_t* total_earliest_arrival_device_;
  uint16_t* line_stop_count_device_;
  std::size_t* line_stop_count_size_device_;
  gpu_nested_fws_multimap_transfers* transfers_device;
};

void search_fwd_gpu(unsigned max_transfers,
                    gpu_device_ptrs ptrs);

gpu_device_ptrs allocate_and_copy_on_device(
    std::vector<std::vector<gpu_dest_arrival>> dest_arrs,
    gpu_fws_multimap_arrival_times arrival_times,
    uint16_t total_earliest_arrival,
    uint16_t* line_stop_count,
    std::size_t line_stop_count_size,
    gpu_nested_fws_multimap_transfers transfers);

void free_on_device(gpu_device_ptrs ptrs);

} // namespace motis::tripbased