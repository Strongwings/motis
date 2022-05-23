#pragma once

#include <queue>
#include <vector>
#include <array>

#include "motis/tripbased/tb_search_common.h"

namespace motis::tripbased {

void search_fwd_gpu(unsigned max_transfers,
                    std::array<std::vector<queue_entry>, 8> queues);

} // namespace motis::tripbased