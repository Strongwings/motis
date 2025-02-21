#pragma once

#include <ctime>
#include <string>
#include <utility>

#include "boost/program_options.hpp"

#include "conf/configuration.h"

#include "motis/loader/loader_options.h"

namespace motis::bootstrap {

struct dataset_settings : public conf::configuration,
                          public motis::loader::loader_options {
  dataset_settings() : configuration("Dataset Settings", "dataset") {
    param(dataset_, "path", "MOTIS Dataset root");
    param(dataset_prefix_, "prefix",
          "station id prefixes (one per path or empty).");
    param(graph_path_, "graph_path",
          "path to read&write the serialized graph from/to "
          "(\"default\": generated from settings)");
    param(write_serialized_, "write_serialized", "Ignore serialized dataset");
    param(write_graph_, "write_graph", "Write bianry schedule graph");
    param(read_graph_, "read_graph", "Read binary schedule graph");
    param(read_graph_mmap_, "read_graph_mmap", "Read using memory mapped file");
    param(cache_graph_, "cache_graph", "Cache binary schedule graph");
    param(apply_rules_, "apply_rules",
          "Apply special rules (through-services, merge-split-services)");
    param(adjust_footpaths_, "adjust_footpaths",
          "Remove footpaths if they do not fit an assumed average speed");
    param(expand_footpaths_, "expand_footpaths",
          "Calculate expanded footpaths");
    param(use_platforms_, "use_platforms",
          "Use separate interchange times for trips stopping at the same "
          "platform");
    param(schedule_begin_, "begin",
          "schedule interval begin (TODAY or YYYYMMDD)");
    param(num_days_, "num_days", "number of days");
    param(planned_transfer_delta_, "planned_transfer_delta",
          "Max. difference between feeder arrival and connector departure for "
          "waiting time rules (minutes)");
    param(wzr_classes_path_, "wzr_classes_path",
          "waiting time rules class mapping");
    param(wzr_matrix_path_, "wzr_matrix_path", "waiting time matrix");
    param(no_local_transport_, "no_local_transport",
          "don't load local transport");
    param(debug_broken_trips_, "debug_broken_trips",
          "print debug information for broken trips");
  }
};

}  // namespace motis::bootstrap
