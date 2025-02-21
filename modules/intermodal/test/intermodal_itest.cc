#include "gtest/gtest.h"

#include <string>

#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/idl.h"

#include "geo/latlng.h"

#include "motis/core/journey/journey.h"
#include "motis/core/journey/message_to_journeys.h"
#include "motis/module/message.h"
#include "motis/test/motis_instance_test.h"
#include "motis/test/schedule/simple_realtime.h"

using namespace geo;
using namespace flatbuffers;
using namespace motis::osrm;
using namespace motis::test;
using namespace motis::test::schedule;
using namespace motis::module;
using namespace motis::routing;
using namespace motis::intermodal;
using motis::test::schedule::simple_realtime::dataset_opt;

namespace motis::intermodal {

struct intermodal_itest : public motis_instance_test {
  intermodal_itest()
      : motis::test::motis_instance_test(dataset_opt,
                                         {"intermodal", "routing", "lookup"}) {
    instance_->register_op(
        "/osrm/one_to_many",
        [](msg_ptr const& msg) {
          auto const req = motis_content(OSRMOneToManyRequest, msg);
          auto one = latlng{req->one()->lat(), req->one()->lng()};

          std::vector<Cost> costs;
          for (auto const& loc : *req->many()) {
            auto dist = distance(one, {loc->lat(), loc->lng()});
            costs.emplace_back(dist / WALK_SPEED, dist);
          }

          message_creator mc;
          mc.create_and_finish(
              MsgContent_OSRMOneToManyResponse,
              CreateOSRMOneToManyResponse(mc, mc.CreateVectorOfStructs(costs))
                  .Union());
          return make_msg(mc);
        },
        {});
  }
};

TEST_F(intermodal_itest, forward) {
  //  Heidelberg Hbf -> Bensheim ( departure: 2015-11-24 13:30:00 )
  auto json = R"(
    {
      "destination": {
        "type": "Module",
        "target": "/intermodal"
      },
      "content_type": "IntermodalRoutingRequest",
      "content": {
        "start_type": "IntermodalOntripStart",
        "start": {
          "position": { "lat": 49.4047178, "lng": 8.6768716},
          "departure_time": 1448368200
        },
        "start_modes": [{
          "mode_type": "Foot",
          "mode": { "max_duration": 600 }
        }],
        "destination_type": "InputPosition",
        "destination": { "lat": 49.6801332, "lng": 8.6200666},
        "destination_modes":  [{
          "mode_type": "Foot",
          "mode": { "max_duration": 600 }
        },{
          "mode_type": "Bike",
          "mode": { "max_duration": 600 }
        }],
        "search_type": "SingleCriterion"
      }
    }
  )";

  auto res = call(make_msg(json));
  auto content = motis_content(RoutingResponse, res);

  ASSERT_EQ(1, content->connections()->size());
  auto const& stops = content->connections()->Get(0)->stops();

  ASSERT_EQ(5, stops->size());

  auto const& start = stops->Get(0);
  EXPECT_STREQ(STATION_START, start->station()->id()->c_str());
  EXPECT_DOUBLE_EQ(49.4047178, start->station()->pos()->lat());
  EXPECT_DOUBLE_EQ(8.6768716, start->station()->pos()->lng());

  auto const& first_station = stops->Get(1);
  EXPECT_STREQ("8000156", first_station->station()->id()->c_str());
  EXPECT_DOUBLE_EQ(49.403567, first_station->station()->pos()->lat());
  EXPECT_DOUBLE_EQ(8.675442, first_station->station()->pos()->lng());

  auto const& last_station = stops->Get(3);
  EXPECT_STREQ("8000031", last_station->station()->id()->c_str());
  EXPECT_DOUBLE_EQ(49.681329, last_station->station()->pos()->lat());
  EXPECT_DOUBLE_EQ(8.616717, last_station->station()->pos()->lng());

  auto const& end = stops->Get(4);
  EXPECT_STREQ(STATION_END, end->station()->id()->c_str());
  EXPECT_DOUBLE_EQ(49.6801332, end->station()->pos()->lat());
  EXPECT_DOUBLE_EQ(8.6200666, end->station()->pos()->lng());

  auto const& transports = content->connections()->Get(0)->transports();
  ASSERT_EQ(3, transports->size());

  ASSERT_EQ(Move_Walk, transports->Get(0)->move_type());
  ASSERT_STREQ("foot",
               reinterpret_cast<motis::Walk const*>(transports->Get(0)->move())
                   ->mumo_type()
                   ->c_str());

  ASSERT_EQ(Move_Walk, transports->Get(0)->move_type());
  ASSERT_STREQ("bike",
               reinterpret_cast<motis::Walk const*>(transports->Get(2)->move())
                   ->mumo_type()
                   ->c_str());
}

TEST_F(intermodal_itest, backward) {
  //  Heidelberg Hbf -> Bensheim ( arrival: 2015-11-24 14:30:00 )
  auto json = R"(
    {
      "destination": {
        "type": "Module",
        "target": "/intermodal"
      },
      "content_type": "IntermodalRoutingRequest",
      "content": {
        "start_type": "IntermodalOntripStart",
        "start": {
          "position": { "lat": 49.6801332, "lng": 8.6200666 },
          "departure_time": 1448371800
        },
        "start_modes": [{
          "mode_type": "Foot",
          "mode": { "max_duration": 600 }
        }],
        "destination_type": "InputPosition",
        "destination": { "lat": 49.4047178, "lng": 8.6768716 },
        "destination_modes":  [{
          "mode_type": "Foot",
          "mode": { "max_duration": 600 }
        }],
        "search_type": "SingleCriterion",
        "search_dir": "Backward"
      }
    }
  )";

  auto res = call(make_msg(json));
  auto content = motis_content(RoutingResponse, res);

  ASSERT_EQ(1, content->connections()->size());
  auto const& stops = content->connections()->Get(0)->stops();

  ASSERT_EQ(5, stops->size());

  auto const& start = stops->Get(0);
  EXPECT_STREQ(STATION_END, start->station()->id()->c_str());
  EXPECT_DOUBLE_EQ(49.4047178, start->station()->pos()->lat());
  EXPECT_DOUBLE_EQ(8.6768716, start->station()->pos()->lng());

  auto const& first_station = stops->Get(1);
  EXPECT_STREQ("8000156", first_station->station()->id()->c_str());
  EXPECT_DOUBLE_EQ(49.403567, first_station->station()->pos()->lat());
  EXPECT_DOUBLE_EQ(8.675442, first_station->station()->pos()->lng());

  auto const& last_station = stops->Get(3);
  EXPECT_STREQ("8000031", last_station->station()->id()->c_str());
  EXPECT_DOUBLE_EQ(49.681329, last_station->station()->pos()->lat());
  EXPECT_DOUBLE_EQ(8.616717, last_station->station()->pos()->lng());

  auto const& end = stops->Get(4);
  EXPECT_STREQ(STATION_START, end->station()->id()->c_str());
  EXPECT_DOUBLE_EQ(49.6801332, end->station()->pos()->lat());
  EXPECT_DOUBLE_EQ(8.6200666, end->station()->pos()->lng());
}

TEST_F(intermodal_itest, not_so_intermodal) {
  //  Heidelberg Hbf -> Bensheim ( departure: 2015-11-24 13:30:00 )
  auto json = R"(
    {
      "destination": {
        "type": "Module",
        "target": "/intermodal"
      },
      "content_type": "IntermodalRoutingRequest",
      "content": {
        "start_type": "OntripStationStart",
        "start": {
          "station": { "id": "8000156", "name": "" },
          "departure_time": 1448368200
        },
        "start_modes": [],
        "destination_type": "InputStation",
        "destination": { "id": "8000031", "name": "" },
        "destination_modes": [],
        "search_type": "SingleCriterion"
      }
    }
  )";

  auto res = call(make_msg(json));
  auto content = motis_content(RoutingResponse, res);

  ASSERT_EQ(1, content->connections()->size());
  auto const& stops = content->connections()->Get(0)->stops();

  ASSERT_EQ(3, stops->size());

  auto const& first_station = stops->Get(0);
  EXPECT_STREQ("8000156", first_station->station()->id()->c_str());
  EXPECT_DOUBLE_EQ(49.403567, first_station->station()->pos()->lat());
  EXPECT_DOUBLE_EQ(8.675442, first_station->station()->pos()->lng());

  auto const& last_station = stops->Get(2);
  EXPECT_STREQ("8000031", last_station->station()->id()->c_str());
  EXPECT_DOUBLE_EQ(49.681329, last_station->station()->pos()->lat());
  EXPECT_DOUBLE_EQ(8.616717, last_station->station()->pos()->lng());
}

}  // namespace motis::intermodal
