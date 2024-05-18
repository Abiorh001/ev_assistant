from agents.agent_tools.charge_station_locator import ChargePointsLocatorTool
from agents.agent_tools.constant import DISTANCE_IN_KM
from agents.agent_tools.utils.charge_station_availability import \
    is_charge_station_available
from agents.agent_tools.utils.google_location import (
    get_distance_and_route_info, get_latitude_longitude)
from core.settings import logger


class EvTripPlannerTool:
    def __init__(self):
        self.charge_points_locator = ChargePointsLocatorTool()

    def segment_trip(self, user_address: str, user_destination_address: str):
        # Calculate the total distance between the user's location and the
        # destination
        total_distance = get_distance_and_route_info(
            origin=user_address, destination=user_destination_address
        )[0]
        # convert to km
        if total_distance is not None:
            total_distance = float(total_distance.split()[0]) * 1.60934
            # Determine the segment length based on the total distance
            if total_distance < 10:
                segment_length = total_distance / 2
            elif total_distance >= 10 and total_distance < 50:
                segment_length = 5
            elif total_distance >= 50 and total_distance < 100:
                segment_length = 20
            else:
                segment_length = 50
            # Calculate the number of segments
            num_segments = int(total_distance / segment_length) + 1
        else:
            logger.error(
                "Error calculating total distance, setting default values to 0"
            )
            total_distance, num_segments, segment_length = 0, 0, 0
        return total_distance, num_segments, segment_length

    def get_charge_point(self, user_address: str, user_destination_address: str):
        # Get the latitude and longitude of the user's address
        user_latitude, user_longitude = get_latitude_longitude(user_address)
        # Initialize start point as user location
        start_latitude, start_longitude = user_latitude, user_longitude

        # Iterate over each segment to find charging points
        total_distance, num_segments, segment_length = self.segment_trip(
            user_address, user_destination_address
        )
        if any(item == 0 for item in (total_distance, num_segments, segment_length)):
            logger.error(
                "Error in calculating the trip segments, returning empty charge points list"
            )
            self.charge_points_locator._charge_stations = []
            return self.charge_points_locator._charge_stations

        for segment in range(1, num_segments + 1):
            # Calculate the distance for the current segment
            segment_distance = min(
                segment_length, total_distance - (segment - 1) * segment_length
            )
            segment_distance = round(segment_distance)
            # Get the closest EV charging stations to the user's location for the current segment
            charge_points = self.charge_points_locator.get_closest_charge_stations(
                start_latitude,
                start_longitude,
                max_results=5,
                max_distance=segment_distance,
            )

            # Append the results to the list of charging points if the UUID is not already present
            for charge_point in charge_points:
                charge_point_uuid = charge_point.get("UUID")
                if charge_point_uuid and charge_point_uuid not in [
                    cp["UUID"] for cp in self.charge_points_locator._charge_stations
                ]:
                    address_info = charge_point.get("AddressInfo", {})
                    charge_point_address = address_info.get("AddressLine1", "")
                    charge_point_city = address_info.get("Town", "")
                    charge_point_state = address_info.get("StateOrProvince", "")
                    charge_point_country = address_info.get("Country", {}).get(
                        "Title", ""
                    )
                    charge_point_full_address = ", ".join(
                        filter(
                            None,
                            [
                                charge_point_address,
                                charge_point_city,
                                charge_point_state,
                                charge_point_country,
                            ],
                        )
                    )
                    distance, duration, steps = get_distance_and_route_info(
                        user_address, charge_point_full_address
                    )
                    distance = round(
                        float(distance.split()[0]) * 1.60934, 2
                    )  # Convert from mile to km
                    distance_str = f"{distance} {DISTANCE_IN_KM}"
                    charge_point.update(
                        {
                            "DistanceToUserLocation": distance_str,
                            "DurationToUserLocation": duration,
                            "StepsDirectionFromUserLocationToChargeStation": [
                                step["html_instructions"] for step in steps
                            ],
                        }
                    )
                    self.charge_points_locator._charge_stations.append(charge_point)

            # Update user's location to the last charging station found

            start_latitude = self.charge_points_locator._charge_stations[-1][
                "AddressInfo"
            ]["Latitude"]
            start_longitude = self.charge_points_locator._charge_stations[-1][
                "AddressInfo"
            ]["Longitude"]

        return self.charge_points_locator._charge_stations

    def ev_trip_planner(self, user_address, user_destination_address, socket_type=None):
        charge_stations = self.get_charge_point(user_address, user_destination_address)
        for charge_point in charge_stations:
            for connection in charge_point["Connections"]:
                connection_type = connection.get("ConnectionType", {}).get("Title", "")
                if socket_type is None or connection_type.startswith(socket_type):
                    if is_charge_station_available(charge_point):
                        self.charge_points_locator._filtered_charging_stations.append(
                            charge_point
                        )
                        break
        sorted_charging_points = sorted(
            self.charge_points_locator._filtered_charging_stations,
            key=lambda x: float(x["DistanceToUserLocation"].split()[0]),
        )
        return sorted_charging_points


# initialize the ev trip planner tool
ev_trip_planner_tool = EvTripPlannerTool()
logger.info("Ev trip planner tool initialized")
# import json
# # # # Example usage:
# user_address = "Brooklyn, NY 11206, United States"
# user_destination_address = "726 Washington Ave, Belleville, NJ 07109, United States"
# socket_type = "Type 1"
# ev_trip_planner = EvTripPlannerTool()
# sorted_charging_points = ev_trip_planner.ev_trip_planner(user_address, user_destination_address, socket_type)
# with open('sorted_charging_points.json', 'w') as f:
#     json.dump(sorted_charging_points, f, indent=4)
