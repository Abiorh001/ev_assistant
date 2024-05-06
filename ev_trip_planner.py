from utils.google_location import get_latitude_longitude, get_distance_and_route_info
from charge_station_locator import get_closest_charge_stations
from utils.charge_station_availability import is_charge_station_available


def segment_trip(user_address, user_destination_address):
    # Get the latitude and longitude of the user's address
    user_latitude, user_longitude = get_latitude_longitude(user_address)
    # Calculate the total distance between the user's location and the
    # destination
    total_distance = get_distance_and_route_info(
        user_address,
        user_destination_address
    )[0]
   
    # convert to km
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

    # Initialize lists to store charging points along the route
    charging_points = []

    # Iterate over each segment to find charging points
    for segment in range(1, num_segments + 1):
        # Calculate the distance for the current segment
        segment_distance = min(
            segment_length,
            total_distance - (segment - 1) * segment_length
        )
        segment_distance = round(segment_distance)
        # Get the closest EV charging stations to the user's location for 
        # he current segment
        results = get_closest_charge_stations(
            user_latitude,
            user_longitude,
            max_distance=segment_distance
        )
        # Append the results to the list of charging points if the UUID is not already present
        # check if each result uuid is unique in the list
        for result in results:
            result_uuid = result["UUID"]
            # check if the result uuid is not already in the list of charging points
            if not any(
                    result_uuid == point["UUID"] for point in charging_points):

                # get the distance of each charging station to the user's location
                charge_station_address = result["AddressInfo"]["AddressLine1"]
                charge_staion_city = result["AddressInfo"]["Town"]
                charge_station_state = result["AddressInfo"]["StateOrProvince"]
                charge_station_country = result["AddressInfo"]["Country"]["Title"]
                charge_station_full_address = f"{charge_station_address}, {charge_staion_city}, {charge_station_state}, {charge_station_country}"
                distance, duration, steps = get_distance_and_route_info(
                    user_address,
                    charge_station_full_address,
                )
                # convert from mile to km
                distance = float(distance.split()[0]) * 1.60934
                distance = round(distance, 2)
                distance_str = str(distance) + " km"
                result_with_distance = result.copy()
                result_with_distance["DistanceToUserLocation"] = distance_str
                result_with_distance["DurationToUserLocation"] = duration
                # get the steps to each charging station
                steps_html_list = []
                for step in steps:
                    steps_html_list.append(step["html_instructions"])
                result_with_distance["StepsDirectionFromUserLocationToChargeStation"] = steps_html_list
                charging_points.append(result_with_distance)

        # Update the user's location to the last charging station found
        user_latitude = charging_points[-1]["AddressInfo"]["Latitude"]
        user_longitude = charging_points[-1]["AddressInfo"]["Longitude"]

    return charging_points


def ev_trip_planner(user_address, user_destination_address, socket_type=None):
    charging_points = segment_trip(user_address, user_destination_address)
    # Filter the charging points based on the socket type and availability
    filtered_charging_points = []
    for charge_point in charging_points:
        for connection in charge_point["Connections"]:
            if socket_type is None or connection["ConnectionType"]["Title"].startswith(
               socket_type) and is_charge_station_available(charge_point):
                filtered_charging_points.append(charge_point)
                break
    # Sort the filtered charging points by distance to the user's location
    sorted_charging_points = sorted(charging_points, key=lambda x: float(x["DistanceToUserLocation"].split()[0]))

    return sorted_charging_points


# import json
# # # # Example usage:
# user_address = "Brooklyn, NY 11206, United States"
# user_destination_address = "726 Washington Ave, Belleville, NJ 07109, United States"
# socket_type = "Type 1"
# ev_trip_planner = ev_trip_planner(user_address, user_destination_address, socket_type)
# with open("Ev_trip_planner.txt", "w") as f:
#     json.dump(ev_trip_planner, f, indent=4)
#     f.write("\n")