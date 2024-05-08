import os

import requests
from dotenv import load_dotenv

from utils.charge_station_availability import is_charge_station_available
from utils.google_location import (get_distance_and_route_info,
                                   get_latitude_longitude)

# Load environment variables
load_dotenv()


opencharge_map_api_key = os.environ.get("OPENCHARGE_MAP_API_KEY")

if opencharge_map_api_key is None:
    raise ValueError("OpenChargeMap API Key is not provided")

# Dictionary to store the charging stations with distance to user's location as key
charge_stations = []


def get_closest_charge_stations(user_latitude, user_longitude, max_results=10, max_distance=None):
    """
    Get the closest charging stations to the user's location using OpenChargeMap API.
    """
    base_url = "https://api.openchargemap.io/v3/poi/"
    headers = {"X-API-Key": opencharge_map_api_key}
    params = {
        "output": "json",
        "maxresults": max_results,
        "latitude": user_latitude,
        "longitude": user_longitude,
        "distance": max_distance,
        "distanceunit": "KM",
    }
    response = requests.get(base_url, params=params, headers=headers)
    if response.status_code == 200:
        results = response.json()
        if results:
            return results
    print("Error: Unable to retrieve charging stations")
    return []


def charge_point_locator_unfiltered(address):
    user_latitude, user_longtitude = get_latitude_longitude(address=address)
    if user_latitude and user_longtitude is not None:
        results = get_closest_charge_stations(user_latitude, user_longtitude)
        for result in results:
            result_uuid = result.get("UUID")
            if result_uuid is not None:
                # check if the result uuid is not already in the list of charging points
                if not any(
                        result_uuid == charge_point["UUID"] for charge_point in charge_stations):

                    # get the distance of each charging station to the user's location
                    charge_station_address = result["AddressInfo"]["AddressLine1"] if result["AddressInfo"]["AddressLine1"] else ""
                    charge_station_city = result["AddressInfo"]["Town"] if result["AddressInfo"]["Town"] else ""
                    charge_station_state = result["AddressInfo"]["StateOrProvince"] if result["AddressInfo"]["StateOrProvince"] else ""
                    charge_station_country = result["AddressInfo"]["Country"]["Title"] if result["AddressInfo"]["Country"]["Title"] else ""
                    charge_station_full_address = f"{charge_station_address}, {charge_station_city}, {charge_station_state}, {charge_station_country}"
                    distance, duration, steps = get_distance_and_route_info(
                        address,
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
                    charge_stations.append(result_with_distance)
            else:
                print("No UUID found for the charging station")
    return charge_stations


# filter the charging points based on the socket type and availability
def charge_points_locator(address, socket_type=None):
    filtered_charging_points = []
    # get the charge station location
    charge_stations = charge_point_locator_unfiltered(address)
    for charge_point in charge_stations:
        for connection in charge_point["Connections"]:
            if socket_type is None or connection["ConnectionType"]["Title"].startswith(
                    socket_type) and is_charge_station_available(charge_point):
                filtered_charging_points.append(charge_point)
                break
    # Sort the filtered charging points by distance to the user's location
    sorted_charging_points = sorted(filtered_charging_points, key=lambda x: float(x["DistanceToUserLocation"].split()[0]))
    return sorted_charging_points
# import json
# user_address = "726 Washington Ave, Belleville, NJ 07109, United States"
# charge_stations = charge_points_locator(user_address, socket_type=None)
# with open("charging_points.json", "w") as file:
#     file.write(json.dumps(charge_stations, indent=4))
