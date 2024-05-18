import os
from typing import Union

import requests
from dotenv import load_dotenv

from agents.agent_tools.constant import DISTANCE_IN_KM
from agents.agent_tools.utils.charge_station_availability import \
    is_charge_station_available
from agents.agent_tools.utils.google_location import (
    get_distance_and_route_info, get_latitude_longitude)
from core.settings import logger

# Load environment variables
load_dotenv()

opencharge_map_api_key = os.getenv("OPENCHARGE_MAP_API_KEY")

if not opencharge_map_api_key:
    logger.error("OpenChargeMap API Key is not provided")
    raise ValueError("OpenChargeMap API Key is not provided")


class ChargePointsLocatorTool:
    def __init__(self):
        self._charge_stations = []
        self._filtered_charging_stations = []

    def get_closest_charge_stations(
        self,
        user_latitude: float,
        user_longitude: float,
        max_results: int,
        max_distance: Union[int, float] = None,
    ) -> list:
        """Get the closest charging stations to the user's location using OpenChargeMap API."""
        base_url = "https://api.openchargemap.io/v3/poi/"
        headers = {"X-API-Key": opencharge_map_api_key}
        params = {
            "output": "json",
            "maxresults": max_results,
            "latitude": user_latitude,
            "longitude": user_longitude,
            "distance": max_distance,
            "distanceunit": DISTANCE_IN_KM,
        }
        try:
            response = requests.get(base_url, params=params, headers=headers)
            response.raise_for_status()
            return response.json() if response.status_code == 200 else []
        except requests.RequestException as e:
            logger.error(
                "Error getting charging stations from OpenChargeMap API: %s", e
            )
            return []

    def charge_point_locator_unfiltered(
        self, address: str, max_distance: int = 5, max_results: int = 10
    ):
        user_latitude, user_longitude = get_latitude_longitude(address=address)
        if user_latitude is None or user_longitude is None:
            logger.error("Error getting user location")
            return []

        charge_points = self.get_closest_charge_stations(
            user_latitude,
            user_longitude,
            max_results,
            max_distance,
        )
        for charge_point in charge_points:
            charge_point_uuid = charge_point.get("UUID")
            if charge_point_uuid and not any(
                charge_point_uuid == cp["UUID"] for cp in self._charge_stations
            ):
                address_info = charge_point.get("AddressInfo", {})
                charge_point_address = address_info.get("AddressLine1", "")
                charge_point_city = address_info.get("Town", "")
                charge_point_state = address_info.get("StateOrProvince", "")
                charge_point_country = address_info.get("Country", {}).get("Title", "")
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
                    address, charge_point_full_address
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
                logger.info("Charge point added to the the charge stations list.")
                self._charge_stations.append(charge_point)

        return self._charge_stations

    def charge_points_locator(self, address: str, socket_type: str = None):
        charge_stations = self.charge_point_locator_unfiltered(address)
        for charge_point in charge_stations:
            for connection in charge_point["Connections"]:
                connection_type = connection.get("ConnectionType", {}).get("Title", "")
                if socket_type is None or connection_type.startswith(socket_type):
                    if is_charge_station_available(charge_point):
                        self._filtered_charging_stations.append(charge_point)
                    break

        sorted_charging_points = sorted(
            self._filtered_charging_stations,
            key=lambda x: float(x["DistanceToUserLocation"].split()[0]),
        )
        logger.info("Sorted charging points by distance to user location.")
        return sorted_charging_points


# initialize the charge station locator tool
charge_station_locator_tool = ChargePointsLocatorTool()
logger.info("Charge station locator tool initialized.")
