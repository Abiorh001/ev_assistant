from typing import Union

from pydantic import BaseModel


class ChargePointsLocatorSchema(BaseModel):
    address: str
    socket_type: Union[str, None] = None


class EvTripPlannerSchema(BaseModel):
    user_address: str
    user_destination_address: str
    socket_type: Union[str, None] = None
