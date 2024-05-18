import logging
import os
import sys

from dotenv import load_dotenv
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.agent import ReActAgent
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI

from agents.agent_tools.charge_station_locator import \
    charge_station_locator_tool
from agents.agent_tools.ev_trip_planner import ev_trip_planner_tool
from agents.agent_tools.schema import (ChargePointsLocatorSchema,
                                       EvTripPlannerSchema)
from prompt import context
from query_engines.ocpp_query_engine import ocpp_query_engine

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI(model="gpt-4-1106-preview", max_tokens=1000, api_key=openai_api_key)


# initialize the charge station locator tool
charge_points_locator = charge_station_locator_tool.charge_points_locator

# initialize the ev trip planner tool
ev_trip_planner = ev_trip_planner_tool.ev_trip_planner
# create charge point locator tool
charge_point_locator_tool = FunctionTool.from_defaults(
    fn=charge_points_locator,
    name="charge_points_locator",
    description="This tool is used to locate the nearest electric vehicle charging stations to the user's location. the function will accept the user's address as input, it optionally accept socket type if addess or None and return the nearest charging stations and it contain distnace from user address, duration in minutes, steps to charge station ensuring users can easily identify the most convenient charging stations.",
    tool_metadata=ToolMetadata(
        fn_schema=ChargePointsLocatorSchema,
        name="charge_points_locator",
        description="This tool is used to locate the nearest electric vehicle charging stations to the user's location. the function will accept the user's address as input, it optionally accept socket type if addess or None and return the nearest charging stations and it contain distnace from user address, duration in minutes, steps to charge station ensuring users can easily identify the most convenient charging stations.",
    ),
)

# create ev trip planner tool
ev_trip_planner_tool = FunctionTool.from_defaults(
    fn=ev_trip_planner,
    name="ev_trip_planner",
    description="This tool is used to plan a trip for an electric vehicle user. It calculates the distance between the user's location and destination, segments the trip based on the distance, and finds the closest charging stations for each segment. The function accepts the user's address, destination address, and sockey_type as input and returns the charging stations for each segment of the trip.",
    tool_metadata=ToolMetadata(
        fn_schema=EvTripPlannerSchema,
        name="ev_trip_planner",
        description="This tool is used to plan a trip for an electric vehicle user. It calculates the distance between the user's location and destination, segments the trip based on the distance, and finds the closest charging stations for each segment. The function accepts the user's address and destination address as input and returns the charging stations for each segment of the trip.",
    ),
)
# create query engine tool for the ocpp query engine
ocpp_query_engine_tool = QueryEngineTool(
    query_engine=ocpp_query_engine,
    metadata=ToolMetadata(
        name="ocpp_query_engine",
        description="This tool is used to answer questions related to Open Charge Point Protocol (OCPP) and electric vehicle charging stations., how to connect to the charging stations and the electric vechicle supply equipments. through the OCPP protocol.",
    ),
)
# create query engine tool to be use in the sub question ocpp query engine
ocpp_query_engine_tools = [
    QueryEngineTool(
        query_engine=ocpp_query_engine,
        metadata=ToolMetadata(
            name="ocpp_query_engine",
            description="This tool is used to answer questions related to Open Charge Point Protocol (OCPP) and electric vehicle charging stations., how to connect to the charging stations and the electric vechicle supply equipments. through the OCPP protocol.",
        ),
    )
]
# create a sub question query engine
ocpp_sub_question_query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=ocpp_query_engine_tools,
    llm=llm,
)

# create query engine tool for sub question ocpp query engine
ocpp_sub_question_query_engine_tool = QueryEngineTool(
    query_engine=ocpp_sub_question_query_engine,
    metadata=ToolMetadata(
        name="ocpp_sub_question_query_engine",
        description="This tool is used to answer sub-questions related to Open Charge Point Protocol (OCPP) and electric vehicle charging stations., how to connect to the charging stations and the electric vechicle supply equipments. through the OCPP protocol. It is used to provide detailed information for specific questions. while breaking the complex questions into sub-questions.",
    ),
)

# tools
tools = [
    ocpp_query_engine_tool,
    ocpp_sub_question_query_engine_tool,
    charge_point_locator_tool,
    ev_trip_planner_tool,
]


# create agent
agent = ReActAgent.from_tools(
    tools=tools,
    verbose=True,
    context=context,
    llm=llm,
)
