from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.agent import ReActAgent
from llama_index.core.query_engine import SubQuestionQueryEngine
from advance_rag import query_engine
from charge_station_locator import charge_points_locator
from prompt import context
from pydantic import BaseModel
from ev_trip_planner import ev_trip_planner


class ChargePointsLocator(BaseModel):
    address: str
    socket_type: str


class EvTripPlanner(BaseModel):
    user_address: str
    user_destination_address: str
    socket_type: str


# create charge point locator tool
charge_point_locator_tool = FunctionTool.from_defaults(
    fn=charge_points_locator,
    name="charge_points_locator",
    description="This tool is used to locate the nearest electric vehicle charging stations to the user's location. the function will accept the user's address as input, it optionally accept socket type if addess or None and return the nearest charging stations and it contain distnace from user address, duration in minutes, steps to charge station ensuring users can easily identify the most convenient charging stations.",
    tool_metadata=ToolMetadata(
        fn_schema=ChargePointsLocator,
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
        fn_schema=EvTripPlanner,
        name="ev_trip_planner",
        description="This tool is used to plan a trip for an electric vehicle user. It calculates the distance between the user's location and destination, segments the trip based on the distance, and finds the closest charging stations for each segment. The function accepts the user's address and destination address as input and returns the charging stations for each segment of the trip.",
    ),
)
# create query engine tool for the query engine
query_engine_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="query_engine",
        description="This tool is used to answer questions related to electric vehicles, electric vechicle supply equipments, and electric vehicle charging stations.",
    ),
)
# create query engine tool to be use in the sub question query engine
query_engine_tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="query_engine",
            description="This tool is used to answer questions related to electric vehicles, electric vechicle supply equipments, and electric vehicle charging stations.",
        ),
    )
]
# create a sub question query engine
sub_question_query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
)

# create query engine tool for sub question query engine
sub_question_query_engine_tool = QueryEngineTool(
    query_engine=sub_question_query_engine,
    metadata=ToolMetadata(
        name="sub_question_query_engine",
        description="This tool is used to answer sub-questions related to the main question. It is used to answer questions related to electric vehicles, electric vechicle supply equipments, and electric vehicle charging stations. and also if the question is very broad, it will break it down into sub-questions and answer them.",
    ),
)

# tools
tools = [
    query_engine_tool,
    sub_question_query_engine_tool,
    charge_point_locator_tool,
    ev_trip_planner_tool,
]

# create agent
agent = ReActAgent.from_tools(
    tools=tools,
    verbose=True,
    context=context,
)
    
