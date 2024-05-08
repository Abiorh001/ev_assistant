import uuid
from core.settings import logger
import asyncio

import uvicorn
from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect

from agents.openai_agent import agent


app = FastAPI()


@app.get("/status")
async def status():
    return {"status": "OK"}


# Creating connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections = set()
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"New connection established with: {websocket.client}")
        logger.info(f"Total active connections: {len(self.active_connections)}")
        
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Connection closed with: {websocket.client}")
        logger.info(f"Total active connections: {len(self.active_connections)}")


# initialize the connection manager
manager = ConnectionManager()


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    try:
        # Accept the WebSocket connection
        await manager.connect(websocket)

        # Generate a unique session ID for this connection
        session_id = str(uuid.uuid4())

        intro_message = f"USER ID: {session_id}: what are you and what can you do?"
        response = await agent.achat(intro_message)
        # stream_response = await agent.astream_chat(intro_message)
        
        # async for response in stream_response.achat_stream:
        #     # Ensure the response is a string
        #     response_str = str(response.message.content)
        #     # Send the response to the client
        #     await websocket.send_text(response_str)
        response_str = str(response)
        await websocket.send_text(response_str)

        # Handle messages until the connection is closed
        while True:
            try:
                # Receive the message from the client
                message = await websocket.receive_text()
                # Process the received message
                updated_message = f"USER ID: {session_id}: {message}"
                logger.debug(f"Received message: {updated_message}")
                # call the agent in async mode to pass the message and get the response
                response = await agent.achat(updated_message)
                logger.debug(f"Generated response: {response}")
                # Ensure the response is a string
                response_str = str(response)
                # Send the response to the client
                await websocket.send_text(response_str)
            except WebSocketDisconnect:
                manager.disconnect(websocket)
                logger.error("WebSocket connection closed. Stopping message sending.")
                break
            except Exception as e:
                logger.error(f"An error occurred {e}")
                manager.disconnect(websocket)
                logger.error("An error occurred. WebSocket connection closed. abnormally.")
                break
    except Exception as e:
        manager.disconnect(websocket)
        logger.error(f"An error occurred {e}")

if __name__ == "__main__":
    logger.info("Starting the server...")
    uvicorn.run(app, host="localhost", port=8765)
