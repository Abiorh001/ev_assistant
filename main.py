import uuid

import uvicorn
from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect

from agents.openai_agent import agent

app = FastAPI()


@app.get("/status")
async def status():
    return {"status": "OK"}


async def process_message(message):
    # Use your AI agent to generate a response
    response = agent.chat(message)
    return response


# Creating connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections = set()
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        print(f"Connection established with {websocket}")
        
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print(f"Connection closed with {websocket}")


# initialize the connection manager
manager = ConnectionManager()


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    try:
        # Accept the WebSocket connection
        await manager.connect(websocket)

        # Generate a unique session ID for this connection
        session_id = str(uuid.uuid4())

        into_message = f"USER ID: {session_id}: what are you and what can you do?"
        response = agent.chat(into_message)
        response_str = str(response)
        await websocket.send_text(response_str)

        # Handle messages until the connection is closed
        while True:
            try:
                message = await websocket.receive_text()
                # Process the received message
                updated_message = f"USER ID: {session_id}: {message}"
                print(f"Received message: {updated_message}")
                response = await process_message(updated_message)
                print(f"Generated response: {response}")
                # Ensure the response is a string
                response_str = str(response)
                await websocket.send_text(response_str)
            except WebSocketDisconnect:
                manager.disconnect(websocket)
                print("WebSocket connection closed. Stopping message sending.")
                break
            except Exception as e:
                print(f"Error: {e}")
                manager.disconnect(websocket)
                print("WebSocket connection closed due to error. Stopping message sending.")
                break
    except Exception as e:
        manager.disconnect(websocket)
        print(f"Error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8765)
