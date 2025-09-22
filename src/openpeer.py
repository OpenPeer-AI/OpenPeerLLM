from typing import List, Dict, Any, Optional
import asyncio
import json
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
import torch

class PeerNetwork:
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.app = FastAPI()
        self.active_peers: Dict[str, WebSocket] = {}
        self.host = host
        self.port = port
        
        # Register WebSocket endpoint
        @self.app.websocket("/ws/{peer_id}")
        async def websocket_endpoint(websocket: WebSocket, peer_id: str):
            await self.connect_peer(websocket, peer_id)
            try:
                while True:
                    data = await websocket.receive_text()
                    await self.broadcast(data, peer_id)
            except Exception:
                await self.disconnect_peer(peer_id)
                
    async def connect_peer(self, websocket: WebSocket, peer_id: str):
        """Connect a new peer to the network"""
        await websocket.accept()
        self.active_peers[peer_id] = websocket
        
    async def disconnect_peer(self, peer_id: str):
        """Remove a peer from the network"""
        if peer_id in self.active_peers:
            await self.active_peers[peer_id].close()
            del self.active_peers[peer_id]
            
    async def broadcast(self, message: str, sender_id: str):
        """Broadcast a message to all peers except the sender"""
        for peer_id, websocket in self.active_peers.items():
            if peer_id != sender_id:
                await websocket.send_text(message)
                
class OpenPeerClient:
    def __init__(self, network_url: str):
        self.network_url = network_url
        self.websocket: Optional[WebSocket] = None
        self.peer_id: Optional[str] = None
        
    async def connect(self, peer_id: str):
        """Connect to the peer network"""
        self.peer_id = peer_id
        self.websocket = await WebSocket.connect(f"{self.network_url}/ws/{peer_id}")
        
    async def send_model_update(self, model_state: Dict[str, torch.Tensor]):
        """Send model state updates to the network"""
        if not self.websocket:
            raise RuntimeError("Not connected to network")
            
        serialized_state = {
            "type": "model_update",
            "peer_id": self.peer_id,
            "state": {k: v.cpu().numpy().tolist() for k, v in model_state.items()}
        }
        await self.websocket.send_text(json.dumps(serialized_state))
        
    async def receive_updates(self):
        """Receive updates from the network"""
        if not self.websocket:
            raise RuntimeError("Not connected to network")
            
        while True:
            data = await self.websocket.receive_text()
            yield json.loads(data)
            
def create_peer_network(host: str = "localhost", port: int = 8000) -> PeerNetwork:
    """Create and start a peer network server"""
    network = PeerNetwork(host, port)
    import uvicorn
    uvicorn.run(network.app, host=host, port=port)
    return network