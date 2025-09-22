import torch
import torch.nn as nn
from typing import Dict, Any, List
import asyncio
import websockets
import json
from pydantic import BaseModel

class PeerMessage(BaseModel):
    message_type: str
    payload: Dict[str, Any]
    peer_id: str

class DecentModel(nn.Module):
    """Base class for decentralized deep learning models"""
    
    def __init__(self):
        super().__init__()
        self.peer_id = self._generate_peer_id()
        self.peers: List[str] = []
        self.websocket = None
        self.state_updates = {}
        
    def _generate_peer_id(self) -> str:
        """Generate a unique peer ID"""
        import uuid
        return str(uuid.uuid4())
        
    async def connect_to_network(self, network_url: str):
        """Connect to the decentralized network"""
        self.websocket = await websockets.connect(network_url)
        await self._register_peer()
        
    async def _register_peer(self):
        """Register this peer with the network"""
        message = PeerMessage(
            message_type="register",
            payload={"model_type": self.__class__.__name__},
            peer_id=self.peer_id
        )
        await self.websocket.send(message.json())
        
    async def broadcast_state_update(self, state_dict: Dict[str, torch.Tensor]):
        """Broadcast model state updates to other peers"""
        message = PeerMessage(
            message_type="state_update",
            payload={"state": self._serialize_state_dict(state_dict)},
            peer_id=self.peer_id
        )
        await self.websocket.send(message.json())
        
    def _serialize_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, List[float]]:
        """Serialize model state for transmission"""
        return {k: v.cpu().numpy().tolist() for k, v in state_dict.items()}
        
    async def receive_state_updates(self):
        """Receive and process state updates from other peers"""
        while True:
            message = await self.websocket.recv()
            data = PeerMessage.parse_raw(message)
            if data.message_type == "state_update":
                self.state_updates[data.peer_id] = self._deserialize_state_dict(
                    data.payload["state"]
                )
                
    def _deserialize_state_dict(self, state_dict: Dict[str, List[float]]) -> Dict[str, torch.Tensor]:
        """Deserialize received model state"""
        return {k: torch.tensor(v) for k, v in state_dict.items()}
        
    def aggregate_states(self):
        """Aggregate state updates from all peers"""
        if not self.state_updates:
            return
            
        # Average all state updates
        aggregated_state = {}
        for key in self.state_updates[list(self.state_updates.keys())[0]].keys():
            tensors = [states[key] for states in self.state_updates.values()]
            aggregated_state[key] = torch.mean(torch.stack(tensors), dim=0)
            
        # Update model with aggregated state
        self.load_state_dict(aggregated_state)
        self.state_updates.clear()
        
    def forward(self, *args, **kwargs):
        """Forward pass - to be implemented by child classes"""
        raise NotImplementedError