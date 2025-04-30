import threading
import time
import logging
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from backend.agent_system.event_bus import EventBus, Event, EventType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class BaseAgent(ABC):
    """
    Base class for all agents in the system.
    Provides common functionality for agent management and communication.
    """
    
    def __init__(self, name: str, polling_interval: int = 3600):
        """
        Initialize a base agent
        
        Args:
            name (str): Name of the agent
            polling_interval (int): Interval between agent runs in seconds
        """
        self.name = name
        self.id = str(uuid.uuid4())
        self.polling_interval = polling_interval
        self.event_bus = EventBus()
        self.logger = logging.getLogger(f"Agent.{name}")
        self.running = False
        self.thread = None
        self.last_run_time = None
        self.status = "initialized"
        
        # Subscribe to relevant events
        self._subscribe_to_events()
        
        self.logger.info(f"Agent {name} initialized with ID {self.id}")
    
    def _subscribe_to_events(self):
        """Subscribe to relevant events on the event bus"""
        # Override in subclasses to subscribe to specific events
        pass
    
    def start(self):
        """Start the agent's background thread"""
        if self.running:
            self.logger.warning(f"Agent {self.name} already running")
            return
        
        self.running = True
        self.status = "starting"
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        
        # Publish agent status event
        self._publish_status_event("started")
        
        self.logger.info(f"Agent {self.name} started")
    
    def stop(self):
        """Stop the agent's background thread"""
        if not self.running:
            self.logger.warning(f"Agent {self.name} not running")
            return
        
        self.running = False
        self.status = "stopping"
        if self.thread:
            self.thread.join(timeout=5.0)
        
        # Publish agent status event
        self._publish_status_event("stopped")
        
        self.logger.info(f"Agent {self.name} stopped")
    
    def _run_loop(self):
        """Main agent loop that runs in background thread"""
        self.logger.info(f"Agent {self.name} run loop started")
        
        while self.running:
            try:
                # Run the agent's main processing logic
                self.status = "running"
                self._publish_status_event("running")
                
                self.run()
                
                self.last_run_time = time.time()
                self.status = "idle"
                self._publish_status_event("idle")
                
                # Sleep until next polling interval
                for _ in range(int(self.polling_interval)):
                    if not self.running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                self.logger.error(f"Error in agent {self.name} run loop: {e}")
                self.status = "error"
                self._publish_status_event("error", {"error": str(e)})
                
                # Sleep for a bit before retrying
                time.sleep(60)
        
        self.logger.info(f"Agent {self.name} run loop stopped")
    
    def _publish_status_event(self, status: str, additional_data: Dict[str, Any] = None):
        """
        Publish agent status event to the event bus
        
        Args:
            status (str): Agent status
            additional_data (Dict[str, Any], optional): Additional data to include in the event
        """
        data = {
            "agent_id": self.id,
            "agent_name": self.name,
            "status": status,
            "last_run_time": self.last_run_time
        }
        
        if additional_data:
            data.update(additional_data)
        
        event = Event(
            event_type=EventType.AGENT_STATUS_UPDATE,
            data=data,
            source=self.name
        )
        
        self.event_bus.publish(event)
    
    def _publish_event(self, event_type: EventType, data: Dict[str, Any] = None):
        """
        Publish an event to the event bus
        
        Args:
            event_type (EventType): Type of event to publish
            data (Dict[str, Any], optional): Event data
        """
        event = Event(
            event_type=event_type,
            data=data or {},
            source=self.name
        )
        
        self.event_bus.publish(event)
    
    @abstractmethod
    def run(self):
        """
        Main method that contains the agent's processing logic.
        Must be implemented by all agent subclasses.
        """
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the agent
        
        Returns:
            Dict[str, Any]: Agent status information
        """
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "running": self.running,
            "last_run_time": self.last_run_time,
            "polling_interval": self.polling_interval
        }
