import threading
import queue
import logging
import time
from enum import Enum
from typing import Dict, List, Callable, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EventType(Enum):
    """Enum for different types of events in the system"""
    STOCK_DATA_UPDATED = "stock_data_updated"
    NEWS_UPDATED = "news_updated"
    MARKET_ANALYSIS_COMPLETED = "market_analysis_completed"
    SENTIMENT_ANALYSIS_COMPLETED = "sentiment_analysis_completed"
    INSTITUTIONAL_ANALYSIS_COMPLETED = "institutional_analysis_completed"
    RISK_ANALYSIS_COMPLETED = "risk_analysis_completed"
    USER_REQUEST = "user_request"
    SYSTEM_ERROR = "system_error"
    AGENT_STATUS_UPDATE = "agent_status_update"

class Event:
    """Class representing an event in the system"""
    
    def __init__(self, event_type: EventType, data: Dict[str, Any] = None, source: str = None):
        """
        Initialize an event
        
        Args:
            event_type (EventType): Type of the event
            data (Dict[str, Any], optional): Event data
            source (str, optional): Source of the event
        """
        self.event_type = event_type
        self.data = data or {}
        self.source = source
        self.timestamp = time.time()
    
    def __str__(self):
        return f"Event(type={self.event_type.value}, source={self.source}, timestamp={self.timestamp})"

class EventBus:
    """
    Event bus for agent communication.
    Implements a publish-subscribe pattern for inter-agent communication.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern to ensure only one EventBus instance exists"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(EventBus, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the EventBus (only once due to singleton pattern)"""
        if self._initialized:
            return
            
        self._subscribers: Dict[EventType, List[Callable[[Event], None]]] = {event_type: [] for event_type in EventType}
        self._event_queue = queue.Queue()
        self._running = False
        self._thread = None
        self._initialized = True
        
        logger.info("EventBus initialized")
    
    def start(self):
        """Start the event processing thread"""
        if self._running:
            logger.warning("EventBus already running")
            return
            
        self._running = True
        self._thread = threading.Thread(target=self._process_events, daemon=True)
        self._thread.start()
        logger.info("EventBus started")
    
    def stop(self):
        """Stop the event processing thread"""
        if not self._running:
            logger.warning("EventBus not running")
            return
            
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("EventBus stopped")
    
    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]):
        """
        Subscribe to an event type
        
        Args:
            event_type (EventType): Type of event to subscribe to
            callback (Callable[[Event], None]): Callback function to be called when event occurs
        """
        if event_type not in self._subscribers:
            logger.error(f"Invalid event type: {event_type}")
            return
            
        if callback not in self._subscribers[event_type]:
            self._subscribers[event_type].append(callback)
            logger.info(f"Subscribed to {event_type.value}")
    
    def unsubscribe(self, event_type: EventType, callback: Callable[[Event], None]):
        """
        Unsubscribe from an event type
        
        Args:
            event_type (EventType): Type of event to unsubscribe from
            callback (Callable[[Event], None]): Callback function to remove
        """
        if event_type not in self._subscribers:
            logger.error(f"Invalid event type: {event_type}")
            return
            
        if callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)
            logger.info(f"Unsubscribed from {event_type.value}")
    
    def publish(self, event: Event):
        """
        Publish an event to the bus
        
        Args:
            event (Event): Event to publish
        """
        self._event_queue.put(event)
        logger.info(f"Published event: {event}")
    
    def _process_events(self):
        """Process events from the queue (runs in a separate thread)"""
        logger.info("Event processing thread started")
        
        while self._running:
            try:
                # Get event with timeout to allow for clean shutdown
                event = self._event_queue.get(timeout=1.0)
                
                # Notify subscribers
                for callback in self._subscribers[event.event_type]:
                    try:
                        callback(event)
                    except Exception as e:
                        logger.error(f"Error in subscriber callback: {e}")
                
                self._event_queue.task_done()
                
            except queue.Empty:
                # Timeout occurred, just continue the loop
                continue
            except Exception as e:
                logger.error(f"Error processing events: {e}")
        
        logger.info("Event processing thread stopped")
    
    def get_queue_size(self):
        """Get the current size of the event queue"""
        return self._event_queue.qsize()
    
    def wait_for_queue_empty(self, timeout=None):
        """
        Wait for the event queue to be empty
        
        Args:
            timeout (float, optional): Maximum time to wait in seconds
        
        Returns:
            bool: True if queue is empty, False if timeout occurred
        """
        try:
            self._event_queue.join(timeout=timeout)
            return True
        except queue.Empty:
            return False
