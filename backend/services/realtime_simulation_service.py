#!/usr/bin/env python3
"""
Real-time ICS Security Simulation Service
-----------------------------------------
This service simulates real-time network traffic by progressively reading from the
cleaned dataset and performing ML inference using the trained ensemble model.
"""

import asyncio
import logging
import pandas as pd
import numpy as np
import json
import joblib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, AsyncGenerator
from dataclasses import dataclass, asdict
import socket
import struct

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ClassificationResult:
    """Data class for classification results"""
    timestamp: str
    packet_id: int
    source_ip: str
    destination_ip: str
    protocol: str
    packet_size: int
    predicted_class: str
    confidence: float
    anomaly_score: float
    features: Dict[str, float]
    attack_type: Optional[str] = None
    severity: str = "normal"

class RealTimeSimulationService:
    """Service for real-time ICS security simulation and ML inference"""
    
    def __init__(self, dataset_path: str = "/app/trained_models/processed_ics_dataset_cleaned.csv"):
        self.dataset_path = dataset_path
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_selectors = None
        self.feature_columns = None
        
        # Simulation state
        self.is_running = False
        self.is_paused = False
        self.current_row_index = 0
        self.playback_speed = 1.0  # Rows per second
        self.dataset = None
        
        # Statistics
        self.total_packets = 0
        self.attack_counts = {}
        self.recent_classifications = []
        self.max_recent_classifications = 1000
        
        # WebSocket connections
        self.active_connections = set()
        
        self._load_models()
        self._load_dataset()
    
    def _load_models(self):
        """Load the trained ML models and preprocessors"""
        try:
            models_dir = Path("/app/trained_models")
            
            # Load ensemble model
            model_path = models_dir / "ensemble_model.pkl"
            if model_path.exists():
                self.model = joblib.load(model_path)
                logger.info("Loaded ensemble model successfully")
            else:
                logger.error(f"Model file not found: {model_path}")
                return
            
            # Load preprocessors
            scaler_path = models_dir / "scalers.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("Loaded scaler successfully")
            
            label_encoder_path = models_dir / "label_encoder.pkl"
            if label_encoder_path.exists():
                self.label_encoder = joblib.load(label_encoder_path)
                logger.info("Loaded label encoder successfully")
            
            feature_selector_path = models_dir / "feature_selectors.pkl"
            if feature_selector_path.exists():
                self.feature_selectors = joblib.load(feature_selector_path)
                logger.info("Loaded feature selectors successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def _load_dataset(self):
        """Load and prepare the dataset for simulation"""
        try:
            # Read only the first few rows to get column info
            sample_df = pd.read_csv(self.dataset_path, nrows=10)
            
            # Identify feature columns (exclude metadata columns)
            exclude_cols = ['timestamp', 'label', 'category', 'file_name', 'src_ip', 'dst_ip']
            self.feature_columns = [col for col in sample_df.columns if col not in exclude_cols]
            
            logger.info(f"Identified {len(self.feature_columns)} feature columns")
            logger.info(f"Dataset path: {self.dataset_path}")
            
            # Get total row count for progress tracking
            with open(self.dataset_path, 'r') as f:
                self.total_packets = sum(1 for line in f) - 1  # Subtract header
            
            logger.info(f"Total packets in dataset: {self.total_packets}")
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
    
    def _int_to_ip(self, ip_int: int) -> str:
        """Convert integer IP to string format"""
        try:
            return socket.inet_ntoa(struct.pack('!I', ip_int))
        except:
            return f"Unknown-{ip_int}"
    
    def _preprocess_features(self, row: pd.Series) -> np.ndarray:
        """Preprocess features for ML inference"""
        try:
            # Extract features
            features = row[self.feature_columns].values.reshape(1, -1)
            
            # Handle missing values
            features = np.nan_to_num(features, nan=0.0)
            
            # Scale features if scaler is available
            if self.scaler is not None:
                features = self.scaler.transform(features)
            
            # Apply feature selection if available
            if self.feature_selectors is not None:
                # Assuming feature_selectors contains the selected feature indices
                if hasattr(self.feature_selectors, 'transform'):
                    features = self.feature_selectors.transform(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error preprocessing features: {e}")
            return np.zeros((1, len(self.feature_columns)))
    
    def _classify_packet(self, row: pd.Series) -> ClassificationResult:
        """Classify a single packet using the ML model"""
        try:
            # Preprocess features
            features = self._preprocess_features(row)
            
            # Get prediction
            if self.model is not None:
                prediction = self.model.predict(features)[0]
                probabilities = self.model.predict_proba(features)[0]
                confidence = np.max(probabilities)
                
                # Get class name if label encoder is available
                if self.label_encoder is not None:
                    try:
                        predicted_class = self.label_encoder.inverse_transform([prediction])[0]
                    except:
                        predicted_class = str(prediction)
                else:
                    predicted_class = str(prediction)
            else:
                # Fallback if model not loaded
                predicted_class = "normal"
                confidence = 0.5
            
            # Determine attack type and severity
            attack_type = None
            severity = "normal"
            
            if predicted_class != "normal" and predicted_class != 0:
                attack_type = predicted_class
                if confidence > 0.9:
                    severity = "critical"
                elif confidence > 0.7:
                    severity = "high"
                elif confidence > 0.5:
                    severity = "medium"
                else:
                    severity = "low"
            
            # Extract IP addresses
            src_ip = self._int_to_ip(int(row.get('src_ip_int', 0)))
            dst_ip = self._int_to_ip(int(row.get('dst_ip_int', 0)))
            
            # Create classification result
            result = ClassificationResult(
                timestamp=row.get('timestamp', datetime.now().isoformat()),
                packet_id=self.current_row_index,
                source_ip=src_ip,
                destination_ip=dst_ip,
                protocol=self._get_protocol(row),
                packet_size=int(row.get('packet_length', 0)),
                predicted_class=predicted_class,
                confidence=confidence,
                anomaly_score=1.0 - confidence if predicted_class != "normal" else confidence,
                features={col: float(row.get(col, 0)) for col in self.feature_columns[:10]},  # First 10 features
                attack_type=attack_type,
                severity=severity
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error classifying packet: {e}")
            # Return a default result
            return ClassificationResult(
                timestamp=datetime.now().isoformat(),
                packet_id=self.current_row_index,
                source_ip="Unknown",
                destination_ip="Unknown",
                protocol="Unknown",
                packet_size=0,
                predicted_class="error",
                confidence=0.0,
                anomaly_score=0.0,
                features={},
                severity="normal"
            )
    
    def _get_protocol(self, row: pd.Series) -> str:
        """Determine protocol from packet features"""
        if row.get('has_modbus', 0) == 1:
            return "Modbus"
        elif row.get('has_tcp', 0) == 1:
            return "TCP"
        elif row.get('has_udp', 0) == 1:
            return "UDP"
        elif row.get('has_icmp', 0) == 1:
            return "ICMP"
        else:
            return "Other"
    
    async def _read_packet_chunk(self, chunk_size: int = 100) -> List[pd.Series]:
        """Read a chunk of packets from the dataset"""
        try:
            # Read chunk starting from current position
            df_chunk = pd.read_csv(
                self.dataset_path,
                skiprows=range(1, self.current_row_index + 1),
                nrows=chunk_size
            )
            
            if df_chunk.empty:
                return []
            
            return [row for _, row in df_chunk.iterrows()]
            
        except Exception as e:
            logger.error(f"Error reading packet chunk: {e}")
            return []
    
    async def start_simulation(self):
        """Start the real-time simulation"""
        if self.is_running:
            logger.warning("Simulation is already running")
            return
        
        self.is_running = True
        self.is_paused = False
        logger.info("Starting real-time simulation")
        
        try:
            while self.is_running and self.current_row_index < self.total_packets:
                if self.is_paused:
                    await asyncio.sleep(0.1)
                    continue
                
                # Read packet chunk
                packets = await self._read_packet_chunk(10)
                
                if not packets:
                    break
                
                # Process each packet
                for packet in packets:
                    if not self.is_running or self.is_paused:
                        break
                    
                    # Classify packet
                    result = self._classify_packet(packet)
                    
                    # Update statistics
                    self.total_packets += 1
                    if result.attack_type:
                        self.attack_counts[result.attack_type] = self.attack_counts.get(result.attack_type, 0) + 1
                    
                    # Add to recent classifications
                    self.recent_classifications.append(result)
                    if len(self.recent_classifications) > self.max_recent_classifications:
                        self.recent_classifications.pop(0)
                    
                    # Broadcast to connected clients
                    await self._broadcast_classification(result)
                    
                    self.current_row_index += 1
                    
                    # Control playback speed
                    await asyncio.sleep(1.0 / self.playback_speed)
        
        except Exception as e:
            logger.error(f"Error in simulation: {e}")
        finally:
            self.is_running = False
            logger.info("Simulation stopped")
    
    async def stop_simulation(self):
        """Stop the real-time simulation"""
        self.is_running = False
        logger.info("Stopping real-time simulation")
    
    async def pause_simulation(self):
        """Pause the real-time simulation"""
        self.is_paused = True
        logger.info("Pausing real-time simulation")
    
    async def resume_simulation(self):
        """Resume the real-time simulation"""
        self.is_paused = False
        logger.info("Resuming real-time simulation")
    
    def set_playback_speed(self, speed: float):
        """Set the playback speed (packets per second)"""
        self.playback_speed = max(0.1, min(10.0, speed))
        logger.info(f"Set playback speed to {self.playback_speed} packets/second")
    
    def reset_simulation(self):
        """Reset simulation to beginning"""
        self.current_row_index = 0
        self.recent_classifications = []
        self.attack_counts = {}
        logger.info("Reset simulation to beginning")
    
    async def _broadcast_classification(self, result: ClassificationResult):
        """Broadcast classification result to all connected WebSocket clients"""
        if not self.active_connections:
            return
        
        message = {
            "type": "classification",
            "data": asdict(result)
        }
        
        # Remove disconnected clients
        disconnected = set()
        for websocket in self.active_connections.copy():
            try:
                await websocket.send_text(json.dumps(message))
            except:
                disconnected.add(websocket)
        
        self.active_connections -= disconnected
    
    def add_websocket_connection(self, websocket):
        """Add a WebSocket connection for real-time updates"""
        self.active_connections.add(websocket)
        logger.info(f"Added WebSocket connection. Total: {len(self.active_connections)}")
    
    def remove_websocket_connection(self, websocket):
        """Remove a WebSocket connection"""
        self.active_connections.discard(websocket)
        logger.info(f"Removed WebSocket connection. Total: {len(self.active_connections)}")
    
    def get_simulation_status(self) -> Dict[str, Any]:
        """Get current simulation status"""
        return {
            "is_running": self.is_running,
            "is_paused": self.is_paused,
            "current_row": self.current_row_index,
            "total_rows": self.total_packets,
            "progress_percent": (self.current_row_index / self.total_packets * 100) if self.total_packets > 0 else 0,
            "playback_speed": self.playback_speed,
            "attack_counts": self.attack_counts,
            "recent_classifications_count": len(self.recent_classifications),
            "active_connections": len(self.active_connections)
        }
    
    def get_recent_classifications(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent classification results"""
        recent = self.recent_classifications[-limit:] if limit else self.recent_classifications
        return [asdict(result) for result in recent]
    
    def get_attack_timeline(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get attack timeline for the last N minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        timeline = []
        for result in self.recent_classifications:
            try:
                result_time = datetime.fromisoformat(result.timestamp.replace('Z', '+00:00'))
                if result_time >= cutoff_time and result.attack_type:
                    timeline.append({
                        "timestamp": result.timestamp,
                        "attack_type": result.attack_type,
                        "severity": result.severity,
                        "confidence": result.confidence,
                        "source_ip": result.source_ip,
                        "destination_ip": result.destination_ip
                    })
            except:
                continue
        
        return timeline
    
    def get_network_graph_data(self) -> Dict[str, Any]:
        """Generate network graph data from recent classifications"""
        nodes = {}
        edges = []
        
        for result in self.recent_classifications[-200:]:  # Last 200 packets
            # Add source node
            if result.source_ip not in nodes:
                nodes[result.source_ip] = {
                    "id": result.source_ip,
                    "ip": result.source_ip,
                    "type": "device",
                    "attack_count": 0,
                    "normal_count": 0
                }
            
            # Add destination node
            if result.destination_ip not in nodes:
                nodes[result.destination_ip] = {
                    "id": result.destination_ip,
                    "ip": result.destination_ip,
                    "type": "device",
                    "attack_count": 0,
                    "normal_count": 0
                }
            
            # Update node statistics
            if result.attack_type:
                nodes[result.source_ip]["attack_count"] += 1
                nodes[result.destination_ip]["attack_count"] += 1
            else:
                nodes[result.source_ip]["normal_count"] += 1
                nodes[result.destination_ip]["normal_count"] += 1
            
            # Add edge
            edge = {
                "source": result.source_ip,
                "target": result.destination_ip,
                "protocol": result.protocol,
                "attack_type": result.attack_type,
                "severity": result.severity,
                "packet_count": 1,
                "timestamp": result.timestamp
            }
            edges.append(edge)
        
        return {
            "nodes": list(nodes.values()),
            "edges": edges
        }

# Global service instance
_simulation_service = None

def get_simulation_service() -> RealTimeSimulationService:
    """Get the global simulation service instance"""
    global _simulation_service
    if _simulation_service is None:
        _simulation_service = RealTimeSimulationService()
    return _simulation_service 