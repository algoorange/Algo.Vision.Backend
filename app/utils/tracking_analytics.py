"""
Advanced tracking analytics utilities for CCTV footage analysis.
Provides movement pattern analysis, anomaly detection, and traffic insights.
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
from datetime import datetime, timedelta

class TrackingAnalyzer:
    """Advanced analytics for object tracking data"""
    
    def __init__(self):
        self.movement_patterns = {}
        self.traffic_zones = {}
        self.speed_statistics = {}
    
    def analyze_track_movements(self, tracks_data: List[Dict]) -> Dict:
        """
        Analyze movement patterns from tracking data
        
        Args:
            tracks_data: List of track dictionaries with trajectory, movement_data, etc.
            
        Returns:
            Dictionary with comprehensive movement analysis
        """
        analysis = {
            "traffic_flow": self._analyze_traffic_flow(tracks_data),
            "speed_analysis": self._analyze_speeds(tracks_data),
            "direction_patterns": self._analyze_directions(tracks_data),
            "anomalies": self._detect_anomalies(tracks_data),
            "zone_activity": self._analyze_zones(tracks_data),
            "temporal_patterns": self._analyze_temporal_patterns(tracks_data),
            "object_interactions": self._detect_interactions(tracks_data)
        }
        
        return analysis
    
    def _analyze_traffic_flow(self, tracks_data: List[Dict]) -> Dict:
        """Analyze overall traffic flow patterns"""
        flow_data = {
            "total_objects": len(tracks_data),
            "moving_objects": 0,
            "stationary_objects": 0,
            "avg_speed": 0.0,
            "dominant_direction": "unknown",
            "flow_density": 0.0
        }
        
        if not tracks_data:
            return flow_data
        
        speeds = []
        directions = []
        
        for track in tracks_data:
            movement_data = track.get("movement_data", {})
            avg_speed = movement_data.get("avg_speed", 0.0)
            track_directions = movement_data.get("directions", [])
            
            speeds.append(avg_speed)
            
            if avg_speed > 2.0:  # Threshold for moving objects
                flow_data["moving_objects"] += 1
                directions.extend(track_directions)
            else:
                flow_data["stationary_objects"] += 1
        
        # Calculate statistics
        if speeds:
            flow_data["avg_speed"] = round(np.mean(speeds), 2)
        
        if directions:
            # Find dominant direction (most common direction range)
            direction_bins = self._bin_directions(directions)
            most_common = max(direction_bins.items(), key=lambda x: x[1])
            flow_data["dominant_direction"] = most_common[0]
        
        # Flow density (objects per frame area - simplified)
        flow_data["flow_density"] = flow_data["moving_objects"] / max(1, len(tracks_data))
        
        return flow_data
    
    def _analyze_speeds(self, tracks_data: List[Dict]) -> Dict:
        """Analyze speed distributions and patterns"""
        speeds = []
        max_speeds = []
        
        for track in tracks_data:
            movement_data = track.get("movement_data", {})
            avg_speed = movement_data.get("avg_speed", 0.0)
            max_speed = movement_data.get("max_speed", 0.0)
            
            speeds.append(avg_speed)
            max_speeds.append(max_speed)
        
        if not speeds:
            return {"avg_speed": 0, "max_speed": 0, "speed_variance": 0}
        
        return {
            "avg_speed": round(np.mean(speeds), 2),
            "max_speed": round(np.max(max_speeds), 2),
            "min_speed": round(np.min(speeds), 2),
            "speed_variance": round(np.var(speeds), 2),
            "speed_std": round(np.std(speeds), 2),
            "speed_distribution": self._categorize_speeds(speeds)
        }
    
    def _analyze_directions(self, tracks_data: List[Dict]) -> Dict:
        """Analyze movement direction patterns"""
        all_directions = []
        
        for track in tracks_data:
            movement_data = track.get("movement_data", {})
            directions = movement_data.get("directions", [])
            all_directions.extend(directions)
        
        if not all_directions:
            return {"dominant_directions": [], "direction_distribution": {}}
        
        # Bin directions into cardinal and intercardinal directions
        direction_bins = self._bin_directions(all_directions)
        
        # Sort by frequency
        sorted_directions = sorted(direction_bins.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "dominant_directions": [d[0] for d in sorted_directions[:3]],
            "direction_distribution": dict(sorted_directions),
            "direction_entropy": self._calculate_entropy(list(direction_bins.values()))
        }
    
    def _detect_anomalies(self, tracks_data: List[Dict]) -> List[Dict]:
        """Detect anomalous movement patterns"""
        anomalies = []
        
        # Calculate normal ranges
        speeds = [track.get("movement_data", {}).get("avg_speed", 0) for track in tracks_data]
        if speeds:
            speed_mean = np.mean(speeds)
            speed_std = np.std(speeds)
            speed_threshold = speed_mean + 2 * speed_std  # 2 standard deviations
        else:
            speed_threshold = float('inf')
        
        for track in tracks_data:
            track_id = track.get("track_id")
            movement_data = track.get("movement_data", {})
            avg_speed = movement_data.get("avg_speed", 0)
            trajectory = track.get("trajectory", [])
            
            # Speed anomalies
            if avg_speed > speed_threshold:
                anomalies.append({
                    "type": "excessive_speed",
                    "track_id": track_id,
                    "value": avg_speed,
                    "threshold": speed_threshold,
                    "description": f"Object moving at {avg_speed:.1f} units/frame (threshold: {speed_threshold:.1f})"
                })
            
            # Erratic movement detection
            if len(trajectory) > 5:
                direction_changes = self._count_direction_changes(trajectory)
                if direction_changes > len(trajectory) * 0.3:  # More than 30% direction changes
                    anomalies.append({
                        "type": "erratic_movement",
                        "track_id": track_id,
                        "value": direction_changes,
                        "description": f"Erratic movement pattern with {direction_changes} direction changes"
                    })
            
            # Long stationary periods
            stationary_frames = movement_data.get("stationary_frames", 0)
            total_frames = len(trajectory)
            if total_frames > 30 and stationary_frames / total_frames > 0.8:
                anomalies.append({
                    "type": "prolonged_stationary",
                    "track_id": track_id,
                    "value": stationary_frames / total_frames,
                    "description": f"Object stationary for {stationary_frames}/{total_frames} frames"
                })
        
        return anomalies
    
    def _analyze_zones(self, tracks_data: List[Dict]) -> Dict:
        """Analyze activity in different zones of the frame"""
        # Simple zone analysis - divide frame into 3x3 grid
        zones = defaultdict(int)
        
        for track in tracks_data:
            trajectory = track.get("trajectory", [])
            for point in trajectory:
                if len(point) >= 2:
                    x, y = point[0], point[1]
                    # Assuming frame dimensions (can be made configurable)
                    zone_x = min(2, int(x / 640 * 3))
                    zone_y = min(2, int(y / 480 * 3))
                    zone_id = f"zone_{zone_x}_{zone_y}"
                    zones[zone_id] += 1
        
        return dict(zones)
    
    def _analyze_temporal_patterns(self, tracks_data: List[Dict]) -> Dict:
        """Analyze temporal patterns in movement"""
        # This would be enhanced with actual timestamps
        patterns = {
            "track_durations": [],
            "avg_track_duration": 0.0,
            "short_tracks": 0,
            "long_tracks": 0
        }
        
        for track in tracks_data:
            duration = len(track.get("trajectory", []))
            patterns["track_durations"].append(duration)
            
            if duration < 10:
                patterns["short_tracks"] += 1
            elif duration > 50:
                patterns["long_tracks"] += 1
        
        if patterns["track_durations"]:
            patterns["avg_track_duration"] = np.mean(patterns["track_durations"])
        
        return patterns
    
    def _detect_interactions(self, tracks_data: List[Dict]) -> List[Dict]:
        """Detect potential interactions between objects"""
        interactions = []
        
        # Group tracks by time overlaps and proximity
        for i, track1 in enumerate(tracks_data):
            for j, track2 in enumerate(tracks_data[i+1:], i+1):
                # Check if tracks have temporal overlap
                frames1 = set(track1.get("frames", []))
                frames2 = set(track2.get("frames", []))
                
                overlap = frames1.intersection(frames2)
                if len(overlap) > 5:  # Significant temporal overlap
                    # Check spatial proximity during overlap
                    min_distance = self._calculate_min_distance_during_overlap(
                        track1, track2, overlap
                    )
                    
                    if min_distance < 50:  # Close proximity threshold
                        interactions.append({
                            "track1": track1.get("track_id"),
                            "track2": track2.get("track_id"),
                            "type1": track1.get("label"),
                            "type2": track2.get("label"),
                            "min_distance": min_distance,
                            "overlap_frames": len(overlap),
                            "interaction_type": self._classify_interaction(min_distance, overlap)
                        })
        
        return interactions
    
    def _bin_directions(self, directions: List[float]) -> Dict[str, int]:
        """Bin directions into cardinal/intercardinal categories"""
        bins = {
            "North": 0, "Northeast": 0, "East": 0, "Southeast": 0,
            "South": 0, "Southwest": 0, "West": 0, "Northwest": 0
        }
        
        for direction in directions:
            # Normalize to 0-360
            normalized = direction % 360
            
            if 337.5 <= normalized or normalized < 22.5:
                bins["East"] += 1
            elif 22.5 <= normalized < 67.5:
                bins["Northeast"] += 1
            elif 67.5 <= normalized < 112.5:
                bins["North"] += 1
            elif 112.5 <= normalized < 157.5:
                bins["Northwest"] += 1
            elif 157.5 <= normalized < 202.5:
                bins["West"] += 1
            elif 202.5 <= normalized < 247.5:
                bins["Southwest"] += 1
            elif 247.5 <= normalized < 292.5:
                bins["South"] += 1
            else:  # 292.5 <= normalized < 337.5
                bins["Southeast"] += 1
        
        return bins
    
    def _categorize_speeds(self, speeds: List[float]) -> Dict[str, int]:
        """Categorize speeds into slow/medium/fast"""
        categories = {"slow": 0, "medium": 0, "fast": 0}
        
        for speed in speeds:
            if speed < 2.0:
                categories["slow"] += 1
            elif speed < 10.0:
                categories["medium"] += 1
            else:
                categories["fast"] += 1
        
        return categories
    
    def _calculate_entropy(self, values: List[int]) -> float:
        """Calculate entropy of a distribution"""
        total = sum(values)
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for value in values:
            if value > 0:
                p = value / total
                entropy -= p * math.log2(p)
        
        return round(entropy, 3)
    
    def _count_direction_changes(self, trajectory: List[Tuple]) -> int:
        """Count significant direction changes in trajectory"""
        if len(trajectory) < 3:
            return 0
        
        changes = 0
        prev_direction = None
        
        for i in range(1, len(trajectory)):
            p1, p2 = trajectory[i-1], trajectory[i]
            direction = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
            
            if prev_direction is not None:
                angle_diff = abs(direction - prev_direction)
                if angle_diff > math.pi / 4:  # 45 degree threshold
                    changes += 1
            
            prev_direction = direction
        
        return changes
    
    def _calculate_min_distance_during_overlap(self, track1: Dict, track2: Dict, overlap_frames: set) -> float:
        """Calculate minimum distance between tracks during overlap"""
        min_distance = float('inf')
        
        traj1 = track1.get("trajectory", [])
        traj2 = track2.get("trajectory", [])
        frames1 = track1.get("frames", [])
        frames2 = track2.get("frames", [])
        
        # Create frame to position mapping
        pos1_by_frame = {frames1[i]: traj1[i] for i in range(min(len(frames1), len(traj1)))}
        pos2_by_frame = {frames2[i]: traj2[i] for i in range(min(len(frames2), len(traj2)))}
        
        for frame in overlap_frames:
            if frame in pos1_by_frame and frame in pos2_by_frame:
                p1, p2 = pos1_by_frame[frame], pos2_by_frame[frame]
                distance = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                min_distance = min(min_distance, distance)
        
        return min_distance if min_distance != float('inf') else 0.0
    
    def _classify_interaction(self, min_distance: float, overlap_frames: set) -> str:
        """Classify the type of interaction based on distance and duration"""
        if min_distance < 20 and len(overlap_frames) > 10:
            return "close_encounter"
        elif min_distance < 30:
            return "proximity"
        elif len(overlap_frames) > 20:
            return "parallel_movement"
        else:
            return "passing"

def generate_tracking_report(tracks_data: List[Dict]) -> str:
    """Generate a comprehensive tracking analysis report"""
    analyzer = TrackingAnalyzer()
    analysis = analyzer.analyze_track_movements(tracks_data)
    
    report = f"""
CCTV TRACKING ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

=== TRAFFIC FLOW SUMMARY ===
Total Objects: {analysis['traffic_flow']['total_objects']}
Moving Objects: {analysis['traffic_flow']['moving_objects']}
Stationary Objects: {analysis['traffic_flow']['stationary_objects']}
Average Speed: {analysis['traffic_flow']['avg_speed']} units/frame
Dominant Direction: {analysis['traffic_flow']['dominant_direction']}
Flow Density: {analysis['traffic_flow']['flow_density']:.2f}

=== SPEED ANALYSIS ===
Average Speed: {analysis['speed_analysis']['avg_speed']} units/frame
Maximum Speed: {analysis['speed_analysis']['max_speed']} units/frame
Speed Variance: {analysis['speed_analysis']['speed_variance']:.2f}

=== DIRECTION PATTERNS ===
Dominant Directions: {', '.join(analysis['direction_patterns']['dominant_directions'][:3])}
Direction Entropy: {analysis['direction_patterns']['direction_entropy']}

=== DETECTED ANOMALIES ===
Total Anomalies: {len(analysis['anomalies'])}
"""
    
    for anomaly in analysis['anomalies'][:5]:  # Show first 5 anomalies
        report += f"- {anomaly['type']}: {anomaly['description']}\n"
    
    report += f"""
=== ZONE ACTIVITY ===
Most Active Zone: {max(analysis['zone_activity'].items(), key=lambda x: x[1])[0] if analysis['zone_activity'] else 'None'}

=== INTERACTIONS ===
Detected Interactions: {len(analysis['object_interactions'])}
"""
    
    return report 