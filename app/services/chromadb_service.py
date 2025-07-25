import chromadb
import os
from typing import List, Dict, Any
import json

class ChromaDBService:
    def __init__(self):
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        # Create or get collections
        self.video_collection = self.client.get_or_create_collection(
            name="video_analysis",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.frame_collection = self.client.get_or_create_collection(
            name="frame_objects",
            metadata={"hnsw:space": "cosine"}
        )
    
    def store_video_analysis(self, video_data: Dict[str, Any]):
        """
        Store video analysis data in ChromaDB
        """
        try:
            # Create document for video analysis
            video_id = video_data.get("video_id")
            summary = video_data.get("natural_language_summary", "")
            tracks = video_data.get("tracks", [])
            
            # Create metadata
            metadata = {
                "video_id": video_id,
                "video_name": video_data.get("file_path", ""),
                "duration": video_data.get("summary", {}).get("duration_seconds", 0),
                "total_tracks": len(tracks),
                "analysis_type": "object_detection"
            }
            
            # Add to collection
            self.video_collection.add(
                documents=[summary],
                metadatas=[metadata],
                ids=[f"video_{video_id}"]
            )
            
            print(f"✅ Stored video analysis in ChromaDB: {video_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error storing video analysis in ChromaDB: {e}")
            return False
    
    def store_frame_objects(self, video_id: str, frames_data: List[Dict[str, Any]]):
        """
        Store frame-by-frame object data in ChromaDB
        """
        try:
            documents = []
            metadatas = []
            ids = []
            
            for frame_data in frames_data:
                frame_id = frame_data.get("frame_id")
                frame_number = frame_data.get("frame_number")
                objects = frame_data.get("objects", [])
                
                # Create document text from objects
                object_descriptions = []
                for obj in objects:
                    obj_desc = f"Track ID {obj.get('track_id')}: {obj.get('object_type')} at position {obj.get('position')} with confidence {obj.get('confidence')}"
                    object_descriptions.append(obj_desc)
                
                document_text = f"Frame {frame_number} contains: {'; '.join(object_descriptions)}"
                
                # Create metadata - convert lists to strings for ChromaDB compatibility
                object_types = list(set([obj.get("object_type", "unknown") for obj in objects]))
                object_types_str = ", ".join(object_types) if object_types else "none"
                
                metadata = {
                    "video_id": video_id,
                    "frame_id": frame_id,
                    "frame_number": frame_number,
                    "frame_time": frame_data.get("frame_time", 0),
                    "total_objects": len(objects),
                    "object_types": object_types_str  # Convert list to string
                }
                
                documents.append(document_text)
                metadatas.append(metadata)
                ids.append(f"frame_{video_id}_{frame_number}")
            
            # Add to collection
            if documents:
                self.frame_collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                print(f"✅ Stored {len(documents)} frames in ChromaDB for video: {video_id}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error storing frame objects in ChromaDB: {e}")
            return False
    
    def search_videos(self, query: str, n_results: int = 5):
        """
        Search videos by query
        """
        try:
            results = self.video_collection.query(
                query_texts=[query],
                n_results=n_results
            )
            return results
        except Exception as e:
            print(f"❌ Error searching videos: {e}")
            return None
    
    def search_frames(self, query: str, n_results: int = 10):
        """
        Search frames by query
        """
        try:
            results = self.frame_collection.query(
                query_texts=[query],
                n_results=n_results
            )
            return results
        except Exception as e:
            print(f"❌ Error searching frames: {e}")
            return None

# Create global instance
chromadb_service = ChromaDBService() 