"""
JAL SANKETH - Safe Place Database Module
Module Path: evacuation/safe_places.py

This module provides functionality to find the nearest safe places (evacuation centers)
for users during flood emergencies in Telangana, India.

Key Features:
-------------
1. Loads safe place database from CSV file
2. Calculates distances using Haversine formula
3. Returns nearest safe places sorted by distance
4. Simple, beginner-friendly implementation
"""

import pandas as pd
import numpy as np
import os
import math
from typing import List, Dict, Tuple, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAFE_PLACES_CSV = os.path.join(PROJECT_ROOT, "data", "safe_places.csv")


class SafePlaceDatabase:
    """
    Database class for managing safe places (evacuation centers).
    """
    
    def __init__(self, csv_path: Optional[str] = None):
        """
        Initialize the safe place database.
        
        Parameters:
        -----------
        csv_path : str, optional
            Path to the safe_places.csv file. Defaults to data/safe_places.csv
        """
        self.csv_path = csv_path or SAFE_PLACES_CSV
        self.safe_places_df = None
        self.load_database()
    
    def load_database(self) -> bool:
        """
        Load the safe places database from CSV file.
        
        Returns:
        --------
        bool : True if loaded successfully, False otherwise
        """
        if not os.path.exists(self.csv_path):
            print(f"Warning: Safe places database not found at {self.csv_path}")
            return False
        
        try:
            self.safe_places_df = pd.read_csv(self.csv_path)
            print(f"Loaded {len(self.safe_places_df)} safe places from database")
            return True
        except Exception as e:
            print(f"Error loading safe places database: {e}")
            return False
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great circle distance between two points 
        on Earth using the Haversine formula.
        
        Parameters:
        -----------
        lat1, lon1 : float
            Latitude and longitude of first point (user location)
        lat2, lon2 : float
            Latitude and longitude of second point (safe place)
            
        Returns:
        --------
        float : Distance in kilometers
        """
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Radius of Earth in kilometers
        r = 6371.0
        
        return c * r
    
    def find_nearest_safe_places(
        self, 
        user_lat: float, 
        user_lon: float, 
        limit: int = 5,
        max_distance_km: Optional[float] = None
    ) -> List[Dict]:
        """
        Find the nearest safe places to a user's location.
        
        Parameters:
        -----------
        user_lat : float
            User's latitude
        user_lon : float
            User's longitude
        limit : int
            Maximum number of safe places to return (default: 5)
        max_distance_km : float, optional
            Maximum distance in kilometers to search (default: no limit)
            
        Returns:
        --------
        List[Dict] : List of safe places sorted by distance (nearest first)
                     Each dictionary contains: name, latitude, longitude, type, 
                     capacity, elevation, distance_km
        """
        if self.safe_places_df is None or self.safe_places_df.empty:
            print("Error: Safe places database is empty or not loaded")
            return []
        
        # Calculate distance to each safe place
        distances = []
        for _, row in self.safe_places_df.iterrows():
            safe_place_lat = row['latitude']
            safe_place_lon = row['longitude']
            
            distance = self.calculate_distance(user_lat, user_lon, safe_place_lat, safe_place_lon)
            
            # Apply maximum distance filter if specified
            if max_distance_km is not None and distance > max_distance_km:
                continue
            
            safe_place_info = {
                'name': row['name'],
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'type': row['type'],
                'capacity': row['capacity'],
                'elevation': row['elevation'],
                'distance_km': round(distance, 2)
            }
            distances.append(safe_place_info)
        
        # Sort by distance (nearest first)
        distances.sort(key=lambda x: x['distance_km'])
        
        # Return only the requested number of results
        return distances[:limit]
    
    def get_safe_place_by_name(self, name: str) -> Optional[Dict]:
        """
        Get information about a specific safe place by name.
        
        Parameters:
        -----------
        name : str
            Name of the safe place to search for
            
        Returns:
        --------
        Dict : Safe place information or None if not found
        """
        if self.safe_places_df is None:
            return None
        
        matching_places = self.safe_places_df[self.safe_places_df['name'].str.contains(name, case=False, na=False)]
        
        if matching_places.empty:
            return None
        
        row = matching_places.iloc[0]
        return {
            'name': row['name'],
            'latitude': row['latitude'],
            'longitude': row['longitude'],
            'type': row['type'],
            'capacity': row['capacity'],
            'elevation': row['elevation']
        }
    
    def get_all_safe_places(self) -> List[Dict]:
        """
        Get all safe places in the database.
        
        Returns:
        --------
        List[Dict] : List of all safe places
        """
        if self.safe_places_df is None:
            return []
        
        return self.safe_places_df.to_dict('records')


def find_nearest_safe_places(
    user_lat: float, 
    user_lon: float, 
    limit: int = 5,
    csv_path: Optional[str] = None
) -> List[Dict]:
    """
    Convenience function to find nearest safe places without creating a database instance.
    
    Parameters:
    -----------
    user_lat : float
        User's latitude
    user_lon : float
        User's longitude
    limit : int
        Maximum number of safe places to return (default: 5)
    csv_path : str, optional
        Path to the safe_places.csv file
        
    Returns:
    --------
    List[Dict] : List of safe places sorted by distance (nearest first)
    """
    db = SafePlaceDatabase(csv_path)
    return db.find_nearest_safe_places(user_lat, user_lon, limit)


# ==============================================================================
# TEST FUNCTION
# ==============================================================================
def test_safe_places_module():
    """
    Test function to verify the safe places module works independently.
    """
    print("=" * 70)
    print("SAFE PLACES MODULE TEST")
    print("=" * 70)
    
    # Test 1: Initialize database
    print("\n1. Testing database initialization...")
    db = SafePlaceDatabase()
    if db.safe_places_df is not None:
        print(f"[OK] Database loaded successfully with {len(db.safe_places_df)} safe places")
    else:
        print("[FAIL] Database failed to load")
        return
    
    # Test 2: Distance calculation
    print("\n2. Testing distance calculation...")
    # Distance between Hyderabad (17.3850, 78.4867) and Warangal (17.9689, 79.5941)
    distance = db.calculate_distance(17.3850, 78.4867, 17.9689, 79.5941)
    print(f"[OK] Distance between Hyderabad and Warangal: {distance:.2f} km")
    
    # Test 3: Find nearest safe places
    print("\n3. Testing nearest safe places search...")
    # Use Hyderabad coordinates as example
    user_lat = 17.3850
    user_lon = 78.4867
    
    nearest = db.find_nearest_safe_places(user_lat, user_lon, limit=3)
    
    if nearest:
        print(f"[OK] Found {len(nearest)} nearest safe places to Hyderabad:")
        for i, place in enumerate(nearest, 1):
            print(f"  {i}. {place['name']} ({place['type']}) - {place['distance_km']} km away")
            print(f"     Capacity: {place['capacity']}, Elevation: {place['elevation']}m")
    else:
        print("[FAIL] No safe places found")
    
    # Test 4: Test with maximum distance filter
    print("\n4. Testing with maximum distance filter (10 km)...")
    nearest_limited = db.find_nearest_safe_places(user_lat, user_lon, limit=5, max_distance_km=10.0)
    if nearest_limited:
        print(f"[OK] Found {len(nearest_limited)} safe places within 10 km")
    else:
        print("[OK] No safe places within 10 km (as expected for test data)")
    
    # Test 5: Get all safe places
    print("\n5. Testing get all safe places...")
    all_places = db.get_all_safe_places()
    print(f"[OK] Total safe places in database: {len(all_places)}")
    
    print("\n" + "=" * 70)
    print("[SUCCESS] TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    test_safe_places_module()
