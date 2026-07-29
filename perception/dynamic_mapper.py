# perception/dynamic_mapper.py
"""
Dynamic Scene & Object Mapper (Open-World Generalized Version).
1. Uses Open-Vocabulary CLIP to recognize 20+ environments.
2. Dynamically creates new Scene nodes in Neo4j if a novel scene is detected.
3. Uses fuzzy matching to map unknown object labels to the ontology.
"""
from difflib import get_close_matches

class DynamicMapper:
    def __init__(self, neo4j_driver, database="neo4j"):
        self.driver = neo4j_driver
        self.database = database
        self._scene_cache = {}
        self._object_cache = {}
        self._refresh_cache()

    def _refresh_cache(self):
        try:
            with self.driver.session(database=self.database) as session:
                scenes = session.run(
                    "MATCH (s:Scene) RETURN s.scene_id AS id, s.name AS name"
                ).data()
                objects = session.run(
                    "MATCH (o:Object) RETURN o.object_id AS id, o.name AS name"
                ).data()
                
                self._scene_cache = {s["name"].lower(): s["id"] for s in scenes}
                self._object_cache = {o["name"].lower(): o["id"] for o in objects}
                
                # Keep legacy aliases as fallbacks
                self._scene_cache.update({
                    "kitchen": "SC2", "office": "SC1", "living room": "SC3", 
                    "factory": "SC4", "bedroom": "SC5"
                })
        except Exception as e:
            print(f"[WARN] DynamicMapper cache refresh failed: {e}")

    def get_all_scenes_for_clip(self):
        """
        Returns a massive open-vocabulary list for CLIP zero-shot classification.
        This allows the system to recognize environments beyond the 4 hardcoded ones.
        """
        neo4j_scenes = list(self._scene_cache.keys()) if self._scene_cache else []
        
        # Open-World Fallback List
        open_vocab_scenes = [
            "kitchen", "office", "living room", "factory", "hospital room",
            "classroom", "bedroom", "bathroom", "supermarket", "restaurant",
            "library", "gym", "park", "street", "garage", "laboratory",
            "warehouse", "hotel room", "cafe", "conference room", "outdoor"
        ]
        
        # Combine and remove duplicates
        return list(set(neo4j_scenes + open_vocab_scenes))

    def ensure_scene_exists(self, scene_name):
        """
        Dynamic Ontology Expansion: 
        If the scene doesn't exist in Neo4j, dynamically create it 
        and assign it a new Scene ID (e.g., SC6, SC7).
        """
        scene_name_lower = scene_name.lower().strip()
        
        # If it already exists in our cache, do nothing
        if scene_name_lower in self._scene_cache:
            return self._scene_cache[scene_name_lower]
            
        # It's a NEW scene! Create it in Neo4j dynamically.
        try:
            with self.driver.session(database=self.database) as session:
                # Find the highest existing Scene ID number
                result = session.run("""
                    MATCH (s:Scene) 
                    RETURN max(toInteger(substring(s.scene_id, 2))) AS max_id
                """).single()
                
                current_max = result["max_id"] if result and result["max_id"] is not None else 5
                new_scene_id = f"SC{current_max + 1}"
                
                # Create the new Scene node
                session.run("""
                    MERGE (s:Scene {scene_id: $id, name: $name})
                """, id=new_scene_id, name=scene_name)
                
                # Update local cache
                self._scene_cache[scene_name_lower] = new_scene_id
                print(f"[INFO] 🌍 Dynamically created new scene in Neo4j: {new_scene_id} ({scene_name})")
                
                return new_scene_id
        except Exception as e:
            print(f"[ERROR] Failed to create dynamic scene: {e}")
            return "SC1" # Ultimate fallback

    def map_scene(self, clip_scene_name):
        """Maps CLIP's scene output to Neo4j scene ID."""
        name_lower = clip_scene_name.lower().replace(" scene", "").strip()
        
        # 1. Exact match in cache
        if name_lower in self._scene_cache:
            return self._scene_cache[name_lower]
        
        # 2. Fuzzy match
        matches = get_close_matches(name_lower, self._scene_cache.keys(), n=1, cutoff=0.6)
        if matches:
            return self._scene_cache[matches[0]]
        
        # 3. NO MATCH FOUND -> Dynamically create this new scene in Neo4j!
        return self.ensure_scene_exists(name_lower)

    def map_object(self, coco_label):
        """Maps a COCO label to Neo4j object ID."""
        label_lower = coco_label.lower().strip()
        
        aliases = {
            "person": "o_human", "human": "o_human",
            "wine glass": "o_glass", "cup": "o_glass",
            "dining table": "o_table", "cell phone": "o_phone",
            "couch": "o_chair", "potted plant": "o_object",
            "tv": "o_computer", "tvmonitor": "o_computer",
        }
        
        if label_lower in aliases:
            return aliases[label_lower]
        if label_lower in self._object_cache:
            return self._object_cache[label_lower]
        
        matches = get_close_matches(label_lower, self._object_cache.keys(), n=1, cutoff=0.7)
        if matches:
            return self._object_cache[matches[0]]
        
        return "o_object"