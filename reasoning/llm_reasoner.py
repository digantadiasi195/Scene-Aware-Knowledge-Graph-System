# reasoning/llm_reasoner.py
"""
LLM High-Level Goal Predictor (Scene-Aware + Noise-Resistant Version).
"""

import json
import ollama

class LLMReasoner:
    def __init__(self, model_name="llama3"):
        self.model = model_name
        print(f"🦙 Initializing LLM Reasoner ({model_name})...")
        ollama.chat(model=self.model, messages=[{'role': 'user', 'content': 'Hello'}])
        print("✅ LLM Reasoner ready.")

    def predict_goal(self, scene_name, scene_id, detected_objects, atomic_action, graph_relations):
        object_labels = [obj.get("label", obj.get("object_id")) for obj in detected_objects]
        unique_objects = list(set(object_labels))
        
        relation_summary = []
        for rel in graph_relations[:3]:
            relation_summary.append(f"{rel.get('subject', 'human')} is {rel.get('relation', 'near')} {rel.get('object', 'object')}")
        relation_text = "; ".join(relation_summary) if relation_summary else "No specific spatial relations detected"
        
        # Categorize objects by scene compatibility
        indoor_objects = ["bed", "cabinet", "remote", "spoon", "blanket", "mirror-stuff", 
                         "refrigerator", "oven", "microwave", "toaster", "sink", "couch", "chair"]
        outdoor_objects = ["car", "bus", "motorcycle", "truck", "bicycle", "tree", "road", 
                          "sky-other-merged", "building-other-merged", "traffic light", "stop sign"]
        
        detected_indoor = [o for o in unique_objects if o in indoor_objects]
        detected_outdoor = [o for o in unique_objects if o in outdoor_objects]
        
        # Flag potential false positives
        noise_warning = ""
        if "street" in scene_name.lower() or "outdoor" in scene_name.lower():
            if detected_indoor:
                noise_warning = f"\n⚠️ NOTE: The following indoor objects were detected but may be FALSE POSITIVES (misclassified vehicle/building parts): {', '.join(detected_indoor)}. IGNORE these when predicting the goal."
        elif "kitchen" in scene_name.lower() or "office" in scene_name.lower() or "living" in scene_name.lower():
            if detected_outdoor:
                noise_warning = f"\n⚠️ NOTE: The following outdoor objects were detected but may be FALSE POSITIVES: {', '.join(detected_outdoor)}. IGNORE these when predicting the goal."
        
        context_prompt = f"""You are an advanced AI assistant for an assistive robot. Analyze the scene and predict the human's INTENT.

### SCENE CONTEXT:
- **Location**: {scene_name} (ID: {scene_id})
- **Visible Objects**: {', '.join(unique_objects) if unique_objects else 'None detected'}
- **Current Atomic Action**: {atomic_action if atomic_action else 'Unknown/Idle'}
- **Spatial Relations**: {relation_text}
{noise_warning}

### YOUR TASK:
Predict what the human is most likely trying to achieve RIGHT NOW.

**CRITICAL RULES:**
1. **SCENE CONSISTENCY FIRST**: The scene type ({scene_name}) is the MOST RELIABLE signal. If detected objects contradict the scene (e.g., "bed" in a "street" scene), they are likely segmentation FALSE POSITIVES. IGNORE them.
2. Only predict goals that are CONSISTENT with the scene type.
3. Only use objects that MAKE SENSE in this environment.
4. If objects contradict the scene, base your prediction on the SCENE TYPE and any plausible objects (people, vehicles, buildings for outdoor; furniture, appliances for indoor).
5. If no clear goal can be determined, say "Observing environment".

### EXAMPLES OF CORRECT REASONING:
- Scene: "street" + Objects: [car, bus, person, building] → Goal: "Commuting" or "Crossing the road"
- Scene: "street" + Objects: [bed, mirror] (FALSE POSITIVES) → IGNORE bed/mirror, focus on street context
- Scene: "kitchen" + Objects: [bottle, bowl, person] → Goal: "Preparing food" or "Drinking"

### RESPONSE FORMAT (JSON only):
{{
  "goal": "Specific high-level goal",
  "confidence": "High" | "Medium" | "Low",
  "reasoning": "One sentence explaining WHY, mentioning which objects you used and which you ignored as false positives",
  "robot_assist": "Specific action the robot should take"
}}

Respond with JSON only, no markdown formatting."""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': context_prompt}],
                options={'temperature': 0.2}  # Lower for more deterministic, scene-consistent reasoning
            )
            
            llm_output = response['message']['content'].strip()
            
            if "```json" in llm_output:
                llm_output = llm_output.split("```json")[1].split("```")[0].strip()
            elif "```" in llm_output:
                llm_output = llm_output.split("```")[1].split("```")[0].strip()
                
            parsed_goal = json.loads(llm_output)
            parsed_goal["engine"] = "LLM (Llama 3)"
            return parsed_goal
            
        except Exception as e:
            print(f"[WARN] LLM Reasoning failed: {e}")
            return {
                "goal": "Observing environment",
                "confidence": "Low",
                "reasoning": f"LLM failed to parse context: {str(e)[:50]}",
                "robot_assist": "Continue monitoring.",
                "engine": "LLM (Fallback)"
            }