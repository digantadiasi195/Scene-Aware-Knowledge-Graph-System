# knowledge_seed.py
"""
ONTOLOGY LAYER -- hand-authored domain knowledge only.

This file intentionally contains NO probabilities and NO mu/sigma
statistics. It only declares what scenes, objects, actions and activities
*exist*, and which nominal action sequences build each activity (this is
domain knowledge -- e.g. "drinking water nominally involves reaching then
drinking" -- exactly like the paper's Fig. 9 graphs). All numeric
statistics (HAS_OBJECT.p, HAS_ACTIVITY.p, AFFORDS.mu/sigma2,
TRANSITIONS_TO.p, NEAR.p) are learned later by stats_learning.py from the
instance layer -- never hardcoded here.
"""

# ---------------------------------------------------------------- SCENES ---
SCENES = [
    {"scene_id": "SC1", "name": "office"},
    {"scene_id": "SC2", "name": "kitchen"},
    {"scene_id": "SC3", "name": "living_room"},
    {"scene_id": "SC4", "name": "bedroom"},
    {"scene_id": "SC5", "name": "bathroom"},
]

# Rich environmental context per scene (lighting / hazards / workspace type).
# This is the "Scene as true contextual variable" extension: these
# properties can later gate or re-weight predictions (e.g. low light ->
# lower confidence in vision-derived geometry; hazardous workspace ->
# raise robot-assistance priority for the same predicted activity).
SCENE_CONDITIONS = [
    {"condition_id": "COND_SC1", "scene_id": "SC1", "lighting": "bright",
     "hazard_level": "low", "workspace_type": "desk_work"},
    {"condition_id": "COND_SC2", "scene_id": "SC2", "lighting": "bright",
     "hazard_level": "medium", "workspace_type": "food_prep"},  # hot surfaces, knives
    {"condition_id": "COND_SC3", "scene_id": "SC3", "lighting": "medium",
     "hazard_level": "low", "workspace_type": "leisure"},
    {"condition_id": "COND_SC4", "scene_id": "SC4", "lighting": "dim",
     "hazard_level": "low", "workspace_type": "rest"},
    {"condition_id": "COND_SC5", "scene_id": "SC5", "lighting": "medium",
     "hazard_level": "high", "workspace_type": "wet_area"},  # slip risk
]

# --------------------------------------------------------------- OBJECTS ---
OBJECTS = [
    {"object_id": "o_glass",     "name": "glass",      "category": "small"},
    {"object_id": "o_bottle",    "name": "bottle",      "category": "small"},
    {"object_id": "o_computer",  "name": "computer",    "category": "large"},
    {"object_id": "o_doorknob",  "name": "door_knob",   "category": "small"},
    {"object_id": "o_milk",      "name": "milk",        "category": "small"},
    {"object_id": "o_bowl",      "name": "bowl",        "category": "small"},
    {"object_id": "o_object",    "name": "object",      "category": "small"},
    {"object_id": "o_book",      "name": "book",        "category": "small"},
    {"object_id": "o_phone",     "name": "phone",       "category": "small"},
    {"object_id": "o_microwave", "name": "microwave",   "category": "large"},
    {"object_id": "o_table",     "name": "table",       "category": "large"},
    {"object_id": "o_door",      "name": "door",        "category": "large"},
]

# --------------------------------------------------------------- ACTIONS ---
ACTIONS = [
    {"action_id": "a1", "name": "approaching"},
    {"action_id": "a2", "name": "reaching"},
    {"action_id": "a3", "name": "pressing"},
    {"action_id": "a4", "name": "pouring"},
    {"action_id": "a5", "name": "placing"},
    {"action_id": "a6", "name": "drinking"},
    {"action_id": "a7", "name": "opening"},
    {"action_id": "a8", "name": "talking"},
    {"action_id": "a9", "name": "passing"},
]

# ------------------------------------------------------------- ACTIVITIES --
# Only the STRUCTURE (which action, on which object, in which order) is
# domain knowledge. No probabilities or geometry parameters live here.
ACTIVITIES = [
    {
        "activity_id": "AC1", "name": "drinking_water",
        "possible_scenes": ["SC1", "SC2", "SC3"],
        "sequences": [
            [("a2", "o_bottle"), ("a4", "o_bottle"), ("a5", "o_bottle"), ("a6", "o_glass")],
            [("a2", "o_bottle"), ("a6", "o_bottle")],
            [("a2", "o_glass"),  ("a6", "o_glass")],
        ],
    },
    {
        "activity_id": "AC2", "name": "activate_computer",
        "possible_scenes": ["SC1"],
        "sequences": [[("a2", "o_computer"), ("a3", "o_computer")]],
    },
    {
        "activity_id": "AC3", "name": "making_cereal",
        "possible_scenes": ["SC2"],
        "sequences": [[("a1", "o_table"), ("a2", "o_milk"), ("a4", "o_milk"), ("a5", "o_bowl")]],
    },
    {
        "activity_id": "AC4", "name": "arranging_books",
        "possible_scenes": ["SC1", "SC3"],
        "sequences": [[("a2", "o_book"), ("a9", "o_book"), ("a5", "o_book")]],
    },
    {
        "activity_id": "AC5", "name": "take_out_warming_food",
        "possible_scenes": ["SC2"],
        "sequences": [[("a2", "o_microwave"), ("a7", "o_microwave"), ("a5", "o_object")]],
    },
    {
        "activity_id": "AC6", "name": "picking_up_phone",
        "possible_scenes": ["SC1", "SC3", "SC4"],
        "sequences": [[("a2", "o_phone"), ("a8", "o_phone")]],
    },
    {
        "activity_id": "AC7", "name": "opening_door",
        "possible_scenes": ["SC1", "SC2", "SC3", "SC4", "SC5"],
        "sequences": [[("a1", "o_door"), ("a2", "o_doorknob"), ("a7", "o_door")]],
    },
]