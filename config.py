"""
Configuration for Qwen 3.5 VLM Detection API
"""

import os

# Model
MODEL_NAME = os.getenv("QWEN_MODEL", "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4")
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTIL", "0.95"))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "4096"))
TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
QUANTIZATION = os.getenv("QUANTIZATION", "moe_wna16")

# Server
PORT = int(os.getenv("PORT", "8000"))

# Detection
CONFIDENCE_THRESHOLD = 0.2

CATEGORIES = [
    "abandoned_vehicle",
    "unsheltered_encampment",
]

CATEGORY_PROMPTS = {
    "abandoned_vehicle": """Find ALL abandoned vehicles in this image. Look carefully at every vehicle.

What makes a vehicle ABANDONED:
- Old, damaged, rusty, or deteriorated appearance
- Flat or missing tires
- Covered with tarp/sheet (owner protecting it but not using it)
- Broken windows or missing parts
- Dusty, dirty, or weathered
- Parked on street for long time
- Looks like it cannot be driven
- Vegetation/weeds growing around wheels
- Extensive rust, body damage, or missing parts
- Faded paint with peeling/chipping
- Missing or expired license plates

DO NOT DETECT:
- Normal parked cars in good condition
- Cars in parking lots or driveways
- Moving or recently used vehicles
- Construction vehicles actively being used

Check EVERY vehicle in the image. Multiple abandoned vehicles may exist.

Output format - list ALL found:
[{"category": "abandoned_vehicle", "bbox": [x1,y1,x2,y2], "confidence": 0.8, "description": "brief reason"}]

Use pixel coordinates [x1,y1,x2,y2] where x1,y1 is top-left corner.
If none found, return: []""",

    "unsheltered_encampment": """Find homeless encampments in this image.

WHAT TO DETECT - actual shelters where people live outdoors:
- TENTS: Dome tents, camping tents (blue, orange, green fabric) - especially under bridges, overpasses, sidewalks
- TARPS used as ROOF: Tarp suspended between poles/trees/fences creating a covered living space
- MAKESHIFT SHELTERS: Cardboard structures, plywood lean-tos, blanket forts

WHERE encampments are found:
- Under bridges and overpasses
- Along sidewalks and streets
- In parks and vacant lots
- Near highway underpasses

DO NOT DETECT as encampments:
- Car covers or tarps draped OVER vehicles (that's just a covered car)
- Construction site materials
- Outdoor furniture with covers
- Vegetation, weeds, or overgrown areas
- Empty fenced areas with plants
- Trash or debris without shelter structures
- Random piles of items without visible shelter

CRITICAL: If you don't see tents or tarps, DO NOT DETECT!

Output: [{"category": "unsheltered_encampment", "bbox": [x1,y1,x2,y2], "confidence": 0.75, "description": "tent under bridge" or "tarp shelter"}]
If none found, return: []""",
}

# Sampling parameters
SAMPLING_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.8,
    "max_tokens": 512,
    "presence_penalty": 1.5,
    "top_k": 20,
}
