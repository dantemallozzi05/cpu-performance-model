import os
import json

# Global json save helper script
def save_json(obj, path):
    os.makedirs(os.path.dirnam(path), exist_ok=True)

    with open(path, "w") as file:
        json.dump(obj, file, indent=2)