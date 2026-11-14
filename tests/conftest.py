import os

# Tests start the app; it must never delete the developer's real caches or data.
os.environ["STORAGE_CLEANUP"] = "false"
