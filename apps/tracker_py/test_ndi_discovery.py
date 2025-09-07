import NDIlib
import time

# Initialize the NDI system
if not NDIlib.initialize():
    print("Failed to initialize NDI library.")
    exit(1)

print("NDI library initialized successfully!")

# Create a find instance
find_instance = NDIlib.find_create_v2()

if not find_instance:
    print("Failed to create NDI find instance.")
    NDIlib.destroy()
    exit(1)

print("Looking for NDI sources... (waiting 5 seconds)")
time.sleep(5)

# Get current list of sources
sources = NDIlib.find_get_current_sources(find_instance)

if not sources:
    print("No NDI sources found.")
else:
    print(f"Found {len(sources)} NDI source(s):")
    for idx, source in enumerate(sources):
        source_name = source.ndi_name
        print(f"  [{idx+1}] {source_name}")

# Cleanup
NDIlib.find_destroy(find_instance)
NDIlib.destroy()
print("NDI library shutdown.")

