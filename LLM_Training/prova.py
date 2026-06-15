import os
from PIL import ImageGrab

counter = 0
output_frames_dir = os.path.join("results", f"frames")
os.makedirs(output_frames_dir, exist_ok=True)

# Definiamo la scatola del secondo monitor (da X=1920 a X=3840)
secondo_monitor_box = (1920, 0, 3840, 1080)

# Cattura lo schermo (all_screens=True permette a Pillow su Mac di vedere oltre il primo monitor)
screenshot = ImageGrab.grab(bbox=secondo_monitor_box, all_screens=True)

frame_path = os.path.join(output_frames_dir, f"frame_{counter}.png")
screenshot.save(frame_path)

print(f"Screen salvato correttamente in: {frame_path}")