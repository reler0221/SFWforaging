# Deinterlace all .MTS video and save them without audio.
import os
import ffmpeg
import time


#%% function
def deinterlace_videos(input_root, output_root, pause_every, pause_seconds):
    for dirpath, _, filenames in os.walk(input_root):
        all_files = []
        for filename in filenames:
            if filename.lower().endswith('.mts') and not filename.startswith("."):
                all_files.append(os.path.join(dirpath, filename))
    total = len(all_files)

    for i, input_path in enumerate(all_files, 1): # i starts as 1
        # Build relative path and corresponding output path
        relative_path = os.path.relpath(input_path, input_root)
        output_path = os.path.join(output_root, os.path.splitext(relative_path)[0] + '.mp4')

        # Skip if already processed
        if os.path.exists(output_path):
            print("Skipping (already exists)", output_path)
            continue


        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        # Skip if already processed
        if os.path.exists(output_path):
            print(f"[{i}/{total}] Skipping (already exists): {output_path}")
            continue

        # FFmpeg deinterlace command using yadif
        print("deinterlacing: ", input_path)

        try:
            (
                ffmpeg.input(input_path)
                .filter("yadif")
                .output(output_path,
                        threads = 6,
                        vcodec="libx264",
                        preset = "fast",
                        crf = 23,
                        an = None
                        )
                .overwrite_output()
                .run(quiet=True)
            )
            print("Saved to: ", output_path)
        except ffmpeg.Error as e:
            print(f"Error processing {input_path}:\n{e.stderr.decode()}")

        if i % pause_every == 0:
            print(f"Cooling down for {pause_seconds} seconds...")
            time.sleep(pause_seconds)



#%% Set paths
input_root = '/Volumes/T5 EVO/Foraging HD/Video'
output_root = '/Volumes/T5 EVO/Foraging HD/Videos_deinterlaced'

#%% Run!
batch_size = 30
pause_length = 30
deinterlace_videos(input_root, output_root, 30, 30)