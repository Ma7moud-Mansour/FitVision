import os

def extract_frames(video_path, output_folder, fps=10):
    # تأكد إن الفولدر موجود
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # command بتاع ffmpeg
    command = f'ffmpeg -i "{video_path}" -vf "fps=5" "{output_folder}/frame_%04d.png"'

    # تشغيل الكوماند
    result = os.system(command)

    if result != 0:
        print("❌ Error in extracting frames")
    else:
        print("✅ Frames extracted successfully")