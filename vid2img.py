import os, sys, glob, cv2
from argparse import ArgumentParser
from tqdm.auto import tqdm
from multiprocessing import Process


def do_vid2img(video_file, output, video_ext, resize_resolution):
    target_folder = os.path.join(
        output, video_file.replace(".%s" % video_ext, "").split(os.sep).pop()
    )

    if not os.path.isdir(target_folder):
        os.makedirs(target_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_id in tqdm(range(frame_count), desc='Processing file %s -> %s\id_04d.jpg' % (video_file, target_folder)):
        ret, frame = cap.read()
        if not ret:
            print("cv2: early break, reason: EOF")
            break

        save_filename = os.path.join(target_folder, "%04d.jpg" % frame_id)
        frame = cv2.resize(frame, resize_resolution)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video-path', type=str, required=True, help='Path to folder storing video files')
    parser.add_argument('--video-ext', default='m4v', type=str, help='Video extension name (Default: m4v)')
    parser.add_argument('--width', default=1280, type=int, help='Converted resolution width (Default: 1280)')
    parser.add_argument('--height', default=720, type=int, help='Converted resolution height (Default: 720)')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel processing per video file (Default: False)')
    parser.add_argument('--output', default='./output', type=str, help='Output folder (Default: ./output)')
    args = parser.parse_args()

    video_file_list = list(glob.glob(os.path.join(args.video_path, "*.%s" % args.video_ext)))
    output = args.output
    parallel = args.parallel
    video_ext = args.video_ext

    if not os.path.isdir(output):
        os.makedirs(output, exist_ok=True)
    
    if parallel:
        video_file_count = len(video_file_list)
        print("Using total %d subprocesses." % video_file_count)
        
        procs = []
        for video_file in video_file_list:
            procs.append(
                Process(target=do_vid2img, args=(video_file, output, video_ext, (args.width, args.height)))
            )
        
        [proc.start() for proc in procs]
        [proc.join() for proc in procs]
    else:
        for video_file in video_file_list:
            do_vid2img(video_file, output, video_ext, (args.width, args.height))