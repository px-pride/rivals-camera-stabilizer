import numpy as np 
import sys
import cv2
import time
import copy
import os
import traceback
import ffmpeg
import subprocess as sp
import os.path as path
from pprint import pprint
from concurrent.futures import ThreadPoolExecutor
from PIL import ImageColor
np.set_printoptions(threshold=np.inf)


def is_point_in_box(point, box, camera_position):
    return ((point[0] >= box[0][0] - camera_position[0]) and
            (point[0] < box[1][0] - camera_position[0]) and
            (point[1] >= box[0][1] - camera_position[1]) and
            (point[1] < box[1][1] - camera_position[1]))


def get_points(frame, feature_detector, feature_sparsity, negate_boxes, camera_position):
    kp = feature_detector.detect(frame, None)
    if len(kp) == 0:
        return []
    pt = np.array([kp[i].pt for i in range(len(kp))])
    pt_key = np.sum(
        pt // feature_sparsity * np.array([frame.shape[0] // feature_sparsity, 1]),
        axis=-1)
    p0 = []
    p0_buckets = dict()
    for i in range(len(pt)):
        p0_key = pt_key[i]
        if p0_key not in p0_buckets:
            p0_buckets[p0_key] = True
            flag = True
            for box in negate_boxes:
                if is_point_in_box(pt[i], box, camera_position):
                    flag = False
                    break
            if flag:
                p0.append(pt[i])
    p0 = np.float32(p0).reshape(-1, 1, 2)  # convert to numpy
    return p0


def preproc_frame(frame, proc_xres, proc_yres, pad_x, pad_y, pad_color):
    colored = frame #[:585, :1040]
    colored = cv2.resize(
        colored, (proc_xres, proc_yres), interpolation = cv2.INTER_NEAREST)
    colored = np.pad(
        colored, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)))
    gray = cv2.cvtColor(colored, cv2.COLOR_BGR2GRAY)  # make grayscale frame
    pad_bgr = ImageColor.getcolor(pad_color, "RGB")[::-1]
    colored[:,:pad_x,:] = pad_bgr
    colored[:,-pad_x:,:] = pad_bgr
    colored[:pad_y,:,:] = pad_bgr
    colored[-pad_y:,:,:] = pad_bgr
    return colored, gray


def read_logic(reader):
    try:
        frame = reader.read()
        return frame
    except Exception as e:
        traceback.print_exc()    


def write_logic(writer, frame_w, tracker, frame_track):
    try:
        tracker.write(frame_track)
        return writer.write(frame_w)
    except Exception as e:
        traceback.print_exc()    


def core_logic(frame_r, 
               frame_ctr,
               out_xres,
               out_yres,
               proc_xres,
               proc_yres,
               pad_x,
               pad_y,
               pad_color,
               feature_detector,
               feature_sparsity,
               negate_boxes,
               draw_features,
               camera_position,
               reference_gray,
               reference_p0):
    try:
        current_colored, current_gray = preproc_frame(
            frame_r, proc_xres, proc_yres, pad_x, pad_y, pad_color)
        current_gray = np.roll(
            current_gray, (-int(camera_position[1]), -int(camera_position[0])), axis=(0,1))

        if frame_ctr == 0 or len(reference_p0) == 0:
            frame_w = current_colored
            frame_w = np.roll(current_colored, (-int(camera_position[1]), -int(camera_position[0])), axis=(0,1))
            if draw_features:
                for box in negate_boxes:
                    frame_w = cv2.rectangle(
                        frame_w,
                        (box[0][0] - camera_position[0], box[0][1] - camera_position[1]),
                        (box[1][0] - camera_position[0], box[1][1] - camera_position[1]),
                        (0, 0, 255), 5)

        else:
            buckets = dict()
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                reference_gray, current_gray, reference_p0, None)
            diff = (p1[:,0,:] - reference_p0[:,0,:]) #* 1080 / proc_yres
            for i in range(len(diff)):
                key_x = int(round(diff[i][0]))
                key_y = int(round(diff[i][1]))
                key = str(key_x) + "," + str(key_y)
                if key not in buckets:
                    buckets[key] = []
                buckets[key].append(p1[i][0])

            if len(buckets) > 0:
                argmax = max(buckets, key=lambda key: len(buckets[key]))  # choose bucket with most elements
                #argmax *= proc_yres / 1080
                current_stage_points = list(buckets[argmax])
                current_non_stage_points = []
                for bucket in buckets:
                    if bucket != argmax:
                        current_non_stage_points += list(buckets[bucket])
                argmax = argmax.split(",")
                adjustment = [int(argmax[0]), int(argmax[1])]
            else:
                current_stage_points = []
                current_non_stage_points = []
                adjustment = [0, 0]
            current_gray = np.roll(current_gray, (-adjustment[1], -adjustment[0]), axis=(0,1))

            # draw all the FAST features for debugging
            camera_position[0] += adjustment[0]
            camera_position[1] += adjustment[1]
            frame_w = np.roll(current_colored, (-int(camera_position[1]), -int(camera_position[0])), axis=(0,1))
            if draw_features:
                for i in range(len(current_stage_points)):
                    center = (int(current_stage_points[i][0]), int(current_stage_points[i][1]))
                    frame_w = cv2.circle(frame_w, center, 4, (255,255,0), -1)
                for i in range(len(current_non_stage_points)):
                    center = (int(current_non_stage_points[i][0]), int(current_non_stage_points[i][1]))
                    frame_w = cv2.circle(frame_w, center, 4, (0,255,0), -1)
                for box in negate_boxes:
                    frame_w = cv2.rectangle(
                        frame_w,
                        (box[0][0] - camera_position[0], box[0][1] - camera_position[1]),
                        (box[1][0] - camera_position[0], box[1][1] - camera_position[1]),
                        (0, 0, 255), 5)

        reference_gray = current_gray
        reference_p0 = get_points(
            reference_gray, feature_detector, feature_sparsity, negate_boxes, camera_position)
        frame_w = cv2.resize(frame_w, (out_xres, out_yres), interpolation = cv2.INTER_NEAREST)

        frame_track = np.zeros((proc_yres, proc_xres, 3), dtype=np.uint8)
        frame_track[camera_position[1] + proc_yres // 2][camera_position[0] + proc_xres // 2][0] = 0
        frame_track[camera_position[1] + proc_yres // 2][camera_position[0] + proc_xres // 2][1] = 0
        frame_track[camera_position[1] + proc_yres // 2][camera_position[0] + proc_xres // 2][2] = 255

        return frame_w, frame_track, camera_position, reference_gray, reference_p0

    except Exception as e:
        traceback.print_exc()


def main(in_file,
         out_file='stabilized.mp4',
         track_file='tracker.mp4',
         out_xres=3840,
         out_yres=2160,
         proc_xres=1920,
         proc_yres=1080,
         pad_x=960,
         pad_y=540,
         pad_color='#00ff00',
         fps=60.0,
         mp4_bitrate=320000,
         feature_sparsity=40,
         feature_threshold=40,
         draw_features=False):

    # input file reader
    reader = cv2.VideoCapture(in_file)

    # output file writer
    if out_file.endswith('.avi'):
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(out_file, fourcc, fps, (out_xres, out_yres))
    elif out_file.endswith('.mp4'):
        writer = (
            ffmpeg.input(
                'pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(
                    out_xres, out_yres), r=fps)
            .output(out_file, pix_fmt='yuv420p', video_bitrate=mp4_bitrate)
            .overwrite_output()
            .run_async(pipe_stdin=True)
            .stdin
        )
    else:
        raise ValueError('Only MP4 and AVI output formats supported.')

    # tracking file writer
    if track_file.endswith('.avi'):
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        tracker = cv2.VideoWriter(track_file, fourcc, fps, (proc_xres, proc_yres))
    elif track_file.endswith('.mp4'):
        tracker = (
            ffmpeg.input(
                'pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(
                    proc_xres, proc_yres), r=fps)
            .output(track_file, pix_fmt='yuv420p', video_bitrate=mp4_bitrate)
            .overwrite_output()
            .run_async(pipe_stdin=True)
            .stdin
        )
    else:
        raise ValueError('Only MP4 and AVI output formats supported.')

    # create feature detector
    feature_detector = cv2.FastFeatureDetector_create()
    feature_detector.setNonmaxSuppression(False) 
    feature_detector.setThreshold(feature_threshold)
    negate_boxes = [
        [(pad_x, pad_y),
         (pad_x + 1920 * proc_xres // 1920, pad_y + 540 * proc_yres // 1080)],
        [(pad_x + 470 * proc_xres // 1920, pad_y + 980 * proc_yres // 1080),
         (pad_x + 1450 * proc_xres // 1920, pad_y + 1080 * proc_yres // 1080)],
        [(pad_x, pad_y + 1020 * proc_yres // 1080),
         (pad_x + 120 * proc_xres // 1920, pad_y + 1080 * proc_yres // 1080)],
        [(pad_x + 1800 * proc_xres // 1920, pad_y + 1020 * proc_yres // 1080),
         (pad_x + 1920 * proc_xres // 1920, pad_y + 1080 * proc_yres // 1080)],
    ] # areas to avoid feature detection

    # state variables
    ret = True
    frame_r = None
    frame_w = None
    frame_track = None
    frame_ctr = 0
    camera_position = [0,0]
    reference_gray = None
    reference_p0 = None

    # multi-threaded main loop
    with ThreadPoolExecutor(max_workers=3) as executor:
        while ret or frame_r is not None or frame_w is not None:
            if frame_w is not None:
                write_thread = executor.submit(
                    write_logic, writer, 
                    cv2.cvtColor(frame_w, cv2.COLOR_BGR2RGB),#.tostring(),
                    tracker, cv2.cvtColor(frame_track, cv2.COLOR_BGR2RGB))
                frame_draw = cv2.resize(
                    frame_w, (1280, 720), interpolation = cv2.INTER_NEAREST)
                cv2.imshow('Frame', frame_draw) # draw frame on screen for debugging
                if cv2.waitKey(1) == 27:
                    print("Terminating early.")
                    break
                if (frame_ctr + 1) % 60 == 0:
                    print("Processed " + str((frame_ctr + 1) // 60) + " second" + (
                        "s" if (frame_ctr + 1) // 60 != 1 else "") + " of video.")
                if not ret and frame_r is None:
                    frame_w = None
                frame_ctr += 1
            if frame_r is not None:
                core_thread = executor.submit(
                    core_logic,
                    frame_r, 
                    frame_ctr,
                    out_xres,
                    out_yres,
                    proc_xres,
                    proc_yres,
                    pad_x,
                    pad_y,
                    pad_color,
                    feature_detector,
                    feature_sparsity,
                    negate_boxes,
                    draw_features,
                    camera_position,
                    reference_gray,
                    reference_p0)  # core logic on current frame
                frame_w, frame_track, camera_position, reference_gray, reference_p0 = core_thread.result()
            if ret:
                read_thread = executor.submit(read_logic, reader)  # read next frame
                ret, frame_r = read_thread.result()
    
    if out_file.endswith(".avi"):
        writer.release()
    elif out_file.endswith("mp4"):
        writer.stdin.close()
        writer.stderr.close()
        writer.communicate()
    if track_file.endswith(".avi"):
        tracker.release()
    elif track_file.endswith("mp4"):
        tracker.stdin.close()
        tracker.stderr.close()
        tracker.communicate()
    reader.release()


if __name__ == '__main__':
    main(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        int(sys.argv[4]),
        int(sys.argv[5]),
        int(sys.argv[6]),
        int(sys.argv[7]),
        int(sys.argv[8]),
        int(sys.argv[9]),
        sys.argv[10],
        float(sys.argv[11]),
        int(sys.argv[12]),
        int(sys.argv[13]),
        int(sys.argv[14]),
        sys.argv[15] == "True")
