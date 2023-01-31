import cv2
import mediapipe as mp
import numpy as np
import tqdm

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


def process_one_video(video: str) -> [bool, list, float]:
    cap = cv2.VideoCapture()
    ret = cap.open(video)
    if not ret:
        print('open video failed: ' + video)
        return [False, [], None]

    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    results_output = []

    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
        for frame in tqdm.tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                results_output.append([False, None])
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results_frame = holistic.process(image)
            results_output.append([True, results_frame])

    cap.release()
    return [True, results_output, frame_rate]


def video2mp_np(video: str) -> dict:
    ret, results_all, frame_rate = process_one_video(video)
    if not ret:
        return {
            'ret': False,
            'mp_data': None,
            'frame_rate': None,
            'error': 'load video failed',
        }
    pose_3d_lms = list(map(lambda x: x[1].pose_world_landmarks.landmark if x[0] and x[
        1].pose_world_landmarks is not None else None, results_all[
                                                       :-5]))  # '-5' for omitting the last several frames that might be null in some videos

    # check if the failed frames are more than 10%, return False or interpolate
    count_failed = 0
    for i in pose_3d_lms:
        if i is None:
            count_failed += 1
    if count_failed >= len(pose_3d_lms) * 0.1:
        return {
            'ret': False,
            'mp_data': None,
            'frame_rate': frame_rate,
            'error': 'too many failed frames',
        }
    else:
        pose_3d_lms = deal_with_failed_frame(pose_3d_lms)

    pose_3d_lms_np = np.zeros((len(pose_3d_lms), 33, 4), dtype=np.float64)  # [frame, lm_index, [x,y,z,visibility]]
    for i in range(len(pose_3d_lms)):
        for j in range(33):
            pose_3d_lms_np[i, j, 0] = pose_3d_lms[i][j].x
            pose_3d_lms_np[i, j, 1] = pose_3d_lms[i][j].y
            pose_3d_lms_np[i, j, 2] = pose_3d_lms[i][j].z
            pose_3d_lms_np[i, j, 3] = pose_3d_lms[i][j].visibility
    return {
        'ret': True,
        'mp_data': pose_3d_lms_np,
        'frame_rate': frame_rate,
        'error': '',
    }


class PseudoLandmark:
    properties = ['x', 'y', 'z', 'visibility']

    def __init__(self, x: float, y: float, z: float, visibility: float):
        self.x, self.y, self.z, self.visibility = x, y, z, visibility

    @staticmethod
    def interpolate(start, end, number2interp: int):
        def multiple_linspace(pairs: list, number: int) -> [PseudoLandmark, ...]:
            increments = [(pair[1] - pair[0]) / (number + 1) for pair in pairs]
            out = [pair[0] for pair in pairs]
            for i in range(1, number + 1):
                out = [out[j] + increments[j] for j in range(len(out))]
                yield out

        interp_values = [list(
            multiple_linspace(
                [(getattr(start_channel, prop), getattr(end_channel, prop)) for prop in PseudoLandmark.properties],
                number=number2interp)) for (start_channel, end_channel) in zip(start, end)]
        # return [[PseudoLandmark(*one) for one in one_channel] for one_channel in interp_values]
        return [[PseudoLandmark(*channel[i]) for channel in interp_values]for i in range(number2interp)]


def deal_with_failed_frame(lm_list: list) -> list:
    # trim the starting ones
    first_not_null = 0
    for i, result in enumerate(lm_list):
        if result is not None:
            first_not_null = i
            break
        else:
            pass

    # trim the ending ones
    last_not_null = len(lm_list) - 1
    while last_not_null > 0:
        if lm_list[last_not_null] is not None:
            break
        else:
            last_not_null -= 1

    # get the trimmed list
    out_list = lm_list[first_not_null:last_not_null + 1]

    # interpolate the failed frames
    i = 1
    while i < len(out_list) - 2:  # -2 for the last one is definitely not None
        if out_list[i] is not None:  # !!!!!!!!!!!!! all True even no landmarks identified!!!!!!!!!!!!!!!!!
            i += 1
        else:
            failed_seg_start = i

            # find the end of the failed segment
            failed_seg_end = i
            for j in range(i + 1, len(out_list) - 1):
                if out_list[j] is None:
                    pass
                else:
                    failed_seg_end = j - 1
                    break

            # interpolate it
            number2interp = failed_seg_end - failed_seg_start + 1
            interp_list = PseudoLandmark.interpolate(out_list[failed_seg_start - 1], out_list[failed_seg_end + 1],
                                                     number2interp)
            for j in range(failed_seg_start, failed_seg_end + 1):
                out_list[j] = interp_list.pop(0)

            # skip the interpolated segment
            i = failed_seg_end + 1

    return out_list
