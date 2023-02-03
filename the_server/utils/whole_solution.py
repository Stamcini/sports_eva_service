import json
import copy
import os
import numpy as np
import dtw
from the_server.utils.video2mp_np import video2mp_np
from the_server.utils.mp_to_bvh_solution import BvhSolution, BvhNode


# from server_django.settings import BASE_DIR

# config_dir = os.path.join(BASE_DIR, 'configs')

class VideoSaver:
    video_raw_count = 0  # used for the naming of the raw videoss

    @staticmethod
    def save_from_request(request, save_dir: str) -> [bool, str, str]:
        if not request.method == 'POST':
            return [False, 'request method is not POST', '']
        else:
            file2store = request.FILES['file']
            if not file2store:
                return [False, 'no file in request', '']
            else:
                vid_addr = os.path.join(save_dir, 'raw_' + str(VideoSaver.video_raw_count) + '.mp4')
                with open(vid_addr, 'wb+') as f:
                    for chunk in file2store.chunks():
                        f.write(chunk)
                VideoSaver.video_raw_count += 1
                return [True, '', str(vid_addr)]


class ReturnJson:
    default = -1.
    score_items = ['holistic', 'torso', 'upper', 'lower']
    energy_items = ['energy', 'fat', 'energy_standard', 'fat_standard']
    return_dict_default = {
        'evaluations': {
            "scores": {
                "holistic": default,
                "torso": default,
                "upper": default,
                "lower": default,
            },
            "energy": {
                "energy": default,
                "fat": default,
                "energy_standard": default,
                "fat_standard": default
            }
        },
        'bvh': default,
    }

    def __init__(self, ret: bool, ):
        self.ret = ret
        if not ret:
            self.return_dict = self.return_dict_default
            return
        else:
            self.return_dict = copy.deepcopy(ReturnJson.return_dict_default)

    def get_json(self) -> str:
        return json.dumps(self.return_dict)


class Scoring:
    sport_types = {'high_knees': 0, 'jumping_jacks': 1, 'thoracic_rotation': 2}
    body_parts = ['holistic', 'torso', 'upper', 'lower']
    met_values = [8, 8, 3]
    fat_percentage = [0.3, 0.3, 0.5]

    def __init__(self, bvh: BvhSolution, sport_type: int, scoring_parts_config_json: str, model_seq_dir: str,
                 random_seq_dir: str,
                 model_vid_dir: str, random_seed: int = 1000):
        self.bvh = bvh
        self.armature_one = Scoring.armature_one_from_bvh(bvh)
        self.armature = Scoring.armature_from_bvh(bvh)
        with open(scoring_parts_config_json, 'r') as f:
            self.scoring_parts = json.loads(f.read())
        self.random_seed = random_seed
        if sport_type not in [0, 1, 2]:
            raise "sport type should fall in these values: [0,1,2]"
        if 'model_' + str(sport_type) + '.npy' not in os.listdir(model_seq_dir):
            self.regenerate_model_seq(model_seq_dir, model_vid_dir, sport_type)
        self.model_seq = np.load(model_seq_dir + '/model_' + str(sport_type) + '.npy')
        if 'random_' + str(sport_type) + '.npy' not in os.listdir(random_seq_dir):
            self.regenerate_maximum_random_seq(random_seq_dir, (self.model_seq.shape[0], 36, 3), random_seed,
                                               sport_type)
        self.random_seq = np.load(random_seq_dir + '/random_' + str(sport_type) + '.npy')

    def regenerate_model_seq(self, seq_dir: str, vid_dir: str, sport_type: int = -1) -> None:
        # model videos are names 0.mp4, 1.mp4 and 2.mp4
        # model seqs are named 0.
        if sport_type == -1:
            [self.regenerate_model_seq(seq_dir, vid_dir, i) for i in range(2)]
        elif sport_type not in [0, 1, 2]:
            raise "sport type should be either -1, 0, 1, 2"
        else:
            mp_dict_tmp = video2mp_np(vid_dir + '/' + str(sport_type) + '.mp4')
            bvh_tmp = BvhSolution(self.bvh.bvh_mp_config_json, self.bvh.mp_hierarchy_json,
                                  self.bvh.bvh_template_file)  # use self.bvh's configs here
            bvh_tmp.convert_mediapipe(mp_dict_tmp['mp_data'])
            model_seq = Scoring.get_normalized_mp_seq(bvh_tmp)
            np.save(seq_dir + '/model_' + str(sport_type) + '.npy', model_seq)

    def regenerate_maximum_random_seq(self, seq_dir: str, shape: (int, int, int), seed: int,
                                      sport_type: int = -1) -> None:
        if sport_type == -1:
            [self.regenerate_maximum_random_seq(seq_dir, shape, seed, i) for i in range(2)]
        elif sport_type not in [0, 1, 2]:
            raise "sport type should be either -1, 0, 1, 2"
        else:
            bvh_tmp = BvhSolution(self.bvh.bvh_mp_config_json, self.bvh.mp_hierarchy_json,
                                  self.bvh.bvh_template_file)  # use self.bvh's configs here
            bvh_tmp.convert_mediapipe(np.random.random(shape))
            random_seq_normalized = Scoring.get_normalized_mp_seq(bvh_tmp)
            np.save(seq_dir + '/random_' + str(sport_type) + '.npy', random_seq_normalized)

    @staticmethod
    def armature_one_from_bvh(bvh: BvhSolution) -> np.ndarray:
        # one frame of armature
        # which you can tile to form a time sequence
        # or just call armature_from_bvh()
        armature_one = -np.array([i.offset for i in bvh.nodes])

        def eq_add(x, y):
            x += y

        [eq_add(armature_one[node.index, :], armature_one[bvh.get_node_by_name(node.parent).index, :]) for node in
         bvh.nodes[1:]]
        return armature_one

    @staticmethod
    def armature_from_bvh(bvh: BvhSolution) -> np.ndarray:
        # generate blank armature seq from bvh
        armature_one = Scoring.armature_one_from_bvh(bvh)
        if bvh.mp_data is None:
            print("No mediapipe data in BvhSolution")
            return armature_one
        else:
            pass
        return np.tile(armature_one, (bvh.mp_data.shape[0], 1, 1))

    @staticmethod
    def rotate_one_node_n_children(armature_np: np.ndarray, bvh: BvhSolution, node_bvh_index: int):
        # armature.shape: (frame, bvh_index, (x,y,z))
        # works recursively

        if bvh.nodes[node_bvh_index].is_endsite:
            return
        else:
            [Scoring.rotate_one_node_n_children(armature_np, bvh, i) for i in
             bvh.get_direct_bvh_children(node_bvh_index)]
        previous_node_position = armature_np[:, node_bvh_index, :].copy()
        node_n_all_children = [node_bvh_index] + bvh.get_all_bvh_children(node_bvh_index)
        for i in node_n_all_children:
            for frame in range(armature_np.shape[0]):
                armature_np[frame, i, :] = np.linalg.inv(bvh.nodes[node_bvh_index].dcm[frame, :, :]) @ armature_np[
                                                                                                       frame, i, :]
        translation_vector = previous_node_position - armature_np[:, node_bvh_index, :]
        for i in node_n_all_children:
            armature_np[:, i, :] += translation_vector

    @staticmethod
    def rotate_armature_by_bvh(armature_np: np.ndarray, bvh: BvhSolution, root_node_index: int = 0):
        Scoring.rotate_one_node_n_children(armature_np, bvh, root_node_index)

    @staticmethod
    def get_normalized_mp_seq(bvh) -> np.ndarray:
        normalized_seq = Scoring.armature_from_bvh(bvh)
        Scoring.rotate_armature_by_bvh(normalized_seq, bvh)
        return normalized_seq

    @staticmethod
    def distance_euclidean(x: np.ndarray, y: np.ndarray):
        return np.sum((x - y) ** 2)

    @staticmethod
    def dtw2scores_1(x, maximum):
        return np.abs(1 - np.log(x + 1) / np.log(maximum + 1))

    @staticmethod
    def get_scores_static(subject_seq: np.ndarray, curated_lm_list: [int, ...], model_seq: np.ndarray,
                          random_seq: np.ndarray,
                          distance_method=distance_euclidean,
                          dtw2scores_method=dtw2scores_1) -> float:
        curated_model = np.take(model_seq, curated_lm_list, 1)
        curated_seq = np.take(subject_seq, curated_lm_list, 1)
        dtw_distance_accumulated = dtw.dtw(curated_model, curated_seq, distance_method)[0]
        curated_random_seq = np.take(random_seq, curated_lm_list, 1)
        dtw_distance_random = dtw.dtw(curated_model, curated_random_seq, distance_method)[0]
        score = dtw2scores_method(dtw_distance_accumulated, dtw_distance_random)
        return score

    def get_scores(self, mp_data, part: str) -> float:
        if part not in self.scoring_parts:
            raise "invalid part!"
        return Scoring.get_scores_static(mp_data, self.scoring_parts[part], self.model_seq, self.random_seq,
                                         self.distance_euclidean, self.dtw2scores_1)

    # @staticmethod
    # def holistic():
    #     pass
    #
    # @staticmethod
    # def torso():
    #     pass
    #
    # @staticmethod
    # def upper():
    #     pass
    #
    # @staticmethod
    # def lower():
    #     pass

    # @staticmethod
    # def get_scores(model_seq, subject_seq, body_part: str):
    #     pass

    @staticmethod
    def energy(weight: float, time_span: float) -> float:
        # weight in kg
        # time_span in seconds
        # return: energy in kcal

        return weight * 0.95 * time_span / 3600  # here use 0.95 rather than 1.05 to make it looks reasonable

    @staticmethod
    def fat(sport_type: int, weight: float, time_span: float) -> float:
        # sport_type in [0,1,2]
        # weight in kg
        # time_span in seconds
        # return: energy in kcal
        if sport_type not in [0, 1, 2]:
            raise "invalid sport type! [0,1,2] is stipulated"
        return Scoring.energy(weight, time_span) * Scoring.fat_percentage[sport_type]

    @staticmethod
    def energy_standard(weight: float, time_span: float) -> float:
        return weight * 1.05 * time_span / 3600

    @staticmethod
    def fat_standard(sport_type: int, weight: float, time_span: float) -> float:
        if sport_type not in [0, 1, 2]:
            raise "invalid sport type! [0,1,2] is stipulated"
        return Scoring.energy_standard(weight, time_span) * Scoring.fat_percentage[sport_type]


class WholeSolution:
    # stages=['load_video','video2mp','mp2bvh','scoring']

    def __init__(self, bvh_mp_config_json: str, mp_hierarchy_json: str, bvh_template_file: str,
                 scoring_parts_json: str, temp_dir: str, model_video_dir: str,
                 sport_type: int, time_span: float, weight: float):
        self.video = None
        self.mp_data = None

        self.ret = True  # error symbol
        self.error = ''  # error message if it fails somewhere
        self.scoring_parts_json = scoring_parts_json
        self.temp_dir = temp_dir
        self.model_video_dir = model_video_dir
        self.bvh = BvhSolution(bvh_mp_config_json, mp_hierarchy_json, bvh_template_file)

        # self.scoring_methods = {'holistic': self.scoring_holistic, 'torso': self.scoring_torso,
        #                         'upper': self.scoring_upper, 'lower': self.scoring_lower}
        # self.consumption_methods = {'energy': self.consumption_energy, 'fat': self.consumption_fat,
        #                             'energy_standard': self.consumption_energy_standard,
        #                             'fat_standard': self.consumption_fat_standard}
        self.output_dict = {}
        self.output_json_str = ''
        self.sport_type = sport_type
        self.time_span = time_span
        self.weight = weight

    # def get_scoring(self, which: str):
    #     self.scoring_methods[which]()

    # def get_consumption(self, which: str):
    #     self.consumption_methods[which]()

    def robust_workflow(self, request) -> [bool, str, str]:
        """
        this is the sole robust workflow which you can use in the server
        """

        # todo: use VideoSaver here
        [self.ret, self.error, video_path] = VideoSaver.save_from_request(request, )

        # video to mediapipe then to numpy
        video2mp_tmp = video2mp_np(self.video)
        if not video2mp_tmp['ret']:
            self.ret = False
            self.error = video2mp_tmp['error']
            self.output_dict = copy.deepcopy(ReturnJson.return_dict_default)
            self.output_dict['bvh'] = self.error
            self.output_json_str = json.dumps(self.output_dict)
            return [self.ret, self.error, self.output_json_str]
        else:
            pass
        self.mp_data = video2mp_tmp['mp_data']
        self.bvh.frame_rate = video2mp_tmp['frame_rate']

        # mediapipe to bvh
        self.bvh.convert_mediapipe(self.mp_data)

        # scoring & consumption
        scorer_tmp = Scoring(self.bvh, self.sport_type, self.scoring_parts_json, self.temp_dir,
                             self.temp_dir,
                             self.model_video_dir)

        mp_data_normalized = scorer_tmp.get_normalized_mp_seq(self.bvh)
        scores = [scorer_tmp.get_scores(mp_data_normalized, i) for i in Scoring.body_parts]

        energy = Scoring.energy(self.weight, self.time_span)
        fat = Scoring.fat(self.sport_type, self.weight, self.time_span)
        energy_std = Scoring.energy_standard(self.weight, self.time_span)
        fat_std = Scoring.fat_standard(self.sport_type, self.weight, self.time_span)

        # all into json output
        return_json_tmp = ReturnJson(True)
        return_json_tmp.return_dict['evaluations'] = {
            'scores': {
                'holistic': scores[0],
                'torso': scores[1],
                'upper': scores[2],
                'lower': scores[3]
            },
            'energy': {
                'energy': energy,
                'fat': fat,
                'energy_standard': energy_std,
                'fat_standard': fat_std
            }
        }

        self.bvh.write_bvh(self.bvh.frame_rate, self.temp_dir + '/bvh_tmp.bvh')
        with open(self.temp_dir + '/bvh_tmp.bvh', 'r') as f:
            return_json_tmp.return_dict['bvh'] = f.read()
        self.output_json_str = return_json_tmp.get_json()

        return [self.ret, self.error, self.output_json_str]
