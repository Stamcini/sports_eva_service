import json
import copy
import os
import numpy as np
import dtw
from .video2mp_np import video2mp_np
from .mp_to_bvh_solution import BvhSolution, BvhNode


class ReturnJson:
    default = -1
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

    def get_json(self):
        return json.dumps(self.return_dict)


class Scoring:
    sport_types = {'high_knees': 0, 'jumping_jacks': 1, 'thoracic_rotation': 2}
    body_parts = ['holistic', 'torso', 'upper', 'lower']

    def __init__(self, bvh: BvhSolution, sport_type: int, scoring_parts_config_json: str, model_seq_dir: str,
                 model_vid_dir: str, random_seed: int = 1000):
        self.bvh = bvh
        self.armature_one = Scoring.armature_one_from_bvh(bvh)
        self.armature = Scoring.armature_from_bvh(bvh)
        with open(scoring_parts_config_json, 'r') as f:
            self.scoring_parts = json.loads(f.read())
        self.random_seed = random_seed
        if sport_type in [0, 1, 2] and str(sport_type) + '.npy' not in os.listdir(model_seq_dir):
            Scoring.regenerate_model_seq(model_seq_dir, model_vid_dir, sport_type)
        else:
            raise "sport type should fall in these values: [0,1,2]"
        self.model_seq = np.load(model_seq_dir + '/' + str(sport_type) + '.npy')
        self.maximum_dtw_dist = {}

    def regenerate_model_seq(self, seq_dir: str, vid_dir: str, sport_type: int = -1):
        # model videos are names 0.mp4, 1.mp4 and 2.mp4
        # model seqs are named 0.
        if sport_type == -1:
            [self.regenerate_model_seq(seq_dir, vid_dir, i) for i in range(2)]
        elif sport_type not in [0, 1, 2]:
            raise "sport type should be either -1, 0, 1, 2"
        else:
            bvh_tmp = BvhSolution()  # use self.bvh's configs here
            Scoring.get_normalized_mp_seq()

    @staticmethod
    def regenerate_maximum_random_seq(shape: (int, int, int), seed: int) -> np.ndarray:
        pass  # todo

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
    def get_scores(subject_seq: np.ndarray, curated_lm_list: [int, ...], model_seq: np.ndarray,
                   distance_method: staticmethod = distance_euclidean,
                   dtw2scores_method: staticmethod = dtw2scores_1) -> float:
        curated_seq = np.take(subject_seq, curated_lm_list, 1)
        dtw_distance_accumulated = dtw.dtw(model_seq, curated_seq, distance_method)
        score = dtw2scores_method(dtw_distance_accumulated, )

        return score

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
    def energy():
        pass

    @staticmethod
    def fat():
        pass

    @staticmethod
    def energy_standard():
        pass

    @staticmethod
    def fat_standard():
        pass


class WholeSolution:
    # stages=['load_video','video2mp','mp2bvh','scoring']

    def __init__(self, video: str, bvh_mp_config_json: str, mp_hierarchy_json: str, bvh_template_file: str,
                 temp_path: str):
        self.video = video
        self.frame_rate = None
        self.mp_data = None

        self.ret = True  # error symbol
        self.error = ''  # error message if it fails somewhere
        self.bvh = BvhSolution(bvh_mp_config_json, mp_hierarchy_json, bvh_template_file)

        self.scoring_methods = {'holistic': self.scoring_holistic, 'torso': self.scoring_torso,
                                'upper': self.scoring_upper, 'lower': self.scoring_lower}
        self.consumption_methods = {'energy': self.consumption_energy, 'fat': self.consumption_fat,
                                    'energy_standard': self.consumption_energy_standard,
                                    'fat_standard': self.consumption_fat_standard}
        self.output_dict = {}
        self.output_json_str = ''

    def scoring_holistic(self):
        pass

    def scoring_torso(self):
        pass

    def scoring_upper(self):
        pass

    def scoring_lower(self):
        pass

    def get_scoring(self, which: str):
        self.scoring_methods[which]()

    def consumption_energy(self):
        pass

    def consumption_fat(self):
        pass

    def consumption_energy_standard(self):
        pass

    def consumption_fat_standard(self):
        pass

    def get_consumption(self, which: str):
        self.consumption_methods[which]()

    def robust_workflow(self) -> [bool, str, str]:
        """
        this is the sole robust workflow which you can use in the server
        """

        # video to mediapipe then to numpy
        video2mp_tmp = video2mp_np(self.video)
        if not video2mp_tmp['ret']:
            self.error = video2mp_tmp['error']
            self.output_dict = copy.deepcopy(ReturnJson.return_dict_default)
            self.output_dict['bvh'] = self.error
            self.output_json_str = json.dumps(self.output_dict)
            return [self.ret, self.error, self.output_json_str]
        else:
            pass
        self.mp_data = video2mp_tmp['mp_data']
        self.frame_rate = video2mp_tmp['frame_rate']

        # mediapipe to bvh
        self.bvh.convert_mediapipe(self.mp_data)

        # scoring & consumption

        # all into json output

        return [self.ret, self.error, self.output_json_str]
