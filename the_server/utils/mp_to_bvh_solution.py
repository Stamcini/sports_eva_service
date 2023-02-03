import numpy as np


class BvhNode:
    # name = ''
    # parent = None
    # children = []
    # offset = np.zeros(3)
    # channels = 'zxy'
    # is_endsite = False
    # is_root = False
    # index = None
    # hier_depth = None
    # mp_index = None
    # mp_joint_type = None
    # dcm = None
    # eulers = None
    # ref_axes = None
    # ref_axes_afterwards = None

    def __init__(self, params: dict):
        self.name = ''
        self.parent = None
        self.children = []
        self.offset = np.zeros(3)
        self.channels = 'zxy'
        self.is_endsite = False
        self.is_root = False
        self.index = None
        self.hier_depth = None
        self.mp_index = None
        self.mp_joint_type = None
        self.dcm = None
        self.eulers = None
        self.ref_axes = None
        self.ref_axes_afterwards = None
        for i in params:
            setattr(self, i, params[i])

    def check_all_set(self):
        pass

    def dcm2eulers(self):
        from ..utils import math3d
        eulers = np.zeros((self.dcm.shape[0], 3))
        for i in range(eulers.shape[0]):
            eulers[i, :] = math3d.quat2euler(math3d.dcm2quat(self.dcm[i, :, :]), order='zxy')
        self.eulers = eulers / 3.141592653589793 * 180  # rad2deg


# class BvhHierarchy:
#     def __init__(self):dsa
#         pass


class BvhSolution:
    # bvh_mp_config_json = None
    # bvh_mp_config = None
    # mp_hierarchy_json = None
    # mp_hierarchy = None
    # bvh_template_file = None
    #
    # nodes = []
    #
    # mp_data: np.ndarray = None
    # frame_rate = None

    def __init__(self, bvh_mp_config_json: str, mp_hierarchy_json: str, bvh_template_file: str):
        self.bvh_mp_config_json = bvh_mp_config_json
        self.bvh_mp_config = None
        self.mp_hierarchy_json = mp_hierarchy_json  # which defines the hierarchy of the mp
        self.mp_hierarchy = None
        self.bvh_template_file = bvh_template_file
        self.nodes = []
        self.mp_data: np.ndarray or None = None
        self.frame_rate = None
        self.initialize()

    def initialize(self):
        import json
        with open(self.bvh_mp_config_json, 'r', encoding='utf-8') as f:
            self.bvh_mp_config = json.load(f)
        for i in self.bvh_mp_config:
            one_node = self.bvh_mp_config[i]
            params_dict = {
                'name': one_node['bvh_name'],
                'parent': one_node['parent'],
                'children': [],
                'offset': np.array([one_node['offset'][axis] for axis in one_node['offset']]),  # in order [x, y, z]
                'channels': 'zxy',
                'is_endsite': one_node['is_endsite'],
                'is_root': one_node['is_root'],
                'index': one_node['bvh_index'],
                'hier_depth': None,
                'mp_index': one_node['mp_index'],
                'mp_joint_type': one_node['mp_joint_type']
            }
            if params_dict['is_root']:
                params_dict['hier_depth'] = 0
            else:
                params_dict['hier_depth'] = getattr(self.get_node_by_name(params_dict['parent']), 'hier_depth') + 1
                self.get_node_by_name(params_dict['parent']).children.append(params_dict['name'])

            node_tmp = BvhNode(params_dict)
            self.nodes.append(node_tmp)

        with open(self.mp_hierarchy_json, 'r', encoding='utf-8') as f:
            self.mp_hierarchy = json.load(f)

    # def nodes2json(self,file:str=None):
    #     import json
    #     contents=json.dumps(self.nodes)
    #     if file is not None:
    #         with open(file,'w',encoding='utf-8') as f:
    #             f.write(contents)
    #     return contents

    def get_node_by_name(self, name) -> BvhNode:
        get_by_name = lambda node: True if getattr(node, 'name') == name else False
        return next(filter(get_by_name, self.nodes))

    def interp_original_mp(self, mp_data_original: np.ndarray) -> np.ndarray:
        data_out = mp_data_original
        mp_get = lambda node: data_out[:, self.bvh_mp_config[node]['mp_index'], :]
        pos_hips = np.expand_dims(0.5 * (mp_get('LeftHip') + mp_get('RightHip')), 1)
        pos_neck = np.expand_dims(0.5 * (mp_get('LeftShoulder') + mp_get('RightShoulder')), 1)
        data_out = np.append(data_out, pos_hips, axis=1)
        data_out = np.append(data_out, pos_neck, axis=1)
        pos_chest = np.expand_dims(0.5 * (mp_get('Hips') + mp_get('Neck')), 1)
        data_out = np.append(data_out, pos_chest, axis=1)
        return data_out

    def convert_mediapipe(self, mp_data: np.ndarray, nodes_list: list = None):
        if nodes_list is None:
            nodes_to_convert = self.nodes
        else:
            nodes_to_convert = [self.get_node_by_name(x) for x in nodes_list]
        self.mp_data = self.interp_original_mp(mp_data)

        for i in nodes_to_convert:
            if i.mp_joint_type == 'vector':
                self.convert_vector(i)
            elif i.mp_joint_type == 'rigid':
                self.convert_rigid_body(i)
            else:
                continue
            i.dcm2eulers()

    def dcm2eulers_update(self, nodes2update: [list, None] = None):
        if nodes2update is None:
            nodes2update = self.nodes
        else:
            pass
        for i in nodes2update:
            if i.mp_joint_type != 'EndSite':
                i.dcm2eulers()

    def get_all_mp_children(self, mp_index: int) -> list:
        # get all the children no matter direct or indirect of an mp index
        # for visualization of the matchstick-man
        all_children = []
        all_children.extend(
            [int(x) for x in filter(lambda x: True if self.mp_hierarchy[x] == mp_index else False, self.mp_hierarchy)])
        children_s_children = []
        for i in all_children:
            children_s_children.extend(self.get_all_mp_children(i))
        all_children.extend(children_s_children)
        return all_children  # [str, ...]

    def get_direct_bvh_children(self, bvh_index: int) -> [int, ...]:
        # direct_children = []
        # direct_children.extend([node.index for node in
        #                         filter(lambda x: True if x.parent == self.nodes[bvh_index].name else False,
        #                                self.nodes)])
        direct_children=[self.get_node_by_name(i).index for i in self.nodes[bvh_index].children]
        return direct_children

    def get_all_bvh_children(self, bvh_index: int) -> [int, ...]:
        all_children = self.get_direct_bvh_children(bvh_index)
        childrens_children = []
        for i in all_children:
            childrens_children.extend(self.get_all_bvh_children(i))
        all_children.extend(childrens_children)
        return all_children

    def get_vector_2mp_nodes(self, mp_node_from: int, mp_node_to: int) -> np.ndarray:
        return self.mp_data[:, mp_node_to, :3] - self.mp_data[:, mp_node_from, :3]  # not normalized yet

    def get_random_ortho_vec(self, x: np.ndarray) -> np.ndarray:
        # for self.convert_ridig_body
        if np.abs(x[0]) >= 1e-10:
            point_in_plane = np.array([x.sum() / x[0], 0, 0])
        elif np.abs(x[1]) >= 1e-10:
            point_in_plane = np.array([0, x.sum() / x[1], 0])
        elif np.abs(x[2]) >= 1e-10:
            point_in_plane = np.array([0, 0, x.sum() / x[2]])
        else:
            raise Exception('Zero vector is rejected')
        return point_in_plane - np.ones(3)

    def get_tri_ortho_vectors(self, two_of_3_vectors: dict, main_ref_axis: str) -> dict:  # {x, y, z}
        """
        params:
           {
                'x':
                'y':
                'z': # one of these three is None
           },

           'x'
        """
        output_dict = {main_ref_axis: two_of_3_vectors[main_ref_axis]}
        null_axis = next(filter(lambda x: True if two_of_3_vectors[x] is None else False, two_of_3_vectors))
        axes_list = ['x', 'y', 'z']
        output_dict[null_axis] = np.cross(two_of_3_vectors[axes_list[axes_list.index(null_axis) - 2]],
                                          two_of_3_vectors[axes_list[axes_list.index(null_axis) - 1]])
        for i in axes_list:
            if i not in output_dict:
                output_dict[i] = np.cross(output_dict[axes_list[axes_list.index(i) - 2]],
                                          output_dict[axes_list[axes_list.index(i) - 1]])
        return output_dict

    def joint_fixed_rotation(self, node: BvhNode, axis_vectors: dict) -> None:
        previous_joint_coordinates = self.mp_data[:, node.mp_index, :3].copy()
        # rotation
        from ..utils import math3d
        rot_matrix = np.zeros((self.mp_data.shape[0], 3, 3))
        for i in range(self.mp_data.shape[0]):
            rot_matrix[i, :, :] = math3d.dcm_from_axis(axis_vectors['x'][i, :], axis_vectors['y'][i, :],
                                                       axis_vectors['z'][i, :], order='xyz')
            for j in self.get_all_mp_children(node.mp_index) + [node.mp_index]:
                self.mp_data[i, j, :3] = rot_matrix[i, :, :] @ self.mp_data[i, j, :3]
            rot_matrix[i, :, :] = rot_matrix[i, :, :]
        node.dcm = rot_matrix

        # translation (move the rotated joint back to its place to keep the body part intact)
        translation_vector = np.zeros((self.mp_data.shape[0], 3))
        for i in range(self.mp_data.shape[0]):
            translation_vector[i, :] = previous_joint_coordinates[i, :] - self.mp_data[i, node.mp_index, :3]
            for j in self.get_all_mp_children(node.mp_index) + [node.mp_index]:
                self.mp_data[i, j, :3] += translation_vector[i, :]

    def get_joint_ref_axes(self, node) -> dict:
        if node.mp_joint_type == 'rigid':
            return self.get_rigid_body_joint_ref_axes(node)
        elif node.mp_joint_type == 'vector':
            return self.get_vector_joint_ref_axes(node)
        else:
            raise Exception('not a valid joint')

    def get_rigid_body_joint_ref_axes(self, node: BvhNode) -> dict:
        # form the axes dict
        axes_dict_primal = {}
        get_vector = lambda x: self.get_vector_2mp_nodes(self.bvh_mp_config[node.name]['ref_axes'][x]['from'],
                                                         self.bvh_mp_config[node.name]['ref_axes'][x]['to']) if \
            self.bvh_mp_config[node.name]['ref_axes'][x]['defined'] else None
        for i in ['x', 'y', 'z']:
            axes_dict_primal[i] = get_vector(i)

        # print([axes_dict_primal[i][0, :] if axes_dict_primal[i] is not None else None for i in axes_dict_primal])

        axes_dict = self.get_tri_ortho_vectors(axes_dict_primal, self.bvh_mp_config[node.name]['main_ref_axis'])
        return axes_dict

    def convert_rigid_body(self, node: BvhNode) -> bool:
        node.ref_axes = self.get_rigid_body_joint_ref_axes(node)

        # fo the rotation and record
        self.joint_fixed_rotation(node, node.ref_axes)

        print('converted bvh node:' + node.name + ', mediapipe node: ' + str(
            node.mp_index) + ', type:' + node.mp_joint_type + '\n')
        return True

    def get_vector_joint_ref_axes(self, node: BvhNode) -> dict:
        main_ref_axis = self.bvh_mp_config[node.name]['main_ref_axis']
        vec = self.get_vector_2mp_nodes(self.bvh_mp_config[node.name]['ref_axes'][main_ref_axis]['from'],
                                        self.bvh_mp_config[node.name]['ref_axes'][main_ref_axis]['to'])
        vec_ortho1 = np.zeros(vec.shape)
        for i in range(vec_ortho1.shape[0]):
            vec_ortho1[i] = self.get_random_ortho_vec(vec[i])
            vec_ortho2 = np.cross(vec, vec_ortho1)
            # it's in left-handed ref-sys, in order vec,ortho1,ortho2

        # form the axes_dict
        axes_dict = {main_ref_axis: vec}
        axes_list = ['x', 'y', 'z']
        index_restored = [axes_list.index(main_ref_axis)]
        axes_dict[axes_list[(index_restored[0] + 2) % 3 - 1]] = vec_ortho1
        index_restored.append((index_restored[0] + 2) % 3 - 1)
        axes_dict[axes_list[(index_restored[1] + 2) % 3 - 1]] = vec_ortho2

        return axes_dict

    def convert_vector(self, node: BvhNode) -> bool:

        node.ref_axes = self.get_vector_joint_ref_axes(node)

        # fo the rotation and record
        self.joint_fixed_rotation(node, node.ref_axes)

        print('converted bvh node:' + node.name + ', mediapipe node: ' + str(
            node.mp_index) + ', type:' + node.mp_joint_type + '\n')
        return True

    def check_after_rot(self, node) -> np.ndarray:
        node.ref_axes_afterwards = self.get_joint_ref_axes(node)

    def write_bvh(self, frame_rate: float, output_file: str):  # no modification on original hierarchy
        self.frame_rate = frame_rate
        with open(self.bvh_template_file, 'r', encoding='utf-8') as f:
            template = f.readlines()
        out_file_handle = open(output_file, 'w', encoding='utf-8')
        out_file_handle.writelines(template[:template.index('MOTION\n')])
        out_file_handle.writelines([
            'MOTION\n',
            'Frames: ' + str(self.mp_data.shape[0]) + '\n',
            'Frame Time: ' + str(1 / self.frame_rate) + '\n'
        ])
        for i in range(self.mp_data.shape[0]):
            tmp = ''
            tmp += '0.000000 ' * 3
            for j in self.nodes:
                if j.mp_joint_type != 'EndSite':
                    tmp += str(j.eulers[i, :])[1:-1] + ' '
            out_file_handle.write(tmp + '\n')

        out_file_handle.close()
