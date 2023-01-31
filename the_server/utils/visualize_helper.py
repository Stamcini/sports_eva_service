import mediapipe as mp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils.mp_to_bvh_solution import BvhSolution

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


def pairwise_coors(data: np.ndarray, pairs: [(int, int), ...]) -> np.ndarray:
    # data: [lm_index,(x,y,z,visibility)]
    out = np.zeros((2, len(pairs), 3))
    for i, pair in enumerate(pairs):
        out[:, i, :] = np.stack((data[pair[0], :3], data[pair[1], :3]))
    return out


def draw_skeleton_init(ax: plt.axes, data: np.ndarray, pairs: [(int, int), ...]) -> list:
    init_coors = pairwise_coors(data[0, :, :], pairs)
    lines = [ax.plot3D([], [], [], color='red', alpha=0.6)[0] for i in range(len(pairs))]
    [line.set_data_3d(tuple([init_coors[:, i, j] for j in (0, 2, 1)])) for i, line in enumerate(lines)]
    return lines


def anim_update(num: int, scatter, lines: list, data: np.ndarray, pairs: [(int, int), ...]):
    # update the scatter points
    scatter._offsets3d = [data[num, :, i] for i in (0, 2, 1)]

    # update the skeleton connections
    pairwise_coor_step = pairwise_coors(data[num, :, :], pairs)
    [line.set_data_3d(tuple([pairwise_coor_step[:, i, j] for j in (0, 2, 1)])) for i, line in enumerate(lines)]

    return scatter, lines


def fig_init() -> []:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.view_init(elev=-165, azim=65)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    return [fig, ax]


def anim_matchstickman(data: np.ndarray, frame_rate: float, *, repeat: bool = True,
                       show: bool = True) -> mpl.animation.FuncAnimation:
    # data is pose_3d_lms_np like, with shape (n,33,3)
    # frame_rate is
    f_pose, ax_pose = fig_init()
    if not show:
        plt.close(f_pose)

    pose_init = data[0, :, :]
    sc_pose = ax_pose.scatter3D(pose_init[:, 0], pose_init[:, 2], pose_init[:, 1])

    lines_pose = draw_skeleton_init(ax_pose, data, mp_holistic.POSE_CONNECTIONS)

    num_steps_pose = data.shape[0]
    pose_ani = animation.FuncAnimation(f_pose, anim_update, num_steps_pose,
                                       fargs=[sc_pose, lines_pose, data, mp_holistic.POSE_CONNECTIONS],
                                       interval=1000 / frame_rate, repeat=repeat)
    return pose_ani


def anim_bvh_armature(data: np.ndarray, frame_rate: float, bvh: BvhSolution, *, repeat: bool = True):
    fig, ax = fig_init()

    pose_init = data[0, :, :]
    scatter = ax.scatter3D(pose_init[:, 0], pose_init[:, 2], pose_init[:, 1])

    pairs = []
    for i, node in enumerate(bvh.nodes):
        if i == 0:
            continue
        else:
            pass
        pairs.append((node.index, bvh.get_node_by_name(node.parent).index))

    lines = draw_skeleton_init(ax, data, pairs)

    anim = mpl.animation.FuncAnimation(fig, anim_update, data.shape[0], fargs=[scatter, lines, data, pairs],
                                       interval=1000 / frame_rate, repeat=repeat)

    return anim
