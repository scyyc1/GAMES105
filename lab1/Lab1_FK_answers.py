import numpy as np
from scipy.spatial.transform import Rotation as R
from task1_forward_kinematics import part1

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)

    return motion_data



def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_name = []
    joint_parent = []
    joint_stack_list = []
    offset_list = []
    my_joint_dict = {}

    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line = [name for name in lines[i].split()]
            next_line = [name for name in lines[i+1].split()]
            if line[0] == "HIERARCHY":
                continue
            if line[0] == "MOTION":
                break
            if line[0] == "ROOT" or line[0] == "JOINT":
                joint_name.append(line[-1])
                joint_stack_list.append(line[-1])
            if line[0] == "End":
                joint_name.append(joint_name[-1]+"_end")
                joint_stack_list.append(joint_name[-1])
            if line[0] == "OFFSET":
                offset_list.append([float(line[1]), float(line[2]), float(line[3])])
            if line[0] == "}":
                joint_index = joint_stack_list.pop()
                if joint_stack_list == []:
                    continue
                else:
                    my_joint_dict[joint_index] = joint_stack_list[-1]
        for i in joint_name:
            if i == "RootJoint":
                joint_parent.append(-1)
            else:
                joint_parent_name = my_joint_dict[i]
                joint_parent.append(joint_name.index(joint_parent_name))
        joint_offset = np.array(offset_list).reshape(-1, 3)
    # print(joint_offset, type(joint_offset), joint_offset.shape)
    # print(joint_parent)
    # exit()
    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    joint_positions = []
    joint_orientations = []
    end_index = []

    for i in joint_name:
        if "_end" in i:
            end_index.append(joint_name.index(i))
    frame_data = motion_data[frame_id]
    frame_data = frame_data.reshape(-1, 3)
    quaternion = R.from_euler('XYZ', frame_data[1:], degrees=True).as_quat()
    for i in end_index:
        quaternion = np.insert(quaternion, i, [0,0,0,1], axis=0)



    for index, parent in enumerate(joint_parent):
        if parent==-1:
            joint_positions.append(frame_data[0])
            joint_orientations.append(quaternion[0])
        else:
            quat = R.from_quat(quaternion)
            rotation = R.as_quat(quat[index] * quat[parent])
            joint_orientations.append(rotation)
            joint_orientations_quat = R.from_quat(joint_orientations)

            offset_rotation = joint_orientations_quat[parent].apply(joint_offset[index])
            joint_positions.append(joint_positions[parent] + offset_rotation)

    joint_positions = np.array(joint_positions)
    joint_orientations = np.array(joint_orientations)

    # exit()

    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """

    motion_dict = {}
    end_index_A = []
    joint_remove_A = []
    joint_remove_T = []



    joint_name_T, joint_parent_T, joint_offset_T = part1_calculate_T_pose(T_pose_bvh_path)
    joint_name_A, joint_parent_A, joint_offset_A = part1_calculate_T_pose(A_pose_bvh_path)
    motion_data_A = load_motion_data(A_pose_bvh_path)
    motion_shape_A = motion_data_A.shape

    root_position = motion_data_A[:, :3]
    motion_data_A = motion_data_A[:, 3:]
    motion_data = np.zeros(motion_data_A.shape)
    print(root_position.shape)
    print((motion_data_A[1]).reshape(-1,3))
    # exit()

    for i in joint_name_A:
        if "_end" not in i:
            joint_remove_A.append(i)

    for i in joint_name_T:
        if "_end" not in i:
            joint_remove_T.append(i)


    for index, name in enumerate(joint_remove_A):
        motion_dict[name] = motion_data_A[:, 3*index:3*(index+1)]

    # print(motion_dict)
    # exit()
    for index, name in enumerate(joint_remove_T):
        if name == "lShoulder":
            motion_dict[name][:, 2] -= 45
        elif name == "rShoulder":
            motion_dict[name][:, 2] += 45
        motion_data[:, 3*index:3*(index+1)] = motion_dict[name]
    # print(motion_dict)

    motion_data = np.concatenate([root_position, motion_data], axis=1)
    # print((motion_data[0]).reshape(-1,3))
    print(motion_data[0])
    # exit()
    return motion_data
