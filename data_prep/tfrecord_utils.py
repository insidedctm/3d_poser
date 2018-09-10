# Based on tfrecord_utils from https://github.com/htung0101/3d_smpl
# code added to convert_to_tfrecords_from_folder to deal with selecting frames
# from videos where the subject is either absent or only partially visible.
import numpy as np
import pickle
import os
import random
import sys
from write_utils import read_syn_to_bin
import struct
from tqdm import tqdm
import tensorflow as tf
import math
import scipy.misc
sys.path.append('../smpl')
from matrix_utils import get_reconstruct_3d, get_reconstruct_3d_from_angles, get_focal_length, get_smpl_transform, \
    get_intrinsic_matrix,get_smpl_transform3, get_extrinsic_matrix, project_camera_points, apply_extrinsic_matrix, \
    apply_extrinsic_matrix2, get_focal_length2, project_blender_points_using_camera_location, \
    project_camera_points_using_f

from chamfer_utils import get_chamfer
sys.path.append('.')

def get_file_list(data_path, quo=0, test=False):
    files = []
    num = 0
    for folder in tqdm(os.listdir(data_path)):
        condition = "_S9_" not in folder
        if test:
            condition = "_S9_" in folder and "1" not in folder
        if condition:
            num += 1
            p = os.path.join(data_path, folder)
            for filename in os.listdir(p):
                # if filename.startswith("h36m_S1_Directions_c0002"):
                with open(os.path.join(p, filename), 'rb') as f_:
                    line = f_.read(4)  # gender
                    line = f_.read(4)  # num_frames
                    num_frames = struct.unpack('i', line)[0]
                    # if num_frames != 100:
                    # print filename, "nframes", num_frames
                """
                if test:
                  for frame_id in range(num_frames - 1):
                    if frame_id %20 == quo:
                      #    files.append(os.path.join(p, filename) + "#" + str(frame_id))
                      #fid = np.random.randint(num_frames-1)
                      files.append(os.path.join(p, filename) + "#" + str(frame_id))
                else:
                """
                fid = np.random.randint(num_frames - 1)
                files.append(os.path.join(p, filename) + "#" + str(fid))

    print("number of folder", num)
    return files

def loadBatchSurreal_fromString(file_string, image_size=128, num_frames=2, \
                                keypoints_num=24, bases_num=10, chamfer_scale=0.5):
    filename, t = file_string.split("#")
    output = dict()
    output[0] = read_syn_to_bin(filename, int(t))
    output[1] = read_syn_to_bin(filename, int(t) + 1)

    return process_frame(output)

def process_frame(output, image_size=128, num_frames=2, \
                  keypoints_num=24, bases_num=10, chamfer_scale=0.5, is_H36M=False):
    # initialise return values
    data_J, data_J_3d_gt, data_J_2d, data_R, data_T, data_beta, data_c, data_chamfer, data_f, \
    data_gender, data_image, data_pose, data_resize_scale, data_seg = \
        initialise_process_frame( bases_num, chamfer_scale, image_size, keypoints_num, num_frames )


    J_2d = output[0]['J_2d']

    # Cropping parameters
    old_2d_center = np.array([(320 - 1) / 2.0, (240 - 1) / 2.0])
    # Use keypoint 0 in frame1 as center
    new_2d_center = np.round(J_2d[0, :] + 10 * (np.random.uniform((2)) - 1)) + 0.5 * np.ones((2))
    s = 1.2
    new_image_size, resize_scale, x_max, x_min, y_max, y_min = \
        get_crop_parameters(J_2d, image_size, new_2d_center, s)
    new_origin = np.array([x_min, y_min])

    # set frame_independent return values
    data_resize_scale[:] = resize_scale
    data_c[:, :] = np.reshape(old_2d_center - new_origin, [-1, 2])



    for frame_id in range(num_frames):
        # crop image
        image = output[frame_id]['image']
        h, w, _ = image.shape
        img_x_min = max(x_min, 0)
        img_x_max = min(x_max, w - 1)
        img_y_min = max(y_min, 0)
        img_y_max = min(y_max, h - 1)
        crop_image = np.zeros((new_image_size, new_image_size, 3), dtype=np.float32)
        crop_image[max(0, -y_min):max(0, -y_min) + img_y_max - img_y_min + 1, \
        max(0, -x_min):max(0, -x_min) + img_x_max - img_x_min + 1, :] \
            = image[img_y_min:img_y_max + 1, img_x_min:img_x_max + 1, :]
        data_image[frame_id, :, :, :] = scipy.misc.imresize(crop_image, [image_size, image_size])

        seg = get_cropped_segmentation(frame_id, image_size, img_x_max, img_x_min, img_y_max, img_y_min, new_image_size,
                                       output, x_min, y_min)


        data_seg[frame_id, :, :] = seg[:, :, 0]
        data_chamfer[frame_id, :, :], _, _ = get_chamfer(seg[:, :, 0], chamfer_scale)

    for frame_id in range(num_frames):
        fx, fy = get_focal_length2(is_H36M)

        if is_H36M: # H36M
            T = output[frame_id]['T']
            angles = np.zeros((3))
            reconstruct_3d = np.zeros(output[frame_id]['J'].T.shape)
        else:       # SURREAL
            T, angles, reconstruct_3d = get_smpl_to_camera_params(frame_id, fx, fy, output)

        # set frame-dependent return values
        data_pose[frame_id, :] = output[frame_id]['pose']
        data_gender = int(output[frame_id]['gender'])
        data_R[frame_id, :3] = np.sin(angles)
        data_R[frame_id, 3:6] = np.cos(angles)
        data_beta[frame_id, :] = output[frame_id]['beta']
        data_f[frame_id, :] = [fx, fy]
        data_T[frame_id, :] = np.reshape(T, [3])
        data_J[frame_id, :, :] = reconstruct_3d.T
        data_J_3d_gt[frame_id, :, :] = output[frame_id]['J']
        data_J_2d[frame_id, :, :] = resize_scale * (output[frame_id]['J_2d'] - np.reshape(new_origin, [1, -1]))

    return data_pose, data_T, data_R, data_beta, data_J, data_J_3d_gt, data_J_2d, data_image / 255.0, \
           data_seg, data_f, data_chamfer, data_c, data_gender, data_resize_scale


def get_smpl_to_camera_params(frame_id, fx, fy, output):
    camLoc = output[0]['camLoc']  # not frame-dependent
    J = output[frame_id]['J_transformed']
    camDist = output[frame_id]['camDist']
    d2 = output[frame_id]['J_2d']
    d3 = output[frame_id]['J']
    num_joints = 24
    # calculate ground truth rotation (angles) and translation (T) and camera parameters (fx, fy)
    d3 = apply_extrinsic_matrix2(camLoc, d3)
    swap_d2_x = False
    T, angles = get_smpl_transform(camDist, d2.T, d3.T, J.T, num_joints, fx, fy, swap_d2_x=swap_d2_x)
    T3, angles3, R3, scale = get_smpl_transform3(d3, J);
    # Use ground truth rotation/translation to convert 3d joint ground truth from SMPL to camera coords
    reconstruct_3d = get_reconstruct_3d(J.T, T3, R3)
    return T, angles, reconstruct_3d


def initialise_process_frame(bases_num, chamfer_scale, image_size, keypoints_num, num_frames):
    data_pose = np.zeros((num_frames, keypoints_num * 3))
    data_T = np.zeros((num_frames, 3))
    data_R = np.zeros((num_frames, 6))
    data_beta = np.zeros((num_frames, bases_num))
    data_J = np.zeros((num_frames, keypoints_num, 3))
    data_J_3d_gt = np.zeros((num_frames, keypoints_num, 3))
    data_J_2d = np.zeros((num_frames, keypoints_num, 2))
    data_image = np.zeros((num_frames, image_size, image_size, 3))
    data_seg = np.zeros((num_frames, image_size, image_size))
    small_image_size = int(chamfer_scale * image_size)
    data_chamfer = np.zeros((num_frames, small_image_size, small_image_size))
    data_f = np.zeros((num_frames, 2))
    data_c = np.zeros((num_frames, 2))
    data_resize_scale = np.zeros((num_frames))
    data_gender = np.zeros(())
    return data_J, data_J_3d_gt, data_J_2d, data_R, data_T, data_beta, data_c, data_chamfer, \
           data_f, data_gender, data_image, data_pose, data_resize_scale, data_seg


def get_crop_parameters(J_2d, image_size, new_2d_center, s):
    crop_size = np.round(s * np.max(np.abs(J_2d - np.reshape(new_2d_center, [1, 1, -1]))))
    new_image_size = int(2 * crop_size)
    x_min = int(math.ceil(new_2d_center[0] - crop_size))
    x_max = int(math.floor(new_2d_center[0] + crop_size))
    y_min = int(math.ceil(new_2d_center[1] - crop_size))
    y_max = int(math.floor(new_2d_center[1] + crop_size))
    resize_scale = float(image_size) / (crop_size * 2.0)
    return new_image_size, resize_scale, x_max, x_min, y_max, y_min


def visualise_projection(T, T3, angles, angles3, d2, frame_id, fx, fy, image, output, reconstruct_3d, scale):
    import matplotlib.pyplot as plt
    pose = np.array(output[frame_id]['pose'])
    beta = np.array(output[frame_id]['beta'])
    print('angles', angles)
    print('angles3', angles3)
    print('T', T)
    print('T3', T3)
    print(scale)
    ax, fig = get_axes()
    project_reconstructed = project_camera_points_using_f(reconstruct_3d, fx, fy)
    project_reconstructed = project_camera_points(reconstruct_3d)
    print(project_reconstructed)
    # project_reconstructed = project_reconstructed + np.expand_dims(np.array([160, 120]), 1)
    project_reconstructed[0, :] = 320 - project_reconstructed[0, :]
    plot_image_and_2djoints(ax, image.astype(int), project_reconstructed.T, True)
    plt.show()
    ax, fig = get_axes()
    plot_image_and_2djoints(ax, image.astype(int), d2, True)
    plt.title('gt 2d')
    plt.show()


def get_cropped_segmentation(frame_id, image_size, img_x_max, img_x_min, img_y_max, img_y_min, new_image_size, output,
                             x_min, y_min):
    seg_float = output[frame_id]['seg'].astype(np.float32)
    crop_seg = np.zeros((new_image_size, new_image_size, 3), dtype=np.float32)
    crop_seg[max(0, -y_min):max(0, -y_min) + img_y_max - img_y_min + 1, \
    max(0, -x_min):max(0, -x_min) + img_x_max - img_x_min + 1] \
        = np.expand_dims(seg_float[img_y_min:img_y_max + 1, \
                         img_x_min:img_x_max + 1], 2)
    seg = scipy.misc.imresize(crop_seg, [image_size, image_size])
    seg[seg < 0.5] = 0
    seg[seg >= 0.5] = 1
    return seg


def visualise_projection_post_cropping(data_J_2d, data_image, frame_id, new_origin, reconstruct_3d, resize_scale):
    import matplotlib.pyplot as plt
    ax, fig = get_axes()
    project_reconstructed = project_camera_points(reconstruct_3d)
    project_reconstructed[0, :] = 320 - project_reconstructed[0, :]
    project_reconstructed -= np.array(np.expand_dims(new_origin, 1))
    print(project_reconstructed)
    print(resize_scale)
    vis_img = data_image[frame_id, :, :, :] * resize_scale
    plot_image_and_2djoints(ax, vis_img.astype(int), project_reconstructed.T, True)
    plt.show()
    ax, fig = get_axes()
    plot_image_and_2djoints(ax, vis_img.astype(int), data_J_2d[frame_id, :, :], True)
    plt.title('gt 2d')
    plt.show()


def _floatList_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _intList_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytesList_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def convert_to_tfrecords_from_folder(folder_name, tf_filename, get_samples=None, test=False, quo=0, with_idx=False,
                                     shuffle=True, visualise_setting=False):
    global visualise_
    visualise_ = visualise_setting
    files = get_file_list(folder_name, quo, test=test)
    print('files', files)
    if shuffle:
        random.shuffle(files)
    num_files = len(files)
    num_frames = 2
    crop_image_size = 128
    keypoints_num = 24
    bases_num = 10
    if not get_samples:
        get_samples = num_files
    print("total samples", get_samples)

    writer = tf.python_io.TFRecordWriter(tf_filename)
    for sample_id in range(get_samples):
        try:
            pose, T, R, beta, J, J_3d_gt, J_2d, image, seg, f, chamfer, c, gender, resize_scale = \
                loadBatchSurreal_fromString(files[sample_id], crop_image_size, num_frames)
            ident = files[sample_id]
            example = get_tfrecord_example(
                pose, T, R, beta, J, J_3d_gt, J_2d, image, seg, f, chamfer, c, gender, resize_scale, with_idx, sample_id, ident)
            writer.write(example.SerializeToString())
        except Exception as e:
            import traceback
            traceback.print_exc() 
            print('skipping record',e)
    writer.close()

def convert_to_tfrecords(data, tf_filename, get_samples=None, test=False, quo=0, with_idx=False,
                                     shuffle=True):
    num_frames = 2
    crop_image_size = 128
    keypoints_num = 24
    bases_num = 10
    num_records = data['pose'].shape[0]

    print('converting {} records to tfrecords format'.format(num_records))
    writer = tf.python_io.TFRecordWriter(tf_filename)
    for sample_id in range(num_records):
        try:
            output = dict()
            output[0] = get_frame_data(data, sample_id)
            output[1] = get_frame_data(data, sample_id) # repeated to match the data format used for Surreal
            pose, T, R, beta, J, J_3d_gt, J_2d, image, seg, f, chamfer, c, gender, resize_scale = \
                process_frame(output, is_H36M=True)
            ident = data['identifier'][sample_id]
            
            example = get_tfrecord_example(
                pose, T, R, beta, J, J_3d_gt, J_2d, image, seg, f, chamfer, c, gender,
                resize_scale, with_idx, sample_id, ident)
            writer.write(example.SerializeToString())
        except Exception as e:
            import traceback
            traceback.print_exc() 
            print('skipping record',e)
    writer.close()

def get_frame_data(data, id):
    output = dict()
    output['gender']        = data['gender']
    output['camLoc']        = np.array([0., 0., 0.])
    output['camDist']       = 0
    output['beta']          = data['beta'][id]
    output['pose']          = data['pose'][id]
    output['f']             = data['f'][id]
    output['R']             = data['R'][id]
    output['T']             = data['T'][id]
    output['J']             = data['J'][id]
    output['J_transformed'] = data['J_transformed'][id]
    output['J_2d']          = data['J_2d'][id]
    output['image']         = data['image'][id]
    output['seg']           = data['seg'][id]
    output['identifier']    = data['identifier'][id]
    return output 

def get_tfrecord_example(pose, T, R, beta, J, J_3d_gt, J_2d, image, seg, f, chamfer, c, gender, \
                         resize_scale, with_idx, sample_id, ident='empty') :
    if with_idx:
                example = tf.train.Example(features=tf.train.Features(feature={
                    'pose': _floatList_feature(pose.flatten()),
                    'beta': _floatList_feature(beta.flatten()),
                    'T': _floatList_feature(T.flatten()),
                    'R': _floatList_feature(R.flatten()),
                    'J': _floatList_feature(J.flatten()),
                    'J_3d_gt': _floatList_feature(J_3d_gt.flatten()),
                    'J_2d': _floatList_feature(J_2d.flatten()),
                    'image': _floatList_feature(image.flatten()),
                    'seg': _floatList_feature(seg.flatten()),
                    'f': _floatList_feature(f.flatten()),
                    'chamfer': _floatList_feature(chamfer.flatten()),
                    'c': _floatList_feature(c.flatten()),
                    'resize_scale': _floatList_feature(resize_scale.flatten()),
                    'gender': _intList_feature([gender]),
                    'identifier': _bytesList_feature([ident]),
                    'idx': _intList_feature([sample_id])}))
    else:
                example = tf.train.Example(features=tf.train.Features(feature={
                    'pose': _floatList_feature(pose.flatten()),
                    'beta': _floatList_feature(beta.flatten()),
                    'T': _floatList_feature(T.flatten()),
                    'R': _floatList_feature(R.flatten()),
                    'J': _floatList_feature(J.flatten()),
                    'J_3d_gt': _floatList_feature(J_3d_gt.flatten()),
                    'J_2d': _floatList_feature(J_2d.flatten()),
                    'image': _floatList_feature(image.flatten()),
                    'seg': _floatList_feature(seg.flatten()),
                    'f': _floatList_feature(f.flatten()),
                    'chamfer': _floatList_feature(chamfer.flatten()),
                    'c': _floatList_feature(c.flatten()),
                    'resize_scale': _floatList_feature(resize_scale.flatten()),
                    'gender': _intList_feature([gender]),
                    'identifier': _bytesList_feature([ident])}))
    return example

def read_and_decode_surreal(tfrecord_file, data_version=3):
    num_frames = 2
    image_size = 128
    keypoints_num = 24
    bases_num = 10
    chamfer_scale = 0.5
    small_image_size = int(chamfer_scale * image_size)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(tfrecord_file)
    features = {
        'pose': tf.FixedLenFeature([num_frames * keypoints_num * 3], tf.float32),
        'beta': tf.FixedLenFeature([num_frames * bases_num], tf.float32),
        'T': tf.FixedLenFeature([num_frames * 3], tf.float32),
        'R': tf.FixedLenFeature([num_frames * 6], tf.float32),
        'J': tf.FixedLenFeature([num_frames * keypoints_num * 3], tf.float32),
        'J_2d': tf.FixedLenFeature([num_frames * keypoints_num * 2], tf.float32),
        'image': tf.FixedLenFeature([num_frames * image_size * image_size * 3], tf.float32),
        'seg': tf.FixedLenFeature([num_frames * image_size * image_size], tf.float32),
        'f': tf.FixedLenFeature([num_frames * 2], tf.float32),
        'chamfer': tf.FixedLenFeature([num_frames * small_image_size * small_image_size], tf.float32),
        'c': tf.FixedLenFeature([num_frames * 2], tf.float32),
        'resize_scale': tf.FixedLenFeature([num_frames], tf.float32),
        'gender': tf.FixedLenFeature([], tf.int64)
    }

    if data_version >= 2:
        features['identifier'] = tf.FixedLenFeature([], tf.string)
    if data_version >= 3:
        features['J_3d_gt'] = tf.FixedLenFeature([num_frames * keypoints_num * 3], tf.float32)

    feature = tf.parse_single_example(
        serialized_example,
        features=features)
    feature['pose'] = tf.reshape(feature['pose'], [num_frames, keypoints_num, 3])
    feature['beta'] = tf.reshape(feature['beta'], [num_frames, bases_num])
    feature['T'] = tf.reshape(feature['T'], [num_frames, 3])
    feature['R'] = tf.reshape(feature['R'], [num_frames, 6])
    feature['J'] = tf.reshape(feature['J'], [num_frames, keypoints_num, 3])
    feature['J_2d'] = tf.reshape(feature['J_2d'], [num_frames, keypoints_num, 2])
    feature['image'] = tf.reshape(feature['image'], [num_frames, image_size, image_size, 3])
    feature['seg'] = tf.reshape(feature['seg'], [num_frames, image_size, image_size])
    feature['chamfer'] = tf.reshape(feature['chamfer'], [num_frames, small_image_size, small_image_size])
    feature['c'] = tf.reshape(feature['c'], [num_frames, 2])
    feature['f'] = tf.reshape(feature['f'], [num_frames, 2])
    feature['resize_scale'] = tf.reshape(feature['resize_scale'], [num_frames])
    if data_version < 2:
        feature['identifier'] = tf.constant(["Empty"])
    if data_version < 3:
        feature['J_3d_gt'] = tf.reshape(feature['J'], [num_frames, keypoints_num, 3])

    return feature['pose'], feature['beta'], feature['T'], feature['R'], feature['J'], feature['J_3d_gt'], \
           feature['J_2d'], feature['image'], feature['seg'], feature['chamfer'], feature['c'], \
           feature['f'], feature['resize_scale'], feature['gender'], feature['identifier']


def read_and_decode_surreal_with_idx(tfrecord_file, data_version=3):
    num_frames = 2
    image_size = 128
    keypoints_num = 24
    bases_num = 10
    chamfer_scale = 0.5
    small_image_size = int(chamfer_scale * image_size)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(tfrecord_file)

    features = {
        'pose': tf.FixedLenFeature([num_frames * keypoints_num * 3], tf.float32),
        'beta': tf.FixedLenFeature([num_frames * bases_num], tf.float32),
        'T': tf.FixedLenFeature([num_frames * 3], tf.float32),
        'R': tf.FixedLenFeature([num_frames * 6], tf.float32),
        'J': tf.FixedLenFeature([num_frames * keypoints_num * 3], tf.float32),
        'J_2d': tf.FixedLenFeature([num_frames * keypoints_num * 2], tf.float32),
        'image': tf.FixedLenFeature([num_frames * image_size * image_size * 3], tf.float32),
        'seg': tf.FixedLenFeature([num_frames * image_size * image_size], tf.float32),
        'f': tf.FixedLenFeature([num_frames * 2], tf.float32),
        'chamfer': tf.FixedLenFeature([num_frames * small_image_size * small_image_size], tf.float32),
        'c': tf.FixedLenFeature([num_frames * 2], tf.float32),
        'resize_scale': tf.FixedLenFeature([num_frames], tf.float32),
        'gender': tf.FixedLenFeature([], tf.int64),
        'idx': tf.FixedLenFeature([], tf.int64),
    }

    if data_version >= 2:
        features['identifier'] = tf.FixedLenFeature([], tf.string)
    if data_version >= 3:
        features['J_3d_gt'] = tf.FixedLenFeature([num_frames * keypoints_num * 3], tf.float32)

    feature = tf.parse_single_example(
        serialized_example,
        features=features)
    feature['pose'] = tf.reshape(feature['pose'], [num_frames, keypoints_num, 3])
    feature['beta'] = tf.reshape(feature['beta'], [num_frames, bases_num])
    feature['T'] = tf.reshape(feature['T'], [num_frames, 3])
    feature['R'] = tf.reshape(feature['R'], [num_frames, 6])
    feature['J'] = tf.reshape(feature['J'], [num_frames, keypoints_num, 3])
    feature['J_2d'] = tf.reshape(feature['J_2d'], [num_frames, keypoints_num, 2])
    feature['image'] = tf.reshape(feature['image'], [num_frames, image_size, image_size, 3])
    feature['seg'] = tf.reshape(feature['seg'], [num_frames, image_size, image_size])
    feature['chamfer'] = tf.reshape(feature['chamfer'], [num_frames, small_image_size, small_image_size])
    feature['c'] = tf.reshape(feature['c'], [num_frames, 2])
    feature['f'] = tf.reshape(feature['f'], [num_frames, 2])
    feature['resize_scale'] = tf.reshape(feature['resize_scale'], [num_frames])

    if data_version < 2:
        feature['identifier'] = tf.constant(["Empty"])
    if data_version < 3:
        feature['J_3d_gt'] = tf.reshape(feature['J'], [num_frames, keypoints_num, 3])

    return feature['pose'], feature['beta'], feature['T'], feature['R'], feature['J'], feature['J_3d_gt'], \
           feature['J_2d'], feature['image'], feature['seg'], feature['chamfer'], feature['c'], \
           feature['f'], feature['resize_scale'], feature['gender'], \
           feature['identifier'], feature['idx']


def inputs_surreal(tf_filenames, batch_size, data_version=3):
    print('inputs_surreal')
    with tf.name_scope('surreal_input'):
        filename_queue = tf.train.string_input_producer(tf_filenames)
        pose, beta, T, R, J, J_3d_gt, J_2d, image, seg, chamfer, c, f, resize_scale, \
            gender, identifier = read_and_decode_surreal(
            filename_queue, data_version=data_version)

        return tf.train.shuffle_batch([pose, beta, T, R, J, J_3d_gt, J_2d, image, seg, chamfer, c, f,
                                       resize_scale, gender, identifier],
                                      batch_size=batch_size,
                                      num_threads=2, capacity=5000, min_after_dequeue=2000)


def inputs_surreal_with_idx(tf_filenames, batch_size, shuffle=True, num_epochs=None, data_version=3):
    print('inputs_surreal_with_idx')
    with tf.name_scope('surreal_input'):
        filename_queue = tf.train.string_input_producer(tf_filenames, shuffle=shuffle, num_epochs=num_epochs)
        pose, beta, T, R, J, J_3d_gt, J_2d, image, seg, chamfer, c, f, resize_scale, \
               gender, identifier, idx = read_and_decode_surreal_with_idx(
            filename_queue, data_version=data_version)

        if not shuffle:
            return tf.train.batch([pose, beta, T, R, J, J_3d_gt, J_2d, image, seg, chamfer, c, f,
                                   resize_scale, gender, identifier, idx],
                                  batch_size=batch_size,
                                  num_threads=2, allow_smaller_final_batch=False)  # ,capacity=80,min_after_dequeue=50)

        else:
            return tf.train.shuffle_batch([pose, beta, T, R, J, J_3d_gt, J_2d, image, seg, chamfer, c, f,
                                           resize_scale, gender, identifier, idx],
                                          batch_size=batch_size,
                                          num_threads=2, capacity=80, min_after_dequeue=50)

