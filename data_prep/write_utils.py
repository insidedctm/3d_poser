# Based on write_utils from https://github.com/htung0101/3d_smpl

import numpy as np
import struct
w = 320
h = 240
def write_syn_to_bin(parsed_data, filename):
  # gender: 1 
  # beta: 100 x 10
  # pose: 100 x 72
  # f : 100 x2
  # R : 100 x 3
  # T : 100 x 3
  # J : 100x24x3
  # J_2d : 100 x 24 x2
  # image: 100 x 24 x 320 x 3 # np.unit8
  # seg: 100 x 240 x 320 #bool
  num_frames = parsed_data['pose'].shape[0]
  # gender[int32], num_frames[int32]
  with open(filename, "wb") as f_:
    f_.write(struct.pack('i', parsed_data['gender'])) 
    f_.write(struct.pack('i', num_frames))
    f_.write(struct.pack('f' * 3, *parsed_data['camLoc']))
    for frame_id in range(num_frames):
      beta = list(parsed_data['beta'][frame_id, :])
      pose = list(parsed_data['pose'][frame_id, :])
      camDist = list(parsed_data['camDist'][frame_id, :])
      J = list(np.reshape(parsed_data['J'][frame_id, :, :], [-1]))
      J_trans = list(np.reshape(parsed_data['J_transformed'][frame_id, :, :], [-1]))
      J_2d = list(np.reshape(parsed_data['J_2d'][frame_id, :, :], [-1]))
      image = list(np.reshape(parsed_data['image'][frame_id, :, :, :].astype(np.float32), [-1]))
      params = beta + pose + J + J_trans + J_2d + camDist + image
      num_elements = len(params)
      f_.write(struct.pack('f' * num_elements, *params))
      seg = list(np.reshape(parsed_data['seg'][frame_id, :, :], [-1])) 
      f_.write(struct.pack('?' * h * w, *seg))
       

def read_syn_to_bin(filename, frame_id):
  
  with open(filename, 'rb') as f_:
    line = f_.read(4)
    gender = struct.unpack('i', line)[0] 
    line = f_.read(4)
    num_frames = struct.unpack('i', line)[0]
    line = f_.read(12)
    camLoc = struct.unpack('f' * 3, line)
    num_elements_in_line = 10 + 72 + 24 * 3 + 24 * 3 + 24 * 2 + 1 + h * w * 3
    # get to the head of requested frame
    _ = f_.read((4 * (num_elements_in_line) + h * w) * frame_id)
    line = f_.read(4 * num_elements_in_line)
    params = struct.unpack('f' * num_elements_in_line, line) 
    line = f_.read(1 * h * w)
    seg = struct.unpack('?' * h * w, line) 
    output = dict()
    output['gender'] = gender
    output['beta'] = params[:10]
    output['pose'] = params[10: 82]
    output['J'] = np.reshape(params[82:82 + 72], [24, 3])
    output['J_transformed'] = np.reshape(params[154:154 + 72], [24, 3])
    output['J_2d'] = np.reshape(params[226:226 + 48], [24, 2])
    output['camDist'] = params[274]
    output['image'] = np.reshape(params[275:275 + h * w * 3], [h, w, 3])
    output['seg'] = np.reshape(seg, [h, w])
    output['camLoc'] = camLoc
    return output
