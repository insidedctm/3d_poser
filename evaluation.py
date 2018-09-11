import numpy as np
import json
from data_prep.matrix_utils import procrustes
import pandas as pd

json_path = 'json/H36M_annotations_testSet copy.json'
JOINTS = [0,1,6,11,2,7,12,3,8,12,5,10,16,24,24,14,25,17,26,18,27,19,30,31]

def evaluate(identifiers, preds_j_3d):
    identifiers = [identifier.split('#')[0] for identifier in identifiers]
    j_3d_gt_lookup = load_json()
    errors_3d = []
    num_records = len(identifiers)
    for identifier, pred_j_3d in zip(identifiers, preds_j_3d):
        j_3d_gt = np.array(j_3d_gt_lookup[identifier])
        j_3d_gt = j_3d_gt[JOINTS, :]
        j_3d_closest = run_procrustes(j_3d_gt, pred_j_3d)
        diff = j_3d_gt - j_3d_closest
        norm_diff = np.linalg.norm(diff, axis=1)
        avg_3d_error = np.mean(norm_diff)
        errors_3d.append(avg_3d_error)
    return np.mean(errors_3d), num_records

def evaluate_SURREAL(identifiers, preds_j_3d_gt, preds_j_3d, output_fname = 'evaluation_output.csv'):
    errors_3d = []
    num_records = len(preds_j_3d_gt)
    output = []
    for identifier, gt, pred in zip(identifiers, preds_j_3d_gt, preds_j_3d):
        j_3d_closest = run_procrustes(gt[0], pred)
        diff = gt[0] - j_3d_closest
        norm_diff = np.linalg.norm(diff, axis=1)
        avg_3d_error = np.mean(norm_diff)
        errors_3d.append(avg_3d_error)
        output.append([identifier, avg_3d_error])
    pd_out = pd.DataFrame(output)
    pd_out.to_csv(output_fname, header=None)
    return np.mean(errors_3d), num_records

def run_procrustes(gt, pred):
    _, Z, tr = procrustes(gt, pred)
    R = tr['rotation']
    t = tr['translation']
    s = tr['scale']
    return Z

def load_json():
    f = open(json_path, 'r')
    data = json.load(f)['root']
    return {d['img_paths'] : d['joint_self_3d'] for d in data}
