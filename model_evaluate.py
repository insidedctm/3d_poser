import os
from bn_model import _3DINNBatchNormalisation
from dropout_model import _3DINNDropout
import tensorflow as tf
import glob
import evaluation

"""
Define flags
"""
flags = tf.app.flags
flags.DEFINE_integer("gpu", 1, "gpu_id")
flags.DEFINE_string("name", "from_scratch", "name of this version")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate")
flags.DEFINE_float("init", 0.01, "std of param init") #? 
flags.DEFINE_integer("max_iter", 20, "Iterations times")
flags.DEFINE_integer("batch_size", 16, "The size of batch images")
flags.DEFINE_integer("num_frames", 2, "The size of batch images")
flags.DEFINE_integer("gf_dim", 32, "The size of batch images")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("alpha", 0.5, "Weights for silhouette loss S_M * C_I[0.5]")
flags.DEFINE_float("sr", 1.0, "sampling rate for visibility")
flags.DEFINE_integer("image_size_h", 128, "")
flags.DEFINE_integer("image_size_w", 128, "")
flags.DEFINE_integer("keypoints_num", 24, "number of keypoints") 
flags.DEFINE_integer("mesh_num", 6890, "number of keypoints") 
flags.DEFINE_integer("bases_num", 10, "number of Base Shapes") # 1 (mean)+ 96

flags.DEFINE_integer("gWidth", 7, "width of Gaussian kernels")
flags.DEFINE_float("gStddev", 0.25, "std for 2d Gaussian heatmaps")
flags.DEFINE_string("data_dir", "../src/output", "Directory name to save the preprocessed data [data]")
flags.DEFINE_string("checkpoint_dir", "checkpoint/", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_integer("checkpoint_frequency", 10000, "Frequency (in terms of iterations) to checkpoint")
flags.DEFINE_integer("summary_frequency", 100, "Frequency (in terms of iterations) to output summary")
flags.DEFINE_integer("sample_frequency", 1000, "Frequency (in terms of iterations) to dump samples to a .mat file")
flags.DEFINE_boolean("sample", False, "Whether to output samples to .mat files")
flags.DEFINE_string("model_dir", None , "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples/", "Directory name to save the image samples [samples]")
flags.DEFINE_string("logs_dir", "logs/", "Directory name to save logs [logs]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_boolean("is_sup_train", True, "True for supervised training on training data,"
                                           "False for unsupervised training [True]")
flags.DEFINE_boolean("is_dryrun", False, "Run one batch to see whether visibility is correct"
                                         "You need to reduce batch size to fit memory [False]")
#flags.DEFINE_boolean("sup_loss", True, "True for using supervised loss")
flags.DEFINE_boolean("key_loss", False, "True for using unsupervised keypoint loss")
flags.DEFINE_integer("data_version", 2, "1=original 3d_smpl format, 2=include identifier, 3=J_3d_gt" )
flags.DEFINE_string("data_name", "H36M", "Either H36M or SURREAL to indicate the data to be evaluated")
flags.DEFINE_string("evaluation_output", "model_evaluation.csv", "")
flags.DEFINE_float("keep_prob", 0.8, "Dropout model dropout (= 1-keep_prob) probability")
flags.DEFINE_string("model_type", "BatchNormalisation", "Either BatchNormalisation or Dropout")
FLAGS = flags.FLAGS

def main(_):
    checkpoint_dir_ = os.path.join(FLAGS.checkpoint_dir, FLAGS.name)
    sample_dir_ = os.path.join(FLAGS.sample_dir, FLAGS.name)
    logs_dir_ = os.path.join(FLAGS.logs_dir, FLAGS.name)
    if not os.path.exists(checkpoint_dir_):
        os.makedirs(checkpoint_dir_)
    if not os.path.exists(sample_dir_):
        os.makedirs(sample_dir_)
    if not os.path.exists(logs_dir_):
        os.makedirs(logs_dir_)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        if FLAGS.model_type == "BatchNormalisation":
          m = _3DINNBatchNormalisation(sess, config=FLAGS, checkpoint_dir=checkpoint_dir_,
                         logs_dir=logs_dir_, sample_dir=sample_dir_)
        else:
          m = _3DINNDropout(sess, config=FLAGS, checkpoint_dir=checkpoint_dir_,
                         logs_dir=logs_dir_, sample_dir=sample_dir_)

        test_filenames = glob.glob('data/test/*.tfrecords')
        print(test_filenames)
        _, _, _, _, m.J_sr_v, m.J_2d_sr_v, m.image_sr_v, _, \
        _, m.c_sr_v, m.f_sr_v, m.resize_scale_sr_v, m.gender_sr_v, m.identifier_sr_v, \
        _, m.idx_sr_v, _, _ , m.J_3d_orig_sr_v = \
            m.get_data(test_filenames, with_idx=True, num_epochs=1, shuffle=False)

        print("### Evaluate  ####")
        preds_identifier, preds_j_3d_gt, preds_j_3d, preds_pose, preds_beta = m.test(FLAGS)

        if FLAGS.data_name == 'H36M':
            avg_mpjpe, num_records = evaluation.evaluate(preds_identifier, preds_j_3d)
        else:
            avg_mpjpe, num_records = evaluation.evaluate_SURREAL(preds_identifier, preds_j_3d_gt, preds_j_3d, \
                                        FLAGS.evaluation_output)             
        print(test_filenames)
        print('Average MPJPE={}; {} records'.format(avg_mpjpe, num_records))

if __name__ == '__main__':
	tf.app.run()
