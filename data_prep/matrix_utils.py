# Based on matrix_utils from https://github.com/htung0101/3d_smpl
# routine added for procrustes analysis and working with and manipulating
# SURREAL and H36M data
from smpl_webuser.serialization import load_model
import numpy as np
from scipy.linalg import orthogonal_procrustes
import inspect
import math
## Assign random pose and shape parameter
from numpy.linalg import norm
import math

def avg_joint_error(m):
  return np.mean(np.sqrt(np.sum(np.square(m),0)))

def get_valid_rotation(R):
  if isRotationMatrix(R):
    return R, 0
  U, s, V  = np.linalg.svd(R)
  new_R = np.matmul(U, V)
  if not abs(np.linalg.det(new_R) - 1) < 0.01:
    end = 2
    U[end, :] = (-1) * U[end, :]
    new_R = np.matmul(U, V)

  return new_R, np.linalg.norm(new_R - R) 

def isRotationMatrix(R):
  Rt = np.transpose(R)
  shouldBeIdentity = np.dot(Rt, R)
  I = np.identity(3, dtype = R.dtype)
  n = np.linalg.norm(I - shouldBeIdentity)
  return n < 1e-6

def rotationMatrixToEulerAngles(R, print_error=False) :
  R, recovered_error = get_valid_rotation(R)
  assert(isRotationMatrix(R))
  
  sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
  singular = sy < 1e-6

  if  not singular :
    x = math.atan2(R[2,1] , R[2,2])
    y = math.atan2(-R[2,0], sy)
    z = math.atan2(R[1,0], R[0,0])
  else:
    x = math.atan2(-R[1,2], R[1,1])
    y = math.atan2(-R[2,0], sy)
    z = 0
  if print_error:
    return np.array([x, y, z]), recovered_error
  else:
    return np.array([x, y, z])



def eulerAnglesToRotationMatrix(theta) :
  R_x = np.array([[1,         0,                  0                   ],
                  [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                  [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                   ])
  R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                  [0,                     1,      0                   ],
                  [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                  ])
  R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
    [math.sin(theta[2]),    math.cos(theta[2]),     0],
    [0,                     0,                      1]
    ])
  R = np.dot(R_z, np.dot( R_y, R_x ))
  return R

def get_reconstruct_3d_from_angles(J, T, angles):
    R_rec = eulerAnglesToRotationMatrix(angles)
    return get_reconstruct_3d(J, T, R_rec)

def get_reconstruct_3d(J, T, R):
    reconstruct_3d = np.matmul(R, J) + np.reshape(T, [3, 1])
    return reconstruct_3d


def get_smpl_transform(camDist, d2, d3, J, num_joints, fx, fy, swap_d2_x=False):

    if swap_d2_x:
        d2[0, :] = (320 - d2[0, :])

    # visualize_smpl_2d(d3_project, xlim=(-160, 160), ylim=(-120, 120), title="projected 3d", figure_id=3)
    # print "piexl error", avg_joint_error(d3_project - centered_2d)
    T2 = np.zeros((3, 1))
    T2[0] = (d2[0, 0] - 160) / fx * d3[2, 0]
    T2[1] = (d2[1, 0] - 120) / fy * d3[2, 1]
    # print fx, fy
    # smpl to 3d center
    J_ones = np.concatenate((J, np.ones((1, num_joints))), 0)
    R_T = np.linalg.lstsq(J_ones.T, d3.T)[0].T
    R = R_T[:3, :3]
    T = np.reshape(R_T[:, 3], [3, 1])

    #T += T2
    angles = rotationMatrixToEulerAngles(R)
    # print angles

    return T, angles

def get_smpl_transform2(cam, smpl):
    # Work out translation (T) and rotation (R) between from SMPL to camera coordinate systems
    # INPUTS:
    #   d3 - set of points in camera coordinates (24 x 3)
    #   J  - set of points in SMPL coordinates (24 x 3)

    # use root node (pelvis) to get translation from SMPL to camera
    T = cam[0,:] - smpl[0,:]

    # use centroid of points to get translation from SMPL to camera
    T_alt = np.mean(cam, axis=0) - np.mean(smpl, axis=0)

    # translate SMPL to camera
    smpl_trans = smpl + T_alt
    print('T', T)
    print('T_alt', T_alt)
    print('cam', cam)
    print('smpl_trans', smpl_trans)

    R = procrustes(smpl_trans, cam)
    angles = rotationMatrixToEulerAngles(R)

    return T_alt, angles

def get_smpl_transform3(cam, smpl):
    # Work out translation (T) and rotation (R) between from SMPL to camera coordinate systems
    # INPUTS:
    #   d3 - set of points in camera coordinates (24 x 3)
    #   J  - set of points in SMPL coordinates (24 x 3)

    _, _, tr = procrustes(cam, smpl)
    T = tr['translation']
    R = tr['rotation']
    s = tr['scale']

    angles = rotationMatrixToEulerAngles(R)

    return T, angles, R, s


def procrustes(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    #transformation values
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform



def get_focal_length(d2, d3, swap_d2_x=False):
    if swap_d2_x:
        d2[0, :] = (320 - d2[0, :])
    d3_project = d3[:2, :] / (d3[2, :])
    # center 2d ground truth
    centered_2d = d2 - np.reshape(d2[:, 0], [2, 1])
    sol = np.linalg.lstsq(np.reshape(d3_project[0, :], [-1, 1]), np.reshape(centered_2d[0, :], [-1, 1]))[0]
    fx = sol[0][0]
    sol = np.linalg.lstsq(np.reshape(d3_project[1, :], [-1, 1]), np.reshape(centered_2d[1, :], [-1, 1]))[0]
    fy = sol[0][0]  # np.sum(centered_2d[1,:])/np.sum(d3_project[1,:])
    return fx, fy

def get_focal_length2(identifier='', is_H36M=False):
    if is_H36M:
        return get_focal_length_h36m(identifier)
    else:
        return get_focal_length_surreal()

def get_focal_length_h36m(identifier):
    print(identifier)
    return 0., 0.

def get_focal_length_surreal():
    intrinsic_matrix = get_intrinsic_matrix()
    fx = intrinsic_matrix[0,0]
    fy = intrinsic_matrix[1,1]
    return fx, fy

def apply_extrinsic_matrix(camDist, d3):
    d3 = d3.T
    matrix_world = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    d3 = np.matmul(matrix_world, d3) + np.reshape(np.array([0, 1, -(camDist - 3)]), [3, 1])
    r_camera = [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]
    d3 = np.matmul(r_camera, d3)
    d3_tmp = d3
    new_center = np.zeros((3, 1))
    new_center[:2] = np.expand_dims(d3[:2, 0], 1)
    d3 = d3 - new_center
    angle_x = -0.00102
    R_x = [[1, 0, 0],
           [0, math.cos(angle_x), -math.sin(angle_x)],
           [0, math.sin(angle_x), math.cos(angle_x)]]
    d3 = np.matmul(R_x, d3)
    return d3.T

def apply_extrinsic_matrix2(camLoc, d3):
    d3 = d3.T
    extrinsic_matrix = get_extrinsic_matrix(np.reshape(camLoc,[3,1]))
    num_points = d3.shape[1]
    pts_3d_homogeneous = np.concatenate([d3, np.ones((1, num_points))], axis=0)
    d3 = np.matmul(extrinsic_matrix, pts_3d_homogeneous)
    d3 = d3.T

    return d3

def project_camera_points_using_f (pts_3d, fx, fy):
    # given 3d points in camera coordinates project to 2d using the given focal length parameters
    # INPUTS
    #   pts_3d - points in 3d camera coordinates to be projected (3 x 24)
    #   fx, fy - focal length parameters
    # RETURNS:
    #   projected points (2 x 24)
    pts_2d = np.zeros( (2, 24) )
    pts_2d[0,:] = ( pts_3d[0,:] / pts_3d[2,:] ) * fx
    pts_2d[1,:] = ( pts_3d[1,:] / pts_3d[2,:] ) * fy
    return pts_2d

def project_camera_points(pts_3d):
    # given 3d points in camera world coordinate system project to the
    # surreal 320 x 240 2d image
    #  INPUTS
    #   pts_3d - points in 3d camera coordinates to be projected (3 x 24)
    # RETURNS:
    #   projected points (2 x 24)
    intrinsic = get_intrinsic_matrix()
    pts_3d_normalised_homogeneous = np.matmul(intrinsic, pts_3d)
    d3_project_x = pts_3d_normalised_homogeneous[0, :] / pts_3d_normalised_homogeneous[2, :]
    d3_project_x = np.expand_dims(d3_project_x, 1)
    d3_project_y = pts_3d_normalised_homogeneous[1, :] / pts_3d_normalised_homogeneous[2, :]
    d3_project_y = np.expand_dims(d3_project_y, 1)
    d3_project = np.concatenate([d3_project_x, d3_project_y], axis=1)
    return d3_project.T

def project_blender_points(pts_3d, intrinsic_matrix, extrinsic_matrix):
    # given 3d points in Blender world coordinate system project to the
    # surreal 320 x 240 2d image using the given extrinsic and intrinsic matrices
    num_points = pts_3d.shape[1]
    pts_3d_homogeneous = np.concatenate([pts_3d, np.ones((1,num_points))], axis=0)
    pts_3d_camera_homogeneous = np.matmul(extrinsic_matrix, pts_3d_homogeneous)
    pts_3d_normalised_homogeneous = np.matmul(intrinsic_matrix, pts_3d_camera_homogeneous)
    d3_project_x = pts_3d_normalised_homogeneous[0,:] / pts_3d_normalised_homogeneous[2,:]
    d3_project_x = np.expand_dims(d3_project_x, 1)
    d3_project_y = pts_3d_normalised_homogeneous[1,:] / pts_3d_normalised_homogeneous[2,:]
    d3_project_y = np.expand_dims(d3_project_y, 1)
    d3_project = np.concatenate([d3_project_x, d3_project_y], axis=1)
    return d3_project

def project_blender_points_using_camera_location(pts_3d, camLoc):
    # given 3d points in Blender world coordinate system project to the
    # surreal 320 x 240 2d image using the given Blender camera location
    intrinsic = get_intrinsic_matrix()
    extrinsic = get_extrinsic_matrix(camLoc)
    return project_blender_points(pts_3d, intrinsic, extrinsic)


def get_intrinsic_matrix():
    # Get the SURREAL intrinsic matrix
    # These are set in Blender(datageneration / main_part1.py)
    res_x_px    = 320    # *scn.render.resolution_x
    res_y_px    = 240    # *scn.render.resolution_y
    f_mm        = 60     # *cam_ob.data.lens
    sensor_w_mm = 32    # *cam_ob.data.sensor_width
    sensor_h_mm = sensor_w_mm * res_y_px / res_x_px # *cam_ob.data.sensor_height(function of others)

    scale = 1 # *scn.render.resolution_percentage / 100
    skew  = 0 # only use rectangular pixels
    pixel_aspect_ratio = 1

    # From similar triangles:
    # sensor_width_in_mm / resolution_x_inx_pix = focal_length_x_in_mm / focal_length_x_in_pix
    fx_px = f_mm * res_x_px * scale / sensor_w_mm
    fy_px = f_mm * res_y_px * scale * pixel_aspect_ratio / sensor_h_mm

    # Center of the image
    u = res_x_px * scale / 2
    v = res_y_px * scale / 2

    # Intrinsic camera matrix
    K = np.array([[fx_px, skew,  u],
                  [0,     fy_px, v],
                  [0,     0    , 1]])

    return K

def get_extrinsic_matrix(T):
    # Get the SURREAL extrinsic matrix
    # function RT = getExtrinsicBlender(T)
    #   returns extrinsic camera matrix
    #
    #   T : translation vector from Blender (*cam_ob.location)
    #   RT: extrinsic computer vision camera matrix
    #   Script based on https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera

    # Take the first 3 columns of the matrix_world in Blender and transpose.
    # This is hard-coded since all images in SURREAL use the same.
    R_world2bcam = np.array([[0, 0, 1], [0, -1, 0], [-1, 0, 0]]).T

    # *cam_ob.matrix_world = Matrix(((0., 0., 1, params['camera_distance']),
    #                               (0., -1, 0., -1.0),
    #                               (-1., 0., 0., 0.),
    #                               (0.0, 0.0, 0.0, 1.0)))

    # Convert camera location to translation vector used in coordinate changes
    T_world2bcam = np.matmul(R_world2bcam, T)
    T_world2bcam = -1 * T_world2bcam

    # Following is needed to convert Blender camera to computer vision camera
    R_bcam2cv = np.array([[ 1,  0,  0 ], [ 0, -1,  0 ], [ 0, 0, -1]])

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = np.matmul(R_bcam2cv, R_world2bcam)
    T_world2cv = np.matmul(R_bcam2cv, T_world2bcam)

    RT = np.concatenate([R_world2cv, T_world2cv], 1)
    return RT
