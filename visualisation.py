from  matplotlib.patches import Circle
import mpl_toolkits.mplot3d as plt3d
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.image import imsave
import sys
sys.path.insert(0,'../smpl')
import numpy as np
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
from smpl_webuser.serialization import load_model

LIMBS = [(0,3), (3,6), (6,9), (9,12), (12,13), (12,14), (13,15), (14,15),\
         (1,4), (4,7), (7,10), (2,5), (5,8), (8,11),  \
         (13,16),(16,18),(18,20),(20,22),  \
         (14,17),(17,19),(19,21),(21,23)]
LIMBCOLORS = ['k','k','k','k','k','k','k','k', \
              'r','r','r','b','b','b','g','g','g','g','c','c','c','c']


def get_axes(subplot=111, projection=None, fig=None):
    if fig is None:
        fig = plt.figure()
    return fig.add_subplot(subplot, projection=projection), fig

def plot_image(ax, img):
  ax.imshow(img)

def plot_image_and_2djoints(ax, img, j_2d, debug=False):
    # plots the img and overlays circles for the joints in j_2d
    # INPUTS:
    #   ax    - matplotlib axes to draw on
    #   img   - the image to display
    #   j_2d  - joints to overlay (24 x 2)
    #   debug - If True outputs the shape of j_2d.shape
    if debug:
        print('j_2d.shape: ', j_2d.shape)
    plot_image(ax, img)
    for jt in range(j_2d.shape[0]):
        circ = Circle((j_2d[jt][0], j_2d[jt][1]), 2)
        ax.add_patch(circ)

def plot_3djoints(ax, j_3d, dotc = 'b', debug=False):
  ax.scatter(j_3d[0], j_3d[1], j_3d[2], c=dotc)
  for ix,(s,e) in enumerate(LIMBS):
    xs = j_3d[0][s], j_3d[0][e]
    ys = j_3d[1][s], j_3d[1][e]
    zs = j_3d[2][s], j_3d[2][e]
    line = plt3d.art3d.Line3D(xs, ys, zs, color=LIMBCOLORS[ix])
    ax.add_line(line)



def output_mesh(mat, idx, mpjpe):
  img = mat['image'][idx][0]
  pose = mat['pose'][idx]
  beta = mat['beta'][idx]
  gender = mat['gender'][idx]
  outimg_path = './sample_img{}_{}.png'.format(idx, mpjpe)
  imsave(outimg_path, img,format='png')

  ## Load SMPL model (here we load the female model)
  if gender == 0:
       m = load_model('../smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl')
  else:
       m = load_model('../smpl/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl')
  m.pose[:] = pose.reshape(m.pose.shape)
  m.betas[:]= beta.reshape(m.betas.shape)
  outmesh_path = './sample_mesh{}_{}.obj'.format(idx, mpjpe)
  with open( outmesh_path, 'w') as fp:
    for v in m.r:
        fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )
    for f in m.f+1: # Faces are 1-based, not 0-based in obj files
        fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )

def render_smpl(pose,beta,angle=0.0):
  ## Load SMPL model (here we load the female model)
  m = load_model('../smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl')
  m.pose[:] = pose.reshape(m.pose.shape)
  m.betas[:]= beta.reshape(m.betas.shape)
  #m.pose[2]=angle

  ## Create OpenDR renderer
  rn = ColoredRenderer()
  ## Assign attributes to renderer
  w, h = (128, 128)

  rn.camera = ProjectPoints(v=m, rt=np.array([0., angle, 0.]), t=np.array([0, 0, 2.]),
                            f=np.array([w,w])/1., c=np.array([w,h])/2., k=np.zeros(5))
  rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
  rn.set(v=m, f=m.f, bgcolor=np.zeros(3))

  ## Construct point light source
  rn.vc = LambertianPointLight(
    f=m.f,
    v=rn.v,
    num_verts=len(m),
    light_pos=np.array([-1000,-1000,-2000]),
    vc=np.ones_like(m)*.9,
    light_color=np.array([1., 1., 1.]))

  ## Show it using OpenCV
  import cv2
  return rn.r

def show_img_and_pose (img, j_2d, pose, beta, angle=np.pi/4):
    # plot 4 windows:
    #   image
    #   image plus 2d points
    #   smpl model no rotation
    #   smpl model 'angle' rotation
    # INPUTS:
    #   img - RGB image to display
    #   j_2d - 2d projection of joints (24 x 2)
    #   pose - SMPL pose
    #   beta - SMPL shape
    #   angle - rotation around z-axis (default 45 degrees)
    LIMBS = [[0,1],[0,2],[0,3],[3,6],[1,4],[2,5],[4,7],[5,8],[7,10],[8,11],[9,13],[9,14],[12,15],
              [12,14],[12,13],[13,16],[14,17],[16,18],[17,19],[18,20],[19,21],[20,22],[21,23]]
    LIMB_COLORS = ['r','r','r','r','b','c','b','c','b','c','r','r','r','r','r','r','r','y','g',
                   'y','g','y','g']
    num_joints = 24
    projectJ = j_2d
    print(projectJ.shape)
    fig = plt.subplot(221)
    fig.imshow(img)
    fig = plt.subplot(222)
    fig.imshow(img)
    for jt in range(num_joints):
        circ = Circle(projectJ[jt,:], 2)
        fig.add_patch(circ)


    for ((pt1,pt2),c) in zip(LIMBS, LIMB_COLORS):
        x0 = projectJ[pt1,0]; x1 = projectJ[pt2,0]; y0 = projectJ[pt1,1]; y1 = projectJ[pt2,1]
        line = Line2D((x0,x1),(y0,y1),color=c)
        fig.add_line(line)
    smpl_img = render_smpl(pose,beta)
    plt.subplot(223)
    plt.imshow(smpl_img)
    smpl_img = render_smpl(pose,beta,angle)
    plt.subplot(224)
    plt.imshow(smpl_img)
    plt.show()

def show_img_and_2d_limbs(img, j_2d):
    # plot 2 windows:
    #   image
    #   image plus 2d points
    # INPUTS:
    #   img - RGB image to display
    #   j_2d - 2d projection of joints (24 x 2)
    LIMBS = [[0, 1], [0, 2], [0, 3], [3, 6], [1, 4], [2, 5], [4, 7], [5, 8], [7, 10], [8, 11], [9, 13], [9, 14],
             [12, 15],
             [12, 14], [12, 13], [13, 16], [14, 17], [16, 18], [17, 19], [18, 20], [19, 21], [20, 22], [21, 23]]
    LIMB_COLORS = ['r', 'r', 'r', 'r', 'b', 'c', 'b', 'c', 'b', 'c', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'y', 'g',
                   'y', 'g', 'y', 'g']
    num_joints = 24
    projectJ = j_2d
    print(projectJ.shape)
    fig = plt.subplot(221)
    fig.imshow(img)
    fig = plt.subplot(222)
    fig.imshow(img)
    for jt in range(num_joints):
        circ = Circle(projectJ[jt, :], 2)
        fig.add_patch(circ)

    for ((pt1, pt2), c) in zip(LIMBS, LIMB_COLORS):
        x0 = projectJ[pt1, 0];
        x1 = projectJ[pt2, 0];
        y0 = projectJ[pt1, 1];
        y1 = projectJ[pt2, 1]
        line = Line2D((x0, x1), (y0, y1), color=c)
        fig.add_line(line)

    plt.show()
