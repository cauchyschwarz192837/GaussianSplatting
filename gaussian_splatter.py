import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import read_points3d_bin, read_cameras_bin, read_images_bin, q2r, \
    jacobian_torch, initialize_sh, inverse_sigmoid, inverse_sigmoid_torch, sample_two_point, Camera
import transforms as tf
import math
import cv2
import numpy as np
from einops import repeat
from tqdm import tqdm
from pykdtree.kdtree import KDTree
EPS=1e-4
    

#---------------------------------------------------------------------------------------------------


def world_to_camera(points, rot, tran):
    _r = points @ rot.T + tran.unsqueeze(0)
    return _r     


"""
column 0: normalized horizontal coordinate
column 1: normalized vertical coordinate
column 2: camera-to-point distance

These three numbers are all the renderer needs to decide which 3-D Gaussians 
project onto which screen tiles and in what order to draw them.
"""
def camera_to_image(points_camera_space):
    points_image_space = [
        points_camera_space[:,0]/points_camera_space[:,2],
        points_camera_space[:,1]/points_camera_space[:,2],
        points_camera_space.norm(dim=-1)
    ]
    return torch.stack(points_image_space, dim=-1)


#-------------------------------------------------------------------------------------------------------------------


class Learnables(nn.Module):
    def __init__(self, position, rgb_color, opacity, quaternion_rotation=None, scale=None, cov=None, init_values=False):
        super().__init__()
        self.init_values = init_values


        if init_values:  # if LEARNABLE
            self.position = nn.parameter.Parameter(position)   # centre x,y,z in world frame
            self.rgb_color = nn.parameter.Parameter(rgb_color)   # colour in linear RGB (stored as logits, later .sigmoid())
            self.opacity = nn.parameter.Parameter(opacity)   # opacity logit
            self.quaternion_rotation = quaternion_rotation if quaternion_rotation is None else nn.parameter.Parameter(quaternion_rotation)   # 	unit quaternion for ellipsoid orientation
            self.scale = scale if scale is None else nn.parameter.Parameter(scale)   # 	radii along the local X,Y,Z axes (log-space or signed, depending on activation)
            self.cov = cov if cov is None else nn.parameter.Parameter(cov)   # 	pre-computed 3-D covariance (alternative to quaternion_rotation + scale)
        else:
            self.position = position
            self.rgb_color = rgb_color
            self.opacity = opacity
            self.quaternion_rotation = quaternion_rotation
            self.scale = scale
            self.cov = cov


    # override
    def to(self, *args, **kwargs):
        self.position.to(*args, **kwargs)
        self.rgb_color.to(*args, **kwargs)
        self.opacity.to(*args, **kwargs)
        if self.quaternion_rotation is not None:
            self.quaternion_rotation.to(*args, **kwargs)
        if self.scale is not None:
            self.scale.to(*args, **kwargs)
        if self.cov is not None:
            self.cov.to(*args, **kwargs)


    # DONE, REORDER
    def filter_me(self, mask):     # A Gaussian can be parameterised in one of two mutually-exclusive ways:
        if self.quaternion_rotation is not None and self.scale is not None:
            assert self.cov is None
            return Learnables(
                position=self.position[mask],
                rgb_color=self.rgb_color[mask],
                opacity=self.opacity[mask],
                quaternion_rotation=self.quaternion_rotation[mask],
                scale=self.scale[mask],
            )
        else:
            assert self.cov is not None
            return Learnables(
                position=self.position[mask],
                rgb_color=self.rgb_color[mask],
                opacity=self.opacity[mask],
                cov=self.cov[mask],
            )
        

    """
    turns per-Gaussian orientation + axis lengths into a proper world-space 3x3 
    covariance that downstream code can project to the screen
    """
    def get_gaussian_3d_cov(self):
        R = q2r(self.quaternion_rotation)
        # self.quaternion_rotation: per-Gaussian unit quaternion [B,4] that orients the ellipsoid’s local axes.
        # q2r(...) → rotation matrices R ∈ ℝ^{B×3×3}. Each row gives the ellipsoid’s orientation in world coordinates.

        _scale = self.scale.abs() + 1e-4
        S = torch.diag_embed(_scale)

        RS = torch.bmm(R, S)
        RSSR = torch.bmm(RS, RS.permute(0,2,1)) # Returns the full 3D covariance per Gaussian in world space (B, 3, 3)
        # permute(0, 2, 1) is just transpose each 3×3 matrix in the batch, guarantee PSD even with small numeric noise
        return RSSR    # symmetric positive semidefinite covariance matrix in world coordinates for each Gaussian


    def reset_opa(self):
        torch.nn.init.uniform_(self.opacity, a=inverse_sigmoid(0.01), b=inverse_sigmoid(0.01)) # So all Gaussians are reset to an opacity of 1 % (nearly transparent).


    """
    Edits the current set of Gaussians to keep the useful ones and add capacity where gradients say more detail is needed
        Delete Gaussians that are either too transparent or too large
        Densify important ones (high gradient):
            Clone small Gaussians (duplicate + nudge)
            Split large Gaussians (shrink the parent, add a sibling sampled from its covariance)
    """
    def adaptive_density_control(self, 
        grad: torch.Tensor,     # per-Gaussian gradient stats for position
        split_or_clone_threshold: float,    # compares against abs(scale), larger: split, smaller/equal: clone
        delete_threshold: float,    # if abs(scale) less than or equal, Gaussian: deleted
        gradient_threshold=0.0002,    # minimum gradient magnitude for densification to trigger
        use_clone=True,
        use_split=True,
        clone_dt=0.01,    # step size to nudge clones along −grad
    ):
        
        assert self.init_values
        
        # DELETION

        delete_mask = (self.opacity > inverse_sigmoid(0.02)) & (self.scale.norm(dim=-1) < delete_threshold)   # self.opacity is in logit space
        self.position = nn.parameter.Parameter(self.position.detach()[delete_mask])    # no longer connected to the graph for gradient computation
        self.rgb_color = nn.parameter.Parameter(self.rgb_color.detach()[delete_mask])
        self.opacity = nn.parameter.Parameter(self.opacity.detach()[delete_mask])
        self.quaternion_rotation = nn.parameter.Parameter(self.quaternion_rotation.detach()[delete_mask])
        self.scale = nn.parameter.Parameter(self.scale.detach()[delete_mask])
        grad = grad[delete_mask]


        # CLONING OR SPLITING SURVIVING GAUSSIANS

        densify_mask = grad.abs().max(-1)[0] > gradient_threshold

        # we will later append() extra Gaussians to these lists before concatenating.
        new_position = [self.position.clone().detach()]    # no longer connected to the graph for gradient computation
        new_rgb_color = [self.rgb_color.clone().detach()]
        new_opacity = [self.opacity.clone().detach()]
        new_quaternion_rotation = [self.quaternion_rotation.clone().detach()]
        new_scale = [self.scale.clone().detach()]

        if densify_mask.any():    # only those with high gradients
            scale_norm = self.scale.norm(dim=-1)
            split_mask = scale_norm > split_or_clone_threshold
            clone_mask = scale_norm <= split_or_clone_threshold
            split_mask = split_mask & densify_mask
            clone_mask = clone_mask & densify_mask

            if clone_mask.any() and use_clone:
                cloned_pos = self.position[clone_mask].clone().detach()
                cloned_pos -= grad[clone_mask] * clone_dt
                cloned_rgb = self.rgb_color[clone_mask].clone().detach()
                cloned_opa = self.opacity[clone_mask].clone().detach()
                cloned_quat = self.quaternion_rotation[clone_mask].clone().detach()
                cloned_scale = self.scale[clone_mask].clone().detach()

                new_position.append(cloned_pos)
                new_rgb_color.append(cloned_rgb)
                new_opacity.append(cloned_opa)
                new_quaternion_rotation.append(cloned_quat)
                new_scale.append(cloned_scale)

            if split_mask.any() and use_split:
                _scale = self.scale.clone().detach() 
                _scale[split_mask] /= 1.6

                new_scale[0] = _scale

                # sampling two positions
                this_cov = self.get_gaussian_3d_cov()[split_mask]
                p1, p2 = sample_two_point(self.position[split_mask], this_cov)

                origin_pos = new_position[0]
                origin_pos[split_mask] = p1.detach()
                new_position[0] = origin_pos
                split_pos = p2.detach()
                split_rgb = self.rgb_color[split_mask].clone().detach()
                split_opa = self.opacity[split_mask].clone().detach()
                split_quat = self.quaternion_rotation[split_mask].clone().detach()
                split_scale = _scale[split_mask].clone()

                new_position.append(split_pos)
                new_rgb_color.append(split_rgb)
                new_opacity.append(split_opa)
                new_quaternion_rotation.append(split_quat)
                new_scale.append(split_scale)
            self.position = nn.parameter.Parameter(torch.cat(new_position))
            self.rgb_color = nn.parameter.Parameter(torch.cat(new_rgb_color))
            self.opacity = nn.parameter.Parameter(torch.cat(new_opacity))
            self.quaternion_rotation = nn.parameter.Parameter(torch.cat(new_quaternion_rotation))
            self.scale = nn.parameter.Parameter(torch.cat(new_scale))



    """
    Take 3D Gaussians in world space (center + 3D covariance from quaternion_rotation & scale) 
    and turn them into 2D Gaussians on the image plane (center in normalized image 
    coords + 2x2 covariance). Also convert color/opacity from logits to display space.

    self.position : [N,3] centers in world coords
    self.quaternion_rotation, self.scale → define each blob's 3D shape
    rot : [3,3] world→camera rotation
    tran: [3] world→camera translation
    scale_activation: how to turn scale into positive radii

    Outputs:
    A new Learnables whose:
        position: [N,3] = (x/Z, y/Z, depth) (normalized image coords + depth)
        cov: [N,2,2] = 2D covariance on the image plane
        rgb_color, opacity: squashed to [0,1]
    """
    def project(self, rot, tran):
        # World → camera → image coordinates (centers only)
        pos_cam_space = world_to_camera(self.position, rot, tran)
        pos_img_space = camera_to_image(pos_cam_space)

        # Jacobian of the projection (for covariance push-forward)
        # Jacobian (matrix of partial derivatives) of the mapping you use in camera_to_image
        jacobian = jacobian_torch(pos_cam_space)


        gaussian_3d_cov = self.get_gaussian_3d_cov()
        # renderer projects 3D Gaussians to the image. The 2D ellipse wedraw comes from pushing this 3D 
        # covariance through the camera Jacobian
        
        # JW = torch.einsum("bij,bjk->bik", jacobian, rot.unsqueeze(dim=0))
        # JW is the Jacobian of the whole mapping from world coords to image coords. 
        # We get it by chain rule, so we multiply projection Jacobian by the rotation (Check math)
        JW = torch.matmul(jacobian, rot.unsqueeze(dim=0))
        JWC = torch.bmm(JW, gaussian_3d_cov)   # can do the covariance math and check this
        gaussian_2d_cov = torch.bmm(JWC, JW.permute(0,2,1))[:, :2, :2]

        gaussian_3ds_image_space = Learnables(     # in image space now
            position=pos_img_space,
            rgb_color=self.rgb_color.sigmoid(),
            opacity=self.opacity.sigmoid(),
            cov=gaussian_2d_cov,
        )

        return gaussian_3ds_image_space



#------------------------------------------------------------------------------------------------------


class Tiles:
    def __init__(self, width, height, focal_x, focal_y, device):
        self.width = width 
        self.height = height
        self.padded_width = int(math.ceil(self.width / 16)) * 16     # padded up to multiple of 16
        self.padded_height = int(math.ceil(self.height / 16)) * 16   # padded up to multiple of 16
        self.focal_x = focal_x
        self.focal_y = focal_y
        self.n_tile_x = self.padded_width // 16
        self.n_tile_y = self.padded_height // 16
        self.device = device

    def __len__(self):
        return self.tiles_top.shape[0]
    

    def crop(self, image):
        # image: padded_height x padded_width x 3
        # output: height x width x 3
        top = int(self.padded_height - self.height) // 2 
        left = int(self.padded_width - self.width) // 2 
        return image[top:top + int(self.height), left:left + int(self.width), :]
    

    """
    create_tiles() computes the tile borders along x & y in image-plane coordinates (not pixels).
    """
    def create_tiles(self):
        # First makes pixel edges:
        self.tiles_left = torch.linspace(-self.padded_width/2, self.padded_width/2, self.n_tile_x + 1, device=self.device)[:-1]
        self.tiles_right = self.tiles_left + 16
        self.tiles_top = torch.linspace(-self.padded_height/2, self.padded_height/2, self.n_tile_y + 1, device=self.device)[:-1]
        self.tiles_bottom = self.tiles_top + 16

        # Dividing pixel coordinates by focal length converts to normalized image plane coordinates
        # tile_geo_length_* is the physical size of a tile on that normalized plane
        self.tile_geo_length_x = 16 / self.focal_x
        self.tile_geo_length_y = 16 / self.focal_y
        self.leftmost = -self.padded_width/2/self.focal_x
        self.topmost = -self.padded_height/2/self.focal_y

        # Then scale by the focal lengths to convert from pixel units to normalized image plane, actually convert all four edge arrays to normalized coords
        self.tiles_left = self.tiles_left / self.focal_x
        self.tiles_top = self.tiles_top / self.focal_y
        self.tiles_right = self.tiles_right / self.focal_x
        self.tiles_bottom = self.tiles_bottom / self.focal_y

        """
        Now every edge value matches the coordinate system of the projected Gaussians 
        (which we compute with camera_to_image, i.e. divide by z, then x/fx, y/fy).

        expand 1D edge arrays to per-tile lists
        """

        self.tiles_left = repeat(self.tiles_left, "b -> (c b)", c=self.n_tile_y)
        self.tiles_right = repeat(self.tiles_right, "b -> (c b)", c=self.n_tile_y)

        self.tiles_top = repeat(self.tiles_top, "b -> (b c)", c=self.n_tile_x)
        self.tiles_bottom = repeat(self.tiles_bottom, "b -> (b c)", c=self.n_tile_x)

        # We want four flat arrays (left/right/top/bottom) where entry k describes tile k (in row-major order)


#---------------------------------------------------------------------------------------------------------------------------



"""
# load the COLMAP reconstruction (3-D points, cameras, and registered images), convert every 3-D point 
# into a learnable 3-D Gaussian (position + colour + opacity + shape), then keep all book-keeping data 
# needed to render those Gaussians from any camera view.
"""

# rendered_img = self.gaussian_splatter(camera_id)
# self.gaussian_splatter is an instance of Gaussian_Splatter
class Gaussian_Splatter(nn.Module):         # SEE w2c ... 
    def __init__(self, colmap_path, image_path, near=0.3, render_downsample=1, spherical_harmonics=False, 
                    opacity_init_value=0.1, scale_init_value=0.02, load_checkpoint=None,
                    test=False,
                 ):
        super().__init__()


        self.device = torch.device("cuda")
        self.near = near    # near-plane distance (frustum culling)
        self.render_downsample = render_downsample 
        

        #*********************************************************************************


        self.points3d = read_points3d_bin(colmap_path/"points3D.bin")    
        # Reconstructed 3D points and which images see them


        self.cameras = read_cameras_bin(colmap_path/"cameras.bin")
        # Intrinsics (camera model, image size, focal length, principal point, distortion coefficients)
        # This builds a dict: camera_id : Camera(id, model, width, height, params)
        # Only params[0] and params[1] (fx, fy) are used later; width/height are used for FOV bounds


        self.images_info = read_images_bin(colmap_path/"images.bin")    
        # Extrinsics (pose for each registered image: rotation and translation), filename, and its 2D–3D correspondences


        self.image_path = image_path


        #*********************************************************************************


        self.test = test
        if not self.test:
            self.parse_imgs()


        # ------------------------------------------------------------------
        # 2) Pull every COLMAP 3-D point into GPU tensors
        # ------------------------------------------------------------------
        _points = []      # world-space XYZ           →  torch.float32  (later shape  [N,3])
        _rgbs   = []      # colour in 0-1 sRGB logits →  torch.float32  (later shape  [N,3] or [N,27])


        for useless, point in self.points3d.items():          # <-- self.points3d came from read_points3d_binary
            # a) coordinates ------------------------------------------------
            _points.append(torch.from_numpy(point.xyz))   # numpy → torch (CPU for now)
            
            # b) colour  ----------------------------------------------------
            #   COLMAP stores uint8 [0..255].  We want logits so that
            #   sigmoid(logit) ≈ original-rgb_color.  “inverse_sigmoid_torch”
            #   does: log(r / (1-r)).
            rgb01 = torch.from_numpy(point.rgb_color / 255.0)   # 0-1 float, [N,3]
            _rgbs.append(inverse_sigmoid_torch(rgb01))    # convert to logits (FULL RANGE)
            #   (if spherical_harmonics == True we will expand to 9 SH coeffs / channel later)


        # ------------------------------------------------------------------
        # 3) Stack & move to GPU
        # ------------------------------------------------------------------
        rgb_color = torch.stack(_rgbs).to(torch.float32).to(self.device)   # [N,3]
        if spherical_harmonics:                                        # optional spherical-harmonics
            rgb_color = initialize_sh(rgb_color)                                 # [N,27]


        _pos = torch.stack(_points).to(torch.float32).to(self.device)  # [N,3]


        if load_checkpoint is None:                      # fresh training run
            # Build a KD-tree on CPU to find each point’s 3 nearest neighbours
            kd_tree  = KDTree(_pos.cpu().numpy())          # scipy-style fast NN
            dist, _  = kd_tree.query(_pos.cpu().numpy(), k=4)  # returns [N,4] distances (self + 3 nn)
            
            mean_min_three_dis = dist[:,1:].mean(axis=1)   # average of 3 neighbour dists  → shape  [N]
            mean_min_three_dis = torch.tensor(mean_min_three_dis, dtype=torch.float32) * scale_init_value  # arbitrary  
            # global multiplicative knob

            """
            mean_min_three_dis is now one radius per point; each row will be
            duplicated to three axes in a moment.
            """

            # ------------------------------------------------------------------
            # 5) Build the big learnable tensor container  (Learnables)
            # ------------------------------------------------------------------
            # Learnables inside here

            self.gaussian_3ds = Learnables(
                position=_pos.to(self.device),    # B x 3          # [N,3]  XYZ
                rgb_color = rgb_color,    # B x 3 or 27      # [N,3]  or [N,27]
                opacity = torch.ones(len(_points)).to(torch.float32).to(self.device) * inverse_sigmoid(opacity_init_value), # B   # initialise α to *opacity_init_value*
                quaternion_rotation = torch.Tensor([1, 0, 0, 0]).unsqueeze(dim=0).repeat(len(_points),1).to(torch.float32).to(self.device), # B x 4      # identity orientation  [N,4]
                scale = torch.ones(len(_points), 3).to(torch.float32).to(self.device) * mean_min_three_dis.unsqueeze(dim=1).to(self.device),
                init_values=True,
            )

        else:
            self.gaussian_3ds = Learnables(
                position=_pos.to(self.device), # B x 3
                rgb_color = rgb_color, # B x 3 or 27
                opacity = torch.ones(len(_points)).to(torch.float32).to(self.device) * inverse_sigmoid(opacity_init_value), # B
                quaternion_rotation = torch.Tensor([1, 0, 0, 0]).unsqueeze(dim=0).repeat(len(_points),1).to(torch.float32).to(self.device), # B x 4
                scale = torch.ones(len(_points), 3).to(torch.float32).to(self.device),
                init_values=True,
            )
        """
        Learnables is a thin wrapper around five learnable tensors
        (position, rgb_color, opacity, quaternion_rotation, scale)
        """


        # ------------------------------------------------------------------
        # 4) Choose initial Gaussian *scale* for every point
        # ------------------------------------------------------------------
        if load_checkpoint is not None:      # then overwrite each tensor with values, IMMEDIATELY OVERWRITE PREVIOUS ELSE BLOCK
            # load checkpoint
            ckpt = torch.load(load_checkpoint)
            self.gaussian_3ds.position = nn.Parameter(ckpt["position"])
            self.gaussian_3ds.opacity = nn.Parameter(ckpt["opacity"])
            self.gaussian_3ds.rgb_color = nn.Parameter(ckpt["rgb_color"])
            self.gaussian_3ds.quaternion_rotation = nn.Parameter(ckpt["quaternion_rotation"])
            self.gaussian_3ds.scale = nn.Parameter(ckpt["scale"])

        self.current_camera = None    # will be filled by set_camera()
        if not self.test:             # "training" mode: immediately preload first image
            self.set_camera(0)        # In “training” mode (test=False), it preloads the first COLMAP frame (index 0) 
    



    def parse_imgs(self):
        img_ids = sorted([im.id for im in self.images_info.values()])
        self.w2c_quats = []
        self.w2c_rots = []
        self.w2c_trans = []
        self.cam_ids = []
        self.imgs = []
        """
        world→camera quaternion (w2c_quats)
        world→camera rotation matrix (w2c_rots)
        world→camera translation (w2c_trans)
        the camera_id used by this image (cam_ids)
        the image tensor (imgs)
        """

        for img_id in tqdm(img_ids):
            img_info = self.images_info[img_id]   # GET, img_info contains: id, qvec, tvec, camera_id, name, ... (read from images.bin)

            cam = self.cameras[img_info.camera_id]    
            # Grab the intrinsics entry (from cameras.bin) that this image refers to (e.g., model + width/height + fx,fy,…). 
            # We don’t use cam immediately here, but we need this mapping later in set_camera().

            image_filename = os.path.join(self.image_path, img_info.name)
            if not os.path.exists(image_filename):
                continue

            _current_image = cv2.imread(image_filename)
            _current_image = cv2.cvtColor(_current_image, cv2.COLOR_BGR2RGB)
            self.imgs.append(torch.from_numpy(_current_image).to(torch.uint8).to(self.device))
            """
            Load with OpenCV (BGR), convert to RGB.
            Store as a uint8 GPU tensor [H, W, 3]
            Later in set_camera(idx), we cast it to float16 and divide by 255 to get [0,1]. ************
            """

            # this object represents the world→camera transform:
            T_world_camera = tf.SE3.from_rotation_and_translation(    # tf.SE3.from_rotation_and_translation(R, t) builds an SE(3) transform from rotation R and translation t.
                tf.SO3(img_info.qvec), img_info.tvec,    # tf.SO3(img_info.qvec) constructs an SO(3) rotation from the quaternion qvec

            )

            self.w2c_quats.append(torch.from_numpy(T_world_camera.rotation().wxyz).to(torch.float32).to(self.device))
            self.w2c_trans.append(torch.from_numpy(T_world_camera.translation()).to(torch.float32).to(self.device))
            self.w2c_rots.append(q2r(self.w2c_quats[-1].unsqueeze(0)).squeeze().to(torch.float32).to(self.device))
            # q2r converts a batch of quaternions [B,4] to rotation matrices [B,3,3]
            # We unsqueeze(0) to make the latest quaternion a batch of 1, then convert, then squeeze() back to [3,3]
            # Store the world→camera rotation matrix on the GPU

            self.cam_ids.append(img_info.camera_id)    
            # Record which camera intrinsics this image uses. Later set_camera(idx) will use this to pick 
            # self.current_camera = self.cameras[self.cam_ids[idx]].

            """
            store per-image arrays: self.w2c_rots[i] = R, self.w2c_trans[i] = t, 
            plus the image and its camera_id.
            """


        
    #------------------------------------------------------------------

    """
    Prepares everything needed to render from a specific camera. 

    When the renderer (Gaussian_Splatter) is asked to draw a frame it must first know which camera to use.

    If idx is an integer → use one of the COLMAP frames already loaded.
    If idx is None → the caller supplies raw extrinsics / intrinsics dicts (useful for GUI or novel-view rendering).

    """ # SEE PARSE_IMAGES
    def set_camera(self, idx, extrinsics=None, intrinsics=None):

        # 1) idx is None → external/GUI camera (novel view)
        # You provide raw camera extrinsics/intrinsics dicts

        if idx is None:
            # print(extrinsics)
            # These say "how to map world points into this camera’s coordinates."
            self.current_w2c_rot = torch.from_numpy(extrinsics["rot"]).to(torch.float32).to(self.device)
            self.current_w2c_tran = torch.from_numpy(extrinsics["tran"]).to(torch.float32).to(self.device)  # 3-vector translation (camera origin in world coords).  
            self.current_w2c_quat = None

            # This branch is for rendering a novel view; there’s no reference photo to compare against
            self.ground_truth = None

            self.current_camera = Camera(   # This dataclass stores what the pinhole needs: image size and focal lengths.
                id=-1, model="pinhole", width=intrinsics["width"], height=intrinsics["height"],
                params = np.array(
                    [intrinsics["focal_x"], intrinsics["focal_y"]]
                ),
            )

            #-------------------------------------------------------------------------------------------------------

            """
            Tiles pads width/height up to multiples of 16, computes per-tile 
            boundaries in normalized image coords (divide by focal)

            Renderer uses this to quickly cull Gaussians per tile.
            """

            self.tile_info = Tiles(    # Tiles pre-computes how the padded screen is split into 16 × 16 tiles and stores geometric lengths for culling.
                math.ceil(intrinsics["width"]), 
                math.ceil(intrinsics["height"]), 
                intrinsics["focal_x"], 
                intrinsics["focal_y"], 
                self.device
            )
            self.tile_info.create_tiles()


        # 2) idx is an integer → one of the COLMAP registered images
        # We use pre-parsed info from parse_imgs()

        else:
            self.current_w2c_quat = self.w2c_quats[idx]
            self.current_w2c_tran = self.w2c_trans[idx]
            self.current_w2c_rot = self.w2c_rots[idx]

            self.ground_truth = self.imgs[idx].to(torch.float16) / 255.     
            # Shape is [H,W,3], dtype fp16, values in [0,1]. This is what training compares to the rendered output (for L1/SSIM/PSNR)

            downsample = max(1, int(self.render_downsample))
            if downsample > 1:
                self.ground_truth = F.interpolate(
                    self.ground_truth.permute(2, 0, 1).unsqueeze(0),  # [1,3,H,W]
                    scale_factor=1.0 / downsample,
                    mode="area"
                ).squeeze(0).permute(1, 2, 0)  # [H,W,3]


            # If this image uses a different camera model/size/focal from the previous one, rebuild the tiling
            if self.cameras[self.cam_ids[idx]] != self.current_camera:
                self.current_camera = self.cameras[self.cam_ids[idx]]
                focal_x = self.current_camera.params[0] / downsample
                focal_y = self.current_camera.params[1] / downsample

                # width then height
                self.tile_info = Tiles(int(self.ground_truth.shape[1]), int(self.ground_truth.shape[0]), focal_x, focal_y, self.device)
                self.tile_info.create_tiles()

        """
        "Tiling": split the (padded) screen into fixed 16 x 16 pixel blocks 
        so we can figure out, per block, which Gaussians affect it. So we 
        don't blend every Gaussian into every pixel — only the small subset overlappinh that block
        """




    def project_and_culling(self):
        # project 3D to 2D
        """
        self.gaussian_3ds.position: [N, 3] world-space centers of all Gaussians
        self.current_w2c_rot: [3, 3] rotation (world→camera)
        self.current_w2c_tran: [3] translation (world→camera)
        """

        gaussian_3ds_pos_camera_space = world_to_camera(self.gaussian_3ds.position, self.current_w2c_rot, self.current_w2c_tran)
        # p_cam = R * p_world + t, output camera-space positions.
        #------------------------------------------------------------------------- Frustum culling 
        # NEAR-PLANE TEST: Keep only points in front of the camera and farther than the near distance.
        # (Z in camera space must be positive and > near)
        valid = gaussian_3ds_pos_camera_space[:,2] > self.near

        # Project to normalized image plane
        gaussian_3ds_pos_image_space = camera_to_image(gaussian_3ds_pos_camera_space)
        """
        Perspective divide:
            x_norm = X/Z
            y_norm = Y/Z
            third column is ||[X,Y,Z]|| (depth magnitude) used later for sorting
        Shape: [N, 3] → columns (x_norm, y_norm, depth).
        """

        # Field-of-view check (loose screen bounds), check both sides
        culling_mask = (gaussian_3ds_pos_image_space[:, 0].abs() < (self.current_camera.width*1.2/2/self.current_camera.params[0]))  & \
                        (gaussian_3ds_pos_image_space[:, 1].abs() < (self.current_camera.height*1.2/2/self.current_camera.params[1]))
        # half-width in normalized coords ≈ (W/2) / fx
        # half-height in normalized coords ≈ (H/2) / fy
        # The * 1.2 is a safety margin (20% larger box) so we don’t drop things that slightly spill in due to Gaussian extent


        valid &= culling_mask # valid now means: in front of camera AND within an expanded view box


        self.gaussian_3ds_valid = self.gaussian_3ds.filter_me(valid) 
        # self.gaussian_3ds_valid is a new Learnables containing only the 
        # filtered subset of Gaussians (its position/rgb_color/opacity/quaternion_rotation/scale are all masked consistently).

        #-------------------------------------------------------------------------

        self.culling_gaussian_3d_image_space = self.gaussian_3ds_valid.project(
            self.current_w2c_rot, 
            self.current_w2c_tran, 
            self.near
        )      # READY TO RASTERIZE
        

#--------------------------------------------------------------------------------------------------------------------


    # --- Helper: conservative 2D footprint from 2x2 covariance ---
    def _gaussian_bbox_from_cov(self, pos_xy: torch.Tensor, cov: torch.Tensor, sigma_mult: float = 3.0, eps: float = 1e-8):
        """
        pos_xy: [K,2] normalized image-plane centers (x_norm, y_norm), batch of Gaussians dimension in front
        cov   : [K,2,2] image-plane covariance for each Gaussian
        sigma_mult: how many stddevs to cover (3 ~ 99.7%)
        returns x0,x1,y0,y1 each shape [K]
        """
        # Axis-aligned approximation: use diagonal entries
        sigma_x = torch.sqrt(torch.clamp(cov[:, 0, 0], min=eps))
        sigma_y = torch.sqrt(torch.clamp(cov[:, 1, 1], min=eps))
        rx = sigma_mult * sigma_x
        ry = sigma_mult * sigma_y
        x0 = pos_xy[:, 0] - rx
        x1 = pos_xy[:, 0] + rx
        y0 = pos_xy[:, 1] - ry
        y1 = pos_xy[:, 1] + ry
        return x0, x1, y0, y1
        # we need a quick, safe "which tiles could this Gaussian touch?" test
    


    # --- Helper: normalized bbox -> tile index ranges ---
    def _bbox_to_tile_ranges(self, 
                            x0: torch.Tensor, x1: torch.Tensor,
                            y0: torch.Tensor, y1: torch.Tensor,
                            left0: float, top0: float,
                            gx: float, gy: float,
                            n_x: int, n_y: int):
        """
        Map normalized coords to tile indices.
        left0/top0: top-left corner of the plane (normalized)
        gx/gy: tile size on the normalized plane (16/fx, 16/fy)
        n_x/n_y: tile counts
        returns ix0,ix1,iy0,iy1 each shape [K], int64 clamped to valid range
        """
        ix0 = torch.floor((x0 - left0) / gx).to(torch.int64)
        ix1 = torch.floor((x1 - left0) / gx).to(torch.int64)
        iy0 = torch.floor((y0 - top0) / gy).to(torch.int64)
        iy1 = torch.floor((y1 - top0) / gy).to(torch.int64)

        ix0 = torch.clamp(ix0, 0, n_x - 1)    # Torch tensors
        ix1 = torch.clamp(ix1, 0, n_x - 1)
        iy0 = torch.clamp(iy0, 0, n_y - 1)
        iy1 = torch.clamp(iy1, 0, n_y - 1)
        return ix0, ix1, iy0, iy1
        # tiles live on the normalized plane. Dividing by the tile size converts coords → tile indices.
    


    # --- Torch-only version of calc_tile_list ---
    def calc_tile_list_torch(self, gauss: Learnables, tile_info: Tiles, MAXP: int, sigma_mult: float = 3.0):
        """
        gauss: Learnables in *image space* (position [K,3], cov [K,2,2], rgb_color/opacity already sigmoid)
        tile_info: Tiles instance (normalized geometry precomputed)
        MAXP: per-tile capacity (drop extra if overflow)
        returns:
        tile_n_point: [T] int32 counts per tile
        tile_gaussian_list: [T,MAXP] int32 of Gaussian indices per tile (-1 pad)
        """
        device = gauss.position.device
        K = gauss.position.shape[0]
        T = tile_info.n_tile_x * tile_info.n_tile_y     # total number of tiles

        tile_n_point = torch.zeros(T, dtype=torch.int32, device=device)
        tile_gaussian_list = torch.full((T, MAXP), -1, dtype=torch.int32, device=device)

        # 1) conservative bbox per Gaussian
        pos_xy = gauss.position[:, :2]  # [K,2]
        x0, x1, y0, y1 = self._gaussian_bbox_from_cov(pos_xy, gauss.cov, sigma_mult=sigma_mult)   # get box boundaries

        # 2) bbox -> tile index ranges
        ix0, ix1, iy0, iy1 = self._bbox_to_tile_ranges(      # return Torch tensors
            x0, x1, y0, y1,
            tile_info.leftmost, tile_info.topmost,
            tile_info.tile_geo_length_x, tile_info.tile_geo_length_y,
            tile_info.n_tile_x, tile_info.n_tile_y
        )

        # 3) fill per-tile lists (simple loops for clarity)
        for g in range(K):   # for each gaussian
            y0g = int(iy0[g].item())
            y1g = int(iy1[g].item())
            x0g = int(ix0[g].item())
            x1g = int(ix1[g].item())

            for iy in range(y0g, y1g + 1):      # iterate tile rows the Gaussian’s footprint covers. (y0g..y1g came from projecting the Gaussian to the image, making a bbox, then mapping that bbox to tile indices.)
                base = iy * tile_info.n_tile_x     # compute the row’s starting tile id in row-major layout, (Row iy tiles are numbered iy*n_tile_x .. iy*n_tile_x + (n_tile_x-1).)
                for ix in range(x0g, x1g + 1):     # iterate tile columns the Gaussian covers on that row
                    t = base + ix                   #  flatten 2D tile coords (iy, ix) → 1D tile id t
                    c = int(tile_n_point[t].item())    
                    # read how many Gaussians are already listed for tile t. 
                    # That value is also the NEXT FREE SLOT INDEX in tile_gaussian_list[t].

                    if c < MAXP:                      # add if capacity left, g is the index of the gaussian
                        tile_gaussian_list[t, c] = g  # tile_gaussian_list: [T, MAXP] int32, filled with indices of Gaussians per tile, using -1 for padding.
                        tile_n_point[t] = c + 1       # increment gaussian count for tile t

        return tile_n_point, tile_gaussian_list
        # compute a tile bbox per Gaussian, then append its index to each tile it overlaps





    # --- Torch-only version of gather_gaussians ---
    def gather_gaussians_torch(self, tile_n_point: torch.Tensor, tile_gaussian_list: torch.Tensor):
        """
        tile_n_point: [T] int32
        tile_gaussian_list: [T, MAXP] int32 with -1 padding

        returns:
        gathered_list: [K'] int64 Gaussian indices in flat, tile-major order
        tile_ids_for_points: [K'] int64 tile id for each gathered item
        tile_n_point_accum: [T+1] int32 prefix sum (segment starts per tile)
        """
        device = tile_gaussian_list.device
        T = tile_gaussian_list.shape[0]

        # segment boundaries per tile
        tile_n_point_accum = torch.cat(
            [torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(tile_n_point, dim=0)],
            dim=0
        )

        # flatten while dropping paddings (-1)
        mask = tile_gaussian_list.ne(-1)
        gathered_list = tile_gaussian_list[mask].to(torch.int64)  # for tensor indexing

        # tile id for every gathered element
        tile_ids = torch.arange(T, dtype=torch.int32, device=device)
        tile_ids_for_points = tile_ids.repeat_interleave(tile_n_point).to(torch.int64)

        return gathered_list, tile_ids_for_points, tile_n_point_accum



    #-----------------------------------------------------------------------------------------------------------------------



    def render(self, out_write: bool = True, sigma_mult: float = 3.0):
        # If there are no Gaussians after culling/projection, immediately return a black image (right size, on GPU)
        if len(self.culling_gaussian_3d_image_space.position) == 0:
            return torch.zeros(self.tile_info.padded_height,
                            self.tile_info.padded_width, 3,
                            device=self.device, dtype=torch.float32)
        
        # K: number of projected Gaussians still alive
        # MAXP: per-tile cap (heuristic), i.e., at most this many Gaussians can be listed for any single tile. Prevents unbounded memory
        K = len(self.culling_gaussian_3d_image_space.position)   # TOTAL
        MAXP = max(1, K // 20)

        # --- Torch-only tile building ---
        tile_n_point, tile_gaussian_list = self.calc_tile_list_torch(
            self.culling_gaussian_3d_image_space,  # position/cov already in image space
            self.tile_info,
            MAXP=MAXP,
            sigma_mult=sigma_mult,
        )
        """
        For each Gaussian, approximate a bounding box on the normalized plane using sigma_mult * sqrt(variances) from its 2 x 2 cov.
        Map that bbox to tile index ranges using leftmost, topmost, and tile_geo_length_{x,y}.
        For every tile (iy, ix) the bbox touches, append the Gaussian's index to tile_gaussian_list[t, :] (if capacity left) 
        and increment tile_n_point[t].
        """

        #---------------------------------------------------------------------------------------
        """
        Outputs (in-place):
            tile_n_point[t] = number of Gaussians that may affect tile t.
            tile_gaussian_list[t, :tile_n_point[t]] = their indices, the rest remain -1. (tile t)

        tile_n_point = min(tile_n_point, MAXP): clip counts to the allocated capacity

        For each projected 2-D Gaussian (each 3-D blob after projection), decide which 16 x 16 tiles of the screen it overlaps. Then build two things the rasterizer needs:
            tile_n_point — how many Gaussians touch each tile
            tile_gaussian_list — the indices of those Gaussians per tile (padded with -1)
        """


        """
        self.culling_gaussian_3d_image_space.position → position shape [K, 3]
        Each row is (x_norm, y_norm, depth), i.e., normalized image-plane coords and a depth for sorting.

        self.culling_gaussian_3d_image_space.cov → cov shape [K, 2, 2]
        The 2 x 2 image-plane covariance of each 2-D Gaussian.

        self.tile_info (from set_camera):

            n_tile_x, n_tile_y: tiles along x/y
            tile_geo_length_x, tile_geo_length_y: physical tile width/height on the normalized plane (i.e., 16/fx and 16/fy)
            leftmost, topmost: coordinates (normalized) of the top-left pixel center

        """

        #---------------------------------------------------------------------------------------------


        if tile_n_point.sum() == 0:
            return torch.zeros(self.tile_info.padded_height, self.tile_info.padded_width, 3, device=self.device, dtype=torch.float32)
        

        """
        Flattens all valid entries (>=0) from tile_gaussian_list into one long vector gathered_list of length M = tile_n_point.sum().
        Builds tile_ids_for_points (same length) where entry j stores which tile the j-th gathered Gaussian belongs to.
        Computes a prefix sum (tile_n_point_accum, length T+1) so you know the segment [accum[t], accum[t+1]) inside 
        gathered_list corresponds to tile t.

        Outputs:
            gathered_list: [M] indices into the K Gaussians.
                one long list of which Gaussians to draw, flattened tile-by-tile.
                Think: "all the Gaussian indices, in tile order, no -1 padding."
            tile_ids_for_points: [M] tile ids, aligned with gathered_list.
                same length as gathered_list. For each entry in gathered_list, 
                this tells you which tile it came from.
                Think: "label for each Gaussian saying 'I belong to tile X'."
            tile_n_point_accum: [T+1] prefix sums (segment offsets per tile).

        gathered_list will contain duplicates (the same Gaussian index repeated once per tile it touches)
        tile_ids_for_points tells us which tile each occurrence belongs to
        tile_n_point_accum is the bookmark array so the renderer processes tile-by-tile
        """

        gathered_list, tile_ids_for_points, tile_n_point_accum = self.gather_gaussians_torch(tile_n_point, tile_gaussian_list)
        # compact ragged per-tile lists into one flat list + a per-element tile id array

        self.tile_gaussians = self.culling_gaussian_3d_image_space.filter_me(gathered_list.long())
        self.n_tile_gaussians = len(self.tile_gaussians.position)
        self.n_gaussians = len(self.gaussian_3ds.position)


        base_tile = self.tile_gaussians.position[..., 2].max()
        id_and_depth = self.tile_gaussians.position[..., 2].to(torch.float32) + tile_ids_for_points.to(torch.float32) * (base_tile + 1)

        """
        Multiply the tile id by (base_tile + 1) and add the depth.
            Because (base_tile+1) is bigger than any depth, all entries for tile 0 end up in range [0, base_tile], for tile 1 in 
            [base_tile + 1, 2 * (base_tile + 1)-1], tile 2 in the next block, etc. So when we sort id_and_depth, we get:
                Group by tile id (tile 0's block first, then tile 1, ...),
                Within each tile, the order is by depth (ascending, i.e., near to far).
        """

        _, sort_indices = torch.sort(id_and_depth)    
        # sort_indices reorders the flattened per-tile list so it’s grouped by tile and depth-sorted within each tile


        self.tile_gaussians = self.tile_gaussians.filter_me(sort_indices)
        # filter_me applies that permutation to all attributes (position/rgb_color/opacity/cov), keeping tensors aligned, REORDER


        """
        build the 16×16 pixel grid (in normalized image-plane coords) for each tile,
        iterate the Gaussians that overlap that tile (sorted front-to-back),
        evaluate their 2-D Gaussian at each pixel to get per-pixel alpha
        alpha-blend color front-to-back
        """


        # ---------------------------------------------------------------- Rasterize and blend


        # Framebuffer / geometry basics from your tile_info
        device = self.device
        H, W = self.tile_info.padded_height, self.tile_info.padded_width   # padded height/width (multiples of 16) for tiling
        n_tx, n_ty = self.tile_info.n_tile_x, self.tile_info.n_tile_y    # number of tiles in X and Y (each tile is 16×16)
        fx, fy = self.tile_info.focal_x, self.tile_info.focal_y  
        # focal lengths (pixels) of the current camera, already downsampled when needed (set in set_camera)
        gx, gy = self.tile_info.tile_geo_length_x, self.tile_info.tile_geo_length_y
        # gx, gy: tile size on the normalized plane (gx = 16/fx, gy = 16/fy). This converts 16 px to "x/Z, y/Z units"
        left0, top0 = self.tile_info.leftmost, self.tile_info.topmost
        # left0, top0: normalized coordinates of the top-left pixel center of the padded frame


        # Output (padded) image buffers
        rendered_image = torch.zeros(H, W, 3, dtype=torch.float32, device=device)   # RGB framebuffer (padded), initialized black


        pos = self.tile_gaussians.position     # [K,3]  (x_norm, y_norm, depth), pos is [K,3] where each row is (x_norm, y_norm, depth)
        cov = self.tile_gaussians.cov     # [K,2,2], cov is [K,2,2] the 2D covariance on the normalized plane
        col = self.tile_gaussians.rgb_color    # col is either [K,3] (plain RGB) or [K,27] if using SH. We reduce SH to the constant term c00 so [K,3]
        opa = self.tile_gaussians.opacity    # per-gaussian opacity (after sigmoid() in your project()).
        # If colors are SH-expanded (27), use the constant c00 term only
        if col.ndim == 2 and col.shape[1] == 27:
            col = col.view(-1, 3, 9)[:, :, 0]  # -> [K,3]


        # Per-pixel spacing in normalized image plan, how far normalized coordinates move per pixel step
        dx = 1.0 / fx
        dy = 1.0 / fy


        T = n_tx * n_ty


        for t in range(T):   # Loop over the tiles
            n = int(tile_n_point[t].item())         # gives how many gaussians overlap tile t. Skip if zero
            if n == 0:
                continue


            # Tile (ty,tx) -> pixel window in the padded framebuffer
            ty = t // n_tx  # row index (how many full rows fit before t)
            tx = t %  n_tx  # column index inside that row
            y0_pix = ty * 16
            y1_pix = y0_pix + 16
            x0_pix = tx * 16
            x1_pix = x0_pix + 16  # tile (ty, tx) covers pixel rows [y0_pix:y1_pix) and columns [x0_pix:x1_pix)


            # Normalized plane coords for the 16×16 pixel centers of this tile
            left_norm = left0 + tx * gx            # tile's left edge in normalized units
            top_norm = top0 + ty * gy            # tile's top edge  in normalized units

            xs = left_norm + (torch.arange(16, device=device) + 0.5) * dx  # x-centers (16)
            ys = top_norm  + (torch.arange(16, device=device) + 0.5) * dy  # y-centers (16)
            X, Y = torch.meshgrid(xs, ys, indexing="xy")  # [16,16] grid, can evaluate each Gaussian’s 2D ellipse at every pixel center inside the tile.


            # Per-tile accumulators (front-to-back "under" compositing)
            acc_rgb = torch.zeros(16, 16, 3, dtype=torch.float32, device=device)   # accumulated color for each pixel in this 16×16 tile after alpha compositing front-to-back.
            acc_alpha = torch.zeros(16, 16, dtype=torch.float32, device=device)    # acc_alpha tracks how much of the pixel is already “covered” by earlier (closer) Gaussians.


            # Gaussians for this tile live in a contiguous block after your global sort
            start = int(tile_n_point_accum[t].item())    # look up start offset (in the globally sorted arrays) for tile t
            end = start + n
            # Compute the exclusive end index for this tile’s block: if there are n entries for this tile, they occupy indices [start, start+n)
            gidx = torch.arange(start, end, device=device)  # indices into the sorted, filtered arrays


            eps = 1e-6  # numeric stability when inverting covariances


            # Rasterize gaussians for this tile, already depth-sorted near->far
            for g in gidx:     # Loop over the Gaussians that affect this tile, in the already depth-sorted order (near to far)
                # Mean and covariance in normalized image plane
                mu_x = pos[g, 0]   # Read the Gaussian’s mean position in normalized image coords (x and y)
                mu_y = pos[g, 1]
                S = cov[g] + eps * torch.eye(2, device=device)  # [2,2] (+eps for invertibility)
                invS = torch.inverse(S)


                # Offsets of all 16×16 pixel centers from the gaussian mean
                dxg = X - mu_x                     # [16,16], for every pixel centre in this tile, compute offset from the Gaussian mean
                dyg = Y - mu_y                     # [16,16]


                # Mahalanobis quadratic form: [dx dy] invS [dx dy]^T
                q11, q12 = invS[0, 0], invS[0, 1] # Extract the unique entries from the symmetric inverse covariance
                q22 = invS[1, 1]              # invS is symmetric; invS[1,0] == invS[0,1]
                quad = (q11 * dxg * dxg) + (2.0 * q12 * dxg * dyg) + (q22 * dyg * dyg)


                # Unnormalized gaussian weight per pixel
                weight = torch.exp(-0.5 * quad)    # [16,16], further means lesser weight, yields a smooth, bell-shaped contribution per pixel.


                # Per-pixel alpha from opacity × weight
                a = torch.clamp(opa[g] * weight, 0.0, 1.0)  # [16,16]

                # Convert that weight into a per-pixel alpha by multiplying the Gaussian’s opacity opa[g] (a scalar in [0,1]).
                # Clamp to [0,1] for numeric stability.


                # Front-to-back alpha compositing (premultiplied)
                one_minus = (1.0 - acc_alpha)                # transmittance so far
                acc_rgb = acc_rgb + (a * one_minus).unsqueeze(-1) * col[g]   # add color
                acc_alpha = acc_alpha + a * one_minus


            rendered_image[y0_pix:y1_pix, x0_pix:x1_pix, :] = acc_rgb


        # ----------------------------------------------------------------


        if out_write:
            img_npy = rendered_image.clip(0,1).detach().cpu().numpy()
            cv2.imwrite("test.png", (img_npy * 255).astype(np.uint8)[...,::-1])

        return rendered_image


    def forward(self, camera_id=None, extrinsics=None, intrinsics=None):
        self.set_camera(camera_id, extrinsics, intrinsics)     # pick intrinsics/extrinsics & ground-truth
        self.project_and_culling()     # drop Gaussians that won't affect the frame
        padded_render_img = self.render(out_write=False)     # draw remaining blobs
        padded_render_img = torch.clamp(padded_render_img, 0, 1) 
        ret = self.tile_info.crop(padded_render_img)    # remove padding; output H×W×3 tensor
        
        return ret
