import struct
import numpy as np
from typing import Dict, Union
from pathlib import Path
import torch
import math


class CameraModel:
    model_id: int
    model_name: str
    num_params: int


class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray


class Point3D:
    id: int
    xyz: np.ndarray
    rgb_color: np.ndarray
    error: Union[float, np.ndarray]
    image_ids: np.ndarray
    point2D_idxs: np.ndarray


class Image():
    id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray
    point3D_ids: np.ndarray


    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}


CAMERA_MODEL_IDS = dict(
    [(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS]
)


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)   


def read_cameras_bin(path_to_model_file: Union[str, Path]) -> Dict[int, Camera]:
    cameras = {}


    with open(path_to_model_file, "rb") as fid:     # open and read .bin file
        num_cameras = read_next_bytes(fid, 8, "Q")[0]   


        for useless2 in range(num_cameras):
            camera_properties = read_next_bytes(fid, num_bytes=24, format_char_sequence="iiQQ")     # 4, 4, 8, 8


            camera_id = camera_properties[0]      # an integer ID for this camera (links to images.bin)
            model_id = camera_properties[1]      # index into COLMAP's list of camera models (e.g. SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, etc.).
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name      # see class
            width = camera_properties[2]      # image width in pixels
            height = camera_properties[3]      # image height in pixels
            num_params = CAMERA_MODEL_IDS[model_id].num_params


            params = read_next_bytes(fid, num_bytes=8 * num_params, format_char_sequence="d" * num_params)    
            # intrinsic parameters, always float64. The meaning and number depend on the model


            # create the Camera instance with collected data given by COLMAP .bin file
            cameras[camera_id] = Camera(id=camera_id, model=model_name, width=width, height=height, params=np.array(params))


    return cameras


def read_images_bin(path_to_model_file: Union[str, Path]) -> Dict[int, Image]:
    images = {}


    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]


        for useless3 in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )


            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])       # ROTATION
            tvec = np.array(binary_image_properties[5:8])       # POSITION
            camera_id = binary_image_properties[8]              # WHICH CAMERA IT WAS TAKEN BY
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]


            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]


            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]


            x_y_id_s = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )

            xys = np.column_stack(
                [tuple(map(float, x_y_id_s[0::3])), tuple(map(float, x_y_id_s[1::3]))]
            )

            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))

            images[image_id] = Image(     # create object of class Image
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )

    return images


def read_points3d_bin(path_to_model_file: Union[str, Path]) -> Dict[int, Point3D]:
    points3D = {}

    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]     # because return is a tuple

        # So the fixed header per point is 43 bytes (8 + 24 + 3 + 8)
        # After that come track_length pairs of (image_id, point2D_idx)

        for useless4 in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd"
            )

            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid,
                num_bytes = 8 * track_length,
                format_char_sequence = "ii" * track_length,
            ) # When COLMAP saves a 3-D point it also records which 2-D key-points in which images see that point. That list is 
              # called the track
              # 1 image, 1 2d point (4 bytes each --> 32 bits)


            image_ids = np.array(tuple(map(int, track_elems[0::2])))    # 0::2 picks elements 0,2,4,… (the image_ids).
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2]))) # 1::2 picks elements 1,3,5,… (the feature indices).


            points3D[point3D_id] = Point3D(
                id=point3D_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs,
            )

            """
            Now downstream code can quickly look up which images and which key-points 
            participate in bundle-adjustment or—like in the Gaussian-splatting code
            : calculate per-image gradients.
            """


    return points3D


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def q2r(qvec):
    # qvec B x 4
    qvec = qvec / qvec.norm(dim=1, keepdim=True)
    rot = [
        1 - 2 * qvec[:, 2] ** 2 - 2 * qvec[:, 3] ** 2,
        2 * qvec[:, 1] * qvec[:, 2] - 2 * qvec[:, 0] * qvec[:, 3],
        2 * qvec[:, 3] * qvec[:, 1] + 2 * qvec[:, 0] * qvec[:, 2],
        2 * qvec[:, 1] * qvec[:, 2] + 2 * qvec[:, 0] * qvec[:, 3],
        1 - 2 * qvec[:, 1] ** 2 - 2 * qvec[:, 3] ** 2,
        2 * qvec[:, 2] * qvec[:, 3] - 2 * qvec[:, 0] * qvec[:, 1],
        2 * qvec[:, 3] * qvec[:, 1] - 2 * qvec[:, 0] * qvec[:, 2],
        2 * qvec[:, 2] * qvec[:, 3] + 2 * qvec[:, 0] * qvec[:, 1],
        1 - 2 * qvec[:, 1] ** 2 - 2 * qvec[:, 2] ** 2,
    ]
    rot = torch.stack(rot, dim=1).reshape(-1, 3, 3)
    return rot


def jacobian_torch(a):
    _rsqr = 1./(a[:, 0]**2 + a[:, 1]**2 + a[:, 2]**2).sqrt()
    _res = [
        1/a[:,2], torch.zeros_like(a[:,0]), -a[:,0]/(a[:,2]**2),
        torch.zeros_like(a[:,0]), 1/a[:,2], -a[:,1]/(a[:,2]**2),
        _rsqr * a[:, 0], _rsqr * a[:, 1], _rsqr * a[:, 2]
    ]
    return torch.stack(_res, dim=-1).reshape(-1, 3, 3)


"""
R(d)=c00​Y00​=RGBR​,
so we start with the correct flat colour everywhere; 
the optimiser later nudges the other 8 coefficients 
away from zero to learn view-dependence
"""
def initialize_sh(rgbs):
    sh_coeff = torch.zeros(rgbs.shape[0], 3, 9, device=rgbs.device, dtype=rgbs.dtype)
    sh_coeff[:, :, 0] = rgbs / 0.28209479177387814     # Y00​ is the constant spherical harmonic.
    return sh_coeff.flatten(1)    # --> (N, 27)
    # Flattens the last two axes so PyTorch treats the 27 values as one parameter vector per Gaussian.


def inverse_sigmoid(y):
    return -math.log(1/y  - 1)


def inverse_sigmoid_torch(y):
    return -torch.log(1/y  - 1)


def sample_two_point(gaussian_pos, gaussian_cov):
    dist = torch.distributions.multivariate_normal.MultivariateNormal(
        gaussian_pos,
        gaussian_cov,
    )
    p1 = dist.sample()
    p2 = dist.sample()
    return p1, p2
