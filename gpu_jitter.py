from __future__ import division
import kornia
import torch
import random
from typing import Tuple, Optional
import config
#lhl mode rib fpr
if 'rib' in config.data_mode:
    class GPURotate(object):
        def __init__(self):
            self.rot_max_deg = 90.
            self.rot_std_deg = 15.
            self.scale_max = 0.2
            # self.scale_std = 0.05
            self.scale_std = 0.15
            self.p_art_sys = 0.2
            self.padding_modes = ['zeros', 'border', 'reflection']
            self.sheer_range = 0.1

        def _sample_padding_mode(self):
            return random.choice(self.padding_modes)

        def _sample_angle(self, batch):
            n = batch.size(0)
            rot_max_deg = 15 if self.artefact_sys else self.rot_max_deg
            #rot_std_deg = 5 if self.artefact_sys else self.rot_std_deg
            degs = torch.randint(low =0, high = 180, size=(n,),
                                             dtype=batch.dtype,
                                             device=batch.device) - 90


            return degs.clamp_(-rot_max_deg, rot_max_deg)

        def _sample_scale(self, batch):

            n = batch.size(0)
            if self.artefact_sys:
                return torch.ones(size=(n,), dtype=batch.dtype, device=batch.device)
            scales = 0.4 * torch.rand(size=(n,),
                                                  dtype=batch.dtype,
                                                  device=batch.device) - 0.2
            # return (1. - scales).clamp_(0.5, 2)
            return 1. - scales.clamp_(-self.scale_max, self.scale_max)

        def _sample_artefact_sys(self):
            self.artefact_sys = random.random() < self.p_art_sys

        def _get_center(self, batch):
            n, _, h, w = batch.size()
            centers = batch.new_empty(size=(n, 2))
            centers[:, 0] = w / 2
            centers[:, 1] = h / 2
            return centers

        def _get_translations(self, batch):
            n = batch.size(0)
            return torch.zeros(size=(n, 2), dtype=batch.dtype, device=batch.device)

        def _get_sheer_factor(self, batch):
            n = batch.size(0)
            sheer_factor = (torch.rand(size=(n,), dtype=batch.dtype, device=batch.device) - 0.5) * 2 * self.sheer_range
            return sheer_factor


        def __call__(self, batch):
            self._sample_artefact_sys()
            # compute the transformation matrix
            # M = kornia.get_rotation_matrix2d(self._get_center(batch),
            #                                  self._sample_angle(batch),
            #                                  self._sample_scale(batch))
            scale_one = self._sample_scale(batch).view(-1, 1)
            #scale_two = self._sample_scale(batch).view(-1, 1)
            scale_2use = torch.cat((scale_one, scale_one), dim =1)
            M = get_rotation_matrix2d(self._get_center(batch),
                                      self._sample_angle(batch),
                                      scale_2use
                                      )
            # M = get_affine_matrix2d(  translations = self._get_translations(batch),
            #                           center= self._get_center(batch),
            #                           angle = self._sample_angle(batch),
            #                           scale = scale_2use,
            #                           sx =self._get_sheer_factor(batch),
            #                           sy =self._get_sheer_factor(batch),
            #                           )

            # apply the transformation to original image
            _, _, h, w = batch.size()
            out = kornia.warp_affine(batch, M, dsize=(h, w),
                                     padding_mode=self._sample_padding_mode())
            if self.artefact_sys:
                out = 0.5 * (out + batch)
            return out

#lhl mode density
if 'density' in config.data_mode:
    class GPURotate(object):
        def __init__(self):
            self.rot_max_deg = 360.
            self.rot_std_deg = 15.
            self.scale_max = 0.15
            # self.scale_std = 0.05
            self.scale_std = 0.15
            self.p_art_sys = 0.2
            self.padding_modes = ['zeros', 'border', 'reflection']
            self.sheer_range = 0.1

        def _sample_padding_mode(self):
            return random.choice(self.padding_modes)

        def _sample_angle(self, batch):
            n = batch.size(0)
            rot_max_deg = 15 if self.artefact_sys else self.rot_max_deg
            #rot_std_deg = 5 if self.artefact_sys else self.rot_std_deg
            degs = torch.randint(low =0, high = 360, size=(n,),
                                             dtype=batch.dtype,
                                             device=batch.device) - 90


            return degs.clamp_(-rot_max_deg, rot_max_deg)

        def _sample_scale(self, batch):
            n = batch.size(0)
            if self.artefact_sys:
                return torch.ones(size=(n,), dtype=batch.dtype, device=batch.device)
            scales = self.scale_max * 2 * torch.rand(size=(n,),
                                                  dtype=batch.dtype,
                                                  device=batch.device) - self.scale_max
            # return (1. - scales).clamp_(0.5, 2)
            return 1. - scales.clamp_(-self.scale_max, self.scale_max)

        def _sample_artefact_sys(self):
            self.artefact_sys = random.random() < self.p_art_sys

        def _get_center(self, batch):
            n, _, h, w = batch.size()
            centers = batch.new_empty(size=(n, 2))
            centers[:, 0] = w / 2
            centers[:, 1] = h / 2
            return centers

        def _get_translations(self, batch):
            n = batch.size(0)
            return torch.zeros(size=(n, 2), dtype=batch.dtype, device=batch.device)

        def _get_sheer_factor(self, batch):
            n = batch.size(0)
            sheer_factor = (torch.rand(size=(n,), dtype=batch.dtype, device=batch.device) - 0.5) * 2 * self.sheer_range
            return sheer_factor


        def __call__(self, batch):
            self._sample_artefact_sys()
            # compute the transformation matrix
            # M = kornia.get_rotation_matrix2d(self._get_center(batch),
            #                                  self._sample_angle(batch),
            #                                  self._sample_scale(batch))
            scale_one = self._sample_scale(batch).view(-1, 1)
            #scale_two = self._sample_scale(batch).view(-1, 1)
            scale_2use = torch.cat((scale_one, scale_one), dim =1)
            M = get_rotation_matrix2d(self._get_center(batch),
                                      self._sample_angle(batch),
                                      scale_2use
                                      )
            # M = get_affine_matrix2d(  translations = self._get_translations(batch),
            #                           center= self._get_center(batch),
            #                           angle = self._sample_angle(batch),
            #                           scale = scale_2use,
            #                           sx =self._get_sheer_factor(batch),
            #                           sy =self._get_sheer_factor(batch),
            #                           )

            # apply the transformation to original image
            _, _, h, w = batch.size()
            out = kornia.warp_affine(batch, M, dsize=(h, w),
                                     padding_mode=self._sample_padding_mode())
            if self.artefact_sys:
                out = 0.5 * (out + batch)
            return out



def get_rotation_matrix2d(
        center: torch.Tensor,
        angle: torch.Tensor,
        scale: torch.Tensor) -> torch.Tensor:
    r"""Calculates an affine matrix of 2D rotation.

    The function calculates the following matrix:

    .. math::
        \begin{bmatrix}
            \alpha & \beta & (1 - \alpha) \cdot \text{x}
            - \beta \cdot \text{y} \\
            -\beta & \alpha & \beta \cdot \text{x}
            + (1 - \alpha) \cdot \text{y}
        \end{bmatrix}

    where

    .. math::
        \alpha = \text{scale} \cdot cos(\text{angle}) \\
        \beta = \text{scale} \cdot sin(\text{angle})

    The transformation maps the rotation center to itself
    If this is not the target, adjust the shift.

    Args:
        center (Tensor): center of the rotation in the source image.
        angle (Tensor): rotation angle in degrees. Positive values mean
            counter-clockwise rotation (the coordinate origin is assumed to
            be the top-left corner).
        scale (Tensor): isotropic scale factor.

    Returns:
        Tensor: the affine matrix of 2D rotation.

    Shape:
        - Input: :math:`(B, 2)`, :math:`(B)` and :math:`(B)`
        - Output: :math:`(B, 2, 3)`

    Example:
        >>> center = torch.zeros(1, 2)
        >>> scale = torch.ones(1)
        >>> angle = 45. * torch.ones(1)
        >>> M = kornia.get_rotation_matrix2d(center, angle, scale)
        tensor([[[ 0.7071,  0.7071,  0.0000],
                 [-0.7071,  0.7071,  0.0000]]])
    """
    if not torch.is_tensor(center):
        raise TypeError("Input center type is not a torch.Tensor. Got {}"
                        .format(type(center)))
    if not torch.is_tensor(angle):
        raise TypeError("Input angle type is not a torch.Tensor. Got {}"
                        .format(type(angle)))
    if not torch.is_tensor(scale):
        raise TypeError("Input scale type is not a torch.Tensor. Got {}"
                        .format(type(scale)))
    if not (len(center.shape) == 2 and center.shape[1] == 2):
        raise ValueError("Input center must be a Bx2 tensor. Got {}"
                         .format(center.shape))
    if not len(angle.shape) == 1:
        raise ValueError("Input angle must be a B tensor. Got {}"
                         .format(angle.shape))
    if not len(scale.shape) == 2:
        raise ValueError("Input scale must be a Bx2 tensor. Got {}"
                         .format(scale.shape))
    if not (center.shape[0] == angle.shape[0] == scale.shape[0]):
        raise ValueError("Inputs must have same batch size dimension. Got center {}, angle {} and scale {}"
                         .format(center.shape, angle.shape, scale.shape))
    # convert angle and apply scale
    rotation: torch.Tensor = kornia.angle_to_rotation_matrix(angle) #* scale.view(-1, 1, 1)
    scale_matrix: torch.Tensor = torch.zeros((scale.shape[0], 3, 3)).to(center.device)
    scale_matrix[:, 0, 0] = scale[:, 0]
    scale_matrix[:, 1, 1] = scale[:, 1]
    scale_matrix[:, 2, 2] = 1.

    alpha: torch.Tensor = rotation[:, 0, 0]
    beta: torch.Tensor = rotation[:, 0, 1]

    # unpack the center to x, y coordinates
    x: torch.Tensor = center[..., 0]
    y: torch.Tensor = center[..., 1]

    # create output tensor
    batch_size: int = center.shape[0]
    one = torch.tensor(1.).to(center.device)
    M: torch.Tensor = torch.zeros(
        batch_size, 3, 3, device=center.device, dtype=center.dtype)
    M[..., 0:2, 0:2] = rotation
    M[..., 0, 2] = (one - alpha) * x - beta * y
    M[..., 1, 2] = beta * x + (one - alpha) * y
    M[..., 2, 2] = 1
    M = torch.bmm(M, scale_matrix)
    M = M[..., :2, :]
    return M

def get_affine_matrix2d(translations: torch.Tensor, center: torch.Tensor, scale: torch.Tensor, angle: torch.Tensor,
                        sx: Optional[torch.Tensor] = None, sy: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""Composes affine matrix from the components.

    Args:
        translations (torch.Tensor): tensor containing the translation vector with shape :math:`(B, 2)`.
        center (torch.Tensor): tensor containing the center vector with shape :math:`(B, 2)`.
        scale (torch.Tensor): tensor containing the scale factor with shape :math:`(B)`.
        sx (torch.Tensor, optional): tensor containing the shear factor in the x-direction with shape :math:`(B)`.
        sy (torch.Tensor, optional): tensor containing the shear factor in the y-direction with shape :math:`(B)`.

    Returns:
        torch.Tensor: the affine transformation matrix :math:`(B, 2, 3)`.
    """

    transform: torch.Tensor = get_rotation_matrix2d(center, -angle, scale)
    transform[..., 2] += translations  # tx/ty
    # pad transform to get Bx3x3
    transform_h = kornia.convert_affinematrix_to_homography(transform)

    if sx is not None or sy is not None:
        x, y = torch.split(center, 1, dim=-1)
        x = x.view(-1)
        y = y.view(-1)
        sx_tan = torch.tan(sx)  # type: ignore
        sy_tan = torch.tan(sy)  # type: ignore
        zeros = torch.zeros_like(sx)  # type: ignore
        ones = torch.ones_like(sx)  # type: ignore
        shear_mat = torch.stack([ones, -sx_tan, sx_tan * y,  # type: ignore   # noqa: E241
                                 -sy_tan, ones + sx_tan * sy_tan, sy_tan * (sx_tan * y + x)],  # noqa: E241
                                dim=-1).view(-1, 2, 3)
        shear_mat = kornia.convert_affinematrix_to_homography(shear_mat)
        transform_h = transform_h @ shear_mat
    return transform_h[..., :2, :]


class GPUCrop(object):
    def __init__(self, is_training=True):
        self.target_size = 224
        self.crop_scale = 0.85
        self.is_training = is_training

    def _sample_crop_params_1d(self, batch, scaler):
        n = batch.size(0)
        t = 1 - self.crop_scale
        if self.is_training:
            u = torch.rand(size=(n,), dtype=batch.dtype, device=batch.device)
        else:
            u = torch.full(size=(n,), fill_value=0.5, dtype=batch.dtype, device=batch.device)
        start = u * t
        return start * scaler, (start + self.crop_scale) * scaler

    def _sample_crop_boxes(self, batch):
        _, _, h, w = batch.shape
        x1, x2 = self._sample_crop_params_1d(batch, w)
        y1, y2 = self._sample_crop_params_1d(batch, h)  # N,
        boxes = torch.stack(
            [
                torch.stack([x1, y1], dim=1),  # Nx2, top-left
                torch.stack([x2, y1], dim=1),  # Nx2, top-right
                torch.stack([x2, y2], dim=1),  # Nx2, bottom-right
                torch.stack([x1, y2], dim=1),  # Nx2, bottom-left
            ], dim=1
        )  # Nx4x2
        return boxes

    def __call__(self, batch):
        boxes = self._sample_crop_boxes(batch)
        return kornia.crop_and_resize(batch, boxes,
                                      (self.target_size, self.target_size))

#pathology
if 'pathology' in config.data_mode:
    class GPUCutout(object):
        def __init__(self):
            self.int_scaler = 1.5
            self.p_cutout = 0.5
            #self.box_sizes = [6, 12, 18, 24]
            self.box_sizes = [4, 8, 12, 16]

        def _sample_min_1d(self, dim):
            sz = random.choice(self.box_sizes)
            return random.randint(0, dim - sz), sz

        def _cutout_one(self, img, intensity):
            if random.random() > self.p_cutout:  # only apply cutout with p_cutout
                return
            d, h, w = img.size()
            z0, cd = self._sample_min_1d(d)
            y0, ch = self._sample_min_1d(h)
            x0, cw = self._sample_min_1d(w)

            img[z0:z0+cd, y0:y0 + ch, x0:x0 + cw] = intensity

        # @staticmethod
        def _sample_cutout_intensity(self, batch):
            n = batch.size(0)
            vmin, vmax = batch.mean(), batch.max()
            return vmin + (vmax * self.int_scaler - vmin) * torch.rand(size=(n,),
                                                     dtype=batch.dtype,
                                                     device=batch.device).clamp_(min=0.5)

        def __call__(self, batch):
            intensities = self._sample_cutout_intensity(batch)
            for img, intensity in zip(batch, intensities):
                self._cutout_one(img, intensity)
            return batch

#guzhe
if 'rib' in config.data_mode:
    class GPUCutout(object):
        def __init__(self):
            self.int_scaler = 1.5
            self.p_cutout = 0.5
            self.box_sizes = [6, 12, 18, 24]

        def _sample_min_1d(self, dim):
            sz = random.choice(self.box_sizes)
            return random.randint(0, dim - sz), sz

        def _cutout_one(self, img, intensity):
            if random.random() > self.p_cutout:  # only apply cutout with p_cutout
                return
            _, h, w = img.size()
            y0, ch = self._sample_min_1d(h)
            x0, cw = self._sample_min_1d(w)
            img[:, y0:y0 + ch, x0:x0 + cw] = intensity

        # @staticmethod
        def _sample_cutout_intensity(self, batch):
            n = batch.size(0)
            vmin, vmax = batch.mean(), batch.max()
            return vmin + (vmax * self.int_scaler - vmin) * torch.rand(size=(n,),
                                                     dtype=batch.dtype,
                                                     device=batch.device).clamp_(min=0.5)

        def __call__(self, batch):
            intensities = self._sample_cutout_intensity(batch)
            for img, intensity in zip(batch, intensities):
                self._cutout_one(img, intensity)
            return batch

#density
if 'density' in config.data_mode:
    class GPUCutout(object):
        def __init__(self):
            self.int_scaler = 1.5
            self.p_cutout = 0.5
            #self.box_sizes = [6, 12, 18, 24]
            self.box_sizes = [2, 4, 8, 12]

        def _sample_min_1d(self, dim):
            sz = random.choice(self.box_sizes)
            return random.randint(0, dim - sz), sz

        def _cutout_one(self, img, intensity):
            if random.random() > self.p_cutout:  # only apply cutout with p_cutout
                return
            d, h, w = img.size()
            z0, cd = self._sample_min_1d(d)
            y0, ch = self._sample_min_1d(h)
            x0, cw = self._sample_min_1d(w)

            img[z0:z0+cd, y0:y0 + ch, x0:x0 + cw] = intensity

        # @staticmethod
        def _sample_cutout_intensity(self, batch):
            n = batch.size(0)
            vmin, vmax = batch.mean(), batch.max()
            return vmin + (vmax * self.int_scaler - vmin) * torch.rand(size=(n,),
                                                     dtype=batch.dtype,
                                                     device=batch.device).clamp_(min=0.5)

        def __call__(self, batch):
            intensities = self._sample_cutout_intensity(batch)
            for img, intensity in zip(batch, intensities):
                self._cutout_one(img, intensity)
            return batch

class Compose(object):
    def __init__(self, transforms):
        self.train_transforms = transforms
        if any(isinstance(t, GPUCrop) for t in self.train_transforms):
            self.test_transforms = [GPUCrop(is_training=False)]
        else:
            self.test_transforms = []

    def __call__(self, data, is_training=True):
        # pre proc
        #is_geo_data = isinstance(data, GeoData)
        #img = data.x if is_geo_data else data
        img = data
        need_squeeze = img.ndimension() == 5

        if need_squeeze:
            img = img.squeeze_(1)  # NxCxHxW
        # n, c, h, w = img.shape
        # if is_geo_data:  # reshape such that all images use the same transform
        #     img = img.reshape(1, n * c, h, w)

        # do transforms
        transforms = self.train_transforms if is_training else self.test_transforms
        for t in transforms:
            img = t(img)

        # post_proc
        # if is_geo_data:
        #     img = img.reshape(n, c, h, w)
        if need_squeeze:
            img = img.unsqueeze_(1)
        # if is_geo_data:
        #     data.x = img
        #     img = data
        return img


#guzhe
if 'rib' in config.data_mode:
    def build_gpu_jitter():
        cutout = GPUCutout()
        rotate = GPURotate()
        # crop = GPUCrop()
        return Compose([cutout, rotate])

#nodule pathology density
if 'density' in config.data_mode:
    def build_gpu_jitter():
        cutout = GPUCutout()
        rotate = GPURotate()
        # crop = GPUCrop()
        return Compose([cutout])
if 'pathology' in config.data_mode:
    def build_gpu_jitter():
        cutout = GPUCutout()
        #rotate = GPURotate()
        # crop = GPUCrop()
        return Compose([cutout])
