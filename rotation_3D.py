from __future__ import division
import numpy as np
import cv2
from PIL import Image
import scipy.misc as misc
from scipy import ndimage
import SimpleITK as sitk


def rotateZMatrix(radians, zoom=1):
    """ Return matrix for rotating about the z-axis by 'radians' radians 
        Here remember we set axis as (z,y,x) because order of array"""
    radians = radians / 180 * np.pi
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[zoom, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def rotateYMatrix(radians, zoom=1):
    """ Return matrix for rotating about the z-axis by 'radians' radians 
        Here remember we set axis as (z,y,x) because order of array"""
    radians = radians / 180 * np.pi
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[c, 0, -s],
                     [0, zoom, 0],
                     [s, 0, c]])


def rotateXMatrix(radians, zoom=1):
    """ Return matrix for rotating about the z-axis by 'radians' radians 
        Here remember we set axis as (z,y,x) because order of array"""
    radians = radians / 180 * np.pi
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, zoom]])


def rotateMatrix(R_x, R_y, R_z, zoom):
    trans_M = np.dot(rotateZMatrix(R_z, zoom[2]), rotateYMatrix(R_y, zoom[1]))
    trans_M = np.dot(trans_M, rotateXMatrix(R_x, zoom[0]))
    return trans_M


def rotation3d(img, R_x, R_y, R_z, zoom=(1, 1, 1), shear=(0, 0, 0, 0, 0, 0)):
    centre_in = 0.5 * np.array(img.shape)
    centre_out = centre_in
    rot = np.dot(rotateMatrix(R_x, R_y, R_z, zoom), shearMatrix(shear))
    offset = centre_in - centre_out.dot(rot)

    img_roted = ndimage.affine_transform(
        img, rot.T, order=3, offset=offset, mode='nearest'
    )
    return img_roted


def rotation3d_itk(img, R_x, R_y, R_z, zoom=(1, 1, 1), shear=(0, 0, 0, 0, 0, 0)):
    centre_in = 0.5 * np.array(img.shape[::-1])
    rot = np.dot(rotateMatrix(R_x, R_y, R_z, zoom), shearMatrix(shear))

    img = sitk.GetImageFromArray(img)
    affine = sitk.AffineTransform(3)
    affine.SetMatrix(np.reshape(rot.T, -1)[::-1])
    centering_transform = sitk.TranslationTransform(3)
    centering_transform.SetOffset(np.array(affine.GetInverse().TransformPoint(centre_in) - centre_in))

    all_transform = sitk.Transform(affine)
    all_transform.AddTransform(centering_transform)
    img = sitk.Resample(img, all_transform)
    img = sitk.GetArrayFromImage(img)

    return img


def shearMatrix(shear=[0, 0, 0, 0, 0, 0]):
    hyx, hzx, hxy, hzy, hxz, hyz = shear
    return np.array([[1, hyx, hzx],
                     [hxy, 1, hzy],
                     [hxz, hyz, 1]])


def shear3d(img, hyx, hzx, hxy, hzy, hxz, hyz):
    centre_in = 0.5 * np.array(img.shape)
    centre_out = centre_in
    shear = shearMatrix(hyx, hzx, hxy, hzy, hxz, hyz)
    offset = centre_in - centre_out.dot(shear)

    img_sheared = ndimage.affine_transform(
        img, shear.T, order=3, offset=offset, mode='nearest'
    )
    return img_sheared
