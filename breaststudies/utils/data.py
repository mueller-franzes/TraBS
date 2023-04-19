import numpy as np 
import SimpleITK as sitk 
from scipy.ndimage import zoom

def get_affine(image):
    # Coppied from TorchIO: 
    # https://github.com/fepegar/torchio/blob/164a1bf3699863ef3a74f2a7694f6f4cf0fff361/torchio/data/io.py#L271
    spacing = np.array(image.GetSpacing())
    direction = np.array(image.GetDirection())
    origin = image.GetOrigin()
    return get_affine2(spacing, direction, origin, lps2ras=True)

def get_affine2(spacing, direction, origin, lps2ras=False):
    spacing = np.asarray(spacing)
    direction = np.asarray(direction)
    origin = np.asarray(origin)
    if len(direction) == 9:
        rotation = direction.reshape(3, 3)
    elif len(direction) == 4:  # ignore first dimension if 2D (1, W, H, 1)
        rotation_2d = direction.reshape(2, 2)
        rotation = np.eye(3)
        rotation[:2, :2] = rotation_2d
        spacing = *spacing, 1
        origin = *origin, 0
    else:
        raise RuntimeError(f'Direction not understood: {direction}')
    flip_xy = np.diag((-1, -1, 1)) if lps2ras else np.diag((1, 1, 1)) # used to switch between LPS and RAS
    rotation = np.dot(flip_xy, rotation)
    rotation_zoom = rotation * spacing
    translation = np.dot(flip_xy, origin)
    affine = np.eye(4)
    affine[:3, :3] = rotation_zoom
    affine[:3, 3] = translation
    return affine 

def sitk_resample_to_shape(img, x, y, z, order=3):
    """
    Resamples Image to given shape

    Parameters
    ----------
    img : SimpleITK.Image
    x : int
        shape in x-direction
    y : int
        shape in y-direction
    z : int
        shape in z-direction
    order : int
        interpolation order

    Returns
    -------
    SimpleITK.Image
        Resampled Image

    """
    if img.GetSize() != (x, y, z):
        img_np = sitk.GetArrayFromImage(img)
        zoom_fac_z = z / img_np.shape[0]
        zoom_fac_y = y / img_np.shape[1]
        zoom_fac_x = x / img_np.shape[2]
        img_np_fixed_size = zoom(img_np,
                                 [zoom_fac_z,
                                  zoom_fac_y,
                                  zoom_fac_x],
                                 order=order)
        img_resampled = sitk.GetImageFromArray(img_np_fixed_size)
        img_resampled = sitk_copy_metadata(img, img_resampled)
        img_resampled.SetDirection(img.GetDirection())
        img_resampled.SetOrigin(img.GetOrigin())

        spacing_x = img.GetSpacing()[0] * (1 + 1 - (zoom_fac_x))
        spacing_y = img.GetSpacing()[1] * (1 + 1 - (zoom_fac_y))
        spacing_z = img.GetSpacing()[2] * (1 + 1 - (zoom_fac_z))
        img_resampled.SetSpacing((spacing_x, spacing_y, spacing_z))
        return img_resampled
    else:
        return img



def sitk_copy_metadata(img_source, img_target):
    """ Copy metadata (=DICOM Tags) from one image to another

    Parameters
    ----------
    img_source : SimpleITK.Image
        Source image
    img_target : SimpleITK.Image
        Source image

    Returns
    -------
    SimpleITK.Image
        Target image with copied metadata
    """
    for k in img_source.GetMetaDataKeys():
        img_target.SetMetaData(k, img_source.GetMetaData(k))
    return img_target