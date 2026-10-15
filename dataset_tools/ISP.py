import numpy as np
from numpy.core.numeric import outer

def apply_gains(bayer_image, wb):
    """Applies white balance to Bayer an image."""
    C, _, _ = bayer_image.shape
    out = bayer_image * wb.reshape(C, 1, 1)
    return out

def raw2LRGB(bayer_image): 
    """RGBG -> linear RGB"""
    lin_rgb = np.stack([
        bayer_image[0, ...], 
        np.mean(bayer_image[[1,3], ...], axis=0), 
        bayer_image[2, ...]], axis=0)
    return lin_rgb

def apply_ccm(image, ccm):
    """Applies color correction matrices."""
    image = np.transpose(image, (1, 2, 0))
    image = image[:, :, None, :]
    ccm = ccm[None, None, :, :]
    out = np.sum(image * ccm, axis=-1)
    out = np.transpose(out, (2, 0, 1))
    return out

def gamma_compression(images, gamma=2.2):
    """Converts from linear to gamma space."""
    out = np.maximum(images, 1e-8) ** (1 / gamma)
    outs = np.clip((out * 255).astype(np.int16), 0, 255).astype(np.float32) / 255
    return outs

def process(bayer_image, wb, cam2rgb, gamma=2.2):
    """Processes Bayer RGBG image into sRGB image."""
    # White balance.
    bayer_image = apply_gains(bayer_image, wb)
    # Binning
    bayer_image = np.clip(bayer_image, 0.0, 1.0)
    image = raw2LRGB(bayer_image)
    # Color correction.
    image = apply_ccm(image, cam2rgb)
    # Gamma compression.
    image = np.clip(image, 0.0, 1.0)
    image = gamma_compression(image, gamma)
    return image

def raw2rgb(packed_raw, raw): 
    """Raw2RGB pipeline (preprocess version)"""
    wb = np.array(raw.camera_whitebalance) 
    wb /= wb[1]
    cam2rgb = get_cam2rgb_matrix(raw)
    out = process(packed_raw, wb=wb, cam2rgb=cam2rgb, gamma=2.2)
    return out

def raw2rawrgb(packed_raw, raw=None): 
    """Raw2RGB pipeline (preprocess version)"""
    # wb = np.array(raw.camera_whitebalance) 
    # wb /= wb[1]
    # cam2rgb = get_cam2rgb_matrix(raw)
    # out = process(packed_raw, wb=wb, cam2rgb=cam2rgb, gamma=2.2)
    out = packed_raw.copy()
    out[1] = (out[1] + out[3]) / 2
    out = np.clip(out, 0., 1.)
    return out[:3]

def get_cam2rgb_matrix(raw):
    xyz2cam = raw.rgb_xyz_matrix[:3, :]
    rgb2xyz = np.array([[0.4124564, 0.3575761, 0.1804375],
                    [0.2126729, 0.7151522, 0.0721750],
                    [0.0193339, 0.1191920, 0.9503041]])
    rgb2cam = np.matmul(xyz2cam, rgb2xyz)
    rgb2cam = rgb2cam / np.sum(rgb2cam, axis=-1, keepdims=True)
    cam2rgb = np.linalg.inv(rgb2cam)
    return cam2rgb

# import rawpy
# raw = rawpy.imread('./dark/2.CR2')
# cam2rgb = get_cam2rgb_matrix(raw)
# print(cam2rgb)
# print(raw.camera_whitebalance)