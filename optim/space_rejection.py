import numpy as np

import geometry.internal_orientation as io
import geometry.external_orientation as eo

def pack_parameters(pp, ld_coeffs, eo_params, n_img=1):
    """
    Pack the parameters into a single list.
    Inputs:
        - pp (list or np.ndarray(2,)): Principal point coordinates [x0, y0].
        - ld_coeffs (list or np.ndarray(6,)): Lens distortion coefficients [k0, k1, k2, k3, p1, p2].
        - eo_params (list or np.ndarray(n_img * 6,)): Exterior orientation parameters for n_img images.
            Each image has 6 parameters: [xc, yc, zc, omega, phi, kappa].
        - n_img (int): Number of images.
    Outputs:
        - params (list): Packed parameters containing the focal length, lens distortion coefficients, and EO parameters.
    """
    params = [
        pp[0],
        pp[1],
        ld_coeffs[0],
        ld_coeffs[1],
        ld_coeffs[2],
        ld_coeffs[3],
        ld_coeffs[4],
        ld_coeffs[5]
    ]
    
    for i in range(n_img):
        params.append(eo_params[6*i + 0])
        params.append(eo_params[6*i + 1])
        params.append(eo_params[6*i + 2])
        params.append(eo_params[6*i + 3])
        params.append(eo_params[6*i + 4])
        params.append(eo_params[6*i + 5])
        
    return params

def unpack_parameters(params, n_img=1):
    """
    Unpack the parameters from a single list.
    Inputs:
        - params (list or np.ndarray): Parameters containing the focal length, lens distortion coefficients, and EO parameters.
        - n_img (int): Number of images.
    Outputs:
        - pp (list): Principal point coordinates [x0, y0].
        - ld_coeffs (list): Lens distortion coefficients [k0, k1, k2, k3, p1, p2].
        - eo_params (list): Exterior orientation parameters for n_img images.
            Each image has 6 parameters: [xc, yc, zc, omega, phi, kappa].
    """
    pp = [
        params[0],
        params[1]
    ]
    ld_coeffs = [
        params[2],
        params[3],
        params[4],
        params[5],
        params[6],
        params[7],
    ]
    
    eo_params = []
    for i in range(n_img):
        eo_params.append(params[8 + 6*i])   # xc
        eo_params.append(params[9 + 6*i])   # yc
        eo_params.append(params[10 + 6*i])  # zc
        eo_params.append(params[11 + 6*i])  # omega
        eo_params.append(params[12 + 6*i])  # phi
        eo_params.append(params[13 + 6*i])  # kappa

    return pp, ld_coeffs, eo_params

# objective function for one image, with EO and IO parameters.
def space_rejection_1img(params, GCPs, f):
    """
    Objective function for space rejection with one image, inspired by Molnar (2021).
    Inputs:
        - params (list or np.ndarray): Parameters containing the focal length, lens distortion coefficients, and EO parameters.
        - GCPs (pandas.DataFrame): Ground Control Points with columns 'xi', 'eta', 'x_gr', 'y_gr', 'z_gr', and 'image_id'.
        - f (float): Focal length of the camera in meters.
    Outputs:
        - res (float): The residual sum of squared differences between the observed and predicted photo coordinates.
    """

    # unpack parameters
    pp, ld_coeffs, eo_params = unpack_parameters(params)
    
    ## IO    
    # photo coordinates
    xp, yp = io.fiducial_to_photo_coordinates(GCPs.xi.values, GCPs.eta.values, pp[0], pp[1])
    GCPs.loc[:, ["xp", "yp"]] = np.array([xp, yp]).T
    
    # lens distortion
    xp_ld, yp_ld = io.correct_lens_distortion(GCPs.xp.values, GCPs.yp.values, ld_coeffs)
    GCPs.loc[:, ["xp_ld", "yp_ld"]] = np.array([xp_ld, yp_ld]).T
    
    ## EO
    assert "x_gr" in GCPs.columns and "y_gr" in GCPs.columns and "z_gr" in GCPs.columns, "GCPs must contain x_gr, y_gr, z_gr columns"
    # collinearity equations
    xp_ld_1, yp_ld_1 = eo.collinearity_equations(
        GCPs.x_gr.values, 
        GCPs.y_gr.values, 
        GCPs.z_gr.values, 
        f,
        eo_params[0],  # xc
        eo_params[1],  # yc
        eo_params[2],  # zc
        eo_params[3],  # omega
        eo_params[4],  # phi
        eo_params[5]   # kappa
    )
    
    ## compute residual
    res = 1/2 * np.linalg.norm(np.array([xp_ld, yp_ld]).T - np.array([xp_ld_1, yp_ld_1]).T, axis=1) ** 2
    res = np.sum(res)
    
    return res

def space_rejection(params, GCPs, f, n_img=1):
    
    # unpack parameters
    pp, ld_coeffs, eo_params = unpack_parameters(params, n_img=n_img)
    
    ## IO    
    # photo coordinates
    xp, yp = io.fiducial_to_photo_coordinates(GCPs.xi.values, GCPs.eta.values, pp[0], pp[1])
    GCPs.loc[:, ["xp", "yp"]] = np.array([xp, yp]).T
    
    # lens distortion
    xp_ld, yp_ld = io.correct_lens_distortion(GCPs.xp.values, GCPs.yp.values, ld_coeffs)
    GCPs.loc[:, ["xp_ld", "yp_ld"]] = np.array([xp_ld, yp_ld]).T
    
    ## EO
    assert "x_gr" in GCPs.columns and "y_gr" in GCPs.columns and "z_gr" in GCPs.columns, "GCPs must contain x_gr, y_gr, z_gr columns"
    # collinearity equations
    
    xyp_ld_1 = np.zeros((len(GCPs), 2))
    for i_image, image in enumerate(GCPs.image.value_counts().index.tolist()):
        # /!\ Assumes that GCPs are sorted by image.
        i_min, i_max = GCPs.loc[GCPs.image == image].index.min(), GCPs.loc[GCPs.image == image].index.max()
        xp_ld_1, yp_ld_1 = eo.collinearity_equations(
            GCPs.loc[GCPs.image == image].x_gr.values, 
            GCPs.loc[GCPs.image == image].y_gr.values, 
            GCPs.loc[GCPs.image == image].z_gr.values, 
            f,
            eo_params[0 + 6*i_image],  # xc
            eo_params[1 + 6*i_image],  # yc
            eo_params[2 + 6*i_image],  # zc
            eo_params[3 + 6*i_image],  # omega
            eo_params[4 + 6*i_image],  # phi
            eo_params[5 + 6*i_image]   # kappa
        )
        xyp_ld_1[i_min:i_max + 1, 0] = xp_ld_1
        xyp_ld_1[i_min:i_max + 1, 1] = yp_ld_1
        
    xp_ld_1, yp_ld_1 = xyp_ld_1[:, 0], xyp_ld_1[:, 1]
    
    ## compute residual
    res = 1/2 * np.linalg.norm(np.array([xp_ld, yp_ld]).T - np.array([xp_ld_1, yp_ld_1]).T, axis=1) ** 2
    res = np.sum(res)
    
    return res
