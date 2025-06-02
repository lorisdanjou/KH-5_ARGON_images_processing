import numpy as np

import geometry.internal_orientation as io
import geometry.external_orientation as eo

def pack_parameters(pp, ld_coeffs, eo_params):
    """
    Pack the parameters into a single list.
    """
    params = [
        pp[0],
        pp[1],
        ld_coeffs[0],
        ld_coeffs[1],
        ld_coeffs[2],
        ld_coeffs[3],
        ld_coeffs[4],
        ld_coeffs[5],
        eo_params[0],
        eo_params[1],
        eo_params[2],
        eo_params[3],
        eo_params[4],
        eo_params[5]
    ]
    return params

def unpack_parameters(params):
    """
    Unpack the parameters from a single list.
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
    eo_params = [
        params[8],
        params[9],
        params[10],
        params[11],
        params[12],
        params[13]
    ]
    return pp, ld_coeffs, eo_params

# objective function for one image, with EO and IO parameters.
def space_rejection_1img(params, GCPs, f):

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
