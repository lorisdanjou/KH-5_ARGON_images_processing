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
        params[6],
        params[7]
    ]
    ld_coeffs = [
        params[8],
        params[9],
        params[10],
        params[11],
        params[12],
        params[13],
    ]
    eo_params = [
        params[14],
        params[15],
        params[16],
        params[17],
        params[18],
        params[19]
    ]
    return pp, ld_coeffs, eo_params

# objective function for one image, with EO and IO parameters.
def objective_function(params, GCPs):

    # unpack parameters
    pp, ld_coeffs, eo_params = unpack_parameters(params)
    
    ## IO    
    # photo coordinates
    xp, yp = io.fiducial_to_photo_coordinates(GCPs.xi.values, GCPs.eta.values, pp[0], pp[1])
    GCPs.loc[:, ["xp", "yp"]] = np.array([xp, yp]).T
    
    # lens distortion
    # xp_p, yp_p = io.lens_distortion(GCPs.xp.values, GCPs.yp.values, ld_coeffs)
    # GCPs.loc[:, ["xp_p", "yp_p"]] = np.array([xp_p, yp_p]).T
    
    ## EO
    assert "x_gr" in GCPs.columns and "y_gr" in GCPs.columns and "z_gr" in GCPs.columns, "GCPs must contain x_gr, y_gr, z_gr columns"
    
    # collinearity equations
    
    ## compute residual
    res = 0
    return res
