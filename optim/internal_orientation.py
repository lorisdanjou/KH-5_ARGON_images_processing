import numpy as np
import geometry.internal_orientation as io

def objective_function(params, FMs_fiducial_true_coords, FMs_image_true_coords): # TODO: rename and or move to space rejection.py
    '''
    Objective function for the optimization process to retrieve parameters to switch from a coordinate system and the other.
    Inputs:
        params: Parameters to optimize (xc, yc, alpha, delta_xi, delta_eta).
        FMs_fiducial_true_coords: True fiducial marks fiducial coordinates (np.array of shape p x 2, p being the number of fiducial marks).
        FMs_image_true_coords: True fiducial marks image coordinates (np.array of shape p x 2).
    Outputs:
        res: Sum of squared differences between inferred and true fiducial coordinates.
    '''
    assert FMs_fiducial_true_coords.shape == FMs_image_true_coords.shape, "Fiducial and image coordinates must have the same shape."
    assert FMs_fiducial_true_coords.shape[1] == 2, "Fiducial and image coordinates must have two columns."
    
    xc, yc, alpha, delta_xi, delta_eta = params[0], params[1], params[2], params[3], params[4]
    
    xi, eta = io.image_to_fiducial_coordinates(
        FMs_image_true_coords[:, 0], FMs_image_true_coords[:, 1], xc, yc, alpha, delta_xi, delta_eta
    )
    FMs_fiducial_inferred_coords = np.array([xi, eta]).T
    
    res = 1/2 * np.linalg.norm(FMs_fiducial_inferred_coords - FMs_fiducial_true_coords, axis=1) ** 2
    res = np.sum(res)
    return res