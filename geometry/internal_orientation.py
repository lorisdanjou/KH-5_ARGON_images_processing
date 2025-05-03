import numpy as np

def image_to_fiducial_coordinates(x, y, xc, yc, alpha, delta_xi, delta_eta):
    '''
    Convert image coordinates to fiducial coordinates using the given parameters.
    Inputs:
        x, y: Image coordinates.
        xc, yc: Center of the fiducial coordinate system.
        alpha: Rotation angle in radians.
        delta_xi, delta_eta: size of pixel.
    Outputs:
        xi, eta: Fiducial coordinates.
    '''
    rotation_matrix = np.array([
        [delta_xi * np.cos(alpha), delta_eta * np.sin(alpha)],
        [-delta_xi * np.sin(alpha), delta_eta * np.cos(alpha)]
    ])
    
    XI = np.zeros((2, 1))
    XI = rotation_matrix @ np.array([[x - xc], [y - yc]])
    return XI[0, 0], XI[1, 0]


def fiducial_to_image_coordinates(xi, eta, xc, yc, alpha, delta_xi, delta_eta):
    '''
    Convert fiducial coordinates to image coordinates using the given parameters.
    Inputs:
        xi, eta: Fiducial coordinates.
        xc, yc: Center of the fiducial coordinate system.
        alpha: Rotation angle in radians.
        delta_xi, delta_eta: size of pixel.
    Outputs:
        x, y: Image coordinates.
    '''
    rotation_matrix = np.array([
        [delta_xi * np.cos(alpha), delta_eta * np.sin(alpha)],
        [-delta_xi * np.sin(alpha), delta_eta * np.cos(alpha)]
    ])
    
    X = np.zeros((2, 1))
    X = np.linalg.inv(rotation_matrix) @ np.array([[xi], [eta]]) + np.array([[xc], [yc]])
    return X[0, 0], X[1, 0]


def objective_function(params, FMs_fiducial_true_coords, FMs_image_true_coords):
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
        
    FMs_fiducial_inferred_coords = np.array(
        [np.array(image_to_fiducial_coordinates(X[0], X[1], xc, yc, alpha, delta_xi, delta_eta)) for X in FMs_image_true_coords]
    )
    
    res = 1/2 * np.linalg.norm(FMs_fiducial_inferred_coords - FMs_fiducial_true_coords, axis=1) ** 2
    res = np.sum(res)
    return res