import numpy as np

"""
TODO:
- move objective functions to utils/optim.py
- chose one method (Molnar?) and delete the others.
"""

# Conversion between photo and fiducial coordinates

def photo_to_fiducial_coordinates(xp, yp, x0, y0):
    '''
    Convert photo coordinates to fiducial coordinates using the given parameters.
    Inputs:
        x, y: Photo coordinates.
        x0, y0: Location of the optical center in the fiducial coordinate system.
    Outputs:
        xi, eta: Fiducial coordinates.
    '''
    return xp + x0, yp + y0

def fiducial_to_photo_coordinates(xi, eta, x0, y0):
    '''
    Convert fiducial coordinates to photo coordinates using the given parameters.
    Inputs:
        xi, eta: Fiducial coordinates.
        x0, y0: Location of the optical center in the fiducial coordinate system.
    Outputs:
        x, y: Photo coordinates.
    '''
    return xi - x0, eta - y0

# Conversion between image and fiducial coordinates

def image_to_fiducial_coordinates_molnar(x, y, xc, yc, alpha, delta_xi, delta_eta):
    '''
    Convert image coordinates to fiducial coordinates as explained in Molnar et al. (2021), using the given parameters.
    Inputs:
        x, y (int, float or np.ndarray(n,)): Image coordinates.
        xc, yc (in or float): Center of the fiducial coordinate system.
        alpha (int or float): Rotation angle in radians.
        delta_xi, delta_eta (int or float): size of pixel.
    Outputs:
        xi, eta (int, float or np.ndarray(n,)): Fiducial coordinates.
    '''
    rotation_matrix = np.array([
        [delta_xi * np.cos(alpha), delta_eta * np.sin(alpha)],
        [-delta_xi * np.sin(alpha), delta_eta * np.cos(alpha)]
    ])
    
    if type(x) in [float, int] and type(y) in [float, int]:
        XI = rotation_matrix @ np.array([[x - xc], [y - yc]])
        return XI[0, 0], XI[1, 0]
    elif type(x) == np.ndarray and type(y) == np.ndarray:
        XI = rotation_matrix @ np.array([x - xc, y - yc])
        return XI[0, :], XI[1, :]
    else:
        raise TypeError("x, y must be either all floats or all numpy arrays")


def fiducial_to_image_coordinates_molnar(xi, eta, xc, yc, alpha, delta_xi, delta_eta):
    '''
    Convert fiducial coordinates to image coordinates as explained in Molnar et al. (2021), using the given parameters.
    Inputs:
        xi, eta (int, float or np.ndarray(n,)): Fiducial coordinates.
        xc, yc (int of float): Center of the fiducial coordinate system.
        alpha (int or float): Rotation angle in radians.
        delta_xi, delta_eta (int or float): size of pixel.
    Outputs:
        x, y (int, float or np.ndarray(n,)): Image coordinates.
    '''
    rotation_matrix = np.array([
        [delta_xi * np.cos(alpha), delta_eta * np.sin(alpha)],
        [-delta_xi * np.sin(alpha), delta_eta * np.cos(alpha)]
    ])
    
    if type(xi) in [float, int] and type(eta) in [float, int]:
        X = np.linalg.inv(rotation_matrix) @ np.array([[xi], [eta]]) + np.array([[xc], [yc]])
        return X[0, 0], X[1, 0]
    elif type(xi) == np.ndarray and type(eta) == np.ndarray:
        X = np.linalg.inv(rotation_matrix) @ np.array([xi, eta]) + np.array([[xc], [yc]]) * np.ones((2, xi.shape[0]))
        return X[0, :], X[1, :]
    else:
        raise TypeError("xi, eta must be either all floats or all numpy arrays")


def objective_function_molnar(params, FMs_fiducial_true_coords, FMs_image_true_coords):
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
    
    xi, eta = image_to_fiducial_coordinates_molnar(
        FMs_image_true_coords[:, 0], FMs_image_true_coords[:, 1], xc, yc, alpha, delta_xi, delta_eta
    )
    FMs_fiducial_inferred_coords = np.array([xi, eta]).T
    
    res = 1/2 * np.linalg.norm(FMs_fiducial_inferred_coords - FMs_fiducial_true_coords, axis=1) ** 2
    res = np.sum(res)
    return res


def image_to_fiducial_coordinates_linear(x, y, matrix):
    '''
    Convert image coordinates to fiducial coordinates using the given parameters.
    Inputs:
        x, y (int, float or np.ndarray(n,)): Image coordinates.
        matrix np.ndarray(2, 2): Linear transformation matrix (np.array).
    Outputs:
        xi, eta (int, float or np.ndarray(n,)): Fiducial coordinates.
    '''
    if type(x) in [float, int] and type(y) in [float, int]:
        XI = matrix @ np.array([[x], [y]])
        return XI[0, 0], XI[1, 0]
    elif type(x) == np.ndarray and type(y) == np.ndarray:
        XI = matrix @ np.array([x, y])
        return XI[0, :], XI[1, :]
    else:
        raise TypeError("x, y must be either all floats or all numpy arrays")


def fiducial_to_image_coordinates_linear(xi, eta, matrix):
    '''
    Convert fiducial coordinates to image coordinates using the given parameters.
    Inputs:
        xi, eta (int, float or np.ndarray(n,)): Fiducial coordinates.
        matrix (np.ndarray(2, 2)): Linear transformation matrix used to convert from image to fiducial coordinates (np.array).
    Outputs:
        x, y (int, float or np.ndarray(n,)): Image coordinates.
    '''
    if type(xi) in [float, int] and type(eta) in [float, int]:
        X = np.linalg.inv(matrix) @ np.array([[xi], [eta]])
        return X[0, 0], X[1, 0]
    elif type(xi) == np.ndarray and type(eta) == np.ndarray:
        X = np.linalg.inv(matrix) @ np.array([xi, eta])
        return X[0, :], X[1, :]
    else:
        raise TypeError("xi, eta must be either all floats or all numpy arrays")


def objective_function_linear(matrix, FMs_fiducial_true_coords, FMs_image_true_coords):
    '''
    Objective function for the optimization process to retrieve parameters to switch from a coordinate system and the other.
    Inputs:
        matrix: Linear transformation matrix (np.array).
        FMs_fiducial_true_coords: True fiducial marks fiducial coordinates (np.array of shape p x 2, p being the number of fiducial marks).
        FMs_image_true_coords: True fiducial marks image coordinates (np.array of shape p x 2).
    Outputs:
        res: Sum of squared differences between inferred and true fiducial coordinates.
    '''
    assert FMs_fiducial_true_coords.shape == FMs_image_true_coords.shape, "Fiducial and image coordinates must have the same shape."
    assert FMs_fiducial_true_coords.shape[1] == 2, "Fiducial and image coordinates must have two columns."
    
    matrix = np.array(matrix).reshape(2, 2)
    
    xi, eta = image_to_fiducial_coordinates_linear(FMs_image_true_coords[:, 0], FMs_image_true_coords[:, 1], matrix)
    FMs_fiducial_inferred_coords = np.array([xi, eta]).T
    
    res = 1/2 * np.linalg.norm(FMs_fiducial_inferred_coords - FMs_fiducial_true_coords, axis=1) ** 2
    res = np.sum(res)
    return res


def least_squares_linear(FMs_fiducial_true_coords, FMs_image_true_coords, alpha=0, a_priori=np.eye(2)): # not working for real image # TODO: vectorize
    '''
    Compute the least squares solution to find the linear transformation matrix that maps image coordinates to fiducial coordinates.
    Inputs:
        FMs_fiducial_true_coords: True fiducial marks fiducial coordinates (np.array of shape p x 2, p being the number of fiducial marks).
        FMs_image_true_coords: True fiducial marks image coordinates (np.array of shape p x 2).
    Outputs:
        matrix: Linear transformation matrix (np.array of shape 2 x 2).
    '''
    assert FMs_fiducial_true_coords.shape == FMs_image_true_coords.shape, "Fiducial and image coordinates must have the same shape."
    assert FMs_fiducial_true_coords.shape[1] == 2, "Fiducial and image coordinates must have two columns."
    
    b = FMs_fiducial_true_coords.reshape(-1, 1)
    A = []
    for i in range(len(FMs_fiducial_true_coords)):
        A.append([FMs_image_true_coords[i, 0], FMs_image_true_coords[i, 1], 0, 0])
        A.append([0, 0, FMs_image_true_coords[i, 0], FMs_image_true_coords[i, 1]])
    A = np.array(A)
    
    if alpha <= 1e-8:
        x = np.linalg.inv(A.T @ A) @ A.T @ b
    else:
        x0 = a_priori.reshape(-1, 1)
        x = np.linalg.inv(A.T @ A + alpha * np.eye(4)) @ (A.T @ b + alpha * x0)
        
    matrix = x.reshape(-1, 2)    
    return matrix


def image_to_fiducial_coordinates_affine(x, y, matrix, translation_vector=np.array([[0], [0]])):
    '''
    Convert image coordinates to fiducial coordinates using the given parameters.
    Inputs:
        x, y (int, float or np.ndarray(n,)): Image coordinates.
        matrix (np.ndarray(2, 2)): Linear transformation matrix (np.array 2x2).
        translation_vector (np.ndarray(2, 1)): Translation vector (np.array 2x1).
    Outputs:
        xi, eta (int, float or np.ndarray(n,): Fiducial coordinates.
    '''
    if type(x) in [float, int] and type(y) in [float, int]:
        XI = matrix @ np.array([[x], [y]]) - translation_vector
        return XI[0, 0], XI[1, 0]
    elif type(x) == np.ndarray and type(y) == np.ndarray:
        XI = matrix @ np.array([x, y]) - translation_vector * np.ones((2, x.shape[0]))
        return XI[0, :], XI[1, :]
    else:
        raise TypeError("xi, eta must be either all floats or all numpy arrays")


def fiducial_to_image_coordinates_affine(xi, eta, matrix, translation_vector=np.array([[0], [0]])):
    '''
    Convert fiducial coordinates to image coordinates using the given parameters.
    Inputs:
        xi, eta (int, float or np.ndarray(n,): Fiducial coordinates.
        matrix (np.ndarray(2, 2)): Linear transformation matrix used to convert from image to fiducial coordinates (np.array 2x2).
        translation_vector (np.ndarray(2, 1)): Translation vector (np.array 2x1).
    Outputs:
        x, y (int, float or np.ndarray(n,): Image coordinates.
    '''
    if type(xi) in [float, int] and type(eta) in [float, int]:
        X = np.linalg.inv(matrix) @ (np.array([[xi], [eta]]) + translation_vector)
        return X[0, 0], X[1, 0]
    elif type(xi) == np.ndarray and type(eta) == np.ndarray:
        X = np.linalg.inv(matrix) @ (np.array([xi, eta]) + translation_vector)
        return X[0, :], X[1, :]
    else:
        raise TypeError("xi, eta must be either all floats or all numpy arrays")


def objective_function_affine(params, FMs_fiducial_true_coords, FMs_image_true_coords): # TODO: vectorize
    '''
    Objective function for the optimization process to retrieve parameters to switch from a coordinate system and the other.
    Inputs:
        m11, m12, m21, m22: Linear transformation matrix coefficients.
        xt, yt: Translation vector components.
        FMs_fiducial_true_coords: True fiducial marks fiducial coordinates (np.array of shape p x 2, p being the number of fiducial marks).
        FMs_image_true_coords: True fiducial marks image coordinates (np.array of shape p x 2).
    Outputs:
        res: Sum of squared differences between inferred and true fiducial coordinates.
    '''
    assert FMs_fiducial_true_coords.shape == FMs_image_true_coords.shape, "Fiducial and image coordinates must have the same shape."
    assert FMs_fiducial_true_coords.shape[1] == 2, "Fiducial and image coordinates must have two columns."
    
    m11, m12, m21, m22, xt, yt = params[0], params[1], params[2], params[3], params[4], params[5]
    matrix = np.array([[m11, m12], [m21, m22]])
    translation_vector = np.array([[xt], [yt]])
        
    xi, eta = image_to_fiducial_coordinates_affine(FMs_image_true_coords[:, 0], FMs_image_true_coords[:, 1], matrix, translation_vector)
    FMs_fiducial_inferred_coords = np.array([xi, eta]).T
    
    res = 1/2 * np.linalg.norm(FMs_fiducial_inferred_coords - FMs_fiducial_true_coords, axis=1) ** 2
    res = np.sum(res)
    return res