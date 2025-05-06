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


def image_to_fiducial_coordinates_linear(x, y, matrix):
    '''
    Convert image coordinates to fiducial coordinates using the given parameters.
    Inputs:
        x, y: Image coordinates.
        matrix: Linear transformation matrix (np.array).
    Outputs:
        xi, eta: Fiducial coordinates.
    '''
    XI = np.zeros((2, 1))
    XI = matrix @ np.array([[x], [y]])
    return XI[0, 0], XI[1, 0]


def fiducial_to_image_coordinates_linear(xi, eta, matrix):
    '''
    Convert fiducial coordinates to image coordinates using the given parameters.
    Inputs:
        xi, eta: Fiducial coordinates.
        matrix: Linear transformation matrix used to convert from image to fiducial coordinates (np.array).
    Outputs:
        x, y: Image coordinates.
    '''
    X = np.zeros((2, 1))
    X = np.linalg.inv(matrix) @ np.array([[xi], [eta]])
    return X[0, 0], X[1, 0]


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
        
    FMs_fiducial_inferred_coords = np.array(
        [np.array(image_to_fiducial_coordinates_linear(X[0], X[1], matrix)) for X in FMs_image_true_coords]
    )
    
    res = 1/2 * np.linalg.norm(FMs_fiducial_inferred_coords - FMs_fiducial_true_coords, axis=1) ** 2
    res = np.sum(res)
    return res


def least_squares_linear(FMs_fiducial_true_coords, FMs_image_true_coords, alpha=0, a_priori=np.eye(2)):
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
    
    # print("cond = ", np.linalg.norm(np.linalg.inv(A.T @ A)) * np.linalg.norm(A.T @ A))
    
    return matrix