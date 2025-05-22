import numpy as np

def geodetic_to_geocentric_cartesian_coordinates(lat, lon, h):
    '''
    Convert ellipsoidal coordinates (latitude, longitude, height) to geocentric Cartesian coordinates (x, y, z).
    Inputs:
        lat: Latitude in radians.
        lon: Longitude in radians.
        h: Height above the ellipsoid in meters.
    Outputs:
        x, y, z: Geocentric Cartesian coordinates in meters.
    '''
    # WGS84 ellipsoid parameters
    a = 6378137.0 
    f = 1 / 298.257223563
    
    e = np.sqrt(2 * f - f**2)
    nu = a / np.sqrt(1 - e**2 * np.sin(lat)**2)
    
    x = (nu + h) * np.cos(lat) * np.cos(lon)
    y = (nu + h) * np.cos(lat) * np.sin(lon)
    z = ((1 - e**2) * nu + h) * np.sin(lat)
    return x, y, z

def geocentric_cartesian_to_geodetic_coordinates(x, y, z):
    '''
    Convert geocentric Cartesian coordinates (x, y, z) to ellipsoidal coordinates (latitude, longitude, height).
    Inputs:
        x, y, z: Geocentric Cartesian coordinates in meters.
    Outputs:
        lat: Latitude in radians.
        lon: Longitude in radians.
        h: Height above the ellipsoid in meters.
    '''
    # WGS84 ellipsoid parameters
    a = 6378137.0 
    f = 1 / 298.257223563
    
    e = np.sqrt(2 * f - f**2)
    b = a * np.sqrt(1 - e**2)
    p = np.sqrt(x**2 + y**2)
    beta = np.arctan(z / ((1 - f) * p))
    eps = e**2 / (1 - e**2)
    
    lat = np.arctan((z + eps * b * np.sin(beta)**3) / (p - e**2 * a * np.cos(beta)**3))
    lon = np.arctan(y / x)
    nu = a / np.sqrt(1 - e**2 * np.sin(lat)**2)
    h = p * np.cos(lat) + z * np.sin(lat) - (a**2/nu)
    
    return lat, lon, h


def geocentric_cartesian_to_local_cartesian_coordinates(x, y, z, lat_c, lon_c):
    '''
    Convert geocentric Cartesian coordinates (x, y, z) to local Cartesian coordinates (x_gr, y_gr, z_gr).
    Inputs:
        x, y, z (int, float or np.ndarray(n,)): Geocentric Cartesian coordinates in meters.
        lat_c (in or float): Latitude of the center point in radians.
        lon_c (int or float): Longitude of the center point in radians.
    Outputs:
        x_gr, y_gr, z_gr (int, float or np.ndarray(n,): Local Cartesian coordinates in meters.
    '''
    h_c = 0
    x_c, y_c, z_c = geodetic_to_geocentric_cartesian_coordinates(lat_c, lon_c, h_c)
    
    matrix = np.array([
        [-np.sin(lon_c), np.cos(lon_c), 0],
        [-np.sin(lat_c) * np.cos(lon_c), -np.sin(lat_c) * np.sin(lon_c), np.cos(lat_c)],
        [np.cos(lat_c) * np.cos(lon_c), np.cos(lat_c) * np.sin(lon_c), np.sin(lat_c)]
    ])
    
    if type(x) in [float, int] and type(y) in [float, int] and type(z) in [float, int]:
        X_gr = matrix @ np.array([[x - x_c], [y - y_c], [z - z_c]])
        return X_gr[0, 0], X_gr[1, 0], X_gr[2, 0]
    elif type(x) == np.ndarray and type(y) == np.ndarray and type(z) == np.ndarray:
        X_gr = matrix @ np.array([x - x_c, y - y_c, z - z_c])
        return X_gr[0, :], X_gr[1, :], X_gr[2, :]
    else:
        raise TypeError("x, y, z must be either all floats or all numpy arrays")


def local_cartesian_to_geocentric_cartesian_coordinates(x_gr, y_gr, z_gr, lat_c, lon_c):
    '''
    Convert local Cartesian coordinates (x_gr, y_gr, z_gr) to geocentric Cartesian coordinates (x, y, z).
    Inputs:
        x_gr, y_gr, z_gr (int, float or np.ndarray(n,)): Local Cartesian coordinates in meters.
        lat_c (int or float): Latitude of the center point in radians.
        lon_c (int or float): Longitude of the center point in radians.
    Outputs:
        x, y, z ((int, float or np.ndarray(n,))): Geocentric Cartesian coordinates in meters.
    '''
    h_c = 0
    x_c, y_c, z_c = geodetic_to_geocentric_cartesian_coordinates(lat_c, lon_c, h_c)
    
    matrix = np.array([
        [-np.sin(lon_c), np.cos(lon_c), 0],
        [-np.sin(lat_c) * np.cos(lon_c), -np.sin(lat_c) * np.sin(lon_c), np.cos(lat_c)],
        [np.cos(lat_c) * np.cos(lon_c), np.cos(lat_c) * np.sin(lon_c), np.sin(lat_c)]
    ])
    
    if type(x_gr) in [float, int] and type(y_gr) in [float, int] and type(z_gr) in [float, int]:
        X = np.linalg.inv(matrix) @ np.array([[x_gr], [y_gr], [z_gr]]) + np.array([[x_c], [y_c], [z_c]])
        return X[0, 0], X[1, 0], X[2, 0]
    elif type(x_gr) == np.ndarray and type(y_gr) == np.ndarray and type(z_gr) == np.ndarray:
        X = np.linalg.inv(matrix) @ np.array([x_gr, y_gr, z_gr]) + np.array([[x_c], [y_c], [z_c]]) * np.ones((3, x_gr.shape[0]))
        return X[0, :], X[1, :], X[2, :]
    raise TypeError("x, y, z must be either all floats or all numpy arrays")


def collinearity_equations(x_gr, y_gr, z_gr, f, xc, yc, zc, omega, phi, kappa):
    '''
    Calculates the photo coordinates (xp, yp) of a point in the image using the colinearity equations.
    Inputs:
        x_gr, y_gr, z_gr (int, float or np.ndarray(n,)): Local Cartesian coordinates of the point in meters.
        f (int or float): Focal length of the camera in meters.
        xc, yc, zc (int or float): Coordinates of the camera in geocentric Cartesian coordinates in meters.
        omega, phi, kappa (int or float): Rotation angles (in radians) around the x, y, and z axes respectively.
    Outputs:
        xp, yp (int, float or np.ndarray(n,): Photo coordinates of the point in the image in pixels.
    '''
    R = np.array([
        [np.cos(phi) * np.cos(kappa), -np.cos(phi) * np.sin(kappa), np.sin(phi)],
        [np.cos(omega) * np.sin(kappa) + np.sin(omega) * np.sin(phi) * np.cos(kappa), np.cos(omega) * np.cos(kappa) - np.sin(omega) * np.sin(phi) * np.sin(kappa), -np.sin(omega) * np.cos(phi)],
        [np.sin(omega) * np.sin(kappa) - np.cos(omega) * np.sin(phi) * np.cos(kappa), np.sin(omega) * np.cos(kappa) + np.cos(omega) * np.sin(phi) * np.sin(kappa), np.cos(omega) * np.cos(phi)]
    ])

    if type(x_gr) in [float, int] and type(y_gr) in [float, int] and type(z_gr) in [float, int]:
        lam = R[2, :] @ (np.array([[x_gr], [y_gr], [z_gr]]) - np.array([[xc], [yc], [zc]]))
        Xp = - f / lam * R[0:2, :] @ (np.array([[x_gr], [y_gr], [z_gr]]) - np.array([[xc], [yc], [zc]]))
        return Xp[0, 0], Xp[1, 0]
    elif type(x_gr) == np.ndarray and type(y_gr) == np.ndarray and type(z_gr) == np.ndarray:        
        lam = R[2, :] @ (np.array([x_gr, y_gr, z_gr]) - (np.array([[xc], [yc], [zc]])) * np.ones((3, x_gr.shape[0])))
        Xp = -f / lam * (R[0:2, :] @ (np.array([x_gr, y_gr, z_gr]) - (np.array([[xc], [yc], [zc]])) * np.ones((3, x_gr.shape[0]))) * np.ones((2, x_gr.shape[0])))
        return Xp[0, :], Xp[1, :]
    raise TypeError("x_gr, y_gr, z_gr must be either all floats or all numpy arrays")


def objective_function(params, f, GCPs_local_cartesian_true_coords, GCPs_photo_true_coords):
    '''
    Objective function for the optimization process.
    Inputs:
        params (list or np.ndarray(6,)): Parameters to be optimized (xc, yc, zc, omega, phi, kappa).
        f (int or float): Focal length of the camera in meters.
        GCPs_local_cartesian_true_coords (np.ndarray(n, 3)): True coordinates of the GCPs in local Cartesian coordinates.
        GCPs_photo_true_coords (np.ndarray(n, 2)): True coordinates of the fiducial marks in the image.
    Outputs:
        res (float): Residual sum of squares.
    '''
    xc, yc, zc, omega, phi, kappa = params[0], params[1], params[2], params[3], params[4], params[5]
    
    xp, yp = collinearity_equations(GCPs_local_cartesian_true_coords[:, 0], GCPs_local_cartesian_true_coords[:, 1], GCPs_local_cartesian_true_coords[:, 2], f, xc, yc, zc, omega, phi, kappa)
    GCPs_fiducial_inferred_coords = np.array([xp, yp]).T
    
    res = 1/2 * np.linalg.norm(GCPs_fiducial_inferred_coords - GCPs_photo_true_coords, axis=1) ** 2
    res = np.sum(res)
    
    return res