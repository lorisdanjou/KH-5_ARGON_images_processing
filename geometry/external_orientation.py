import numpy as np

def ellipsoidal_to_geocentric_cartesian_coordinates(lat, lon, h):
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

def geocentric_cartesian_to_ellipsoidal_coordinates(x, y, z):
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
        x, y, z: Geocentric Cartesian coordinates in meters.
        lat_c: Latitude of the center point in radians.
        lon_c: Longitude of the center point in radians.
    Outputs:
        x_gr, y_gr, z_gr: Local Cartesian coordinates in meters.
    '''
    h_c = 0
    x_c, y_c, z_c = ellipsoidal_to_geocentric_cartesian_coordinates(lat_c, lon_c, h_c)
    
    matrix = np.array([
        [-np.sin(lon_c), np.cos(lon_c), 0],
        [-np.sin(lat_c) * np.cos(lon_c), -np.sin(lat_c) * np.sin(lon_c), np.cos(lat_c)],
        [np.cos(lat_c) * np.cos(lon_c), np.cos(lat_c) * np.sin(lon_c), np.sin(lat_c)]
    ])
    
    X_gr = np.zeros((3, 1))
    X_gr = matrix @ np.array([[x - x_c], [y - y_c], [z - z_c]])
    return X_gr[0, 0], X_gr[1, 0], X_gr[2, 0]


def local_cartesian_to_geocentric_cartesian_coordinates(x_gr, y_gr, z_gr, lat_c, lon_c):
    '''
    Convert local Cartesian coordinates (x_gr, y_gr, z_gr) to geocentric Cartesian coordinates (x, y, z).
    Inputs:
        x_gr, y_gr, z_gr: Local Cartesian coordinates in meters.
        lat_c: Latitude of the center point in radians.
        lon_c: Longitude of the center point in radians.
    Outputs:
        x, y, z: Geocentric Cartesian coordinates in meters.
    '''
    h_c = 0
    x_c, y_c, z_c = ellipsoidal_to_geocentric_cartesian_coordinates(lat_c, lon_c, h_c)
    
    matrix = np.array([
        [-np.sin(lon_c), np.cos(lon_c), 0],
        [-np.sin(lat_c) * np.cos(lon_c), -np.sin(lat_c) * np.sin(lon_c), np.cos(lat_c)],
        [np.cos(lat_c) * np.cos(lon_c), np.cos(lat_c) * np.sin(lon_c), np.sin(lat_c)]
    ])
    
    X = np.zeros((3, 1))
    X = np.linalg.inv(matrix) @ np.array([[x_gr], [y_gr], [z_gr]]) + np.array([[x_c], [y_c], [z_c]])
    return X[0, 0], X[1, 0], X[2, 0]