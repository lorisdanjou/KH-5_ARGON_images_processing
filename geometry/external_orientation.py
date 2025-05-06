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