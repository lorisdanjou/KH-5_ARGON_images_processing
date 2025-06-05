import sys
sys.path.insert(0, "D:/OneDrive/Documents/Cours/4A/SFE/KH-5_ARGON_images_processing")

import numpy as np
import geometry.internal_orientation as gio
import geometry.external_orientation as geo


NUM_TESTS = 10


'''
Internal orientation tests
'''

def test_photo_fiducial():
    assert gio.photo_to_fiducial_coordinates(0., 0., 1., 1.) == (1., 1.)
    assert gio.fiducial_to_photo_coordinates(1., 1., 1., 1.) == (0., 0.)
    
    for _ in range(NUM_TESTS):
        xpyp = np.random.rand(2) * 127 - 63.5
        x0y0 = np.random.rand(2) * 127 - 63.5
        
        xi, eta = gio.photo_to_fiducial_coordinates(xpyp[0], xpyp[1], x0y0[0], x0y0[1])
        xp, yp = gio.fiducial_to_photo_coordinates(xi, eta, x0y0[0], x0y0[1])
        
        assert np.isclose(xpyp[0], xp, atol=1e-12)
        assert np.isclose(xpyp[1], yp, atol=1e-12)
    
def test_fiducial_image():
    for _ in range(NUM_TESTS):
        xy = np.random.rand(2) * 19000
        x, y = float(xy[0]), float(xy[1])
        params = np.random.rand(5) * np.array([19000, 19000, 2*np.pi, 127/19000, 127/19000])
        
        xi, eta = gio.image_to_fiducial_coordinates(xy[0], xy[1], *params)
        x, y = gio.fiducial_to_image_coordinates(xi, eta, *params)
        
        assert np.isclose(xy[0], x, atol=1e-12)
        assert np.isclose(xy[1], y, atol=1e-12)

# def test_lens_distortion():
#     assert False # TODO: to implement


'''
External orientation tests
'''

def test_geodetic_geocentric():
    for _ in range(NUM_TESTS):
        latlonh = np.random.rand(3) * np.array([2., 2., 1000.]) + np.array([78., 15., 0.])
        lat, lon, h = float(latlonh[0]) * np.pi/180, float(latlonh[1]) * np.pi/180, float(latlonh[2])
        
        x, y, z = geo.geodetic_to_geocentric_cartesian_coordinates(lat, lon, h)
        lat0, lon0, h0 = geo.geocentric_cartesian_to_geodetic_coordinates(x, y, z)
        assert np.isclose(lat, lat0, atol=1e-12)
        assert np.isclose(lon, lon0, atol=1e-12)
        assert np.isclose(h, h0, atol=1e-12)
        
def test_geocentric_local_cartesian():
    for _ in range(NUM_TESTS):
        xyz = np.random.rand(3) * 1.e3 + 3.e6
        x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        lat_c, lon_c = np.random.rand() * 2. + 78., np.random.rand() * 2. + 15.
        lat_c, lon_c = float(lat_c) * np.pi/180, float(lon_c) * np.pi/180
        
        x_gr, y_gr, z_gr = geo.geocentric_cartesian_to_local_cartesian_coordinates(x, y, z, lat_c, lon_c)
        x0, y0, z0 = geo.local_cartesian_to_geocentric_cartesian_coordinates(x_gr, y_gr, z_gr, lat_c, lon_c)
        
        assert np.isclose(x, x0, atol=1e-12)
        assert np.isclose(y, y0, atol=1e-12)
        assert np.isclose(z, z0, atol=1e-12)
    
# def test_collinearity_equations():
#     assert False
        
if __name__ == "__main__":
    test_photo_fiducial()
    test_fiducial_image()
    # test_lens_distortion()
    test_geodetic_geocentric()
    test_geocentric_local_cartesian()
    # test_collinearity_equations()
    print("All tests passed!")