import numpy as np

def p(X, Y, Z, coeffs):
    return coeffs[0] + \
        coeffs[1] * Y + \
        coeffs[2] * X + \
        coeffs[3] * Z + \
        coeffs[4] * Y * X + \
        coeffs[5] * Y * Z + \
        coeffs[6] * X * Z + \
        coeffs[7] * Y**2 + \
        coeffs[8] * X**2 + \
        coeffs[9] * Z**2 + \
        coeffs[10] * X * Y * Z + \
        coeffs[11] * Y**3 + \
        coeffs[12] * Y * X**2 + \
        coeffs[13] * Y * Z**2 + \
        coeffs[14] * Y**2 * X + \
        coeffs[15] * X**3 + \
        coeffs[16] * X * Z**2 + \
        coeffs[17] * Y**2 * Z + \
        coeffs[18] * X**2 * Z + \
        coeffs[19] * Z**3
        
def RPC(X, Y, Z, coeffs_1, coeffs_2, coeffs_3, coeffs_4):
    x, y = p(X, Y, Z, coeffs_1)/p(X, Y, Z, coeffs_2), p(X, Y, Z, coeffs_3)/p(X, Y, Z, coeffs_4)
    return x, y
