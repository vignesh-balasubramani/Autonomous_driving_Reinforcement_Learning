from __future__ import division
import math
import numpy as np

def addition(vector_A, vector_B):
    Ax, Ay = vector_A
    Bx, By = vector_B
    return Ax + Bx, Ay + By

def add(vector_A, vector_B):
    Ax, Ay = vector_A
    Bx, By = vector_B
    return Ax + Bx, Ay + By

def subtraction(vector_A, vector_B):
    Ax, Ay = vector_A
    Bx, By = vector_B
    return Ax - Bx, Ay - By

def sub(vector_A, vector_B):
    Ax, Ay = vector_A
    Bx, By = vector_B
    return Ax - Bx, Ay - By

def multiply(vector, factor):
    x, y = vector
    return factor * x, factor * y

def mul(factor, vector):
    x, y = vector
    return factor * x, factor * y

def divide(vector, factor):
    x, y = vector
    return x / factor, y / factor

def div(vector, factor):
    x, y = vector
    return x / factor, y / factor

def rotate(vector, center, angle):
    Vx, Vy = vector
    Cx, Cy = center

    rad_angle = math.radians(angle)
    sin_angle = math.sin(rad_angle)
    cos_angle = math.cos(rad_angle)

    Vx = Vx - Cx
    Vy = Vy - Cy

    x = Vx*cos_angle - Vy*sin_angle
    y = Vx*sin_angle + Vy*cos_angle

    x = x + Cx
    y = y + Cy
    return x, y

def rot(vector, center, angle):
    Vx, Vy = vector
    Cx, Cy = center

    rad_angle = math.radians(angle)
    sin_angle = math.sin(rad_angle)
    cos_angle = math.cos(rad_angle)

    Vx = Vx - Cx
    Vy = Vy - Cy

    x = Vx*cos_angle - Vy*sin_angle
    y = Vx*sin_angle + Vy*cos_angle

    x = x + Cx
    y = y + Cy
    return x, y

def distance(A):
    x, y = A
    return math.sqrt(x**2 + y**2)

def dist(A):
    x, y = A
    return math.sqrt(x**2 + y**2)

def quad_distance(A):
    x, y = A
    return x**2 + y**2

def qdist(A):
    x, y = A
    return x**2 + y**2

def normal(angle):
    rad_angle = math.radians(angle)
    x = math.cos(rad_angle)
    y = math.sin(rad_angle)
    return x, y

def norm(angle):
    rad_angle = math.radians(angle)
    x = math.cos(rad_angle)
    y = math.sin(rad_angle)
    return x, y

def perpendicular(vector):
    x, y = vector
    return -y, x

def perp(vector):
    x, y = vector
    return -y, x

def dot_product(vector_A, vector_B):
    Ax, Ay = vector_A
    Bx, By = vector_B
    return Ax*Bx + Ay*By

def dot(vector_A, vector_B):
    Ax, Ay = vector_A
    Bx, By = vector_B
    return Ax*Bx + Ay*By

def mdot(matrix, vector):
    result = []
    for row in matrix:
        result.append(lines(row, vector))
    return result

def lines(row, vector):
    res = 0.0
    for i, val in enumerate(row):
        res += val * vector[i]
    return res

def ccw(A, B, C):
    Ax, Ay = A
    Bx, By = B
    Cx, Cy = C
    return (Bx - Ax) * (Cy - Ay) > (By - Ay) * (Cx - Ax)

def cw(A, B, C):
    Ax, Ay = A
    Bx, By = B
    Cx, Cy = C
    return (Bx - Ax) * (Cy - Ay) < (By - Ay) * (Cx - Ax)

def direction():
    pass

def neg(vector):
    x, y = vector
    return -x, -y

def equal(vector_A, vector_B):
    Ax, Ay = vector_A
    Bx, By = vector_B
    if Ax == Bx and Ay == By:
        return True
    else:
        return False

def shortest_distance(center, line):
    center = np.array(center)
    line = np.array(line)
    distances = []
    for pt in line:
        d = math.dist(center, pt)
        distances.append(d)

    return min(distances)
