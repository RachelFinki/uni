import math
import random
import numpy
import matplotlib.pyplot as plt
mass = 9.109*10**(-31)
charge = -1.602*10**(-19)
tau = 10 ** -15
electric_field = 30


def location_per_collision(x_initial, y_initial, vx_initial, vy_initial):
    acceleration = charge * electric_field / mass
    x_final = x_initial + vx_initial * tau + 0.5 * acceleration * (tau ** 2)
    y_final = y_initial + vy_initial*tau
    return x_final, y_final


def random_angle():
    return random.random()*2*math.pi


def collision():
    """

    :return: the velocity after a collision
    """
    angle = random_angle()
    vx = 0.0002 * math.cos(angle)
    vy = 0.0002 * math.sin(angle)
    return vx, vy


def hundred_collisions_process(x, y, vx, vy):
    """
    :param vy_initial: starting speed at y direction
    :param vx_initial: starting speed at x direction
    :param x_initial: start x coordinate
    :param y_initial: start y coordinate
    :return: x,y coordinates after 100 collisions
    """
    x_coordinates = []
    y_coordinates = []
    for i in range(100):
        x, y = location_per_collision(x, y, vx, vy)
        vx, vy = collision()
        x_coordinates.append(x)
        y_coordinates.append(y)
    x_coordinates = numpy.array(x_coordinates)
    y_coordinates = numpy.array(y_coordinates)
    return x_coordinates, y_coordinates


x_coordinates, y_coordinates = hundred_collisions_process(0, 0, 0, 0)

plt.plot(x_coordinates, y_coordinates)
plt.show()
