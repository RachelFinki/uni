import math
import random
mass = 9.109
charge = -1.602
tau = 10 ** -15
electric_field = 30


def location_per_collision(x_initial, y_initial, vx_initial, vy_initial):
    acceleration = charge * electric_field / mass
    x_final = x_initial + vx_initial * tau + 0.5 * acceleration * (tau ** 2)
    y_final = y_initial + vy_initial*tau
    return x_final, y_final


def random_angle():
    return random.random()*2*math.pi
1bb

def collision():
    """

    :return: the velocity after a collision
    """
    angle = random_angle()
    vx = 0.0002*math.cos(angle)
    vy = 0.0002 * math.sin(angle)
    return vx, vy


def 100