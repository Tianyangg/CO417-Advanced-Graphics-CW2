import matplotlib.pyplot as plt
import math
import numpy as np

"""
Part 1
"""
def reflectance_both(ni, nt, theta_i):
    """
    Calculate the fresnel with parrellel, perpendicular and total
    Support matrix input of theta_i much faster than a looooooop
    :param ni: incident material eta_i
    :param nt: another material eta_t
    :param theta_i: incident angle
    :return:
    """
    # calculate theta_t using snell's law
    theta_t = np.arcsin(np.sin(theta_i) * ni / nt)
    # print(theta_t)

    R_parallel = np.power((nt * np.cos(theta_i) - ni * np.cos(theta_t)) / (nt * np.cos(theta_i) + ni * np.cos(theta_t)), 2)
    R_perp = np.power((ni * np.cos(theta_i) - nt * np.cos(theta_t)) / (ni * np.cos(theta_i) + nt * np.cos(theta_t)), 2)
    unpolarized = np.multiply(np.add(R_parallel, R_perp), 0.5)
    return R_parallel, R_perp, unpolarized


def brewster(ni, nt):
    """

    :param ni: incident material eta_i
    :param nt: another material eta_t
    :return:
    """
    return math.degrees(math.atan(nt / ni))

def critical(ni, nt):
    """

    :param ni: incident material eta_i
    :param nt: another material eta_t
    :return: critical angle when ni > nt
    """
    return math.degrees(math.asin(nt / ni))

def fresnel(ni, nt):
    N = 90
    upperbound = 90  # change if ni > nt for the B... angle
    if ni <= nt:
        brewster_angle = brewster(ni, nt)  # measured in degree
        plt.vlines(brewster_angle, 0., 1., linestyles = "dashed", label = "Brewster")
        print("Brewster_angle activated: ", brewster_angle)
        # Report the reflectance of both components normal and Brewster’s angle
        two_angle = np.array([0, brewster_angle])
        r_pp, r_ss, ttt = reflectance_both(ni, nt, np.radians(two_angle))
        print("normal rp: ", r_pp[0])
        print("normal rs: ", r_ss[0])
        print("brewster rp: ", r_pp[1])
        print("brewster rs: ", r_ss[1])
    else:
        brewster_angle = brewster(ni, nt)  # measured in degree
        print("Brewster_angle activated: ", brewster_angle)
        two_angle = np.array([0, brewster_angle])
        r_pp, r_ss, ttt = reflectance_both(ni, nt, np.radians(two_angle))
        print("normal rp: ", r_pp[0])
        print("normal rs: ", r_ss[0])
        print("brewster rp: ", r_pp[1])
        print("brewster rs: ", r_ss[1])

        critical_angle = critical(ni, nt)
        upperbound = critical_angle
        plt.vlines(critical_angle, 0., 1., linestyles = "dashed", label = "critical")
        print("critical angle activated: ", critical_angle)

    x = np.linspace(0, upperbound, N)
    # r_p = np.zeros(N)
    # r_s = np.zeros(N)
    # unpolar = np.zeros(N)
    r_p, r_s, unpolar = reflectance_both(ni, nt, np.radians(x))

    axes = plt.gca()
    axes.set_xlim([0, 90])
    axes.set_ylim([0, 1])

    # plt.plot(x, r_p, label = "$R_p$")
    # plt.plot(x, r_s, label = "$R_s$")
    plt.plot(x, unpolar, label = "reference (unpolarized)")
    # plt.xlabel("Angle")
    # plt.ylabel("Reflectance percentage")
    plt.legend()
    # plt.title("Parallel and perpendicular components of reflectance\n $\eta_i$ = " + str(ni) + ", $\eta_t$ =  " + str(nt))
    plt.show()

def schlic_approx(r0, theta):
    """
    approximation
    :param r0: reflectance and normal incidence
    :param theta: support matrix
    :return: calculated value for schlic_approx
    """
    # print("hey", np.power((1 - np.cos(theta)), 5))
    return np.array(r0 + np.multiply((1. - r0), np.power((1. - np.cos(theta)), 5)), dtype=float)


def schlick_approx_plot(ni, nt):

    upperbound = 90
    N = 90
    if ni > nt:
        critical_angle = critical(ni, nt)
        upperbound = critical_angle
    # get r0
    r_pp, r_ss, r0 = reflectance_both(ni, nt, np.radians(0))
    print("r0", r0)
    theta = np.linspace(0, upperbound, N)
    sch = schlic_approx(r0, np.arcsin(max(1, (ni / nt)) * np.sin(np.radians(theta))))
    plt.plot(theta, sch, label = "approx")
    plt.xlabel("Angle")
    plt.ylabel("Reflectance percentage")
    # plt.legend()
    plt.title("Schlick’s approximation of Fresnel reflectance \n $\eta_i$ = " + str(ni) + ", $\eta_t$ =  " + str(nt))
    # plt.show()

if __name__ == '__main__':
    # fresnel(1.0, 1.45)
    schlick_approx_plot(1., 1.45)
    # fresnel(1.45, 1.0)
    fresnel(1., 1.45)
#