import numpy as np


def sd_ros(x):
    aa = 100

    x1 = x[0]
    x2 = x[1]

    f = aa * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2

    g1 = 2 * aa * (x2 - x1 ** 2) * (-2 * x1) - 2 * (1 - x1)
    g2 = aa * (x2 - x1 ** 2) * 2

    g = np.array([g1, g2])

    B11 = 2 * aa * (x2 - x1 ** 2) * (-2) + 2 + 2 * aa * (-2 * x1) * (-2 * x1)
    B12 = 2 * aa * (-2 * x1)
    B22 = 2 * aa

    Hessian = np.array([[B11, B12], [B12, B22]])

    return f, g, Hessian


def zoom_w(x, p, alpha_lo, alpha_hi, c1, c2):

    # The following is an implementation of Algorithm 3.6, Nocedal and Wright, page 61.
    # Note: This code is a simple modified version of a code by Davide Taviani, https://gist.github.com/Heliosmaster/1043132

    maxit = 20

    f0, g0, _ = sd_ros(x)
    df0 = np.dot(p, g0)

    j = 0
    while 1:
        # I use bracketing and bisection to estimate the best value of alpha,
        # meaning the trial step-length is the middle point of [alpha_lo,alpha_hi]
        alpha_j = (alpha_lo+alpha_hi)/2

        f, g, _ = sd_ros(x+alpha_j*p)
        df = np.dot(p, g)

        f_lo, _, _ = sd_ros(x+alpha_lo*p)

        # Test for sufficient decrease or comparison with alpha_lo.
        if ((f > f0 + c1*alpha_j*df0) or (f >= f_lo)):
            alpha_hi = alpha_j  # Narrow the interval
        else:
            if np.abs(df) <= -c2*df0:  # SW2 satisfied
                alpha_star = alpha_j
                break
            if df*(alpha_hi-alpha_lo) >= 0:
                alpha_hi = alpha_lo
            alpha_lo = alpha_j  # Interval is now [alpha,alpha_lo]
        if j == maxit:
            alpha_star = alpha_j
            break
        j = j + 1
    return alpha_star


def lsa(x, p, alpha_i):
    # The following is an implementaiton of Algorithm 3.5, Nocedal and Wright,
    # page 60.

    # Note: This code is a simple modified version of a code by Davide Taviani,
    # https://gist.github.com/Heliosmaster/1043132

    # alpha_1 = a_i
    # a0 = a_{i-1}

    # Initialization of default parameters
    # Here, c1 and c2 are the parameters s recommended by Nocedal and Wright
    # (p. 62)

    c1 = 1e-4
    c2 = 0.9
    maxit = 100

    alpha_0 = 0
    alpha_max = 10*alpha_i

    f0, g0, _ = sd_ros(x)
    df0 = np.dot(p, g0)

    i = 1

    while 1:
        fold, _, _ = sd_ros(x+alpha_0*p)
        f, g, _ = sd_ros(x+alpha_i*p)
        df = np.dot(g, p)

        if ((f > f0+c1*alpha_i*df0) or ((i > 1) and (f > fold))):  # Check for SW1
            alpha_star = zoom_w(x, p, alpha_0, alpha_i, c1, c2)
            return alpha_star

        if (np.abs(df) <= -c2*df):  # Check for SW2
            alpha_star = alpha_i
            return alpha_star

        if (df >= 0):
            alpha_star = zoom_w(x, p, alpha_i, alpha_0, c1, c2)
            return alpha_star

        if i == maxit:
            print("Maximum iterations reached")
            alpha_star = alpha_i
            return alpha_star

        # Update for next loop iteration
        i = i + 1
        alpha_0 = alpha_i

        # Update alpha using linear iterpolation
        rho = 0.8
        alpha_i = rho*alpha_0 + (1-rho)*alpha_max
