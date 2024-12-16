import numpy as np
from scipy.special import binom
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.measure import grid_points_in_poly

# credit: https://stackoverflow.com/a/50751932


def bernstein(n, k, t):
    return binom(n, k) * t**k * (1.0 - t) ** (n - k)


def bezier(points, numpoints=100):
    N = len(points)
    t = np.linspace(0, 1, num=numpoints)
    curve = np.zeros((numpoints, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve


class Segment:
    def __init__(self, p1, p2, angle1, angle2, r=0.3, numpoints=100):
        self.p1 = p1
        self.p2 = p2
        self.angle1 = angle1
        self.angle2 = angle2
        self.numpoints = numpoints
        d = np.sqrt(np.sum((self.p2 - self.p1) ** 2))
        self.r = r * d
        self.p = np.zeros((4, 2))
        self.p[0, :] = self.p1[:]
        self.p[3, :] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self, r):
        self.p[1, :] = self.p1 + np.array(
            [r * np.cos(self.angle1), r * np.sin(self.angle1)]
        )
        self.p[2, :] = self.p2 + np.array(
            [r * np.cos(self.angle2 + np.pi), r * np.sin(self.angle2 + np.pi)]
        )
        self.curve = bezier(self.p, self.numpoints)


def get_curve(points, **kw):
    segments = []
    for i in range(len(points) - 1):
        seg = Segment(
            points[i, :2], points[i + 1, :2], points[i, 2], points[i + 1, 2], **kw
        )
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve


def ccw_sort(p):
    """Counter ClockWise angle sort"""
    d = p - np.mean(p, axis=0)
    s = np.arctan2(d[:, 0], d[:, 1])
    return p[np.argsort(s), :]


def get_bezier_curve(points, rad=0.2, numpoints=100):
    """given an array of points, create a curve through
    those points.
    *rad* is a number between 0 and 1 to steer the distance of
          control points.
    """

    points = ccw_sort(points)
    points = np.append(points, np.atleast_2d(points[0, :]), axis=0)
    d = np.diff(points, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    # only positive angles
    ang = np.where(ang >= 0, ang, ang + 2 * np.pi)
    ang1 = ang
    ang2 = np.roll(ang, 1)

    ang = 0.5 * (ang2 + ang1) + (np.abs(ang2 - ang1) > np.pi) * np.pi
    ang = np.append(ang, [ang[0]])
    points = np.append(points, np.atleast_2d(ang).T, axis=1)
    s, c = get_curve(points, r=rad, numpoints=numpoints)
    return c, points


def get_random_points(n=5, scale=256, mindst=None, rec=0):
    """create n random points in the unit square, which are *mindst*
    apart, then scale them."""
    mindst = mindst or 0.7 / n
    a = np.random.rand(n, 2)
    # calculate distances between following points
    d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0) ** 2, axis=1))
    if np.all(d >= mindst) or rec >= 200:
        return a * scale
    else:
        return get_random_points(n=n, scale=scale, mindst=mindst, rec=rec + 1)


if __name__ == "__main__":
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.set_xlim(0, 256)
    ax1.set_ylim(0, 256)

    ax2.set_xlim(0, 256)
    ax2.set_ylim(0, 256)

    n = 7
    rad = 0.4
    edgy = 0
    fraction = 1 / 4
    scale = 256 * np.sqrt(fraction)

    a = get_random_points(
        n=7, scale=scale
    )  # 64 = 256 / sqrt(16) so that the bbox is 1/16 of the image
    offset = np.random.rand(1, 2) * scale

    a += offset
    curve_verts, _ = get_bezier_curve(a, rad=rad, numpoints=5)
    x, y = curve_verts.T
    ax1.plot(x, y)

    # Create a Rectangle patch
    rect = patches.Rectangle(
        (0, 0), 256, 256, linewidth=1, edgecolor="r", facecolor="none"
    )
    rect2 = patches.Rectangle(
        offset[0], scale, scale, linewidth=1, edgecolor="r", facecolor="none"
    )

    # Add the patch to the Axes
    ax1.add_patch(rect)
    ax1.add_patch(rect2)

    mask = grid_points_in_poly((256, 256), curve_verts)

    ax2.imshow(mask, cmap="gray")

    plt.show()
