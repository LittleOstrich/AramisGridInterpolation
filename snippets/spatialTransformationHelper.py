import numpy as np
import numpy as np
from sklearn.decomposition import PCA
from skspatial.objects import Plane
from skspatial.objects import Points
from skspatial.plotting import plot_3d


def calc_cos_phi(a, b, c):
    return c / np.sqrt(a * a + b * b + c * c)


def calc_sin_phi(a, b, c):
    return np.sqrt((a * a + b * b) / (a * a + b * b + c * c))


def calc_u1(a, b, c):
    return b / np.sqrt(a * a + b * b)


def calc_u2(a, b, c):
    return -a / np.sqrt(a * a + b * b)


def get_transform_matrix(plane):
    a, b, c, d = plane
    cos_phi = calc_cos_phi(a, b, c)
    sin_phi = calc_sin_phi(a, b, c)
    u1 = calc_u1(a, b, c)
    u2 = calc_u2(a, b, c)
    out = np.array([
        [cos_phi + u1 * u1 * (1 - cos_phi), u1 * u2 * (1 - cos_phi), u2 * sin_phi, 0],
        [u1 * u2 * (1 - cos_phi), cos_phi + u2 * u2 * (1 - cos_phi), -u1 * sin_phi, 0],
        [-u2 * sin_phi, u1 * sin_phi, cos_phi, -d / c],
        [0, 0, 0, 1]
    ])
    return out


def transform_plane(plane):
    t = get_transform_matrix(plane)
    t_inv = np.linalg.inv(t)
    new_plane = np.dot(plane, t_inv)
    print("new plane:")
    print(new_plane)
    return new_plane


plane = [-0.9437203588952212, 0.1681597233102734, 0.2848055330622773, -0.5959815443123256]
plane_xy = transform_plane(plane)
cartesian = plane_xy
a = cartesian[0]
b = cartesian[1]
c = cartesian[2]
d = cartesian[3]
p1 = [0.5 * -d / a, 0.5 * -d / b, 0]
p2 = [0, -d / b, 0]
p3 = [0, 0, -d / c]

plane = Plane.from_points(p1, p2, p3)
plot_3d(
    plane.plotter(alpha=0.2, lims_x=(-5, 5), lims_y=(-5, 5)),
)

# cartesian = plane_xy
# a = cartesian[0]
# b = cartesian[1]
# c = cartesian[2]
# d = cartesian[3]
#
# p1 = [-d / a, 0, 0]
# p2 = [0, -d / b, 0]
# p3 = [0, 0, -d / c]
#
# plane = Plane.from_points(p1, p2, p3)
# data = [[1, 2, 2], [1.1, 2, 5.3], [1.3, 2, 7], [2, 0, 7.2], [-1, 0, 0]]
# points = Points(data)
# plot_3d(
#     points.plotter(c='k', s=50, depthshade=False),
#     plane.plotter(alpha=0.2, lims_x=(-5, 5), lims_y=(-5, 5)),
# )
#
# print(plane_xy)
# print("Finally done!")
