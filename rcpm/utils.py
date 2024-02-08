import numpy as np
import torch


def spherical_to_euclidean(theta_phi):
    single = theta_phi.ndim == 1
    if torch.is_tensor(theta_phi):
        if not single:
            theta_phi = theta_phi.squeeze(1)
        theta, phi = torch.split(theta_phi, 1, 1)
        return torch.cat((
            torch.sin(phi) * torch.cos(theta),
            torch.sin(phi) * torch.sin(theta),
            torch.cos(phi)
        ), 1)

    else:
        if single:
            theta_phi = np.expand_dims(theta_phi, 0)
        theta, phi = np.split(theta_phi, 2, 1)
        return np.concatenate((
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ), 1)


def euclidean_to_spherical(xyz):
    single = xyz.ndim == 1
    if single:
        xyz = np.expand_dims(xyz, 0)
    x, y, z = np.split(xyz, 3, 1)
    return np.concatenate((
        np.arctan2(y, x),
        np.arccos(z)
    ), 1)


def fibonacci_sphere(samples=1000):
    theta = np.pi * (1 + 5 ** 0.5)

    points = []
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta += np.pi * (3. - 5. ** 0.5)
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])

    return torch.tensor(points)


def rotation3D_matrix(A, B):
    """Calculate the 3D rotation matrix to align vector A with vector B."""
    dot_prod = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    angle = np.arccos(dot_prod / (norm_A * norm_B))
    if angle == 0:
        return np.eye(3)
    else:
        axis = np.cross(A, B) / np.linalg.norm(np.cross(A, B))
        skew_symm = np.array([[0, -axis[2], axis[1]],
                              [axis[2], 0, -axis[0]],
                              [-axis[1], axis[0], 0]])
        rot_matrix = np.cos(angle) * np.eye(3) + (1 - np.cos(angle)) * np.outer(axis,
                                                                                axis) + np.sin(
            angle) * skew_symm
        return rot_matrix


def S1euclideantospherical(euc_coords):
    batch = euc_coords.ndim > 1

    if torch.is_tensor(euc_coords):
        if not batch:
            euc_coords = torch.unsqueeze(euc_coords, 0)
        theta = torch.atan2(euc_coords[:, 1], euc_coords[:, 0])
        if not batch:
            theta = theta[0]
        # return theta - theta.div(2 * np.pi, rounding_mode="floor") * 2 * np.pi
    else:
        if not batch:
            euc_coords = np.expand_dims(euc_coords, 0)
        theta = np.arctan2(euc_coords[:, 1], euc_coords[:, 0])
        if not batch:
            theta = theta[0]
    return theta  # - theta // (2 * np.pi) * 2 * np.pi


def S1sphericaltoeuclidean(theta):
    if torch.is_tensor(theta):
        x = torch.cos(theta)
        y = torch.sin(theta)
        return torch.stack([x, y], dim=-1)
    else:
        x = np.cos(theta)
        y = np.sin(theta)
        return np.stack([x, y], axis=-1)


def TORUSeuclideantospherical(eucl):
    if not torch.is_tensor(eucl):
        eucl = torch.as_tensor(eucl)
    batch = eucl.ndim > 1
    if not batch:
        eucl = torch.unsqueeze(eucl, 0)
    thetas = torch.stack((S1euclideantospherical(eucl[:, :2]),
                          S1euclideantospherical(eucl[:, 2:])), 1)
    if not batch:
        thetas = thetas[0]
    return thetas


def TORUSsphericaltoeuclidean(thetas):
    batch = thetas.ndim > 1
    if torch.is_tensor(thetas):
        if not batch:
            thetas = torch.unsqueeze(thetas, 0)
        eucl = torch.concat((S1sphericaltoeuclidean(thetas[:, 0]),
                             S1sphericaltoeuclidean(thetas[:, 1])), 1)
    else:
        if not batch:
            thetas = np.expand_dims(thetas, 0)
        eucl = np.concatenate((S1sphericaltoeuclidean(thetas[:, 0]),
                               S1sphericaltoeuclidean(thetas[:, 1])), 1)
    if not batch:
        eucl = eucl[0]

    return eucl


def productS1toTorus(theta1, theta2, R=1, r=0.4):
    x = (R + r * np.cos(theta1)) * np.cos(theta2)
    y = (R + r * np.cos(theta1)) * np.sin(theta2)
    z = r * np.sin(theta1)
    return x, y, z


def S1eucltoTorus(prod_coords, as_tensor=False, R=1, r=0.4):
    cos1 = prod_coords[:, 0]
    sin1 = prod_coords[:, 1]
    cos2 = prod_coords[:, 2]
    sin2 = prod_coords[:, 3]

    x = (R + r * cos1) * cos2
    y = (R + r * cos1) * sin2
    z = r * sin1
    if as_tensor:
        return torch.stack([x, y, z], dim=-1)
    else:
        return x, y, z


def TORUStangentto3d(eucl, as_tensor=False):
    batch = eucl.ndim > 1
    if not batch:
        eucl = torch.unsqueeze(eucl, 0)
    cooords3d = torch.stack((eucl[:, 0] + eucl[:, 2], eucl[:, 1], eucl[:, 3]), 1)
    return cooords3d


## COVERAGE ##
def order_contour_points(points):
    # Calculate the centroid
    centroid = torch.mean(points, dim=0)

    # Calculate the angles between points and the centroid
    angles = torch.atan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])

    # Sort the points based on angles
    sorted_indices = torch.argsort(angles)
    sorted_points = points[sorted_indices]

    return sorted_points

