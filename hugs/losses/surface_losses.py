import torch
import torch
import matplotlib.cm as cm


def fit_plane(points):
    """
    Fit a plane to a set of 3D points using Singular Value Decomposition (SVD).

    Parameters:
        points (torch.Tensor): A tensor of shape (N, 3) representing 3D points.

    Returns:
        normal (torch.Tensor): The normal vector of the fitted plane.
        point_on_plane (torch.Tensor): A point on the fitted plane.
    """
    # Compute the centroid of the points
    centroid = torch.mean(points, dim=1, keepdim=True)

    # Subtract the centroid to center the points
    centered_points = points - centroid

    # Use SVD to compute the normal vector of the plane
    _, _, V = torch.svd(centered_points)
    
    # The normal vector is the last column of V
    normal = V[..., 2]

    # A point on the plane is the centroid
    point_on_plane = centroid

    return normal, point_on_plane


def fit_plane_weighted_pca(points, weights):
    """
    Fit a plane to a set of 3D points using Singular Value Decomposition (SVD).

    Parameters:
        points (torch.Tensor): A tensor of shape (N, 3) representing 3D points.

    Returns:
        normal (torch.Tensor): The normal vector of the fitted plane.
        point_on_plane (torch.Tensor): A point on the fitted plane.
    """
    # https://stats.stackexchange.com/questions/113485/weighted-principal-components-analysis

    if weights.shape[-1] != 1:
        weights = weights.unsqueeze(-1)
        
    if points.device != 'cuda':
        weights = weights.cuda()
        points = points.cuda()
        
    weights_sum = torch.sum(weights, dim=1, keepdim=True)
    centroid = torch.sum(points * weights, dim=1, keepdim=True) / weights_sum

    # Subtract the centroid to center the points
    centered_points = points - centroid
    
    weighted_cov = centered_points.transpose(-1, -2) @ (torch.diag_embed(weights.squeeze(-1)) @ centered_points)
    weighted_cov = weighted_cov / weights_sum
    
    # Use SVD to compute the normal vector of the plane
    
    _, _, V = torch.svd(weighted_cov)
    
    # The normal vector is the last column of V
    normal = V[..., 2]

    # A point on the plane is the centroid
    point_on_plane = centroid

    return normal, point_on_plane


def project_point_onto_plane(points, normal, point_on_plane):
    """
    Project a point onto a plane.

    Parameters:
        point (torch.Tensor): The point in 3D space as a tensor of shape (3,).
        normal (torch.Tensor): The normal vector of the plane.
        point_on_plane (torch.Tensor): A point on the plane.

    Returns:
        projected_point (torch.Tensor): The projected point onto the plane.
    """
    ft = normal.unsqueeze(1) @ point_on_plane.transpose(-1, -2)
    st = normal.unsqueeze(1).unsqueeze(1) @ points.unsqueeze(2).transpose(-1, -2)
    denom = normal[:, None] @ normal[:, None].transpose(-1, -2)
    dist = (ft.squeeze(-1) - st.squeeze(-1).squeeze(-1)) / denom.squeeze(-1)
    
    projected_point = points + dist.unsqueeze(-1) * normal.unsqueeze(1)
    return projected_point, dist


def point_plane_normal_loss(knn_normals, plane_normal, weights=None, reduction='mean'):
    
    # since the sign of the normal is arbitrary, we need to check both
    pos_loss = 1 - torch.cosine_similarity(knn_normals, plane_normal[:, None], dim=-1)
    neg_loss = 1 - torch.cosine_similarity(knn_normals, -plane_normal[:, None], dim=-1)
    loss = torch.stack([pos_loss, neg_loss], dim=-1).min(-1).values
    if weights is not None:
        loss = loss * weights
        
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    
    
def point_plane_dist_loss(knn_pos, plane_normal, plane_point, weights=None, reduction='mean'):
    _, dist = project_point_onto_plane(knn_pos, plane_normal, plane_point)
    dist = dist * dist
    if weights is not None:
        dist = dist * weights
        
    if reduction == 'mean':
        return dist.mean()
    elif reduction == 'sum':
        return dist.sum()
    elif reduction == 'none':
        return dist


def get_spatial_w(points):
    norm = torch.norm(points - points.mean(dim=1, keepdim=True), dim=-1)
    denom = norm.max().pow(2)
    return torch.exp(- norm / denom)


def plot_plane(point_on_plane, normal, points, ax):
    d = -point_on_plane.dot(normal)
    # Plot the tangent plane
    min, max = points.min(), points.max()

    xx, yy = torch.meshgrid(
        torch.linspace(min, max, 10),
        torch.linspace(min, max, 10)
    )
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

    ax.plot_surface(xx, yy, z, alpha=0.4, label='Tangent Plane')
    h
