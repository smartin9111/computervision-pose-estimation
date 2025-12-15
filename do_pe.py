import open3d as o3d
import copy
import numpy as np
from tqdm import tqdm
import random

# PARAMETERS

DEBUG = False

SIDE_DEFAULT = "BOTTOM"

VOXEL = 0.003
PLANE_DIST = 0.006

ZMIN = 0.0015
ZMAX = 0.25

OBJ_EPS = 0.04
OBJ_MINPTS = 10

STL_SCALE = 1.0
STL_AXIS_FIX = None
MESH_SAMPLE_N = 20000

ICP_MAX_CORR = 0.01
ICP_ITERS = 300
YAW_SAMPLES = 120


def do_pose_estimation(scene_pointcloud, object_pointcloud):

    # --- helpers ---
    def unit(v): return v / (np.linalg.norm(v) + 1e-12)

    def make_T(R, t):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def plane_basis_from_normal(n):
        n = unit(n)
        ref = np.array([1, 0, 0])
        if abs(np.dot(n, ref)) > 0.9:
            ref = np.array([0, 1, 0])
        x = unit(np.cross(n, ref))
        y = unit(np.cross(n, x))
        return x, y

    def build_table_frame_from_plane(plane_pcd, n):
        origin = np.asarray(plane_pcd.get_center(), float)
        x_tbl, y_tbl = plane_basis_from_normal(n)
        z_tbl = unit(n)
        R = np.column_stack([x_tbl, y_tbl, z_tbl])
        T_scene_from_table = make_T(R, origin)
        return T_scene_from_table, np.linalg.inv(T_scene_from_table)

    def preprocess_pcd(pcd, voxel_):
        q = copy.deepcopy(pcd)
        q = q.voxel_down_sample(voxel_)
        if len(q.points) > 50:
            q, _ = q.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        q.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_ * 5, max_nn=30))
        q.normalize_normals()
        return q

    def filter_by_z_tbl(pcd, zlo, zhi):
        z = np.asarray(pcd.points)[:, 2]
        idx = np.where((z >= zlo) & (z <= zhi))[0]
        return pcd.select_by_index(idx.tolist())

    def pick_object_cluster(cand_tbl, ref_center_tbl, eps, obj_minpts):
        labels = np.array(cand_tbl.cluster_dbscan(
            eps=eps, min_points=obj_minpts))
        if labels.size == 0 or labels.max() < 0:
            return None
        pts = np.asarray(cand_tbl.points)
        best_lbl = None
        best_dist = None
        for lbl in range(labels.max() + 1):
            idx = np.where(labels == lbl)[0]
            if idx.size == 0:
                continue
            c = pts[idx].mean(axis=0)
            d = np.linalg.norm(c - ref_center_tbl)
            if best_dist is None or d < best_dist:
                best_dist, d
                best_lbl = lbl
        if best_lbl is None:
            return None
        keep = np.where(labels == best_lbl)[0]
        return cand_tbl.select_by_index(keep.tolist())

    def estimate_init_T_from_obb(src, tgt):
        obb_s = src.get_oriented_bounding_box()
        obb_t = tgt.get_oriented_bounding_box()
        R = obb_t.R @ obb_s.R.T
        t = obb_t.center - R @ obb_s.center
        return make_T(R, t)

    def run_icp(src, tgt, T_init, max_corr, iters):
        src = copy.deepcopy(src)
        tgt = copy.deepcopy(tgt)
        src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=max_corr * 2, max_nn=30))
        tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=max_corr * 2, max_nn=30))
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=int(iters))
        reg = o3d.pipelines.registration.registration_icp(
            src, tgt,
            max_correspondence_distance=float(max_corr),
            init=T_init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=criteria
        )
        return reg.transformation, reg.fitness, reg.inlier_rmse

    def yaw_rot_z(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0],
                         [s,  c, 0],
                         [0,  0, 1]], float)

    # ---  Plane detection ---
    scene = preprocess_pcd(scene_pointcloud, VOXEL)
    plane_model, inliers = scene.segment_plane(
        distance_threshold=PLANE_DIST, ransac_n=3, num_iterations=2000)
    a, b, c, d = plane_model
    n = unit([a, b, c])
    if np.dot(n, [0, 0, 1]) < 0:
        n = -n
    plane_pcd = scene.select_by_index(inliers)
    wo_plane = scene.select_by_index(inliers, invert=True)

    # --- TABLE frame ---
    T_scene_from_table, T_table_from_scene = build_table_frame_from_plane(
        plane_pcd, n)
    wo_tbl = copy.deepcopy(wo_plane)
    wo_tbl.transform(T_table_from_scene)

    # --- Side-filter ---
    side = SIDE_DEFAULT
    zmin, zmax = ZMIN, ZMAX

    z = np.asarray(wo_tbl.points)[:, 2]
    if side.upper() == "TOP":
        mask = (z > zmin) & (z < zmax)
        zlo, zhi = zmin, zmax
    else:
        mask = (z < -zmin) & (z > -zmax)
        zlo, zhi = -zmax, -zmin

    above_tbl = wo_tbl.select_by_index(np.where(mask)[0].tolist())
    roi_tbl = filter_by_z_tbl(above_tbl, zlo, zhi)

    # --- Object cluster ---
    ref_center = np.asarray(roi_tbl.get_center(), float)
    obj_tbl = pick_object_cluster(roi_tbl, ref_center, OBJ_EPS, OBJ_MINPTS)
    if obj_tbl is None or len(obj_tbl.points) < 60:
        labels = np.array(roi_tbl.cluster_dbscan(
            eps=OBJ_EPS, min_points=OBJ_MINPTS))
        if labels.size > 0 and labels.max() >= 0:
            best = np.bincount(labels[labels >= 0]).argmax()
            obj_tbl = roi_tbl.select_by_index(
                np.where(labels == best)[0].tolist())
        else:
            if DEBUG:
                print("[WARN] No cluster found, using ROI directly")
            obj_tbl = roi_tbl

    # --- Model prep ---
    model_input = object_pointcloud
    if isinstance(model_input, o3d.geometry.TriangleMesh):
        mesh = copy.deepcopy(model_input)
        if STL_AXIS_FIX is not None:
            mesh.transform(STL_AXIS_FIX)
        if abs(STL_SCALE - 1.0) > 1e-12:
            mesh.scale(STL_SCALE, center=(0, 0, 0))
        mesh.compute_vertex_normals()
        model_pcd = mesh.sample_points_uniformly(
            number_of_points=MESH_SAMPLE_N)
    elif isinstance(model_input, o3d.geometry.PointCloud):
        model_pcd = preprocess_pcd(model_input, VOXEL)
    else:
        raise TypeError("object_pointcloud must be TriangleMesh or PointCloud")

    # --- Multi-init ICP ---
    inits = []
    inits.append(("OBB", estimate_init_T_from_obb(model_pcd, obj_tbl)))
    t = obj_tbl.get_center() - model_pcd.get_center()
    inits.append(("CENTROID", make_T(np.eye(3), t)))

    for i, ang in enumerate(np.linspace(0, 2*np.pi, YAW_SAMPLES, endpoint=False)):
        Rz = yaw_rot_z(ang)
        inits.append((f"YAW[{i}]", make_T(Rz, t)))

    for axis, Rf in [("FLIP_X", np.diag([-1, 1, 1])),
                     ("FLIP_Y", np.diag([1, -1, 1])),
                     ("FLIP_Z", np.diag([1, 1, -1]))]:
        inits.append((axis, make_T(Rf, t)))

    best = None
    for name, T0 in inits:
        T_try, fit, rmse = run_icp(
            model_pcd, obj_tbl, T0, ICP_MAX_CORR, ICP_ITERS // 2)
        if DEBUG:
            print(f"[DBG][INIT] {name} -> fit={fit:.4f}, rmse={rmse:.6f}")
        if best is None or fit > best[1] or (fit == best[1] and rmse < best[2]):
            best = (T_try, fit, rmse, name)

    if best is None:
        raise RuntimeError("No viable ICP init")

    T_best, fit_best, rmse_best, name_best = best
    if DEBUG:
        print(
            f"[DBG] BEST init={name_best}, fit={fit_best:.4f}, rmse={rmse_best:.6f}")

    T_final, fit_final, rmse_final = run_icp(model_pcd, obj_tbl, T_best,
                                             ICP_MAX_CORR, ICP_ITERS)
    if DEBUG:
        print(f"[DBG] FINAL ICP fit={fit_final:.4f}, rmse={rmse_final:.6f}")

    T_world = T_scene_from_table @ T_final
    return T_world
