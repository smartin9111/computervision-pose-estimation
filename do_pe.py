#!/usr/bin/env python3
import open3d as o3d
import numpy as np
import copy
import random

# ============================================================
# do_pe.py
# Pose estimation: scene_pointcloud + object_pointcloud -> 4x4 T_scene_from_object
# Cél: rotációs hiba csökkentése (multi-init + (opcionális) jitter + coarse-to-fine ICP)
# Közben: nagy gyorsítás (normálok cache, no deepcopy ICP-ben, early reject, FAST_MODE)
# ============================================================

# ----------------------------
# MÓD VÁLTÁS
# ----------------------------
FAST_MODE = True   # True: gyors / False: lassabb, de több próbát tesz

DEBUG = True

# ----------------------------
# PARAMÉTEREK
# ----------------------------
# Downsample / denoise
VOXEL_SCENE = 0.003
VOXEL_MODEL = 0.002
OUTLIER_NB = 20
OUTLIER_STD = 2.0

# Plane
PLANE_DIST = 0.006
PLANE_ITERS = 2000

# TABLE z-szelet (z=0 a plane a TABLE frame-ben)
ZMIN = 0.0015
ZMAX = 0.25

# DBSCAN jelöltek
DBSCAN_EPS = 0.02
DBSCAN_MINPTS = 20

# Méretszűrés
SIZE_MAX_MULT = 3.0
SIZE_MIN_MULT = 0.25

# ICP stage-ek (coarse -> fine)
ICP_STAGES_FAST = [
    (0.020, 35),
    (0.010, 55),
]
ICP_STAGES_ACCURATE = [
    (0.030, 60),
    (0.015, 80),
    (0.008, 120),
]

# Score = fitness - W*rmse
SCORE_RMSE_WEIGHT = 2.0

# Multi-init rotációk (fekvő/álló + yaw)
YAW_DEG_FAST = [0, 90, 180, 270]
BASE_ROTS_FAST = ["I", "RX90", "RY90", "RX180"]

YAW_DEG_ACCURATE = [0, 45, 90, 120, 135, 180, 225, 240, 270, 315]
BASE_ROTS_ACCURATE = ["I", "RX90", "RX-90", "RY90", "RY-90", "RX180", "RY180"]

# Random restarts (init jitter) – FAST módban minimális
N_RANDOM_RESTARTS_FAST = 2
N_RANDOM_RESTARTS_ACCURATE = 12
JITTER_YAW_SIGMA_DEG = 8.0
JITTER_ROLLPITCH_SIGMA_DEG = 2.5
JITTER_TRANS_SIGMA_M = 0.0025  # ~2.5mm

# Oldalak próbálása
TRY_SIDES = ["TOP", "BOTTOM"]

# Klaszter limit
MAX_CLUSTERS_TO_TRY_FAST = 8
MAX_CLUSTERS_TO_TRY_ACCURATE = 25

# STL scale/axis fix (ha kell)
STL_SCALE = 1.0
STL_AXIS_FIX = None  # None, "ROT_Z_180", "FLIP_X", stb.

# Early reject (durva stage után)
EARLY_REJECT_FITNESS = 0.30


# ----------------------------
# CACHE (modell preprocess újrafelhasználás)
# ----------------------------
_MODEL_CACHE = {
    "key": None,
    "model": None,
    "center": None,
    "diag": None,
}


# ----------------------------
# Helper
# ----------------------------
def unit(v):
    v = np.asarray(v, dtype=float)
    return v / (np.linalg.norm(v) + 1e-12)


def preprocess(pcd, voxel):
    q = copy.deepcopy(pcd)
    q = q.voxel_down_sample(float(voxel))
    if len(q.points) > 200:
        q, _ = q.remove_statistical_outlier(
            nb_neighbors=int(OUTLIER_NB), std_ratio=float(OUTLIER_STD)
        )
    return q


def ensure_normals(pcd, radius, max_nn=30):
    # normálok kellenek point-to-plane-hoz
    if (not pcd.has_normals()) or (len(pcd.normals) != len(pcd.points)):
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=float(radius), max_nn=int(max_nn))
        )
        pcd.normalize_normals()


def plane_basis_from_normal(n):
    n = unit(n)
    ref = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(n, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    x = unit(np.cross(n, ref))
    y = unit(np.cross(n, x))
    return x, y


def make_T(R, t):
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(t, dtype=float)
    return T


def build_table_frame_from_plane(plane_pcd, n):
    origin = np.asarray(plane_pcd.get_center(), dtype=float)
    x_tbl, y_tbl = plane_basis_from_normal(n)
    z_tbl = unit(n)
    R_scene_from_table = np.column_stack([x_tbl, y_tbl, z_tbl])
    T_scene_from_table = make_T(R_scene_from_table, origin)
    T_table_from_scene = np.linalg.inv(T_scene_from_table)
    return T_scene_from_table, T_table_from_scene


def bbox_extent(pcd):
    bb = pcd.get_axis_aligned_bounding_box()
    return np.asarray(bb.get_extent(), dtype=float)


def axis_fix_matrix(name):
    if name is None:
        return np.eye(4, dtype=float)
    name = name.upper()
    fixes = {
        "FLIP_X": np.array([[-1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=float),
        "FLIP_Y": np.array([[1, 0, 0, 0],
                            [0, -1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=float),
        "FLIP_Z": np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, -1, 0],
                            [0, 0, 0, 1]], dtype=float),
        "ROT_Z_180": np.array([[-1, 0, 0, 0],
                               [0, -1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]], dtype=float),
    }
    if name not in fixes:
        raise ValueError(f"Ismeretlen STL_AXIS_FIX: {name}")
    return fixes[name]


def rot_x(deg):
    a = np.deg2rad(float(deg))
    c, s = np.cos(a), np.sin(a)
    R = np.array([[1, 0, 0],
                  [0, c, -s],
                  [0, s,  c]], dtype=float)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    return T


def rot_y(deg):
    a = np.deg2rad(float(deg))
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, 0, s],
                  [0, 1, 0],
                  [-s, 0, c]], dtype=float)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    return T


def rot_z(deg):
    a = np.deg2rad(float(deg))
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]], dtype=float)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    return T


def base_rot(name):
    if name == "I":
        return np.eye(4, dtype=float)
    if name == "RX90":
        return rot_x(90)
    if name == "RX-90":
        return rot_x(-90)
    if name == "RY90":
        return rot_y(90)
    if name == "RY-90":
        return rot_y(-90)
    if name == "RX180":
        return rot_x(180)
    if name == "RY180":
        return rot_y(180)
    raise ValueError(name)


def T_rotate_about_center(Trot, center_xyz):
    c = np.asarray(center_xyz, dtype=float).reshape(3)
    R = Trot[:3, :3]
    t = c - R @ c
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def estimate_init_T_from_obb(source_pcd, target_pcd):
    # return: T_target_from_source
    obb_s = source_pcd.get_oriented_bounding_box()
    obb_t = target_pcd.get_oriented_bounding_box()

    R_s = np.asarray(obb_s.R, dtype=float)
    c_s = np.asarray(obb_s.center, dtype=float)
    R_t = np.asarray(obb_t.R, dtype=float)
    c_t = np.asarray(obb_t.center, dtype=float)

    R = R_t @ R_s.T
    t = c_t - R @ c_s

    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _try_make_p2plane_estimator_with_robust(max_corr):
    try:
        loss = o3d.pipelines.registration.TukeyLoss(k=float(max_corr))
        return o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
    except Exception:
        return o3d.pipelines.registration.TransformationEstimationPointToPlane()


def run_icp_point_to_plane_fast(src, tgt, T_init, max_corr, iters):
    # Feltételezzük: src/tgt normals már megvannak
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=int(iters))
    estimator = _try_make_p2plane_estimator_with_robust(max_corr)
    reg = o3d.pipelines.registration.registration_icp(
        src, tgt,
        max_correspondence_distance=float(max_corr),
        init=np.asarray(T_init, dtype=float),
        estimation_method=estimator,
        criteria=criteria
    )
    return reg.transformation, float(reg.fitness), float(reg.inlier_rmse)


def run_icp_multistage(src, tgt, T_init, stages):
    T = np.asarray(T_init, dtype=float)
    fit, rmse = 0.0, 1e9

    # early reject: csak az első stage után dobjuk ki a nagyon rosszakat
    first_mc, first_it = stages[0]
    T, fit, rmse = run_icp_point_to_plane_fast(src, tgt, T, first_mc, first_it)
    if fit < EARLY_REJECT_FITNESS:
        return T, fit, rmse

    for (mc, it) in stages[1:]:
        T, fit, rmse = run_icp_point_to_plane_fast(src, tgt, T, mc, it)
    return T, fit, rmse


def score_fit_rmse(fit, rmse):
    return float(fit) - SCORE_RMSE_WEIGHT * float(rmse)


def jitter_init(T0, model_center):
    dyaw = random.gauss(0.0, JITTER_YAW_SIGMA_DEG)
    droll = random.gauss(0.0, JITTER_ROLLPITCH_SIGMA_DEG)
    dpitch = random.gauss(0.0, JITTER_ROLLPITCH_SIGMA_DEG)

    Trot = rot_z(dyaw) @ rot_x(droll) @ rot_y(dpitch)
    Tpre = T_rotate_about_center(Trot, model_center)

    T = np.asarray(T0 @ Tpre, dtype=float)
    T[:3, 3] += np.array([
        random.gauss(0.0, JITTER_TRANS_SIGMA_M),
        random.gauss(0.0, JITTER_TRANS_SIGMA_M),
        random.gauss(0.0, JITTER_TRANS_SIGMA_M),
    ], dtype=float)
    return T


def _get_model_cached(object_pointcloud):
    # Cache kulcs: ugyanaz az objektum instance -> gyors
    key = (id(object_pointcloud), len(object_pointcloud.points))

    if _MODEL_CACHE["key"] == key and _MODEL_CACHE["model"] is not None:
        return _MODEL_CACHE["model"], _MODEL_CACHE["center"], _MODEL_CACHE["diag"]

    model = copy.deepcopy(object_pointcloud)
    model.transform(axis_fix_matrix(STL_AXIS_FIX))
    if abs(float(STL_SCALE) - 1.0) > 1e-12:
        model.scale(float(STL_SCALE), center=(0, 0, 0))

    model = preprocess(model, VOXEL_MODEL)

    # normál a modellre egyszer (ICP-hez)
    ensure_normals(model, radius=float(VOXEL_MODEL) * 6.0)

    center = np.asarray(model.get_center(), dtype=float)
    diag = float(np.linalg.norm(bbox_extent(model)))

    _MODEL_CACHE["key"] = key
    _MODEL_CACHE["model"] = model
    _MODEL_CACHE["center"] = center
    _MODEL_CACHE["diag"] = diag

    return model, center, diag


# ============================================================
# A KÉRT FÜGGVÉNY
# ============================================================
def do_pose_estimation(scene_pointcloud, object_pointcloud):
    """
    scene_pointcloud: Open3D PointCloud (scene)
    object_pointcloud: Open3D PointCloud (duck template, STL-ből mintázva)
    return: 4x4 T_scene_from_object
    """

    # ---- param set ----
    if FAST_MODE:
        stages = ICP_STAGES_FAST
        yaw_list = YAW_DEG_FAST
        base_list = BASE_ROTS_FAST
        max_clusters = MAX_CLUSTERS_TO_TRY_FAST
        n_restarts = N_RANDOM_RESTARTS_FAST
    else:
        stages = ICP_STAGES_ACCURATE
        yaw_list = YAW_DEG_ACCURATE
        base_list = BASE_ROTS_ACCURATE
        max_clusters = MAX_CLUSTERS_TO_TRY_ACCURATE
        n_restarts = N_RANDOM_RESTARTS_ACCURATE

    # --- preprocess scene ---
    scene = preprocess(scene_pointcloud, VOXEL_SCENE)
    if len(scene.points) < 300:
        return np.eye(4, dtype=float)

    # --- model cache ---
    model, model_center, stl_diag = _get_model_cached(object_pointcloud)
    if len(model.points) < 200:
        return np.eye(4, dtype=float)

    # --- plane detektálás ---
    plane_model, inliers = scene.segment_plane(
        distance_threshold=float(PLANE_DIST),
        ransac_n=3,
        num_iterations=int(PLANE_ITERS)
    )
    a, b, c, d = plane_model
    n = unit([a, b, c])

    # normál +Z felé
    if float(np.dot(n, np.array([0, 0, 1.0], dtype=float))) < 0:
        n = -n

    plane_pcd = scene.select_by_index(inliers)
    wo_plane = scene.select_by_index(inliers, invert=True)

    # --- TABLE frame ---
    T_scene_from_table, T_table_from_scene = build_table_frame_from_plane(
        plane_pcd, n)

    wo_tbl = copy.deepcopy(wo_plane)
    wo_tbl.transform(T_table_from_scene)

    pts = np.asarray(wo_tbl.points)
    if pts.shape[0] < DBSCAN_MINPTS:
        return np.eye(4, dtype=float)
    z = pts[:, 2]

    best = {
        "score": -1e18,
        "T_tbl": None,
        "fit": 0.0,
        "rmse": 1e9,
        "side": None,
        "lbl": None
    }

    for side in TRY_SIDES:
        if side == "TOP":
            mask = (z > ZMIN) & (z < ZMAX)
        else:
            mask = (z < -ZMIN) & (z > -ZMAX)

        candidates = wo_tbl.select_by_index(np.where(mask)[0].tolist())
        if len(candidates.points) < DBSCAN_MINPTS:
            continue

        labels = np.array(candidates.cluster_dbscan(
            eps=float(DBSCAN_EPS), min_points=int(DBSCAN_MINPTS)))
        if labels.size == 0 or labels.max() < 0:
            continue

        # legnagyobb klaszterek
        cand_lbls = list(range(int(labels.max()) + 1))
        sizes = [(l, int(np.sum(labels == l))) for l in cand_lbls]
        sizes.sort(key=lambda x: -x[1])
        sizes = sizes[:int(max_clusters)]

        for lbl, _sz in sizes:
            idx = np.where(labels == lbl)[0]
            cl = candidates.select_by_index(idx.tolist())
            if len(cl.points) < 80:
                continue

            # méretszűrés
            diag = float(np.linalg.norm(bbox_extent(cl)))
            if diag > SIZE_MAX_MULT * stl_diag:
                continue
            if diag < SIZE_MIN_MULT * stl_diag:
                continue

            # klaszter normals egyszer (sok init fut rá)
            ensure_normals(cl, radius=float(stages[0][0]) * 2.0)

            # OBB init
            T0 = estimate_init_T_from_obb(model, cl)

            # determinisztikus multi-init (base + yaw)
            for base_name in base_list:
                B = base_rot(base_name)
                for yaw in yaw_list:
                    Zr = rot_z(yaw)

                    # forgatás a modell saját középpontja körül
                    Tpre = T_rotate_about_center(B @ Zr, model_center)
                    T_init = T0 @ Tpre

                    # 1) determinisztikus
                    T, fit, rmse = run_icp_multistage(
                        model, cl, T_init, stages)
                    sc = score_fit_rmse(fit, rmse)
                    if sc > best["score"]:
                        best.update(score=sc, T_tbl=T, fit=fit,
                                    rmse=rmse, side=side, lbl=lbl)

                    # 2) jitter restarts (kevesebb FAST módban)
                    for _ in range(int(n_restarts)):
                        Tj = jitter_init(T_init, model_center)
                        T2, fit2, rmse2 = run_icp_multistage(
                            model, cl, Tj, stages)
                        sc2 = score_fit_rmse(fit2, rmse2)
                        if sc2 > best["score"]:
                            best.update(score=sc2, T_tbl=T2, fit=fit2,
                                        rmse=rmse2, side=side, lbl=lbl)

    if best["T_tbl"] is None:
        return np.eye(4, dtype=float)

    T_scene_from_object = T_scene_from_table @ best["T_tbl"]

    if DEBUG:
        print(
            f"[DBG] BEST side={best['side']} cluster={best['lbl']} fitness={best['fit']:.4f} rmse={best['rmse']:.6f} FAST_MODE={FAST_MODE}")

    return T_scene_from_object
