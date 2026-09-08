import json
import os
import math
import sys

import SimpleITK as sitk
import numpy as np
from scipy.interpolate import CubicSpline
import slicer
from skimage.morphology import skeletonize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import cv2
from math import acos, degrees

spinal_canal_labels = {
    "dural_sac": 31,
    "spinal_canal": 32,
    "cauda_equina": 33
}

vertebrae = {
    "th11": 1,
    "th12": 5,
    "l1": 9,
    "l2": 13,
    "l3": 17,
    "l4": 21,
    "l5": 25
    # "s1": 29
}


def save_cuboid_to_nifti(origin, size, dir_cos, spacing=(1, 1, 1), filename='output.nii.gz'):
    image_data = np.ones((round(size[2]), round(size[1]), round(size[0])), dtype=np.float32)

    dir_cos = np.array(dir_cos).ravel()
    direction = [dir_cos[0], dir_cos[3], dir_cos[6],
                 dir_cos[1], dir_cos[4], dir_cos[7],
                 dir_cos[2], dir_cos[5], dir_cos[8]]
    sitk_image = sitk.GetImageFromArray(image_data)
    sitk_image.SetSpacing(spacing)
    sitk_image.SetOrigin(origin)
    sitk_image.SetDirection(direction)

    sitk.WriteImage(sitk_image, filename)


def postprocess_label_image(image):
    component_image = sitk.ConnectedComponent(image)
    sorted_component_image = sitk.RelabelComponent(component_image, sortByObjectSize=True)
    return sorted_component_image == 1


def find_closest_i(points, reference_point):
    closest = np.finfo(np.float32).max
    closest_i = 0
    for i, point in enumerate(points):
        d2 = np.sum((np.array(point) - np.array(reference_point)) ** 2)
        if d2 < closest:
            closest_i = i
            closest = d2
    return closest_i


def vec_from_spline(points, key):
    markup = slicer.build_markups([slicer.build_markup_fiducial(points, "p")])
    with open(f"./target/{key}/y_vec_points.json", "w") as f:
        json.dump(markup, f)
    centered = points - np.mean(points, axis=0)
    # analiza głównych składowych
    pca = PCA(n_components=3)
    pca.fit(centered)
    principal_vector = pca.components_[0]
    with open(f"./target/{key}/y_vector.json", "w") as f:
        json.dump(slicer.build_markups([slicer.build_markup_vector(points[0], points[0] + principal_vector, 'y')]), f)
    return principal_vector / np.linalg.norm(principal_vector)


def detect_orientation(closest, points, key):
    start_i = max(closest - 10, 0)
    end_i = min(closest + 10, len(points) - 1)
    selected_points = points[start_i:end_i + 1]
    return vec_from_spline(selected_points, key)


def estimate_gap(points):
    threshold = 3.2
    expected_step = 3.2

    remaining_indices = [0]
    missing_indices = []

    for i in range(1, len(points)):
        gap = abs(points[i, 2] - points[i - 1, 2])

        if gap >= threshold:
            num_missing = int(gap // expected_step)

            for j in range(0, num_missing):
                missing_indices.append(i + j)

        remaining_indices.append(i + len(missing_indices))

    return remaining_indices, missing_indices


def approximate(points):
    points = np.array(points)
    remaining_indices, missing_indices = estimate_gap(points)
    t = np.linspace(0, 1, num=len(points))
    cs_x = CubicSpline(t, points[:, 0], bc_type='natural')
    cs_y = CubicSpline(t, points[:, 1], bc_type='natural')
    cs_z = CubicSpline(t, points[:, 2], bc_type='natural')

    t_missing = np.interp(missing_indices, remaining_indices, t)
    x_smooth = cs_x(t_missing)
    y_smooth = cs_y(t_missing)
    z_smooth = cs_z(t_missing)
    smooth_coordinates = np.column_stack((x_smooth, y_smooth, z_smooth))  # Convert to NumPy array

    all_points = np.vstack((points, smooth_coordinates))
    all_points = all_points[np.argsort(np.concatenate((remaining_indices, missing_indices)))]

    markup = slicer.build_markups([slicer.build_markup_fiducial(all_points, "p")])
    with open("./target/approx.json", "w") as f:
        json.dump(markup, f)
    return all_points


def prepare_spinal_canal_points(image):
    canal = (image == spinal_canal_labels["spinal_canal"]) | (image == spinal_canal_labels["cauda_equina"]) | (
            image == spinal_canal_labels["dural_sac"])
    closed_canal = canal
    data = sitk.GetArrayFromImage(closed_canal)
    skeleton_lee = skeletonize(data, method='lee')

    skeleton_lee = skeleton_lee.astype(np.uint8)
    res = sitk.GetImageFromArray(skeleton_lee)
    res.CopyInformation(closed_canal)

    shape_filter = sitk.LabelShapeStatisticsImageFilter()
    shape_filter.ComputeOrientedBoundingBoxOn()
    shape_filter.Execute(res)

    origin = shape_filter.GetOrientedBoundingBoxOrigin(1)
    direction = shape_filter.GetOrientedBoundingBoxDirection(1)
    direction = [direction[0], direction[3], direction[6],
                 direction[1], direction[4], direction[7],
                 direction[2], direction[5], direction[8]]
    size = shape_filter.GetOrientedBoundingBoxSize(1)
    resample = sitk.ResampleImageFilter()
    resample.SetOutputOrigin(origin)
    resample.SetOutputDirection(direction)
    spacing = [1.0, 1.0, 2.0]
    resample.SetOutputSpacing(spacing)
    resampled_size = np.ceil(np.array(size).astype(np.float32) / np.array(spacing).astype(np.float32))
    resample.SetSize(resampled_size.astype(np.int32).tolist())
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    canal_resampled = resample.Execute(res)

    z_size = canal_resampled.GetSize()[2]

    canal_points = []
    for z_slice in range(z_size):
        canal_slice = canal_resampled[:, :, z_slice]
        shape_filter = sitk.LabelShapeStatisticsImageFilter()
        shape_filter.Execute(canal_slice)
        num_labels = shape_filter.GetNumberOfLabels()
        if num_labels > 0:
            centroid_2d = shape_filter.GetCentroid(1)
            centroid_index_2d = canal_slice.TransformPhysicalPointToContinuousIndex(centroid_2d)
            centroid_index_3d = [centroid_index_2d[0], centroid_index_2d[1], z_slice]
            centroid_3d = canal_resampled.TransformContinuousIndexToPhysicalPoint(centroid_index_3d)
            canal_points.append(centroid_3d)

    markups = slicer.build_markups([slicer.build_markup_fiducial(canal_points, 'b')])
    with open("./target/canal_points.json", "w") as f:
        json.dump(markups, f)
    return approximate(canal_points)


def cobb_angle(v1, v2, x_vec, y_vec):
    u1 = np.array([np.dot(v1, x_vec), np.dot(v1, y_vec)])
    u2 = np.array([np.dot(v2, x_vec), np.dot(v2, y_vec)])

    dot = np.dot(u1, u2)
    mag1 = np.linalg.norm(u1)
    mag2 = np.linalg.norm(u2)

    cos_angle = dot / (mag1 * mag2)
    angle = math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))

    cross = u1[0] * u2[1] - u1[1] * u2[0]
    signed_angle = angle if cross >= 0 else -angle

    direction = "prawostronna" if signed_angle > 0 else "lewostronna" if signed_angle < 0 else "neutral"

    return signed_angle, direction


def extract_measurement_slice(image, plane_anchor, x_vector, y_vector, z_vector, name):
    resample_direction = [
        x_vector[0], y_vector[0], z_vector[0],
        x_vector[1], y_vector[1], z_vector[1],
        x_vector[2], y_vector[2], z_vector[2],
    ]

    size = [250, 250, 1]
    spacing = [0.5, 0.5, 1.0]
    # resample_origin = plane_anchor - x_vector * 125 - y_vector * 125
    resample_origin = plane_anchor

    save_cuboid_to_nifti(resample_origin, size, [x_vector, y_vector, z_vector], spacing=spacing,
                         filename=f"./target/{name}/plane.nii.gz")
    resample = sitk.ResampleImageFilter()
    resample.SetSize(size)
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    resample.SetOutputSpacing(spacing)
    resample.SetOutputOrigin(resample_origin.tolist())
    resample.SetOutputDirection(resample_direction)
    resample.SetOutputPixelType(sitk.sitkUInt8)
    resampled = resample.Execute(image)
    # sitk.WriteImage(resampled, f"./target/{name}/resampled.nii.gz")
    return resampled


def extract_vertebra_contour(flat: np.ndarray, name):
    def select_corner_points(points):
        sum_xy = points[:, 0] + points[:, 1]
        diff_xy = points[:, 0] - points[:, 1]

        top_left = points[np.argmin(sum_xy)]
        bottom_right = points[np.argmax(sum_xy)]
        top_right = points[np.argmax(diff_xy)]
        bottom_left = points[np.argmin(diff_xy)]

        return np.array((top_left, top_right, bottom_left, bottom_right))

    # wygładzenie szumu z obrazu
    cleaned = cv2.morphologyEx(flat, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=2)
    # filtr Canny detekcja krawędzi
    edges = cv2.Canny(cleaned, 0, 1)
    # plt.imshow(edges)
    # plt.title(f"Krawędzie {name.upper()}")
    # plt.axis("off")
    # plt.show()
    # znalezienie konturów
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = np.vstack(contours).squeeze(1)

    # plt.figure(figsize=(6, 6))
    # plt.scatter(contours[:, 0], contours[:, 1], color='b', alpha=0.3, s=5, label="Kontury - findContours")
    # plt.title(f"Kontur {name.upper()}")
    # otoczka wypukła
    hull = cv2.convexHull(contours)[:, 0, :]
    # plt.scatter(hull[:, 0], hull[:, 1], color='r', alpha=0.5, s=15, label=f"Otoczka wypukła - convexHull")

    approx = cv2.approxPolyDP(hull, cv2.arcLength(hull, True) * 0.01, True)[:, 0, :]
    # plt.scatter(approx[:, 0], approx[:, 1], color='orange', alpha=0.7, s=35, label=f"Wielomian - approxPolyDP")
    corner_points = select_corner_points(approx)
    # plt.scatter(corner_points[:, 0], corner_points[:, 1], color='g', alpha=0.8, s=50, label=f"Wybrane narożne punkty")
    # plt.gca().invert_yaxis()
    # plt.legend()
    # plt.axis("off")
    # plt.show()
    return corner_points


def fit_plane_svd(points):
    """Dopasowuje płaszczyznę do zbioru punktów przy użyciu SVD."""
    points = np.asarray(points)
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered)
    normal = vh[-1]  # najmniejsza wartość osobliwa
    return centroid, normal


def project_onto_plane(points, plane_point, normal):
    """Rzutuje punkty ortogonalnie na płaszczyznę."""
    normal = normal / np.linalg.norm(normal)
    projected = []
    for p in points:
        v = p - plane_point
        dist = np.dot(v, normal)
        projected.append(p - dist * normal)
    return np.array(projected)


def algo(image):
    canal_points = prepare_spinal_canal_points(image)
    points = np.array(canal_points)
    diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
    mask = np.insert(diffs > 1e-6, 0, True)  # keep first point, remove near-duplicates
    points = points[mask]
    n_points = len(points)

    # Parameterize by cumulative distance
    distances = np.cumsum(np.r_[0, np.linalg.norm(np.diff(points, axis=0), axis=1)])

    # Create cubic splines
    spline_x = CubicSpline(distances, points[:, 0])
    spline_y = CubicSpline(distances, points[:, 1])
    spline_z = CubicSpline(distances, points[:, 2])

    # Sample twice as many points
    new_distances = np.linspace(distances[0], distances[-1], n_points * 2)

    # Interpolate
    new_points = np.vstack([
        spline_x(new_distances),
        spline_y(new_distances),
        spline_z(new_distances)
    ]).T
    markup = slicer.build_markups([slicer.build_markup_fiducial(new_points, "p")])
    with open("./target/approx2.json", "w") as f:
        json.dump(markup, f)

    canal_points = new_points

    keys = list(vertebrae.keys())
    vertebrae_vectors = {}
    upper_points = {}
    lower_points = {}
    for (key, value) in vertebrae.items():
        try:
            os.makedirs(f"./target/{key}", exist_ok=True)
            body = image == value
            shape_filter = sitk.LabelShapeStatisticsImageFilter()
            shape_filter.Execute(body)
            body_centroid = shape_filter.GetCentroid(1)
            closest_i = find_closest_i(canal_points, body_centroid)

            y_vec = detect_orientation(closest_i, canal_points, key)  # superior-inferior
            y_vec /= np.linalg.norm(y_vec)
            if y_vec[2] > 0:
                y_vec = -y_vec

            with open(f"./target/{key}/y_vector.json", "w") as f:
                json.dump(slicer.build_markups(
                    [slicer.build_markup_vector(np.array(body_centroid), np.array(body_centroid) + y_vec * 20, 'z')]),
                    f)

            # Medial-lateral: from vertebral centroid to spinal canal
            z_vec = np.array(canal_points[closest_i]) - np.array(body_centroid)
            with open(f"./target/{key}/z_vector.json", "w") as f:
                json.dump(slicer.build_markups(
                    [slicer.build_markup_vector(np.array(body_centroid), np.array(body_centroid) + z_vec, 'z')]), f)
            z_vec /= np.linalg.norm(z_vec)

            # Anterior-posterior: perpendicular to both iloczyn wektorowy
            x_vec = np.cross(z_vec, y_vec)
            x_vec /= np.linalg.norm(x_vec)

            z_vec = np.cross(y_vec, x_vec)
            z_vec /= np.linalg.norm(z_vec)

            anchor = np.array(body_centroid) - x_vec * 50 - y_vec * 50
            resampled = extract_measurement_slice(body, anchor, x_vec, y_vec, z_vec, key)
            flat = sitk.GetArrayFromImage(sitk.BinaryFillhole(resampled[:, :, 0]))
            # if key == 'l2':
            #     plt.imshow(flat)
            #     plt.title(f"Resampled L2 vertebra")
            #     plt.axis("off")
            #     plt.show()
            points = extract_vertebra_contour(flat, key)

            start_top = resampled.TransformContinuousIndexToPhysicalPoint(
                [float(points[0][0]), float(points[0][1]), 0.0])
            end_top = resampled.TransformContinuousIndexToPhysicalPoint([float(points[1][0]), float(points[1][1]), 0.0])
            start_bot = resampled.TransformContinuousIndexToPhysicalPoint(
                [float(points[2][0]), float(points[2][1]), 0.0])
            end_bot = resampled.TransformContinuousIndexToPhysicalPoint([float(points[3][0]), float(points[3][1]), 0.0])
            markup = slicer.build_markups([slicer.build_markup_fiducial([start_top, end_top, start_bot, end_bot], "p")])
            os.makedirs('./target/contours', exist_ok=True)
            with open(f"./target/contours/{key}.json", "w") as f:
                json.dump(markup, f)
            upper_points[key] = (start_top, end_top)
            lower_points[key] = (start_bot, end_bot)
        except Exception as e:
            print(f"Błąd dla kręgu {key}: {e}", file=sys.stderr)

    max_cobb_angle = -math.inf
    max_dir = None
    max_pair = ''
    for i in range(len(keys)):
        for j in range(i + 2, len(keys)):
            upper_key = keys[i]
            lower_key = keys[j]
            if not (upper_key in upper_points) or not (lower_key in lower_points):
                continue
            pts = np.array([upper_points[upper_key], lower_points[upper_key], upper_points[lower_key],
                            lower_points[lower_key]]).reshape(-1, 3)
            os.makedirs(f"./target/{upper_key}_{lower_key}", exist_ok=True)
            with open(f"./target/{upper_key}_{lower_key}/before_mapping_points.json", "w") as f:
                json.dump(slicer.build_markups([slicer.build_markup_vector(upper_points[upper_key][0], upper_points[upper_key][1], "UPPER"),
                                                slicer.build_markup_vector(lower_points[lower_key][0], lower_points[lower_key][1], "LOWER")]), f)
            plane_point, normal = fit_plane_svd(pts)

            c_upper = np.mean(pts[0:4], axis=0)
            c_lower = np.mean(pts[4:8], axis=0)
            spine_vec = c_upper - c_lower
            z_vec = normal
            x_vec = np.cross(spine_vec, z_vec)
            x_vec /= np.linalg.norm(x_vec)
            y_vec = np.cross(z_vec, x_vec)
            y_vec /= np.linalg.norm(y_vec)

            origin = plane_point - x_vec * 50 - y_vec * 60
            save_cuboid_to_nifti(origin, [120, 150, 1], [x_vec, y_vec, z_vec], spacing=[1, 1, 1],
                                 filename=f"./target/{upper_key}_{lower_key}/common_plane.nii.gz")

            projected = project_onto_plane(pts, plane_point, normal)

            upper_pts = np.array([projected[0], projected[1]])
            lower_pts = np.array([projected[6], projected[7]])
            with open(f"./target/{upper_key}_{lower_key}/mapped_points.json", "w") as f:
                json.dump(slicer.build_markups([slicer.build_markup_vector(upper_pts[1], upper_pts[0], "UPPER"),
                                                slicer.build_markup_vector(lower_pts[1], lower_pts[0], "LOWER")]), f)

            upper_top_vec = np.array(upper_pts[1]) - np.array(upper_pts[0])
            lower_bot_vec = np.array(lower_pts[1]) - np.array(lower_pts[0])
            if upper_key == 'l3' and lower_key == 'l5':
                None
                # print(f"{upper_top_vec} {lower_bot_vec}")

            x, direction = cobb_angle(upper_top_vec, lower_bot_vec, x_vec, y_vec)
            x = abs(x)
            if x > max_cobb_angle:
                max_cobb_angle = x
                max_pair = f"{upper_key}-{lower_key}"
                max_dir = direction
            # print(f"Angle between {upper_key} and  {lower_key} = {x:0.2f} {direction}")
    print(f"{max_pair}: {max_cobb_angle:.3f} {max_dir}")


def main():
    if len(sys.argv) == 2:
        nifti_file = sys.argv[1]
    else:
        nifti_file = '../data/case64/seg.nii.gz'
    image = sitk.ReadImage(nifti_file)
    algo(image)


# todo kierunek skoliozy (prawoboczna/lewoboczna)
if __name__ == "__main__":
    main()
