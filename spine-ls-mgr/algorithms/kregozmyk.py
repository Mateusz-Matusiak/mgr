import json
import os

import SimpleITK as sitk
import numpy as np
from scipy.interpolate import CubicSpline

import slicer

import matplotlib.pyplot as plt
import cv2
from skimage.morphology import skeletonize
from sklearn.decomposition import PCA


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


def vec_from_spline(points):
    centered = points - np.mean(points, axis=0)
    pca = PCA(n_components=3)
    pca.fit(centered)
    principal_vector = pca.components_[0]
    # with open(f"./target/{key}/y_vector.json", "w") as f:
    #     json.dump(slicer.build_markups([slicer.build_markup_vector(points[0], points[0] + principal_vector, 'y')]), f)
    return principal_vector / np.linalg.norm(principal_vector)


def detect_orientation(closest, points):
    start_i = max(closest - 10, 0)
    end_i = min(closest + 10, len(points) - 1)
    selected_points = points[start_i:end_i + 1]
    return vec_from_spline(selected_points)


def extract_measurement_slice(image, plane_anchor, x_vector, y_vector, z_vector, name):
    resample_direction = [
        x_vector[0], y_vector[0], z_vector[0],
        x_vector[1], y_vector[1], z_vector[1],
        x_vector[2], y_vector[2], z_vector[2],
    ]

    size = [500, 500, 1]
    spacing = [0.5, 0.5, 1.0]
    # resample_origin = plane_anchor - x_vector * 125 - y_vector * 125
    resample_origin = plane_anchor

    # save_cuboid_to_nifti(resample_origin, size, [x_vector, y_vector, z_vector], spacing=spacing,
    #                      filename=f"./target/{name}/plane.nii.gz")
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

    cleaned = cv2.morphologyEx(flat, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=2)
    edges = cv2.Canny(cleaned, flat.min(), flat.max() - flat.min())
    # plt.imshow(edges)
    # plt.title(f"Edges of {name}")
    # plt.show()
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = np.vstack(contours).squeeze(1)

    # plt.figure(figsize=(6, 6))
    # plt.scatter(contours[:, 0], contours[:, 1], color='b', linewidth=0.1, label="Contrours")
    # plt.title(f"Contour {name}")
    hull = cv2.convexHull(contours)[:, 0, :]
    # plt.scatter(hull[:, 0], hull[:, 1], color='r', linewidth=0.1, label=f"Convex hull")
    epsilon = 0.01 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.01, True)[:, 0, :]
    corner_points = select_corner_points(approx)
    # plt.scatter(corner_points[:, 0], corner_points[:, 1], color='g', linewidth=0.1, label=f"Selected points")
    # plt.gca().invert_yaxis()
    # plt.legend()
    # plt.show()
    return corner_points


def find_measurement_points(image, plane_anchor, x_vector, y_vector, z_vector, name):
    os.makedirs(f"./target/{name}", exist_ok=True)
    resampled = extract_measurement_slice(image, plane_anchor, x_vector, y_vector, z_vector, name)
    flat = sitk.GetArrayFromImage(sitk.BinaryFillhole(resampled[:, :, 0]))
    points = extract_vertebra_contour(flat, name)

    start_top = resampled.TransformContinuousIndexToPhysicalPoint([float(points[0][0]), float(points[0][1]), 0.0])
    end_top = resampled.TransformContinuousIndexToPhysicalPoint([float(points[1][0]), float(points[1][1]), 0.0])
    start_bot = resampled.TransformContinuousIndexToPhysicalPoint([float(points[2][0]), float(points[2][1]), 0.0])
    end_bot = resampled.TransformContinuousIndexToPhysicalPoint([float(points[3][0]), float(points[3][1]), 0.0])
    markup = slicer.build_markups([slicer.build_markup_fiducial([start_top, end_top, start_bot, end_bot], "p")])
    with open(f"./target/{name}/contour.json", "w") as f:
        json.dump(markup, f)
    vertebrae_points[name] = (start_top, end_top, start_bot, end_bot)
    return start_top, end_top, start_bot, end_bot


def estimate_gap(points):
    threshold = 3.2
    expected_step = 3.2

    remaining_indices = [0]
    missing_indices = []

    for i in range(1, len(points)):
        gap = abs(points[i, 2] - points[i - 1, 2])

        if gap >= threshold:
            num_missing = int(gap // expected_step)

            for j in range(1, num_missing + 1):
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


def prepare_spinal_canal_points():
    canal = (image == spinal_canal_labels["dural_sac"]) | (image == spinal_canal_labels["spinal_canal"]) | \
            (image == spinal_canal_labels["cauda_equina"])
    closed_canal = sitk.BinaryMorphologicalClosing(canal, kernelRadius=(5, 5, 5))
    sitk.WriteImage(closed_canal, "closed_canal.nii.gz")
    data = sitk.GetArrayFromImage(closed_canal)
    skeleton_lee = skeletonize(data, method='lee')

    skeleton_lee = skeleton_lee.astype(np.uint8)
    res = sitk.GetImageFromArray(skeleton_lee)
    res.CopyInformation(closed_canal)
    sitk.WriteImage(res, './target/skeleton.nii.gz')

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
    spacing = [1.0, 1.0, 3.0]
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

    return approximate(canal_points)


def postprocess_vertebra(image, vertebra):
    upper = image == vertebra
    return postprocess_label_image(upper)


def get_y_vec(closest, canal_points):
    y_vec = detect_orientation(closest, canal_points)
    y_vec /= np.linalg.norm(y_vec)
    if y_vec[2] > 0:
        y_vec = -y_vec
    return y_vec



#todo fix 41
os.makedirs("./target", exist_ok=True)
image = sitk.ReadImage("../data/case1/seg.nii.gz")

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
    "l5": 25,
    "s1": 29
}

vertebrae_points = {}

canal_points = prepare_spinal_canal_points()

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

canal_points = new_points
markups = slicer.build_markups([slicer.build_markup_fiducial(canal_points, 'c')])
with open("./target/canal_points.json", "w") as f:
    json.dump(markups, f)

keys = list(vertebrae.keys())
values = list(vertebrae.values())

for (key1, value1), (key2, value2) in zip(zip(keys, values), zip(keys[1:], values[1:])):
    upper = postprocess_vertebra(image, value1)
    filter = sitk.LabelShapeStatisticsImageFilter()
    filter.Execute(upper)
    upper_centroid = filter.GetCentroid(1)
    lower = postprocess_vertebra(image, value2)
    filter = sitk.LabelShapeStatisticsImageFilter()
    filter.Execute(lower)
    lower_centroid = filter.GetCentroid(1)
    upper_closest = find_closest_i(canal_points, upper_centroid)
    lower_closest = find_closest_i(canal_points, lower_centroid)
    y_vec_upper = get_y_vec(upper_closest, canal_points)
    y_vec_lower = get_y_vec(lower_closest, canal_points)
    with open(f"./target/{key2}/y_vector.json", "w") as f:
        json.dump(slicer.build_markups(
            [slicer.build_markup_vector(np.array(lower_centroid), np.array(lower_centroid) + y_vec_lower * 20, 'y')]),
            f)
    x_vec_upper = np.array(canal_points[upper_closest]) - np.array(lower_centroid)
    x_vec_upper /= np.linalg.norm(x_vec_upper)
    x_vec_lower = np.array(canal_points[lower_closest]) - np.array(lower_centroid)
    x_vec_lower /= np.linalg.norm(x_vec_lower)
    with open(f"./target/{key2}/x_vector.json", "w") as f:
        json.dump(slicer.build_markups(
            [slicer.build_markup_vector(np.array(lower_centroid), np.array(lower_centroid) + x_vec_lower * 20, 'x')]),
            f)
    z_vec_upper = np.cross(x_vec_upper, y_vec_upper)
    z_vec_lower = np.cross(x_vec_lower, y_vec_lower)
    with open(f"./target/{key2}/z_vector.json", "w") as f:
        json.dump(slicer.build_markups(
            [slicer.build_markup_vector(np.array(lower_centroid), np.array(lower_centroid) + z_vec_lower * 20, 'z')]),
            f)
    anchor_upper = np.array(upper_centroid) - x_vec_upper * 100 - y_vec_upper * 100
    anchor_lower = np.array(lower_centroid) - x_vec_lower * 100 - y_vec_lower * 100

    ((up_top_end, up_top_start, up_bot_end, up_bot_start)) = vertebrae_points[
        key1] if key1 in vertebrae_points.keys() else find_measurement_points(upper, anchor_upper, x_vec_upper,
                                                                              y_vec_upper,
                                                                              z_vec_upper, key1)

    ((low_top_end, low_top_start, low_bot_end, low_bot_start)) = vertebrae_points[
        key2] if key2 in vertebrae_points.keys() else find_measurement_points(lower, anchor_lower, x_vec_lower,
                                                                              y_vec_lower,
                                                                              z_vec_lower, key2)

    AB = np.array(low_top_end) - np.array(low_top_start)

    with open(f"./target/{key1}/AB_VECTOR.json", 'w') as f:
        json.dump(slicer.build_markups([slicer.build_markup_vector(low_top_end, low_top_start, "AB")]), f)
    with open(f"./target/{key1}/AP_VECTOR.json", 'w') as f:
        json.dump(slicer.build_markups([slicer.build_markup_vector(up_bot_start, low_top_start, "AP")]), f)
    AP = np.array(up_bot_start) - np.array(low_top_start)
    result = np.array(low_top_start) + np.dot(AP, AB) / np.dot(AB, AB) * AB

    one_vector = np.array(result) - np.array(low_top_start)
    x1 = np.linalg.norm(one_vector)
    second_vector = np.array(low_top_end) - np.array(low_top_start)
    x2 = np.linalg.norm(second_vector)
    direction = np.dot(one_vector, second_vector)
    with open(f"./target/{key1}/mapped_point.json", 'w') as f:
        json.dump(slicer.build_markups([slicer.build_markup_fiducial([result], "m")]), f)
    print(f"Result of {'kregozmyk' if direction > 0 else 'tylozmyk'} between {key1} and {key2}: {x1 / x2:0.3f}")
