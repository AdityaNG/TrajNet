import math

import numpy as np


def intersection(start1, end1, start2, end2):
    if start2 < start1:
        start1, end1, start2, end2 = start2, end2, start1, end1
    if end1 < start2:
        return (None, None)
    return (max([start1, start2]), min([end1, end2]))


def union(start1, end1, start2, end2):
    return (min([start1, start2]), max([end1, end2]))


def intersection_of_group(group):
    start, end = group[0]
    for start_i, end_i in group:
        start, end = intersection(start, end, start_i, end_i)
    return start, end


def union_of_group(group):
    start, end = group[0]
    for start_i, end_i in group:
        start, end = union(start, end, start_i, end_i)
    return start, end


def IOU_of_group(group):
    start_intersect, end_intersect = intersection_of_group(group)
    start_union, end_union = union_of_group(group)
    if end_union == start_union:
        return float("inf")
    return float(end_intersect - start_intersect) / float(
        end_union - start_union
    )


def translate_rotate(p, x, y, theta):
    p_dtype = p.dtype
    p = np.array(
        [
            p[0] * math.cos(theta) - p[1] * math.sin(theta),
            p[0] * math.sin(theta) + p[1] * math.cos(theta),
        ],
        dtype=p_dtype,
    )
    # p = p[0]-x, p[1]-y
    p = p - np.array([x, y], dtype=p_dtype)

    return p
