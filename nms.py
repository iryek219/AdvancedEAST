# coding=utf-8
import numpy as np

import cfg
import qhull_2d as qhull
import min_bounding_rect as minbr

import matplotlib.pyplot as plt


def should_merge(region, i, j):
    neighbor = {(i, j - 1)}
    return not region.isdisjoint(neighbor)


def region_neighbor(region_set):
    region_pixels = np.array(list(region_set))
    j_min = np.amin(region_pixels, axis=0)[1] - 1
    j_max = np.amax(region_pixels, axis=0)[1] + 1
    i_m = np.amin(region_pixels, axis=0)[0] + 1
    region_pixels[:, 0] += 1
    neighbor = {(region_pixels[n, 0], region_pixels[n, 1]) for n in
                range(len(region_pixels))}
    neighbor.add((i_m, j_min))
    neighbor.add((i_m, j_max))
    return neighbor


def region_group(region_list):
    S = [i for i in range(len(region_list))]
    D = []
    while len(S) > 0:
        m = S.pop(0)
        if len(S) == 0:
            # S has only one element, put it to D
            D.append([m])
        else:
            D.append(rec_region_merge(region_list, m, S))
    return D


def rec_region_merge(region_list, m, S):
    rows = [m]
    tmp = []
    for n in S:
        if not region_neighbor(region_list[m]).isdisjoint(region_list[n]) or \
                not region_neighbor(region_list[n]).isdisjoint(region_list[m]):
            # 第m与n相交
            # Intersection of m and n // google translate
            tmp.append(n)
    for d in tmp:
        S.remove(d)
    for e in tmp:
        rows.extend(rec_region_merge(region_list, e, S))
    return rows


def nms(predict, activation_pixels, vertex_threshold=cfg.side_vertex_pixel_threshold):
    region_list = []
    for i, j in zip(activation_pixels[0], activation_pixels[1]):
        merge = False
        for k in range(len(region_list)):
            if should_merge(region_list[k], i, j):
                region_list[k].add((i, j))
                merge = True
                # Fixme 重叠文本区域处理，存在和多个区域邻接的pixels，先都merge试试
                # Fixme Overlap text area processing, there are pixels adjacent to multiple areas, try merge first // Google Translate
                # break
        if not merge:
            region_list.append({(i, j)})

    #fig, ax = plt.subplot()

    rg = region_group(region_list)
    word_list = np.zeros((len(rg),4,2))
    #group_list = 
    for g, gi in zip(rg, range(len(rg))):
        group_members = []
        for row in g:
            for ij in region_list[row]:
                group_members.append((float(ij[1]), float(ij[0])) )
        if len(group_members)>0:
            xy_points = np.array(group_members)
            xy_points = (xy_points+0.5) * cfg.pixel_size
            hull_points = qhull.qhull2D(xy_points)
            hull_points = hull_points[::-1]
            print('Convex hull points: \n', hull_points, "\n")
            # Find minimum area bounding rectangle
            (rot_angle, area, width, height, center_point, word_list[gi]) \
                = minbr.minBoundingRect(hull_points)
            #plt.scatter(xy_points[:,1], xy_points[:,0])
            #plt.plot(word_list[gi][:,1], word_list[gi][:,0])
    #plt.show()

    D = region_group(region_list)
    quad_list = np.zeros((len(D), 4, 2))
    score_list = np.zeros((len(D), 4))
    for group, g_th in zip(D, range(len(D))):
        total_score = np.zeros((4, 2))
        for row in group:
            for ij in region_list[row]:
                score = predict[ij[0], ij[1], 1]
                if score >= vertex_threshold:
                    ith_score = predict[ij[0], ij[1], 2:3]
                    if not (cfg.trunc_threshold <= ith_score < 1 -
                            cfg.trunc_threshold):
                        ith = int(np.around(ith_score))
                        total_score[ith * 2:(ith + 1) * 2] += score
                        px = (ij[1] + 0.5) * cfg.pixel_size
                        py = (ij[0] + 0.5) * cfg.pixel_size
                        t37 = np.reshape(predict[ij[0], ij[1], 3:7], (2, 2))
                        p_v = [px, py] + t37
                        #p_v = [px, py] + np.reshape(predict[ij[0], ij[1], 3:7], (2, 2))
                        quad_list[g_th, ith * 2:(ith + 1) * 2] += score * p_v
        score_list[g_th] = total_score[:, 0]
        quad_list[g_th] /= (total_score + cfg.epsilon)
    return score_list, quad_list, word_list

