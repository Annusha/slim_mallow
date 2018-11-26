#!/usr/bin/env python

"""Log parser
"""

__author__ = 'Anna Kukleva'
__date__ = 'September 2018'


import numpy as np
import os
import pandas as pd
from collections import defaultdict
import re


def grid_search_parser(path):
    counter = 0
    frames = []
    mof = []
    set_idx = -1
    set_lines = []
    avg_mof = []
    bg = []
    with open(path, 'r') as f:
        for line in f:
            if 'SET' in line:
                if set_idx > -1:
                    if not bg:
                        yield mof[0], avg_mof[0], mof[-1], avg_mof[-1], set_lines[-1], frames[-1]
                    else:
                        yield mof[0], avg_mof[0], bg[0], mof[-1], avg_mof[-1], bg[-1], set_lines[-1], frames[-1]
                print(line)
                counter = 0
                frames = []
                mof = []
                avg_mof = []
                set_idx += 1
                bg = []
                set_lines.append(line)
            if 'Iteration' in line:
                line = line.split()[-1]
                counter = int(line)
            if 'MoF' in line:
                if 'old' in line:
                    continue
                print(counter, line)
                line = line.split()[-1]
                mof.append(float(line))
            if 'frames' in line:
                line = line.split()[-5]
                frames.append(int(line))
            if 'average class mof' in line:
                line = line.split()[-1]
                avg_mof.append(float(line))
                print(counter, line)
            if 'total background' in line:
                line = line.split()[-1]
                bg.append(int(line))
    try:
        yield mof[0], avg_mof[0], mof[-1], avg_mof[-1], set_lines[-1], frames[-1]
    except IndexError:
        pass


def create_table(path):
    # table_path = '/media/data/kukleva/lab/Breakfast/tables'
    table_path = '/media/data/kukleva/lab/YTInstructions/tables'
    data = {}
    for values in grid_search_parser(path):
        try:
            f_mof, f_av_mof, l_mof, l_av_mof, set_n, frames = values
        except ValueError:
            f_mof, f_av_mof, f_bg, l_mof, l_av_mof, l_bg, set_n, frames = values

        data[set_n] = [f_mof, f_av_mof, f_bg, l_mof, l_av_mof, l_bg, frames]

    df = pd.DataFrame(data)
    print(df)
    name = path.split('/')[-1] + '.csv'
    df.to_csv(os.path.join(table_path, name))


def params_parser(path):

    params = defaultdict(list)
    set_idx = 0

    def reset(line):
        nonlocal params, set_idx
        params = defaultdict(list)
        set_idx += 1

        search = re.search(r'dim:\s*(\d*)\s*epochs:\s*(\d*)\s*,\s*lr:\s*(\d*.\d*e-\d*)\s*', line)
        params['dim'] = int(search.group(1))
        params['epochs'] = int(search.group(2))
        params['lr'] = float(search.group(3))
        params['idx'] = set_idx

    with open(path, 'r') as f:
        frames = []
        for line in f:
            if 'wrap - <function pose_training' in line:
                yield params

            if 'SET' in line:
                print('%d : %s' % (set_idx, line))
                reset(line)

            if 'clustering - MoF val' in line:
                params['cl_mof'].append(float(line.split()[-1]))

            if 'accuracy_corpus - MoF val' in line:
                params['mof'].append(float(line.split()[-1]))
                params['frames'].append(frames[-1])

            if 'accuracy_corpus - pure vit MoF val' in line:
                params['!o_mof'].append(float(line.split()[-1]))
                params['!o_frames'].append(frames[-1])

            # if 'training_embed.py - training - loss:' in line:
            #     params['loss'].append(float(line.split()[-1]))

            if 'mof_val - frames true:' in line:
                search = re.search(r'frames true:\s*(\d*)\s*frames overall :\s*(\d*)', line)
                frames.append(int(search.group(1)))

            if 'average class mof:' in line:
                params['av_mof'].append(float(line.split()[-1]))


def create_table_params(path, prefix=''):
    table_path = '/media/data/kukleva/lab/Breakfast/tables'
    # table_path = '/media/data/kukleva/lab/YTInstructions/tables'
    data = defaultdict(list)

    for params in params_parser(path):
        if not params:
            continue

        name = (params['dim'], params['epochs'], params['lr'])
        if not name[0]:
            name = 'set'
        # data[name].append(params['loss'][-1])
        data[name].append(int(100 * params['cl_mof'][0]))

        # mof = [int(i * 100) for i in params['mof']]
        mof = [int(100 * params['mof'][0]), int(100 * params['!o_mof'][0])]  # for pure vit
        data[name].append(mof)

        av_mof = [int(i * 100) for i in params['av_mof']]
        data[name].append(av_mof)

        frames = [params['frames'][0], params['!o_frames'][0]]
        data[name].append(frames)
        data[name].append(frames[-1])
        # data[name].append(params['frames'])
        # data[name].append(params['frames'][1])


    df = pd.DataFrame(data)
    # decimals = pd.Series([2], index=['cl_mof'])
    # df.round(decimals)
    print(df)
    name = prefix + path.split('/')[-1] + '.csv'
    df.to_csv(os.path.join(table_path, name))


def yti_parser(p):
    with open(p) as f:
        action = ''
        mof = 0
        mof_bg = 0
        trh = 0
        f1 = 0
        for line in f:
            if 'accuracy_corpus - Action:' in line:
                action = line.split()[-1]
            if 'accuracy_corpus - MoF val:' in line:
                mof = line.split()[-1]
            if 'mof_classes - mof with bg:' in line:
                mof_bg = line.split()[-1]
            if 'all_actions - bg_trh' in line:
                trh = line.split()[-1]
            if 'f1_score.py - f1 - f1 score:' in line:
                f1 = line.split()[-1]
            if 'utils.py - wrap - <function pose_training ' in line:
                print('Action: %s, %s\n'
                      'mof: %s\n'
                      'mof with bg: %s\n'
                      'f1 score: %s\n' % (action, trh, mof, mof_bg, f1))

        print('__________________________________________\n')




if __name__ == '__main__':
    # path = '/media/data/kukleva/lab/logs_debug/grid_search_mlp_tea_pipeline_mlp_full_2_20_gmm(1)_adj_lr1.0e-03_ep30(2018-09-17 21:49:14.462660)'
    # create_table(path)

    path = '/media/data/kukleva/lab/Breakfast/logs/grid.vit._coffee_mlp_!pose_full_vae1_time10.0_epochs30_embed20_n2_ordering_gmm1_one_!gt_lr0.001_lr_zeros_b0_v1_l0_c1_pipeline(2018-10-24 23:06:18.032832)'
    create_table_params(path)


