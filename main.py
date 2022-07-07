import itertools
import os

import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sn
import matplotlib.pyplot as plt


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def convert_name(index):
    names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
             'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
             'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
             'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
             'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
             'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
             'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
             'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
             'hair drier', 'toothbrush']  # class names
    return names[index]


def deduplicate(a):
    return list(dict.fromkeys(a))


def combinations_list(a):
    return list(itertools.permutations(a, 2))


def polar_cordinates(act, video, frame, objlist, combination):
    distances = []
    if combination:
        indexes = []
        for oi in list(dict.fromkeys(objlist)):
            temp_i = stat_frame[(stat_frame['action'] == act) & (stat_frame['video'] == video) & (stat_frame['frame'] == frame) & (stat_frame['classes'] == oi)].index.values
            for eleme in range(len(temp_i)):
                indexes.append(temp_i[eleme])
        indperm = list(itertools.permutations(indexes, 2))
        for inds in indperm:
            xcenter1 = stat_frame['xcenter'].iloc[inds[0]]
            ycenter1 = stat_frame['ycenter'].iloc[inds[0]]
            xcenter2 = stat_frame['xcenter'].iloc[inds[1]]
            ycenter2 = stat_frame['ycenter'].iloc[inds[1]]
            x = xcenter2 - xcenter1
            y = ycenter2 - ycenter1
            dis = np.sqrt((x ** 2) + (y ** 2))
            ang = np.arctan2(y, x)
            distances.append([dis, ang])
    return distances


def frame_mean(obj, objc, combinations, polar_val):
    mean = [0, 0]
    iteration = 0
    for co in range(len(combinations)):
        if combinations[co] == (obj, objc):
            iteration = iteration + 1
            mean[0] = polar_val[co][0] + mean[0]
            mean[1] = polar_val[co][1] + mean[1]
    mean[:] = [x / iteration for x in mean]
    return mean


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    action_path = Path('C:\\Users\\gap201\\Documents\\UCF_Yolo\\detect\\')
    list_actions = os.listdir(action_path)
    for action in list_actions:
        print(action)
        file_path = 'C:\\Users\\gap201\\Documents\\UCF_Yolo\\detect\\' + action + '\\'
        stat_frame = pd.read_csv(Path('C:\\Users\\gap201\\Documents\\UCF_Yolo\\Actions\\detect' + action + '.txt'), index_col=None)
        stat_frame = stat_frame.rename(columns={"class": "classes"})
        # Objects distributions -> how many of an object per frame ( x-> obj number y-> number of frames lines-> top t obj classes)

        list_class_count = stat_frame.groupby(['action', 'video', 'frame', 'classes']).size()
        list_class_distribution = list_class_count.groupby(['classes', list_class_count.values]).size()
        df = pd.DataFrame()
        df[['classes', 'class_dit']] = list(list_class_distribution.index)
        df['values'] = list_class_distribution.values
        top_classes = []
        t_val = 5
        for d in df['class_dit'].unique():
            temp_df = df[df.class_dit.isin([d])].nlargest(t_val, 'values')
            top_classes.append(temp_df['classes'].unique())
        top_classes = np.concatenate(top_classes, axis=0)
        top_class_list = list(dict.fromkeys(top_classes))
        for c in top_class_list:
            sub_lis = df[df.classes.isin([c])]
            plt.plot(sub_lis['class_dit'], sub_lis['values'], label=convert_name(c))
        plt.legend(ncol=3, loc="upper right")
        plt.title('Object Distribution')
        plt.xlabel('Number of objects')
        plt.ylabel('Number of frames')
        plt.savefig(Path(file_path + 'Objects_per_Frame'))
        plt.close()
        # plt.show(block=False)

        # Objt count -> mean apperance of objects during the video (x -> obj classes y-> average frame apperance)

        list_class_count_per_video = stat_frame.groupby(['action', 'video', 'classes']).size()
        list_frames_max_per_video = stat_frame.groupby(['action', 'video'])['frame'].max()
        list_class_average_per_video = list_class_count_per_video / list_frames_max_per_video
        df_mean = pd.DataFrame()
        df_mean[['action', 'video', 'classes']] = list(list_class_average_per_video.index)
        df_mean['values'] = list_class_average_per_video.values
        bar_df = pd.DataFrame()
        for tc in top_class_list:
            sub_mean = df_mean[df_mean.classes.isin([tc])]
            bins = [0, 1, 2, 3, 4, 5, 6]
            labels = [1, 2, 3, 4, 5, 6]
            sub_mean = pd.cut(sub_mean['values'], bins=bins, labels=labels).value_counts()
            sub_mean = sub_mean.sort_index()
            bar_df[convert_name(tc)] = sub_mean
        ax = bar_df.plot.bar()
        ax.set_title('Object Distribution')
        ax.set_xlabel('Number of objects')
        ax.set_ylabel('Video Time (%)')
        fig = ax.get_figure()
        fig.savefig(Path(file_path + 'Objects_per_Video'))

        # Obj position scatter plot

        list_dimension_mean = stat_frame.groupby(['action', 'classes']).mean()
        list_dimension_mean[['action', 'classes']] = list(list_dimension_mean.index)
        df_scatter = pd.DataFrame()
        plt.figure()
        for tcs in top_class_list:
            df_scatter = list_dimension_mean[list_dimension_mean.classes.isin([tcs])]
            plt.scatter(df_scatter['xcenter'], df_scatter['ycenter'], label=convert_name(tcs))
        plt.legend(ncol=3, loc="upper right")
        plt.grid()
        plt.title('Object Positions')
        plt.xlabel('Normalized x')
        plt.ylabel('Normalized y')
        plt.savefig(Path(file_path + 'Objects_mean_Position_per_Video'))
        plt.close()
        # plt.show(block=False)

        # Relative positions

        list_class_in_frame = stat_frame.groupby(['action', 'video', 'frame']).classes.apply(list).reset_index()
        list_combinations_in_frame = list_class_in_frame
        list_combinations_in_frame['combinations'] = list_combinations_in_frame.classes.apply(combinations_list)
        list_combinations_in_frame['polar_dist'] = list_combinations_in_frame.apply(lambda x: polar_cordinates(x.action, x.video, x.frame, x.classes, x.combinations), axis=1)
        centers = top_class_list  # [0] #stat_frame['classes'].unique()
        ''' # mean
        for indc in centers:
            sub_relative = list_combinations_in_frame[list_combinations_in_frame['classes'].apply(lambda x: indc in x)]
            value_mean_dist = []
            value_mean_angle = []
            value_mean_class = []
            for idc in top_class_list:
                temp_sub_relative = sub_relative[sub_relative['combinations'].apply(lambda x: (indc,idc) in x)]
                temp_sub_relative['frame_obj_mean'] = temp_sub_relative.apply(lambda x: frame_mean(indc,idc,x.combinations,x.polar_dist), axis=1)
                temp_sub_relative[['distance', 'angle']] = pd.DataFrame(temp_sub_relative.frame_obj_mean.tolist(), index=temp_sub_relative.index)
                value_mean_dist.append(temp_sub_relative['distance'].mean())
                value_mean_angle.append(temp_sub_relative['angle'].mean())
                value_mean_class.append(idc)
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
            for a in range(len(value_mean_angle)):
                plt.polar(value_mean_angle[a], value_mean_dist[a], 'ro')
                plt.text(value_mean_angle[a], value_mean_dist[a], '%s' %(convert_name(value_mean_class[a])),horizontalalignment='center',verticalalignment='top')
            plt.show(block=False)
            '''
        for indc in centers:
            sub_relative = list_combinations_in_frame[list_combinations_in_frame['classes'].apply(lambda x: indc in x)]
            for idc in top_class_list:
                fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
                temp_sub_relative = sub_relative[sub_relative['combinations'].apply(lambda x: (indc, idc) in x)]
                for combi, pval in zip(temp_sub_relative.combinations.items(), temp_sub_relative.polar_dist.items()):
                    polar_values = pval[1]
                    combinations_set = combi[1]
                    for c in range(len(combinations_set)):
                        if combinations_set[c] == (indc, idc):
                            plt.polar(polar_values[c][1], polar_values[c][0], 'ro')
                            # plt.text(polar_values[c][1], polar_values[c][0], '%s' % (convert_name(idc)), horizontalalignment='center', verticalalignment='top')
                plt.title('Relative Postions of ' + convert_name(idc) + ' to ' + convert_name(indc))
                plt.savefig(Path(file_path + convert_name(idc) + '_to_' + convert_name(indc) + '_Relative_position'))
                plt.close()
                # plt.show(block=False)

        # Confusion

        df_matrix = pd.DataFrame(columns=range(80))
        df_matrix.insert(0, 'classes', range(80))
        list_class_in_frame = stat_frame.groupby(['action', 'video', 'frame']).classes.apply(list).reset_index()
        list_class_in_frame['classes'] = list_class_in_frame['classes'].apply(deduplicate)
        df_mat_temp = pd.DataFrame()
        for cl in stat_frame['classes'].unique():
            df_mat_temp = list_class_in_frame[list_class_in_frame['classes'].apply(lambda x: cl in x)]
            size_temp = len(df_mat_temp)
            v_temp = np.zeros(80)
            conf = df_mat_temp['classes'].apply(pd.Series.value_counts).sum().sort_index()
            for ind in conf.index.values.tolist():
                v_temp[int(ind)] = v_temp[int(ind)] + conf.at[ind]
            df_matrix.at[cl, range(80)] = (v_temp / size_temp) * 100
        df_matrix = df_matrix.fillna(0)
        for cls in df_matrix['classes'].unique():
            df_matrix = df_matrix.rename(index={cls: convert_name(cls)})
            df_matrix = df_matrix.rename(columns={cls: convert_name(cls)})
        df_matrix = df_matrix.drop(columns=['classes'])
        plt.figure(figsize=(20, 17))
        sn.heatmap(df_matrix, annot=False)
        plt.savefig(Path(file_path + 'Confusion_Matrix'))
        plt.close()
        # plt.show(block=False)

        # -> video
        '''
        for vid in list_class_in_frame['video'].unique():
            df_matrix = pd.DataFrame(columns=range(80))
            df_matrix.insert(0, 'classes', range(80))
            df_mat_temp = pd.DataFrame()
            vid_mat = list_class_in_frame[list_class_in_frame.video.isin([vid])]
            for cl in stat_frame['classes'].unique():
                df_mat_temp = vid_mat[vid_mat['classes'].apply(lambda x: cl in x)]
                size_temp = len(df_mat_temp)
                v_temp = np.zeros(80)
                if not df_mat_temp.empty :
                    conf = df_mat_temp['classes'].apply(pd.Series.value_counts).sum().sort_index()
                    for ind in conf.index.values.tolist():
                        v_temp[int(ind)] = v_temp[int(ind)] + conf.at[ind]
                    df_matrix.at[cl, range(80)] = (v_temp / size_temp) * 100
            df_matrix = df_matrix.fillna(0)
            for cls in df_matrix['classes'].unique():
                df_matrix = df_matrix.rename(index={cls: convert_name(cls)})
                df_matrix = df_matrix.rename(columns={cls: convert_name(cls)})
            df_matrix = df_matrix.drop(columns=['classes'])
            plt.figure(figsize=(20, 17))
            sn.heatmap(df_matrix, annot=False)
            p = Path("C:\\Users\\gap201\\Documents\\UCF_Yolo\\detect\\ApplyEyeMakeup\\"+vid+"_confusion.png")
            plt.savefig(p)
            plt.close()
        '''

        # More stats
        '''
        list_dimension_max = stat_frame.groupby(['action', 'classes']).max()
        list_dimension_min = stat_frame.groupby(['action', 'classes']).min()
        list_dimension_mean_per_video = stat_frame.groupby(['action', 'video', 'classes']).mean()
        list_dimension_max_per_video = stat_frame.groupby(['action', 'video', 'classes']).max()
        list_dimension_min_per_video = stat_frame.groupby(['action', 'video', 'classes']).min()
        '''
