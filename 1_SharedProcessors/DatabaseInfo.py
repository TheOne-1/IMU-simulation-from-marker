import numpy as np


class DatabaseInfo:
    @staticmethod
    def get_trial_num(sub=0, speed=0):
        array = np.array([[16, 19, 25, 32, 40, 46, 55, 61, 67, 76, 49, 73],
                          [17, 20, 26, 31, 41, 47, 56, 62, 68, 77, 50, 74],
                          [18, 21, 27, 33, 42, 48, 57, 63, 69, 78, 51, 75]])
        return array[speed, sub]

    @staticmethod
    def get_all_column_names():
        return ['TimeStamp', 'FrameNumber', 'LHEAD.PosX', 'LHEAD.PosY', 'LHEAD.PosZ',
                'THEAD.PosX', 'THEAD.PosY', 'THEAD.PosZ', 'RHEAD.PosX', 'RHEAD.PosY', 'RHEAD.PosZ',
                'FHEAD.PosX', 'FHEAD.PosY', 'FHEAD.PosZ', ' C7.PosX', ' C7.PosY', ' C7.PosZ',
                'T10.PosX', 'T10.PosY', 'T10.PosZ', 'SACR.PosX', 'SACR.PosY', 'SACR.PosZ',
                'NAVE.PosX', 'NAVE.PosY', 'NAVE.PosZ', 'XYPH.PosX', 'XYPH.PosY', 'XYPH.PosZ',
                'STRN.PosX', 'STRN.PosY', 'STRN.PosZ', 'BBAC.PosX', 'BBAC.PosY', 'BBAC.PosZ',
                'LSHO.PosX', 'LSHO.PosY', 'LSHO.PosZ', 'LDELT.PosX', 'LDELT.PosY', 'LDELT.PosZ',
                'LLEE.PosX', 'LLEE.PosY', 'LLEE.PosZ', 'LMEE.PosX', 'LMEE.PosY', 'LMEE.PosZ',
                'LFRM.PosX', 'LFRM.PosY', 'LFRM.PosZ', 'LMW.PosX', 'LMW.PosY', 'LMW.PosZ',
                'LLW.PosX', 'LLW.PosY', 'LLW.PosZ', 'LFIN.PosX', 'LFIN.PosY', 'LFIN.PosZ',
                'RSHO.PosX', 'RSHO.PosY', 'RSHO.PosZ', 'RDELT.PosX', 'RDELT.PosY', 'RDELT.PosZ',
                'RLEE.PosX', 'RLEE.PosY', 'RLEE.PosZ', 'RMEE.PosX', 'RMEE.PosY', 'RMEE.PosZ',
                'RFRM.PosX', 'RFRM.PosY', 'RFRM.PosZ', 'RMW.PosX', 'RMW.PosY', 'RMW.PosZ',
                'RLW.PosX', 'RLW.PosY', 'RLW.PosZ', 'RFIN.PosX', 'RFIN.PosY', 'RFIN.PosZ',
                'LASIS.PosX', 'LASIS.PosY', 'LASIS.PosZ', 'RASIS.PosX', 'RASIS.PosY', 'RASIS.PosZ',
                'LPSIS.PosX', 'LPSIS.PosY', 'LPSIS.PosZ', 'RPSIS.PosX', 'RPSIS.PosY', 'RPSIS.PosZ',
                'LGTRO.PosX', 'LGTRO.PosY', 'LGTRO.PosZ', 'FLTHI.PosX', 'FLTHI.PosY', 'FLTHI.PosZ',
                'LLEK.PosX', 'LLEK.PosY', 'LLEK.PosZ', 'LATI.PosX', 'LATI.PosY', 'LATI.PosZ',
                'LLM.PosX', 'LLM.PosY', 'LLM.PosZ', 'LHEE.PosX', 'LHEE.PosY', 'LHEE.PosZ',
                'LTOE.PosX', 'LTOE.PosY', 'LTOE.PosZ', 'LMT5.PosX', 'LMT5.PosY', 'LMT5.PosZ',
                'RGTRO.PosX', 'RGTRO.PosY', 'RGTRO.PosZ', 'FRTHI.PosX', 'FRTHI.PosY', 'FRTHI.PosZ',
                'RLEK.PosX', 'RLEK.PosY', 'RLEK.PosZ', 'RATI.PosX', 'RATI.PosY', 'RATI.PosZ',
                'RLM.PosX', 'RLM.PosY', 'RLM.PosZ', 'RHEE.PosX', 'RHEE.PosY', 'RHEE.PosZ',
                'RTOE.PosX', 'RTOE.PosY', 'RTOE.PosZ', 'RMT5.PosX', 'RMT5.PosY', 'RMT5.PosZ',
                'FP1.CopX', 'FP1.CopY', 'FP1.CopZ', 'FP1.ForX', 'FP1.ForY', 'FP1.ForZ',
                'FP1.MomX', 'FP1.MomY', 'FP1.MomZ', 'FP2.CopX', 'FP2.CopY', 'FP2.CopZ',
                'FP2.ForX', 'FP2.ForY', 'FP2.ForZ', 'FP2.MomX', 'FP2.MomY', 'FP2.MomZ',
                'Cortex Time', 'D-Flow Time', 'ReferenceFY_FP1', 'Time']

    @staticmethod
    def get_necessary_columns():
        return [
            'TimeStamp', 'FrameNumber', 'Time', 'LHEAD.PosX', 'LHEAD.PosY', 'LHEAD.PosZ',
            'THEAD.PosX', 'THEAD.PosY', 'THEAD.PosZ', 'RHEAD.PosX', 'RHEAD.PosY', 'RHEAD.PosZ',
            'FHEAD.PosX', 'FHEAD.PosY', 'FHEAD.PosZ', ' C7.PosX', ' C7.PosY', ' C7.PosZ',
            'T10.PosX', 'T10.PosY', 'T10.PosZ', 'SACR.PosX', 'SACR.PosY', 'SACR.PosZ',
            'NAVE.PosX', 'NAVE.PosY', 'NAVE.PosZ', 'XYPH.PosX', 'XYPH.PosY', 'XYPH.PosZ',
            'STRN.PosX', 'STRN.PosY', 'STRN.PosZ', 'BBAC.PosX', 'BBAC.PosY', 'BBAC.PosZ',
            'LSHO.PosX', 'LSHO.PosY', 'LSHO.PosZ', 'LDELT.PosX', 'LDELT.PosY', 'LDELT.PosZ',
            'LLEE.PosX', 'LLEE.PosY', 'LLEE.PosZ', 'LMEE.PosX', 'LMEE.PosY', 'LMEE.PosZ',
            'LFRM.PosX', 'LFRM.PosY', 'LFRM.PosZ', 'LMW.PosX', 'LMW.PosY', 'LMW.PosZ',
            'LLW.PosX', 'LLW.PosY', 'LLW.PosZ', 'LFIN.PosX', 'LFIN.PosY', 'LFIN.PosZ',
            'RSHO.PosX', 'RSHO.PosY', 'RSHO.PosZ', 'RDELT.PosX', 'RDELT.PosY', 'RDELT.PosZ',
            'RLEE.PosX', 'RLEE.PosY', 'RLEE.PosZ', 'RMEE.PosX', 'RMEE.PosY', 'RMEE.PosZ',
            'RFRM.PosX', 'RFRM.PosY', 'RFRM.PosZ', 'RMW.PosX', 'RMW.PosY', 'RMW.PosZ',
            'RLW.PosX', 'RLW.PosY', 'RLW.PosZ', 'RFIN.PosX', 'RFIN.PosY', 'RFIN.PosZ',
            'LASIS.PosX', 'LASIS.PosY', 'LASIS.PosZ', 'RASIS.PosX', 'RASIS.PosY', 'RASIS.PosZ',
            'LPSIS.PosX', 'LPSIS.PosY', 'LPSIS.PosZ', 'RPSIS.PosX', 'RPSIS.PosY', 'RPSIS.PosZ',
            'LGTRO.PosX', 'LGTRO.PosY', 'LGTRO.PosZ', 'FLTHI.PosX', 'FLTHI.PosY', 'FLTHI.PosZ',
            'LLEK.PosX', 'LLEK.PosY', 'LLEK.PosZ', 'LATI.PosX', 'LATI.PosY', 'LATI.PosZ',
            'LLM.PosX', 'LLM.PosY', 'LLM.PosZ', 'LHEE.PosX', 'LHEE.PosY', 'LHEE.PosZ',
            'LTOE.PosX', 'LTOE.PosY', 'LTOE.PosZ', 'LMT5.PosX', 'LMT5.PosY', 'LMT5.PosZ',
            'RGTRO.PosX', 'RGTRO.PosY', 'RGTRO.PosZ', 'FRTHI.PosX', 'FRTHI.PosY', 'FRTHI.PosZ',
            'RLEK.PosX', 'RLEK.PosY', 'RLEK.PosZ', 'RATI.PosX', 'RATI.PosY', 'RATI.PosZ',
            'RLM.PosX', 'RLM.PosY', 'RLM.PosZ', 'RHEE.PosX', 'RHEE.PosY', 'RHEE.PosZ',
            'RTOE.PosX', 'RTOE.PosY', 'RTOE.PosZ', 'RMT5.PosX', 'RMT5.PosY', 'RMT5.PosZ',
            'FP1.CopX', 'FP1.CopY', 'FP1.CopZ', 'FP1.ForX', 'FP1.ForY', 'FP1.ForZ',
            'FP2.CopX', 'FP2.CopY', 'FP2.CopZ',
            'FP2.ForX', 'FP2.ForY', 'FP2.ForZ']

    @staticmethod
    def get_marker_column_num():
        return range(3, 144)

    @staticmethod
    def get_force_column_num():
        return range(144, 156)

    def get_file_names(self, sub=0, speed=0, path=''):
        trial_num_str = str(self.get_trial_num(sub, speed))
        mocap_name = path + 'T0' + trial_num_str + '\\mocap-0' + trial_num_str + '.txt'
        record_name = path + 'T0' + trial_num_str + '\\record-0' + trial_num_str + '.txt'
        meta_name = path + 'T0' + trial_num_str + '\\meta-0' + trial_num_str + '.yml'
        return [mocap_name, record_name, meta_name]

    @staticmethod
    def get_segment_marker_names(segment):
        segment_marker = {
            'trunk': [
                ' C7.PosX', ' C7.PosY', ' C7.PosZ',
                'T10.PosX', 'T10.PosY', 'T10.PosZ',
                'XYPH.PosX', 'XYPH.PosY', 'XYPH.PosZ',  # 胸骨箭状突起 Xiphoid process of the sternum
                'STRN.PosX', 'STRN.PosY', 'STRN.PosZ',  # On the jugular notch of the sternum
                # the above markers are more important
                # 'BBAC.PosX', 'BBAC.PosY', 'BBAC.PosZ',        # Inferior angle of the right scapula
                # 'NAVE.PosX', 'NAVE.PosY', 'NAVE.PosZ',        # navel, 肚脐
                # 'SACR.PosX', 'SACR.PosY', 'SACR.PosZ',        # 骶骨, 连接pelvis和trunk
                # 'LSHO.PosX', 'LSHO.PosY', 'LSHO.PosZ',        # left shoulder
                # 'RSHO.PosX', 'RSHO.PosY', 'RSHO.PosZ'         # right shoulder
            ],
            'pelvis': [
                'LASIS.PosX', 'LASIS.PosY', 'LASIS.PosZ',
                'RASIS.PosX', 'RASIS.PosY', 'RASIS.PosZ',
                'LPSIS.PosX', 'LPSIS.PosY', 'LPSIS.PosZ',
                'RPSIS.PosX', 'RPSIS.PosY', 'RPSIS.PosZ',
                # 'LGTRO.PosX', 'LGTRO.PosY', 'LGTRO.PosZ',  # center of the left greater trochanter
                # 'RGTRO.PosX', 'RGTRO.PosY', 'RGTRO.PosZ'
            ],
            'l_thigh': [
                'LGTRO.PosX', 'LGTRO.PosY', 'LGTRO.PosZ',  # center of the left greater trochanter
                'FLTHI.PosX', 'FLTHI.PosY', 'FLTHI.PosZ',  # 1/3 on the line between the LFTRO and LLEK
                'LLEK.PosX', 'LLEK.PosY', 'LLEK.PosZ',  # left knee
            ],
            'r_thigh': [
                'RGTRO.PosX', 'RGTRO.PosY', 'RGTRO.PosZ',  # center of the right greater trochanter
                'FRTHI.PosX', 'FRTHI.PosY', 'FRTHI.PosZ',  # 1/3 on the line between the RFTRO and RLEK
                'RLEK.PosX', 'RLEK.PosY', 'RLEK.PosZ',  # right knee
            ],
            'l_shank': [
                'LLEK.PosX', 'LLEK.PosY', 'LLEK.PosZ',  # left knee
                'LATI.PosX', 'LATI.PosY', 'LATI.PosZ',  # left anterior of the tibia
                'LLM.PosX', 'LLM.PosY', 'LLM.PosZ',  # left lateral malleoulus of the ankle
            ],
            'r_shank': [
                'RLEK.PosX', 'RLEK.PosY', 'RLEK.PosZ',  # right knee
                'RATI.PosX', 'RATI.PosY', 'RATI.PosZ',  # right anterior of the tibia
                'RLM.PosX', 'RLM.PosY', 'RLM.PosZ',  # right lateral malleoulus of the ankle
            ],
            'l_feet': [
                'LLM.PosX', 'LLM.PosY', 'LLM.PosZ',  # left lateral malleoulus of the ankle
                'LHEE.PosX', 'LHEE.PosY', 'LHEE.PosZ',  # left heel
                'LTOE.PosX', 'LTOE.PosY', 'LTOE.PosZ',  # left toe
                'LMT5.PosX', 'LMT5.PosY', 'LMT5.PosZ',  # left 5th metatarsal
            ],
            'r_feet': [
                'RLM.PosX', 'RLM.PosY', 'RLM.PosZ',  # right lateral malleoulus of the ankle
                'RHEE.PosX', 'RHEE.PosY', 'RHEE.PosZ',  # right heel
                'RTOE.PosX', 'RTOE.PosY', 'RTOE.PosZ',  # right toe
                'RMT5.PosX', 'RMT5.PosY', 'RMT5.PosZ',  # right 5th metatarsal
            ],
        }
        return segment_marker[segment]

    @staticmethod
    def get_center_marker_names(segment):
        segment_marker = {
            'trunk': [' C7.PosX', ' C7.PosY', ' C7.PosZ'],

            'pelvis': ['LPSIS.PosX', 'LPSIS.PosY', 'LPSIS.PosZ',
                       'RPSIS.PosX', 'RPSIS.PosY', 'RPSIS.PosZ'],

            'l_thigh': ['LGTRO.PosX', 'LGTRO.PosY', 'LGTRO.PosZ',  # center of the left greater trochanter
                        'LLEK.PosX', 'LLEK.PosY', 'LLEK.PosZ'],  # left knee

            'r_thigh': ['RGTRO.PosX', 'RGTRO.PosY', 'RGTRO.PosZ',  # center of the right greater trochanter
                        'RLEK.PosX', 'RLEK.PosY', 'RLEK.PosZ'],  # right knee

            'l_shank': ['LLEK.PosX', 'LLEK.PosY', 'LLEK.PosZ',
                        'LLM.PosX', 'LLM.PosY', 'LLM.PosZ'],  # left knee

            'r_shank': ['RLEK.PosX', 'RLEK.PosY', 'RLEK.PosZ',
                        'RLM.PosX', 'RLM.PosY', 'RLM.PosZ'],  # right knee


            'l_feet': ['LTOE.PosX', 'LTOE.PosY', 'LTOE.PosZ',  # left toe
                       'LMT5.PosX', 'LMT5.PosY', 'LMT5.PosZ'],  # left 5th metatarsal

            'r_feet': ['RTOE.PosX', 'RTOE.PosY', 'RTOE.PosZ',  # right toe
                       'RMT5.PosX', 'RMT5.PosY', 'RMT5.PosZ']  # right 5th metatarsal
        }
        return segment_marker[segment]

    @staticmethod
    def get_force_column_names():
        return ['FP1.ForX', 'FP1.ForY', 'FP1.ForZ', 'FP2.ForX', 'FP2.ForY', 'FP2.ForZ']

    @staticmethod
    def get_cop_column_names():
        return [
            # 'FP1.CopX', 'FP1.CopY',
            'FP2.CopX', 'FP2.CopY']
