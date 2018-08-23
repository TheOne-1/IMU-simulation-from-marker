from OffsetClass import *


rotation_offsets = OneAxisRotation.get_one_axis_rotation('trunk')
rotation_len = len(rotation_offsets)

for x in range(rotation_len):
    y = rotation_offsets[x]