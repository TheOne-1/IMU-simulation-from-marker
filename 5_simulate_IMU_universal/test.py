from OffsetClass import *

axis_1 = OneAxisOffset('trunk', 'x', range(2))
axis_2 = OneAxisOffset('trunk', 'z', range(2, 4))
axis_3 = OneAxisOffset('trunk', 'z', range(5, 8))


temp = MultiAxisOffset()
temp.add_offset_axis(axis_1)
temp.add_offset_axis(axis_2)
temp.add_offset_axis(axis_3)

comb = temp.get_combos()
for item in comb:
    for offset in item:
        offset.to_string()
    print('')


