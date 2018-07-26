# this function returns combos between different segments
def get_segment_combos(self):
    x = get_segment_combos_recursive(self.__list_all)
    return x


def get_segment_combos_recursive(list_all):
    if len(list_all) == 1:
        return_list = [list_all[0], list_all[1]]
    else:
        return_list = []
        first_segment = list_all[0]
        last_return_list = get_segment_combos_recursive(list_all[1:])
        for last_combo in last_return_list:
            new_combo = first_segment[0]
            new_combo.extend(last_combo)
            return_list.append(new_combo)
            new_combo = first_segment[1]
            new_combo.extend(last_combo)
            return_list.append(new_combo)

    return return_list















