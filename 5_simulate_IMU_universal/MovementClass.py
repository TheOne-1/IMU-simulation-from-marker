import numpy as np

class MoveOneAxis:

    def __init__(self, segment):
        self.__segment = segment



    def __iter__(self):
        return self

    def __next__(self):

    def generate_move(self, axis_name, iterable_object):
        if axis_name not in ['x', 'z', 'theta']:
            raise RuntimeError('wrong movement axis name')
        moves = []
        if axis_name is 'x':
            for item in iterable_object:
                move = Move(x_offset=item)
                moves.append(move)
        if axis_name is 'z':
            for item in iterable_object:
                move = Move(z_offset=item)
                moves.append(move)
        if axis_name is 'theta':
            for item in iterable_object:
                move = Move(x_offset=item)
                moves.append(move)



        self.__moves = moves





class Move:

    def __init__(self, x_offset=0, z_offset=0, y_offset=0, theta_offset=0):
        self.__x_offset = x_offset
        self.__z_offset = z_offset
        self.__y_offset = y_offset
        self.__theta_offset = theta_offset

