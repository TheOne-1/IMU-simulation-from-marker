
class People:
    def __init__(self, name):
        self._name = name


class Chinese(People):
    def __init__(self, name):
        super().__init__(name)
        print(self._name)



my_chinese = Chinese('Tian')