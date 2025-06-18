from dataclasses import dataclass
import json

@dataclass
class Coordinate:
    x: float
    y: float

    def to_tuple(self):
        return (self.x, self.y)

    def shift(self, x, y):
        return Coordinate(x=self.x+x, y=self.y+y)
    
    def scale(self, x, y):
        return Coordinate(x=self.x*x, y=self.y*y)

@dataclass
class Bbox:
    topleft: Coordinate
    topright: Coordinate
    bottomright: Coordinate
    bottomleft: Coordinate

    def to_list(self):
        return [
            self.topleft.to_tuple(),
            self.topright.to_tuple(),
            self.bottomright.to_tuple(),
            self.bottomleft.to_tuple(),
        ]

    def to_outbound_rectangle(self):
        return Bbox.from_wh(
            self.max_x - self.min_x,
            self.max_y - self.min_y
        )

    @classmethod
    def from_wh(cls, width, height):
        return cls(
            topleft=Coordinate(0, 0),
            topright=Coordinate(width, 0),
            bottomright=Coordinate(width, height),
            bottomleft=Coordinate(0, height),
        )

    @classmethod
    def from_np(cls, np_bbox):
        assert(np_bbox.shape == (4,2))
        return cls(
            topleft=Coordinate(*np_bbox[0]),
            topright=Coordinate(*np_bbox[1]),
            bottomright=Coordinate(*np_bbox[2]),
            bottomleft=Coordinate(*np_bbox[3]),
        )

    def arrange(self):
        coordinates = sorted(self.to_list(), key=lambda x: x[0])
        topleft, bottomleft = sorted(coordinates[:2], key=lambda x: x[1])
        topright, bottomright = sorted(coordinates[2:], key=lambda x: x[1])
        return Bbox(
            topleft=Coordinate(*topleft),
            topright=Coordinate(*topright),
            bottomright=Coordinate(*bottomright),
            bottomleft=Coordinate(*bottomleft),
        )
    
    def dumps(self):
        return json.dumps(self.to_list())

    def add_margin(self, margin):
        return Bbox(
            topleft=self.topleft.shift(-margin, -margin),
            topright=self.topright.shift(margin, -margin),
            bottomright=self.bottomright.shift(margin, margin),
            bottomleft=self.bottomleft.shift(-margin, margin)
        )
    
    def shift(self, x, y):
        return Bbox(
            topleft=self.topleft.shift(x, y),
            topright=self.topright.shift(x, y),
            bottomright=self.bottomright.shift(x, y),
            bottomleft=self.bottomleft.shift(x, y)
        )
    
    def scale(self, x, y):
        return Bbox(
            topleft=self.topleft.scale(x, y),
            topright=self.topright.scale(x, y),
            bottomright=self.bottomright.scale(x, y),
            bottomleft=self.bottomleft.scale(x, y)
        )

    @classmethod
    def loads(cls, json_dump):
        json_dump = json.loads(json_dump)
        return cls(
            topleft=Coordinate(*json_dump[0]),
            topright=Coordinate(*json_dump[1]),
            bottomright=Coordinate(*json_dump[2]),
            bottomleft=Coordinate(*json_dump[3]),
        )
    
    @staticmethod
    def save(path, bboxs):
        with open(path, 'w') as fout:
            for bbox in bboxs:
                print(bbox.dumps(), file=fout)

    @classmethod
    def load(cls, path):
        output = []
        with open(path, 'r') as fin:
            for line in fin:
                output.append(cls.loads(line))
        return output

    @property
    def min_x(self):
        xs, _ = list(zip(*self.to_list()))
        return min(xs)

    @property
    def min_y(self):
        _, ys = list(zip(*self.to_list()))
        return min(ys)

    @property
    def max_x(self):
        xs, _ = list(zip(*self.to_list()))
        return max(xs)

    @property
    def max_y(self):
        _, ys = list(zip(*self.to_list()))
        return max(ys)
