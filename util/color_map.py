color_map = {
    'cityscapes': {
        0:  (128,  64, 128),    # road
        1:  (244,  35, 232),    # sidewalk
        2:  ( 70,  70,  70),    # building
        3:  (102, 102, 156),    # wall
        4:  (190, 153, 153),    # fence
        5:  (153, 153, 153),    # pole
        6:  (250, 170,  30),    # traffic light
        7:  (220, 220,   0),    # traffic sign
        8:  (107, 142,  35),    # vegetation
        9:  (152, 251, 152),    # terrain
        10: ( 70, 130, 180),    # sky
        11: (220,  20,  60),    # person
        12: (255,   0,   0),    # rider
        13: (  0,   0, 142),    # car
        14: (  0,   0,  70),    # truck
        15: (  0,  60, 100),    # bus
        16: (  0,  80, 100),    # train
        17: (  0,   0, 230),    # motorcycle
        18: (119,  11,  32),    # bicycle
    },
    'acdc': {
        0:  (128,  64, 128),    # road
        1:  (244,  35, 232),    # sidewalk
        2:  ( 70,  70,  70),    # building
        3:  (102, 102, 156),    # wall
        4:  (190, 153, 153),    # fence
        5:  (153, 153, 153),    # pole
        6:  (250, 170,  30),    # traffic light
        7:  (220, 220,   0),    # traffic sign
        8:  (107, 142,  35),    # vegetation
        9:  (152, 251, 152),    # terrain
        10: ( 70, 130, 180),    # sky
        11: (220,  20,  60),    # person
        12: (255,   0,   0),    # rider
        13: (  0,   0, 142),    # car
        14: (  0,   0,  70),    # truck
        15: (  0,  60, 100),    # bus
        16: (  0,  80, 100),    # train
        17: (  0,   0, 230),    # motorcycle
        18: (119,  11,  32),    # bicycle
    },
    'syn_city': {
        0:  (128,  64, 128),    # road
        1:  (244,  35, 232),    # sidewalk
        2:  ( 70,  70,  70),    # building
        3:  (102, 102, 156),    # wall
        4:  (190, 153, 153),    # fence
        5:  (153, 153, 153),    # pole
        6:  (250, 170,  30),    # traffic light
        7:  (220, 220,   0),    # traffic sign
        8:  (107, 142,  35),    # vegetation
        9:  ( 70, 130, 180),    # sky
        10: (220,  20,  60),    # person
        11: (255,   0,   0),    # rider
        12: (  0,   0, 142),    # car
        13: (  0,  60, 100),    # bus
        14: (  0,   0, 230),    # motorcycle
        15: (119,  11,  32),    # bicycle
    },
    'surround_school': {
        0:  (128,  64, 128),    # road
        1:  (244,  35, 232),    # sidewalk
        2:  ( 70,  70,  70),    # building
        3:  (102, 102, 156),    # guard rail
        4:  (153, 153, 153),    # pole
        5:  (250, 170,  30),    # traffic light
        6:  (220, 220,   0),    # traffic sign
        7:  (107, 142,  35),    # tree
        8:  (152, 251, 152),    # terrain
        9:  ( 70, 130, 180),    # sky
        10: (220,  20,  60),    # person
        11: (255,   0,   0),    # rider
        12: (  0,   0, 142),    # car
        13: (  0,   0,  70),    # truck
        14: (  0,  60, 100),    # bus
        15: (  0,   0, 230),    # motorcycle
        16: (119,  11,  32),    # bicycle
    },
    'HYRoad': {
        0:  (128,  64, 128),    # road
        1:  (244,  35, 232),    # sidewalk
        2:  ( 70,  70,  70),    # building
        3:  (153, 153, 153),    # pole
        4:  (250, 170,  30),    # traffic light
        5:  (220, 220,   0),    # traffic sign
        6:  (107, 142,  35),    # vegetation
        7:  (100,  55,  22),    # trunk
        8:  (152, 251, 152),    # terrain
        9:  ( 70, 130, 180),    # sky
        10: (220,  20,  60),    # person
        11: (  0,   0, 142),    # car
        12: (119,  11,  32),    # bicycle
        13: (168, 161, 192),    # static
        14: (  0,   0,   0),    # background
    },
    'HYRoad_3cls': {
        0:  (  0,   0,   0),    # background
        1:  (153, 153, 153),    # pole
        2:  (100,  55,  22),    # trunk
    },
    'parking_fisheye': {
        0:  (  0,   0,   0),    # background
        1:  (  0,   0, 142),    # parking
        2:  (220,  20,  60),    # white line
        3:  (111,  74,   0),    # yellow line
        4:  (128,  64, 128),    # crossing
        5:  (250, 170, 160),    # arrow
        6:  (250, 170,  30),    # sidewalk
        7:  (220, 200,   0),    # other
    },
    'bev_2023': {
        0:  (  0,   0,   0),    # background
        1:  (  0,   0, 142),    # parking
        2:  (220,  20,  60),    # white line
        3:  (111,  74,   0),    # yellow line
        4:  (128,  64, 128),    # crossing
        5:  (250, 170, 160),    # arrow
        6:  (250, 170,  30),    # slope
    },
    'bev_2024': {
        0:  (  0,   0,   0),    # background
        1:  (  0,   0, 142),    # parking
        2:  (220,  20,  60),    # white line
        3:  (111,  74,   0),    # yellow line
        4:  (128,  64, 128),    # crossing
        5:  (250, 170, 160),    # arrow
        6:  (250, 170,  30),    # slope
    },
    'bev_20234': {
        0:  (  0,   0,   0),    # background
        1:  (  0,   0, 142),    # parking
        2:  (220,  20,  60),    # white line
        3:  (111,  74,   0),    # yellow line
        4:  (128,  64, 128),    # crossing
        5:  (250, 170, 160),    # arrow
        6:  (250, 170,  30),    # slope
    },
    'bev_20234_6cls': {
        0:  (  0,   0,   0),    # background
        1:  (220,  20,  60),    # parking and white line
        2:  (111,  74,   0),    # yellow line
        3:  (128,  64, 128),    # crossing
        4:  (250, 170, 160),    # arrow
        5:  (250, 170,  30),    # slope
    },
    'avm_seg': {
        0:  (  0,   0,   0),    # background
        1:  (  0,   0, 255),    # free space
        2:  (255, 255, 255),    # marker
        3:  (255,   0,   0),    # vehicle
        4:  (  0, 255,   0),    # other objects
    },
    'kyxz': {
        0:  (  0,   0,   0),    # background
        1:  (  0, 255,   0),    # drivable
        2:  (255,   0,   0),    # dangerous
        3:  (  0,   0, 255),    # positive
        4:  (250, 140,  22),    # negative
        5:  (114,  46, 209),    # water
    },
    'gtav': {
        0:  (128,  64, 128),    # road
        1:  (244,  35, 232),    # sidewalk
        2:  ( 70,  70,  70),    # building
        3:  (102, 102, 156),    # wall
        4:  (190, 153, 153),    # fence
        5:  (153, 153, 153),    # pole
        6:  (250, 170,  30),    # traffic light
        7:  (220, 220,   0),    # traffic sign
        8:  (107, 142,  35),    # vegetation
        9:  (152, 251, 152),    # terrain
        10: ( 70, 130, 180),    # sky
        11: (220,  20,  60),    # person
        12: (255,   0,   0),    # rider
        13: (  0,   0, 142),    # car
        14: (  0,   0,  70),    # truck
        15: (  0,  60, 100),    # bus
        16: (  0,  80, 100),    # train
        17: (  0,   0, 230),    # motorcycle
        18: (119,  11,  32),    # bicycle
    },
    'syn': {
        0:  (128,  64, 128),    # road
        1:  (244,  35, 232),    # sidewalk
        2:  ( 70,  70,  70),    # building
        3:  (102, 102, 156),    # wall
        4:  (190, 153, 153),    # fence
        5:  (153, 153, 153),    # pole
        6:  (250, 170,  30),    # traffic light
        7:  (220, 220,   0),    # traffic sign
        8:  (107, 142,  35),    # vegetation
        9:  ( 70, 130, 180),    # sky
        10: (220,  20,  60),    # person
        11: (255,   0,   0),    # rider
        12: (  0,   0, 142),    # car
        13: (  0,  60, 100),    # bus
        14: (  0,   0, 230),    # motorcycle
        15: (119,  11,  32),    # bicycle
    },
}