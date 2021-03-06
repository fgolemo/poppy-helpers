JOINT_LIMITS = [
    (-90, 90),
    (-90, 90),
    (-90, 90),
    (-90, 90),
    (-90, 90),
    (-90, 90)
]

# Pusher
JOINT_LIMITS_P = [
    (-90, 90),
    (-90, 90),
    (-90, 90)
]

VEL_MAX = 150
VEL_MIN = -150

BALL_STATES = {
    "in_cup": 1,
    "in_air": 2,
    "on_floor": 3
}

JOINT_LIMITS_MAXMIN = [-150, 150]
JOINT_LIMITS_SPEED = 300

REST_POS = [0, 0, 0, 0, 0, 0]
REST_HALF_POS = [0, -90, 90,0,-90,0]

SWORDFIGHT_RANDOM_NOISE = [
    (-90, 90),
    (-30, 30),
    (-30, 30),
    (-45, 45),
    (-30, 30),
    (-30, 30)
]
SWORDFIGHT_REST_DEF = [0, -10, -8, 7, -7, -5]
SWORDFIGHT_REST_ATT = [0, -5, -3, 0, 0, 0]

MOVES = {
    "def0": [-14.52, -20.09, 65.54, 98.97, -21.85, -53.81],
    "def1": [8.36, -27.13, 65.84, -75.51, -62.32, -25.37],
    "def2": [-39.15, -10.7, 101.91, 96.92, 20.97, -70.53],
    "def3": [-13.64, 2.49, 41.79, 82.55, -85.19, 27.71]
    # "def0": [24.78, -12.17, 63.78, -52.35, 17.16, -83.72],
    # "def1": [14.81, -31.23, 89.88, 87.24, -95.16, -3.96],
    # "def2": [-5.72, -57.92, 117.74, -66.42, -18.62, -75.81],
    # "def3": [35.34, -36.8, 92.82, 77.27, -65.25, -58.8]
}

MOVE_EVERY_N_STEPS = 250

# in ms
MAX_REFRESHRATE = 10

SIM_VELOCITY_SCALING = 97