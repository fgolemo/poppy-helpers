from poppy_helpers.constants import JOINT_LIMITS, VEL_MAX, VEL_MIN


class Normalizer(object):
    def __init__(self):
        self.pos_diffs = [JOINT_LIMITS[i][1] - JOINT_LIMITS[i][0] for i in range(6)]
        self.vel_diff = VEL_MAX - VEL_MIN

    def normalize_pos(self, pos):
        out = []
        for i in range(6):
            shifted = (pos[i] - JOINT_LIMITS[i][0]) / self.pos_diffs[i]  # now it's in [0,1]
            norm = shifted * 2 - 1
            out.append(norm)
        return out

    def denormalize_pos(self, pos):
        out = []
        for i in range(6):
            shifted = (pos[i] + 1) / 2  # now it's within [0,1]
            denorm = shifted * self.pos_diffs[i] + JOINT_LIMITS[i][0]
            out.append(denorm)
        return out

    def normalize_vel(self, vel):
        out = []
        for i in range(6):
            shifted = (vel[i] - VEL_MIN) / self.vel_diff  # now it's in [0,1]
            norm = shifted * 2 - 1
            out.append(norm)
        return out

    def denormalize_vel(self, vel):
        out = []
        for i in range(6):
            shifted = (vel[i] + 1) / 2  # now it's within [0,1]
            denorm = shifted * self.vel_diff + VEL_MIN
            out.append(denorm)
        return out