import random

import numpy as np

from poppy_helpers.constants import SWORDFIGHT_RANDOM_NOISE, REST_POS, MOVES


class Randomizer(object):

    def random_from_rest(self, rest_pos, noise, scaling = 1.0):
        out = []
        for i in range(6):
            new_pos = int(rest_pos[i] + scaling * np.random.randint(
                low=noise[i][0],
                high=noise[i][1],
                size=1)[0])
            out.append(new_pos)

        return out

    def random_sf(self):
        return self.random_from_rest(REST_POS, SWORDFIGHT_RANDOM_NOISE)

    def random_def_stance(self):
        return MOVES["def{}".format(random.randint(0,3))]
