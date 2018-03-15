import time
from pypot.robot import from_remote
from poppy_helpers.controller import SwordFightController

def startup_swordfight(host_att, host_def):

    print ("=== connecting to poppies")
    poppy_def = from_remote('{}.local'.format(host_def), 4242) # def = flogo4
    poppy_att = from_remote('{}.local'.format(host_att), 4242) # att = flogo2
    print ("=== connection established")

    controller_def = SwordFightController(poppy_def, mode="def")
    controller_att = SwordFightController(poppy_att, mode="att")

    controller_att.compliant(False)
    controller_def.compliant(False)

    controller_att.set_max_speed(50)
    controller_def.set_max_speed(150)

    controller_def.rest()
    controller_att.rest()

    print ("=== resetting robots")

    time.sleep(2)
    return controller_att, controller_def