import logging

import numpy as np

from config.parameter_carrier import ParameterCarrier

log = logging.getLogger()


class Divide:

    def __init__(self, cc: ParameterCarrier):
        self.cc = cc

    # divide parameter, output is array[x_divide_number, y_divide_number, x_increase, y_increase]
    def level1_divide_parameter(self, total_density, trajectory_number, border2):
        divide_threshold = 60
        if self.cc.c_parameter is not False:
            initial_parameter = self.cc.c_parameter
            # Compute K according to the paper: K = sqrt(trajectory_number / c)
            para = np.round(np.sqrt(trajectory_number / initial_parameter))
            log.info(f"Computing K according to the paper for c = {initial_parameter}")
        else:
            # This is the original code version
            initial_parameter = 600
            para = np.floor(np.sqrt(total_density / initial_parameter))
            log.info(f"Computing K according to the repository for c = {initial_parameter}")
        # This seems initial_parameter to be $c$ in the paper
        # Initial value in Code: c=600
        # In the papers: Brinkhoff: c=5000, Porto: c=1200, GeoLife: c=500
        log.info(f"Para for divide: {para}")
        # This para appears to be $K$ in the paper
        assert para > 1, 'need no dividing'
        if para > divide_threshold:
            para = divide_threshold

        self.cc.K = para

        x_divide_number = para
        y_divide_number = para
        x_divide_number = int(x_divide_number)
        y_divide_number = int(y_divide_number)

        top = border2[0]
        bot = border2[1]
        lef = border2[2]
        rig = border2[3]

        x_increase = 1 / x_divide_number * (rig - lef)
        y_increase = 1 / y_divide_number * (top - bot)
        divide_parameter1 = np.array([x_divide_number, y_divide_number, x_increase, y_increase])
        log.info(f"L1 Divide parameter (= K): {divide_parameter1[0]}")
        return divide_parameter1

    def subdividing_parameter(self, noisy_density):
        # noisy_density := d_i
        if self.cc.pop is not False:
            # This is the paper version: kappa = (d_i * K * pop / (2 * 10^7))^(1/2)
            # However, the number of resulting states is way too large. The equation in the paper must be incorrect.
            factor= self.cc.K * self.cc.pop / (2 * 10^7)
            subdivide_parameter1 = int(np.ceil(np.sqrt(noisy_density * factor)))
        else:
            # Default from the repository
            initial_parameter = 200
            subdivide_parameter1 = int(np.ceil(np.sqrt(noisy_density / initial_parameter)))
        # log.info(f"Subdivide parameter: {subdivide_parameter1}")
        return subdivide_parameter1




