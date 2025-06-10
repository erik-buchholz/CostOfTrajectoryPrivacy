class ParameterCarrier:

    def __init__(self, args):
        self.dataset_file_name = args['dataset_file_name']
        self.total_epsilon = args['total_epsilon']
        self.epsilon_partition = args['epsilon_partition']
        self.trajectory_number_to_generate = args['trajectory_number_to_generate']
        if 'c_parameter' in args:
            # c required for computation of K (Level 1 Grid Size)
            self.c_parameter = args['c_parameter']
        else:
            self.c_parameter = False
        if 'pop' in args:
            # Population density for each dataset
            self.pop = args['pop']
        else:
            self.pop = False
        self.K = 0  # Level 1 Grid Size



