import itertools
import fire

def run_experiment(config):

    # your code

def retrieve_config(sweep_step):
    grid = {
        'VAR1': ['a', 'b'],
        'VAR2': [3, 5, 7],
        'VAR3': [True, False]
    }

    grid_setups = list(
        dict(zip(grid.keys(), values)) for values in itertools.product(*grid.values())
    )
    step_grid = grid_setups[sweep_step - 1]  # slurm variable will start from 1

    config = {
        'sweep_step': sweep_step,
        'VAR1': step_grid['VAR1'],
        'VAR2': step_grid['VAR2'],
        'VAR3': step_grid['VAR3']
    }

    return config

def main(sweep_step):
    config = retrieve_config(sweep_step)
    run_experiment(config)

if __name__ == '__main__':
    fire.Fire(main)
