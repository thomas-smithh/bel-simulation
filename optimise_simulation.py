import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import skewnorm
from scipy.optimize import minimize_scalar
from scipy.stats import skewnorm
from simulation import Simulation
from hyperopt import fmin, tpe, Trials, space_eval, STATUS_OK, hp
from hyperopt.fmin import generate_trials_to_calculate
import random
import os
import logging
import tracemalloc
import psutil

def scaled_gaussian(x, mu, sigma):
    max_val = 1 / (sigma * np.sqrt(2 * np.pi))
    pdf_val = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    y = pdf_val / max_val
    return y

class ScaledSkewNormal:

    def __init__(self, a, mu, sigma):
        neg_pdf = lambda x: -skewnorm.pdf(x, a, mu, sigma)
        self.result = minimize_scalar(neg_pdf, bounds=(mu - 3*sigma, mu + 3*sigma), method='bounded')
        self.pdf_max_value = skewnorm.pdf(self.result.x, a, mu, sigma)
        self.a = a
        self.mu = mu
        self.sigma = sigma

    def get_value(self, x):
        pdf_val = skewnorm.pdf(x, self.a, self.mu, self.sigma)
        return pdf_val / self.pdf_max_value

def lumen_vol_dist(x):
    if 0 <= x <= 0.5:
        return 1
    elif 0.5 < x <= 1:
        return -2 * x + 2
    else:
        return 1e-30

def time_dist(x):
    return (x / 10) + 1e-30

def lumen_com_dist(x):
    if np.isnan(x):
        return 1e-30
    else:
        return 1 / np.exp(0.5*x)

def derive_target(
    mean_separation: float,
    #lumen_com: float,
    sphericity: float,
    n_cells: float,
    lumen_vol: float,
    hull_vol: float,
    max_time: float
) -> float:
    """
    Derive the value to be maximised from the results of the simulation.
    
    Inputs:
        results: The results of the simulation.
    
    Returns:
        The value to be maximised.
    """

    skew_normal_n_cells = ScaledSkewNormal(5, -80, 500)
    skew_normal_sphericity = ScaledSkewNormal(-5, 1.05, 0.3)
    
    # lumen_com_optimisation_values = lumen_com_dist(lumen_com)
    mean_separation_optimisation_value = scaled_gaussian(mean_separation, mu=-0.2, sigma=0.3)
    sphericity_optimisation_value = skew_normal_sphericity.get_value(sphericity)
    n_cells_optimisation_value = skew_normal_n_cells.get_value(n_cells)
    lumen_vol_optimisation_value = lumen_vol_dist(lumen_vol/hull_vol)
    time_optimisation_value = time_dist(max_time)

    target_value = (
        mean_separation_optimisation_value *
        sphericity_optimisation_value *
        n_cells_optimisation_value *
        lumen_vol_optimisation_value *
        time_optimisation_value
        # lumen_com_optimisation_values
    )

    return target_value

def get_next_run_number(path):
    files = os.listdir(path)
    run_numbers = [int(file.split('_Run')[1].split('.')[0]) for file in files if 'parquet' in file]
    if len(run_numbers) == 0:
        return 0
    else:
        return max(run_numbers) + 1

def target_function(
    params: dict
) -> float:
    """
    """

    radius_scaling = random.uniform(0.5, 1.7)
    volume_scaling = random.uniform(0.01, 0.1)
    simulation = Simulation(N_bodies=70)

    run_number = get_next_run_number("F:\\Bel_Simulation\\Optimisation Output with Initial Parameter Space Probing 3")

    simulation.execute(
        alpha=params['alpha'],
        beta=params['beta'],
        A_eq_star_scaling=params['A_eq_star_scaling'],
        P_star=params['P_star'],
        radius_scaling=radius_scaling,
        volume_scaling=volume_scaling,
        #max_reset_count=20,
        run_number=run_number,
        write_results=True,
        write_path="F:\\Bel_Simulation\\Optimisation Output with Initial Parameter Space Probing 3"
    )
    
    results = simulation.results.iloc[-1]

    target = -derive_target(
        results['mean_separation'],
        #results['lumen_distance_from_com'],
        results['sphericity'],
        results['final_N_bodies'],
        results['lumen_volume'],
        results['hull_volume'],
        results['t']
    )


    return {'loss': target, 'status': STATUS_OK}


# def log_memory_usage():
#     process = psutil.Process(os.getpid())
#     memory_info = process.memory_info()
#     logging.info(f"Memory Usage: RSS={memory_info.rss / 1024 ** 2} MB, VMS={memory_info.vms / 1024 ** 2} MB")


#     tracemalloc.start()

#         # Your simulation and optimization code
#         log_memory_usage()
#         # More of your code
#         log_memory_usage()
#         tracemalloc.stop()

# if __name__ == "__main__":
#     main()




if __name__ == '__main__':

    search_space = {
        'alpha': hp.uniform('alpha', 0, 1000),
        'beta': hp.uniform('beta', 0, 1000),
        'A_eq_star_scaling': hp.uniform('A_eq_star_scaling', 0, 0.99),
        'P_star': hp.uniform('P_star', 0, 1000),
    }

    beta = np.random.uniform(0.008351, 0.999717, 5)
    alpha =  np.random.uniform(0.000147, 0.004797, 5)
    A_eq_star_scaling = np.random.uniform(0.078842, 0.387942, 5)
    P_star = np.random.uniform(460.438517, 904.892311, 5) 


    initial_probing = []

    for alpha, beta, A_eq_star_scaling, P_star in zip(alpha, beta, A_eq_star_scaling, P_star):
        initial_probing.append(
            {
                'alpha': alpha,
                'beta': beta,
                'A_eq_star_scaling': A_eq_star_scaling,
                'P_star': P_star,
            }
        )

    # for params in tqdm(initial_probing):
    #     target_function(params)

    trials = generate_trials_to_calculate(initial_probing)

    algo = tpe.suggest
    max_evals = 100_000_000

    best = fmin(
        fn=target_function,
        space=search_space,
        algo=algo,
        max_evals=max_evals,
        trials=trials,
        verbose=True,
        trials_save_file='BEL_trials_14_Optimising_with_initial_probing_lumenvolcost_fccstart_pos.pkl'
    )