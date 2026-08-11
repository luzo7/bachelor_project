import time
import json
import numpy as np
import cocoex
import re
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig


_BBOB_BEST_PARAMETER_FILE = '._bbob_problem_best_parameter.txt'
_BBIOBJ_BEST_PARAMETER_FILE = '._bbob_biobj_problem_best_parameter.txt'

def _compute_best_value(problem, best_parameters):
    """Return the optimum for each objective given a problem and the parameters of the optimum"""
    suite_name = "bbob-mixint"

    if int(problem.number_of_objectives) > 1:
        pattern = r"_f(\d+)_i(\d+).*?_f(\d+)_i(\d+)"
        match = re.search(pattern, problem.name) # Example problem.name: bbob_f015_i02_d40__bbob_f015_i04_d40
        
        if match:
            f1_idx = int(match.group(1)) # function 1
            i1_idx = int(match.group(2)) # instance of function 1
            
            f2_idx = int(match.group(3)) # function 2
            i2_idx = int(match.group(4)) # instance of function 2
        else:
            return None
        
        # Getting first problem
        suite_filter = f"function_indices: {f1_idx} dimensions: 40 instance_indices: {i1_idx}"
        suite = cocoex.Suite(suite_name, "", suite_filter)
        problem1 = suite[0]

        # Getting second problem
        suite_filter = f"function_indices: {f2_idx} dimensions: 40 instance_indices: {i2_idx}"
        suite = cocoex.Suite(suite_name, "", suite_filter)
        problem2 = suite[0]

        return [problem1(best_parameters[0]), problem2(best_parameters[1])]
    else:
        # Single-objective, directily calculates best value
        return problem(best_parameters)
        

def _read_problem_optimum(problem):
    """Return the optimum known by COCO for the current problem"""
    objective_count = int(problem.number_of_objectives)
    variable_count = int(problem.dimension)

    parameter_file = _BBIOBJ_BEST_PARAMETER_FILE if objective_count > 1 else _BBOB_BEST_PARAMETER_FILE

    # Function exposed by interface.pyx, calls functions in coco_problem.c that print the best params in a txt file
    problem._best_parameter('print')

    # Reading the best params from txt
    try:
        best_parameters = np.loadtxt(parameter_file)
    except Exception:
        best_parameters = np.array([])

    # Shape check
    if objective_count > 1:
        if best_parameters.ndim == 2 and best_parameters.shape == (2, variable_count): # 2xD
            pass
        elif best_parameters.ndim == 2 and best_parameters.shape == (variable_count, 2): # Dx2
            best_parameters = best_parameters.T
        elif best_parameters.ndim == 1 and best_parameters.size == 2 * variable_count: # 2D flat vectors
            best_parameters = best_parameters.reshape(2, variable_count)
        else:
            print(f'Unexpected best-parameter shape returned by COCO for MOO: {best_parameters.shape}', flush=True)
            return None        
    else:
        if best_parameters.ndim != 1 or best_parameters.size != variable_count:
            print(f'Unexpected best-parameter length returned by COCO: got {best_parameters.size}, expected {variable_count}', flush=True)
            return None

    best_value = _compute_best_value(problem, best_parameters)

    return {
        'best_parameter': best_parameters.tolist(),
        'best_value': best_value,
    }

class DimensionsExperiment():
    def __init__(self, 
                 name, 
                 objective_f, 
                 n_iterations, 
                 max_trials, 
                 is_moo, 
                 generation_strategy = None, 
                 seed = None):
        
        self.name = name
        self.problem = objective_f
        self.dim = objective_f.dimension
        self.n_iterations = n_iterations
        self.max_trials = max_trials
        self.generation_strategy = generation_strategy
        self.seed = seed
        self.is_moo = is_moo
        
        # Setup directories
        output_dir = 'coco_results/'
        self.json_filename = f'{output_dir}data/{self.name}.json'
        
        self.suite_name = 'bbob-biobj-mixint' if self.is_moo else 'bbob-mixint'

    def _params_to_numpy(self, param_dict):
        """Converts from Ax params to numpy array needed by COCO's problems"""
        x_list = [param_dict[f"x{k}"] for k in range(self.dim)]
        return np.array(x_list, dtype=np.float64)

    def _setup_ax_parameters(self):
        """Creates the Ax parameters array"""
        parameters = []
        
        n_integers = getattr(self.problem, "number_of_integer_variables", 0) # Needed if we use non mixint suites
        
        for k, (lb, ub) in enumerate(zip(self.problem.lower_bounds, self.problem.upper_bounds)):
            if k < n_integers:
                param_type = "int"
                bounds = (int(np.ceil(lb)), int(np.floor(ub)))
            else:
                param_type = "float"
                bounds = (float(lb), float(ub))
                
            parameters.append(
                RangeParameterConfig(
                    name=f"x{k}",
                    bounds=bounds,
                    parameter_type=param_type,
                )
            )
            
        return parameters

    def run_experiment(self):
        self.start_time = time.perf_counter()

        result = {
            'name': self.name,
            'start_time': self.start_time,
            'dim': self.dim,
            'is_moo': self.is_moo,
            'best_recommended': None,
            'end_time': None,
            'scores': {}
        }

        best_score = float("inf")
        scores_history = {}

        parameters = self._setup_ax_parameters()
        
        client = Client(random_seed=self.seed)
        client.configure_experiment(parameters=parameters)

        # Minimize the obcjectives
        if self.is_moo:
            client.configure_optimization(objective="-obj_0, -obj_1")
        else:
            client.configure_optimization(objective="-obj_0")
            
        if self.generation_strategy:
            client.set_generation_strategy(generation_strategy=self.generation_strategy)

        result['problem_optimum'] = _read_problem_optimum(self.problem)

        result_folder = re.sub('_i\\d+', '', self.name)
        observer = cocoex.Observer(self.suite_name, f"result_folder: {result_folder} algorithm_name: {self.name.split('_')[0]}")
        self.problem.observe_with(observer)
        
        for i in range(self.n_iterations):
            try:
                trials = client.get_next_trials(max_trials=self.max_trials)

                for idx, param in trials.items():
                    X_np = self._params_to_numpy(param)
                    f_eval = self.problem(X_np)
                    
                    if self.is_moo:
                        # f_eval is an array [val1, val2]
                        raw_data = {"obj_0": f_eval[0], "obj_1": f_eval[1]}
                        scores_history[idx] = list(f_eval)
                    else:
                        # f_eval is a float
                        scalar_val = float(f_eval[0]) if isinstance(f_eval, (np.ndarray, list)) else float(f_eval)
                        raw_data = {"obj_0": scalar_val}
                        scores_history[idx] = scalar_val
                        
                        # Tracking best_score only in Single-Objective
                        if scalar_val < best_score:
                            best_score = scalar_val
                            result['best_observed'] = {'index': idx, 'score': best_score, 'params': param}


                    client.complete_trial(trial_index=idx, raw_data=raw_data)


                result['scores'] = scores_history

                if not self.is_moo:
                    result['mean_score'] = np.mean(list(scores_history.values()))
                
                with open(self.json_filename, "w") as f:
                    json.dump(result, f, indent=4)
                    
            except Exception as e:
                print(f"[{self.name}] ERROR at iteration {i}: {str(e)}. Data saved up to last iteration.")
                break 

        self.problem.free() # Needed for C COCO's backend
        
        if not self.is_moo:
            try:
                best_params, posterior, best_idx, _ = client.get_best_parameterization()
                result['best_recommended'] = {
                    'index': best_idx,
                    'params': best_params,
                    'posterior': posterior
                }
            except Exception as e:
                print(f"[{self.name}] Error extracting best_parameterization: {e}")

        self.end_time = time.perf_counter()
        result['end_time'] = self.end_time

        with open(self.json_filename, "w") as f:
            json.dump(result, f, indent=4)
            
        print(f"[{self.name}] Experiment completed in {self.end_time - self.start_time:.2f}s")