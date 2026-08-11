import sys
import torch
import cocoex

from ax.adapter.registry import Generators
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.transition_criterion import MinTrials
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from ax.generation_strategy.generation_strategy import GenerationStrategy
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement

from coco_experiments import DimensionsExperiment


MOO_F_ID_THRESHOLD = 24 # Identify bi-obj functions using their id
N_ITERATIONS = 400
MAX_TRIALS = 5 # Parallel trials generated


def _sobol_node(seed, transition_to = None):
    """Creates a random generation node with Sobol sequence"""
    criteria = (
        [MinTrials(threshold=20, transition_to=transition_to, use_all_trials_in_exp=True)] if transition_to else []
    )

    return GenerationNode(
        name="Sobol",
        generator_specs=[
            GeneratorSpec(generator_enum=Generators.SOBOL, generator_kwargs={"seed": seed})
        ],
        transition_criteria=criteria,
    )


def _bo_strategy(node_name, generator_enum, seed, device, is_moo):
    """Builds the generation strategy based on the given BO approach and nubmer of objectives"""
    acqf = qLogNoisyExpectedHypervolumeImprovement if is_moo else qLogNoisyExpectedImprovement

    bo_node = GenerationNode(
        name=node_name,
        generator_specs=[GeneratorSpec(
            generator_enum=generator_enum,
            generator_kwargs={"torch_device": device, "botorch_acqf_class": acqf},
        )],
    )

    return GenerationStrategy(
        name=f"Sobol+{node_name}",
        nodes=[_sobol_node(seed, transition_to=node_name), bo_node],
    )


def build_generation_strategy(algo, seed, device, is_moo):
    match algo:
        case "random":
            return GenerationStrategy(name="Random", nodes=[_sobol_node(seed)])
        case "standard":
            return _bo_strategy("Standard GP", Generators.BOTORCH_MODULAR, seed, device, is_moo)
        case "saasbo":
            return _bo_strategy("SAASBO", Generators.SAASBO, seed, device, is_moo)
        case "mixed":
            return _bo_strategy("BO_MIXED", Generators.BO_MIXED, seed, device, is_moo)
        case _:
            raise ValueError(f"Unknown algorithm: {algo!r}")


def load_coco_function(f_id, dim, instance, is_moo):
    suite_name = "bbob-biobj-mixint" if is_moo else "bbob-mixint"
    suite_filter = f"function_indices: {f_id} dimensions: {dim} instance_indices: {instance}"
    suite = cocoex.Suite(suite_name, "", suite_filter)
    return suite[0]


def main():
    _, algo, f_id_str, dim_str, run_id_str, device_str = sys.argv
    dim = int(dim_str)
    instance = int(run_id_str)
    device = torch.device(device_str)
    f_id = int(f_id_str)
    seed = instance # Different seeds for each run

    is_moo = f_id > MOO_F_ID_THRESHOLD
    objective_f = load_coco_function(f_id, dim, instance, is_moo)

    generation_strategy = build_generation_strategy(algo, dim, seed, device, is_moo)

    exp = DimensionsExperiment(
        name=f"{algo}_f{f_id}_i{instance}_{dim}D",
        objective_f=objective_f,
        n_iterations=N_ITERATIONS,
        max_trials=MAX_TRIALS,
        is_moo=is_moo,
        generation_strategy=generation_strategy,
        seed=seed
    )
    exp.run_experiment()


if __name__ == "__main__":
    main()