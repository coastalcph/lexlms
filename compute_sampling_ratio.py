import json
import numpy as np


def compute_sampling_rates(alpha=0.2):

    with open(f'datasets_specs.json') as dataset_specs_file:
        tokens = json.load(dataset_specs_file)

    probabilities = {}
    # compute probabilities
    for key, value in tokens.items():
        probabilities[key] = value / np.sum([tokens[key] for key in tokens.keys()])

    sampling_rates = {}
    # compute sampling rates
    for key, value in tokens.items():
        sampling_rates[key] = np.power(probabilities[key], alpha) / np.sum([np.power(probabilities[key], alpha) for key in probabilities.keys()])

    return tokens, probabilities, sampling_rates


if __name__ == "__main__":
    print(compute_sampling_rates(0.2)[1])
    print(compute_sampling_rates(0.2)[2])
    print(compute_sampling_rates(0.5)[2])


