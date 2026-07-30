"""TOPAS script generator for the MOBO development example (testing_mode ZDT1).

This does not produce a runnable TOPAS simulation. It records the current
decision variables as comments so ``TopasObjectiveFunction`` can read them back
in testing mode. A real study swaps the body for genuine TOPAS input.
"""


PARAM_PREFIX = "# PARAM "


def GenerateTopasScripts(BaseDirectory, iteration, **variable_dict):
    """Generate the TOPAS script(s) for one optimization iteration.

    :param BaseDirectory: Base directory for simulation outputs.
    :param iteration: Current iteration number.
    :param **variable_dict: Current parameter values keyed by parameter name.

    :returns: Tuple ``(scripts, names)`` -- a list of scripts (each a list of
        lines) and the matching list of base filenames.
    """
    script = ["# MOBO Development Example (ZDT1 in testing_mode)"]
    script.append(f"# iteration {iteration}")
    # Iterate in insertion order, which is the optimizer's ParameterNames order
    # (VariableDict is built from it and ``**kwargs`` preserves that order).
    # Sorting here would reorder the vector lexicographically, so "x10" would
    # land between "x1" and "x2" and the objective would score a permuted design.
    for key, value in variable_dict.items():
        script.append(f"{PARAM_PREFIX}{key} = {value}")
    return [script], ["MOBODevelopmentExample"]
