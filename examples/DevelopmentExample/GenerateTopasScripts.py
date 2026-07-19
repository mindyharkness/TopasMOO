"""
TOPAS script generator for the Development Example. This is run in testing mode, so does not produce a runnable TOPAS simulation.
It records the current decision variables as comments so ``TopasObjectiveFunction`` can read them back in testing mode.
"""

def GenerateTopasScripts(BaseDirectory, iteration, **variable_dict):
    """Generate the TOPAS script(s) for one optimization iteration.

    This is the minimal generator for the ZDT1 development example. It does not
    produce a runnable TOPAS simulation; instead it records the current decision
    variables as comments so ``TopasObjectiveFunction`` can read them back in
    testing mode. The structure (a function with this exact name and signature,
    returning a list of scripts and a list of names) is what the framework
    requires, so a real study can swap the body for genuine TOPAS input.

    :param BaseDirectory: Base directory for simulation outputs.
    :param iteration: Current iteration number.
    :param **variable_dict: Current parameter values keyed by parameter name
        (ZDT1 uses ``x1`` through ``x5``).

    :returns: A tuple ``(scripts, names)`` where ``scripts`` is a list of scripts
        (each a list of lines) and ``names`` is the matching list of base
        filenames. Return multiple entries if your simulation needs multiple
        files.
    """
    # ZDT1 uses x1 through x5. Store the parameters in comments so
    # TopasObjectiveFunction can read them back in testing mode; a real
    # application would instead emit world geometry, beam source, scoring
    # volumes, and a physics list here.
    script = ["# ZDT1 Benchmark Test script for development"]
    script.append(f"# This is iteration {iteration}")
    # Record the current decision variables
    for i in range(1, 6):
        script.append(f"# x{i} = {variable_dict.get(f'x{i}', 0.5)}")

    # Return as a list of scripts and a list of names.
    return [script], ["DevelopmentExample"]
