"""
This file contains the objective functions for the optimization.
It is called by the Optimizer object, and receives the results of the Topas simulation.
It returns a list of two objectives:
1. Profile accuracy (lower is better - minimize error)
2. Beam efficiency (lower return value = better efficiency)
"""

import os
from pathlib import Path

import numpy as np
from TopasOpt.utilities import WaterTankData


def CalculateProfileError(TopasResults, GroundTruthResults):
    """
    Calculate the RMS error between desired and actual profile and PDD.
    Uses normalized values to account for different particle counts.

    Returns the mean absolute error as a measure of profile accuracy.
    Lower is better.
    """
    # Define points for profile extraction
    Xpts = np.linspace(GroundTruthResults.x.min(), GroundTruthResults.x.max(), 100)
    Ypts = np.zeros(Xpts.shape)
    Zpts = GroundTruthResults.PhantomSizeZ * np.ones(Xpts.shape)

    OriginalProfile = GroundTruthResults.ExtractDataFromDoseCube(Xpts, Ypts, Zpts)
    OriginalProfileNorm = OriginalProfile * 100 / OriginalProfile.max()
    CurrentProfile = TopasResults.ExtractDataFromDoseCube(Xpts, Ypts, Zpts)
    CurrentProfileNorm = CurrentProfile * 100 / CurrentProfile.max()
    ProfileDifference = OriginalProfileNorm - CurrentProfileNorm

    # Define points for depth dose
    Zpts = GroundTruthResults.z
    Xpts = np.zeros(Zpts.shape)
    Ypts = np.zeros(Zpts.shape)

    OriginalDepthDose = GroundTruthResults.ExtractDataFromDoseCube(Xpts, Ypts, Zpts)
    CurrentDepthDose = TopasResults.ExtractDataFromDoseCube(Xpts, Ypts, Zpts)
    OriginalDepthDoseNorm = OriginalDepthDose * 100 / np.max(OriginalDepthDose)
    CurrentDepthDoseNorm = CurrentDepthDose * 100 / np.max(CurrentDepthDose)
    DepthDoseDifference = OriginalDepthDoseNorm - CurrentDepthDoseNorm

    ProfileError = np.mean(abs(ProfileDifference)) + np.mean(abs(DepthDoseDifference))
    return ProfileError


def CalculateBeamEfficiency(TopasResults):
    """
    Calculate beam efficiency as a measure of how many particles pass through
    the collimator. We use the maximum dose as a proxy for particle transmission.

    We want to maximize efficiency, but since optimizers minimize, we return
    the negative of efficiency (or equivalently, return 1/efficiency).

    Lower return value = better efficiency.
    """
    # Simple efficiency metric: negative of peak dose
    # Higher peak dose = more particles transmitted = better efficiency
    peak_dose = np.max(TopasResults.DoseCube)

    # Return negative so minimization favors higher peak dose
    efficiency_objective = -peak_dose

    return efficiency_objective


def TopasObjectiveFunction(ResultsLocation, iteration):
    """
    Multi-objective function for aperture optimization.

    Returns a list of two objectives:
    1. Profile accuracy (lower is better - minimize error)
    2. Beam efficiency (lower return value = better efficiency)

    :param ResultsLocation: Path to results directory
    :param iteration: Current iteration number
    :return: List of objective values [profile_error, efficiency_objective]
    """

    ResultsFile = ResultsLocation / f"WaterTank_itt_{iteration}.bin"
    path, file = os.path.split(ResultsFile)
    CurrentResults = WaterTankData(path, file)

    # Load ground truth data
    # Update this path to point to your ground truth data
    GroundTruthDataPath = str(
        Path(__file__).parent.parent.parent
        / "TopasOpt-master"
        / "docsrc"
        / "_resources"
        / "ApertureOpt"
        / "Results"
    )
    GroundTruthDataFile = "WaterTank"

    # If ground truth doesn't exist, create a simple target
    try:
        GroundTruthResults = WaterTankData(GroundTruthDataPath, GroundTruthDataFile)
    except:
        # Fallback: use current results as "ground truth" for testing
        print("Warning: Could not load ground truth data. Using simplified objectives.")
        # In this case, just use dose-based objectives
        GroundTruthResults = CurrentResults

    # Calculate objectives
    objective1_profile_error = CalculateProfileError(CurrentResults, GroundTruthResults)
    objective2_efficiency = CalculateBeamEfficiency(CurrentResults)

    # Return as list (TopasMOO requirement)
    return [objective1_profile_error, objective2_efficiency]
