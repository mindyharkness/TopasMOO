from __future__ import annotations

from TopasMOO.io import ReadInMultiObjectiveLogFile


def test_read_multiobjective_log_parses_iteration_parameters_and_objectives(tmp_path) -> None:
    log_file = tmp_path / "OptimizationLogs.txt"
    log_file.write_text(
        "\n".join(
            [
                "Iteration: 0, x1: 0.10, x2: 0.90, ObjectiveFunction_1: 1.20, ObjectiveFunction_2: 2.40",
                "Iteration: 1, x1: 0.20, x2: 0.80, ObjectiveFunction_1: 1.10, ObjectiveFunction_2: 2.10",
            ]
        )
    )

    parsed = ReadInMultiObjectiveLogFile(str(log_file))

    assert parsed["Iteration"] == [0.0, 1.0]
    assert parsed["x1"] == [0.1, 0.2]
    assert parsed["x2"] == [0.9, 0.8]
    assert parsed["ObjectiveFunction_1"] == [1.2, 1.1]
    assert parsed["ObjectiveFunction_2"] == [2.4, 2.1]
