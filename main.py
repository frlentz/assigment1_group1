"""Entry point: load one question's data, build and solve the model, save results and figures.

    python main.py                          # base case of Q1_caseA
    python main.py --question Q2_linear     # another case
    python main.py --scenarios              # also run the example sensitivity scenarios

Results (CSV, TXT, PNG) are written to ``results/<question>/``. Extend ``run_scenarios``
with your own scenarios, or add a new function per question, as your analysis grows.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib

from src.data_loader import InputData, load_question, list_questions
from src.model import FlexibleConsumerModel, Results 
from src.model_q2 import LinearDisutilityModel, NonSeparableDisutilityModel, QuadraticDisutilityModel
from src.model_q3 import MinEnergyModel, BatteryModel
from src.plotting import plot_duals, plot_inputs, plot_scenario_comparison, plot_schedule
from src.scenarios import scale_prices, scale_pv, set_tariffs

RESULTS_DIR = Path(__file__).resolve().parent / "results"

MODELS = {"Q2_linear": LinearDisutilityModel,
          "Q2_quadratic": QuadraticDisutilityModel,
          "Q2_nonseparable": NonSeparableDisutilityModel,  # 2.e: memory 0.7 set in the class
          "Q3": MinEnergyModel,
          "Q3_battery": BatteryModel,
          # Add new models here
          }

# Questions without a data file of their own, and the case whose data they reuse
DATA_CASES = {"Q2_nonseparable": "Q2_quadratic"}


def load_case(question: str) -> InputData:
    """Load the data of ``question``, or of the case it reuses in ``DATA_CASES``, under the name ``question``."""
    return replace(load_question(DATA_CASES.get(question, question)), question=question)


def run_base_case(question: str, out: Path, show: bool) -> Results | None:
    data = load_case(question)
    print(data.summary(), "\n")
    plot_inputs(data, save_to=out / "inputs.png")

    # Lookup the correct model implemented based on question string
    # model = FlexibleConsumerModel(data).build()
    model = MODELS.get(question, FlexibleConsumerModel)(data).build()
    try:
        results = model.solve()
    except NotImplementedError as e:
        print(f"[skipped] {e}")
        return None

    print(results, "\n")
    results.save(out)
    plot_schedule(results, data, save_to=out / "schedule.png")
    plot_duals(results, data, save_to=out / "duals.png")
    if show:
        matplotlib.pyplot.show()
    return results


def run_scenarios(question: str, out: Path) -> dict[str, Results]:
    """Example sensitivity analysis. Replace with the scenarios you design in Question 1.g."""
    base = load_case(question)
    scenarios = {
        "base": base,
        "flat_prices": scale_prices(base, factor=0.0, keep_mean=True),
        "double_spread": scale_prices(base, factor=2.0, keep_mean=True),
        "no_tariffs": set_tariffs(base, import_tariff=0.0, export_tariff=0.0),
        "no_pv": scale_pv(base, factor=0.0),
    }
    runs: dict[str, Results] = {}
    for name, data in scenarios.items():
        #results = FlexibleConsumerModel(data).build().solve()
        results = MODELS.get(question, FlexibleConsumerModel)(data).build().solve()
        results.save(out, tag=name)
        runs[name] = results
        print(f"{name:>14}: cost {results.objective:8.2f} DKK | import {results.hourly['import'].sum():5.1f} kWh"
              f" | export {results.hourly['export'].sum():5.1f} kWh")
    plot_scenario_comparison(runs, "objective", save_to=out / "scenarios_cost.png")
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--question", default="Q1_caseA", choices=list_questions() + list(DATA_CASES), help="data case to use")
    parser.add_argument("--scenarios", action="store_true", help="also run the example sensitivity scenarios")
    parser.add_argument("--show", action="store_true", help="open the figures in a window")
    args = parser.parse_args()

    out = RESULTS_DIR / args.question
    out.mkdir(parents=True, exist_ok=True)
    if not args.show:
        matplotlib.use("Agg")

    base = run_base_case(args.question, out, args.show)
    if args.scenarios and base is not None:
        run_scenarios(args.question, out)
    print(f"\nOutputs written to {out}")


if __name__ == "__main__":
    main()
