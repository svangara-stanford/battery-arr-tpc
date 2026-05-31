import pandas as pd

from battery_aar.paper_reproduction.bayesgap import BayesGapConfig, run_closed_loop
from battery_aar.paper_reproduction.policy_space import generate_policy_space


def test_bayesgap_tiny_policy_space(tmp_path):
    policies = pd.DataFrame(
        {
            "C1": [3.6, 4.0, 4.4, 5.2],
            "C2": [3.6, 4.0, 4.4, 5.2],
            "C3": [3.6, 4.0, 4.4, 5.2],
            "C4": [4.0, 3.5, 3.0, 2.5],
        }
    )
    policy_csv = tmp_path / "policies.csv"
    policies.to_csv(policy_csv, index=False)
    pred = tmp_path / "pred.csv"
    policies.assign(Prediction=[800, 900, 700, 850]).to_csv(pred, index=False)
    results = run_closed_loop(policy_csv, tmp_path / "bg", [pred], BayesGapConfig(bsize=2, seed=1))
    assert len(results) == 2
    assert (tmp_path / "bg" / "round_1_next_batch.csv").exists()


def test_bayesgap_real_policy_space_round0(tmp_path):
    policy_csv = tmp_path / "policies_all.csv"
    generate_policy_space().to_csv(policy_csv, index=False)
    results = run_closed_loop(policy_csv, tmp_path / "bg", config=BayesGapConfig(bsize=4, seed=0))
    assert len(results[0]["next_batch"]) == 4


def test_final_posterior_ranking_has_all_policies_and_paper_protocols(tmp_path):
    policy_csv = tmp_path / "policies_all.csv"
    generate_policy_space().to_csv(policy_csv, index=False)
    run_closed_loop(policy_csv, tmp_path / "bg", config=BayesGapConfig(bsize=4, seed=0))
    ranking = pd.read_csv(tmp_path / "bg" / "final_posterior_ranking.csv")
    check = pd.read_csv(tmp_path / "bg" / "final_paper_top_protocol_check.csv")
    assert len(ranking) == 224
    assert len(check) == 3
    assert check["exists_in_policy_space"].all()
