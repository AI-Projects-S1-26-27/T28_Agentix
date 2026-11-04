import pickle

import numpy as np
import pandas as pd

from data.src.parse_real_wesad_full import parse_subject


def test_parser_preserves_transient_label_and_missing_heart_rate(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    raw_dir = tmp_path / "data" / "raw"
    hr_dir = raw_dir / "S2_E4_Data"
    (tmp_path / "data" / "processed").mkdir(parents=True)
    hr_dir.mkdir(parents=True)

    labels = np.ones(1400, dtype=int)
    labels[350] = 0
    labels[1050] = 2
    recording = {
        "signal": {
            "wrist": {
                "EDA": np.arange(8, dtype=float),
                "TEMP": np.arange(8, dtype=float),
            }
        },
        "label": labels,
    }
    with (raw_dir / "S2.pkl").open("wb") as stream:
        pickle.dump(recording, stream)

    (hr_dir / "HR.csv").write_text(
        "1700000000\n1\n72.0\nNaN\n",
        encoding="utf-8",
    )

    result = parse_subject("S2")
    assert result is not None

    parsed = pd.read_csv(tmp_path / "data" / "processed" / "wesad_s2_real.csv")
    assert parsed["stress_label"].tolist() == [0, 2]
    assert parsed.loc[0, "heart_rate_bpm"] == 72.0
    assert pd.isna(parsed.loc[1, "heart_rate_bpm"])


def test_parser_uses_eda_timestamp_without_heart_rate_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw_dir = tmp_path / "data" / "raw"
    eda_dir = raw_dir / "S2_E4_Data"
    (tmp_path / "data" / "processed").mkdir(parents=True)
    eda_dir.mkdir(parents=True)

    recording = {
        "signal": {
            "wrist": {
                "EDA": np.arange(4, dtype=float),
                "TEMP": np.arange(4, dtype=float),
            }
        },
        "label": np.ones(700, dtype=int),
    }
    with (raw_dir / "S2.pkl").open("wb") as stream:
        pickle.dump(recording, stream)

    (eda_dir / "EDA.csv").write_text(
        "1700000100\n4\n0.1\n",
        encoding="utf-8",
    )

    result = parse_subject("S2")
    parsed = pd.read_csv(tmp_path / "data" / "processed" / "wesad_s2_real.csv")

    assert result is not None
    assert parsed.loc[0, "timestamp"] == "2023-11-14T22:15:00Z"
    assert pd.isna(parsed.loc[0, "heart_rate_bpm"])


def test_parser_leaves_timestamp_empty_when_metadata_is_unavailable(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    raw_dir = tmp_path / "data" / "raw"
    (tmp_path / "data" / "processed").mkdir(parents=True)
    raw_dir.mkdir(parents=True)

    recording = {
        "signal": {
            "wrist": {
                "EDA": np.arange(4, dtype=float),
                "TEMP": np.arange(4, dtype=float),
            }
        },
        "label": np.ones(700, dtype=int),
    }
    with (raw_dir / "S2.pkl").open("wb") as stream:
        pickle.dump(recording, stream)

    result = parse_subject("S2")
    parsed = pd.read_csv(tmp_path / "data" / "processed" / "wesad_s2_real.csv")

    assert result is not None
    assert result["missing_timestamps"] == 1
    assert pd.isna(parsed.loc[0, "timestamp"])
    assert "No E4 timestamp metadata found" in capsys.readouterr().out
