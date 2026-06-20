from etfs.share_class import (
    classify_share_class,
    distributing_ids_when_accumulating_exists,
    normalize_fund_name,
)


def test_korea_acc_dist_pair():
    names = {
        "acc": "iShares MSCI Korea UCITS ETF (Acc)",
        "dist": "iShares MSCI Korea UCITS ETF (Dist)",
    }
    excluded = distributing_ids_when_accumulating_exists(names)
    assert excluded == {"dist"}
    assert normalize_fund_name(names["acc"]) == normalize_fund_name(names["dist"])


def test_vanguard_sp500_pair():
    names = {
        "acc": "Vanguard S&P 500 UCITS ETF (USD) Accumulating",
        "dist": "Vanguard S&P 500 UCITS ETF (USD) Distributing",
    }
    excluded = distributing_ids_when_accumulating_exists(names)
    assert excluded == {"dist"}


def test_acc_only_no_exclusion():
    names = {"solo": "iShares Core S&P 500 UCITS ETF USD (Acc)"}
    assert distributing_ids_when_accumulating_exists(names) == set()


def test_classify_share_class():
    assert classify_share_class("Foo UCITS ETF (Acc)") == "acc"
    assert classify_share_class("Foo UCITS ETF Distributing") == "dist"
