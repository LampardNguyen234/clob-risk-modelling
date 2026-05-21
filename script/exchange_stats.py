"""
Exchange stats fetcher and monthly summarizer.

This script pulls explorer stats from the Supabase RPC used in docs/volume_stats.md.
It keeps the request headers unchanged, splits any date range into calendar-month
windows, and computes monthly exchange-level metrics.
"""

from __future__ import annotations

import argparse
import calendar
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

import numpy as np
import pandas as pd
import requests


API_URL = "https://pwngicypignjctnkjrmw.supabase.co/rest/v1/rpc/get_explorer_data"

# Keep these headers identical to docs/volume_stats.md.
REQUEST_HEADERS = {
    "apikey": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InB3bmdpY3lwaWduamN0bmtqcm13Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3Njk1MTQxODYsImV4cCI6MjA4NTA5MDE4Nn0.Wj-EzEsl4B-p-aX-VRBdcVjFE_593oHuoML_304Un1E",
    "content-type": "application/json",
    "origin": "https://perpvision.0xtria.dev",
    "referer": "https://perpvision.0xtria.dev/",
}

DEFAULT_METRICS = ["volume", "new_users", "cumulative_users"]
FLOW_METRICS = {"volume", "new_users"}
SNAPSHOT_METRICS = {"cumulative_users"}
REPORT_METRICS = ["new_users", "cumulative_users", "volume", "volume_per_user"]
DAILY_STAT_METRICS = [
    "daily_volume",
    "cumulative_users",
    "daily_new_users",
]
SUMMARY_STATS = ["min", "p5", "mean", "median", "p95", "p99", "max"]


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def month_end(day: date) -> date:
    last_day = calendar.monthrange(day.year, day.month)[1]
    return date(day.year, day.month, last_day)


def iter_month_windows(start: date, end: date) -> Iterator[Tuple[str, str]]:
    """
    Split a date range into calendar-month windows.

    Each yielded window is guaranteed to stay within one calendar month and
    therefore satisfies the "max 1-month duration" request constraint.
    """
    current = start
    while current <= end:
        window_end = min(month_end(current), end)
        yield current.isoformat(), window_end.isoformat()
        current = window_end + timedelta(days=1)


def fetch_explorer_data(
    session: requests.Session,
    slugs: Sequence[str],
    metrics: Sequence[str],
    from_date: str,
    to_date: str,
) -> pd.DataFrame:
    payload = {
        "p_slugs": list(slugs),
        "p_metrics": list(metrics),
        "p_from": from_date,
        "p_to": to_date,
    }

    response = session.post(
        API_URL,
        headers=REQUEST_HEADERS,
        data=json.dumps(payload),
        timeout=60,
    )
    response.raise_for_status()

    data = response.json()
    if not data:
        return pd.DataFrame(columns=["ts", "project_slug", "metric", "value", "granularity"])

    return pd.DataFrame(data)


def fetch_explorer_data_for_range(
    slugs: Sequence[str],
    metrics: Sequence[str],
    from_date: str,
    to_date: str,
) -> pd.DataFrame:
    start = parse_date(from_date)
    end = parse_date(to_date)
    session = requests.Session()
    frames: List[pd.DataFrame] = []

    for window_from, window_to in iter_month_windows(start, end):
        frame = fetch_explorer_data(session, slugs, metrics, window_from, window_to)
        if frame.empty:
            continue
        frame["request_from"] = window_from
        frame["request_to"] = window_to
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["ts", "project_slug", "metric", "value", "granularity"])

    return pd.concat(frames, ignore_index=True)


def normalize_raw_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(columns=["ts", "project_slug", "metric", "value", "granularity", "month"])

    df = raw_df.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df["month"] = df["ts"].dt.to_period("M").astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def build_daily_metric_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_raw_df(raw_df)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "ts",
                "project_slug",
                "daily_new_users",
                "daily_volume",
                "cumulative_users",
                "volume_per_user",
                "volume_per_new_user",
            ]
        )

    daily_df = (
        df.pivot_table(
            index=["ts", "project_slug"],
            columns="metric",
            values="value",
            aggfunc="last",
        )
        .reset_index()
        .sort_values(["project_slug", "ts"])
    )

    daily_df.columns.name = None
    rename_map = {
        "new_users": "daily_new_users",
        "volume": "daily_volume",
        "cumulative_users": "cumulative_users",
    }
    daily_df = daily_df.rename(columns=rename_map)

    for column in ["daily_new_users", "daily_volume", "cumulative_users"]:
        if column not in daily_df.columns:
            daily_df[column] = np.nan

    numeric_columns = [
        "daily_new_users",
        "daily_volume",
        "cumulative_users",
    ]
    for column in numeric_columns:
        daily_df[column] = pd.to_numeric(daily_df[column], errors="coerce")

    daily_df["daily_new_users"] = daily_df["daily_new_users"].fillna(0.0)
    fallback_cumulative_users = daily_df.groupby("project_slug")["daily_new_users"].cumsum()
    daily_df["cumulative_users"] = daily_df["cumulative_users"].fillna(fallback_cumulative_users)
    daily_df["volume_per_user"] = daily_df["daily_volume"] / daily_df["cumulative_users"].replace(0, np.nan)
    daily_df["volume_per_new_user"] = (
        daily_df["daily_volume"] / daily_df["daily_new_users"].replace(0, np.nan)
    )

    return daily_df


def _series_stat_map(series: pd.Series) -> dict:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {stat: None for stat in SUMMARY_STATS}

    return {
        "min": float(clean.min()),
        "p5": float(clean.quantile(0.05)),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "p95": float(clean.quantile(0.95)),
        "p99": float(clean.quantile(0.99)),
        "max": float(clean.max()),
    }


def build_exchange_stat_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    daily_df = build_daily_metric_frame(raw_df)
    if daily_df.empty:
        return pd.DataFrame(columns=["project_slug"])

    metric_columns = {
        "daily_volume": "daily_volume",
        "cumulative_users": "cumulative_users",
        "daily_new_users": "daily_new_users",
    }

    rows = []
    for project_slug, group in daily_df.groupby("project_slug", sort=True):
        row = {"project_slug": project_slug}
        for output_prefix, column in metric_columns.items():
            stats = _series_stat_map(group[column])
            for stat_name, value in stats.items():
                row[f"{output_prefix}_{stat_name}"] = value

        row["avg_volume_per_user"] = float(group["volume_per_user"].dropna().mean())
        row["avg_volume_per_new_user"] = float(group["volume_per_new_user"].dropna().mean())
        rows.append(row)

    return pd.DataFrame(rows).sort_values("project_slug").reset_index(drop=True)


def build_monthly_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_raw_df(raw_df)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "month",
                "project_slug",
                "new_users",
                "cumulative_users",
                "volume",
                "volume_per_user",
            ]
        )

    rows = []
    for (month, project_slug), group in df.groupby(["month", "project_slug"], sort=True):
        metrics = {}
        for metric, metric_group in group.groupby("metric", sort=False):
            series = metric_group.sort_values("ts")["value"]
            if metric in FLOW_METRICS:
                metrics[metric] = float(series.sum())
            elif metric in SNAPSHOT_METRICS:
                metrics[metric] = float(series.iloc[-1])
            else:
                metrics[metric] = float(series.sum())

        new_users = metrics.get("new_users", 0.0)
        cumulative_users = metrics.get("cumulative_users")
        volume = metrics.get("volume", 0.0)
        volume_per_user = volume / cumulative_users if cumulative_users else None

        rows.append(
            {
                "month": month,
                "project_slug": project_slug,
                "new_users": new_users,
                "cumulative_users": cumulative_users,
                "volume": volume,
                "volume_per_user": volume_per_user,
            }
        )

    summary = pd.DataFrame(rows).sort_values(["month", "project_slug"]).reset_index(drop=True)
    return summary


def build_monthly_report_tables(summary_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if summary_df.empty:
        return {
            metric: pd.DataFrame(index=pd.Index([], name="month"))
            for metric in REPORT_METRICS
        }

    tables: dict[str, pd.DataFrame] = {}
    ordered_slugs = sorted(summary_df["project_slug"].dropna().unique().tolist())

    for metric in REPORT_METRICS:
        table = (
            summary_df.pivot_table(
                index="month",
                columns="project_slug",
                values=metric,
                aggfunc="last",
            )
            .reindex(columns=ordered_slugs)
            .sort_index()
        )
        table.columns.name = None
        tables[metric] = table

    return tables


def build_monthly_report_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    tables = build_monthly_report_tables(summary_df)
    if not tables:
        return pd.DataFrame(index=pd.Index([], name="month"))

    combined = pd.concat(tables, axis=1)
    combined = combined.reindex(REPORT_METRICS, axis=1, level=0)
    combined.index.name = "month"
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch monthly exchange stats and compute summary metrics."
    )
    parser.add_argument(
        "--slugs",
        default="hyperliquid,lighter,extended,variational,nado,pacifica",
        help="Comma-separated exchange slugs.",
    )
    parser.add_argument(
        "--from-date",
        required=True,
        help="Start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--to-date",
        required=True,
        help="End date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metrics list.",
    )
    parser.add_argument(
        "--output",
        default="data/exchange_stats/monthly_summary.csv",
        help="Path for the monthly summary CSV.",
    )
    parser.add_argument(
        "--raw-output",
        default="data/exchange_stats/raw_rows.csv",
        help="Path for the raw long-form CSV.",
    )
    args = parser.parse_args()

    slugs = parse_csv_list(args.slugs)
    metrics = parse_csv_list(args.metrics)

    raw_df = fetch_explorer_data_for_range(
        slugs=slugs,
        metrics=metrics,
        from_date=args.from_date,
        to_date=args.to_date,
    )
    summary_df = build_monthly_summary(raw_df)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)

    raw_output_path = Path(args.raw_output)
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(raw_output_path, index=False)

    print(f"Saved summary to {output_path}")
    print(f"Saved raw rows to {raw_output_path}")


if __name__ == "__main__":
    main()
