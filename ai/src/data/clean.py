"""Clean and standardize raw FamilyMart schedule Excel files.

Handles known data quality issues:
  - "1530.0" → 15.5 (Excel auto-format of 15:30)
  - "x" → NaN (non-standard absence marking)
  - Blank/empty cells → NaN (off-duty)
  - Overnight shifts (23→7, 23→9) → correct hour calculation
  - Mixed types in columns (str vs float)

Outputs:
  - cleaned_schedules.csv: standardized format with columns:
    date, employee_id, shift_start, shift_end, shift_hours, shift_class
  - schedule_summary.csv: per-employee monthly summary

Usage:
    python -m data.clean
    python -m data.clean --input-dir ../data --output-dir ../data/cleaned
"""

import argparse
import glob
import os
from pathlib import Path

import pandas as pd


def parse_time_value(val) -> float | None:
    """Parse a time value from raw Excel data; None means off-duty."""
    if pd.isna(val):
        return None
    if isinstance(val, str):
        val = val.strip()
        if val.lower() in ("x", "", "-"):
            return None
        try:
            val = float(val)
        except ValueError:
            return None

    val = float(val)

    if val > 24:
        hours = int(val) // 100
        minutes = int(val) % 100
        if 0 <= hours <= 23 and 0 <= minutes <= 59:
            return hours + minutes / 60.0
        return None

    return val


def compute_shift_hours(start: float, end: float) -> float:
    """Shift duration in hours, handling overnight shifts."""
    if end > start:
        return end - start
    return (24.0 - start) + end


def classify_shift(start: float) -> int:
    """0=off, 1=morning(6-11), 2=afternoon(11-15), 3=evening/night(15-24+)."""
    if 6 <= start < 11:
        return 1
    elif 11 <= start < 15:
        return 2
    elif 15 <= start <= 24 or 0 <= start < 6:
        return 3
    return 0


def load_and_clean_excel(file_path: str) -> pd.DataFrame:
    """Load a single Excel file and return cleaned records."""
    df = pd.read_excel(file_path)

    df = df.rename(columns={"date": "day"})

    if "year" in df.columns and "month" in df.columns and "day" in df.columns:
        df["date"] = pd.to_datetime(
            dict(year=df.year, month=df.month, day=df.day),
            errors="coerce",
        )
        df = df.dropna(subset=["date"])
    else:
        return pd.DataFrame()

    all_cols = df.columns.tolist()
    employees = sorted(
        set(
            col.replace("_s", "").replace("_e", "")
            for col in all_cols
            if col.endswith(("_s", "_e"))
        )
    )

    records = []
    for _, row in df.iterrows():
        date = row.date
        for emp in employees:
            start_col = f"{emp}_s"
            end_col = f"{emp}_e"
            if start_col not in df.columns or end_col not in df.columns:
                continue

            start = parse_time_value(row.get(start_col))
            end = parse_time_value(row.get(end_col))

            if start is None or end is None:
                continue

            shift_hours = compute_shift_hours(start, end)
            shift_class = classify_shift(start)

            emp_id = emp.replace("member_", "").strip().upper()

            records.append(
                {
                    "date": date,
                    "employee_id": emp_id,
                    "shift_start": round(start, 2),
                    "shift_end": round(end, 2),
                    "shift_hours": round(shift_hours, 2),
                    "shift_class": shift_class,
                    "day_of_week": date.weekday(),
                }
            )

    return pd.DataFrame(records)


def clean_all(input_dir: str, output_dir: str) -> None:
    """Process all Excel files in input_dir and write cleaned CSVs."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    xlsx_files = sorted(glob.glob(str(input_path / "*.xlsx")))
    if not xlsx_files:
        print(f"No .xlsx files found in {input_path}")
        return

    all_dfs = []
    for f in xlsx_files:
        print(f"Processing: {os.path.basename(f)}")
        df = load_and_clean_excel(f)
        if not df.empty:
            all_dfs.append(df)
            print(f"  -> {len(df)} shift records")
        else:
            print("  -> skipped (no valid data)")

    if not all_dfs:
        print("No valid data found.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values(["date", "employee_id"]).reset_index(drop=True)
    combined = combined.drop_duplicates(subset=["date", "employee_id", "shift_start"])

    schedules_path = output_path / "cleaned_schedules.csv"
    combined.to_csv(schedules_path, index=False)
    print(f"\nCleaned schedules: {schedules_path} ({len(combined)} records)")

    summary = (
        combined.groupby("employee_id")
        .agg(
            total_shifts=("shift_hours", "count"),
            total_hours=("shift_hours", "sum"),
            avg_shift_hours=("shift_hours", "mean"),
            morning_shifts=("shift_class", lambda x: (x == 1).sum()),
            afternoon_shifts=("shift_class", lambda x: (x == 2).sum()),
            evening_shifts=("shift_class", lambda x: (x == 3).sum()),
            first_date=("date", "min"),
            last_date=("date", "max"),
        )
        .round(2)
        .reset_index()
    )

    summary_path = output_path / "schedule_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Schedule summary: {summary_path}")

    print("\nDataset overview:")
    print(f"  Date range: {combined.date.min().date()} to {combined.date.max().date()}")
    print(f"  Employees: {combined.employee_id.nunique()}")
    print(f"  Total shifts: {len(combined)}")
    print(
        "  Distinct shift patterns: "
        f"{combined[['shift_start', 'shift_end']].drop_duplicates().shape[0]}"
    )
    print("\nPer-employee summary:")
    print(summary.to_string(index=False))


def main() -> None:
    default_input = Path(__file__).resolve().parent.parent.parent / "data"
    parser = argparse.ArgumentParser(description="Clean and standardize schedule Excel data")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(default_input),
        help="Directory containing .xlsx files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(default_input / "cleaned"),
        help="Output directory for cleaned CSVs",
    )

    args = parser.parse_args()
    clean_all(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
