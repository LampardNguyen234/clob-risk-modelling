"""
Extended Exchange Historical Data Crawler

Fetches historical kline data from Extended exchange:
- Trades (actual trade prices)
- Mark Prices
- Index Prices

Data is saved incrementally to CSV files with final validation.
"""

import requests
import pandas as pd
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
import argparse
import random
from concurrent.futures import ThreadPoolExecutor, as_completed


class ExtendedCrawler:
    """Simple, reliable crawler for Extended exchange kline data."""

    MAINNET_URL = "https://api.starknet.extended.exchange/api/v1"
    TESTNET_URL = "https://api.starknet.sepolia.extended.exchange/api/v1"
    MAX_CANDLES_PER_REQUEST = 2800

    CANDLE_TYPE_MAPPING = {
        'trades': 'trades',
        'mark_price': 'mark-prices',
        'index_price': 'index-prices'
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        data_dir: str = "./data",
        network: str = "mainnet",
        rate_limit_delay: float = 0.06,
        max_retries: int = 5
    ):
        self.api_key = api_key
        self.data_dir = Path(data_dir)
        self.base_url = self.MAINNET_URL if network == "mainnet" else self.TESTNET_URL
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.session = requests.Session()

        headers = {'User-Agent': 'ExtendedCrawler/2.0'}
        if self.api_key:
            headers['X-Api-Key'] = self.api_key
        self.session.headers.update(headers)

    def _make_request(self, market: str, candle_type: str, params: dict) -> List:
        """Make API request with retry logic."""
        api_type = self.CANDLE_TYPE_MAPPING.get(candle_type, candle_type)
        url = f"{self.base_url}/info/candles/{market}/{api_type}"

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                time.sleep(self.rate_limit_delay)
                return response.json()
            except requests.exceptions.HTTPError as e:
                if response.status_code == 429 and attempt < self.max_retries:
                    wait_time = (2 ** (attempt + 1)) + random.uniform(0, 1)
                    print(f"  Rate limit hit. Retry {attempt + 1}/{self.max_retries} after {wait_time:.2f}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"HTTP Error: {e}")
                    return []
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"  Request error: {e}. Retry {attempt + 1}/{self.max_retries} after {wait_time:.2f}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Error: {e}")
                    return []
        return []

    def _process_response(self, data: List) -> pd.DataFrame:
        """Convert API response to DataFrame."""
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)

        # Handle nested response structure
        if 'data' in df.columns and 'status' in df.columns:
            df = pd.json_normalize(df['data'])

        # Rename columns
        column_mapping = {
            'T': 'open_time',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume'
        }
        df = df.rename(columns=column_mapping)

        # Convert types (make timezone-aware in UTC)
        if 'open_time' in df.columns:
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)

        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = df[col].astype(float)

        return df

    def fetch_and_save_incrementally(
        self,
        market: str,
        candle_type: str,
        interval: str,
        start_date: str,
        end_date: str,
        output_file: Path,
        batch_size: int = 50000
    ) -> int:
        """
        Fetch historical data and save incrementally in batches.
        Only fetches missing data gaps if file already exists.

        Args:
            market: Trading pair (e.g., 'BTC-USD')
            candle_type: 'trades', 'mark_price', or 'index_price'
            interval: Candle interval (e.g., '1m', '5m', '1h')
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            output_file: Path to save CSV file
            batch_size: Number of records to accumulate before saving

        Returns:
            Total number of records fetched
        """
        start_dt = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)

        start_time = int(start_dt.timestamp() * 1000)
        end_time = int(end_dt.timestamp() * 1000)

        print(f"\nFetching {candle_type} for {market} ({interval})")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Batch size: {batch_size} records")
        print("-" * 80)

        # Check existing file for gaps
        existing_df = None
        fetch_ranges = [(start_time, end_time)]  # Default: fetch entire range

        if output_file.exists():
            try:
                existing_df = pd.read_csv(output_file)
                if not existing_df.empty and 'open_time' in existing_df.columns:
                    existing_df['open_time'] = pd.to_datetime(existing_df['open_time'], utc=True)
                    first_time = int(existing_df['open_time'].iloc[0].timestamp() * 1000)
                    last_time = int(existing_df['open_time'].iloc[-1].timestamp() * 1000)

                    print(f"Found existing file with {len(existing_df)} records:")
                    print(f"  First record: {existing_df['open_time'].iloc[0]}")
                    print(f"  Last record: {existing_df['open_time'].iloc[-1]}")

                    # Determine what ranges need to be fetched
                    fetch_ranges = []
                    if start_time < first_time:
                        fetch_ranges.append((start_time, first_time - 1))
                        print(f"  Gap BEFORE: {start_date} to {existing_df['open_time'].iloc[0]}")
                    if last_time < end_time:
                        fetch_ranges.append((last_time + 1, end_time))
                        print(f"  Gap AFTER: {existing_df['open_time'].iloc[-1]} to {end_date}")

                    if not fetch_ranges:
                        print("  ✓ No gaps detected - data already complete for this range")
                        return len(existing_df)

                    print(f"  Fetching {len(fetch_ranges)} gap(s)...")
            except Exception as e:
                print(f"Could not read existing file: {e}")
                print("Will fetch entire range")
                existing_df = None
                fetch_ranges = [(start_time, end_time)]

        current_batch = []
        total_records = 0
        batch_count = 0

        # Fetch data for each gap range
        for range_idx, (range_start, range_end) in enumerate(fetch_ranges, 1):
            print(f"\nFetching range {range_idx}/{len(fetch_ranges)}: " +
                  f"{datetime.fromtimestamp(range_start/1000, tz=timezone.utc)} to " +
                  f"{datetime.fromtimestamp(range_end/1000, tz=timezone.utc)}")

            current_end = range_end
            iteration = 0
            previous_earliest = None

            # Fetch backwards from range_end
            while current_end > range_start:
                iteration += 1

                params = {
                    'interval': interval,
                    'limit': self.MAX_CANDLES_PER_REQUEST,
                    'endTime': current_end
                }

                print(f"  Iteration {iteration}, type {candle_type}: Fetching before {datetime.fromtimestamp(current_end/1000, tz=timezone.utc)}")

                data = self._make_request(market, candle_type, params)
                df = self._process_response(data)

                if df.empty:
                    print("    No data returned. Stopping.")
                    break

                original_length = len(df)
                print(f"    API returned {original_length} records: {df['open_time'].min()} to {df['open_time'].max()}")

                # Filter to this specific range
                range_start_dt = pd.to_datetime(range_start, unit='ms', utc=True)
                range_end_dt = pd.to_datetime(range_end, unit='ms', utc=True)
                df = df[df['open_time'] >= range_start_dt]
                df = df[df['open_time'] <= range_end_dt]

                if df.empty:
                    print("    All records outside range. Stopping.")
                    break

                earliest = df['open_time'].min()
                latest = df['open_time'].max()
                print(f"    After filtering: {len(df)} records from {earliest} to {latest}")

                # Check for infinite loop
                if previous_earliest is not None and earliest >= previous_earliest:
                    print(f"    WARNING: Not progressing backwards. API limit reached at {earliest}")
                    break
                previous_earliest = earliest

                # Add to current batch
                current_batch.append(df)
                total_records += len(df)

                # Check if we should save the batch
                current_batch_size = sum(len(b) for b in current_batch)
                if current_batch_size >= batch_size:
                    self._save_batch(current_batch, output_file, batch_count, existing_df is not None)
                    batch_count += 1
                    current_batch = []

                # Update for next iteration (subtract 1ms to avoid overlap)
                current_end = int(earliest.timestamp() * 1000) - 1

                # Stop if we got less than max (reached beginning of data)
                if original_length < self.MAX_CANDLES_PER_REQUEST:
                    print(f"    API returned {original_length} < {self.MAX_CANDLES_PER_REQUEST}. Reached beginning.")
                    break

        # Save any remaining data in the last batch
        if current_batch:
            self._save_batch(current_batch, output_file, batch_count, existing_df is not None)

        # If we fetched new data and have existing data, we need to merge
        if total_records > 0 and existing_df is not None:
            print(f"\nTotal NEW records fetched: {total_records}")
            print("Merging with existing data...")
            return total_records
        elif total_records > 0:
            print(f"\nTotal records fetched: {total_records}")
            return total_records
        else:
            return len(existing_df) if existing_df is not None else 0

    def _save_batch(self, batch: List[pd.DataFrame], output_file: Path, batch_number: int, append_mode: bool = False):
        """Save a batch of DataFrames to file (in descending order since we fetch backwards)."""
        if not batch:
            return

        # Combine batch and sort descending (newest first)
        batch_df = pd.concat(batch, ignore_index=True)
        batch_df = batch_df.sort_values('open_time', ascending=False).reset_index(drop=True)

        # Create directory if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # When filling gaps, always append. Otherwise, write header only for first batch.
        if append_mode:
            mode = 'a'
            header = False
        else:
            mode = 'a' if output_file.exists() else 'w'
            header = not output_file.exists()

        batch_df.to_csv(output_file, mode=mode, header=header, index=False)

        action = "appended" if mode == 'a' else "written"
        print(f"    → Saved batch {batch_number + 1}: {len(batch_df)} records {action} to file")

    def clean_and_finalize(self, filepath: Path) -> bool:
        """
        Read the saved file, remove duplicates, sort, and save back.
        This ensures the final file is 100% correct.

        Args:
            filepath: Path to the CSV file

        Returns:
            True if successful
        """
        if not filepath.exists():
            print(f"ERROR: File {filepath} does not exist")
            return False

        print("\n" + "=" * 80)
        print("CLEANING AND FINALIZING")
        print("=" * 80)

        # Read the file
        print(f"Reading: {filepath}")
        df = pd.read_csv(filepath)
        original_count = len(df)
        print(f"Original rows: {original_count}")

        if df.empty:
            print("ERROR: File is empty")
            return False

        # Parse datetime
        df['open_time'] = pd.to_datetime(df['open_time'])

        # Remove duplicates
        before_dedup = len(df)
        df = df.drop_duplicates(subset=['open_time'], keep='first')
        duplicates_removed = before_dedup - len(df)
        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} duplicates")
        else:
            print("No duplicates found")

        # Sort by time
        df = df.sort_values('open_time').reset_index(drop=True)
        print("Sorted by open_time")

        # Save back to file
        df.to_csv(filepath, index=False)
        print(f"Final rows: {len(df)}")
        print(f"Saved clean data to: {filepath}")

        return True

    def validate_data(
        self,
        filepath: Path,
        expected_start: str,
        expected_end: str
    ) -> bool:
        """Validate saved data."""
        if not filepath.exists():
            print("ERROR: File does not exist")
            return False

        print("\n" + "=" * 80)
        print("VALIDATION")
        print("=" * 80)

        df = pd.read_csv(filepath)
        print(f"File: {filepath}")
        print(f"Total rows: {len(df)}")

        if df.empty:
            print("ERROR: File is empty")
            return False

        # Parse dates
        df['open_time'] = pd.to_datetime(df['open_time'])
        actual_start = df['open_time'].min()
        actual_end = df['open_time'].max()

        expected_start_dt = datetime.strptime(expected_start, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        expected_end_dt = datetime.strptime(expected_end, '%Y-%m-%d').replace(tzinfo=timezone.utc)

        print(f"\nDate Range:")
        print(f"  Expected: {expected_start} to {expected_end}")
        print(f"  Actual:   {actual_start} to {actual_end}")

        # Check for duplicates
        duplicates = df.duplicated(subset=['open_time']).sum()
        if duplicates > 0:
            print(f"\n✗ WARNING: {duplicates} duplicate timestamps found")
        else:
            print(f"\n✓ No duplicates")

        # Check if data is sorted
        is_sorted = df['open_time'].is_monotonic_increasing
        if is_sorted:
            print("✓ Data is sorted")
        else:
            print("✗ WARNING: Data is not sorted")

        # Check coverage
        coverage_ok = True
        if actual_start <= expected_start_dt:
            print(f"✓ Start coverage OK")
        else:
            print(f"✗ Start coverage INCOMPLETE (gap: {expected_start_dt} to {actual_start})")
            coverage_ok = False

        if actual_end >= expected_end_dt - pd.Timedelta(days=1):
            print(f"✓ End coverage OK")
        else:
            print(f"✗ End coverage INCOMPLETE (gap: {actual_end} to {expected_end_dt})")
            coverage_ok = False

        print("=" * 80)

        return duplicates == 0 and is_sorted and coverage_ok


def process_single_candle_type(
    crawler: ExtendedCrawler,
    market: str,
    candle_type: str,
    interval: str,
    start_date: str,
    end_date: str,
    batch_size: int
) -> tuple:
    """
    Process a single candle type (fetch, clean, validate).

    Returns:
        Tuple of (candle_type, success, message)
    """
    try:
        # Prepare output file
        subdir = crawler.data_dir / "extended" / candle_type
        filename = f"{market}_{interval}.csv"
        output_file = subdir / filename

        # Fetch and save incrementally
        total_records = crawler.fetch_and_save_incrementally(
            market=market,
            candle_type=candle_type,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            output_file=output_file,
            batch_size=batch_size
        )

        if total_records > 0:
            # Clean and finalize the file
            crawler.clean_and_finalize(output_file)

            # Validate
            is_valid = crawler.validate_data(
                filepath=output_file,
                expected_start=start_date,
                expected_end=end_date
            )

            if is_valid:
                return (candle_type, True, f"✓ Completed and validated: {output_file}")
            else:
                return (candle_type, True, f"⚠ Completed with warnings: {output_file}")
        else:
            return (candle_type, False, f"No data fetched")

    except Exception as e:
        return (candle_type, False, f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Extended Exchange Historical Data Crawler')
    parser.add_argument('--api-key', type=str, help='Optional API key')
    parser.add_argument('--symbols', type=str, required=True, help='Comma-separated symbols (e.g., BTC,ETH)')
    parser.add_argument('--quote', type=str, default='USD', help='Quote currency (default: USD)')
    parser.add_argument('--data-type', type=str, default='all',
                        help='Comma-separated types: trades, mark_price, index_price, or all')
    parser.add_argument('--interval', type=str, default='1m', help='Candle interval (e.g., 1m, 5m, 1h)')
    parser.add_argument('--start-date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--data-dir', type=str, default='./data', help='Output directory')
    parser.add_argument('--network', type=str, default='mainnet', choices=['mainnet', 'testnet'])
    parser.add_argument('--batch-size', type=int, default=50000, help='Records per batch before saving')
    parser.add_argument('--rate-limit-delay', type=float, default=0.06)
    parser.add_argument('--max-retries', type=int, default=5)

    args = parser.parse_args()

    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(',')]
    markets = [f"{s.upper()}-{args.quote.upper()}" if '-' not in s else s for s in symbols]

    # Parse candle types
    if args.data_type == 'all':
        candle_types = ['trades', 'mark_price', 'index_price']
    else:
        candle_types = [ct.strip() for ct in args.data_type.split(',')]

    print("=" * 80)
    print("EXTENDED EXCHANGE DATA CRAWLER")
    print("=" * 80)
    print(f"Markets: {', '.join(markets)}")
    print(f"Candle types: {', '.join(candle_types)}")
    print(f"Interval: {args.interval}")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Batch size: {args.batch_size}")
    print(f"Execution: Sequential for symbols, parallel for data types")
    print("=" * 80)

    crawler = ExtendedCrawler(
        api_key=args.api_key,
        data_dir=args.data_dir,
        network=args.network,
        rate_limit_delay=args.rate_limit_delay,
        max_retries=args.max_retries
    )

    # Process each market sequentially, but candle types in parallel
    for market_idx, market in enumerate(markets, 1):
        print("\n" + "#" * 80)
        print(f"# [{market_idx}/{len(markets)}] Processing market: {market}")
        print("#" * 80)

        # Process candle types in parallel for this market
        with ThreadPoolExecutor(max_workers=len(candle_types)) as executor:
            # Submit all candle type tasks
            futures = {
                executor.submit(
                    process_single_candle_type,
                    crawler,
                    market,
                    candle_type,
                    args.interval,
                    args.start_date,
                    args.end_date,
                    args.batch_size
                ): candle_type
                for candle_type in candle_types
            }

            # Process results as they complete
            for future in as_completed(futures):
                candle_type, success, message = future.result()
                print("\n" + "=" * 80)
                print(f"[{market}][{candle_type}] {message}")
                print("=" * 80)

    print("\n" + "=" * 80)
    print("COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
