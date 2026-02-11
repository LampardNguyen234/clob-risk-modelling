"""
Binance Historical Data Crawler

This script crawls historical data from Binance:

Spot API:
- Spot Klines (Candlestick Data)

USDS-Margined Futures API:
- Klines (Candlestick Data)
- Mark Price Klines
- Index Price Klines
- Premium Index Klines
- Funding Rate History

Data is saved to CSV files in the data directory.
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import random


class BinanceCrawler:
    """Crawler for Binance Spot and USDS-Margined Futures historical data."""

    BASE_URL = "https://fapi.binance.com"
    SPOT_BASE_URL = "https://api.binance.com"

    # API endpoints
    ENDPOINTS = {
        'spot': '/api/v3/klines',
        'klines': '/fapi/v1/klines',
        'mark_price': '/fapi/v1/markPriceKlines',
        'index_price': '/fapi/v1/indexPriceKlines',
        'premium_index': '/fapi/v1/premiumIndexKlines',
        'funding_rate': '/fapi/v1/fundingRate'
    }

    # Column names for the DataFrame (12 columns total)
    COLUMNS = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ]

    def __init__(self, data_dir: str = "../data", rate_limit_delay: float = 0.1, max_retries: int = 5):
        """
        Initialize the crawler.

        Args:
            data_dir: Directory to save CSV files
            rate_limit_delay: Delay between API requests in seconds
            max_retries: Maximum number of retries for rate-limited requests
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.session = requests.Session()

    def _make_request(self, endpoint: str, params: dict, use_spot_api: bool = False) -> List:
        """
        Make API request with error handling and exponential backoff retry.

        Args:
            endpoint: API endpoint path
            params: Query parameters
            use_spot_api: If True, use spot API base URL; otherwise use futures API

        Returns:
            List of kline data
        """
        base_url = self.SPOT_BASE_URL if use_spot_api else self.BASE_URL
        url = f"{base_url}{endpoint}"

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                time.sleep(self.rate_limit_delay)
                return response.json()
            except requests.exceptions.HTTPError as e:
                # Handle rate limiting (429) with exponential backoff
                if response.status_code == 429:
                    if attempt < self.max_retries:
                        # Exponential backoff with jitter: 2^attempt * base_delay + random jitter
                        base_delay = 2 ** (attempt+1)
                        jitter = random.uniform(0, 1)
                        wait_time = base_delay + jitter

                        print(f"  Rate limit hit (429). Retry {attempt + 1}/{self.max_retries} after {wait_time:.2f}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"Error: Max retries ({self.max_retries}) exceeded for rate limiting")
                        print(f"URL: {url}")
                        print(f"Params: {params}")
                        return []
                else:
                    # Other HTTP errors - don't retry
                    print(f"HTTP Error: {e}")
                    print(f"URL: {url}")
                    print(f"Params: {params}")
                    return []
            except requests.exceptions.RequestException as e:
                # Connection errors, timeouts, etc. - retry with backoff
                if attempt < self.max_retries:
                    base_delay = 2 ** attempt
                    jitter = random.uniform(0, 1)
                    wait_time = base_delay + jitter

                    print(f"  Request error: {e}. Retry {attempt + 1}/{self.max_retries} after {wait_time:.2f}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Error: Max retries ({self.max_retries}) exceeded")
                    print(f"URL: {url}")
                    print(f"Params: {params}")
                    return []

        return []

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1500
    ) -> pd.DataFrame:
        """
        Fetch kline/candlestick data.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Kline interval (e.g., '1m', '5m', '1h', '1d')
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            limit: Number of records per request (max 1500)

        Returns:
            DataFrame with kline data
        """
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }

        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time

        data = self._make_request(self.ENDPOINTS['klines'], params)

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=self.COLUMNS)
        df = self._process_dataframe(df, keep_all_columns=True)
        return df

    def fetch_spot_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch spot kline/candlestick data.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Kline interval (e.g., '1s', '1m', '5m', '1h', '1d')
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            limit: Number of records per request (max 1000, default 1000)

        Returns:
            DataFrame with spot kline data (open_time, open, high, low, close, volume)
        """
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }

        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time

        data = self._make_request(self.ENDPOINTS['spot'], params, use_spot_api=True)

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=self.COLUMNS)

        # Convert timestamps to datetime
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

        # Convert price and volume columns to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        # Keep only essential columns for spot data
        spot_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time']
        df = df[spot_cols]

        return df

    def fetch_mark_price_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1500
    ) -> pd.DataFrame:
        """
        Fetch mark price kline data.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Kline interval (e.g., '1m', '5m', '1h', '1d')
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            limit: Number of records per request (max 1500)

        Returns:
            DataFrame with mark price kline data
        """
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }

        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time

        data = self._make_request(self.ENDPOINTS['mark_price'], params)

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=self.COLUMNS)
        df = self._process_dataframe(df)
        return df

    def fetch_index_price_klines(
        self,
        pair: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1500
    ) -> pd.DataFrame:
        """
        Fetch index price kline data.

        Args:
            pair: Trading pair (e.g., 'BTCUSDT')
            interval: Kline interval
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            limit: Number of records per request (max 1500)

        Returns:
            DataFrame with index price kline data
        """
        params = {
            'pair': pair,
            'interval': interval,
            'limit': limit
        }

        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time

        data = self._make_request(self.ENDPOINTS['index_price'], params)

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=self.COLUMNS)
        df = self._process_dataframe(df)
        return df

    def fetch_premium_index_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1500
    ) -> pd.DataFrame:
        """
        Fetch premium index kline data.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Kline interval
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            limit: Number of records per request (max 1500)

        Returns:
            DataFrame with premium index kline data
        """
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }

        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time

        data = self._make_request(self.ENDPOINTS['premium_index'], params)

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=self.COLUMNS)
        df = self._process_dataframe(df)
        return df

    def fetch_funding_rate_history(
        self,
        symbol: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch funding rate history.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            limit: Number of records per request (max 1000)

        Returns:
            DataFrame with funding rate history
        """
        params = {
            'symbol': symbol,
            'limit': limit
        }

        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time

        data = self._make_request(self.ENDPOINTS['funding_rate'], params)

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df = self._process_funding_rate_dataframe(df)
        return df

    def _process_funding_rate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and clean the funding rate DataFrame.

        Args:
            df: Raw DataFrame from API

        Returns:
            Processed DataFrame
        """
        if df.empty:
            return df

        # Convert timestamp to datetime
        df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')

        # Convert numeric columns to float
        df['fundingRate'] = df['fundingRate'].astype(float)
        df['markPrice'] = df['markPrice'].astype(float)

        return df

    def _process_dataframe(self, df: pd.DataFrame, keep_all_columns: bool = False) -> pd.DataFrame:
        """
        Process and clean the DataFrame.

        Args:
            df: Raw DataFrame from API
            keep_all_columns: If True, keep all columns; otherwise keep only essential columns

        Returns:
            Processed DataFrame
        """
        if df.empty:
            return df

        # Convert timestamps to datetime
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

        # Convert price columns to float
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = df[col].astype(float)

        # Convert volume columns to float if keeping all columns
        if keep_all_columns:
            for col in ['volume', 'quote_volume', 'taker_buy_base', 'taker_buy_quote']:
                if col in df.columns:
                    df[col] = df[col].astype(float)
            # Convert trades column to int
            if 'trades' in df.columns:
                df['trades'] = df['trades'].astype(int)

        else:
            # Keep only essential columns for other kline types
            essential_cols = ['open_time', 'open', 'high', 'low', 'close', 'close_time']
            df = df[essential_cols]

        return df

    def fetch_historical_data(
        self,
        data_type: str,
        symbol: str,
        interval: Optional[str],
        start_date: str,
        end_date: str,
        output_file: Optional[Path] = None,
        batch_size: int = 30000
    ) -> pd.DataFrame:
        """
        Fetch historical data for a date range with automatic pagination.
        Saves batches incrementally if output_file is provided.

        Args:
            data_type: Type of data ('klines', 'mark_price', 'index_price', 'premium_index', 'funding_rate')
            symbol: Trading pair symbol
            interval: Kline interval (not required for funding_rate)
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            output_file: Optional file path to save batches incrementally
            batch_size: Number of records to accumulate before saving to file

        Returns:
            Complete DataFrame for the date range
        """
        start_dt = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)

        start_time = int(start_dt.timestamp() * 1000)
        end_time = int(end_dt.timestamp() * 1000)

        # Check existing file for gaps
        time_col = 'fundingTime' if data_type == 'funding_rate' else 'open_time'
        existing_df = None
        fetch_ranges = [(start_time, end_time)]  # Default: fetch entire range

        if output_file and output_file.exists():
            try:
                existing_df = pd.read_csv(output_file)
                if not existing_df.empty and time_col in existing_df.columns:
                    existing_df[time_col] = pd.to_datetime(existing_df[time_col])
                    first_time = int(existing_df[time_col].iloc[0].timestamp() * 1000)
                    last_time = int(existing_df[time_col].iloc[-1].timestamp() * 1000)

                    print(f"Found existing file with {len(existing_df)} records:")
                    print(f"  First record: {existing_df[time_col].iloc[0]}")
                    print(f"  Last record: {existing_df[time_col].iloc[-1]}")

                    # Determine what ranges need to be fetched
                    fetch_ranges = []
                    if start_time < first_time:
                        fetch_ranges.append((start_time, first_time - 1))
                        print(f"  Gap detected before first record: {start_date} to {existing_df[time_col].iloc[0]}")
                    if last_time < end_time:
                        fetch_ranges.append((last_time + 1, end_time))
                        print(f"  Gap detected after last record: {existing_df[time_col].iloc[-1]} to {end_date}")

                    if not fetch_ranges:
                        print("  No gaps detected - data already complete for this range")
                        return existing_df
            except Exception as e:
                print(f"Could not read existing file: {e}")
                fetch_ranges = [(start_time, end_time)]

        all_data = []
        total_records = 0
        batch_count = 0
        current_batch = []

        print(f"Fetching {data_type} for {symbol} from {start_date} to {end_date}...")
        print(f"Batch size: {batch_size} records per save")

        # Fetch data for each range (handles gaps)
        for range_start, range_end in fetch_ranges:
            print(f"\nFetching range: {datetime.fromtimestamp(range_start/1000, tz=timezone.utc)} to {datetime.fromtimestamp(range_end/1000, tz=timezone.utc)}")
            current_start = range_start

            while current_start < range_end:
                if data_type == 'spot':
                    df = self.fetch_spot_klines(
                        symbol=symbol,
                        interval=interval,
                        start_time=current_start,
                        end_time=range_end,
                        limit=1000
                    )
                    time_col = 'close_time'
                    max_limit = 1000
                elif data_type == 'klines':
                    df = self.fetch_klines(
                        symbol=symbol,
                        interval=interval,
                        start_time=current_start,
                        end_time=range_end,
                        limit=1500
                    )
                    time_col = 'close_time'
                    max_limit = 1500
                elif data_type == 'mark_price':
                    df = self.fetch_mark_price_klines(
                        symbol=symbol,
                        interval=interval,
                        start_time=current_start,
                        end_time=range_end,
                        limit=1500
                    )
                    time_col = 'close_time'
                    max_limit = 1500
                elif data_type == 'index_price':
                    df = self.fetch_index_price_klines(
                        pair=symbol,
                        interval=interval,
                        start_time=current_start,
                        end_time=range_end,
                        limit=1500
                    )
                    time_col = 'close_time'
                    max_limit = 1500
                elif data_type == 'premium_index':
                    df = self.fetch_premium_index_klines(
                        symbol=symbol,
                        interval=interval,
                        start_time=current_start,
                        end_time=range_end,
                        limit=1500
                    )
                    time_col = 'close_time'
                    max_limit = 1500
                elif data_type == 'funding_rate':
                    df = self.fetch_funding_rate_history(
                        symbol=symbol,
                        start_time=current_start,
                        end_time=range_end,
                        limit=1000
                    )
                    time_col = 'fundingTime'
                    max_limit = 1000
                else:
                    raise ValueError(f"Invalid data_type: {data_type}")

                if df.empty:
                    break

                print(f"  Fetched {len(df)} {data_type} records. Last timestamp: {df[time_col].iloc[-1]}")

                all_data.append(df)
                current_batch.append(df)
                total_records += len(df)

                # Check if we should save accumulated batch to file
                current_batch_size = sum(len(b) for b in current_batch)
                if output_file and current_batch_size >= batch_size:
                    # Combine current batch and prepare for saving
                    batch_df = pd.concat(current_batch, ignore_index=True)

                    # Drop close_time before saving (keep for pagination logic)
                    # Exception: keep all columns for klines data only
                    if data_type not in ['funding_rate', 'klines'] and 'close_time' in batch_df.columns:
                        batch_df = batch_df.drop('close_time', axis=1)

                    # Save to file
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    mode = 'a' if output_file.exists() and batch_count > 0 else 'w'
                    header = not output_file.exists() or batch_count == 0
                    batch_df.to_csv(output_file, mode=mode, header=header, index=False)
                    print(f"  --> Saved batch {batch_count + 1}: {len(batch_df)} records to file (total: {total_records})")

                    current_batch = []
                    batch_count += 1

                # Update start time for next batch
                current_start = int(df[time_col].iloc[-1].timestamp() * 1000) + 1

                # Break if we got less than requested (end of available data)
                if len(df) < max_limit:
                    break

        # Save any remaining records in current_batch
        if output_file and current_batch:
            batch_df = pd.concat(current_batch, ignore_index=True)

            # Drop close_time before saving (except for klines where we keep all columns)
            if data_type not in ['funding_rate', 'klines'] and 'close_time' in batch_df.columns:
                batch_df = batch_df.drop('close_time', axis=1)

            output_file.parent.mkdir(parents=True, exist_ok=True)
            mode = 'a' if output_file.exists() and batch_count > 0 else 'w'
            header = not output_file.exists() or batch_count == 0
            batch_df.to_csv(output_file, mode=mode, header=header, index=False)
            print(f"  --> Saved final batch {batch_count + 1}: {len(batch_df)} records to file (total: {total_records})")

        # Merge new data with existing data if applicable
        if not all_data and existing_df is not None:
            print("No new data fetched - using existing data")
            return existing_df

        if not all_data:
            print("No data available")
            return pd.DataFrame()

        # Combine newly fetched data
        result = pd.concat(all_data, ignore_index=True)

        # Drop close_time for kline data (except for klines where we keep all columns)
        if data_type not in ['funding_rate', 'klines'] and 'close_time' in result.columns:
            result = result.drop('close_time', axis=1)

        # Merge with existing data if we had gaps
        if existing_df is not None:
            # Ensure existing_df has same columns as result
            dedup_col = 'fundingTime' if data_type == 'funding_rate' else 'open_time'
            result = pd.concat([existing_df, result], ignore_index=True)
            result = result.drop_duplicates(subset=[dedup_col])
            result = result.sort_values(dedup_col).reset_index(drop=True)

            # Overwrite file with complete sorted data
            if output_file:
                output_file.parent.mkdir(parents=True, exist_ok=True)
                result.to_csv(output_file, index=False)
                print(f"\nMerged with existing data and saved complete file: {len(result)} total records")
        else:
            # Just deduplicate and sort the new data
            dedup_col = 'fundingTime' if data_type == 'funding_rate' else 'open_time'
            result = result.drop_duplicates(subset=[dedup_col])
            result = result.sort_values(dedup_col).reset_index(drop=True)
            print(f"\nTotal records fetched: {len(result)}")

        return result

    def save_to_csv(self, df: pd.DataFrame, data_type: str, filename: str):
        """
        Save DataFrame to CSV file in data_type subdirectory.

        Args:
            df: DataFrame to save
            data_type: Type of data (spot, klines, mark_price, index_price, premium_index, funding_rate)
            filename: Output filename
        """
        # Map data type to subdirectory name
        subdir_name = 'spot_klines' if data_type == 'spot' else data_type
        subdir = self.data_dir / subdir_name
        subdir.mkdir(exist_ok=True)
        filepath = subdir / filename
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")


def fetch_single_data_type(
    crawler: BinanceCrawler,
    data_type: str,
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
    batch_size: int
) -> tuple:
    """
    Fetch a single data type.

    Args:
        crawler: BinanceCrawler instance
        data_type: Type of data to fetch
        symbol: Trading pair symbol
        interval: Kline interval
        start_date: Start date
        end_date: End date
        batch_size: Batch size for saving

    Returns:
        Tuple of (data_type, success, message)
    """
    try:
        # Prepare output file path
        if data_type == 'funding_rate':
            filename = f"{symbol}.csv"
        else:
            filename = f"{symbol}_{interval}.csv"

        # Map data type to subdirectory name
        subdir_name = 'spot_klines' if data_type == 'spot' else data_type
        subdir = crawler.data_dir / subdir_name
        output_file = subdir / filename

        # Fetch data with incremental batch saving
        df = crawler.fetch_historical_data(
            data_type=data_type,
            symbol=symbol,
            interval=interval if data_type != 'funding_rate' else None,
            start_date=start_date,
            end_date=end_date,
            output_file=output_file,
            batch_size=batch_size
        )

        if not df.empty:
            return (data_type, True, f"Completed: {output_file}")
        else:
            return (data_type, False, f"No data fetched for {data_type}")
    except Exception as e:
        return (data_type, False, f"Error fetching {data_type}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Crawl Binance historical data')
    parser.add_argument('--data-type',
                       type=str,
                       default='all',
                       help='Type of data to fetch. Can be: all, spot, klines, mark_price, index_price, premium_index, funding_rate. '
                            'Multiple types can be comma-separated: spot,klines,mark_price')
    parser.add_argument('--symbols',
                       type=str,
                       default='BTCUSDT',
                       help='Trading pair symbols (comma-separated). Example: BTCUSDT,ETHUSDT')
    parser.add_argument('--interval',
                       type=str,
                       default='1m',
                       help='Kline interval (e.g., 1m, 5m, 1h, 1d) - not used for funding_rate')
    parser.add_argument('--start-date',
                       type=str,
                       required=True,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date',
                       type=str,
                       required=True,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--data-dir',
                       type=str,
                       default='./data',
                       help='Directory to save CSV files')
    parser.add_argument('--batch-size',
                       type=int,
                       default=30000,
                       help='Number of records to accumulate before saving to file')
    parser.add_argument('--rate-limit-delay',
                       type=float,
                       default=0.1,
                       help='Delay between API requests in seconds (default: 0.1)')
    parser.add_argument('--max-retries',
                       type=int,
                       default=5,
                       help='Maximum number of retries for rate-limited requests (default: 5)')

    args = parser.parse_args()

    # Parse symbols (comma-separated)
    symbols = [s.strip() for s in args.symbols.split(',')]
    print(f"Symbols: {', '.join(symbols)}")

    crawler = BinanceCrawler(
        data_dir=args.data_dir,
        rate_limit_delay=args.rate_limit_delay,
        max_retries=args.max_retries
    )

    # Parse data types
    all_data_types = ['spot', 'klines', 'mark_price', 'index_price', 'premium_index', 'funding_rate']

    if args.data_type == 'all':
        data_types = all_data_types
    else:
        # Parse comma-separated data types
        requested_types = [dt.strip() for dt in args.data_type.split(',')]

        # Validate data types
        invalid_types = [dt for dt in requested_types if dt not in all_data_types]
        if invalid_types:
            print(f"Error: Invalid data types: {', '.join(invalid_types)}")
            print(f"Valid options: {', '.join(all_data_types)}")
            return

        data_types = requested_types

    print(f"Fetching data types: {', '.join(data_types)}")
    print(f"Execution: Sequential for symbols, parallel for data types")
    print("=" * 80)

    # Process each symbol sequentially
    for symbol_idx, symbol in enumerate(symbols, 1):
        print(f"\n{'#' * 80}")
        print(f"# [{symbol_idx}/{len(symbols)}] Processing symbol: {symbol}")
        print(f"{'#' * 80}\n")

        # Fetch data types in parallel for this symbol
        with ThreadPoolExecutor(max_workers=len(data_types)) as executor:
            # Submit all tasks
            futures = {
                executor.submit(
                    fetch_single_data_type,
                    crawler,
                    data_type,
                    symbol,
                    args.interval,
                    args.start_date,
                    args.end_date,
                    args.batch_size
                ): data_type
                for data_type in data_types
            }

            # Process results as they complete
            for future in as_completed(futures):
                data_type, success, message = future.result()
                print("\n" + "=" * 80)
                print(f"[{symbol}][{data_type}] {message}")
                print("=" * 80)

    print(f"\n{'#' * 80}")
    print(f"# All symbols completed!")
    print(f"{'#' * 80}")


if __name__ == "__main__":
    main()
