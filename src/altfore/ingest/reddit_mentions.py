"""Build a Reddit ticker-mentions dataset from subreddit RSS feeds."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import feedparser
import pandas as pd


LOGGER = logging.getLogger(__name__)
RSS_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/atom+xml,application/rss+xml,application/xml;q=0.9,*/*;q=0.8",
}

RSS_FEEDS: tuple[str, ...] = (
    "https://www.reddit.com/r/wallstreetbets/.rss",
    "https://www.reddit.com/r/stocks/.rss",
    "https://www.reddit.com/r/investing/.rss",
)

# Match 1-5 uppercase letters with optional leading '$'
TICKER_PATTERN = re.compile(r"\$?[A-Z]{1,5}\b")

STOPLIST = {
    "A",
    "I",
    "AM",
    "DD",
    "CEO",
    "GDP",
    "CPI",
    "IMO",
    "TLDR",
    "YOLO",
    "USA",
    "ETF",
}


@dataclass(frozen=True)
class PostRecord:
    """Normalized RSS entry fields used for downstream parsing."""

    subreddit: str
    post_id: str
    post_title: str
    post_url: str
    published_at: str
    raw_text: str


def extract_subreddit_name(rss_url: str) -> str:
    """Extract subreddit name from a Reddit RSS URL."""
    path_parts = [part for part in urlparse(rss_url).path.split("/") if part]
    if "r" in path_parts:
        r_index = path_parts.index("r")
        if r_index + 1 < len(path_parts):
            return path_parts[r_index + 1]
    return "unknown"


def fetch_feed_entries(rss_url: str) -> list[feedparser.FeedParserDict]:
    """Fetch and parse a single RSS feed, returning entry objects."""
    request = Request(rss_url, headers=RSS_REQUEST_HEADERS)
    with urlopen(request, timeout=20) as response:
        feed_bytes = response.read()

    parsed_feed = feedparser.parse(feed_bytes)
    if getattr(parsed_feed, "bozo", False):
        bozo_exc = getattr(parsed_feed, "bozo_exception", None)
        LOGGER.warning("Feed parse warning for %s: %s", rss_url, bozo_exc)
    return list(getattr(parsed_feed, "entries", []))


def extract_raw_text(entry: feedparser.FeedParserDict) -> str:
    """Combine title and summary text into a single raw text field."""
    title = (getattr(entry, "title", "") or "").strip()
    summary = (getattr(entry, "summary", "") or "").strip()
    return f"{title}\n{summary}".strip()


def normalize_post_record(
    subreddit: str, entry: feedparser.FeedParserDict
) -> PostRecord | None:
    """Normalize feed entry fields into a PostRecord; skip unusable records."""
    post_id = (getattr(entry, "id", "") or getattr(entry, "link", "") or "").strip()
    if not post_id:
        return None

    post_title = (getattr(entry, "title", "") or "").strip()
    post_url = (getattr(entry, "link", "") or "").strip()
    published_at = (getattr(entry, "published", "") or "").strip()
    raw_text = extract_raw_text(entry)

    return PostRecord(
        subreddit=subreddit,
        post_id=post_id,
        post_title=post_title,
        post_url=post_url,
        published_at=published_at,
        raw_text=raw_text,
    )


def extract_ticker_mentions(raw_text: str) -> dict[str, int]:
    """Extract ticker mentions from text and count mentions per ticker."""
    matches = TICKER_PATTERN.findall(raw_text or "")
    mention_counts: dict[str, int] = {}

    for token in matches:
        ticker = token.lstrip("$").upper()
        if not ticker:
            continue
        if ticker in STOPLIST:
            continue
        mention_counts[ticker] = mention_counts.get(ticker, 0) + 1

    return mention_counts


def build_mentions_rows(posts: Iterable[PostRecord]) -> list[dict[str, object]]:
    """Create mention-level rows (one row per ticker mention group per post)."""
    rows: list[dict[str, object]] = []

    for post in posts:
        ticker_counts = extract_ticker_mentions(post.raw_text)
        for ticker, mention_count in ticker_counts.items():
            rows.append(
                {
                    "source": "reddit_rss",
                    "subreddit": post.subreddit,
                    "post_id": post.post_id,
                    "post_title": post.post_title,
                    "post_url": post.post_url,
                    "published_at": post.published_at,
                    "ticker": ticker,
                    "mention_count_in_post": int(mention_count),
                    "raw_text": post.raw_text,
                }
            )

    return rows


def build_mentions_dataframe(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build and clean mention-level DataFrame."""
    columns = [
        "source",
        "subreddit",
        "post_id",
        "post_title",
        "post_url",
        "published_at",
        "ticker",
        "mention_count_in_post",
        "raw_text",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows, columns=columns)
    df = df.drop_duplicates(subset=["post_id", "ticker"], keep="first").copy()

    dt_series = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    sort_key = dt_series.fillna(pd.Timestamp.min.tz_localize("UTC"))
    df = df.assign(_published_sort=sort_key).sort_values(
        by="_published_sort", ascending=False
    )
    df = df.drop(columns=["_published_sort"]).reset_index(drop=True)

    return df


def build_daily_dataframe(mentions_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mentions into a daily dataset by subreddit and ticker."""
    columns = ["date", "subreddit", "ticker", "post_count", "total_mentions"]
    if mentions_df.empty:
        return pd.DataFrame(columns=columns)

    daily_df = mentions_df.copy()
    parsed_dates = pd.to_datetime(daily_df["published_at"], errors="coerce", utc=True)
    daily_df["date"] = parsed_dates.dt.strftime("%Y-%m-%d")
    daily_df["date"] = daily_df["date"].fillna("unknown")

    aggregated = (
        daily_df.groupby(["date", "subreddit", "ticker"], as_index=False)
        .agg(
            post_count=("post_id", "nunique"),
            total_mentions=("mention_count_in_post", "sum"),
        )
        .sort_values(
            by=["date", "subreddit", "ticker"], ascending=[False, True, True]
        )
        .reset_index(drop=True)
    )
    return aggregated[columns]


def save_datasets(
    mentions_df: pd.DataFrame, daily_df: pd.DataFrame, dataset_dir: Path
) -> tuple[Path, Path]:
    """Persist mention-level and daily aggregate CSV files."""
    dataset_dir.mkdir(parents=True, exist_ok=True)

    mentions_path = dataset_dir / "reddit_mentions.csv"
    daily_path = dataset_dir / "reddit_mentions_daily.csv"

    mentions_df.to_csv(mentions_path, index=False)
    daily_df.to_csv(daily_path, index=False)

    return mentions_path, daily_path


def run_build_reddit_mentions(project_root: Path) -> None:
    """Execute end-to-end ingestion and dataset creation."""
    all_posts: list[PostRecord] = []

    for rss_url in RSS_FEEDS:
        subreddit = extract_subreddit_name(rss_url)
        LOGGER.info("Processing feed: %s", rss_url)
        try:
            entries = fetch_feed_entries(rss_url)
        except Exception as exc:  # defensive: feedparser/network edge cases
            LOGGER.error("Failed to fetch feed %s: %s", rss_url, exc)
            continue

        LOGGER.info("Fetched %d posts from r/%s", len(entries), subreddit)
        for entry in entries:
            post = normalize_post_record(subreddit=subreddit, entry=entry)
            if post is not None:
                all_posts.append(post)

    mention_rows = build_mentions_rows(all_posts)
    mentions_df = build_mentions_dataframe(mention_rows)
    daily_df = build_daily_dataframe(mentions_df)

    dataset_dir = project_root / "dataset"
    mentions_path, daily_path = save_datasets(mentions_df, daily_df, dataset_dir)

    LOGGER.info("Wrote %d mention rows to %s", len(mentions_df), mentions_path)
    LOGGER.info("Wrote %d daily aggregate rows to %s", len(daily_df), daily_path)
