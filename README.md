# AltFore: Stock Forecasting using Alternative Inputs

## Dataset

`dataset/` contains generated Reddit ticker-mention data:

- `reddit_mentions.csv`: mention-level records (one row per post-ticker pair)
- `reddit_mentions_daily.csv`: daily aggregates by date, subreddit, and ticker

## Scripts

- `scripts/build_reddit_mentions.py`: pulls RSS posts from selected subreddits, extracts ticker mentions, and writes both dataset files

## Build Reddit Mentions

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the ingestion script from the project root:

```bash
python scripts/build_reddit_mentions.py
```

Output files are written to `dataset/`:
- `dataset/reddit_mentions.csv`
- `dataset/reddit_mentions_daily.csv`

## Build Model Dataset

Build the unified forecasting dataset (prices + Reddit features):

```bash
python scripts/build_model_dataset.py
```

Outputs:
- `dataset/prices_daily.csv`
- `dataset/model_dataset.csv`
