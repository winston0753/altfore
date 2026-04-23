# AltFore: Stock Forecasting using Alternative Inputs

## Reddit RSS Mentions Dataset

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
