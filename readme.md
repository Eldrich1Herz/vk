# VK S-Commerce Content Potential Analysis

This project analyzes the content potential of products in VK's social commerce ecosystem. Using four CSV datasets from the test launch, the script calculates engagement, conversion, revenue, and author reward metrics to rank product categories and formats.

## Project Structure
```
project/
  ├── data/
  │    ├── df_offers.csv
  │    ├── df_stats.csv
  │    ├── df_orders.csv
  │    └── df_placements.csv
  ├── main.py
  ├── requirements.txt
  └── README.md
```

## Input Data

### df_offers.csv
- hash_offer_id
- offer_created_at
- placement_format
- hash_seller_id
- hash_model_id
- category
- price

### df_stats.csv
- hash_placement_id
- views
- clicks

### df_orders.csv
- hash_order_id
- order_created_at
- hash_placement_id
- order_status_code
- order_status_code_changed_at
- GMV
- reward_author

### df_placements.csv
- hash_placement_id
- placement_created_at
- placement_format
- hash_author_id
- hash_offer_id
- is_published
- published_at

## 🧠 Purpose
Identify:
- Key categories with the highest content potential
- Formats (post/clip) with the best performance
- Profitability for both VK and authors

The script computes metrics and outputs ranked categories.

## Computed Metrics
- CTR (click-through rate)
- CVR (conversion to completed order)
- GMV per view
- Reward per view
- Engagement rate
- Cancel rate
- GMV per click
- Time to first order

All metrics are aggregated on **category × placement_format**.

## ⚖Weight Model (from strategy analysis)
The final score uses weights aligned with content-focused strategy:
- CTR — 0.35
- CVR — 0.25
- GMV per view — 0.25
- Reward per view — 0.15

## 🔄 Flexible Column Matching
If uploaded CSVs use nonstandard names (e.g., `impressions` instead of `views`), the script automatically detects them using alias mapping.

## ▶️ How to Run
```
pip install -r requirements.txt
python main.py
```

Optional custom weights:
```
python main.py --weights 0.4 0.3 0.2 0.1
```

## 📤 Output Files
Stored in the `results/` directory:
- **content_potential_by_category_format.csv** – final ranking
- **metrics_full.csv** – all metrics per placement

## 📝 Notes
- Missing values are automatically filled.
- All datetime columns are parsed when present.
- The analysis is fully reproducible.

## 📬 Support
If needed, additional files or modular separation (`src/` structure) can be added.

