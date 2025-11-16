from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import sys

COLUMN_ALIASES = {
    'views': ['views','impressions','shown'],
    'clicks': ['clicks','click','tap','taps'],
    'GMV': ['GMV','amount','order_value','price_sum'],
    'reward_author': ['reward_author','reward','author_reward'],
    'hash_placement_id': ['hash_placement_id','placement_id','pid'],
    'hash_offer_id': ['hash_offer_id','offer_id','oid'],
    'category': ['category','cat','product_category'],
    'placement_format': ['placement_format','format','type'],
    'hash_author_id': ['hash_author_id','author_id'],
    'hash_seller_id': ['hash_seller_id','seller_id'],
}

def resolve_column(df, target):
    if target in df.columns:
        return target
    for alias in COLUMN_ALIASES.get(target, []):
        if alias in df.columns:
            return alias
    return None

DATA_DIR = Path('data')
RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS = {
    'n_ctr': 0.35,
    'n_cvr': 0.25,
    'n_gmv_per_view': 0.25,
    'n_reward_per_view': 0.15
}

FILES = {
    'offers': DATA_DIR / 'df_offers.csv',
    'stats': DATA_DIR / 'df_stats.csv',
    'orders': DATA_DIR / 'df_orders.csv',
    'placements': DATA_DIR / 'df_placements.csv'
}

def safe_read_csv(path):
    if not path.exists():
        print(f"ERROR: file not found: {path}")
        sys.exit(1)
    return pd.read_csv(path)


def minmax_series(s):
    if s.max() == s.min():
        return pd.Series(0.0, index=s.index)
    return (s - s.min()) / (s.max() - s.min())

def run_analysis(weights=WEIGHTS, files=FILES, results_dir=RESULTS_DIR):
    # 1. Load
    df_offers = safe_read_csv(files['offers'])
    df_stats = safe_read_csv(files['stats'])
    df_orders = safe_read_csv(files['orders'])
    df_placements = safe_read_csv(files['placements'])

    date_cols = {
        'df_offers': ['offer_created_at'],
        'df_orders': ['order_created_at', 'order_status_code_changed_at'],
        'df_placements': ['placement_created_at', 'published_at']
    }
    for col in date_cols['df_orders']:
        if col in df_orders.columns:
            df_orders[col] = pd.to_datetime(df_orders[col], errors='coerce')
    for col in date_cols['df_placements']:
        if col in df_placements.columns:
            df_placements[col] = pd.to_datetime(df_placements[col], errors='coerce')
    for col in date_cols['df_offers']:
        if col in df_offers.columns:
            df_offers[col] = pd.to_datetime(df_offers[col], errors='coerce')

    df = df_placements.merge(df_offers, on='hash_offer_id', how='left', suffixes=('_placement','_offer'))
    df = df.merge(df_stats, on='hash_placement_id', how='left')

    df_orders_all = df_orders.copy()

    for col in ['views', 'clicks', 'GMV', 'reward_author']:
        if col in df.columns:
            df[col] = df[col].fillna(0)
        else:
            df[col] = 0

    if 'order_status_code' in df_orders_all.columns:
        df_orders_all['is_created'] = (df_orders_all['order_status_code'] == 2).astype(int)
        df_orders_all['is_cancelled'] = (df_orders_all['order_status_code'] == 3).astype(int)
        df_orders_all['is_completed'] = (df_orders_all['order_status_code'] == 5).astype(int)
    else:
        df_orders_all['is_created'] = 0
        df_orders_all['is_cancelled'] = 0
        df_orders_all['is_completed'] = 0

    orders_agg = df_orders_all.groupby('hash_placement_id').agg(
        orders_total=('hash_order_id','count'),
        orders_completed=('is_completed','sum'),
        orders_created=('is_created','sum'),
        orders_cancelled=('is_cancelled','sum'),
        gmv_sum=('GMV','sum'),
        reward_sum=('reward_author','sum'),
        first_order_at=('order_created_at','min')
    ).reset_index()

    df = df.merge(orders_agg, on='hash_placement_id', how='left')

    fill_zero_cols = ['views','clicks','gmv_sum','reward_sum','orders_total','orders_completed','orders_created','orders_cancelled']
    for c in fill_zero_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    df['ctr'] = np.where(df['views'] > 0, df['clicks'] / df['views'], 0.0)

    df['cvr'] = np.where(df['clicks'] > 0, df['orders_completed'] / df['clicks'], 0.0)
    df['gmv_per_view'] = np.where(df['views'] > 0, df['gmv_sum'] / df['views'], 0.0)
    df['reward_per_view'] = np.where(df['views'] > 0, df['reward_sum'] / df['views'], 0.0)

    df['engagement_rate'] = np.where(df['views'] > 0, (df['clicks'] + df['orders_completed']) / df['views'], 0.0)
    df['cancel_rate'] = np.where(df['orders_created'] > 0, df['orders_cancelled'] / df['orders_created'], 0.0)
    df['gmv_per_click'] = np.where(df['clicks'] > 0, df['gmv_sum'] / df['clicks'], 0.0)

    if ('published_at' in df.columns) and ('first_order_at' in df.columns):
        df['time_to_first_order_hours'] = (df['first_order_at'] - df['published_at']).dt.total_seconds() / 3600.0
    else:
        df['time_to_first_order_hours'] = np.nan

    agg_cols = ['category', 'placement_format'] if 'placement_format' in df.columns else ['category']
    agg = df.groupby(agg_cols).agg(
        placements_count=('hash_placement_id','nunique'),
        views_total=('views','sum'),
        clicks_total=('clicks','sum'),
        orders_completed_total=('orders_completed','sum'),
        gmv_total=('gmv_sum','sum'),
        reward_total=('reward_sum','sum')
    ).reset_index()

    agg['ctr'] = np.where(agg['views_total']>0, agg['clicks_total'] / agg['views_total'], 0.0)
    agg['cvr'] = np.where(agg['clicks_total']>0, agg['orders_completed_total'] / agg['clicks_total'], 0.0)
    agg['gmv_per_view'] = np.where(agg['views_total']>0, agg['gmv_total'] / agg['views_total'], 0.0)
    agg['reward_per_view'] = np.where(agg['views_total']>0, agg['reward_total'] / agg['views_total'], 0.0)

    agg['n_ctr'] = minmax_series(agg['ctr'])
    agg['n_cvr'] = minmax_series(agg['cvr'])
    agg['n_gmv_per_view'] = minmax_series(agg['gmv_per_view'])
    agg['n_reward_per_view'] = minmax_series(agg['reward_per_view'])

    score_terms = []
    total_weight = 0.0
    for key, w in weights.items():
        total_weight += w
        if key not in agg.columns:
            print(f"WARNING: weight key {key} not in aggregated columns")
        score_terms.append(w * agg[key])

    if abs(total_weight - 1.0) > 1e-6:
        print(f"NOTE: weights sum to {total_weight:.4f} (not 1). Scores will be scaled accordingly.")

    agg['content_potential_score'] = sum(score_terms)

    out_top = agg.sort_values('content_potential_score', ascending=False)
    out_top.to_csv(results_dir / 'content_potential_by_category_format.csv', index=False)

    df_out_cols = [
        'hash_placement_id','hash_offer_id','category','placement_format','hash_author_id','hash_seller_id',
        'views','clicks','orders_total','orders_completed','gmv_sum','reward_sum',
        'ctr','cvr','gmv_per_view','reward_per_view','engagement_rate','cancel_rate','gmv_per_click','time_to_first_order_hours'
    ]

    df_out_cols = [c for c in df_out_cols if c in df.columns]
    df[df_out_cols].to_csv(results_dir / 'metrics_full.csv', index=False)

    print('Done. Results saved to:')
    print(' -', results_dir / 'content_potential_by_category_format.csv')
    print(' -', results_dir / 'metrics_full.csv')

    return agg, df

def parse_args():
    p = argparse.ArgumentParser(description='VK s-commerce content potential analysis')
    p.add_argument('--data-dir', default='data', help='Path to data directory')
    p.add_argument('--results-dir', default='results', help='Path to results directory')
    p.add_argument('--weights', nargs=4, type=float, metavar=('W_CTR','W_CVR','W_GMV','W_REWARD'),
                   help='Optional weights for [ctr,cvr,gmv_per_view,reward_per_view] that sum to 1')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    DATA_DIR = Path(args.data_dir)
    RESULTS_DIR = Path(args.results_dir)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    FILES = {
        'offers': DATA_DIR / 'df_offers.csv',
        'stats': DATA_DIR / 'df_stats.csv',
        'orders': DATA_DIR / 'df_orders.csv',
        'placements': DATA_DIR / 'df_placements.csv'
    }

    if args.weights:
        w = args.weights
        if abs(sum(w) - 1.0) > 1e-6:
            print('ERROR: weights must sum to 1. Using default weights from README.')
        else:
            WEIGHTS = {'n_ctr': w[0], 'n_cvr': w[1], 'n_gmv_per_view': w[2], 'n_reward_per_view': w[3]}

    run_analysis(weights=WEIGHTS, files=FILES, results_dir=RESULTS_DIR)
