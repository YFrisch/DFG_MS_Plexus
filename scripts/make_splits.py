import pickle
from pathlib import Path

import click
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

# ROOT = Path('F:/DATA/dfg_plexus/')
ROOT = Path('/media/yannik/Intenso/DATA/dfg_plexus/')

def parse_seeds(seeds: str):
    return [int(seed) for seed in seeds.split(",")]

@click.command()
@click.option('--root', type=click.Path(exists=True), default=ROOT)
@click.option('--target', type=click.Path(exists=False), default=ROOT / "splits")
@click.option('--fts_file', type=str, default="radiomics_features___combined_SA.csv")
@click.option('--train_file', type=str, default="train_idx_SA.pkl")
@click.option('--test_file', type=str, default="test_idx_SA.pkl")
@click.option('--test_split_size', type=float, default=0.3)
@click.option('--random_seed', type=int, default=42)
@click.option('--cv_seeds', type=str, default="0,1,2,3,4")
def make_splits(
    root: Path,
    target: Path,
    fts_file: str,
    train_file: str,
    test_file: str,
    test_split_size: float,
    random_seed: int,
    cv_seeds: str,
):

    root, target, fts_file = Path(root), Path(target), Path(fts_file)

    target.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(root / fts_file, delimiter=";")

    label = df['label'].astype(int)

    idx = np.arange(len(label))

    # Fixed outer train/test split.
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_split_size,
        random_state=random_seed,
        stratify=np.asarray(label)
    )

    print(f"{test_idx=}")

    with open(target / train_file, 'wb') as f:
        pickle.dump(train_idx, f)

    with open(target / test_file, 'wb') as f:
        pickle.dump(test_idx, f)

    print(f"Saved outer train split: {target / train_file}")
    print(f"Saved outer test split:  {target / test_file}")
    print(f"{len(train_idx)=}, {len(test_idx)=}")

    train_labels = label[train_idx]

    seeds = parse_seeds(cv_seeds)
    for seed in seeds:

        skf = StratifiedKFold(
            n_splits=len(seeds),
            shuffle=True,
            random_state=seed,
        )

        for fold, (subtrain_pos, val_pos) in enumerate(
            skf.split(train_idx, train_labels)
        ):
            subtrain_idx = train_idx[subtrain_pos]
            val_idx = train_idx[val_pos]

            subtrain_file = target / f"train_idx_seed_{seed}_fold_{fold}.pkl"
            val_file = target / f"val_idx_seed_{seed}_fold_{fold}.pkl"

            with open(subtrain_file, "wb") as f:
                pickle.dump(subtrain_idx, f)

            with open(val_file, "wb") as f:
                pickle.dump(val_idx, f)

            print(
                f"Seed {seed}, fold {fold}: "
                f"train={len(subtrain_idx)}, val={len(val_idx)}"
            )

if __name__ == "__main__":
    make_splits()
