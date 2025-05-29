"""The script is used to aggregate the annotation from Label Studio output files"""
import argparse
import pandas as pd
from functools import reduce
from pathlib import Path
import os
import sys

sys.path.append(os.getcwd())
from src.utils.aggregations import eq_aggregation, max_aggregation,soft_aggregation

def main(args: argparse.ArgumentParser):
    """Produce the aggegataed annotation file.

    The logic of the scipt is to collect csv files with annotation from
    different people to one data batch.

    After the collecting is done, the aggregation is applyed. Three aggregation
    types are available:
    - equal - the final label is set iff all N annotators have the same annotation 
    - maximum - the classic majority voting strategy
    - soft - in the case when the multilabel is presented it brakes the labels
    into a list and choces labels that apear more then num_annotators // 2 + 1.

    Args:
        args (argparse.ArgumentParser): The script arguments
    """
    base = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    columns_to_preserve = ["data_id", "text", "annotation"]
    
    dfs_merged = []
    for iteration in base.iterdir():
        if iteration.stem.startswith("."):
            continue
        dfs = []
        if args.use_only != "":
            if any(y!=iteration.name for y in args.use_only.split(";")):
                continue
        for f in iteration.iterdir():
            if f.suffix == ".csv":
                df_ = pd.read_csv(f, index_col=0)
                # this is known issue
                df_.annotation = df_.annotation.apply(lambda x: x.lower()
                                                      .replace("клиничесике проявления", "клинические проявления")
                                                      .replace("проблемы в семье ", "проблемы в семье"))
                df_ = df_[columns_to_preserve]
                splitted_name = f.stem.split("_")
                df_["iter"] = int(int(splitted_name[3]))
                df_ = df_.rename({"annotation": f"annotation_{splitted_name[-1]}"}, axis=1)
                dfs.append(df_)
        dfs = [x.drop_duplicates("data_id") for x in dfs]
        df_merged = reduce(lambda left, right: pd.merge(left,right,on=columns_to_preserve[:-1] + ["iter"]), dfs)
        dfs_merged.append(df_merged)
    df_overall = pd.concat(dfs_merged, ignore_index=True)
    annot_cols = [x for x in df_overall.columns if "annotation" in x]
    overlap = len(annot_cols)
    df_overall["equal"] = df_overall.loc[:, annot_cols].agg(eq_aggregation,
                             num_overlaps=overlap,
                             axis='columns')
    
    df_overall["maximum"] = df_overall.loc[:,annot_cols].agg(max_aggregation,
                             num_overlaps=overlap,
                             axis='columns')
    
    df_overall["soft"] = df_overall.loc[:, annot_cols].agg(soft_aggregation,
                             num_overlaps=overlap,
                             axis='columns')
    df_matched = df_overall[~df_overall[args.agg_method_to_match].isna()]
    df_unmatched = df_overall[df_overall[args.agg_method_to_match].isna()]
    if args.postfix == "":
        fname_matched = f"aggregated_{args.agg_method_to_match}_matched.csv"
        fname_unmatched = f"aggregated_{args.agg_method_to_match}_unmatched.csv"
    else:
        fname_matched = f"aggregated_{args.agg_method_to_match}_matched_{args.postfix}.csv"
        fname_unmatched = f"aggregated_{args.agg_method_to_match}_unmatched_{args.postfix}.csv"
        
    df_matched.to_csv(out_dir/fname_matched, sep="|")
    df_unmatched.to_csv(out_dir/fname_unmatched, sep="|")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src_dir',
        type=str,
        required=True,
        help='The path to dir with overlap annotation'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        required=True,
        help='The path to dir where the results will be stored'
    )
    parser.add_argument(
        '--agg_method_to_match',
        choices=["equal", "maximum", "soft"],
        default="soft",
        help='The aggregation method that will be used to separate matched examples'
    )
    parser.add_argument(
        '--postfix',
        default="",
        help='Postfix for result file names'
    )
    parser.add_argument(
        '--use_only',
        default="",
        help='The iteration numbers to be used separated by semicolumn'
    )

    args = parser.parse_args()
    main(args)