#!/usr/bin/python3
DESCRIPTION = """Plot performance data generated from nvblox's timing framework.

The script creates plots by scraping text logfiles generated from the nvblox::timing framework.

Example usage:
 * Plot mean timings comparison from two logfiles:
       nvblox_plot_timing_data.py --input-logfiles log1.txt log2.txt --mode timings --label mean_time_s

 * Plot total time for "depth" and "color" with a max namespace level of 1:
       nvblox_plot_timing_data.py --input-logfiles log1.txt --mode timings --label total_time_s --regexp ".*depth|.*color.*" --max-namespace-level 1

"""

import argparse
from argparse import RawTextHelpFormatter
import re
import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure
import os

# For each mode: Tokens in the input text to look for when searching for a data block
START_TOKENS = {
    "timings":
    "namespace/tag - NumSamples - TotalTime - (Mean +- StdDev) - [Min,Max]",
    "rates": "namespace/tag - NumSamples (Window Length) - Mean",
    "delays":
    "namespace/tag - NumSamples (Window Length) - Mean Delay (seconds)"
}

# After a start token, this delimiter is expected to appear before and after a data block
DELIMITER = re.escape("-----------")

# Mapping between labels and column indics
COLUMN_INDICES = {
    "timings": {
        "num_samples_s": 1,
        "total_time_s": 2,
        "mean_time_s": 3
    },
    "rates": {
        "num_samples": 1,
        "frequency_hz": 2
    },
    "delays": {
        "num_samples": 1,
        "delay_s": 2
    }
}


def filter_data(raw_data, include_regexp, max_namespace_level) -> str:
    """Remove num-numeric characters and optionally apply an inclusion regexp."""
    filtered_data = raw_data.replace(",", " ")
    filtered_data = re.sub('[\+\-()\[\]]', '', filtered_data)
    if include_regexp:
        regex = re.compile(include_regexp)
        filtered_data = "\n".join(
            [line for line in filtered_data.split("\n") if regex.search(line)])

    if max_namespace_level:
        filtered_data = "\n".join([
            line for line in filtered_data.split("\n")
            if line.count("/") <= max_namespace_level
        ])

    return filtered_data


def extract_data(logfile, start_token, include_regexp,
                 max_namespace_level) -> pd.DataFrame:
    """Find the last occurance of a datablock in a logfile. Return it as a pandas dataframe."""
    with open(logfile, 'r') as file:
        data = file.read()

    # Remove any ROS-generated prefixes for each line
    data = re.sub(r'\[.*\d\] ', '', data)

    # Remove trailing whitespace
    data = data.replace(" \n", "\n")

    # Extract data between start and end tokens
    start_token = re.escape(start_token)
    regexp = rf"(?s){start_token}\n{DELIMITER}\n(.*?)\n{DELIMITER}"
    pattern = re.compile(regexp, re.DOTALL)
    matches = pattern.findall(data)

    if not matches:
        print("No data found between the specified tokens.")
        return

    # We pick the last one if the block appear multiple times
    filtered_data = filter_data(matches[-1], include_regexp,
                                max_namespace_level)
    print("Data block from log:")
    print(filtered_data)

    # Create a dataframe out of the  filtered data
    rows = [
        row.strip() for row in filtered_data.strip().split('\n')
        if row.strip()
    ]
    df = pd.DataFrame([row.split() for row in rows])
    df.set_index(0, inplace=True)

    return df


def create_plot(data_frames, column, label, data_names) -> Figure:
    """Return a plot figure for a given column in all data frames."""
    data_dict = {}
    for name, df in zip(data_names, data_frames):
        try:
            data_dict[name] = pd.to_numeric(df[column])
        except KeyError:
            print(df)
            print(
                f"ERROR: Invalid column: {column}. Select one from the table above"
            )
            exit(1)

    fig = px.bar(data_dict,
                 x=data_names,
                 barmode="group",
                 labels=data_names,
                 log_x=False)
    fig.update_layout(xaxis_title=label,
                      yaxis_title="",
                      title="nvblox performance measurement")
    fig.update_xaxes(tickformat=",.3s")  # SI prefix with 3 significant digits

    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=DESCRIPTION,
                                     formatter_class=RawTextHelpFormatter)

    parser.add_argument('--input-logfiles',
                        "-i",
                        type=str,
                        nargs="+",
                        help='Path to the logfiles',
                        required=True)
    parser.add_argument('--mode',
                        "-m",
                        type=str,
                        choices=["timings", "rates", "delays"],
                        required=True)
    parser.add_argument('--label',
                        "-l",
                        type=str,
                        help='What to plot',
                        required=True)
    parser.add_argument(
        '--regexp',
        "-r",
        type=str,
        help='Regexp to include, for example \"depth.*|occupancy.*\"',
        default=None)
    parser.add_argument('--output-html',
                        "-o",
                        type=str,
                        help='Path to output html file',
                        default=None)
    parser.add_argument('--output-json',
                        "-j",
                        type=str,
                        help='Path to output json file',
                        default=None)
    parser.add_argument(
        '--max-namespace-level',
        '-n',
        type=int,
        help='Max number of namespace, as in namespace1/namespace2/...',
        default=None)

    return parser.parse_args()


def main(args):

    start_token = START_TOKENS[args.mode]
    try:
        column = COLUMN_INDICES[args.mode][args.label]
    except KeyError:
        print(f"ERROR: Invalid key for mode: {args.mode}")
        print(f"Use one of: [" +
              str(", ".join(COLUMN_INDICES[args.mode].keys())) + "]")
        exit(1)

    dfs = []
    for logfile in args.input_logfiles:
        df = extract_data(logfile, start_token, args.regexp,
                          args.max_namespace_level)
        if df is None:
            exit(1)
        dfs.append(df)

    fig = create_plot(dfs, column, args.label,
                      [os.path.basename(path) for path in args.input_logfiles])

    if args.output_html:
        fig.write_html(args.output_html)
        print(f"Wrote: {args.output_html}")

    if args.output_json:
        df.to_json(args.output_json, indent=4)
        print(f"Wrote: {args.output_json}")

    if not args.output_json and not args.output_html:
        fig.show()


if __name__ == '__main__':
    main(parse_args())
