import os
import pdb
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt


def get_data(results_dir, save_dir, save=True):

    if os.path.exists(f"{save_dir}/all_results.pkl"):
        with open(f"{save_dir}/all_results.pkl", "rb") as f:
            results = pickle.load(f)

        layer_types = set()
        for intervention in results:
            for layer_number in results[intervention]:
                for layer_type in results[intervention][layer_number]:
                    layer_types.add(layer_type)

        return results, layer_types

    info = dict()
    results = dict()
    layer_types = set()

    for intervention in ["dropout", "rank-reduction"]:

        info[intervention] = []
        results[intervention] = dict()

        fnames = glob.glob(f"{results_dir}/{intervention}/*/accuracy*")
        print(f"Found {len(fnames)} files of type {intervention}")
        # Format
        # f"{home_dir}plots/{args.intervention}/{save_as}/accuracy-{rate}-{args.dtpts}-{args.lnum}.p"

        for fname in fnames:

            fname_words = fname.split("/")
            suffix = fname_words[-1][:-len(".p")].split("-")

            if len(suffix) != 4:
                print(f"Fname is not in right format. {fname}")
                continue

            layer_type = fname_words[-2]
            rate = float(suffix[1])
            dtpts = int(suffix[2])
            layer_number = suffix[3]

            if suffix[0] != "accuracy" or dtpts != 22000:
                continue

            if layer_number not in results[intervention]:
                results[intervention][layer_number] = dict()

            if layer_type not in results[intervention][layer_number]:
                results[intervention][layer_number][layer_type] = []

            layer_types.add(layer_type)

            print(
                f"Reading file with intervention type {layer_type}, intervention rate {rate}, layer number {layer_number}")
            with open(fname, "rb") as f:

                try:
                    accuracies = pickle.load(f)
                except:
                    print(f"Had trouble in opening {fname}")
                    continue

                info_dict = {
                    "layer_type": layer_type,
                    "rate": rate,
                    "dtps": dtpts,
                    "layer_number": layer_number,
                    "accuracy": accuracies
                }

                info[intervention].append(info_dict)
                results[intervention][layer_number][layer_type].append((rate, accuracies))

    if save:
        with open(f"{save_dir}/all_results.pkl", "wb") as f:
            pickle.dump(results, f)

        with open(f"{save_dir}/accuracy_results.pkl", "wb") as f:
            pickle.dump(info, f)

    return results, layer_types


save_dir = "../plots/intervention-plots"
results_dir = "../plots/intervention-plots"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

results, layer_types = get_data(results_dir=results_dir,
                                save_dir=save_dir)

ncols = 4
nrows = 2

layer_types = list(layer_types)
assert len(layer_types) <= nrows * ncols, \
    f"Make the grid larger than {nrows * ncols} as there are {len(layer_types)}"

for intervention in results:

    for layer_number in results[intervention]:

        plot_data = dict()
        for layer_type, rate_acc in results[intervention][layer_number].items():

            sorted_rate_acc = sorted(rate_acc, key=lambda x: x[0])

            x = []
            y = []
            std = []
            for rate, accuracies in sorted_rate_acc:

                rank_pct_to_keep = 100.0 - rate * 10
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)

                x.append(rank_pct_to_keep)
                y.append(mean_acc)
                std.append(std_acc)

            x = np.array(x)
            y = np.array(y)
            std = np.array(std)

            plot_data[layer_type] = (x, y, std)

        plt.clf()

        fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(15, 4))
        # add an artist, in this case a nice label in the middle...
        for row in range(nrows):
            for col in range(ncols):

                type_id = row * ncols + col

                if type_id >= len(layer_types):
                    continue
                layer_type = layer_types[type_id]

                if layer_type in plot_data:
                    x, y, std = plot_data[layer_type]
                    axs[row, col].set_title(layer_type)
                    axs[row, col].plot(x, y, color="blue", marker="o")

        if int(layer_number) == 28:
            fig.suptitle(f"Accuracy (y-axis) vs Rank percentage to keep (x-axis) for layer number all.")
        else:
            fig.suptitle(f"Accuracy (y-axis) vs Rank percentage to keep (x-axis) for layer number {layer_number}.")
        # plt.xlabel("Rank percentage to keep")
        # plt.ylabel("Accuracy")
        # plt.fill_between(x, y + std, y - std, color="blue", alpha=0.2)

        plt.tight_layout()

        if not os.path.exists(f"{save_dir}/result_plots/{intervention}"):
            os.makedirs(f"{save_dir}/result_plots/{intervention}")
        plt.savefig(
            f"{save_dir}/result_plots/{intervention}/plot_layer_{layer_number}.png")

        plt.close()
