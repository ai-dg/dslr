import sys
import os
from describe import ft_recovering_data_from_dataset, ft_recover_numeric_values_from_columns, ft_is_float
from histogram import ft_reverse_dict
import matplotlib
import random
import shutil


matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt


class Data(object):
    data_csv: dict
    notes_by_students : dict

HOUSE_COLORS = {
    "Gryffindor": "red",
    "Ravenclaw": "blue",
    "Slytherin": "green",
    "Hufflepuff": "gold"
}


def ft_correlation(x, y):
    if len(x) != len(y):
        raise ValueError("Les listes doivent avoir la même taille.")

    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denominator_x = sum((x[i] - mean_x) ** 2 for i in range(n)) ** 0.5
    denominator_y = sum((y[i] - mean_y) ** 2 for i in range(n)) ** 0.5

    if denominator_x == 0 or denominator_y == 0:
        return 0  # pas de variation → pas de corrélation

    return numerator / (denominator_x * denominator_y)


def ft_generate_plots_between_two_subjects(data1, name1, data2, name2, output_dir, reverse_data):
    points_by_house = {}

    for row in reverse_data:
        house = row.get("Hogwarts House")
        x_val = row.get(name1)
        y_val = row.get(name2)

        try:
            x = float(x_val)
            y = float(y_val)
        except (ValueError, TypeError):
            continue

        if x == 0 or y == 0 or not house:
            continue

        points_by_house.setdefault(house, []).append((x, y))

    # Calcule corrélation sur tous les points
    all_x = [x for pts in points_by_house.values() for x, _ in pts]
    all_y = [y for pts in points_by_house.values() for _, y in pts]
    corr = ft_correlation(all_x, all_y)

    if corr > 0.7:
        level = "Strong"
    elif corr >= 0.3:
        level = "Moderate"
    else:
        level = "Weak"

    # Création du plot
    plt.figure(figsize=(10, 6))
    for house, pts in points_by_house.items():
        xs = [x for x, y in pts]
        ys = [y for x, y in pts]
        plt.scatter(xs, ys, alpha=0.5, label=house, color=HOUSE_COLORS.get(house, "black"), s=10)

    plt.text(
        0.05, 0.95,
        f"{level} correlation: {corr:.2f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.7, boxstyle="round")
    )

    plt.xlabel(name1)
    plt.ylabel(name2)
    plt.title(f"Comparison between {name1} and {name2}")
    plt.legend(
        loc="center left",        
        bbox_to_anchor=(1, 0.5),
        fontsize=10,
        frameon=True
    )

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./{output_dir}/{name1}_vs_{name2}.png", bbox_inches="tight")
    plt.close()
    print(f"Correlation between {name1} and {name2}: {corr:.4f}")



def ft_generate_plots_by_subjects(data, reverse_data):
    output_dir = "./plots_scatter"

    # Supprime tout le dossier (même s’il contient des fichiers)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Recrée le dossier vide
    os.makedirs(output_dir)

    subjects = list(data.keys())
    if "Index" in subjects:
        subjects.remove("Index")

    all_pairs = [(a, b) for i, a in enumerate(subjects) for b in subjects[i+1:]]
    selected_pairs = random.sample(all_pairs, 10)



    for subj1, subj2 in selected_pairs:
        ft_generate_plots_between_two_subjects(
            data[subj1], subj1,
            data[subj2], subj2,
            output_dir,
            reverse_data
        )

def main():

    try:
        path = sys.argv[1]
        if len(sys.argv) != 2:
            print("Specify the path of dataset.")
            exit(1)

    except (ValueError, IndexError):
        print("Incorrect path or filename.")
        exit(1)

    data = Data()
    data.data_csv = ft_recovering_data_from_dataset(path)
    data.notes_by_students = ft_recover_numeric_values_from_columns(data.data_csv)

    del data.notes_by_students["Index"]

    reverse_data = ft_reverse_dict(data.data_csv)
    ft_generate_plots_by_subjects(data.notes_by_students, reverse_data)

    



if __name__ == "__main__":
    main()