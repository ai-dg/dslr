from re import L
from statistics import correlation
import sys
import os
from describe import ft_recovering_data_from_dataset, ft_recover_numeric_values_from_columns, ft_is_float
from histogram import ft_reverse_dict
import matplotlib
import random
import shutil
from math import sqrt


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
        raise ValueError("The lists of correlations must have the same length.")

    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = 0
    sum_denom_x = 0
    sum_denom_y = 0

    for i in range(n):
        sum_numerator = (x[i] - mean_x) * (y[i] - mean_y)
        numerator += sum_numerator
        sum_denom_x += (x[i] - mean_x) ** 2
        sum_denom_y += (y[i] - mean_y) ** 2

    denominator_x = sqrt(sum_denom_x)
    denominator_y = sqrt(sum_denom_y)

    if denominator_x == 0 or denominator_y == 0:
        return 0

    correlation = numerator / (denominator_x * denominator_y)

    return correlation


def ft_generate_plots_between_two_subjects(name1, name2, output_dir, reverse_data):
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

        if house not in points_by_house:
            points_by_house[house] = []
        points_by_house[house].append((x, y))

    all_x = []
    all_y = []

    # On prends la liste des valeurs en x et en y pour calculer la correlation de deux sujets
    for points in points_by_house.values():
        for x, y in points:
            all_x.append(x)
            all_y.append(y)

    corr = ft_correlation(all_x, all_y)


    if corr > 0.7:
        level = "Strong"
    elif corr >= 0.3:
        level = "Moderate"
    else:
        level = "Weak"

    # Mise en place des nuages de points separes par maisons et leurs couleurs, s est pour le diametre, et alpha la transparence
    plt.figure(figsize=(10, 6))
    for house, points in points_by_house.items():
        x_points = []
        y_points = []
        for x, y in points:
            x_points.append(x)
            y_points.append(y)
        plt.scatter(x_points, y_points, alpha=0.5, label=house, color=HOUSE_COLORS.get(house, "black"), s=10)

    # Mise en place de la box texte sur le graphique de la correlation
    plt.text(
        # Coordonnees, x = 5% et y = 95%
        0.05, 0.95,
        # Message a afficher
        f"{level} correlation: {corr:.2f}",
        # Transforme les coordoonees en pourcentage
        transform=plt.gca().transAxes,
        # Taille du texte 
        fontsize=12,
        # Aligne le texte par le haut
        verticalalignment="top",
        # Ajoute un fonc blanc semi-transparent autour du texte et coin arrondi
        bbox=dict(facecolor="white", alpha=0.7, boxstyle="round")
    )

    plt.xlabel(name1)
    plt.ylabel(name2)
    plt.title(f"Comparison between {name1} and {name2}")
    
    # Mise en place de la legende des 4 maisons et leurs couleurs respectifs
    plt.legend(
        # Point d'ancrage, c'est a dire le centre gauche de la legende, pas le graphique
        loc="center left",
        # Coordonnees de l'ancrage, en pourcentage (x, y)        
        bbox_to_anchor=(1, 0.5),
        fontsize=10,
        # Ajoute une bordure (cadre)
        frameon=True
    )

    plt.grid(True)

    # Mise en echelle du graphique par rapport a la taille de l'image
    plt.tight_layout()
    plt.savefig(f"./{output_dir}/{name1}_vs_{name2}.png", bbox_inches="tight")
    plt.close()
    print(f"Correlation between {name1} and {name2}: {corr:.4f}")



def ft_generate_plots_by_subjects(data, reverse_data):
    output_dir = "./plots_scatter"

    # Supprime les dossiers, vue qu'il va creer de maniere random les pairs de features/sujets
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Recreer le dossier
    os.makedirs(output_dir)

    subjects = list(data.keys())
    if "Index" in subjects:
        subjects.remove("Index")

    # print(subjects[1:])
    all_pairs = []

    for i, a in enumerate(subjects):
            # print(i)
            # print(a)
            for b in subjects[i + 1:]:
                all_pairs.append((a,b))


    selected_pairs = random.sample(all_pairs, 10)



    for subj1, subj2 in selected_pairs:
        ft_generate_plots_between_two_subjects(
            subj1,
            subj2,
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