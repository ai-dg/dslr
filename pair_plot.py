import sys
import os
from describe import ft_recovering_data_from_dataset, ft_recover_numeric_values_from_columns, ft_is_float
from histogram import ft_reverse_dict

import matplotlib
import shutil


matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt

class Data(object):
    data_csv: dict
    notes_by_students : dict
    list_of_pair_subjects : dict
    reverse_data: list




class Data_pairs(object):
    pair1: list
    name1: str
    pair2: list
    name2: str
    fig = None
    axes = None
    ax = None
    points_by_color: dict

HOUSE_COLORS = {
    "Gryffindor": "red",
    "Ravenclaw": "blue",
    "Slytherin": "green",
    "Hufflepuff": "gold",
}
   

def ft_putting_pairs_into_graph(data: Data_pairs):
    if data.name1 == data.name2:
        for house, points in data.points_by_color.items():
            ys = []
            for _, y in points:
                ys.append(y)
            data.ax.hist(ys, bins=20, alpha=0.5, label=house, color=HOUSE_COLORS.get(house, "black"))
        data.ax.set_title(data.name1, fontsize=7)
    else:
        for house, points in data.points_by_color.items():
            xs = []
            ys = []
            for x, y in points:
                xs.append(x)
                ys.append(y)
            data.ax.scatter(xs, ys, s=8, alpha=0.5, label=house, color=HOUSE_COLORS.get(house, "black"))

    data.ax.set_xticks([])
    data.ax.set_yticks([])




def ft_choising_values_for_plotting(notes_by_students, reverse_data):
    output_dir = "./plots_pair_plot"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    headers = list(notes_by_students.keys())
    print(headers)
    size = len(headers)

    fig, axes = plt.subplots(size, size, figsize=(18, 18))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i, name1 in enumerate(headers):
        for j, name2 in enumerate(headers):
            
            data = Data_pairs()
            data.name1 = name1
            data.name2 = name2
            data.fig = fig
            data.axes = axes
            data.ax = axes[i][j]
            data.points_by_color = {}
            
            

            for student in reverse_data:
                house = student.get("Hogwarts House")
                if not house or name1 not in student or name2 not in student:
                    continue
                try:
                    x = float(student[name2])
                    y = float(student[name1])
                except:
                    continue
                if x == 0 or y == 0:
                    continue
                data.points_by_color.setdefault(house, []).append((x, y))
            

            ft_putting_pairs_into_graph(data)


    plt.suptitle("Pair Plot of Numerical Subjects (colored by house)", fontsize=14)

    
    handles, labels = data.ax.get_legend_handles_labels()

    # print(handles)
    print(labels)
    fig.legend(
        handles, labels,
        loc='center left',
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8 
    )

    # rect=[left, bottom, right, top] en pourcentages
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{output_dir}/pair_plot_colored.png", bbox_inches="tight")




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
    data.reverse_data = ft_reverse_dict(data.data_csv)

    del data.notes_by_students["Index"]
    # print(data.list_of_pair_subjects)
    ft_choising_values_for_plotting(data.notes_by_students, data.reverse_data)
    # print("ok")


if __name__ == "__main__":
    main()