import sys
import os
from describe import ft_recovering_data_from_dataset, ft_recover_numeric_values_from_columns, ft_is_float
import matplotlib

matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt


class Data(object):
    data_csv: dict
    notes_by_house : dict

def ft_reverse_dict(data):
    keys = list(data.keys())
    nbr_rows = len(data[keys[0]])
    rows = []

    for i in range(nbr_rows):
        row = {}
        for key in keys:
            row[key] = data[key][i]
        rows.append(row)
    
    return rows

def ft_check_number_of_keys(reverse_data):
    expected_keys = set(reverse_data[0].keys())
    print(f"Expected number of columns: {len(expected_keys)}")

    for i, row in enumerate(reverse_data):
        actual_keys = set(row.keys())
        if actual_keys != expected_keys:
            missing = expected_keys - actual_keys
            extra = actual_keys - expected_keys
            print(f"Line {i} incomplete, key: ({len(actual_keys)})")
            if missing:
                print(f"Missing keys : {missing}")
            if extra:
                print(f"Surplus keys  : {extra}")


def ft_take_notes_by_house_and_by_subjects(data, data_subjects):
    notes_by_house = {}
    house_list = []

    for house in data["Hogwarts House"]:
        if house not in notes_by_house:
            notes_by_house[house] = {}
            house_list.append(house)


    for col in data_subjects.keys():
        for house in house_list:
            notes_by_house[house][col] = []

    reverse_data = ft_reverse_dict(data)

    for row in reverse_data:
        house = row.get("Hogwarts House")
        if not house:
            continue
        for col in data_subjects.keys():
            val = row.get(col)
            if ft_is_float(val):
                notes_by_house[house][col].append(float(val))


    # print(data_subjects["Arithmancy"])
    return notes_by_house


def ft_put_subject_into_plot(data, subject, output_dir):
    griff = data["Gryffindor"][subject]
    griff = [x for x in griff if x != 0]
    raven = data["Ravenclaw"][subject]
    raven = [x for x in raven if x != 0]
    slyth = data["Slytherin"][subject]
    slyth = [x for x in slyth if x != 0]
    huffle = data["Hufflepuff"][subject]
    huffle = [x for x in huffle if x != 0]

    total_students = len(griff) + len(raven) + len(slyth) + len(huffle)

    plt.close()
    plt.clf()
    plt.figure(figsize=(10, 6))
    
    bins = 100 # Combien de tranches, divise les score
    alpha = 0.3 # Transparence

    plt.hist(griff, color="red", bins=bins, alpha=alpha, label="Gryffindor", edgecolor="black")
    plt.hist(raven, color="blue", bins=bins, alpha=alpha, label="Ravenclaw", edgecolor="black")
    plt.hist(slyth, color="green", bins=bins, alpha=alpha, label="Slytherin", edgecolor="black")
    plt.hist(huffle, color="gold", bins=bins, alpha=alpha, label="Hufflepuff", edgecolor="black")
    plt.title(f"Distribution of '{subject}' scores by house")
    plt.xlabel("Score")
    plt.ylabel(f"Number of students per score range\n({total_students} out of 1599 students with available scores)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./{output_dir}/{subject}.png")
    # plt.show()
    plt.close()
    plt.clf()


def ft_put_histogram_into_plot(data):
    output_dir = "./plots_histogram"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    subjects = data["Gryffindor"].keys()
    for subject in subjects:
        ft_put_subject_into_plot(data, subject, output_dir)


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
    data.numeric_col = ft_recover_numeric_values_from_columns(data.data_csv)

    del data.numeric_col["Index"]
    data.notes_by_house = ft_take_notes_by_house_and_by_subjects(data.data_csv, data.numeric_col)
    ft_put_histogram_into_plot(data.notes_by_house)


if __name__ == "__main__":
    main()