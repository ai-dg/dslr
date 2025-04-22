import csv
import sys
import math
import json
import time
from histogram import ft_reverse_dict
from describe import ft_recovering_data_from_dataset, ft_is_list_numeric
from pair_plot import Data_pairs


class Data(object):
    data_csv: dict
    numeric_features : dict
    numeric_col_with_houses : dict
    binary_houses : list
    x : list
    y : list
    theta_values : list[float]
    x_i : float
    sig : list[float]
    list_of_subjects : list
    nb_features : int
    error : float
    to_export : dict


def ft_recover_numeric_values_and_houses(data):
    numeric_col = {}
    for key, values in data.items():
        if ft_is_list_numeric(values) or key == "Hogwarts House":
            numeric_col[key] = values

    # print(numeric_col.keys())
    
    return numeric_col

def ft_get_binary_houses(data, house_vs):
    binary = []

    for data in data["Hogwarts House"]:
        if data == house_vs:
            binary.append(1)
        else:
            binary.append(0)

    return binary


def ft_get_x_and_y_for_matrix(data1, data2):

    x = {}
    y = {}

    x = ft_reverse_dict(data1)
    y = data2

    return x, y

def ft_normalize_features(data_dict):
    normalized = {}
    for key, values in data_dict.items():
        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        if std == 0:
            normalized[key] = [0 for _ in values]
        else:
            normalized[key] = [(v - mean) / std for v in values]
    return normalized


def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def ft_calculate_log_loss(data: Data):
    alpha = 0.01
    m = len(data.y)

    gradients = [0.0] * len(data.theta_values)

    total_error = 0.0

    for i in range(m):
        x_i = [1.0]  # biais
        for key in data.list_of_subjects:
            x_i.append(data.x[i][key])
        
        error = data.sig[i] - data.y[i]  # h(xᵢ) - yᵢ
        total_error += abs(error)

        for j in range(len(data.theta_values)):
            gradients[j] += error * x_i[j]

    for j in range(len(data.theta_values)):
        gradients[j] = (alpha / m) * gradients[j]
        data.theta_values[j] -= gradients[j]  # mise à jour de theta
    
    data.error = total_error / m



def ft_forward_pass_and_sigmoid(data: Data):
    data.nb_features = len(data.numeric_features.keys())
    data.list_of_subjects = list(data.numeric_features.keys())
    # data.theta_values = [0.0] * (data.nb_features + 1)
    data.sig = []

    for i in range(len(data.x)):
        x_i = [1.0]  # biais
        for key in data.list_of_subjects:
            x_i.append(data.x[i][key])

        z = sum(theta * feature for theta, feature in zip(data.theta_values, x_i))
        s = sigmoid(z)
        data.sig.append(s)

        # print(f"Échantillon {i}: z = {z:.4f}, sigmoid = {s:.4f}")

    ft_calculate_log_loss(data)

    # print(data.theta_values)

def ft_bgd_training_loop(data, house):
    iterations = 1000

    data.theta_values = [0.0] * (len(data.numeric_features.keys()) + 1)


    for i in range(iterations):
        ft_forward_pass_and_sigmoid(data)

    # print(data.theta_values)
    # print(len(data.theta_values))
    print(f"📉 BGD Log loss for {house}: {data.error:.6f}")

def ft_sgd_training_loop(data: Data, house: str):
    alpha = 0.01
    epochs = 20  # nombre de passes sur l’ensemble des données

    data.theta_values = [0.0] * (len(data.numeric_features.keys()) + 1)
    m = len(data.y)
    subjects = list(data.numeric_features.keys())

    for epoch in range(epochs):
        for i in range(m):
            x_i = [1.0]  # biais
            for key in subjects:
                x_i.append(data.x[i][key])

            # calcul de zᵢ et prédiction
            z = sum(theta * xi for theta, xi in zip(data.theta_values, x_i))
            h = sigmoid(z)
            error = h - data.y[i]

            # mise à jour de chaque θⱼ
            for j in range(len(data.theta_values)):
                data.theta_values[j] -= alpha * error * x_i[j]

    # calcul final du log loss (optionnel ici)
    total_error = 0
    for i in range(m):
        x_i = [1.0] + [data.x[i][k] for k in subjects]
        z = sum(theta * xi for theta, xi in zip(data.theta_values, x_i))
        h = sigmoid(z)
        total_error += abs(h - data.y[i])
    data.error = total_error / m

    print(f"📉 SGD Log loss for {house}: {data.error:.6f}")

def ft_mini_batch_training_loop(data: Data, house: str):
    batch_size=32
    epochs=20
    alpha=0.01
    m = len(data.y)
    subjects = list(data.numeric_features.keys())
    data.theta_values = [0.0] * (len(subjects) + 1)

    for epoch in range(epochs):
        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)

            gradients = [0.0] * len(data.theta_values)

            for i in range(start, end):
                x_i = [1.0] + [data.x[i][key] for key in subjects]
                z = sum(theta * xi for theta, xi in zip(data.theta_values, x_i))
                h = sigmoid(z)
                error = h - data.y[i]

                for j in range(len(data.theta_values)):
                    gradients[j] += error * x_i[j]

            # Mise à jour des poids (moyenne des gradients)
            batch_len = end - start
            for j in range(len(data.theta_values)):
                gradients[j] = gradients[j] / batch_len
                data.theta_values[j] -= alpha * gradients[j]

    # Calcul final du log loss
    total_error = 0
    for i in range(m):
        x_i = [1.0] + [data.x[i][key] for key in subjects]
        z = sum(theta * xi for theta, xi in zip(data.theta_values, x_i))
        h = sigmoid(z)
        total_error += abs(h - data.y[i])
    data.error = total_error / m

    print(f"📉 Mini-batch GD Log loss for {house}: {data.error:.6f}")



def ft_saving_values_to_json(path, json_object):


    # Écriture dans le fichier JSON
    with open(path, "w") as file:
        json.dump(json_object, file, indent=4)
    

    print(f"✅ Modèle sauvegardé dans {path}")

def ft_training_one_vs_all_model(house, data_csv, training_method):
    data_house = {}
    
    data = Data()
    data.numeric_col_with_houses = ft_recover_numeric_values_and_houses(data_csv)
    del data.numeric_col_with_houses["Index"]
    # del data.numeric_col_with_houses["Transfiguration"]
    # del data.numeric_col_with_houses["Astronomy"]
    # del data.numeric_col_with_houses["Herbology"]
    # del data.numeric_col_with_houses["Muggle Studies"]
    # del data.numeric_col_with_houses["Defense Against the Dark Arts"]
    # del data.numeric_col_with_houses["Care of Magical Creatures"]
    # del data.numeric_col_with_houses["Flying"]

    data.binary_houses = ft_get_binary_houses(data.numeric_col_with_houses, house)
    data.numeric_features = data.numeric_col_with_houses.copy()

    del data.numeric_features["Hogwarts House"]
    data.numeric_features = ft_normalize_features(data.numeric_features)
    data.x, data.y = ft_get_x_and_y_for_matrix(data.numeric_features, data.binary_houses)
    
    data.list_of_subjects = list(data.numeric_features.keys())
    
    

    if training_method == "BGD":
        ft_bgd_training_loop(data, house)
    elif training_method == "SGD":
        ft_sgd_training_loop(data, house)
    elif training_method == "M-Batch":
        ft_mini_batch_training_loop(data, house)
    else:
        print("❌ Unknown training method.")
        exit(1)

    

    theta_dict = {
        "bias": data.theta_values[0]
    }

    for i, key in enumerate(data.list_of_subjects):
        theta_dict[key] = data.theta_values[i + 1]
    
    data_house = {
        "target" : house,
        "theta" : theta_dict,
    }

    return data_house

def main():

    try:
        path = sys.argv[1]
        if len(sys.argv) != 2:
            print("Specify the path of dataset.")
            exit(1)

    except (ValueError, IndexError):
        print("Incorrect path or filename.")
        exit(1)

    valid_methods = {"BGD", "SGD", "M-Batch"}
    training_method = ""
    

    while training_method not in valid_methods:
        training_method = input("Choose training method (BGD, SGD, M-Batch): ").strip()
        if training_method not in valid_methods:
            print("❌ Invalid method. Please choose from: BGD, SGD, M-Batch.")


    data = Data()    
    data.data_csv = ft_recovering_data_from_dataset(path)

    house_list = []
    for house in data.data_csv["Hogwarts House"]:
        if house not in house_list:
            house_list.append(house)
    
    data.to_export = {}

    start = time.time()


    for house in house_list:
        data.to_export[house] = ft_training_one_vs_all_model(house, data.data_csv, training_method)

    end = time.time()

    elapsed_time = end - start
    print(f"Training time: {elapsed_time:.2f} seconds")



    data.data_csv.clear()
    path_json = "./trained_model.json"
    ft_saving_values_to_json(path_json, data.to_export)


if __name__ == "__main__":
    main()
