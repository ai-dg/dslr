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

        sum_std = 0
        for v in values:
            sum_std += (v - mean) ** 2

        std = math.sqrt(sum_std / len(values))

        normalized_values = []

        if std == 0:
            for _ in values:
                normalized_values.append(0)
        else:
            for v in values:
                normalized_values.append((v - mean) / std)

        normalized[key] = normalized_values

    return normalized


def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def ft_calculate_batch_gradient_descent(data: Data):
    # Learning rate
    alpha = 0.01
    m = len(data.y)

    gradients = [0.0] * len(data.theta_values)

    total_error = 0.0
    total_loss = 0
    epsilon = 1e-15

    for i in range(m):
        # biais
        x_i = [1.0]  
        for key in data.list_of_subjects:
            x_i.append(data.x[i][key])
        
        # h(xᵢ) - yᵢ
        error = data.sig[i] - data.y[i]  
        total_error += abs(error)

        prediction = data.sig[i]
        label = data.y[i]

        if prediction in (0, 1):
            prediction = max(min(prediction, epsilon), 1 - epsilon)
        
        log_loss = - (label * math.log(prediction) + (1 - label) * math.log(1 - prediction))
        total_loss += log_loss

        for j in range(len(data.theta_values)):
            gradients[j] += error * x_i[j]

    for j in range(len(data.theta_values)):
        gradients[j] = (alpha / m) * gradients[j]
        # mise à jour de theta
        data.theta_values[j] -= gradients[j]  
    
    data.error = total_loss / m





def ft_forward_pass_and_sigmoid(data: Data):
    data.nb_features = len(data.numeric_features.keys())
    data.list_of_subjects = list(data.numeric_features.keys())
    # data.theta_values = [0.0] * (data.nb_features + 1)
    data.sig = []

    for i in range(len(data.x)):
        # biais
        x_i = [1.0]  
        for key in data.list_of_subjects:
            x_i.append(data.x[i][key])


        z = 0
        for theta, feature in zip(data.theta_values, x_i):
            z += theta * feature

        # Logistic Regression - Prediction
        s = sigmoid(z)

        data.sig.append(s)

        # print(f"Échantillon {i}: z = {z:.4f}, sigmoid = {s:.4f}")

    ft_calculate_batch_gradient_descent(data)

    # print(data.theta_values)

def ft_bgd_training_loop(data, house):
    iterations = 1000

    # print(data.numeric_features.keys())

    data.theta_values = [0.0] * (len(data.numeric_features.keys()) + 1)


    for i in range(iterations):
        ft_forward_pass_and_sigmoid(data)

    # print(data.theta_values)
    # print(len(data.theta_values))
    print(f"📉 BGD average Log loss for {house}: {data.error:.6f}")

def ft_sgd_training_loop(data: Data, house: str):
    alpha = 0.01
    # Nombre de passes sur l’ensemble des donnees
    epochs = 20  

    data.theta_values = [0.0] * (len(data.numeric_features.keys()) + 1)
    m = len(data.y)
    subjects = list(data.numeric_features.keys())

    epsilon = 1e-15

    for epoch in range(epochs):
        total_loss = 0
        for i in range(m):
            x_i = [1.0]  # biais
            for key in subjects:
                x_i.append(data.x[i][key])

            z = sum(theta * xi for theta, xi in zip(data.theta_values, x_i))
            h = sigmoid(z)
            error = h - data.y[i]

            prediction = h
            label = data.y[i]

            if prediction in (0, 1):
                prediction = max(min(prediction, epsilon), 1 - epsilon)
            
            log_loss = - (label * math.log(prediction) + (1 - label) * math.log(1 - prediction))
            total_loss += log_loss

            for j in range(len(data.theta_values)):
                data.theta_values[j] -= alpha * error * x_i[j]


    data.error = total_loss / m

    print(f"📉 SGD average Log loss for {house}: {data.error:.6f}")

def ft_mini_batch_training_loop(data: Data, house: str):
    alpha=0.01
    
    batch_size=32
    epochs=20
    
    m = len(data.y)
    
    subjects = list(data.numeric_features.keys())
    data.theta_values = [0.0] * (len(subjects) + 1)
    
    epsilon = 1e-15

    for epoch in range(epochs):
        total_loss = 0
        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)

            gradients = [0.0] * len(data.theta_values)

            for i in range(start, end):
                x_i = [1.0] + [data.x[i][key] for key in subjects]
                z = sum(theta * xi for theta, xi in zip(data.theta_values, x_i))
                h = sigmoid(z)
                error = h - data.y[i]

                prediction = h
                label = data.y[i]

                if prediction in (0, 1):
                    prediction = max(min(prediction, epsilon), 1 - epsilon)
                
                log_loss = - (label * math.log(prediction) + (1 - label) * math.log(1 - prediction))
                total_loss += log_loss

                for j in range(len(data.theta_values)):
                    gradients[j] += error * x_i[j]

            batch_len = end - start
            for j in range(len(data.theta_values)):
                gradients[j] = gradients[j] / batch_len
                data.theta_values[j] -= alpha * gradients[j]


    data.error = total_loss / m

    print(f"📉 Mini-batch GD average Log loss for {house}: {data.error:.6f}")



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
