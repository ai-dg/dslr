import csv
import sys
import math
import json
from histogram import ft_reverse_dict
from describe import ft_recovering_data_from_dataset, ft_is_list_numeric
from pair_plot import Data_pairs
from logreg_train import ft_normalize_features

class Data(object):
    data_csv: dict
    data_csv_test : dict
    data_json : dict
    numeric_features : dict
    biais: float
    poids : list[float]
    predictions : list
    predictions_str : list


def ft_load_data_from_json(path):
    data = {}
    try:
        with open(path, "r") as file:
            data = json.load(file)
    except (FileExistsError, FileNotFoundError):
        print("Error coming from json.")
        exit(1)
        
    return data

def ft_recovering_numeric_features_from_csv_test(data : Data):
    
    data.numeric_features = {}

    for keys, values in data.data_csv_test.items():
        if ft_is_list_numeric(values):
            data.numeric_features[keys] = values


    del data.numeric_features["Index"]
    # del data.numeric_features["Transfiguration"]
    # del data.numeric_features["Astronomy"]
    # del data.numeric_features["Herbology"]
    # del data.numeric_features["Muggle Studies"]
    # del data.numeric_features["Defense Against the Dark Arts"]
    # del data.numeric_features["Care of Magical Creatures"]
    # del data.numeric_features["Flying"]

def sigmoid(z):
    # Sigmoid different pour eviter l'overflow
    z = max(min(z, 500), -500)
    return 1 / (1 + math.exp(-z))


def ft_predict_value_for_each_subject(x_i, theta_dict):
    z = 0
    for theta, x in zip(theta_dict.values(), x_i):
        z += theta * x

    result = sigmoid(z)
    return result


def ft_predict_all_houses(data : Data):
    reverse_data = ft_reverse_dict(data.numeric_features)
    subjects = list(data.numeric_features.keys())

    data.predictions_str = []

    for idx, student in enumerate(reverse_data):
        house_probs = {}

        for house, infos in data.data_json.items():
            theta = infos["theta"]

            # biais
            x_i = [1.0]  
            for subj in subjects:
                value = student[subj]
                x_i.append(value)

            proba = ft_predict_value_for_each_subject(x_i, theta)
            # print(proba)
            house_probs[house] = proba
            # print(house_probs)

            
        predicted_house = max(house_probs, key=house_probs.get)
        data.predictions_str.append((idx, predicted_house))
        # break


def ft_save_predictions_to_csv(data):
    path_csv = "./houses.csv"

    with open(path_csv, "w", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["Index", "Hogwarts House"])
        for idx, house in data.predictions_str:
            csv_writer.writerow([idx, house])

    print(f"✅ Results exported into {path_csv}")
        

def ft_calculate_percent_error(data : Data):
    correct = 0
    total = 0

    for idx, actual_house in enumerate(data.data_csv_test["Hogwarts House"]):
        predicted_house = data.predictions_str[idx][1]
        if predicted_house == actual_house:
            correct += 1
        total += 1

    errors = total - correct
    accuracy = (correct / total) * 100
    error_rate = 100 - accuracy

    print(f"❌ Errors: {errors} / {total}")
    print(f"📉 Errors percent: {error_rate:.2f}%")
    print(f"✅ Accuracy: {accuracy:.2f}%")



def main():
    try:
        path = sys.argv[1]
        if len(sys.argv) != 2:
            print("Specify the path of dataset.")
            exit(1)

    except (ValueError, IndexError):
        print("Incorrect path or filename.")
        exit(1)
    
    path_json = "./trained_model.json"

    data = Data()

    data.data_json = ft_load_data_from_json(path_json)
    data.data_csv_test = ft_recovering_data_from_dataset(path)

    ft_recovering_numeric_features_from_csv_test(data)

    data.numeric_features = ft_normalize_features(data.numeric_features)

    ft_predict_all_houses(data)


    ft_calculate_percent_error(data)


    ft_save_predictions_to_csv(data)






if __name__ == "__main__":
    main()


#     def ft_separate_theta_values(data_json : dict, data : Data):
#     data.poids = []

#     for keys, sub_dict in data_json.items():
#         if keys == "theta":
#             for key, val in sub_dict.items():
#                 if key == "bias":
#                     data.biais = float(val)
#                 else:
#                     data.poids.append(float(val))

#     # print(data.biais)
#     # print(data.poids)


# def ft_predictions(data: Data):
#     reverse_data = ft_reverse_dict(data.numeric_features)

#     predictions = []

#     for i, row in enumerate(reverse_data):
#         x_i = [1.0]
#         for key in data.numeric_features:
#             x_i.append(row[key])

#         z = sum(w * x for w, x in zip([data.biais] + data.poids, x_i))
#         s = sigmoid(z)

#         prediction = 1 if s >= 0.5 else 0
#         predictions.append(prediction)

#     return predictions



# def ft_calculate_predictions_from_theta(data):
#     data.predictions = ft_predictions(data)
#     data.predictions_str = []

#     for i, pred in enumerate(data.predictions):
#         if pred == 1:
#             house = data.data_json["target"]
#         else:
#             house = "Other"
#         data.predictions_str.append((i, house))  # Stocker index + nom de maison
#         # print(f"Étudiant {i} → {house}")


# def ft_save_predictions_to_csv(data):
#     path_csv = "./houses.csv"

#     with open(path_csv, "w", newline="") as file:
#         csv_writer = csv.writer(file, delimiter=",", quotechar="|")

#         csv_writer.writerow(["Index", "Hogwarts House"])

#         for index, house in data.predictions_str:
#             csv_writer.writerow([index, house])

#     print(f"✅ Résultats sauvegardés dans {path_csv}")