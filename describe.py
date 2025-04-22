import csv
import sys
import math
import os


class Data(object):
    data_csv: dict
    numeric_col: dict
    stats: dict


def ft_is_float(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def ft_is_list_numeric(lst):
    nbr_string = 0

    for value in lst:
        if not ft_is_float(value):
            nbr_string += 1

    threshold = (nbr_string / len(lst)) * 100

    if threshold >= 90:
        return False
    else:
        return True



def ft_special_cases_in_float(value):
    try:
        value_converted = float(value)
        if (
            value_converted == float("inf")
            or value_converted == float("-inf")
            or value_converted != value_converted
        ):
            return 0.0
        return value_converted
    except:
        return 0


def ft_smart_convert(lst):
    values_to_convert = []
    for val in lst:
        if ft_is_float(val):
            converted = ft_special_cases_in_float(val)
            values_to_convert.append(converted)
        else:
            print(f"Non-numeric value found: {val} replaced by 0")
            values_to_convert.append(0)
    return values_to_convert



def ft_recovering_data_from_dataset(path):
    data = {}
    if not os.path.exists(path):
        print(f"File {path} not found.")
        exit(1)
    if not os.access(path, 2):
        print(f"Cannot access to the file {path}.")
        exit(1)

    with open(path, "r") as file:
        reader = csv.DictReader(file, delimiter=",", quotechar="|")

        for header in reader.fieldnames:
            data[header] = []

        for row in reader:
            for key, value in row.items():
                data[key].append(value)

    for key, values in data.items():
        if ft_is_list_numeric(values):
            data[key] = ft_smart_convert(values)

    if not data:
        print("Dataset is empty or invalid.")
        exit(1)

    return data


def ft_recover_numeric_values_from_columns(data):
    numeric_col = {}
    for key, values in data.items():
        if ft_is_list_numeric(values):
            numeric_col[key] = values

    # print(numeric_col.keys())
    
    return numeric_col


def ft_get_values_for_percents(sorted_list, p):
    if not sorted_list:
        return 0

    n = len(sorted_list)

    k = (n - 1) * p
    f = int(k)
    c = min(f + 1, n - 1)

    if f == c:
        return sorted_list[f]
    else:
        d = k - f
        return sorted_list[f] + (sorted_list[c] - sorted_list[f]) * d


def ft_recover_stats_from_numeric_cols(data):

    stats = {}
    for keys in data.keys():
        stats[keys] = {
            "Count": 0,
            "Mean": 0,
            "Std": 0,
            "Min": 0,
            "25%": 0,
            "50%": 0,
            "75%": 0,
            "Max": 0,
            "Range": 0,
            "Variance": 0,
            "CV (%)": 0,
            "IQR": 0,
            "Low IQR": 0,
            "High IQR": 0,
            "Skewness": 0,
            "Kurtosis": 0,
        }

    # Count, mean, std and min values
    for key, values in data.items():
        n = len(values)
        if n < 2:
            print(f"Not enough data for feature: {key}")
            continue

        stats[key]["Count"] = len(values)

        sum_values = 0
        sum_squared_diff = 0

        for value in values:
            sum_values += value

        mean = sum_values / len(values)
        stats[key]["Mean"] = mean

        for value in values:
            sum_squared_diff += (value - mean) ** 2
        
        variance = sum_squared_diff / n
        
        std = math.sqrt(sum_squared_diff / (len(values) - 1))
        stats[key]["Std"] = std
        stats[key]["Variance"] = variance
        if mean != 0:
            stats[key]["CV (%)"] = (std / abs(mean)) * 100
        else:
            stats[key]["CV (%)"] = 0

        stats[key]["Min"] = min(values)
        stats[key]["Max"] = max(values)
        stats[key]["Range"] = stats[key]["Max"] - stats[key]["Min"]

        # 25%, 50%, 75%
        sorted_values = sorted(values)
        stats[key]["25%"] = ft_get_values_for_percents(sorted_values, 0.25)
        stats[key]["50%"] = ft_get_values_for_percents(sorted_values, 0.50)
        stats[key]["75%"] = ft_get_values_for_percents(sorted_values, 0.75)
        stats[key]["IQR"] = stats[key]["75%"] - stats[key]["25%"]
        stats[key]["Low IQR"] = stats[key]["25%"] - (1.5 * stats[key]["IQR"])
        stats[key]["High IQR"] = stats[key]["75%"] + (1.5 * stats[key]["IQR"])

        # Skewness (Tails direction and data distribution)
        skew_sum = 0.0
        for value in values:
            skew_sum += (value - mean) ** 3


        skew_numer = skew_sum / n
        if std != 0:
            skew_denom = std **3
        else:
            skew_denom = 1
        stats[key]["Skewness"] = skew_numer / skew_denom

        # Kurtosis (Flatenning and data distribution)
        sum_kurt = 0.0
        for value in values:
            sum_kurt += (value - mean) ** 4
            
        kurt_numer = sum_kurt / n

        if std != 0:
            kurt_denom = std ** 4
        else:
            kurt_denom = 1

        stats[key]["Kurtosis"] = (kurt_numer / kurt_denom) - 3  
        

    return stats

def cut_long_names(col, width):
    if len(col) <= width - 2:
        return col
    else:
        return col[:width - 4] + "..."


def ft_show_stats_in_terminal(data):
    content = ""

    col_width = 18

    del data["Index"]
    
    headers = list(data.keys())
    stat_labels = [
        "Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max",
        "Range", "Variance", "CV (%)", "IQR", "Low IQR", "High IQR", "Skewness", "Kurtosis"
    ]

    content += f"{'':{col_width}}"
    for col in headers:
        short_col = cut_long_names(col, col_width - 1)
        content += f"{short_col:>{col_width}}"

    content += "\n"

    for stat in stat_labels:
        content += f"{stat:<{col_width}}"
        for col in headers:
            val = data[col].get(stat, "")
            formatted = f"{val:>{col_width}.6f}"
            content += formatted
        content += "\n"

    try:
        with open("stats.txt", "w") as file:
            file.write(content)
    except (FileExistsError, FileNotFoundError):
        print(f"Error writing in {file}")
        exit(1)

    print(content)
    


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
    data.stats = ft_recover_stats_from_numeric_cols(data.numeric_col)

    ft_show_stats_in_terminal(data.stats)


if __name__ == "__main__":
    main()
