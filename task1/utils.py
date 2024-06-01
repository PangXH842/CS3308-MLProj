import os
import datetime
import matplotlib.pyplot as plt

def log(logf, message):
    print(message)
    with open(logf, 'a') as f:
        f.write(message + "\n")

def init_log(filename, args):
    # Create log file
    log_path = "./logs/"
    os.makedirs(log_path, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")
    log_file = log_path + f"{timestamp}.log"
    with open(log_file, 'w') as f:
        pass
    log(log_file, filename)
    log(log_file, f"{args}")

    return log_file

def generate_line_graph(x_data, y1_data, y2_data, x_label, y_label, title, legend_labels, save_path):
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    
    plt.plot(x_data, y1_data, linestyle='-', label=legend_labels[0])
    plt.plot(x_data, y2_data, linestyle='-', label=legend_labels[1])
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()

    # plt.xticks(np.arange(1,11), x_data, rotation=0)
    
    plt.grid(True)  # Add grid lines
    
    plt.savefig(save_path)
