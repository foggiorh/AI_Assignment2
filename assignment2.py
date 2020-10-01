import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
import numpy as np
import math
import random
import csv

# Define Max Iterarions, error Epsilons, and learning rate alpha
max_iterations = 5000
epsilon_a = .00001
epsilon_b = 100
epsilon_c = 400
epsilon_alt_b = 200
epsilon_alt_c = 700
alpha = 0.1  # 0.1

# Read Datasets
df_a = pd.read_csv("data_A.csv")
df_a.columns = ['Price', 'Weight', 'Type']
df_b = pd.read_csv("data_B.csv")
df_b.columns = ['Price', 'Weight', 'Type']
df_c = pd.read_csv("data_C.csv")
df_c.columns = ['Price', 'Weight', 'Type']
df_alt_a = pd.read_csv("data_ALT_A.csv")
df_alt_a.columns = ['Price', 'Weight', 'Type']
df_alt_b = pd.read_csv("data_ALT_B.csv")
df_alt_b.columns = ['Price', 'Weight', 'Type']
df_alt_c = pd.read_csv("data_ALT_C.csv")
df_alt_c.columns = ['Price', 'Weight', 'Type']


def normalize_data(data):
    values = data.values
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled_values = min_max_scaler.fit_transform(values)
    scaled_data = pd.DataFrame(scaled_values)
    scaled_data.columns = ['Price', 'Weight', 'Type']
    # scaled_data['Type'] = scaled_data['Type'].map({0: 'Small', 1: 'Big'})
    return scaled_data


def get_input_array(data):
    data[['Bias']] = 1
    return data[['Price', 'Weight', 'Bias']].to_numpy()


def get_desired_out_array(data):
    return data['Type'].to_numpy()


def sign_activation(net):
    if(net >= 0):
        return 1
    else:
        return 0


def soft_activation(net, gain):
    return 1 / (1 + (math.exp(-1 * gain * net)))
    # return (math.tanh(net * gain) + 1) / 2


def print_data(iteration, entry, net, error, learn, weights, total_error):
    print("Iteration: %d\tEntry:%d\tNet: %5.2f\tError: %6.3f\tLearn: %6.3f\tWeights: %6.2f, %6.2f, %6.2f\tTotal Error: %6.5f"
          % (iteration, entry, net, error, learn, weights[0], weights[1], weights[2], total_error))


def print_total_error(iteration, total_error):
    print("Iteration: %d\t Total Error: %f" % (iteration, total_error))


def train_hard(input_arr, desired_out, alpha, epsilon, max_iterations):
    y_w = random.uniform(-.5, .5)
    x_w = random.uniform(-.5, .5)
    b = random.uniform(-.5, .5)
    #weights = [-.5, -.5, .5]
    weights = [y_w, x_w, b]
    original_weights = weights
    for i in range(max_iterations):
        total_error = 0
        for j in range(len(input_arr)):
            net = 0
            for k in range(len(weights)):
                net = net + input_arr[j][k]*weights[k]
            out = sign_activation(net)
            error = desired_out[j] - out
            total_error = total_error + math.pow(error, 2)
            learn = alpha * error
            # print_data(i, j, net, error, learn, weights, total_error)
            for z in range(len(weights)):
                weights[z] = weights[z] + learn * input_arr[j][z]
        # print_total_error(i, total_error)
        if(total_error < epsilon):
            break
    return weights, original_weights


def train_soft(input_arr, desired_out, alpha, epsilon, max_iterations, gain):
    y_w = random.uniform(-.5, .5)
    x_w = random.uniform(-.5, .5)
    b = random.uniform(-.5, .5)
    #weights = [-.5, -.5, .5]
    weights = [y_w, x_w, b]
    original_weights = weights
    for i in range(max_iterations):
        total_error = 0
        for j in range(len(input_arr)):
            net = 0
            for k in range(len(weights)):
                net = net + input_arr[j][k]*weights[k]
            out = soft_activation(net, gain)
            error = desired_out[j] - out
            total_error = total_error + math.pow(error, 2)
            learn = alpha * error
            # print_data(i, j, net, error, learn, weights, total_error)
            for z in range(len(weights)):
                weights[z] = weights[z] + (learn * input_arr[j][z])
        # print_total_error(i, total_error)
        if(total_error < epsilon):
            break

    return weights, original_weights


def get_percent_train_and_test(input, output, percentage):
    c = 0
    indicesUsed = {}
    percInput = []
    percOutput = []
    percTestIn = []
    percTestOut = []
    while(c < (len(input) * percentage)):
        while(True):
            randomIndex = random.randint(0, (len(input) - 1))
            if(randomIndex not in indicesUsed):
                indicesUsed[randomIndex] = 1
                percInput.append(input[randomIndex])
                percOutput.append(output[randomIndex])
                break
        c += 1
    for i in range(len(input)):
        if i not in indicesUsed:
            percTestIn.append(input[i])
            percTestOut.append(output[i])

    return (percInput, percOutput, percTestIn, percTestOut)


def normalize_and_train_hard(datase_name, data, alpha, epsilon, max_iterations, perc):
    scaled_data = normalize_data(data)
    input_arr = get_input_array(scaled_data)
    output_arr = get_desired_out_array(scaled_data)
    inputTrain, outputTrain, inputTest, outputTest = get_percent_train_and_test(
        input_arr, output_arr, perc)
    result_weights, original_weights = train_hard(
        inputTrain, outputTrain, alpha, epsilon, max_iterations)
    print("%s Final Weights: " % (datase_name) + str(result_weights))
    return result_weights, original_weights, inputTest, outputTest


def normalize_and_train_soft(dataset_name, data, alpha, epsilon, max_iterations, gain, perc):
    scaled_data = normalize_data(data)
    input_arr = get_input_array(scaled_data)
    output_arr = get_desired_out_array(scaled_data)
    inputTrain, outputTrain, inputTest, outputTest = get_percent_train_and_test(
        input_arr, output_arr, perc)
    result_weights, original_weights = train_soft(
        inputTrain, outputTrain, alpha, epsilon, max_iterations, gain)
    print("%s Final Weights: " % (dataset_name) + str(result_weights))
    return result_weights, original_weights, inputTest, outputTest, gain


def test_hard(weights, testIn, testOut):
    pred = []
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(testIn)):
        net = 0
        for j in range(len(testIn[i])):
            net = net + testIn[i][j] * weights[j]
        out = sign_activation(net)
        pred.append(out)
    for i in range(len(pred)):
        if(pred[i] == 0):
            if(testOut[i] == 0):
                tp = tp + 1
            if(testOut[i] == 1):
                fp = fp + 1
        if(pred[i] == 1):
            if(testOut[i] == 1):
                tn = tn + 1
            if(testOut[i] == 0):
                fn = fn + 1
    acc = (tp + tn) / (tp + tn + fn + fp) * 100
    tp_perc = (tp / (tp + fn)) * 100
    fp_perc = (fp / (fp + tn)) * 100
    tn_perc = (tn / (tn + fp)) * 100
    fn_perc = (fn / (fn + tp)) * 100
    error = ((fp + fn) / (tp + tn + fn + fp) * 100)
    print("accuracy: %f, tp_perc: %f, fp_perc: %f, tn_perc: %f, fn_perc: %f " %
          (acc, tp_perc, fp_perc, tn_perc, fn_perc))
    return pred, testOut


def test_soft(weights, testIn, testOut, gain):
    pred = []
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(testIn)):
        net = 0
        for j in range(len(testIn[i])):
            net = net + testIn[i][j] * weights[j]
        #out = soft_activation(net, gain)
        if (net > 0):
            pred.append(1)
        else:
            pred.append(0)
    for i in range(len(pred)):
        if(pred[i] == 0):
            if(testOut[i] == 0):
                tp = tp + 1
            if(testOut[i] == 1):
                fp = fp + 1
        if(pred[i] == 1):
            if(testOut[i] == 1):
                tn = tn + 1
            if(testOut[i] == 0):
                fn = fn + 1
    acc = (tp + tn) / (tp + tn + fn + fp) * 100
    tp_perc = (tp / (tp + fn)) * 100
    fp_perc = (fp / (fp + tn)) * 100
    tn_perc = (tn / (tn + fp)) * 100
    fn_perc = (fn / (fn + tp)) * 100
    error = ((fp + fn) / (tp + tn + fn + fp) * 100)
    print("accuracy: %f, tp_perc: %f, fp_perc: %f, tn_perc: %f, fn_perc: %f " %
          (acc, tp_perc, fp_perc, tn_perc, fn_perc))
    return pred, testOut


def graph_results(title, dataset, final_weights, original_weights, filename):
    data = normalize_data(dataset)
    sns.set(rc={'figure.figsize': (10, 8)})
    sns.set_style("whitegrid")
    sns.scatterplot(data=data, x='Weight',
                    y='Price', hue='Type', linewidth=0)
    yf_weight = final_weights[0]
    xf_weight = final_weights[1]
    f_bias = final_weights[2]

    fm = (-1 * xf_weight) / (yf_weight)
    fb = (-1 * f_bias) / yf_weight
    fx = np.linspace(-0.1, 1.1, 100)
    fy = (fm * fx) + fb
    plt.plot(fx, fy, label="Final Line", color='green')

    yo_weight = original_weights[0]
    xo_weight = original_weights[1]
    o_bias = original_weights[2]

    om = (-1 * xo_weight) / (yo_weight)
    ob = (-1 * o_bias) / yo_weight
    ox = np.linspace(-0.1, 1.1, 100)
    oy = (om * ox) + ob
    plt.plot(ox, oy, label="Initial Line", linestyle='dashed', color='red')

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title(title)
    plt.ylim(-.1, 1.1)
    plt.xlim(-.1, 1.1)
    plt.show()
    # plt.savefig(filename)
    # plt.close()


def confusion_matrix(title, predicted, actual, filename):
    data = {'Actual': actual,
            'Predicted': predicted}
    df = pd.DataFrame.from_dict(data)
    confusion_matrix = pd.crosstab(df['Actual'], df['Predicted'], rownames=[
        'Actual'], colnames=['Predicted'])
    sns.heatmap(confusion_matrix, annot=True, fmt='d')
    plt.title(title)
    plt.show()
    # plt.savefig(filename)
    # plt.close()


groupa_hard_75, original_weights, groupa_hard_75_in, groupa_hard_75_out = normalize_and_train_hard(
    "Group A", df_a, 0.3, epsilon_a, max_iterations, .75)
groupa_hard_25, original_weights, groupa_hard_25_in, groupa_hard_25_out = normalize_and_train_hard(
    "Group A", df_a, 0.3, epsilon_a, max_iterations, .25)
pred, testOut = test_hard(
    groupa_hard_75, groupa_hard_75_in, groupa_hard_75_out)
graph_results("Group A Hard Activation 75% Train, 25% Test, Alpha: 0.3",
              df_a, groupa_hard_75, original_weights, 'GroupA_Hard_75.png')
confusion_matrix(
    "Group A Hard Activation 75% Train, 25% Test, Alpha: 0.3", pred, testOut, 'GroupA_Hard_75_CM.png')
pred, testOut = test_hard(
    groupa_hard_25, groupa_hard_25_in, groupa_hard_25_out)
graph_results("Group A Hard Activation 25% Train, 75% Test, Alpha: 0.3",
              df_a, groupa_hard_25, original_weights, 'GroupA_Hard_25.png')
confusion_matrix(
    "Group A Hard Activation 25% Train, 75% Test, Alpha: 0.3", pred, testOut, 'GroupA_Hard_25_CM.png')

groupb_hard_75, original_weights, groupb_hard_75_in, groupb_hard_75_out = normalize_and_train_hard(
    "Group B", df_b, 0.3, epsilon_b, max_iterations, .75)
groupb_hard_25, original_weights, groupb_hard_25_in, groupb_hard_25_out = normalize_and_train_hard(
    "Group B", df_b, 0.3, epsilon_b, max_iterations, .25)
pred, testOut = test_hard(
    groupb_hard_75, groupb_hard_75_in, groupb_hard_75_out)
graph_results("Group B Hard Activation 75% Train, 25% Test, Alpha: 0.3",
              df_b, groupb_hard_75, original_weights, 'GroupB_Hard_75.png')
confusion_matrix(
    "Group B Hard Activation 75% Train, 25% Test, Alpha: 0.3", pred, testOut, 'GroupB_Hard_75_CM.png')
pred, testOut = test_hard(
    groupb_hard_25, groupb_hard_25_in, groupb_hard_25_out)
graph_results("Group B Hard Activation 25% Train, 75% Test, Alpha: 0.3",
              df_b, groupb_hard_25, original_weights, 'GroupB_Hard_25.png')
confusion_matrix(
    "Group B Hard Activation 25% Train, 75% Test, Alpha: 0.3", pred, testOut, 'GroupB_Hard_25_CM.png')

groupc_hard_75, original_weights, groupc_hard_75_in, groupc_hard_75_out = normalize_and_train_hard(
    "GroupC", df_c, 0.3, epsilon_c, max_iterations, .75)
groupc_hard_25, original_weights, groupc_hard_25_in, groupc_hard_25_out = normalize_and_train_hard(
    "GroupC", df_c, 0.3, epsilon_c, max_iterations, .25)
pred, testOut = test_hard(
    groupc_hard_75, groupc_hard_75_in, groupc_hard_75_out)
graph_results("Group C Hard Activation 75% Train, 25% Test, Alpha: 0.3",
              df_c, groupc_hard_75, original_weights, 'GroupC_Hard_75.png')
confusion_matrix(
    "Group C Hard Activation 75% Train, 25% Test, Alpha: 0.3", pred, testOut, 'GroupC_Hard_75_CM.png')
pred, testOut = test_hard(
    groupc_hard_25, groupc_hard_25_in, groupc_hard_25_out)
graph_results("Group C Hard Activation 25% Train, 75% Test, Alpha: 0.3",
              df_c, groupc_hard_25, original_weights, 'GroupC_Hard_25.png')
confusion_matrix(
    "Group C Hard Activation 25% Train, 75% Test, Alpha: 0.3", pred, testOut, 'GroupC_Hard_25_CM.png')


groupa_soft_75, original_weights, groupa_soft_75_in, groupa_soft_75_out, gainA_75 = normalize_and_train_soft(
    "GroupA", df_a, 0.3, epsilon_a, max_iterations, .2, .75)
groupa_soft_25, original_weights, groupa_soft_25_in, groupa_soft_25_out, gainB_25 = normalize_and_train_soft(
    "GroupA", df_a, 0.3, epsilon_a, max_iterations, .2, .25)
pred, testOut = test_soft(
    groupa_soft_75, groupa_soft_75_in, groupa_soft_75_out, gainA_75)
graph_results("Group A Soft Activation 75% Train, 25% Test, Alpha: 0.3, Gain: 0.2",
              df_a, groupa_soft_75, original_weights, 'GroupA_Soft_75.png')
confusion_matrix(
    "Group A Soft Activation 75% Train, 25% Test, Alpha: 0.3, Gain: 0.2", pred, testOut, 'GroupA_Soft_75_CM.png')
pred, testOut = test_soft(
    groupa_soft_25, groupa_soft_25_in, groupa_soft_25_out, gainB_25)
graph_results("Group A Soft Activation 25% Train, 75% Test, Alpha: 0.3, Gain: 0.2",
              df_a, groupa_soft_25, original_weights, 'GroupA_Soft_25.png')
confusion_matrix(
    "Group A Soft Activation 25% Train, 75% Test, Alpha: 0.3, Gain: 0.2", pred, testOut, 'GroupA_Soft_25_CM.png')

groupb_soft_75, original_weights, groupb_soft_75_in, groupb_soft_75_out, gainB_75 = normalize_and_train_soft(
    "GroupB", df_b, 0.3, epsilon_b, max_iterations, .2, .75)
groupb_soft_25, original_weights, groupb_soft_25_in, groupb_soft_25_out, gainB_25 = normalize_and_train_soft(
    "GroupB", df_b, 0.3, epsilon_b, max_iterations, .2, .25)
pred, testOut = test_soft(
    groupb_soft_75, groupb_soft_75_in, groupb_soft_75_out, gainB_75)
graph_results("Group B Soft Activation 75% Train, 25% Test, Alpha: 0.3, Gain: 0.2",
              df_b, groupb_soft_75, original_weights, 'GroupB_Soft_75.png')
confusion_matrix(
    "Group B Soft Activation 75% Train, 25% Test, Alpha: 0.3, Gain: 0.2", pred, testOut, 'GroupB_Soft_75_CM.png')
pred, testOut = test_soft(
    groupb_soft_25, groupb_soft_25_in, groupb_soft_25_out, gainB_25)
graph_results("Group B Soft Activation 25% Train, 75% Test, Alpha: 0.3, Gain: 0.2",
              df_b, groupb_soft_25, original_weights, 'GroupB_Soft_25.png')
confusion_matrix(
    "Group B Soft Activation 25% Train, 75% Test, Alpha: 0.3, Gain: 0.2", pred, testOut, 'GroupB_Soft_25_CM.png')

groupc_soft_75, original_weights, groupc_soft_75_in, groupc_soft_75_out, gainC_75 = normalize_and_train_soft(
    "GroupC", df_c, 0.1, epsilon_c, max_iterations, .1, .75)
groupc_soft_25, original_weights, groupc_soft_25_in, groupc_soft_25_out, gainC_25 = normalize_and_train_soft(
    "GroupC", df_c, 0.1, epsilon_c, max_iterations, .1, .25)
pred, testOut = test_soft(
    groupc_soft_75, groupc_soft_75_in, groupc_soft_75_out, gainC_75)
graph_results("Group C Soft Activation 75% Train, 25% Test, Alpha: 0.1, Gain: 0.1",
              df_c, groupc_soft_75, original_weights, 'GroupC_Soft_75.png')
confusion_matrix(
    "Group C Soft Activation 75% Train, 25% Test, Alpha: 0.1, Gain: 0.1", pred, testOut, 'GroupC_Soft_75_CM.png')
pred, testOut = test_soft(
    groupc_soft_25, groupc_soft_25_in, groupc_soft_25_out, gainC_25)
graph_results("Group C Soft Activation 25% Train, 75% Test, Alpha: 0.1, Gain: 0.1",
              df_c, groupc_soft_25, original_weights, 'GroupC_Soft_25.png')
confusion_matrix(
    "Group C Soft Activation 25% Train, 75% Test, Alpha: 0.1, Gain: 0.1", pred, testOut, 'GroupC_Soft_25_CM.png')

alta_hard_75, original_weights, alta_hard_75_in, alta_hard_75_out = normalize_and_train_hard(
    "Alt A", df_alt_a, 0.3, epsilon_a, max_iterations, .75)
alta_hard_25, original_weights, alta_hard_25_in, alta_hard_25_out = normalize_and_train_hard(
    "Alt A", df_alt_a, 0.3, epsilon_a, max_iterations, .25)
pred, testOut = test_hard(
    alta_hard_75, alta_hard_75_in, alta_hard_75_out)
graph_results("Alt Data A Hard Activation 75% Train, 25% Test, Alpha: 0.3",
              df_alt_a, alta_hard_75, original_weights, 'AltA_Hard_75.png')
confusion_matrix(
    "Alt Data A Hard Activation 75% Train, 25% Test, Alpha: 0.3", pred, testOut, 'AltA_Hard_75_CM.png')
pred, testOut = test_hard(
    alta_hard_25, alta_hard_25_in, alta_hard_25_out)
graph_results("Alt Data A Hard Activation 25% Train, 75% Test, Alpha: 0.3",
              df_alt_a, alta_hard_25, original_weights, 'AltA_Hard_25.png')
confusion_matrix(
    "Alt Data A Hard Activation 25% Train, 75% Test, Alpha: 0.3", pred, testOut, 'AltA_Hard_25_CM.png')

altb_hard_75, original_weights, altb_hard_75_in, altb_hard_75_out = normalize_and_train_hard(
    "Alt B", df_alt_b, 0.3, epsilon_b, max_iterations, .75)
altb_hard_25, original_weights, altb_hard_25_in, altb_hard_25_out = normalize_and_train_hard(
    "Alt B", df_alt_b, 0.3, epsilon_b, max_iterations, .25)
pred, testOut = test_hard(
    altb_hard_75, altb_hard_75_in, altb_hard_75_out)
graph_results("Alt Data B Hard Activation 75% Train, 25% Test, Alpha: 0.3",
              df_alt_b, altb_hard_75, original_weights, 'AltB_Hard_75.png')
confusion_matrix(
    "Alt Data B Hard Activation 75% Train, 25% Test, Alpha: 0.3", pred, testOut, 'AltB_Hard_75_CM.png')
pred, testOut = test_hard(
    altb_hard_25, altb_hard_25_in, altb_hard_25_out)
graph_results("Alt Data B Hard Activation 25% Train, 75% Test, Alpha: 0.3",
              df_alt_b, altb_hard_25, original_weights, 'AltB_Hard_25.png')
confusion_matrix(
    "Alt Data B Hard Activation 25% Train, 75% Test, Alpha: 0.3", pred, testOut, 'AltB_Hard_25_CM.png')

altc_hard_75, original_weights, altc_hard_75_in, altc_hard_75_out = normalize_and_train_hard(
    "Alt C", df_alt_c, 0.3, epsilon_c, max_iterations, .75)
altc_hard_25, original_weights, altc_hard_25_in, altc_hard_25_out = normalize_and_train_hard(
    "Alt C", df_alt_c, 0.3, epsilon_c, max_iterations, .25)
pred, testOut = test_hard(
    altc_hard_75, altc_hard_75_in, altc_hard_75_out)
graph_results("Alt Data C Hard Activation 75% Train, 25% Test, Alpha: 0.3",
              df_alt_c, altc_hard_75, original_weights, 'AltC_Hard_75.png')
confusion_matrix(
    "Alt Data C Hard Activation 75% Train, 25% Test, Alpha: 0.3", pred, testOut, 'AltC_Hard_75_CM.png')
pred, testOut = test_hard(
    altc_hard_25, altc_hard_25_in, altc_hard_25_out)
graph_results("Alt Data C Hard Activation 25% Train, 75% Test, Alpha: 0.3",
              df_alt_c, altc_hard_25, original_weights, 'AltC_Hard_25.png')
confusion_matrix(
    "Alt Data C Hard Activation 25% Train, 75% Test, Alpha: 0.3", pred, testOut, 'AltC_Hard_25_CM.png')


alta_soft_75, original_weights, alta_soft_75_in, alta_soft_75_out, gainA_75 = normalize_and_train_soft(
    "AltA", df_alt_a, 0.3, epsilon_a, max_iterations, .2, .75)
alta_soft_25, original_weights, alta_soft_25_in, alta_soft_25_out, gainB_25 = normalize_and_train_soft(
    "AltA", df_alt_a, 0.3, epsilon_a, max_iterations, .2, .25)
pred, testOut = test_soft(
    alta_soft_75, alta_soft_75_in, alta_soft_75_out, gainA_75)
graph_results("Alt Data A Soft Activation 75% Train, 25% Test, Alpha: 0.3, Gain: 0.2",
              df_alt_a, alta_soft_75, original_weights, 'AltA_Soft_75.png')
confusion_matrix(
    "Alt Data A Soft Activation 75% Train, 25% Test, Alpha: 0.3, Gain: 0.2", pred, testOut, 'AltA_Soft_75_CM.png')
pred, testOut = test_soft(
    alta_soft_25, alta_soft_25_in, alta_soft_25_out, gainB_25)
graph_results("Alt Data A Soft Activation 25% Train, 75% Test, Alpha: 0.3, Gain: 0.2",
              df_alt_a, alta_soft_25, original_weights, 'AltA_Soft_25.png')
confusion_matrix(
    "Alt Data A Soft Activation 25% Train, 75% Test, Alpha: 0.3, Gain: 0.2", pred, testOut, 'AltA_Soft_25_CM.png')

altb_soft_75, original_weights, altb_soft_75_in, altb_soft_75_out, gainB_75 = normalize_and_train_soft(
    "AltB", df_alt_b, 0.3, epsilon_b, max_iterations, .2, .75)
altb_soft_25, original_weights, altb_soft_25_in, altb_soft_25_out, gainB_25 = normalize_and_train_soft(
    "AltB", df_alt_b, 0.3, epsilon_b, max_iterations, .2, .25)
pred, testOut = test_soft(
    altb_soft_75, altb_soft_75_in, altb_soft_75_out, gainB_75)
graph_results("Alt Data B Soft Activation 75% Train, 25% Test, Alpha: 0.3, Gain: 0.2",
              df_alt_b, altb_soft_75, original_weights, 'AltB_Soft_75.png')
confusion_matrix(
    "Alt Data B Soft Activation 75% Train, 25% Test, Alpha: 0.3, Gain: 0.2", pred, testOut, 'AltB_Soft_75_CM.png')
pred, testOut = test_soft(
    altb_soft_25, altb_soft_25_in, altb_soft_25_out, gainB_25)
graph_results("Alt Data B Soft Activation 25% Train, 25% Test, Alpha: 0.3, Gain: 0.2",
              df_alt_b, altb_soft_25, original_weights, 'AltB_Soft_25.png')
confusion_matrix(
    "Alt Data B Soft Activation 25% Train, 75% Test, Alpha: 0.3, Gain: 0.2", pred, testOut, 'AltB_Soft_25_CM.png')

altc_soft_75, original_weights, altc_soft_75_in, altc_soft_75_out, gainC_75 = normalize_and_train_soft(
    "AltC", df_alt_c, 0.3, epsilon_c, max_iterations, .2, .75)
altc_soft_25, original_weights, altc_soft_25_in, altc_soft_25_out, gainC_25 = normalize_and_train_soft(
    "AltC", df_alt_c, 0.3, epsilon_c, max_iterations, .2, .25)
pred, testOut = test_soft(
    altc_soft_75, altc_soft_75_in, altc_soft_75_out, gainC_75)
graph_results("Alt Data C Soft Activation 75% Train, 25% Test, Alpha: 0.3, Gain: 0.2",
              df_alt_c, altc_soft_75, original_weights, 'AltC_Soft_75.png')
confusion_matrix(
    "Alt Data C Soft Activation 75% Train, 25% Test, Alpha: 0.3, Gain: 0.2", pred, testOut, 'AltC_Soft_75_CM.png')
pred, testOut = test_soft(
    altc_soft_25, altc_soft_25_in, altc_soft_25_out, gainC_25)
graph_results("Alt Data C Soft Activation 25% Train, 25% Test, Alpha: 0.3, Gain: 0.2",
              df_alt_c, altc_soft_25, original_weights, 'AltC_Soft_25.png')
confusion_matrix(
    "Alt Data C Soft Activation 25% Train, 75% Test, Alpha: 0.3, Gain: 0.2", pred, testOut, 'AltC_Soft_25_CM.png')


def brute_force_hard(data_name, data, epsilon, max_iterations, percent_train):
    alpha = .01
    while alpha < 1:
        weights, test_input, test_out = normalize_and_train_hard(
            data_name, data, alpha, epsilon, max_iterations, percent_train)
        acc, tp_perc, fp_perc, tn_perc, fn_perc, error = test_hard(
            weights, test_input, test_out)
        with open("brute_force_hard_results.csv", mode='a+') as out_file:
            writer = csv.writer(out_file, delimiter=',', quotechar='"')
            writer.writerow([data_name, percent_train, alpha, acc, tp_perc,
                             tn_perc, fp_perc, fn_perc, error])
        alpha = alpha + .01


def brute_force_soft(data_name, data, epsilon, max_iterations, percent_train):
    alpha = .01
    while alpha < 1:
        gain = .01
        while gain < 1:
            weights, test_input, test_out, gain = normalize_and_train_soft(
                data_name, data, alpha, epsilon, max_iterations, gain, percent_train)
            acc, tp_perc, fp_perc, tn_perc, fn_perc, error = test_soft(
                weights, test_input, test_out, gain)
            with open("brute_force_soft_results.csv", mode='a+') as out_file:
                writer = csv.writer(out_file, delimiter=',', quotechar='"')
                writer.writerow([data_name, percent_train, alpha, gain, acc, tp_perc,
                                 tn_perc, fp_perc, fn_perc, error])
            gain = gain + .01
        alpha = alpha + .01


# brute_force_hard("GroupA", df_a, epsilon_a, max_iterations, .75)
# brute_force_hard("GroupA", df_a, epsilon_a, max_iterations, .25)
# brute_force_hard("GroupB", df_b, epsilon_b, max_iterations, .75)
# brute_force_hard("GroupB", df_b, epsilon_b, max_iterations, .25)
# brute_force_hard("GroupC", df_c, epsilon_c, max_iterations, .75)
# brute_force_hard("GroupC", df_c, epsilon_c, max_iterations, .25)

# brute_force_hard("Alt_A", df_alt_a, epsilon_a, max_iterations, .75)
# brute_force_hard("Alt_A", df_alt_a, epsilon_a, max_iterations, .25)
# brute_force_hard("Alt_B", df_alt_b, epsilon_alt_b, max_iterations, .75)
# brute_force_hard("Alt_B", df_alt_b, epsilon_alt_b, max_iterations, .25)
# brute_force_hard("Alt_C", df_alt_c, epsilon_alt_c, max_iterations, .75)
# brute_force_hard("Alt_C", df_alt_c, epsilon_alt_c, max_iterations, .25)

# brute_force_soft("GroupA", df_a, epsilon_a, max_iterations, .75)
# brute_force_soft("GroupA", df_a, epsilon_a, max_iterations, .25)
# brute_force_soft("GroupB", df_b, epsilon_b, max_iterations, .75)
# brute_force_soft("GroupB", df_b, epsilon_b, max_iterations, .25)
# brute_force_soft("GroupC", df_c, epsilon_c, max_iterations, .75)
# brute_force_soft("GroupC", df_c, epsilon_c, max_iterations, .25)

# brute_force_soft("Alt_A", df_alt_a, epsilon_a, max_iterations, .75)
# brute_force_soft("Alt_A", df_alt_a, epsilon_a, max_iterations, .25)
# brute_force_soft("Alt_B", df_alt_b, epsilon_alt_b, max_iterations, .75)
# brute_force_soft("Alt_B", df_alt_b, epsilon_alt_b, max_iterations, .25)
# brute_force_soft("Alt_C", df_alt_c, epsilon_alt_c, max_iterations, .75)
# brute_force_soft("Alt_C", df_alt_c, epsilon_alt_c, max_iterations, .25)
