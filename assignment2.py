import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import math
import random
import csv

pd.options.mode.chained_assignment = None

# Define Max Iterarions, error Epsilons, and learning rate alpha
max_iterations = 5000
epsilon_a = .00001
epsilon_b = 100
epsilon_c = 400
epsilon_alt_b = 200
epsilon_alt_c = 700
alpha = 0.1  # 0.1

# Read Datasets
a_25 = pd.read_csv("./splits/a_0.25.csv")
a_75 = pd.read_csv("./splits/a_0.75.csv")
a_25.columns = ['Price', 'Weight', 'Type']
a_75.columns = ['Price', 'Weight', 'Type']
# df_a.columns = ['Price', 'Weight', 'Type']
# df_b = pd.read_csv("data_B.csv")
# df_b.columns = ['Price', 'Weight', 'Type']
# df_c = pd.read_csv("data_C.csv")
# df_c.columns = ['Price', 'Weight', 'Type']
# df_alt_a = pd.read_csv("data_ALT_A.csv")
# df_alt_a.columns = ['Price', 'Weight', 'Type']
# df_alt_b = pd.read_csv("data_ALT_B.csv")
# df_alt_b.columns = ['Price', 'Weight', 'Type']
# df_alt_c = pd.read_csv("data_ALT_C.csv")
# df_alt_c.columns = ['Price', 'Weight', 'Type']
c_25 = pd.read_csv("./splits/c_0.25.csv")
c_75 = pd.read_csv("./splits/c_0.75.csv")
c_25.columns = ['Price', 'Weight', 'Type']
c_75.columns = ['Price', 'Weight', 'Type']
alt_c_25 = pd.read_csv("./splits/alt_c_0.25.csv")
alt_c_75 = pd.read_csv("./splits/alt_c_0.75.csv")
alt_c_25.columns = ['Price', 'Weight', 'Type']
alt_c_75.columns = ['Price', 'Weight', 'Type']


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
    # weights = [-.5, -.5, .5]
    weights = [y_w, x_w, b]
    original_weights = weights
    train_error = 0
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
        train_error = total_error
        if(total_error < epsilon):
            break
    return weights, train_error


def train_soft(input_arr, desired_out, alpha, epsilon, max_iterations, gain):
    y_w = random.uniform(-.5, .5)
    x_w = random.uniform(-.5, .5)
    b = random.uniform(-.5, .5)
    # weights = [-.5, -.5, .5]
    weights = [y_w, x_w, b]
    original_weights = weights
    train_error = 0
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
        train_error = total_error
        if(total_error < epsilon):
            break

    return weights, train_error


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


# def get_random_sample(data, percent, name):
#     train, test = train_test_split(data, test_size=percent, shuffle=True)
#     train.to_csv('splits/' + name + str(1 - percent), index = False)
#     test.to_csv('splits/' + name + str(percent), index = False)
#     train_in = get_input_array(train)
#     train_out = get_desired_out_array(train)
#     test_in = get_input_array(test)
#     test_out = get_desired_out_array(test)
#     return train, test, train_in, train_out, test_in, test_out

# get_random_sample(df_c, 0.75, 'c_')


def normalize_and_train_hard(dataset_name, test, train, alpha, epsilon, max_iterations, perc):
    # scaled_data = normalize_data(data)
    # input_arr = get_input_array(scaled_data)
    # output_arr = get_desired_out_array(scaled_data)
    # inputTrain, outputTrain, inputTest, outputTest = get_percent_train_and_test(
    #     input_arr, output_arr, perc)
    scaled_train = normalize_data(train)
    scaled_test = normalize_data(test)

    inputTrain = get_input_array(scaled_train)
    inputTest = get_input_array(scaled_test)
    outputTrain = get_desired_out_array(scaled_train)
    outputTest = get_desired_out_array(scaled_test)
    # train, test, inputTrain, outputTrain, inputTest, outputTest = get_random_sample(
    #     scaled_data, perc)
    result_weights, training_error = train_hard(
        inputTrain, outputTrain, alpha, epsilon, max_iterations)
    print("%s Final Weights: %s Train_Error: %f" %
          (dataset_name, str(result_weights), training_error))
    return result_weights, inputTrain, inputTest, outputTest, scaled_train, scaled_test


def normalize_and_train_soft(dataset_name, test, train, alpha, epsilon, max_iterations, gain, perc):
    # scaled_data = normalize_data(data)
    # input_arr = get_input_array(scaled_data)
    # output_arr = get_desired_out_array(scaled_data)
    # inputTrain, outputTrain, inputTest, outputTest = get_percent_train_and_test(
    #     input_arr, output_arr, perc)
    scaled_train = normalize_data(train)
    scaled_test = normalize_data(test)

    inputTrain = get_input_array(scaled_train)
    inputTest = get_input_array(scaled_test)
    outputTrain = get_desired_out_array(scaled_train)
    outputTest = get_desired_out_array(scaled_test)

    # train, test, inputTrain, outputTrain, inputTest, outputTest = get_random_sample(
    #     scaled_data, perc)
    result_weights, training_error = train_soft(
        inputTrain, outputTrain, alpha, epsilon, max_iterations, gain)
    print("%s Final Weights: %s Train_Error: %f" %
          (dataset_name, str(result_weights), training_error))
    return result_weights, inputTrain, inputTest, outputTest, gain, scaled_train, scaled_test,


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
    print("accuracy: %f, tp_perc: %f, fp_perc: %f, tn_perc: %f, fn_perc: %f, error: %f " %
          (acc, tp_perc, fp_perc, tn_perc, fn_perc, error))
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
    print("accuracy: %f, tp_perc: %f, fp_perc: %f, tn_perc: %f, fn_perc: %f, error: %f" %
          (acc, tp_perc, fp_perc, tn_perc, fn_perc, error))
    return pred, testOut


def graph_results(title, dataset, final_weights):
    #data = normalize_data(dataset)
    sns.set(rc={'figure.figsize': (10, 8)})
    sns.set_style("whitegrid")
    sns.scatterplot(data=dataset, x='Weight',
                    y='Price', hue='Type', linewidth=0)
    yf_weight = final_weights[0]
    xf_weight = final_weights[1]
    f_bias = final_weights[2]

    fm = (-1 * xf_weight) / (yf_weight)
    fb = (-1 * f_bias) / yf_weight
    fx = np.linspace(-0.1, 1.1, 100)
    fy = (fm * fx) + fb
    plt.plot(fx, fy, label="Final Line", color='green')

    # yo_weight = original_weights[0]
    # xo_weight = original_weights[1]
    # o_bias = original_weights[2]

    # om = (-1 * xo_weight) / (yo_weight)
    # ob = (-1 * o_bias) / yo_weight
    # ox = np.linspace(-0.1, 1.1, 100)
    # oy = (om * ox) + ob
    # plt.plot(ox, oy, label="Initial Line", linestyle='dotted', color='red')

    plt.legend(loc='best', borderaxespad=0.)
    plt.title(title)
    plt.ylim(-.1, 1.1)
    plt.xlim(-.1, 1.1)
    plt.show()
    # plt.savefig(filename)
    # plt.close()


def confusion_matrix(title, predicted, actual):
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


# groupa_hard_75, groupa_hard_75_train, groupa_hard_75_test, groupa_hard_75_out, train, test = normalize_and_train_hard(
#     "Group A", a_25, a_75, 0.3, epsilon_a, max_iterations, .25)
# groupa_hard_25, groupa_hard_25_train, groupa_hard_25_test, groupa_hard_25_out, train, test = normalize_and_train_hard(
#     "Group A", a_75, a_25, 0.3, epsilon_a, max_iterations, .75)
# pred, testOut = test_hard(
#     groupa_hard_75, groupa_hard_75_test, groupa_hard_75_out)
# graph_results("[Training] Group A Hard Activation 75% Train, 25% Test, Alpha: 0.3",
#               train, groupa_hard_75)
# graph_results("[Testing] Group A Hard Activation 75% Train, 25% Test, Alpha: 0.3",
#               test, groupa_hard_75)
# confusion_matrix(
#     "Group A Hard Activation 75% Train, 25% Test, Alpha: 0.3", pred, testOut)
# pred, testOut = test_hard(
#     groupa_hard_25, groupa_hard_25_test, groupa_hard_25_out)
# graph_results("[Training] Group A Hard Activation 25% Train, 75% Test, Alpha: 0.3",
#               train, groupa_hard_25)
# graph_results("[Testing] Group A Hard Activation 25% Train, 75% Test, Alpha: 0.3",
#               test, groupa_hard_25)
# confusion_matrix(
#     "Group A Hard Activation 25% Train, 75% Test, Alpha: 0.3", pred, testOut)

# groupb_hard_75, groupb_hard_75_train, groupb_hard_75_test, groupb_hard_75_out, train, test = normalize_and_train_hard(
#     "Group B", df_b, 0.3, epsilon_b, max_iterations, .25)
# groupb_hard_25, groupb_hard_25_train, groupb_hard_25_test, groupb_hard_25_out, train, test = normalize_and_train_hard(
#     "Group B", df_b, 0.3, epsilon_b, max_iterations, .75)
# pred, testOut = test_hard(
#     groupb_hard_75, groupb_hard_75_test, groupb_hard_75_out)
# graph_results("[Training] Group B Hard Activation 75% Train, 25% Test, Alpha: 0.3",
#               train, groupb_hard_75)
# graph_results("[Testing] Group B Hard Activation 75% Train, 25% Test, Alpha: 0.3",
#               test, groupb_hard_75)
# confusion_matrix(
#     "Group B Hard Activation 75% Train, 25% Test, Alpha: 0.3", pred, testOut)
# pred, testOut = test_hard(
#     groupb_hard_25, groupb_hard_25_test, groupb_hard_25_out)
# graph_results("[Training] Group B Hard Activation 25% Train, 75% Test, Alpha: 0.3",
#               train, groupb_hard_25)
# graph_results("[Testing] Group B Hard Activation 25% Train, 75% Test, Alpha: 0.3",
#               test, groupb_hard_25)
# confusion_matrix(
#     "Group B Hard Activation 25% Train, 75% Test, Alpha: 0.3", pred, testOut)

groupc_hard_75, groupc_hard_75_train, groupc_hard_75_test, groupc_hard_75_out, train, test = normalize_and_train_hard(
    "GroupC", c_25, c_75, 0.3, epsilon_c, max_iterations, .25)
groupc_hard_25, groupc_hard_25_train, groupc_hard_25_test, groupc_hard_25_out, train, test = normalize_and_train_hard(
    "GroupC", c_25, c_75, 0.3, epsilon_c, max_iterations, .75)
pred, testOut = test_hard(
    groupc_hard_75, groupc_hard_75_test, groupc_hard_75_out)
graph_results("[Training] Group C Hard Activation 75% Train, 25% Test, Alpha: 0.3",
              train, groupc_hard_75)
graph_results("[Testing] Group C Hard Activation 75% Train, 25% Test, Alpha: 0.3",
              test, groupc_hard_75)
confusion_matrix(
    "Group C Hard Activation 75% Train, 25% Test, Alpha: 0.3", pred, testOut)
pred, testOut = test_hard(
    groupc_hard_25, groupc_hard_25_test, groupc_hard_25_out)
graph_results("[Training] Group C Hard Activation 25% Train, 75% Test, Alpha: 0.3",
              train, groupc_hard_25)
graph_results("[Testing] Group C Hard Activation 25% Train, 75% Test, Alpha: 0.3",
              test, groupc_hard_25)
confusion_matrix(
    "Group C Hard Activation 25% Train, 75% Test, Alpha: 0.3", pred, testOut)

# groupa_soft_75, groupa_soft_75_train, groupa_soft_75_test, groupa_soft_75_out, gainA_75, train, test = normalize_and_train_soft(
#     "GroupA", df_a, 0.3, epsilon_a, max_iterations, .2, .25)
# groupa_soft_25, groupa_soft_25_train, groupa_soft_25_test, groupa_soft_25_out, gainB_25, train, test = normalize_and_train_soft(
#     "GroupA", df_a, 0.3, epsilon_a, max_iterations, .2, .75)
# pred, testOut = test_soft(
#     groupa_soft_75, groupa_soft_75_test, groupa_soft_75_out, gainA_75)
# graph_results("[Training] Group A Soft Activation 75% Train, 25% Test, Alpha: 0.3, Gain: 0.2",
#               train, groupa_soft_75)
# graph_results("[Testing] Group A Soft Activation 75% Train, 25% Test, Alpha: 0.3, Gain: 0.2",
#               test, groupa_soft_75)
# confusion_matrix(
#     "Group A Soft Activation 75% Train, 25% Test, Alpha: 0.3, Gain: 0.2", pred, testOut)
# pred, testOut = test_soft(
#     groupa_soft_25, groupa_soft_25_test, groupa_soft_25_out, gainB_25)
# graph_results("[Training] Group A Soft Activation 25% Train, 75% Test, Alpha: 0.3, Gain: 0.2",
#               train, groupa_soft_25)
# graph_results("[Testing] Group A Soft Activation 25% Train, 75% Test, Alpha: 0.3, Gain: 0.2",
#               test, groupa_soft_25)
# confusion_matrix(
#     "Group A Soft Activation 25% Train, 75% Test, Alpha: 0.3, Gain: 0.2", pred, testOut)

# groupb_soft_75, groupb_soft_75_train, groupb_soft_75_test, groupb_soft_75_out, gainB_75, train, test = normalize_and_train_soft(
#     "GroupB", df_b, 0.3, epsilon_b, max_iterations, .2, .25)
# groupb_soft_25, groupb_soft_25_train, groupb_soft_25_test, groupb_soft_25_out, gainB_25, train, test = normalize_and_train_soft(
#     "GroupB", df_b, 0.3, epsilon_b, max_iterations, .2, .75)
# pred, testOut = test_soft(
#     groupb_soft_75, groupb_soft_75_test, groupb_soft_75_out, gainB_75)
# graph_results("[Training] Group B Soft Activation 75% Train, 25% Test, Alpha: 0.3, Gain: 0.2",
#               train, groupb_soft_75)
# graph_results("[Testing] Group B Soft Activation 75% Train, 25% Test, Alpha: 0.3, Gain: 0.2",
#               test, groupb_soft_75)
# confusion_matrix(
#     "Group B Soft Activation 75% Train, 25% Test, Alpha: 0.3, Gain: 0.2", pred, testOut)
# pred, testOut = test_soft(
#     groupb_soft_25, groupb_soft_25_test, groupb_soft_25_out, gainB_25)
# graph_results("[Training] Group B Soft Activation 25% Train, 75% Test, Alpha: 0.3, Gain: 0.2",
#               train, groupb_soft_25)
# graph_results("[Testing] Group B Soft Activation 25% Train, 75% Test, Alpha: 0.3, Gain: 0.2",
#               test, groupb_soft_25)
# confusion_matrix(
#     "Group B Soft Activation 25% Train, 75% Test, Alpha: 0.3, Gain: 0.2", pred, testOut)

groupc_soft_75, groupc_soft_75_train, groupc_soft_75_test, groupc_soft_75_out, gainC_75, train, test = normalize_and_train_soft(
    "GroupC", c_25, c_75, 0.1, epsilon_c, max_iterations, .1, .25)
groupc_soft_25, groupc_soft_25_train, groupc_soft_25_test, groupc_soft_25_out, gainC_25, train, test = normalize_and_train_soft(
    "GroupC", c_25, c_75, 0.1, epsilon_c, max_iterations, .1, .75)
pred, testOut = test_soft(
    groupc_soft_75, groupc_soft_75_test, groupc_soft_75_out, gainC_75)
graph_results("[Training] Group C Soft Activation 75% Train, 25% Test, Alpha: 0.1, Gain: 0.1",
              train, groupc_soft_75)
graph_results("[Training]Group C Soft Activation 75% Train, 25% Test, Alpha: 0.1, Gain: 0.1",
              test, groupc_soft_75)
confusion_matrix(
    "Group C Soft Activation 75% Train, 25% Test, Alpha: 0.1, Gain: 0.1", pred, testOut)
pred, testOut = test_soft(
    groupc_soft_25, groupc_soft_25_test, groupc_soft_25_out, gainC_25)
graph_results("[Training] Group C Soft Activation 25% Train, 75% Test, Alpha: 0.1, Gain: 0.1",
              train, groupc_soft_25)
graph_results("[Testing] Group C Soft Activation 25% Train, 75% Test, Alpha: 0.1, Gain: 0.1",
              test, groupc_soft_25)
confusion_matrix(
    "Group C Soft Activation 25% Train, 75% Test, Alpha: 0.1, Gain: 0.1", pred, testOut)

# alta_hard_75, alta_hard_75_train, alta_hard_75_test, alta_hard_75_out, train, test = normalize_and_train_hard(
#     "Alt A", df_alt_a, 0.3, epsilon_a, max_iterations, .25)
# alta_hard_25, alta_hard_25_train, alta_hard_25_test, alta_hard_25_out, train, test = normalize_and_train_hard(
#     "Alt A", df_alt_a, 0.3, epsilon_a, max_iterations, .75)
# pred, testOut = test_hard(
#     alta_hard_75, alta_hard_75_test, alta_hard_75_out)
# graph_results("[Training] Alt Data A Hard Activation 75% Train, 25% Test, Alpha: 0.3",
#               train, alta_hard_75)
# graph_results("[Testing] Alt Data A Hard Activation 75% Train, 25% Test, Alpha: 0.3",
#               test, alta_hard_75)
# confusion_matrix(
#     "Alt Data A Hard Activation 75% Train, 25% Test, Alpha: 0.3", pred, testOut)
# pred, testOut = test_hard(
#     alta_hard_25, alta_hard_25_test, alta_hard_25_out)
# graph_results("[Training] Alt Data A Hard Activation 25% Train, 75% Test, Alpha: 0.3",
#               train, alta_hard_25)
# graph_results("[Testing] Alt Data A Hard Activation 25% Train, 75% Test, Alpha: 0.3",
#               test, alta_hard_25)
# confusion_matrix(
#     "Alt Data A Hard Activation 25% Train, 75% Test, Alpha: 0.3", pred, testOut)

# altb_hard_75, altb_hard_75_train, altb_hard_75_test, altb_hard_75_out, train, test = normalize_and_train_hard(
#     "Alt B", df_alt_b, 0.3, epsilon_b, max_iterations, .25)
# altb_hard_25, altb_hard_25_train, altb_hard_25_test, altb_hard_25_out, train, test = normalize_and_train_hard(
#     "Alt B", df_alt_b, 0.3, epsilon_b, max_iterations, .75)
# pred, testOut = test_hard(
#     altb_hard_75, altb_hard_75_test, altb_hard_75_out)
# graph_results("[Training] Alt Data B Hard Activation 75% Train, 25% Test, Alpha: 0.3",
#               train, altb_hard_75)
# graph_results("[Testing] Alt Data B Hard Activation 75% Train, 25% Test, Alpha: 0.3",
#               test, altb_hard_75)
# confusion_matrix(
#     "Alt Data B Hard Activation 75% Train, 25% Test, Alpha: 0.3", pred, testOut)
# pred, testOut = test_hard(
#     altb_hard_25, altb_hard_25_test, altb_hard_25_out)
# graph_results("[Training] Alt Data B Hard Activation 25% Train, 75% Test, Alpha: 0.3",
#               train, altb_hard_25)
# graph_results("[Testing] Alt Data B Hard Activation 25% Train, 75% Test, Alpha: 0.3",
#               test, altb_hard_25)
# confusion_matrix(
#     "Alt Data B Hard Activation 25% Train, 75% Test, Alpha: 0.3", pred, testOut)

altc_hard_75, altc_hard_75_train, altc_hard_75_test, altc_hard_75_out, train, test = normalize_and_train_hard(
    "Alt C", alt_c_25, alt_c_75, 0.3, epsilon_c, max_iterations, .25)
altc_hard_25, altc_hard_25_train, altc_hard_25_test, altc_hard_25_out, train, test = normalize_and_train_hard(
    "Alt C", alt_c_75, alt_c_25, 0.3, epsilon_c, max_iterations, .75)
pred, testOut = test_hard(
    altc_hard_75, altc_hard_75_test, altc_hard_75_out)
graph_results("[Training] Alt Data C Hard Activation 75% Train, 25% Test, Alpha: 0.3",
              train, altc_hard_75)
graph_results("[Testing] Alt Data C Hard Activation 75% Train, 25% Test, Alpha: 0.3",
              test, altc_hard_75)
confusion_matrix(
    "Alt Data C Hard Activation 75% Train, 25% Test, Alpha: 0.3", pred, testOut)
pred, testOut = test_hard(
    altc_hard_25, altc_hard_25_test, altc_hard_25_out)
graph_results("[Training] Alt Data C Hard Activation 25% Train, 75% Test, Alpha: 0.3",
              train, altc_hard_25)
graph_results("[Testing] Alt Data C Hard Activation 25% Train, 75% Test, Alpha: 0.3",
              test, altc_hard_25)
confusion_matrix(
    "Alt Data C Hard Activation 25% Train, 75% Test, Alpha: 0.3", pred, testOut)

# alta_soft_75, alta_soft_75_train, alta_soft_75_test, alta_soft_75_out, gainA_75, train, test = normalize_and_train_soft(
#     "AltA", df_alt_a, 0.3, epsilon_a, max_iterations, .2, .25)
# alta_soft_25, alta_soft_25_train, alta_soft_25_test, alta_soft_25_out, gainB_25, train, test = normalize_and_train_soft(
#     "AltA", df_alt_a, 0.3, epsilon_a, max_iterations, .2, .75)
# pred, testOut = test_soft(
#     alta_soft_75, alta_soft_75_test, alta_soft_75_out, gainA_75)
# graph_results("[Training] Alt Data A Soft Activation 75% Train, 25% Test, Alpha: 0.3, Gain: 0.2",
#               train, alta_soft_75)
# graph_results("[Testing] Alt Data A Soft Activation 75% Train, 25% Test, Alpha: 0.3, Gain: 0.2",
#               test, alta_soft_75)
# confusion_matrix(
#     "Alt Data A Soft Activation 75% Train, 25% Test, Alpha: 0.3, Gain: 0.2", pred, testOut)
# pred, testOut = test_soft(
#     alta_soft_25, alta_soft_25_test, alta_soft_25_out, gainB_25)
# graph_results("[Training] Alt Data A Soft Activation 25% Train, 75% Test, Alpha: 0.3, Gain: 0.2",
#               train, alta_soft_25)
# graph_results("[Testing] Alt Data A Soft Activation 25% Train, 75% Test, Alpha: 0.3, Gain: 0.2",
#               test, alta_soft_25)
# confusion_matrix(
#     "Alt Data A Soft Activation 25% Train, 75% Test, Alpha: 0.3, Gain: 0.2", pred, testOut)

# altb_soft_75, altb_soft_75_train, altb_soft_75_test, altb_soft_75_out, gainB_75, train, test = normalize_and_train_soft(
#     "AltB", df_alt_b, 0.3, epsilon_b, max_iterations, .2, .25)
# altb_soft_25, altb_soft_25_train, altb_soft_25_test, altb_soft_25_out, gainB_25, train, test = normalize_and_train_soft(
#     "AltB", df_alt_b, 0.3, epsilon_b, max_iterations, .2, .75)
# pred, testOut = test_soft(
#     altb_soft_75, altb_soft_75_test, altb_soft_75_out, gainB_75)
# graph_results("[Training] Alt Data B Soft Activation 75% Train, 25% Test, Alpha: 0.3, Gain: 0.2",
#               train, altb_soft_75)
# graph_results("[Testing] Alt Data B Soft Activation 75% Train, 25% Test, Alpha: 0.3, Gain: 0.2",
#               test, altb_soft_75)
# confusion_matrix(
#     "Alt Data B Soft Activation 75% Train, 25% Test, Alpha: 0.3, Gain: 0.2", pred, testOut)
# pred, testOut = test_soft(
#     altb_soft_25, altb_soft_25_test, altb_soft_25_out, gainB_25)
# graph_results("[Training] Alt Data B Soft Activation 25% Train, 25% Test, Alpha: 0.3, Gain: 0.2",
#               train, altb_soft_25)
# graph_results("[Testing] Alt Data B Soft Activation 25% Train, 25% Test, Alpha: 0.3, Gain: 0.2",
#               test, altb_soft_25)
# confusion_matrix(
#     "Alt Data B Soft Activation 25% Train, 75% Test, Alpha: 0.3, Gain: 0.2", pred, testOut)

altc_soft_75, altc_soft_75_train, altc_soft_75_test, altc_soft_75_out, gainC_75, train, test = normalize_and_train_soft(
    "AltC", alt_c_25, alt_c_75, 0.3, epsilon_c, max_iterations, .2, .25)
altc_soft_25, altc_soft_25_train, altc_soft_25_test, altc_soft_25_out, gainC_25, train, test = normalize_and_train_soft(
    "AltC", alt_c_75, alt_c_25, 0.3, epsilon_c, max_iterations, .2, .75)
pred, testOut = test_soft(
    altc_soft_75, altc_soft_75_test, altc_soft_75_out, gainC_75)
graph_results("[Training] Alt Data C Soft Activation 75% Train, 25% Test, Alpha: 0.3, Gain: 0.2",
              train, altc_soft_75)
graph_results("[Testing] Alt Data C Soft Activation 75% Train, 25% Test, Alpha: 0.3, Gain: 0.2",
              test, altc_soft_75)
confusion_matrix(
    "Alt Data C Soft Activation 75% Train, 25% Test, Alpha: 0.3, Gain: 0.2", pred, testOut)
pred, testOut = test_soft(
    altc_soft_25, altc_soft_25_test, altc_soft_25_out, gainC_25)
graph_results("[Training] Alt Data C Soft Activation 25% Train, 25% Test, Alpha: 0.3, Gain: 0.2",
              train, altc_soft_25)
graph_results("[Testing] Alt Data C Soft Activation 25% Train, 25% Test, Alpha: 0.3, Gain: 0.2",
              test, altc_soft_25)
confusion_matrix(
    "Alt Data C Soft Activation 25% Train, 75% Test, Alpha: 0.3, Gain: 0.2", pred, testOut)
