import matplotlib.pyplot as plt
import copy

def plot_hist(dico_hist):
    '''Plots the training and test accuracy of a dictionary (history.history)'''
    plt.plot(dico_hist["accuracy"])
    plt.plot(dico_hist["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()

def plot_global_history(dic1,dic2):
    '''Plots the train and test accuracies of two dictionnaries histories'''
    dic = copy.deepcopy(dic1)
    for key in dic:
        for value in dic2[key]:
            dic[key].append(value)
    plot_hist(dic)
