import pandas as pd
import numpy as np
from util.locdefs import geodesicDistance
from sklearn import linear_model, neighbors
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from math import sin, cos, sqrt, atan2, radians
import seaborn as sns


def get_errors(original_points,pred_positions):
    errors = []
    for (lat, lon), (pred_lat, pred_lon) in zip(original_points[["lat", "lon"]].values,
                                                pred_positions):
        error = geodesicDistance((lat, lon), (pred_lat, pred_lon)) * 1000
        errors.append(error)
    return errors
	
def show_stats(errors):
    print("Min Error (in meters):{}".format(np.min(errors)))
    print("Max Error (in meters):{}".format(np.max(errors)))
    print("Mean Error (in meters):{}".format(np.mean(errors)))
    print("Std. Deviation (in meters):{}".format(np.std(errors)))
	
def show_stats_graphs(errors):
    # Individual errors
    plt.plot(range(len(errors)),
            errors,
            color='blue',
            linestyle='dashed',
            marker='o',
            markerfacecolor='red',
            markersize=10)
    plt.title('Errors')
    plt.xlabel('Index')
    plt.ylabel('Error (m)')
    plt.show()
    plt.savefig('errors.png')    

    # Histogram
    plt.title('Histogram of errors')
    plt.ylabel('# of samples')
    plt.xlabel('Error (m)')
    plt.hist(errors, 10)
    plt.show()
    plt.savefig('histogram.png')  

    # Cumulative
    plt.title('Cumulative error')
    plt.xlabel('Error (m)')
    plt.ylabel('% of samples')
    X = np.linspace(1., max(errors), 100)
    Y = []
    for x in X:
        Y.append(100*len([e for e in errors if e<x])/len(errors))

    plt.plot(X, Y)
    plt.show()
    plt.savefig('cumulative.png') 
	
def show_box_plots(errors_list, names):
    all_errors = []
    [all_errors.extend(errors) for errors in errors_list]
    labels = []
    [labels.extend([names[i]]*len(errors_list[i])) for i in range(len(errors_list))]
    df_data = {'Error (m)': all_errors,
               'Approach': labels}
    df = pd.DataFrame(df_data)
    plt.title('Boxplots')
    sns.boxplot(x="Approach", y="Error (m)", data=df,width=0.8)
    plt.show()
    plt.savefig('cumulative.png') 