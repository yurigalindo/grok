from pandas import read_csv
import matplotlib.pyplot as plt


def metric_plots(filename):
        
    # filename = "data/metrics 80k.csv" # path to metric file
    metric = False
    try:    
        frame = read_csv(filename, index_col=None, header=0)
        metric = True
    except:
        print(f"No metric for {filename}")

    if metric:    
        frame = frame.sort_values('epoch')

        # To overcome sparse logging
        frame.fillna(method='ffill', inplace=True)
        
        frame.plot(x='step', y=['train_accuracy','val_accuracy'], kind='line')
        plt.title("Plots")
        plt.xscale('log')
        plt.savefig(f"{filename.split('/')[-2]}.png")
        plt.show()

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Plotting accuracy metrics')
    parser.add_argument('--metric',
                        help='Provide the path to the metric file')
    args = parser.parse_args()

    metric_plots(args.metric)
