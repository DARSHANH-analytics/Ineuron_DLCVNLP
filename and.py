from utils.model import perceptron
from utils.all_utils import prepare_data, save_model,save_plot
import pandas as pd

def main(data,ETA,epoch,filename,plot_filename):

    df = pd.DataFrame(data)
    print(df)

    X,y = prepare_data(df)

    model = perceptron(lr=ETA, epochs=EPOCHS)
    model.fit(X, y)

    _ = model.total_loss()

    save_model(model, filename)

    save_plot(df, plot_filename, model)

if __name__ == "__main__": # entry point
    AND = {'x1':[0,0,1,1],
            'x2':[0,1,0,1],
            'y':[0,0,0,1]
            }

    ETA = 0.3 # 0 and 1
    EPOCHS = 20
    filename = 'and.model'
    plot_filename = 'and.png'
    main(AND,ETA,EPOCHS,filename,plot_filename)