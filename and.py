from utils.model import perceptron
from utils.all_utils import prepare_data, save_model,save_plot
import pandas as pd
import logging
import os

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename= os.path.join(log_dir,"running_logs.log"),level=logging.INFO, format=logging_str, filemode="a")

def main(data,ETA,epoch,filename,plot_filename):

    df = pd.DataFrame(data)
    logging.info(df)

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
    try:
        logging.info(">>>>> starting training >>>>>")
        filename = 'and.model'
        plot_filename = 'and.png'
        main(AND,ETA,EPOCHS,filename,plot_filename)
        logging.info("<<<<< training done successfully<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e