from utils.model import perceptron
from utils.all_utils import prepare_data, save_model,save_plot
import pandas as pd

AND = {'x1':[0,0,1,1],
      'x2':[0,1,0,1],
      'y':[0,0,0,1]}
dfand = pd.DataFrame(AND)
print(dfand)

X,y = prepare_data(dfand)

ETA = 0.3 # 0 and 1
EPOCHS = 20

model = perceptron(lr=ETA, epochs=EPOCHS)
model.fit(X, y)

_ = model.total_loss()

save_model(model, "and.model")

save_plot(dfand, "and.png", model)