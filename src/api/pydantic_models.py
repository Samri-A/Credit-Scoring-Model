import mlflow.pyfunc

class AddN(mlflow.pyfunc.PythonModel):
    def __init__(self, n):
        self.n = n

    def predict(self, context, model_input):
        return model_input + self.n

# Instantiate and choose your save directory
model = AddN(n=5)
local_path = "/Users/HP/Desktop/Tenx/Credit-Scoring-Model/models/add_n_model"

mlflow.pyfunc.save_model(
    path=local_path,
    python_model=model
)
