import pickle

with open("models/best_model.pkl", "rb") as file:
    model = pickle.load(file)

print(type(model))