from dataSlope import DataModel
from hillModel import HillModel
import matplotlib.pyplot as plt

def setup() -> DataModel:
    print("choose the slope model")
    model_name = (input("Enter slope model name"))
    dataModel = DataModel(model_name)
    print("Selected slope model:", dataModel.name)
    return dataModel

def plot_profile(hillModel) -> None:
    xs,ys=hillModel.sample()
    plt.figure()
    plt.plot(xs, ys)
    plt.title(f"Ski Jump Hill Profile - {hillModel.slope_data.name}")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.grid(True)
    plt.axis("equal")  # utile per non deformare le pendenze
    plt.show()

def main() -> None:
    dataModel=setup()
    hillModel=HillModel(dataModel)
    print("Hill model created with the selected slope model.")
    plot_profile(hillModel)


if __name__ == "__main__":
    main()
