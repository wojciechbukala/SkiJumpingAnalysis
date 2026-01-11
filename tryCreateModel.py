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
    (xs1, ys1), (xs2, ys2) = hillModel.sample() 
    plt.figure()
    plt.plot(xs1, ys1)
    plt.plot(xs2, ys2)
    plt.title(f"Ski Jump Hill Profile - {hillModel.name}")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.grid(True)
    plt.axis("equal")
    plt.show()

def main() -> None:
    dataModel=setup()
    hillModel=HillModel(dataModel)
    print("Hill model created with the selected slope model.")
    plot_profile(hillModel)


if __name__ == "__main__":
    main()
