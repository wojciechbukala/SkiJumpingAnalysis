from dataSlope import DataModel
from hillModel import HillModel
import matplotlib.pyplot as plt

def setup() -> DataModel:
    print("choose the slope model")
    model_name = (input("Enter slope model name"))
    dataModel = DataModel(model_name)
    print("Selected slope model:", dataModel.name)
    return dataModel

def plot_profile(hillModel, x_max: float | None = None) -> None:
    (xs1, ys1), (xs2, ys2) = hillModel.sample(x_max)
    plt.figure()
    plt.plot(xs1, ys1)
    plt.plot(xs2, ys2)
    plt.title(f"Ski Jump Hill Profile - {hillModel.name}")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.grid(True)
    plt.axis("equal")
    # set x-limits to start at the inrun start and end at requested x_max (or hill end)
    x_start = float(hillModel.x_inrun_start)
    x_end = float(x_max) if x_max is not None else float(hillModel.x_end)
    plt.xlim(x_start, x_end)
    plt.show()

def main() -> None:
    dataModel=setup()
    hillModel=HillModel(dataModel)
    print("Hill model created with the selected slope model.")
    # max distance to plot
    plot_profile(hillModel, x_max=300.0)


if __name__ == "__main__":
    main()
