import numpy as np
from scipy.signal import savgol_filter

def smooth_trajectory(traj: list[tuple[float, float, float]], window_size: int = 15, poly_order: int = 2):
    """
    Applies a Savitzky-Golay filter to smooth the jumper's trajectory.
    
    This filter is ideal for sports analysis because it smooths the data 
    while preserving the original shape (parabola) and trends better 
    than a simple moving average.

    Args:
        traj (list): A list of tuples containing (x, y, frame_index).
        window_size (int): The length of the filter window. Must be an ODD integer.
        poly_order (int): The order of the polynomial used to fit the samples.
                          Must be less than window_size.

    Returns:
        list: The smoothed trajectory as a list of (x, y, frame_index) tuples.
    """
    
    # Validation: The filter requires at least 'window_size' points to operate.
    # Also, the polynomial order must be smaller than the window size.
    if not traj or len(traj) < window_size or poly_order >= window_size:
        # Return original data if smoothing is not mathematically possible
        return traj

    # Extract coordinates into separate numpy arrays for vector processing
    xs = np.array([p[0] for p in traj])
    ys = np.array([p[1] for p in traj])
    frames = [p[2] for p in traj]

    # Apply the filter to X and Y independently.
    # Mode 'nearest' helps handle edges by extending the data boundary.
    smoothed_x = savgol_filter(xs, window_size, poly_order, mode='nearest')
    smoothed_y = savgol_filter(ys, window_size, poly_order, mode='nearest')

    # Re-assemble the data back into the original list-of-tuples format
    return list(zip(smoothed_x, smoothed_y, frames))