from multiprocessing import cpu_count

from ..logger import logger
from .corner import plot_corner
from .gif_generator import make_gif
from .image_stacking import horizontal_concat, vertical_concat


def get_num_workers():
    """Get the number of workers for parallel processing
    """
    total_cpus_available = cpu_count()
    num_workers = 4
    if total_cpus_available > 64:
        num_workers = 16
    elif total_cpus_available > 32:
        num_workers = 8
    elif total_cpus_available < 16:
        num_workers = 4
    logger.warning(
        f"Using {num_workers}/{total_cpus_available} workers for parallel processing"
        "[Total number of CPUs not used to avoid memory issues]"
    )
    # TODO: check -- are we still having memory issues?
    return num_workers


def safe_savefig(fig, fname, *args, **kwargs):
    """Save the figure to a file."""
    try:
        fig.savefig(fname, *args, **kwargs)
    except Exception as e:
        logger.error(f"Could not save figure to {fname}: {e}")
