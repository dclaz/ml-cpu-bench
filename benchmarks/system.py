import psutil
import platform
import pkg_resources


def get_system_info() -> dict:
    """
    Gathers system information.
    Returns:
        dict: A dictionary containing the following system information:
            - os (str): The operating system name and version.
            - python_version (str): The version of Python being used.
            - cpu_model (str): The model of the CPU.
            - physical_cores (int): The number of physical CPU cores.
            - logical_cores (int): The number of logical CPU cores.
            - ram_gb (float): The total amount of RAM in gigabytes.
    """
    """Gathers system information."""
    info = {}
    info["os"] = platform.system() + " " + platform.release()
    info["python_version"] = platform.python_version()
    info["cpu_model"] = platform.processor()
    info["physical_cores"] = psutil.cpu_count(logical=False)
    info["logical_cores"] = psutil.cpu_count(logical=True)
    info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)

    return info


def get_package_versions():
    """
    Gathers versions of relevant Python packages.

    This function checks the installed versions of a predefined list of Python
    packages and returns a dictionary with the package names as keys and their
    respective versions as values. If a package is not installed, its value
    will be "Not installed".

    Returns:
        dict: A dictionary where the keys are package names and the values are
              the installed versions or "Not installed" if the package is not found.
    """
    packages = [
        "numpy",
        "scipy",
        "scikit-learn",
        "pandas",
        "umap-learn",
        "lightgbm",
        "optuna",
    ]
    versions = {}
    for package in packages:
        try:
            versions[package] = pkg_resources.get_distribution(package).version
        except pkg_resources.DistributionNotFound:
            versions[package] = "Not installed"
    return versions

