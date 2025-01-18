import sys
import importlib

OK = "\x1b[42m[ OK ]\x1b[0m"
FAIL = "\x1b[41m[FAIL]\x1b[0m"

try:
    from packaging.version import Version
except ImportError:
    print(
        FAIL, "'packaging' package not installed, install it with conda or pip"
    )
    sys.exit(1)

# first check the python version
print("Using python in", sys.prefix)
print(sys.version)
pyversion_str = f"{sys.version_info.major}.{sys.version_info.minor}"
pyversion = Version(pyversion_str)

if pyversion < Version("3.8"):
    print(
        FAIL,
        (
            "Python version 3.8 or above is required,"
            f" but {pyversion_str} is installed."
        ),
    )
    sys.exit(1)
print()


def import_version(pkg, min_ver, fail_msg=""):
    mod = None
    try:
        mod = importlib.import_module(pkg)
        if pkg in {"PIL"}:
            try:
                ver = mod.__version__
            except AttributeError:
                try:
                    ver = mod.VERSION
                except AttributeError:
                    try:
                        ver = mod.PILLOW_VERSION
                    except Exception:
                        raise
        else:
            ver = mod.__version__
        if Version(ver) < Version(min_ver):
            print(
                FAIL,
                (
                    f"{lib} version {min_ver} or higher required, but"
                    f" {ver} installed."
                ),
            )
        else:
            print(OK, f"{pkg} version {ver}")
    except ImportError:
        print(FAIL, f"{pkg} not installed. {fail_msg}")
    return mod


# add your requirements here
requirements = {
    "streamlit": "1.40.2",
    "pandas": "2.1.4",
    "scikit-learn": "1.4.2",
    "plotly": "5.24.1",
    "shap": "0.46.0",
    "matplotlib": "3.7.5"
}

# now the dependencies
for lib, required_version in list(requirements.items()):
    import_version(lib, required_version)
