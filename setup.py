from setuptools import find_packages, setup

# Leer dependencias desde requirements.txt
def read_requirements():
    with open("requirements.txt") as f:
        return f.read().splitlines()

setup(
    name="equipo21mna_coil2000",
    version="0.1.0",
    description="Pipeline de Machine Learning para el dataset Insurance Company Benchmark (COIL 2000).",
    author="Alejandro Jesús Mondragón Jiménez, Dylan Ulises Quiroz Hernández, Javier Alejandro Pérez Garza, Ricardo Diez Gutiérrez González",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=read_requirements(),
    python_requires=">=3.8",  
    entry_points={
        "console_scripts": [
            "eda_clean=src.data.eda_clean:main",
            "preprocessing=src.features.preprocessing:main",
            "train_model=src.models.train_model:main",
        ],
    },
)
