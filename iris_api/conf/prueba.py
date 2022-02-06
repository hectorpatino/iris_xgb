
from pathlib import Path
import iris_api

print(Path(iris_api.__file__))
print(Path(iris_api.__file__).resolve())
print(Path(iris_api.__file__).resolve().parent)
print(Path(iris_api.__file__).resolve().parent.parent)
print(type(str(Path(iris_api.__file__).resolve().parent.parent.parent)))
print(Path(iris_api.__file__).resolve().parent.parent.parent / "otro")