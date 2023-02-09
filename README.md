# pyKZK

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-blue)](./LICENSE.md)
![Status](https://github.com/djps/pykzk/actions/workflows/main.yml/badge.svg)
![Coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.github.com%2Fdjps%2F05580cecfa0faf2ba85b2753e7bc4d7e%23file-pykzk-cobertura-coverage-json)

<!-- ![flake8-info](dist/flake8-badge.svg) -->


<!-- hopefully works in the near future
![GitHub](https://img.shields.io/github/license/djps/lyapunov?style=plastic)

![GitHub](https://img.shields.io/github/license/djps/MatrixCompletion?style=plastic&label=LICENSE)

![GitHub](https://img.shields.io/github/license/djps/jaxdiff?style=plastic)

![GitHub](https://img.shields.io/github/license/djps/jwave?style=plastic)

![GitHub](https://img.shields.io/github/license/djps/jaxwell?style=plastic) -->

<!-- [![DOI:10.1007/978-3-319-76207-4_15](https://zenodo.org/badge/DOI/10.1007/978-3-319-76207-4_15.svg)](https://doi.org/10.1007/978-3-319-76207-4_15) -->
<!-- [![Citation Badge](https://api.juleskreuer.eu/citation-badge.php?doi=10.1126/science.1058040)](https://juleskreuer.eu/projekte/citation-badge/) -->

Python implementation of the FDA axisymmetric KZK HITU simulator[^1], which can be accessed at [here](https://github.com/jsoneson/HITU_Simulator).

[^1]: J. E. Soneson "[A User‐Friendly Software Package for HIFU Simulation](https://doi.org/10.1063/1.3131405)" _AIP Conf. Proc._ **1113**(1)  pp 165 (2009)


> **Warning** a work in progress

## Table of Contents

* [Getting Started](#getting-started)
  * [Dependencies](#dependencies)
* [Usage](#todo)
* [Roadmap](#todo)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)

## Getting Started



### Dependencies

This needs the following packages:

* numpy
* scipy
* matplotlib
* tqdm

They can be found in [requirements.txt](./requirements.txt)

## Usage

From terminal
```console
foo@bar:~$ python3 runner.py
```

## TODO

- [x] testing
- [x] actions with coverage, flake8, version
- [x] requirements.txt



## Contributing

Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch 
```console 
foo@bar:~$ git checkout -b feature/AmazingFeature
```
3. Commit your Changes 
```console 
foo@bar:~$ git commit -m 'Add some AmazingFeature'
```
4. Push to the Branch 
```console 
foo@bar:~$ git push origin feature/AmazingFeature
```
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for more information.


<!-- CONTACT -->
## Contact

Email me


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* [HITU Simulator [FDA]](https://www.fda.gov/about-fda/cdrh-offices/hitu-simulator)
* [HITU Simulator [github]](https://github.com/jsoneson/HITU_Simulator)
