# GPS and infinitesimal strain analysis (GPS Strain)

Main Project Resources: [PAJER, Luke](mailto:luke.pajer@gmail.com); [CRONIN, Vincent](mailto:vince_cronin@baylor.edu)

_Last Updated: October 2020_

[![GETSI page](https://img.shields.io/badge/GETSI-page-F78C26.svg)](https://serc.carleton.edu/getsi/teaching_materials/gps_strain/unit4.html)
[![License](https://img.shields.io/badge/LICENSE-mit-43B02A.svg)](/LICENSE)
[![jupyterlab](https://img.shields.io/badge/jupyterlab-0.35.4-F37821.svg)](https://jupyterlab.readthedocs.io/en/stable/)
[![python](https://img.shields.io/badge/python-3.6.5-yellow.svg)](https://jupyterlab.readthedocs.io/en/stable/)

-----

# PROJECT OVERVIEW

The `GPS_Strain` Python package is a simple translation of the methods developed by Vince Cronin and Phil Resor. The purpose of this package is to make the method available to those who are interested in GPS and infinitesimal strain analysis and are more comfortable using Python.

_From [the GETSI teaching materials](https://serc.carleton.edu/getsi/teaching_materials/gps_strain/unit4.html):_

> This module was designed for structural geology courses but can also be successfully used in geophysics, tectonics, or geohazards courses or possibly even a physics or engineering course seeking practical applications. It can be done at almost any point during the term. The module assumes that students have had a basic physical geology introduction to plate tectonics, faults, and earthquakes.

In addition to teaching purposes, the actual analysis can be used for other assessments. See the Victoria E. Worrell thesis titled ["The Seismo-Lineament Analysis Method (SLAM) Applied to the South Napa Earthquake and Antecedent Events"](https://baylor-ir.tdl.org/bitstream/handle/2104/9796/WORRELL-THESIS-2016.pdf?sequence=1&isAllowed=y) to see an example of how this method may be used in practice. 

If there are any issues or concerns with the python package, please reach out to [Luke Pajer](mailto:luke.pajer@gmail.com). For any questions regarding the GPS strain method, please reach out to [Vince Cronin](mailto:vince_cronin@baylor.edu).

-----

# CONTRIBUTORS

This project is an open project, and contributions are welcome from any individual. All contributors to this project are bound by a [code of conduct](/CODE_OF_CONDUCT.md). Please review and follow this code of conduct as part of your contribution.

#### Contributions to the GPS_Strain Python Package
- [Luke Pajer](mailto:luke.pajer@gmail.com) [![orcid](https://img.shields.io/badge/orcid-0000--0002--5218--7650-brightgreen.svg)](https://orcid.org/0000-0002-5218-7650)

#### GPS and infinitesimal strain analysis method Authors/Developers
- [Vince Cronin](mailto:vince_cronin@baylor.edu) [![orcid](https://img.shields.io/badge/orcid-0000--0002--3069--6470-brightgreen.svg)](https://orcid.org/0000-0002-3069-6470)
- [Phil Resor](mailto:presor@wesleyan.edu) [![orcid](https://img.shields.io/badge/orcid-0000--0003--3071--5085-brightgreen.svg)](https://orcid.org/0000-0003-3071-5085)

### Tips for Contributing

Issues and bug reports are always welcome.  Code clean-up, and feature additions can be done either through pull requests to [project forks]() or branches.

All products of the SLAM project are licensed under an [MIT License](LICENSE) unless otherwise noted.

-----

## HOW TO USE THIS REPOSITORY

This repository is available to be 

Base overview for SLAM map generation (_see the [GPS_Strain Wiki](https://github.com/The-Geology-Guy/GPS_Strain/wiki) for more information_):
1. Find GPS Sites -- _either using the `get_stations` function or simply find 3 on the UNAVCO site_
2. Get site coordinates and velocities using the `site_data` function
3. Process the strain data using the `strain_data` class
4. Use the processed strain data to produce one of the two types of maps: (1) Strain Ellipse Map or (2) Map Symbol Map

Once again, this is a simple overview of a typical GPS Strain task. This is in no way the limit of what can be done. See the [GPS_Strain Wiki](https://github.com/The-Geology-Guy/GPS_Strain/wiki) for more information.

### System Requirements

This project is developed using Python. There should be no issues with these projects running on Mac, Windows, or Linux. If there are any issues, please submit an issue and it will be investigated.

### Data Resources used in GPS_Strain

#### A. Data Sources

- [UNAVCO Web Services](https://www.unavco.org/data/web-services/documentation/documentation.html#!/GNSS47GPS/getPositionByStationId) is used for the station locations and relative station velocity.  

#### B. Physical Maps

- [Stamen Map Tile Sets](http://maps.stamen.com/#watercolor/12/37.7706/-122.3782) are used to generate the physical maps in this package. The Stamen map tile sets are copyright Stamen Design, under a Creative Commons Attribution (CC BY 3.0) license.

### Key Outputs

GPS_Strain provides the user a map with seismo-lineament bounds defined. Below are two examples:

#### Example of a Strain Ellipse Map with Legend and Data
<p align="center"><img src="images/Finished_Maps/AC63_strain.jpg" width=800/></p>

#### Example of a Map Symbol plotted with Legend and Strain Ellipse
![image](images/Finished_Maps/AC63_symbol.jpg)

-----

## <a name="section9">REFERENCES</a>

**_Note: The reference section is still not complete. This is in progress. Only links to resources appear at the moment._**

- https://serc.carleton.edu/getsi/teaching_materials/gps_strain/overview.html
- https://croninprojects.org/Vince/Course/IntroStructGeol/index.htm
- https://www.unavco.org/data/web-services/documentation/documentation.html#!/GNSS47GPS/getPositionByStationId
- https://gited.io/notebook/57/MTFfU3RyYWluX2VsbGlwc2UuaXB5bmI=/aHR0cHM6Ly9naXRodWIuY29tL29uZHJvbGV4YS9zZzIvYmxvYi9tYXN0ZXIvMTFfU3RyYWluX2VsbGlwc2UuaXB5bmI=
- https://www.continuummechanics.org/deformationgradient.html
- https://gist.github.com/jakebathman/719e8416191ba14bb6e700fc2d5fccc5
- http://maps.stamen.com/ 
    - The Stamen map tile sets are copyright Stamen Design, under a Creative Commons Attribution (CC BY 3.0) license.
