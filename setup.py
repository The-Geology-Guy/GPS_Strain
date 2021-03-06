from setuptools import setup
import os
import sys

_here = os.path.abspath(os.path.dirname(__file__))

if sys.version_info[0] < 3:
    with open(os.path.join(_here, 'README.md')) as f:
        long_description = f.read()
else:
    with open(os.path.join(_here, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()

version = {}
with open(os.path.join(_here, 'GPS_Strain', 'version.py')) as f:
    exec(f.read(), version)

setup(
    name='GPS_Strain',
    version=version['__version__'],
    description=('GPS and infinitesimal strain analysis'),
    long_description=long_description,
    author='Luke Pajer',
    author_email='luke.pajer@gmail.com',
    url='https://github.com/The-Geology-Guy/GPS_Strain',
    license='mit',
    packages=['GPS_Strain'],
    install_requires=[
        'pandas',
        'numpy',
        'math',
        'pylab',
        'scipy',
        'cartopy',
        'mpl_toolkits',
        'matplotlib',
        'requests',
        'io',
        'dateutil',
    ],
    include_package_data=True,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',],
    )
