from glob import glob

from setuptools import find_packages, setup


package_name = "balls_detector"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    include_package_data=True,
    zip_safe=True,
    maintainer="cat",
    maintainer_email="cat@lubancat",
    description="OpenCV detector for RoboCup red, blue, and yellow balls",
    license="Apache-2.0",
)
