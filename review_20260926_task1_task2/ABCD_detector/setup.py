from glob import glob

from setuptools import find_packages, setup


package_name = "abcd_detector"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
    ],
    package_data={"abcd_detector": ["assets/letters/*.png"]},
    install_requires=["setuptools"],
    include_package_data=True,
    zip_safe=True,
    maintainer="cat",
    maintainer_email="cat@lubancat",
    description="OpenCV detector for white A/B/C/D letter blocks",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "abcd_detector_node = abcd_detector.letter_detector_node:main",
        ],
    },
)
