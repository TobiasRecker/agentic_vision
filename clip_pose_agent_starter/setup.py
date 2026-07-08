from glob import glob
from setuptools import find_packages, setup

package_name = "clip_pose_agent_starter"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/configs", glob("configs/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/scripts", glob("scripts/*.sh")),
    ],
    install_requires=[
        "setuptools",
        "numpy>=1.24",
        "opencv-python>=4.8",
        "PyYAML>=6.0",
        "scipy>=1.10",
    ],
    zip_safe=True,
    maintainer="rosmatch",
    maintainer_email="rosmatch@example.com",
    description="Agentic vision tools for multi-view clip capture and pose estimation.",
    license="TODO",
    entry_points={
        "console_scripts": [
            "clip_object_capture_session = clip_pose_pipeline.clip_object_capture_session:main",
        ],
    },
)
