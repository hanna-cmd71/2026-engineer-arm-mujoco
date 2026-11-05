from setuptools import find_packages, setup


package_name = "jgarm_control"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="hanna-cmd71",
    maintainer_email="2628165913@qq.com",
    description="KDL and Pinocchio controller for JGarm",
    license="Proprietary",
    entry_points={
        "console_scripts": [
            "control_node = jgarm_control.control_node:main",
            "trajectory_bridge_node = jgarm_control.trajectory_bridge_node:main",
            "viewer_node = jgarm_control.viewer_node:main",
        ],
    },
)
