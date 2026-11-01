from setuptools import setup

package_name = 'wyze_ptz_proxy'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'requests'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='ROS2 Proxy Node for Wyze Cam Pan v3 via docker-wyze-bridge',
    license='MIT',
    entry_points={
        'console_scripts': [
            'wyze_ptz_proxy_node = wyze_ptz_proxy.wyze_ptz_proxy_node:main',
        ],
    },
)
