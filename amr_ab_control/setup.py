from glob import glob

from setuptools import find_packages, setup

package_name = 'amr_ab_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README_KO.md', 'SYSTEM_REQUIREMENTS.md']),
        ('share/' + package_name + '/maps', glob('maps/*')),
        ('share/' + package_name + '/models', glob('models/*.pt')),
        ('share/' + package_name + '/docs', glob('docs/*')),
        ('lib/' + package_name, ['scripts/record_test.sh']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hv-03',
    maintainer_email='tjsdnrla02@naver.com',
    description='TurtleBot4 AMR A/B patrol + YOLO car approach',
    license='TODO',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'amr_ab_controller = amr_ab_control.amr_ab_controller:main',
            'amr_ab_controller_patrol = amr_ab_control.amr_ab_controller_patrol:main',  # 백업: A/B 순찰 버전
            'webcam_event = amr_ab_control.webcam_event:main',
            'turtle_cam_view = amr_ab_control.turtle_cam_view:main',
            'ab_nav_only = amr_ab_control.ab_nav_only:main',
            'bag_to_mp4 = amr_ab_control.bag_to_mp4:main',
        ],
    },
)
