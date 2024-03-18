from skbuild import setup
import argparse

import io,os,sys
this_directory = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="slar",
    version="0.1",
    include_package_data=True,
    author=['Carolyn Smith, Sam Young, Ka Vang Tsang, Kazu Terao'],
    description='Sinusoidal Representation Network',
    license='MIT',
    keywords='siren',
    scripts=['bin/train-siren.py','bin/download_icarus_ckpt.sh','bin/update-siren-weight.py','bin/download_2x2_ckpt.sh'],
    packages=['slar'],
    package_data={'slar': ['config/*.yaml']},
    install_requires=[
        'h5py',
        'pyyaml',
        'numpy',
        'scikit-build',
        'torch',
        'photonlib',
        'gdown',
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
)
