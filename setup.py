from setuptools import setup, find_packages

setup(
    name='como_tracks',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
    ],
    author='Sleiman Safaoui',
    author_email='sleiman.safaoui@utdallas.edu',
    description='Tracks for the COMO autonomous car platform at UTDallas',
    url='https://github.com/The-SS/como_tracks',
)
