from setuptools import setup

setup(
    name='libmg',
    version='1.0',
    packages=['libmg'],
    url='',
    license='MIT',
    author='Matteo Belenchia',
    author_email='matteo.belenchia@unicam.it',
    description='libmg package',
    install_requires=['lark', 'numpy', 'scipy', 'spektral', 'tensorflow', 'tensorflow-gpu', 'tensorflow_addons']
)
