from setuptools import setup

setup(
    name='toumei',
    version='0.1',
    packages=['toumei', 'toumei.probe', 'toumei.objectives', 'toumei.parameterization', 'toumei.misc', 'toumei.models'],
    url='https://github.com/LuanAdemi/toumei',
    license='GPL-3.0',
    author='Luan Ademi',
    author_email='luan.ademi@student.kit.edu',
    description='A library for easy feature visualization in pytorch'
)
