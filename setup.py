from setuptools import setup, find_packages

classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Education',
    'Operating System :: Microsoft :: Windows :: Windows 10',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3'
]

setup(
    name = 'AquaNutriOpt',
    version='2.0',
    description='Finds the optimal placement of BMPs and Technologies in a network in order to reduce the nutrient loadings at the lake with multiple objectives taking different year nutrient loadings in consideration',
    Long_description = open('README.txt').read() + '\n\n' + open('CHANGELOG.txt').read(),
    url='https://github.com/Ashim-Khanal/AquaNutriOpt',
    author='Ashim Khanal',
    author_email = 'ashimkhanal18@gmail.com',
    license = 'MIT',
    classifiers=classifiers,
    keywords='Optimization, Environment, Best Management Practices, Aqua, Technologies, Nutrient Load Reduction, environemnt protection, Engineering, Environmental Modeling, Environmental Software',
    packages=find_packages(),
    install_requires=['']
)

