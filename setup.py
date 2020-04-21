from setuptools import setup

with open('README.md', 'r') as f:
    long_description = f.read()

version = {}
with open("mlpce/version.py") as f:
    exec(f.read(), version)

setup(name='mlpce',
      version=version['__version__'],
      description='Machine Learning Prediction Confidence Estimation',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Mark Ewing',
      author_email='b.mark@ewingsonline.com',
      url='https://github.com/bmewing/mlpce',
      packages=['mlpce'],
      license='MIT',
      tests_require=['pytest'],
      install_requires=['numpy', 'pandas'],
      keywords=['machine learning', 'prediction', 'prediction variance', 'confidence'],
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
      ]
)