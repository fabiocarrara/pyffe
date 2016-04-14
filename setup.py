from setuptools import setup

setup(name='pyffe',
      version='0.1',
      description='Tools and utils for PyCaffe',
      # url='http://github.com/fabiocarrara/pyffe',
      author='Fabio Carrara',
      author_email='fabio.carrara@isti.cnr.it',
      license='MIT',
      packages=['pyffe', 'pyffe.models'],
      zip_safe=False,
      requires=['functools32', 'tqdm', 'pandas', 'lmdb', 'caffe']
)
