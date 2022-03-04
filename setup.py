from distutils.core import setup

setup(name='torch_trainer',
      version='0.1',
      packages=['torch_trainer'],
      license='Creative Commons Attribution-Noncommercial-Share Alike license',
      description='PyTorch trainer',
      long_description='PyTorch trainer',
      entry_points = {'console_scripts': [
          'torch_trainer=torch_trainer.command_line:main']},
      author='Pablo Lemos',
      author_email='p.lemos@sussex.ac.uk',
      url='https://github.com/Pablo-Lemos/torch_trainer'
      )