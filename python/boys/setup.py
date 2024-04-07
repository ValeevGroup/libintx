import setuptools
from distutils.core import setup, Extension
from setuptools.command.build_ext import build_ext
import os
import subprocess

boys = Extension(
  'boys',
  language = 'c++',
  sources = ['./src/boys/boys.cc', '../src/libintx/boys/boys.cc'],
  extra_compile_args = [ '-std=c++14' ],
  include_dirs = [ '../include', './pybind11/include', '../src/libintx', '../src' ],
)
 
# class CMakeExtension(Extension):
#   def __init__(self, name, sourcedir=''):
#     Extension.__init__(self, name, sources=[])
#     self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
  def run(self):
    for ext in self.extensions:
      self.build_extension(ext)

  def build_extension(self, ext):

    print(dir(ext), ext.name)
    
    ext_fullpath = self.get_ext_fullpath(ext.name)
    ext_basename = os.path.basename(ext_fullpath)
    ext_dirname = os.path.dirname(ext_fullpath)

    if not os.path.exists(self.build_temp):
      os.makedirs(self.build_temp)

    cfg = 'Debug' if self.debug else 'Release'
    build_args = ['--config', cfg]
    subprocess.check_call(
      ['cmake', '--build', ".", '--target', 'libintx-python'] + build_args,
      #cwd=self.build_temp
    )

    if not os.path.exists(ext_dirname):
      os.makedirs(ext_dirname)
    self.copy_file(ext_basename, ext_fullpath)
    os.remove(ext_basename)

    print()  # Add an empty line for cleaner output

setup(
  name = 'libintx',
  #packages=['libintx'],
  #package_dir={'': 'src'},
  ext_modules = [ Extension("libintx", sources=[]) ],
  cmdclass = dict(build_ext=CMakeBuild),
  test_suite = 'tests',
)

setup(
  name = 'boys',
  ext_modules = [boys],
  test_suite = 'tests'
)
