import setuptools
from distutils.core import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
import os
import subprocess

class CMakeExtension(Extension):
  def __init__(self, name, sourcedir=''):
    Extension.__init__(self, name, sources=[])
    self.sourcedir = os.path.abspath(sourcedir)

class BuildExtension(_build_ext):

  def build_extension(self, ext):
    if isinstance(ext,CMakeExtension):
      return self.build_cmake_extension(ext)
    return _build_ext.build_extension(self,ext)

  def build_cmake_extension(self, ext):

    print("building '%s' CMake extension" % ext.name)

    ext_fullpath = self.get_ext_fullpath(ext.name)
    ext_basename = os.path.basename(ext_fullpath)
    ext_dirname = os.path.dirname(ext_fullpath)

    if not os.path.exists(self.build_temp):
      os.makedirs(self.build_temp)

    cfg = 'Debug' if self.debug else 'Release'
    build_args = ['--config', cfg]
    subprocess.check_call(
      ['cmake', '--build', ".", '--target', '%s-python' % ext.name] + build_args,
      #cwd=self.build_temp
    )

    if not os.path.exists(ext_dirname):
      os.makedirs(ext_dirname)
    self.copy_file(ext_basename, ext_fullpath)
    os.remove(ext_basename)

setup(
  name = 'libintx',
  #packages=['libintx'],
  #package_dir={'': 'src'},
  ext_modules = [ CMakeExtension("boys"), CMakeExtension("libintx") ],
  cmdclass = { "build_ext" : BuildExtension },
  test_suite = 'tests',
)
