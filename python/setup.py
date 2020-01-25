import setuptools
from distutils.core import setup, Extension

module = Extension(
  'boys',
  language = 'c++',
  sources = ['./boys.cc', '../src/libintx/boys/boys.cc'],
  extra_compile_args = [ '-std=c++14' ],
  include_dirs = [ '../include', '../src/libintx', '../src' ],
)

setup(
  name = 'boys',
  version = '0.1',
  #description = 'boys',
  ext_modules = [module],
  #test_suite = 'tests'
)
