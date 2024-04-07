def file(*args):
  import sys
  if sys.version_info >= (3,9):
    import importlib.resources as importlib_resources
  else:
    import importlib_resources as importlib_resources
  package = globals()["__package__"]
  files = importlib_resources.files(package)
  return files.joinpath(*args)
