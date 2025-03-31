add_library(
  libintx.hf.guess
  EXCLUDE_FROM_ALL
  guess.cc
)
target_link_libraries(libintx.hf.guess libintx.md4)
