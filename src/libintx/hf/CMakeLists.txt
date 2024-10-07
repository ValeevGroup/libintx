add_library(
  libintx.hf
  EXCLUDE_FROM_ALL
  engine.cc
  )
set_target_properties(libintx.hf PROPERTIES OUTPUT_NAME intx.hf)

add_library(
  libintx.hf.guess
  EXCLUDE_FROM_ALL
  guess.cc
)
target_link_libraries(libintx.hf.guess libintx.md4)

target_link_libraries(
  libintx.hf
  libintx.ao.screening
  libintx.hf.guess
  libintx.md2
  libintx.blas
  )
