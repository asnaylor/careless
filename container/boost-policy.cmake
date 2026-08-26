# DIALS 3.30 predates CMake's removal of FindBoost. Use its compatibility
# finder so we can point at the Boost ABI bundled by the cctbx wheel.
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 OLD)
endif()

# FindBoost exposes the versioned Python target, while DIALS and dxtbx link
# the generic target normally supplied by BoostConfig.cmake.
if(NOT TARGET Boost::python)
  add_library(Boost::python INTERFACE IMPORTED)
  set_target_properties(
    Boost::python
    PROPERTIES INTERFACE_LINK_LIBRARIES Boost::python312
  )
endif()

# The cctbx wheel's Boost.Thread library has no dynamic chrono or atomic
# dependency, although FindBoost still names these optional interface targets.
foreach(component chrono atomic)
  if(NOT TARGET Boost::${component})
    add_library(Boost::${component} INTERFACE IMPORTED)
  endif()
endforeach()
