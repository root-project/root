# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/runner/work/root/root/build/FREETYPE-prefix/src/FREETYPE")
  file(MAKE_DIRECTORY "/home/runner/work/root/root/build/FREETYPE-prefix/src/FREETYPE")
endif()
file(MAKE_DIRECTORY
  "/home/runner/work/root/root/build/FREETYPE-prefix/src/FREETYPE-build"
  "/home/runner/work/root/root/build/FREETYPE-prefix"
  "/home/runner/work/root/root/build/FREETYPE-prefix/tmp"
  "/home/runner/work/root/root/build/FREETYPE-prefix/src/FREETYPE-stamp"
  "/home/runner/work/root/root/build/FREETYPE-prefix/src"
  "/home/runner/work/root/root/build/FREETYPE-prefix/src/FREETYPE-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/runner/work/root/root/build/FREETYPE-prefix/src/FREETYPE-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/runner/work/root/root/build/FREETYPE-prefix/src/FREETYPE-stamp${cfgdir}") # cfgdir has leading slash
endif()
