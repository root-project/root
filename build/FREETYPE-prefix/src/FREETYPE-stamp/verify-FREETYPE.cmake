# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

if("/home/runner/work/root/root/graf2d/freetype/src/freetype-2.12.1.tar.gz" STREQUAL "")
  message(FATAL_ERROR "LOCAL can't be empty")
endif()

if(NOT EXISTS "/home/runner/work/root/root/graf2d/freetype/src/freetype-2.12.1.tar.gz")
  message(FATAL_ERROR "File not found: /home/runner/work/root/root/graf2d/freetype/src/freetype-2.12.1.tar.gz")
endif()

if("SHA256" STREQUAL "")
  message(WARNING "File cannot be verified since no URL_HASH specified")
  return()
endif()

if("efe71fd4b8246f1b0b1b9bfca13cfff1c9ad85930340c27df469733bbb620938" STREQUAL "")
  message(FATAL_ERROR "EXPECT_VALUE can't be empty")
endif()

message(VERBOSE "verifying file...
     file='/home/runner/work/root/root/graf2d/freetype/src/freetype-2.12.1.tar.gz'")

file("SHA256" "/home/runner/work/root/root/graf2d/freetype/src/freetype-2.12.1.tar.gz" actual_value)

if(NOT "${actual_value}" STREQUAL "efe71fd4b8246f1b0b1b9bfca13cfff1c9ad85930340c27df469733bbb620938")
  message(FATAL_ERROR "error: SHA256 hash of
  /home/runner/work/root/root/graf2d/freetype/src/freetype-2.12.1.tar.gz
does not match expected value
  expected: 'efe71fd4b8246f1b0b1b9bfca13cfff1c9ad85930340c27df469733bbb620938'
    actual: '${actual_value}'
")
endif()

message(VERBOSE "verifying file... done")
