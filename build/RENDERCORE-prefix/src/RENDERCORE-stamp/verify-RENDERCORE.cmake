# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

if("/home/runner/work/root/root/builtins/rendercore/RenderCore-1.7.tar.gz" STREQUAL "")
  message(FATAL_ERROR "LOCAL can't be empty")
endif()

if(NOT EXISTS "/home/runner/work/root/root/builtins/rendercore/RenderCore-1.7.tar.gz")
  message(FATAL_ERROR "File not found: /home/runner/work/root/root/builtins/rendercore/RenderCore-1.7.tar.gz")
endif()

if("SHA256" STREQUAL "")
  message(WARNING "File cannot be verified since no URL_HASH specified")
  return()
endif()

if("46cf6171ae0e16ba2f99789daaeb202146072af874ea530f06a0099c66c3e9b1" STREQUAL "")
  message(FATAL_ERROR "EXPECT_VALUE can't be empty")
endif()

message(VERBOSE "verifying file...
     file='/home/runner/work/root/root/builtins/rendercore/RenderCore-1.7.tar.gz'")

file("SHA256" "/home/runner/work/root/root/builtins/rendercore/RenderCore-1.7.tar.gz" actual_value)

if(NOT "${actual_value}" STREQUAL "46cf6171ae0e16ba2f99789daaeb202146072af874ea530f06a0099c66c3e9b1")
  message(FATAL_ERROR "error: SHA256 hash of
  /home/runner/work/root/root/builtins/rendercore/RenderCore-1.7.tar.gz
does not match expected value
  expected: '46cf6171ae0e16ba2f99789daaeb202146072af874ea530f06a0099c66c3e9b1'
    actual: '${actual_value}'
")
endif()

message(VERBOSE "verifying file... done")
