# Figure out if we can track VC revisions.
function(find_first_existing_file out_var)
  foreach(file ${ARGN})
    if(EXISTS "${file}")
      set(${out_var} "${file}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
endfunction()

macro(find_first_existing_vc_file out_var path)
  if ( LLVM_APPEND_VC_REV )
    find_program(git_executable NAMES git git.exe git.cmd)
    # Run from a subdirectory to force git to print an absolute path.
    execute_process(COMMAND ${git_executable} rev-parse --git-dir
      WORKING_DIRECTORY ${path}/cmake
      RESULT_VARIABLE git_result
      OUTPUT_VARIABLE git_dir
      ERROR_QUIET)
    if(git_result EQUAL 0)
      string(STRIP "${git_dir}" git_dir)
      set(${out_var} "${git_dir}/logs/HEAD")
      # some branchless cases (e.g. 'repo') may not yet have .git/logs/HEAD
      if (NOT EXISTS "${git_dir}/logs/HEAD")
        file(WRITE "${git_dir}/logs/HEAD" "")
      endif()
    else()
      find_first_existing_file(${out_var}
        "${path}/.svn/wc.db"   # SVN 1.7
        "${path}/.svn/entries" # SVN 1.6
      )
    endif()
  endif()
endmacro()

find_first_existing_vc_file(llvm_vc "${LLVM_MAIN_SRC_DIR}")

# The VC revision include that we want to generate.
set(version_inc "${CMAKE_CURRENT_BINARY_DIR}/VCSRevision.h")

set(get_svn_script "${LLVM_CMAKE_PATH}/GenerateVersionFromCVS.cmake")

# Disable this version check as it will cause constant rebuilds because
# this code doesn't correctly handle out embedded LLVM.
# This can just be dropped when we upgrade to LLVM 6.0 where this check will
# be disabled by the LLVM_APPEND_VC_REV=OFF we set in interpreter/CMakeLists.txt
#if(DEFINED llvm_vc)
#  # Create custom target to generate the VC revision include.
#  add_custom_command(OUTPUT "${version_inc}"
#    DEPENDS "${llvm_vc}" "${get_svn_script}"
#    COMMAND
#    ${CMAKE_COMMAND} "-DSOURCE_DIR=${LLVM_MAIN_SRC_DIR}"
#                     "-DNAME=LLVM_REVISION"
#                     "-DHEADER_FILE=${version_inc}"
#                     -P "${get_svn_script}")
#
#  # Mark the generated header as being generated.
#  set_source_files_properties("${version_inc}"
#    PROPERTIES GENERATED TRUE
#               HEADER_FILE_ONLY TRUE)
#else()
#  file(WRITE "${version_inc}" "")
#endif()

# Instead of the code above, just write a dummy file if none exists yet. If
# one already exists, we shouldn't touch it to prevent rebuilding.
if (NOT EXISTS "${version_inc}")
  file(WRITE "${version_inc}" "")
endif()

add_custom_target(llvm_vcsrevision_h DEPENDS "${version_inc}")
