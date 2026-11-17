# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# Patch step of the builtins, see ROOT_BUILTIN_PATCH_COMMAND in RootMacros.cmake:
#
#   cmake -DSOURCE_DIR=<dir> -DSTAMP_DIR=<dir> -P apply_builtin_patches.cmake
#         -- [<git apply option>...] <patch>...
#
# ExternalProject re-runs the patch step not only on a freshly downloaded source
# tree, but also whenever the patch command changes, and then on the tree that
# is already patched. Therefore, the patches are always applied to the pristine
# sources: the first run commits them to a git repository in the source
# directory (unless the sources are a git clone already), and later runs reset
# the tree to that commit before applying the patches again.

set(apply_args)
set(patches)
set(after_separator FALSE)
math(EXPR last_arg "${CMAKE_ARGC} - 1")
foreach(i RANGE ${last_arg})
  set(arg "${CMAKE_ARGV${i}}")
  if(NOT after_separator)
    if(arg STREQUAL "--")
      set(after_separator TRUE)
    endif()
  elseif(arg MATCHES "^-")
    list(APPEND apply_args ${arg})
  else()
    list(APPEND patches ${arg})
  endif()
endforeach()

# Git is looked up here and not passed in the patch command, so that a different
# path to git doesn't change the command, which would re-run the patch step.
find_package(Git REQUIRED)

# Never convert line endings, so that resetting restores the sources as they
# were, and work also in build trees owned by another user.
set(git ${GIT_EXECUTABLE} -c core.autocrlf=false -c core.safecrlf=false -c core.fsmonitor=false
        -c gc.auto=0 -c safe.directory=*)

function(run_git)
  execute_process(COMMAND ${git} ${ARGN} WORKING_DIRECTORY ${SOURCE_DIR} COMMAND_ERROR_IS_FATAL ANY)
endfunction()

# Returns in <result_var> whether the git command succeeded, without output.
function(try_git result_var)
  execute_process(COMMAND ${git} ${ARGN} WORKING_DIRECTORY ${SOURCE_DIR}
                  RESULT_VARIABLE result OUTPUT_QUIET ERROR_QUIET)
  if(result EQUAL 0)
    set(${result_var} TRUE PARENT_SCOPE)
  else()
    set(${result_var} FALSE PARENT_SCOPE)
  endif()
endfunction()

# Left behind if an earlier run was interrupted.
file(REMOVE ${SOURCE_DIR}/.git/index.lock)

set(has_commit FALSE)
if(EXISTS ${SOURCE_DIR}/.git)
  try_git(has_commit rev-parse -q --verify HEAD)
endif()

if(has_commit)
  # Undo the patches of an earlier run.
  run_git(reset -q --hard)
  run_git(clean -q -d -f -f -x)
else()
  # The repository also makes git apply take the paths in the patches relative
  # to the source directory, when the build directory is inside a repository.
  run_git(init -q)

  # The patches might have been applied already without committing the pristine
  # sources, by the patch commands of the builtins before this script was used.
  # Undo those that are there, in reverse order. If this doesn't result in the
  # pristine sources, applying the patches below fails, and the sources are
  # downloaded again with the next build.
  set(reversed_patches ${patches})
  list(REVERSE reversed_patches)
  foreach(patch IN LISTS reversed_patches)
    try_git(reversible apply ${apply_args} --reverse --check ${patch})
    if(reversible)
      run_git(apply ${apply_args} --reverse ${patch})
    endif()
  endforeach()

  run_git(add -A -f)
  run_git(-c user.name=ROOT -c user.email=root@localhost -c commit.gpgsign=false
          commit -q --no-verify --allow-empty -m "Pristine sources")
endif()

# One patch at a time, because git apply would otherwise take the -p level it
# guessed for a patch in the traditional diff format also for the next patches.
foreach(patch IN LISTS patches)
  execute_process(COMMAND ${git} apply ${apply_args} ${patch} WORKING_DIRECTORY ${SOURCE_DIR}
                  RESULT_VARIABLE apply_result)
  if(NOT apply_result EQUAL 0)
    # The commit might not hold the pristine sources. Removing the stamp of the
    # download step makes the next build start again from a new download.
    file(GLOB_RECURSE download_stamps ${STAMP_DIR}/*-download)
    if(download_stamps)
      file(REMOVE ${download_stamps})
      message(FATAL_ERROR "Failed to apply ${patch} to ${SOURCE_DIR}. "
                          "The sources will be downloaded again with the next build.")
    endif()
    message(FATAL_ERROR "Failed to apply ${patch} to ${SOURCE_DIR}. "
                        "Remove that directory to start again from the pristine sources.")
  endif()
endforeach()
