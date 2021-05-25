include(RootMacros)

message(DEPRECATION "RootNewMacros.cmake has been renamed to RootMacros.cmake and is deprecated. "
  "Including this file is no longer necessary, as ROOT macros are now available after a call to "
  "find_package(ROOT). Please use 'include(\${ROOT_USE_FILE})' if you still need to inherit "
  "compilation options from ROOT as well as enabling the macros.")
