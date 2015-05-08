# Arguments:
#   install_dir
#   source_dir

file(GLOB release_libs  ${source_dir}/build/*_release/lib*)
file(GLOB debug_libs    ${source_dir}/build/*_debug/lib*)

file(INSTALL ${release_libs} DESTINATION ${install_dir}/lib USE_SOURCE_PERMISSIONS)
file(INSTALL ${source_dir}/include/ DESTINATION ${install_dir}/include USE_SOURCE_PERMISSIONS)
