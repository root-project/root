add_library(CDT INTERFACE)
target_include_directories(CDT INTERFACE .)
add_library(CDT::CDT ALIAS CDT)
