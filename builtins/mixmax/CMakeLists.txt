add_library(mixmax INTERFACE)
target_include_directories(mixmax INTERFACE .)
add_library(mixmax::mixmax ALIAS mixmax)
