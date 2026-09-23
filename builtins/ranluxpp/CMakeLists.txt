# RANLUX++ is an LCG equivalent of RANLUX using 576 bit numbers.

# Described in
# A. Sibidanov, *A revision of the subtract-with-borrow random numbergenerators*,
# *Computer Physics Communications*, 221(2017), 299-303,
# preprint https://arxiv.org/pdf/1705.03123.pdf

# The code is loosely based on the Assembly implementation by A. Sibidanov
# available at https://github.com/sibidanov/ranluxpp/.

add_library(ranluxpp INTERFACE)
target_include_directories(ranluxpp INTERFACE .)
add_library(ranluxpp::ranluxpp ALIAS ranluxpp)
