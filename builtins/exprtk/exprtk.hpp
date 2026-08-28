#ifndef EXPRTK_FWD_H
#define EXPRTK_FWD_H
// See https://github.com/ArashPartow/exprtk
// https://www.partow.net/programming/lexertk/index.html
// libexprtk-dev is a package of Debian, in principle we could install it and pull the header from there
// At the moment, we use however a subpart of that package with just with the interesting functionality
// that is 2k lines instead of 50k lines of single-header-library
// We would also need https://github.com/Kitware/VTK/blob/master/CMake/FindExprTk.cmake
// See also https://www.reddit.com/r/cpp/comments/1758cth/compile_speed_experiment_using_clang_17s_standard/

#include "lexertk.hpp"
namespace exprtk {
namespace lexer = lexertk; // To be drop-in compatible if later we want to use the full exprtk package (header-only
                           // library) as external dependency from package managers
}
#endif
