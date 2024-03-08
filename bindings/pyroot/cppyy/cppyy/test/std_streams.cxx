#include "std_streams.h"

#ifndef _WIN32
template class std::basic_ios<char,std::char_traits<char> >;
#endif
