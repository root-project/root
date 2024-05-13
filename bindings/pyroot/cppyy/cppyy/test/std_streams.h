#ifndef STD_STREAMS_H 
#define STD_STREAMS_H 1

#ifndef __CLING__
#include <ios>
#endif
#include <iostream>

#ifndef __CLING__
#ifndef _WIN32
extern template class std::basic_ios<char,std::char_traits<char> >;
#endif
#endif

#endif // STD_STREAMS_H
