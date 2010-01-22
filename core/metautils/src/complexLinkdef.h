#include <complex>
#ifndef __hpux
using namespace std;
#endif

#pragma create TClass std::complex<int>+;
#pragma create TClass std::complex<long>+;
#pragma create TClass std::complex<float>+;
#pragma create TClass std::complex<double>+;

#ifdef G__NATIVELONGLONG
#pragma create TClass std::complex<long long>+;
// #pragma create TClass std::complex<long double>+;
#endif