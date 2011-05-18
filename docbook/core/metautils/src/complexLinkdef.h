#include <complex>
#ifndef __hpux
using namespace std;
#endif

#pragma create TClass complex<int>+;
#pragma create TClass complex<long>+;
#pragma create TClass complex<float>+;
#pragma create TClass complex<double>+;

#ifdef G__NATIVELONGLONG
#pragma create TClass complex<long long>+;
// #pragma create TClass complex<long double>+;
#endif
