// lib/dll_stl/cmplx.h

#include <complex>

#ifndef __hpux
using namespace std;
#endif

#ifdef __MAKECINT__

#ifndef G__COMPLEX_DLL
#define G__COMPLEX_DLL
#endif

#pragma link C++ global G__COMPLEX_DLL;

#pragma link C++ class complex<int> ;
#pragma link C++ class complex<unsigned int> ;
#pragma link C++ class complex<long> ;
#pragma link C++ class complex<unsigned long> ;
#pragma link C++ class complex<double> ;

#ifdef G__NATIVELONGLONG
#pragma link C++ class complex<long long> ;
#pragma link C++ class complex<unsigned long long> ;
#pragma link C++ class complex<long double> ;
#endif



#endif
