#include <complex>
#include "root_std_complex.h"

#pragma extra_include "root_std_complex.h";

#pragma create TClass complex<int>+;
#pragma create TClass complex<long>+;
#pragma create TClass complex<float>+;
#pragma create TClass complex<double>+;

#pragma create TClass _root_std_complex<int>+;
#pragma create TClass _root_std_complex<long>+;
#pragma create TClass _root_std_complex<float>+;
#pragma create TClass _root_std_complex<double>+;

#ifdef G__NATIVELONGLONG
// #pragma create TClass complex<long long>+;
#pragma create TClass _root_std_complex<long long>+;
// #pragma create TClass complex<long double>+;
#endif

// IO Rules
// From OSx to ROOT
// There the datamembers are called __re_ and __im_
// #pragma read sourceClass="complex<float>" \
//              targetClass="complex<float>" \
//              checksum=[3671150135] \
//              source="float __re_;" \
//              target="" \
//              code="{ newObj->real(onfile.__re_);}"
//
// #pragma read sourceClass="complex<float>" \
//              targetClass="complex<float>" \
//              checksum=[3671150135] \
//              source="float __im_;" \
//              target="" \
//              code="{ newObj->imag(onfile.__im_);}"
