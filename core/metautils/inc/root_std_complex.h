/*************************************************************************
 * This file defines a complex class which has an layout identical to the one
 * of ROOT5. The file was located in cint/lib/prec_stl/complex
 * This class is used to provide to the ROOT6 typesystem a backward compatible
 * and platform independent information for the complex numbers.
 *
 ************************************************************************/


#ifndef _ROOT_STD_COMPLEX_INCLUDED
#define _ROOT_STD_COMPLEX_INCLUDED

#include <complex>

template<class T> class _root_std_complex {
    T _real;
    T _imag;
};

// Asserts about the size of the complex
static_assert(sizeof(_root_std_complex<double>) == sizeof(complex<double>),
              "The size of complex<T> and _root_std_complex<T> do not match!");
static_assert(sizeof(_root_std_complex<float>) == sizeof(complex<float>),
              "The size of complex<T> and _root_std_complex<T> do not match!");
static_assert(sizeof(_root_std_complex<long>) == sizeof(complex<long>),
              "The size of complex<T> and _root_std_complex<T> do not match!");
static_assert(sizeof(_root_std_complex<int>) == sizeof(complex<int>),
              "The size of complex<T> and _root_std_complex<T> do not match!");

#endif
