/* The -*- C++ -*- 2D, 3D, 4D Vector and 2x2, 3x3, 4x4 Matrix classes.
   (Based on the "Graphics Gems IV", Edited by Paul S. Heckbert, Academic
   Press, 1994, ISBN 0-12-336156-9, original source code files algebra3.h
   and algebra3aux.h by Jean-Francois Doue and John Nagle.
   Modified by Jacek M. Holeczek, 05.2001, 01.2011.)
   You are free to use and modify this code in any way you like.
   No warranty of any kind. No liability for any defects or damages. */

#ifndef __ALGEBRA3_CXX__
#define __ALGEBRA3_CXX__

#ifndef __ALGEBRA3_CXX_DEBUG__
//#define __ALGEBRA3_CXX_DEBUG__ 1
#endif /* __ALGEBRA3_CXX_DEBUG__ */

#include "algebra3"

void runalgebra (void) { /* Dummy ROOT initialization routine */ }

#include "RConfigure.h"

#if defined(__MAKECINT__) || defined(__ALGEBRA3_CXX_DEBUG__)

#ifdef __FIX_MAKROSCHROTT__
#undef __FIX_MAKROSCHROTT__
#endif /* __FIX_MAKROSCHROTT__ */

#ifdef _MSC_VER
#define __FIX_MAKROSCHROTT__ 1
#else /* _MSC_VER */
#define __FIX_MAKROSCHROTT__ 0
#endif /* _MSC_VER */



/* Namespace aux */



#ifdef __MAKECINT__

#pragma link C++ namespace aux;

#endif /* __MAKECINT__ */



/* Enum types */



#ifdef __MAKECINT__

#pragma link C++ enum aux::EAxisXYZW;
#pragma link C++ enum aux::EAxis1234;
#pragma link C++ enum aux::EPlaneABCD;
#pragma link C++ enum aux::EColorRGB;
#pragma link C++ enum aux::EPhongADSS;

#endif /* __MAKECINT__ */



/* RTTI class of < long double > */



namespace aux {

  template class rtti< long double >; // RTTI of < long double >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< long double >+;

#endif /* __MAKECINT__ */



/* RTTI class of < vec2< long double > > */



namespace aux {

  template class rtti< vec2< long double > >; // RTTI of < vec2< long double > >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< aux::vec2< long double > >+;

#endif /* __MAKECINT__ */



/* 2D Vector class and friends of < long double > */



namespace aux {

  template class vec2< long double >; // 2D Vector of < long double >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::vec2< long double >+;

#endif /* __MAKECINT__ */

namespace aux {

  template vec2< long double > operator - (const vec2< long double >& v); // -v1
  template vec2< long double > operator + (const vec2< long double >& a, const vec2< long double >& b); // v1 + v2
  template vec2< long double > operator - (const vec2< long double >& a, const vec2< long double >& b); // v1 - v2
  template vec2< long double > operator * (const vec2< long double >& a, const long double d); // v1 * 3.0
  template vec2< long double > operator * (const long double d, const vec2< long double >& a); // 3.0 * v1
  template vec2< long double > operator / (const vec2< long double >& a, const long double d); // v1 / 3.0
  template vec2< long double > operator * (const mat2< long double >& a, const vec2< long double >& v); // linear transform
  template vec2< long double > operator * (const mat3< long double >& a, const vec2< long double >& v); // M . v
  template vec2< long double > operator * (const vec2< long double >& v, const mat3< long double >& a); // v . M
  template vec3< long double > operator ^ (const vec2< long double >& a, const vec2< long double >& b); // cross product
  template long double operator * (const vec2< long double >& a, const vec2< long double >& b); // dot product
  template bool operator == (const vec2< long double >& a, const vec2< long double >& b); // v1 == v2 ?
  template bool operator != (const vec2< long double >& a, const vec2< long double >& b); // v1 != v2 ?

  template ostream& operator << (ostream& s, const vec2< long double >& v); // output to stream
  template istream& operator >> (istream& s, vec2< long double >& v); // input from stream

  template void swap(vec2< long double >& a, vec2< long double >& b); // swap v1 & v2
  template vec2< long double > min(const vec2< long double >& a, const vec2< long double >& b); // min(v1, v2)
  template vec2< long double > max(const vec2< long double >& a, const vec2< long double >& b); // max(v1, v2)
  template vec2< long double > prod(const vec2< long double >& a, const vec2< long double >& b); // term by term *
  template vec2< long double > conj(const vec2< long double >& a);

} /* namespace aux */



/* RTTI class of < vec3< long double > > */



namespace aux {

  template class rtti< vec3< long double > >; // RTTI of < vec3< long double > >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< aux::vec3< long double > >+;

#endif /* __MAKECINT__ */



/* 3D Vector class and friends of < long double > */



namespace aux {

  template class vec3< long double >; // 3D Vector of < long double >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::vec3< long double >+;

#endif /* __MAKECINT__ */

namespace aux {

  template vec3< long double > operator - (const vec3< long double >& v); // -v1
  template vec3< long double > operator + (const vec3< long double >& a, const vec3< long double >& b); // v1 + v2
  template vec3< long double > operator - (const vec3< long double >& a, const vec3< long double >& b); // v1 - v2
  template vec3< long double > operator * (const vec3< long double >& a, const long double d); // v1 * 3.0
  template vec3< long double > operator * (const long double d, const vec3< long double >& a); // 3.0 * v1
  template vec3< long double > operator / (const vec3< long double >& a, const long double d); // v1 / 3.0
  template vec3< long double > operator * (const mat3< long double >& a, const vec3< long double >& v); // linear transform
  template vec3< long double > operator * (const mat4< long double >& a, const vec3< long double >& v); // M . v
  template vec3< long double > operator * (const vec3< long double >& v, const mat4< long double >& a); // v . M
  template vec3< long double > operator ^ (const vec3< long double >& a, const vec3< long double >& b); // cross product
  template long double operator * (const vec3< long double >& a, const vec3< long double >& b); // dot product
  template bool operator == (const vec3< long double >& a, const vec3< long double >& b); // v1 == v2 ?
  template bool operator != (const vec3< long double >& a, const vec3< long double >& b); // v1 != v2 ?

  template ostream& operator << (ostream& s, const vec3< long double >& v); // output to stream
  template istream& operator >> (istream& s, vec3< long double >& v); // input from stream

  template void swap(vec3< long double >& a, vec3< long double >& b); // swap v1 & v2
  template vec3< long double > min(const vec3< long double >& a, const vec3< long double >& b); // min(v1, v2)
  template vec3< long double > max(const vec3< long double >& a, const vec3< long double >& b); // max(v1, v2)
  template vec3< long double > prod(const vec3< long double >& a, const vec3< long double >& b); // term by term *
  template vec3< long double > conj(const vec3< long double >& a);

} /* namespace aux */



/* RTTI class of < vec4< long double > > */



namespace aux {

  template class rtti< vec4< long double > >; // RTTI of < vec4< long double > >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< aux::vec4< long double > >+;

#endif /* __MAKECINT__ */



/* 4D Vector class and friends of < long double > */



namespace aux {

  template class vec4< long double >; // 4D Vector of < long double >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::vec4< long double >+;

#endif /* __MAKECINT__ */

namespace aux {

  template vec4< long double > operator - (const vec4< long double >& v); // -v1
  template vec4< long double > operator + (const vec4< long double >& a, const vec4< long double >& b); // v1 + v2
  template vec4< long double > operator - (const vec4< long double >& a, const vec4< long double >& b); // v1 - v2
  template vec4< long double > operator * (const vec4< long double >& a, const long double d); // v1 * 3.0
  template vec4< long double > operator * (const long double d, const vec4< long double >& a); // 3.0 * v1
  template vec4< long double > operator / (const vec4< long double >& a, const long double d); // v1 / 3.0
  template vec4< long double > operator * (const mat4< long double >& a, const vec4< long double >& v); // M . v
  template vec4< long double > operator * (const vec4< long double >& v, const mat4< long double >& a); // v . M
  template long double operator * (const vec4< long double >& a, const vec4< long double >& b); // dot product
  template bool operator == (const vec4< long double >& a, const vec4< long double >& b); // v1 == v2 ?
  template bool operator != (const vec4< long double >& a, const vec4< long double >& b); // v1 != v2 ?

  template ostream& operator << (ostream& s, const vec4< long double >& v); // output to stream
  template istream& operator >> (istream& s, vec4< long double >& v); // input from stream

  template void swap(vec4< long double >& a, vec4< long double >& b); // swap v1 & v2
  template vec4< long double > min(const vec4< long double >& a, const vec4< long double >& b); // min(v1, v2)
  template vec4< long double > max(const vec4< long double >& a, const vec4< long double >& b); // max(v1, v2)
  template vec4< long double > prod(const vec4< long double >& a, const vec4< long double >& b); // term by term *
  template vec4< long double > conj(const vec4< long double >& a);

} /* namespace aux */



/* RTTI class of < mat2< long double > > */



namespace aux {

  template class rtti< mat2< long double > >; // RTTI of < mat2< long double > >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< aux::mat2< long double > >+;

#endif /* __MAKECINT__ */



/* 2x2 Matrix class and friends of < long double > */



namespace aux {

  template class mat2< long double >; // 2x2 Matrix of < long double >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::mat2< long double >+;

#endif /* __MAKECINT__ */

namespace aux {

  template mat2< long double > operator - (const mat2< long double >& a); // -m1
  template mat2< long double > operator + (const mat2< long double >& a, const mat2< long double >& b); // m1 + m2
  template mat2< long double > operator - (const mat2< long double >& a, const mat2< long double >& b); // m1 - m2
  template mat2< long double > operator * (const mat2< long double >& a, const mat2< long double >& b); // m1 * m2
  template mat2< long double > operator * (const mat2< long double >& a, const long double d); // m1 * 3.0
  template mat2< long double > operator * (const long double d, const mat2< long double >& a); // 3.0 * m1
  template mat2< long double > operator / (const mat2< long double >& a, const long double d); // m1 / 3.0
  template bool operator == (const mat2< long double >& a, const mat2< long double >& b); // m1 == m2 ?
  template bool operator != (const mat2< long double >& a, const mat2< long double >& b); // m1 != m2 ?

  template ostream& operator << (ostream& s, const mat2< long double >& m); // output to stream
  template istream& operator >> (istream& s, mat2< long double >& m); // input from stream

  template void swap(mat2< long double >& a, mat2< long double >& b); // swap m1 & m2
  template mat2< long double > conj(const mat2< long double >& a);
  template mat2< long double > diagonal(const vec2< long double >& v);
  template mat2< long double > diagonal(const long double x0, const long double y1);
  template long double trace(const mat2< long double >& a);

} /* namespace aux */



/* RTTI class of < mat3< long double > > */



namespace aux {

  template class rtti< mat3< long double > >; // RTTI of < mat3< long double > >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< aux::mat3< long double > >+;

#endif /* __MAKECINT__ */



/* 3x3 Matrix class and friends of < long double > */



namespace aux {

  template class mat3< long double >; // 3x3 Matrix of < long double >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::mat3< long double >+;

#endif /* __MAKECINT__ */

namespace aux {

  template mat3< long double > operator - (const mat3< long double >& a); // -m1
  template mat3< long double > operator + (const mat3< long double >& a, const mat3< long double >& b); // m1 + m2
  template mat3< long double > operator - (const mat3< long double >& a, const mat3< long double >& b); // m1 - m2
  template mat3< long double > operator * (const mat3< long double >& a, const mat3< long double >& b); // m1 * m2
  template mat3< long double > operator * (const mat3< long double >& a, const long double d); // m1 * 3.0
  template mat3< long double > operator * (const long double d, const mat3< long double >& a); // 3.0 * m1
  template mat3< long double > operator / (const mat3< long double >& a, const long double d); // m1 / 3.0
  template bool operator == (const mat3< long double >& a, const mat3< long double >& b); // m1 == m2 ?
  template bool operator != (const mat3< long double >& a, const mat3< long double >& b); // m1 != m2 ?

  template ostream& operator << (ostream& s, const mat3< long double >& m); // output to stream
  template istream& operator >> (istream& s, mat3< long double >& m); // input from stream

  template void swap(mat3< long double >& a, mat3< long double >& b); // swap m1 & m2
  template mat3< long double > conj(const mat3< long double >& a);
  template mat3< long double > diagonal(const vec3< long double >& v);
  template mat3< long double > diagonal(const long double x0, const long double y1, const long double z2);
  template long double trace(const mat3< long double >& a);

} /* namespace aux */



/* RTTI class of < mat4< long double > > */



namespace aux {

  template class rtti< mat4< long double > >; // RTTI of < mat4< long double > >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< aux::mat4< long double > >+;

#endif /* __MAKECINT__ */



/* 4x4 Matrix class and friends of < long double > */



namespace aux {

  template class mat4< long double >; // 4x4 Matrix of < long double >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::mat4< long double >+;

#endif /* __MAKECINT__ */

namespace aux {

  template mat4< long double > operator - (const mat4< long double >& a); // -m1
  template mat4< long double > operator + (const mat4< long double >& a, const mat4< long double >& b); // m1 + m2
  template mat4< long double > operator - (const mat4< long double >& a, const mat4< long double >& b); // m1 - m2
  template mat4< long double > operator * (const mat4< long double >& a, const mat4< long double >& b); // m1 * m2
  template mat4< long double > operator * (const mat4< long double >& a, const long double d); // m1 * 3.0
  template mat4< long double > operator * (const long double d, const mat4< long double >& a); // 3.0 * m1
  template mat4< long double > operator / (const mat4< long double >& a, const long double d); // m1 / 3.0
  template bool operator == (const mat4< long double >& a, const mat4< long double >& b); // m1 == m2 ?
  template bool operator != (const mat4< long double >& a, const mat4< long double >& b); // m1 != m2 ?

  template ostream& operator << (ostream& s, const mat4< long double >& m); // output to stream
  template istream& operator >> (istream& s, mat4< long double >& m); // input from stream

  template void swap(mat4< long double >& a, mat4< long double >& b); // swap m1 & m2
  template mat4< long double > conj(const mat4< long double >& a);
  template mat4< long double > diagonal(const vec4< long double >& v);
  template mat4< long double > diagonal(const long double x0, const long double y1, const long double z2, const long double w3);
  template long double trace(const mat4< long double >& a);

} /* namespace aux */



/* 2D functions and 3D functions of < long double > */



namespace aux {

  template mat2< long double > identity1D< long double >(void); // identity 1D
  template mat2< long double > translation1D(const long double & v); // translation 1D
  template mat2< long double > scaling1D(const long double & scaleVal); // scaling 1D
  template mat3< long double > identity2D< long double >(void); // identity 2D
  template mat3< long double > translation2D(const vec2< long double >& v); // translation 2D
#if __FIX_MAKROSCHROTT__ == 0
  template mat3< long double > rotation2D(const vec2< long double >& Center, const rtti< long double >::value_type angleDeg); // rotation 2D
#endif /* __FIX_MAKROSCHROTT__ */
  template mat3< long double > scaling2D(const vec2< long double >& scaleVec); // scaling 2D
  template mat4< long double > identity3D< long double >(void); // identity 3D
  template mat4< long double > translation3D(const vec3< long double >& v); // translation 3D
#if __FIX_MAKROSCHROTT__ == 0
  template mat4< long double > rotation3D(vec3< long double > Axis, const rtti< long double >::value_type angleDeg); // rotation 3D
#endif /* __FIX_MAKROSCHROTT__ */
  template mat4< long double > scaling3D(const vec3< long double >& scaleVec); // scaling 3D
  template mat4< long double > perspective3D(const long double d); // perspective 3D

} /* namespace aux */



/* AUX functions of < long double > */



namespace aux {

  template vec2< long double > MultiplyElementwise(const vec2< long double >& v1, const vec2< long double >& v2); // element by element multiplication
  template vec3< long double > MultiplyElementwise(const vec3< long double >& v1, const vec3< long double >& v2); // element by element multiplication
  template vec4< long double > MultiplyElementwise(const vec4< long double >& v1, const vec4< long double >& v2); // element by element multiplication

  template mat3< long double > rotation2D(const mat2< long double >& m); // convert 2x2 rotation matrix to 3x3
  template vec2< long double > ExtractTranslation(const mat3< long double >& pose); // extract translation vector
  template vec2< long double > ExtractScaling(const mat3< long double >& mat);
  template mat2< long double > ExtractRotation(const mat3< long double >& pose); // extract rotation matrix from transformation matrix
#if __FIX_MAKROSCHROTT__ == 0
  template rtti< long double >::value_type PointToLine(const vec2< long double >& point, const vec3< long double >& line); // unsigned distance from a point to a line (2D)
#endif /* __FIX_MAKROSCHROTT__ */
  template mat4< long double > rotation3D(const mat3< long double >& m); // convert 3x3 rotation matrix to 4x4
  template vec3< long double > ExtractTranslation(const mat4< long double >& pose); // extract translation vector
  template vec3< long double > ExtractScaling(const mat4< long double >& mat);
  template mat3< long double > ExtractRotation(const mat4< long double >& pose); // extract rotation matrix from transformation matrix
#if __FIX_MAKROSCHROTT__ == 0
  template rtti< long double >::value_type PointToPlane(const vec3< long double >& point, const vec4< long double >& plane); // unsigned distance from a point to a plane (3D)
#endif /* __FIX_MAKROSCHROTT__ */

  template long double fmin(long double x, long double y);
  template long double fmax(long double x, long double y);

  template vec2< long double > fmin(const vec2< long double >& v1, const vec2< long double >& v2);
  template vec2< long double > fmax(const vec2< long double >& v1, const vec2< long double >& v2);

  template vec3< long double > fmin(const vec3< long double >& v1, const vec3< long double >& v2);
  template vec3< long double > fmax(const vec3< long double >& v1, const vec3< long double >& v2);

  template vec4< long double > fmin(const vec4< long double >& v1, const vec4< long double >& v2);
  template vec4< long double > fmax(const vec4< long double >& v1, const vec4< long double >& v2);

} /* namespace aux */



/* RTTI class of < float > */



namespace aux {

  template class rtti< float >; // RTTI of < float >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< float >+;

#endif /* __MAKECINT__ */



/* RTTI class of < vec2< float > > */



namespace aux {

  template class rtti< vec2< float > >; // RTTI of < vec2< float > >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< aux::vec2< float > >+;

#endif /* __MAKECINT__ */



/* 2D Vector class and friends of < float > */



namespace aux {

  template class vec2< float >; // 2D Vector of < float >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::vec2< float >+;

#endif /* __MAKECINT__ */

namespace aux {

  template vec2< float > operator - (const vec2< float >& v); // -v1
  template vec2< float > operator + (const vec2< float >& a, const vec2< float >& b); // v1 + v2
  template vec2< float > operator - (const vec2< float >& a, const vec2< float >& b); // v1 - v2
  template vec2< float > operator * (const vec2< float >& a, const float d); // v1 * 3.0
  template vec2< float > operator * (const float d, const vec2< float >& a); // 3.0 * v1
  template vec2< float > operator / (const vec2< float >& a, const float d); // v1 / 3.0
  template vec2< float > operator * (const mat2< float >& a, const vec2< float >& v); // linear transform
  template vec2< float > operator * (const mat3< float >& a, const vec2< float >& v); // M . v
  template vec2< float > operator * (const vec2< float >& v, const mat3< float >& a); // v . M
  template vec3< float > operator ^ (const vec2< float >& a, const vec2< float >& b); // cross product
  template float operator * (const vec2< float >& a, const vec2< float >& b); // dot product
  template bool operator == (const vec2< float >& a, const vec2< float >& b); // v1 == v2 ?
  template bool operator != (const vec2< float >& a, const vec2< float >& b); // v1 != v2 ?

  template ostream& operator << (ostream& s, const vec2< float >& v); // output to stream
  template istream& operator >> (istream& s, vec2< float >& v); // input from stream

  template void swap(vec2< float >& a, vec2< float >& b); // swap v1 & v2
  template vec2< float > min(const vec2< float >& a, const vec2< float >& b); // min(v1, v2)
  template vec2< float > max(const vec2< float >& a, const vec2< float >& b); // max(v1, v2)
  template vec2< float > prod(const vec2< float >& a, const vec2< float >& b); // term by term *
  template vec2< float > conj(const vec2< float >& a);

} /* namespace aux */



/* RTTI class of < vec3< float > > */



namespace aux {

  template class rtti< vec3< float > >; // RTTI of < vec3< float > >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< aux::vec3< float > >+;

#endif /* __MAKECINT__ */



/* 3D Vector class and friends of < float > */



namespace aux {

  template class vec3< float >; // 3D Vector of < float >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::vec3< float >+;

#endif /* __MAKECINT__ */

namespace aux {

  template vec3< float > operator - (const vec3< float >& v); // -v1
  template vec3< float > operator + (const vec3< float >& a, const vec3< float >& b); // v1 + v2
  template vec3< float > operator - (const vec3< float >& a, const vec3< float >& b); // v1 - v2
  template vec3< float > operator * (const vec3< float >& a, const float d); // v1 * 3.0
  template vec3< float > operator * (const float d, const vec3< float >& a); // 3.0 * v1
  template vec3< float > operator / (const vec3< float >& a, const float d); // v1 / 3.0
  template vec3< float > operator * (const mat3< float >& a, const vec3< float >& v); // linear transform
  template vec3< float > operator * (const mat4< float >& a, const vec3< float >& v); // M . v
  template vec3< float > operator * (const vec3< float >& v, const mat4< float >& a); // v . M
  template vec3< float > operator ^ (const vec3< float >& a, const vec3< float >& b); // cross product
  template float operator * (const vec3< float >& a, const vec3< float >& b); // dot product
  template bool operator == (const vec3< float >& a, const vec3< float >& b); // v1 == v2 ?
  template bool operator != (const vec3< float >& a, const vec3< float >& b); // v1 != v2 ?

  template ostream& operator << (ostream& s, const vec3< float >& v); // output to stream
  template istream& operator >> (istream& s, vec3< float >& v); // input from stream

  template void swap(vec3< float >& a, vec3< float >& b); // swap v1 & v2
  template vec3< float > min(const vec3< float >& a, const vec3< float >& b); // min(v1, v2)
  template vec3< float > max(const vec3< float >& a, const vec3< float >& b); // max(v1, v2)
  template vec3< float > prod(const vec3< float >& a, const vec3< float >& b); // term by term *
  template vec3< float > conj(const vec3< float >& a);

} /* namespace aux */



/* RTTI class of < vec4< float > > */



namespace aux {

  template class rtti< vec4< float > >; // RTTI of < vec4< float > >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< aux::vec4< float > >+;

#endif /* __MAKECINT__ */



/* 4D Vector class and friends of < float > */



namespace aux {

  template class vec4< float >; // 4D Vector of < float >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::vec4< float >+;

#endif /* __MAKECINT__ */

namespace aux {

  template vec4< float > operator - (const vec4< float >& v); // -v1
  template vec4< float > operator + (const vec4< float >& a, const vec4< float >& b); // v1 + v2
  template vec4< float > operator - (const vec4< float >& a, const vec4< float >& b); // v1 - v2
  template vec4< float > operator * (const vec4< float >& a, const float d); // v1 * 3.0
  template vec4< float > operator * (const float d, const vec4< float >& a); // 3.0 * v1
  template vec4< float > operator / (const vec4< float >& a, const float d); // v1 / 3.0
  template vec4< float > operator * (const mat4< float >& a, const vec4< float >& v); // M . v
  template vec4< float > operator * (const vec4< float >& v, const mat4< float >& a); // v . M
  template float operator * (const vec4< float >& a, const vec4< float >& b); // dot product
  template bool operator == (const vec4< float >& a, const vec4< float >& b); // v1 == v2 ?
  template bool operator != (const vec4< float >& a, const vec4< float >& b); // v1 != v2 ?

  template ostream& operator << (ostream& s, const vec4< float >& v); // output to stream
  template istream& operator >> (istream& s, vec4< float >& v); // input from stream

  template void swap(vec4< float >& a, vec4< float >& b); // swap v1 & v2
  template vec4< float > min(const vec4< float >& a, const vec4< float >& b); // min(v1, v2)
  template vec4< float > max(const vec4< float >& a, const vec4< float >& b); // max(v1, v2)
  template vec4< float > prod(const vec4< float >& a, const vec4< float >& b); // term by term *
  template vec4< float > conj(const vec4< float >& a);

} /* namespace aux */



/* RTTI class of < mat2< float > > */



namespace aux {

  template class rtti< mat2< float > >; // RTTI of < mat2< float > >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< aux::mat2< float > >+;

#endif /* __MAKECINT__ */



/* 2x2 Matrix class and friends of < float > */



namespace aux {

  template class mat2< float >; // 2x2 Matrix of < float >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::mat2< float >+;

#endif /* __MAKECINT__ */

namespace aux {

  template mat2< float > operator - (const mat2< float >& a); // -m1
  template mat2< float > operator + (const mat2< float >& a, const mat2< float >& b); // m1 + m2
  template mat2< float > operator - (const mat2< float >& a, const mat2< float >& b); // m1 - m2
  template mat2< float > operator * (const mat2< float >& a, const mat2< float >& b); // m1 * m2
  template mat2< float > operator * (const mat2< float >& a, const float d); // m1 * 3.0
  template mat2< float > operator * (const float d, const mat2< float >& a); // 3.0 * m1
  template mat2< float > operator / (const mat2< float >& a, const float d); // m1 / 3.0
  template bool operator == (const mat2< float >& a, const mat2< float >& b); // m1 == m2 ?
  template bool operator != (const mat2< float >& a, const mat2< float >& b); // m1 != m2 ?

  template ostream& operator << (ostream& s, const mat2< float >& m); // output to stream
  template istream& operator >> (istream& s, mat2< float >& m); // input from stream

  template void swap(mat2< float >& a, mat2< float >& b); // swap m1 & m2
  template mat2< float > conj(const mat2< float >& a);
  template mat2< float > diagonal(const vec2< float >& v);
  template mat2< float > diagonal(const float x0, const float y1);
  template float trace(const mat2< float >& a);

} /* namespace aux */



/* RTTI class of < mat3< float > > */



namespace aux {

  template class rtti< mat3< float > >; // RTTI of < mat3< float > >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< aux::mat3< float > >+;

#endif /* __MAKECINT__ */



/* 3x3 Matrix class and friends of < float > */



namespace aux {

  template class mat3< float >; // 3x3 Matrix of < float >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::mat3< float >+;

#endif /* __MAKECINT__ */

namespace aux {

  template mat3< float > operator - (const mat3< float >& a); // -m1
  template mat3< float > operator + (const mat3< float >& a, const mat3< float >& b); // m1 + m2
  template mat3< float > operator - (const mat3< float >& a, const mat3< float >& b); // m1 - m2
  template mat3< float > operator * (const mat3< float >& a, const mat3< float >& b); // m1 * m2
  template mat3< float > operator * (const mat3< float >& a, const float d); // m1 * 3.0
  template mat3< float > operator * (const float d, const mat3< float >& a); // 3.0 * m1
  template mat3< float > operator / (const mat3< float >& a, const float d); // m1 / 3.0
  template bool operator == (const mat3< float >& a, const mat3< float >& b); // m1 == m2 ?
  template bool operator != (const mat3< float >& a, const mat3< float >& b); // m1 != m2 ?

  template ostream& operator << (ostream& s, const mat3< float >& m); // output to stream
  template istream& operator >> (istream& s, mat3< float >& m); // input from stream

  template void swap(mat3< float >& a, mat3< float >& b); // swap m1 & m2
  template mat3< float > conj(const mat3< float >& a);
  template mat3< float > diagonal(const vec3< float >& v);
  template mat3< float > diagonal(const float x0, const float y1, const float z2);
  template float trace(const mat3< float >& a);

} /* namespace aux */



/* RTTI class of < mat4< float > > */



namespace aux {

  template class rtti< mat4< float > >; // RTTI of < mat4< float > >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< aux::mat4< float > >+;

#endif /* __MAKECINT__ */



/* 4x4 Matrix class and friends of < float > */



namespace aux {

  template class mat4< float >; // 4x4 Matrix of < float >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::mat4< float >+;

#endif /* __MAKECINT__ */

namespace aux {

  template mat4< float > operator - (const mat4< float >& a); // -m1
  template mat4< float > operator + (const mat4< float >& a, const mat4< float >& b); // m1 + m2
  template mat4< float > operator - (const mat4< float >& a, const mat4< float >& b); // m1 - m2
  template mat4< float > operator * (const mat4< float >& a, const mat4< float >& b); // m1 * m2
  template mat4< float > operator * (const mat4< float >& a, const float d); // m1 * 3.0
  template mat4< float > operator * (const float d, const mat4< float >& a); // 3.0 * m1
  template mat4< float > operator / (const mat4< float >& a, const float d); // m1 / 3.0
  template bool operator == (const mat4< float >& a, const mat4< float >& b); // m1 == m2 ?
  template bool operator != (const mat4< float >& a, const mat4< float >& b); // m1 != m2 ?

  template ostream& operator << (ostream& s, const mat4< float >& m); // output to stream
  template istream& operator >> (istream& s, mat4< float >& m); // input from stream

  template void swap(mat4< float >& a, mat4< float >& b); // swap m1 & m2
  template mat4< float > conj(const mat4< float >& a);
  template mat4< float > diagonal(const vec4< float >& v);
  template mat4< float > diagonal(const float x0, const float y1, const float z2, const float w3);
  template float trace(const mat4< float >& a);

} /* namespace aux */



/* 2D functions and 3D functions of < float > */



namespace aux {

  template mat2< float > identity1D< float >(void); // identity 1D
  template mat2< float > translation1D(const float & v); // translation 1D
  template mat2< float > scaling1D(const float & scaleVal); // scaling 1D
  template mat3< float > identity2D< float >(void); // identity 2D
  template mat3< float > translation2D(const vec2< float >& v); // translation 2D
#if __FIX_MAKROSCHROTT__ == 0
  template mat3< float > rotation2D(const vec2< float >& Center, const rtti< float >::value_type angleDeg); // rotation 2D
#endif /* __FIX_MAKROSCHROTT__ */
  template mat3< float > scaling2D(const vec2< float >& scaleVec); // scaling 2D
  template mat4< float > identity3D< float >(void); // identity 3D
  template mat4< float > translation3D(const vec3< float >& v); // translation 3D
#if __FIX_MAKROSCHROTT__ == 0
  template mat4< float > rotation3D(vec3< float > Axis, const rtti< float >::value_type angleDeg); // rotation 3D
#endif /* __FIX_MAKROSCHROTT__ */
  template mat4< float > scaling3D(const vec3< float >& scaleVec); // scaling 3D
  template mat4< float > perspective3D(const float d); // perspective 3D

} /* namespace aux */



/* AUX functions of < float > */



namespace aux {

  template vec2< float > MultiplyElementwise(const vec2< float >& v1, const vec2< float >& v2); // element by element multiplication
  template vec3< float > MultiplyElementwise(const vec3< float >& v1, const vec3< float >& v2); // element by element multiplication
  template vec4< float > MultiplyElementwise(const vec4< float >& v1, const vec4< float >& v2); // element by element multiplication

  template mat3< float > rotation2D(const mat2< float >& m); // convert 2x2 rotation matrix to 3x3
  template vec2< float > ExtractTranslation(const mat3< float >& pose); // extract translation vector
  template vec2< float > ExtractScaling(const mat3< float >& mat);
  template mat2< float > ExtractRotation(const mat3< float >& pose); // extract rotation matrix from transformation matrix
#if __FIX_MAKROSCHROTT__ == 0
  template rtti< float >::value_type PointToLine(const vec2< float >& point, const vec3< float >& line); // unsigned distance from a point to a line (2D)
#endif /* __FIX_MAKROSCHROTT__ */
  template mat4< float > rotation3D(const mat3< float >& m); // convert 3x3 rotation matrix to 4x4
  template vec3< float > ExtractTranslation(const mat4< float >& pose); // extract translation vector
  template vec3< float > ExtractScaling(const mat4< float >& mat);
  template mat3< float > ExtractRotation(const mat4< float >& pose); // extract rotation matrix from transformation matrix
#if __FIX_MAKROSCHROTT__ == 0
  template rtti< float >::value_type PointToPlane(const vec3< float >& point, const vec4< float >& plane); // unsigned distance from a point to a plane (3D)
#endif /* __FIX_MAKROSCHROTT__ */

  template float fmin(float x, float y);
  template float fmax(float x, float y);

  template vec2< float > fmin(const vec2< float >& v1, const vec2< float >& v2);
  template vec2< float > fmax(const vec2< float >& v1, const vec2< float >& v2);

  template vec3< float > fmin(const vec3< float >& v1, const vec3< float >& v2);
  template vec3< float > fmax(const vec3< float >& v1, const vec3< float >& v2);

  template vec4< float > fmin(const vec4< float >& v1, const vec4< float >& v2);
  template vec4< float > fmax(const vec4< float >& v1, const vec4< float >& v2);

} /* namespace aux */



/* RTTI class of < double > */



namespace aux {

  template class rtti< double >; // RTTI of < double >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< double >+;

#endif /* __MAKECINT__ */



/* RTTI class of < vec2< double > > */



namespace aux {

  template class rtti< vec2< double > >; // RTTI of < vec2< double > >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< aux::vec2< double > >+;

#endif /* __MAKECINT__ */



/* 2D Vector class and friends of < double > */



namespace aux {

  template class vec2< double >; // 2D Vector of < double >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::vec2< double >+;

#endif /* __MAKECINT__ */

namespace aux {

  template vec2< double > operator - (const vec2< double >& v); // -v1
  template vec2< double > operator + (const vec2< double >& a, const vec2< double >& b); // v1 + v2
  template vec2< double > operator - (const vec2< double >& a, const vec2< double >& b); // v1 - v2
  template vec2< double > operator * (const vec2< double >& a, const double d); // v1 * 3.0
  template vec2< double > operator * (const double d, const vec2< double >& a); // 3.0 * v1
  template vec2< double > operator / (const vec2< double >& a, const double d); // v1 / 3.0
  template vec2< double > operator * (const mat2< double >& a, const vec2< double >& v); // linear transform
  template vec2< double > operator * (const mat3< double >& a, const vec2< double >& v); // M . v
  template vec2< double > operator * (const vec2< double >& v, const mat3< double >& a); // v . M
  template vec3< double > operator ^ (const vec2< double >& a, const vec2< double >& b); // cross product
  template double operator * (const vec2< double >& a, const vec2< double >& b); // dot product
  template bool operator == (const vec2< double >& a, const vec2< double >& b); // v1 == v2 ?
  template bool operator != (const vec2< double >& a, const vec2< double >& b); // v1 != v2 ?

  template ostream& operator << (ostream& s, const vec2< double >& v); // output to stream
  template istream& operator >> (istream& s, vec2< double >& v); // input from stream

  template void swap(vec2< double >& a, vec2< double >& b); // swap v1 & v2
  template vec2< double > min(const vec2< double >& a, const vec2< double >& b); // min(v1, v2)
  template vec2< double > max(const vec2< double >& a, const vec2< double >& b); // max(v1, v2)
  template vec2< double > prod(const vec2< double >& a, const vec2< double >& b); // term by term *
  template vec2< double > conj(const vec2< double >& a);

} /* namespace aux */



/* RTTI class of < vec3< double > > */



namespace aux {

  template class rtti< vec3< double > >; // RTTI of < vec3< double > >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< aux::vec3< double > >+;

#endif /* __MAKECINT__ */



/* 3D Vector class and friends of < double > */



namespace aux {

  template class vec3< double >; // 3D Vector of < double >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::vec3< double >+;

#endif /* __MAKECINT__ */

namespace aux {

  template vec3< double > operator - (const vec3< double >& v); // -v1
  template vec3< double > operator + (const vec3< double >& a, const vec3< double >& b); // v1 + v2
  template vec3< double > operator - (const vec3< double >& a, const vec3< double >& b); // v1 - v2
  template vec3< double > operator * (const vec3< double >& a, const double d); // v1 * 3.0
  template vec3< double > operator * (const double d, const vec3< double >& a); // 3.0 * v1
  template vec3< double > operator / (const vec3< double >& a, const double d); // v1 / 3.0
  template vec3< double > operator * (const mat3< double >& a, const vec3< double >& v); // linear transform
  template vec3< double > operator * (const mat4< double >& a, const vec3< double >& v); // M . v
  template vec3< double > operator * (const vec3< double >& v, const mat4< double >& a); // v . M
  template vec3< double > operator ^ (const vec3< double >& a, const vec3< double >& b); // cross product
  template double operator * (const vec3< double >& a, const vec3< double >& b); // dot product
  template bool operator == (const vec3< double >& a, const vec3< double >& b); // v1 == v2 ?
  template bool operator != (const vec3< double >& a, const vec3< double >& b); // v1 != v2 ?

  template ostream& operator << (ostream& s, const vec3< double >& v); // output to stream
  template istream& operator >> (istream& s, vec3< double >& v); // input from stream

  template void swap(vec3< double >& a, vec3< double >& b); // swap v1 & v2
  template vec3< double > min(const vec3< double >& a, const vec3< double >& b); // min(v1, v2)
  template vec3< double > max(const vec3< double >& a, const vec3< double >& b); // max(v1, v2)
  template vec3< double > prod(const vec3< double >& a, const vec3< double >& b); // term by term *
  template vec3< double > conj(const vec3< double >& a);

} /* namespace aux */



/* RTTI class of < vec4< double > > */



namespace aux {

  template class rtti< vec4< double > >; // RTTI of < vec4< double > >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< aux::vec4< double > >+;

#endif /* __MAKECINT__ */



/* 4D Vector class and friends of < double > */



namespace aux {

  template class vec4< double >; // 4D Vector of < double >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::vec4< double >+;

#endif /* __MAKECINT__ */

namespace aux {

  template vec4< double > operator - (const vec4< double >& v); // -v1
  template vec4< double > operator + (const vec4< double >& a, const vec4< double >& b); // v1 + v2
  template vec4< double > operator - (const vec4< double >& a, const vec4< double >& b); // v1 - v2
  template vec4< double > operator * (const vec4< double >& a, const double d); // v1 * 3.0
  template vec4< double > operator * (const double d, const vec4< double >& a); // 3.0 * v1
  template vec4< double > operator / (const vec4< double >& a, const double d); // v1 / 3.0
  template vec4< double > operator * (const mat4< double >& a, const vec4< double >& v); // M . v
  template vec4< double > operator * (const vec4< double >& v, const mat4< double >& a); // v . M
  template double operator * (const vec4< double >& a, const vec4< double >& b); // dot product
  template bool operator == (const vec4< double >& a, const vec4< double >& b); // v1 == v2 ?
  template bool operator != (const vec4< double >& a, const vec4< double >& b); // v1 != v2 ?

  template ostream& operator << (ostream& s, const vec4< double >& v); // output to stream
  template istream& operator >> (istream& s, vec4< double >& v); // input from stream

  template void swap(vec4< double >& a, vec4< double >& b); // swap v1 & v2
  template vec4< double > min(const vec4< double >& a, const vec4< double >& b); // min(v1, v2)
  template vec4< double > max(const vec4< double >& a, const vec4< double >& b); // max(v1, v2)
  template vec4< double > prod(const vec4< double >& a, const vec4< double >& b); // term by term *
  template vec4< double > conj(const vec4< double >& a);

} /* namespace aux */



/* RTTI class of < mat2< double > > */



namespace aux {

  template class rtti< mat2< double > >; // RTTI of < mat2< double > >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< aux::mat2< double > >+;

#endif /* __MAKECINT__ */



/* 2x2 Matrix class and friends of < double > */



namespace aux {

  template class mat2< double >; // 2x2 Matrix of < double >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::mat2< double >+;

#endif /* __MAKECINT__ */

namespace aux {

  template mat2< double > operator - (const mat2< double >& a); // -m1
  template mat2< double > operator + (const mat2< double >& a, const mat2< double >& b); // m1 + m2
  template mat2< double > operator - (const mat2< double >& a, const mat2< double >& b); // m1 - m2
  template mat2< double > operator * (const mat2< double >& a, const mat2< double >& b); // m1 * m2
  template mat2< double > operator * (const mat2< double >& a, const double d); // m1 * 3.0
  template mat2< double > operator * (const double d, const mat2< double >& a); // 3.0 * m1
  template mat2< double > operator / (const mat2< double >& a, const double d); // m1 / 3.0
  template bool operator == (const mat2< double >& a, const mat2< double >& b); // m1 == m2 ?
  template bool operator != (const mat2< double >& a, const mat2< double >& b); // m1 != m2 ?

  template ostream& operator << (ostream& s, const mat2< double >& m); // output to stream
  template istream& operator >> (istream& s, mat2< double >& m); // input from stream

  template void swap(mat2< double >& a, mat2< double >& b); // swap m1 & m2
  template mat2< double > conj(const mat2< double >& a);
  template mat2< double > diagonal(const vec2< double >& v);
  template mat2< double > diagonal(const double x0, const double y1);
  template double trace(const mat2< double >& a);

} /* namespace aux */



/* RTTI class of < mat3< double > > */



namespace aux {

  template class rtti< mat3< double > >; // RTTI of < mat3< double > >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< aux::mat3< double > >+;

#endif /* __MAKECINT__ */



/* 3x3 Matrix class and friends of < double > */



namespace aux {

  template class mat3< double >; // 3x3 Matrix of < double >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::mat3< double >+;

#endif /* __MAKECINT__ */

namespace aux {

  template mat3< double > operator - (const mat3< double >& a); // -m1
  template mat3< double > operator + (const mat3< double >& a, const mat3< double >& b); // m1 + m2
  template mat3< double > operator - (const mat3< double >& a, const mat3< double >& b); // m1 - m2
  template mat3< double > operator * (const mat3< double >& a, const mat3< double >& b); // m1 * m2
  template mat3< double > operator * (const mat3< double >& a, const double d); // m1 * 3.0
  template mat3< double > operator * (const double d, const mat3< double >& a); // 3.0 * m1
  template mat3< double > operator / (const mat3< double >& a, const double d); // m1 / 3.0
  template bool operator == (const mat3< double >& a, const mat3< double >& b); // m1 == m2 ?
  template bool operator != (const mat3< double >& a, const mat3< double >& b); // m1 != m2 ?

  template ostream& operator << (ostream& s, const mat3< double >& m); // output to stream
  template istream& operator >> (istream& s, mat3< double >& m); // input from stream

  template void swap(mat3< double >& a, mat3< double >& b); // swap m1 & m2
  template mat3< double > conj(const mat3< double >& a);
  template mat3< double > diagonal(const vec3< double >& v);
  template mat3< double > diagonal(const double x0, const double y1, const double z2);
  template double trace(const mat3< double >& a);

} /* namespace aux */



/* RTTI class of < mat4< double > > */



namespace aux {

  template class rtti< mat4< double > >; // RTTI of < mat4< double > >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< aux::mat4< double > >+;

#endif /* __MAKECINT__ */



/* 4x4 Matrix class and friends of < double > */



namespace aux {

  template class mat4< double >; // 4x4 Matrix of < double >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::mat4< double >+;

#endif /* __MAKECINT__ */

namespace aux {

  template mat4< double > operator - (const mat4< double >& a); // -m1
  template mat4< double > operator + (const mat4< double >& a, const mat4< double >& b); // m1 + m2
  template mat4< double > operator - (const mat4< double >& a, const mat4< double >& b); // m1 - m2
  template mat4< double > operator * (const mat4< double >& a, const mat4< double >& b); // m1 * m2
  template mat4< double > operator * (const mat4< double >& a, const double d); // m1 * 3.0
  template mat4< double > operator * (const double d, const mat4< double >& a); // 3.0 * m1
  template mat4< double > operator / (const mat4< double >& a, const double d); // m1 / 3.0
  template bool operator == (const mat4< double >& a, const mat4< double >& b); // m1 == m2 ?
  template bool operator != (const mat4< double >& a, const mat4< double >& b); // m1 != m2 ?

  template ostream& operator << (ostream& s, const mat4< double >& m); // output to stream
  template istream& operator >> (istream& s, mat4< double >& m); // input from stream

  template void swap(mat4< double >& a, mat4< double >& b); // swap m1 & m2
  template mat4< double > conj(const mat4< double >& a);
  template mat4< double > diagonal(const vec4< double >& v);
  template mat4< double > diagonal(const double x0, const double y1, const double z2, const double w3);
  template double trace(const mat4< double >& a);

} /* namespace aux */



/* 2D functions and 3D functions of < double > */



namespace aux {

  template mat2< double > identity1D< double >(void); // identity 1D
  template mat2< double > translation1D(const double & v); // translation 1D
  template mat2< double > scaling1D(const double & scaleVal); // scaling 1D
  template mat3< double > identity2D< double >(void); // identity 2D
  template mat3< double > translation2D(const vec2< double >& v); // translation 2D
#if __FIX_MAKROSCHROTT__ == 0
  template mat3< double > rotation2D(const vec2< double >& Center, const rtti< double >::value_type angleDeg); // rotation 2D
#endif /* __FIX_MAKROSCHROTT__ */
  template mat3< double > scaling2D(const vec2< double >& scaleVec); // scaling 2D
  template mat4< double > identity3D< double >(void); // identity 3D
  template mat4< double > translation3D(const vec3< double >& v); // translation 3D
#if __FIX_MAKROSCHROTT__ == 0
  template mat4< double > rotation3D(vec3< double > Axis, const rtti< double >::value_type angleDeg); // rotation 3D
#endif /* __FIX_MAKROSCHROTT__ */
  template mat4< double > scaling3D(const vec3< double >& scaleVec); // scaling 3D
  template mat4< double > perspective3D(const double d); // perspective 3D

} /* namespace aux */



/* AUX functions of < double > */



namespace aux {

  template vec2< double > MultiplyElementwise(const vec2< double >& v1, const vec2< double >& v2); // element by element multiplication
  template vec3< double > MultiplyElementwise(const vec3< double >& v1, const vec3< double >& v2); // element by element multiplication
  template vec4< double > MultiplyElementwise(const vec4< double >& v1, const vec4< double >& v2); // element by element multiplication

  template mat3< double > rotation2D(const mat2< double >& m); // convert 2x2 rotation matrix to 3x3
  template vec2< double > ExtractTranslation(const mat3< double >& pose); // extract translation vector
  template vec2< double > ExtractScaling(const mat3< double >& mat);
  template mat2< double > ExtractRotation(const mat3< double >& pose); // extract rotation matrix from transformation matrix
#if __FIX_MAKROSCHROTT__ == 0
  template rtti< double >::value_type PointToLine(const vec2< double >& point, const vec3< double >& line); // unsigned distance from a point to a line (2D)
#endif /* __FIX_MAKROSCHROTT__ */
  template mat4< double > rotation3D(const mat3< double >& m); // convert 3x3 rotation matrix to 4x4
  template vec3< double > ExtractTranslation(const mat4< double >& pose); // extract translation vector
  template vec3< double > ExtractScaling(const mat4< double >& mat);
  template mat3< double > ExtractRotation(const mat4< double >& pose); // extract rotation matrix from transformation matrix
#if __FIX_MAKROSCHROTT__ == 0
  template rtti< double >::value_type PointToPlane(const vec3< double >& point, const vec4< double >& plane); // unsigned distance from a point to a plane (3D)
#endif /* __FIX_MAKROSCHROTT__ */

  template double fmin(double x, double y);
  template double fmax(double x, double y);

  template vec2< double > fmin(const vec2< double >& v1, const vec2< double >& v2);
  template vec2< double > fmax(const vec2< double >& v1, const vec2< double >& v2);

  template vec3< double > fmin(const vec3< double >& v1, const vec3< double >& v2);
  template vec3< double > fmax(const vec3< double >& v1, const vec3< double >& v2);

  template vec4< double > fmin(const vec4< double >& v1, const vec4< double >& v2);
  template vec4< double > fmax(const vec4< double >& v1, const vec4< double >& v2);

} /* namespace aux */



/* RTTI class of < complex<float> > */



namespace aux {

  template class rtti< complex<float> >; // RTTI of < complex<float> >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< complex<float> >+;

#endif /* __MAKECINT__ */



/* RTTI class of < vec2< complex<float> > > */



namespace aux {

  template class rtti< vec2< complex<float> > >; // RTTI of < vec2< complex<float> > >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< aux::vec2< complex<float> > >+;

#endif /* __MAKECINT__ */



/* 2D Vector class and friends of < complex<float> > */



namespace aux {

  template class vec2< complex<float> >; // 2D Vector of < complex<float> >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::vec2< complex<float> >+;

#endif /* __MAKECINT__ */

namespace aux {

  template vec2< complex<float> > operator - (const vec2< complex<float> >& v); // -v1
  template vec2< complex<float> > operator + (const vec2< complex<float> >& a, const vec2< complex<float> >& b); // v1 + v2
  template vec2< complex<float> > operator - (const vec2< complex<float> >& a, const vec2< complex<float> >& b); // v1 - v2
  template vec2< complex<float> > operator * (const vec2< complex<float> >& a, const complex<float> d); // v1 * 3.0
  template vec2< complex<float> > operator * (const complex<float> d, const vec2< complex<float> >& a); // 3.0 * v1
  template vec2< complex<float> > operator / (const vec2< complex<float> >& a, const complex<float> d); // v1 / 3.0
  template vec2< complex<float> > operator * (const mat2< complex<float> >& a, const vec2< complex<float> >& v); // linear transform
  template vec2< complex<float> > operator * (const mat3< complex<float> >& a, const vec2< complex<float> >& v); // M . v
  template vec2< complex<float> > operator * (const vec2< complex<float> >& v, const mat3< complex<float> >& a); // v . M
  template vec3< complex<float> > operator ^ (const vec2< complex<float> >& a, const vec2< complex<float> >& b); // cross product
  template complex<float> operator * (const vec2< complex<float> >& a, const vec2< complex<float> >& b); // dot product
  template bool operator == (const vec2< complex<float> >& a, const vec2< complex<float> >& b); // v1 == v2 ?
  template bool operator != (const vec2< complex<float> >& a, const vec2< complex<float> >& b); // v1 != v2 ?

  template ostream& operator << (ostream& s, const vec2< complex<float> >& v); // output to stream
  template istream& operator >> (istream& s, vec2< complex<float> >& v); // input from stream

  template void swap(vec2< complex<float> >& a, vec2< complex<float> >& b); // swap v1 & v2
  template vec2< complex<float> > min(const vec2< complex<float> >& a, const vec2< complex<float> >& b); // min(v1, v2)
  template vec2< complex<float> > max(const vec2< complex<float> >& a, const vec2< complex<float> >& b); // max(v1, v2)
  template vec2< complex<float> > prod(const vec2< complex<float> >& a, const vec2< complex<float> >& b); // term by term *
  template vec2< complex<float> > conj(const vec2< complex<float> >& a);

} /* namespace aux */



/* RTTI class of < vec3< complex<float> > > */



namespace aux {

  template class rtti< vec3< complex<float> > >; // RTTI of < vec3< complex<float> > >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< aux::vec3< complex<float> > >+;

#endif /* __MAKECINT__ */



/* 3D Vector class and friends of < complex<float> > */



namespace aux {

  template class vec3< complex<float> >; // 3D Vector of < complex<float> >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::vec3< complex<float> >+;

#endif /* __MAKECINT__ */

namespace aux {

  template vec3< complex<float> > operator - (const vec3< complex<float> >& v); // -v1
  template vec3< complex<float> > operator + (const vec3< complex<float> >& a, const vec3< complex<float> >& b); // v1 + v2
  template vec3< complex<float> > operator - (const vec3< complex<float> >& a, const vec3< complex<float> >& b); // v1 - v2
  template vec3< complex<float> > operator * (const vec3< complex<float> >& a, const complex<float> d); // v1 * 3.0
  template vec3< complex<float> > operator * (const complex<float> d, const vec3< complex<float> >& a); // 3.0 * v1
  template vec3< complex<float> > operator / (const vec3< complex<float> >& a, const complex<float> d); // v1 / 3.0
  template vec3< complex<float> > operator * (const mat3< complex<float> >& a, const vec3< complex<float> >& v); // linear transform
  template vec3< complex<float> > operator * (const mat4< complex<float> >& a, const vec3< complex<float> >& v); // M . v
  template vec3< complex<float> > operator * (const vec3< complex<float> >& v, const mat4< complex<float> >& a); // v . M
  template vec3< complex<float> > operator ^ (const vec3< complex<float> >& a, const vec3< complex<float> >& b); // cross product
  template complex<float> operator * (const vec3< complex<float> >& a, const vec3< complex<float> >& b); // dot product
  template bool operator == (const vec3< complex<float> >& a, const vec3< complex<float> >& b); // v1 == v2 ?
  template bool operator != (const vec3< complex<float> >& a, const vec3< complex<float> >& b); // v1 != v2 ?

  template ostream& operator << (ostream& s, const vec3< complex<float> >& v); // output to stream
  template istream& operator >> (istream& s, vec3< complex<float> >& v); // input from stream

  template void swap(vec3< complex<float> >& a, vec3< complex<float> >& b); // swap v1 & v2
  template vec3< complex<float> > min(const vec3< complex<float> >& a, const vec3< complex<float> >& b); // min(v1, v2)
  template vec3< complex<float> > max(const vec3< complex<float> >& a, const vec3< complex<float> >& b); // max(v1, v2)
  template vec3< complex<float> > prod(const vec3< complex<float> >& a, const vec3< complex<float> >& b); // term by term *
  template vec3< complex<float> > conj(const vec3< complex<float> >& a);

} /* namespace aux */



/* RTTI class of < vec4< complex<float> > > */



namespace aux {

  template class rtti< vec4< complex<float> > >; // RTTI of < vec4< complex<float> > >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< aux::vec4< complex<float> > >+;

#endif /* __MAKECINT__ */



/* 4D Vector class and friends of < complex<float> > */



namespace aux {

  template class vec4< complex<float> >; // 4D Vector of < complex<float> >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::vec4< complex<float> >+;

#endif /* __MAKECINT__ */

namespace aux {

  template vec4< complex<float> > operator - (const vec4< complex<float> >& v); // -v1
  template vec4< complex<float> > operator + (const vec4< complex<float> >& a, const vec4< complex<float> >& b); // v1 + v2
  template vec4< complex<float> > operator - (const vec4< complex<float> >& a, const vec4< complex<float> >& b); // v1 - v2
  template vec4< complex<float> > operator * (const vec4< complex<float> >& a, const complex<float> d); // v1 * 3.0
  template vec4< complex<float> > operator * (const complex<float> d, const vec4< complex<float> >& a); // 3.0 * v1
  template vec4< complex<float> > operator / (const vec4< complex<float> >& a, const complex<float> d); // v1 / 3.0
  template vec4< complex<float> > operator * (const mat4< complex<float> >& a, const vec4< complex<float> >& v); // M . v
  template vec4< complex<float> > operator * (const vec4< complex<float> >& v, const mat4< complex<float> >& a); // v . M
  template complex<float> operator * (const vec4< complex<float> >& a, const vec4< complex<float> >& b); // dot product
  template bool operator == (const vec4< complex<float> >& a, const vec4< complex<float> >& b); // v1 == v2 ?
  template bool operator != (const vec4< complex<float> >& a, const vec4< complex<float> >& b); // v1 != v2 ?

  template ostream& operator << (ostream& s, const vec4< complex<float> >& v); // output to stream
  template istream& operator >> (istream& s, vec4< complex<float> >& v); // input from stream

  template void swap(vec4< complex<float> >& a, vec4< complex<float> >& b); // swap v1 & v2
  template vec4< complex<float> > min(const vec4< complex<float> >& a, const vec4< complex<float> >& b); // min(v1, v2)
  template vec4< complex<float> > max(const vec4< complex<float> >& a, const vec4< complex<float> >& b); // max(v1, v2)
  template vec4< complex<float> > prod(const vec4< complex<float> >& a, const vec4< complex<float> >& b); // term by term *
  template vec4< complex<float> > conj(const vec4< complex<float> >& a);

} /* namespace aux */



/* RTTI class of < mat2< complex<float> > > */



namespace aux {

  template class rtti< mat2< complex<float> > >; // RTTI of < mat2< complex<float> > >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< aux::mat2< complex<float> > >+;

#endif /* __MAKECINT__ */



/* 2x2 Matrix class and friends of < complex<float> > */



namespace aux {

  template class mat2< complex<float> >; // 2x2 Matrix of < complex<float> >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::mat2< complex<float> >+;

#endif /* __MAKECINT__ */

namespace aux {

  template mat2< complex<float> > operator - (const mat2< complex<float> >& a); // -m1
  template mat2< complex<float> > operator + (const mat2< complex<float> >& a, const mat2< complex<float> >& b); // m1 + m2
  template mat2< complex<float> > operator - (const mat2< complex<float> >& a, const mat2< complex<float> >& b); // m1 - m2
  template mat2< complex<float> > operator * (const mat2< complex<float> >& a, const mat2< complex<float> >& b); // m1 * m2
  template mat2< complex<float> > operator * (const mat2< complex<float> >& a, const complex<float> d); // m1 * 3.0
  template mat2< complex<float> > operator * (const complex<float> d, const mat2< complex<float> >& a); // 3.0 * m1
  template mat2< complex<float> > operator / (const mat2< complex<float> >& a, const complex<float> d); // m1 / 3.0
  template bool operator == (const mat2< complex<float> >& a, const mat2< complex<float> >& b); // m1 == m2 ?
  template bool operator != (const mat2< complex<float> >& a, const mat2< complex<float> >& b); // m1 != m2 ?

  template ostream& operator << (ostream& s, const mat2< complex<float> >& m); // output to stream
  template istream& operator >> (istream& s, mat2< complex<float> >& m); // input from stream

  template void swap(mat2< complex<float> >& a, mat2< complex<float> >& b); // swap m1 & m2
  template mat2< complex<float> > conj(const mat2< complex<float> >& a);
  template mat2< complex<float> > diagonal(const vec2< complex<float> >& v);
  template mat2< complex<float> > diagonal(const complex<float> x0, const complex<float> y1);
  template complex<float> trace(const mat2< complex<float> >& a);

} /* namespace aux */



/* RTTI class of < mat3< complex<float> > > */



namespace aux {

  template class rtti< mat3< complex<float> > >; // RTTI of < mat3< complex<float> > >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< aux::mat3< complex<float> > >+;

#endif /* __MAKECINT__ */



/* 3x3 Matrix class and friends of < complex<float> > */



namespace aux {

  template class mat3< complex<float> >; // 3x3 Matrix of < complex<float> >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::mat3< complex<float> >+;

#endif /* __MAKECINT__ */

namespace aux {

  template mat3< complex<float> > operator - (const mat3< complex<float> >& a); // -m1
  template mat3< complex<float> > operator + (const mat3< complex<float> >& a, const mat3< complex<float> >& b); // m1 + m2
  template mat3< complex<float> > operator - (const mat3< complex<float> >& a, const mat3< complex<float> >& b); // m1 - m2
  template mat3< complex<float> > operator * (const mat3< complex<float> >& a, const mat3< complex<float> >& b); // m1 * m2
  template mat3< complex<float> > operator * (const mat3< complex<float> >& a, const complex<float> d); // m1 * 3.0
  template mat3< complex<float> > operator * (const complex<float> d, const mat3< complex<float> >& a); // 3.0 * m1
  template mat3< complex<float> > operator / (const mat3< complex<float> >& a, const complex<float> d); // m1 / 3.0
  template bool operator == (const mat3< complex<float> >& a, const mat3< complex<float> >& b); // m1 == m2 ?
  template bool operator != (const mat3< complex<float> >& a, const mat3< complex<float> >& b); // m1 != m2 ?

  template ostream& operator << (ostream& s, const mat3< complex<float> >& m); // output to stream
  template istream& operator >> (istream& s, mat3< complex<float> >& m); // input from stream

  template void swap(mat3< complex<float> >& a, mat3< complex<float> >& b); // swap m1 & m2
  template mat3< complex<float> > conj(const mat3< complex<float> >& a);
  template mat3< complex<float> > diagonal(const vec3< complex<float> >& v);
  template mat3< complex<float> > diagonal(const complex<float> x0, const complex<float> y1, const complex<float> z2);
  template complex<float> trace(const mat3< complex<float> >& a);

} /* namespace aux */



/* RTTI class of < mat4< complex<float> > > */



namespace aux {

  template class rtti< mat4< complex<float> > >; // RTTI of < mat4< complex<float> > >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< aux::mat4< complex<float> > >+;

#endif /* __MAKECINT__ */



/* 4x4 Matrix class and friends of < complex<float> > */



namespace aux {

  template class mat4< complex<float> >; // 4x4 Matrix of < complex<float> >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::mat4< complex<float> >+;

#endif /* __MAKECINT__ */

namespace aux {

  template mat4< complex<float> > operator - (const mat4< complex<float> >& a); // -m1
  template mat4< complex<float> > operator + (const mat4< complex<float> >& a, const mat4< complex<float> >& b); // m1 + m2
  template mat4< complex<float> > operator - (const mat4< complex<float> >& a, const mat4< complex<float> >& b); // m1 - m2
  template mat4< complex<float> > operator * (const mat4< complex<float> >& a, const mat4< complex<float> >& b); // m1 * m2
  template mat4< complex<float> > operator * (const mat4< complex<float> >& a, const complex<float> d); // m1 * 3.0
  template mat4< complex<float> > operator * (const complex<float> d, const mat4< complex<float> >& a); // 3.0 * m1
  template mat4< complex<float> > operator / (const mat4< complex<float> >& a, const complex<float> d); // m1 / 3.0
  template bool operator == (const mat4< complex<float> >& a, const mat4< complex<float> >& b); // m1 == m2 ?
  template bool operator != (const mat4< complex<float> >& a, const mat4< complex<float> >& b); // m1 != m2 ?

  template ostream& operator << (ostream& s, const mat4< complex<float> >& m); // output to stream
  template istream& operator >> (istream& s, mat4< complex<float> >& m); // input from stream

  template void swap(mat4< complex<float> >& a, mat4< complex<float> >& b); // swap m1 & m2
  template mat4< complex<float> > conj(const mat4< complex<float> >& a);
  template mat4< complex<float> > diagonal(const vec4< complex<float> >& v);
  template mat4< complex<float> > diagonal(const complex<float> x0, const complex<float> y1, const complex<float> z2, const complex<float> w3);
  template complex<float> trace(const mat4< complex<float> >& a);

} /* namespace aux */



/* 2D functions and 3D functions of < complex<float> > */



namespace aux {

  template mat2< complex<float> > identity1D< complex<float> >(void); // identity 1D
  template mat2< complex<float> > translation1D(const complex<float> & v); // translation 1D
  template mat2< complex<float> > scaling1D(const complex<float> & scaleVal); // scaling 1D
  template mat3< complex<float> > identity2D< complex<float> >(void); // identity 2D
  template mat3< complex<float> > translation2D(const vec2< complex<float> >& v); // translation 2D
#if __FIX_MAKROSCHROTT__ == 0
  template mat3< complex<float> > rotation2D(const vec2< complex<float> >& Center, const rtti< complex<float> >::value_type angleDeg); // rotation 2D
#endif /* __FIX_MAKROSCHROTT__ */
  template mat3< complex<float> > scaling2D(const vec2< complex<float> >& scaleVec); // scaling 2D
  template mat4< complex<float> > identity3D< complex<float> >(void); // identity 3D
  template mat4< complex<float> > translation3D(const vec3< complex<float> >& v); // translation 3D
#if __FIX_MAKROSCHROTT__ == 0
  template mat4< complex<float> > rotation3D(vec3< complex<float> > Axis, const rtti< complex<float> >::value_type angleDeg); // rotation 3D
#endif /* __FIX_MAKROSCHROTT__ */
  template mat4< complex<float> > scaling3D(const vec3< complex<float> >& scaleVec); // scaling 3D
  template mat4< complex<float> > perspective3D(const complex<float> d); // perspective 3D

} /* namespace aux */



/* AUX functions of < complex<float> > */



namespace aux {

  template vec2< complex<float> > MultiplyElementwise(const vec2< complex<float> >& v1, const vec2< complex<float> >& v2); // element by element multiplication
  template vec3< complex<float> > MultiplyElementwise(const vec3< complex<float> >& v1, const vec3< complex<float> >& v2); // element by element multiplication
  template vec4< complex<float> > MultiplyElementwise(const vec4< complex<float> >& v1, const vec4< complex<float> >& v2); // element by element multiplication

  template mat3< complex<float> > rotation2D(const mat2< complex<float> >& m); // convert 2x2 rotation matrix to 3x3
  template vec2< complex<float> > ExtractTranslation(const mat3< complex<float> >& pose); // extract translation vector
  template vec2< complex<float> > ExtractScaling(const mat3< complex<float> >& mat);
  template mat2< complex<float> > ExtractRotation(const mat3< complex<float> >& pose); // extract rotation matrix from transformation matrix
#if __FIX_MAKROSCHROTT__ == 0
  template rtti< complex<float> >::value_type PointToLine(const vec2< complex<float> >& point, const vec3< complex<float> >& line); // unsigned distance from a point to a line (2D)
#endif /* __FIX_MAKROSCHROTT__ */
  template mat4< complex<float> > rotation3D(const mat3< complex<float> >& m); // convert 3x3 rotation matrix to 4x4
  template vec3< complex<float> > ExtractTranslation(const mat4< complex<float> >& pose); // extract translation vector
  template vec3< complex<float> > ExtractScaling(const mat4< complex<float> >& mat);
  template mat3< complex<float> > ExtractRotation(const mat4< complex<float> >& pose); // extract rotation matrix from transformation matrix
#if __FIX_MAKROSCHROTT__ == 0
  template rtti< complex<float> >::value_type PointToPlane(const vec3< complex<float> >& point, const vec4< complex<float> >& plane); // unsigned distance from a point to a plane (3D)
#endif /* __FIX_MAKROSCHROTT__ */

  template complex<float> fmin(complex<float> x, complex<float> y);
  template complex<float> fmax(complex<float> x, complex<float> y);

  template vec2< complex<float> > fmin(const vec2< complex<float> >& v1, const vec2< complex<float> >& v2);
  template vec2< complex<float> > fmax(const vec2< complex<float> >& v1, const vec2< complex<float> >& v2);

  template vec3< complex<float> > fmin(const vec3< complex<float> >& v1, const vec3< complex<float> >& v2);
  template vec3< complex<float> > fmax(const vec3< complex<float> >& v1, const vec3< complex<float> >& v2);

  template vec4< complex<float> > fmin(const vec4< complex<float> >& v1, const vec4< complex<float> >& v2);
  template vec4< complex<float> > fmax(const vec4< complex<float> >& v1, const vec4< complex<float> >& v2);

} /* namespace aux */



/* RTTI class of < complex<double> > */



namespace aux {

  template class rtti< complex<double> >; // RTTI of < complex<double> >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< complex<double> >+;

#endif /* __MAKECINT__ */



/* RTTI class of < vec2< complex<double> > > */



namespace aux {

  template class rtti< vec2< complex<double> > >; // RTTI of < vec2< complex<double> > >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< aux::vec2< complex<double> > >+;

#endif /* __MAKECINT__ */



/* 2D Vector class and friends of < complex<double> > */



namespace aux {

  template class vec2< complex<double> >; // 2D Vector of < complex<double> >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::vec2< complex<double> >+;

#endif /* __MAKECINT__ */

namespace aux {

  template vec2< complex<double> > operator - (const vec2< complex<double> >& v); // -v1
  template vec2< complex<double> > operator + (const vec2< complex<double> >& a, const vec2< complex<double> >& b); // v1 + v2
  template vec2< complex<double> > operator - (const vec2< complex<double> >& a, const vec2< complex<double> >& b); // v1 - v2
  template vec2< complex<double> > operator * (const vec2< complex<double> >& a, const complex<double> d); // v1 * 3.0
  template vec2< complex<double> > operator * (const complex<double> d, const vec2< complex<double> >& a); // 3.0 * v1
  template vec2< complex<double> > operator / (const vec2< complex<double> >& a, const complex<double> d); // v1 / 3.0
  template vec2< complex<double> > operator * (const mat2< complex<double> >& a, const vec2< complex<double> >& v); // linear transform
  template vec2< complex<double> > operator * (const mat3< complex<double> >& a, const vec2< complex<double> >& v); // M . v
  template vec2< complex<double> > operator * (const vec2< complex<double> >& v, const mat3< complex<double> >& a); // v . M
  template vec3< complex<double> > operator ^ (const vec2< complex<double> >& a, const vec2< complex<double> >& b); // cross product
  template complex<double> operator * (const vec2< complex<double> >& a, const vec2< complex<double> >& b); // dot product
  template bool operator == (const vec2< complex<double> >& a, const vec2< complex<double> >& b); // v1 == v2 ?
  template bool operator != (const vec2< complex<double> >& a, const vec2< complex<double> >& b); // v1 != v2 ?

  template ostream& operator << (ostream& s, const vec2< complex<double> >& v); // output to stream
  template istream& operator >> (istream& s, vec2< complex<double> >& v); // input from stream

  template void swap(vec2< complex<double> >& a, vec2< complex<double> >& b); // swap v1 & v2
  template vec2< complex<double> > min(const vec2< complex<double> >& a, const vec2< complex<double> >& b); // min(v1, v2)
  template vec2< complex<double> > max(const vec2< complex<double> >& a, const vec2< complex<double> >& b); // max(v1, v2)
  template vec2< complex<double> > prod(const vec2< complex<double> >& a, const vec2< complex<double> >& b); // term by term *
  template vec2< complex<double> > conj(const vec2< complex<double> >& a);

} /* namespace aux */



/* RTTI class of < vec3< complex<double> > > */



namespace aux {

  template class rtti< vec3< complex<double> > >; // RTTI of < vec3< complex<double> > >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< aux::vec3< complex<double> > >+;

#endif /* __MAKECINT__ */



/* 3D Vector class and friends of < complex<double> > */



namespace aux {

  template class vec3< complex<double> >; // 3D Vector of < complex<double> >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::vec3< complex<double> >+;

#endif /* __MAKECINT__ */

namespace aux {

  template vec3< complex<double> > operator - (const vec3< complex<double> >& v); // -v1
  template vec3< complex<double> > operator + (const vec3< complex<double> >& a, const vec3< complex<double> >& b); // v1 + v2
  template vec3< complex<double> > operator - (const vec3< complex<double> >& a, const vec3< complex<double> >& b); // v1 - v2
  template vec3< complex<double> > operator * (const vec3< complex<double> >& a, const complex<double> d); // v1 * 3.0
  template vec3< complex<double> > operator * (const complex<double> d, const vec3< complex<double> >& a); // 3.0 * v1
  template vec3< complex<double> > operator / (const vec3< complex<double> >& a, const complex<double> d); // v1 / 3.0
  template vec3< complex<double> > operator * (const mat3< complex<double> >& a, const vec3< complex<double> >& v); // linear transform
  template vec3< complex<double> > operator * (const mat4< complex<double> >& a, const vec3< complex<double> >& v); // M . v
  template vec3< complex<double> > operator * (const vec3< complex<double> >& v, const mat4< complex<double> >& a); // v . M
  template vec3< complex<double> > operator ^ (const vec3< complex<double> >& a, const vec3< complex<double> >& b); // cross product
  template complex<double> operator * (const vec3< complex<double> >& a, const vec3< complex<double> >& b); // dot product
  template bool operator == (const vec3< complex<double> >& a, const vec3< complex<double> >& b); // v1 == v2 ?
  template bool operator != (const vec3< complex<double> >& a, const vec3< complex<double> >& b); // v1 != v2 ?

  template ostream& operator << (ostream& s, const vec3< complex<double> >& v); // output to stream
  template istream& operator >> (istream& s, vec3< complex<double> >& v); // input from stream

  template void swap(vec3< complex<double> >& a, vec3< complex<double> >& b); // swap v1 & v2
  template vec3< complex<double> > min(const vec3< complex<double> >& a, const vec3< complex<double> >& b); // min(v1, v2)
  template vec3< complex<double> > max(const vec3< complex<double> >& a, const vec3< complex<double> >& b); // max(v1, v2)
  template vec3< complex<double> > prod(const vec3< complex<double> >& a, const vec3< complex<double> >& b); // term by term *
  template vec3< complex<double> > conj(const vec3< complex<double> >& a);

} /* namespace aux */



/* RTTI class of < vec4< complex<double> > > */



namespace aux {

  template class rtti< vec4< complex<double> > >; // RTTI of < vec4< complex<double> > >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< aux::vec4< complex<double> > >+;

#endif /* __MAKECINT__ */



/* 4D Vector class and friends of < complex<double> > */



namespace aux {

  template class vec4< complex<double> >; // 4D Vector of < complex<double> >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::vec4< complex<double> >+;

#endif /* __MAKECINT__ */

namespace aux {

  template vec4< complex<double> > operator - (const vec4< complex<double> >& v); // -v1
  template vec4< complex<double> > operator + (const vec4< complex<double> >& a, const vec4< complex<double> >& b); // v1 + v2
  template vec4< complex<double> > operator - (const vec4< complex<double> >& a, const vec4< complex<double> >& b); // v1 - v2
  template vec4< complex<double> > operator * (const vec4< complex<double> >& a, const complex<double> d); // v1 * 3.0
  template vec4< complex<double> > operator * (const complex<double> d, const vec4< complex<double> >& a); // 3.0 * v1
  template vec4< complex<double> > operator / (const vec4< complex<double> >& a, const complex<double> d); // v1 / 3.0
  template vec4< complex<double> > operator * (const mat4< complex<double> >& a, const vec4< complex<double> >& v); // M . v
  template vec4< complex<double> > operator * (const vec4< complex<double> >& v, const mat4< complex<double> >& a); // v . M
  template complex<double> operator * (const vec4< complex<double> >& a, const vec4< complex<double> >& b); // dot product
  template bool operator == (const vec4< complex<double> >& a, const vec4< complex<double> >& b); // v1 == v2 ?
  template bool operator != (const vec4< complex<double> >& a, const vec4< complex<double> >& b); // v1 != v2 ?

  template ostream& operator << (ostream& s, const vec4< complex<double> >& v); // output to stream
  template istream& operator >> (istream& s, vec4< complex<double> >& v); // input from stream

  template void swap(vec4< complex<double> >& a, vec4< complex<double> >& b); // swap v1 & v2
  template vec4< complex<double> > min(const vec4< complex<double> >& a, const vec4< complex<double> >& b); // min(v1, v2)
  template vec4< complex<double> > max(const vec4< complex<double> >& a, const vec4< complex<double> >& b); // max(v1, v2)
  template vec4< complex<double> > prod(const vec4< complex<double> >& a, const vec4< complex<double> >& b); // term by term *
  template vec4< complex<double> > conj(const vec4< complex<double> >& a);

} /* namespace aux */



/* RTTI class of < mat2< complex<double> > > */



namespace aux {

  template class rtti< mat2< complex<double> > >; // RTTI of < mat2< complex<double> > >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< aux::mat2< complex<double> > >+;

#endif /* __MAKECINT__ */



/* 2x2 Matrix class and friends of < complex<double> > */



namespace aux {

  template class mat2< complex<double> >; // 2x2 Matrix of < complex<double> >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::mat2< complex<double> >+;

#endif /* __MAKECINT__ */

namespace aux {

  template mat2< complex<double> > operator - (const mat2< complex<double> >& a); // -m1
  template mat2< complex<double> > operator + (const mat2< complex<double> >& a, const mat2< complex<double> >& b); // m1 + m2
  template mat2< complex<double> > operator - (const mat2< complex<double> >& a, const mat2< complex<double> >& b); // m1 - m2
  template mat2< complex<double> > operator * (const mat2< complex<double> >& a, const mat2< complex<double> >& b); // m1 * m2
  template mat2< complex<double> > operator * (const mat2< complex<double> >& a, const complex<double> d); // m1 * 3.0
  template mat2< complex<double> > operator * (const complex<double> d, const mat2< complex<double> >& a); // 3.0 * m1
  template mat2< complex<double> > operator / (const mat2< complex<double> >& a, const complex<double> d); // m1 / 3.0
  template bool operator == (const mat2< complex<double> >& a, const mat2< complex<double> >& b); // m1 == m2 ?
  template bool operator != (const mat2< complex<double> >& a, const mat2< complex<double> >& b); // m1 != m2 ?

  template ostream& operator << (ostream& s, const mat2< complex<double> >& m); // output to stream
  template istream& operator >> (istream& s, mat2< complex<double> >& m); // input from stream

  template void swap(mat2< complex<double> >& a, mat2< complex<double> >& b); // swap m1 & m2
  template mat2< complex<double> > conj(const mat2< complex<double> >& a);
  template mat2< complex<double> > diagonal(const vec2< complex<double> >& v);
  template mat2< complex<double> > diagonal(const complex<double> x0, const complex<double> y1);
  template complex<double> trace(const mat2< complex<double> >& a);

} /* namespace aux */



/* RTTI class of < mat3< complex<double> > > */



namespace aux {

  template class rtti< mat3< complex<double> > >; // RTTI of < mat3< complex<double> > >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< aux::mat3< complex<double> > >+;

#endif /* __MAKECINT__ */



/* 3x3 Matrix class and friends of < complex<double> > */



namespace aux {

  template class mat3< complex<double> >; // 3x3 Matrix of < complex<double> >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::mat3< complex<double> >+;

#endif /* __MAKECINT__ */

namespace aux {

  template mat3< complex<double> > operator - (const mat3< complex<double> >& a); // -m1
  template mat3< complex<double> > operator + (const mat3< complex<double> >& a, const mat3< complex<double> >& b); // m1 + m2
  template mat3< complex<double> > operator - (const mat3< complex<double> >& a, const mat3< complex<double> >& b); // m1 - m2
  template mat3< complex<double> > operator * (const mat3< complex<double> >& a, const mat3< complex<double> >& b); // m1 * m2
  template mat3< complex<double> > operator * (const mat3< complex<double> >& a, const complex<double> d); // m1 * 3.0
  template mat3< complex<double> > operator * (const complex<double> d, const mat3< complex<double> >& a); // 3.0 * m1
  template mat3< complex<double> > operator / (const mat3< complex<double> >& a, const complex<double> d); // m1 / 3.0
  template bool operator == (const mat3< complex<double> >& a, const mat3< complex<double> >& b); // m1 == m2 ?
  template bool operator != (const mat3< complex<double> >& a, const mat3< complex<double> >& b); // m1 != m2 ?

  template ostream& operator << (ostream& s, const mat3< complex<double> >& m); // output to stream
  template istream& operator >> (istream& s, mat3< complex<double> >& m); // input from stream

  template void swap(mat3< complex<double> >& a, mat3< complex<double> >& b); // swap m1 & m2
  template mat3< complex<double> > conj(const mat3< complex<double> >& a);
  template mat3< complex<double> > diagonal(const vec3< complex<double> >& v);
  template mat3< complex<double> > diagonal(const complex<double> x0, const complex<double> y1, const complex<double> z2);
  template complex<double> trace(const mat3< complex<double> >& a);

} /* namespace aux */



/* RTTI class of < mat4< complex<double> > > */



namespace aux {

  template class rtti< mat4< complex<double> > >; // RTTI of < mat4< complex<double> > >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< aux::mat4< complex<double> > >+;

#endif /* __MAKECINT__ */



/* 4x4 Matrix class and friends of < complex<double> > */



namespace aux {

  template class mat4< complex<double> >; // 4x4 Matrix of < complex<double> >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::mat4< complex<double> >+;

#endif /* __MAKECINT__ */

namespace aux {

  template mat4< complex<double> > operator - (const mat4< complex<double> >& a); // -m1
  template mat4< complex<double> > operator + (const mat4< complex<double> >& a, const mat4< complex<double> >& b); // m1 + m2
  template mat4< complex<double> > operator - (const mat4< complex<double> >& a, const mat4< complex<double> >& b); // m1 - m2
  template mat4< complex<double> > operator * (const mat4< complex<double> >& a, const mat4< complex<double> >& b); // m1 * m2
  template mat4< complex<double> > operator * (const mat4< complex<double> >& a, const complex<double> d); // m1 * 3.0
  template mat4< complex<double> > operator * (const complex<double> d, const mat4< complex<double> >& a); // 3.0 * m1
  template mat4< complex<double> > operator / (const mat4< complex<double> >& a, const complex<double> d); // m1 / 3.0
  template bool operator == (const mat4< complex<double> >& a, const mat4< complex<double> >& b); // m1 == m2 ?
  template bool operator != (const mat4< complex<double> >& a, const mat4< complex<double> >& b); // m1 != m2 ?

  template ostream& operator << (ostream& s, const mat4< complex<double> >& m); // output to stream
  template istream& operator >> (istream& s, mat4< complex<double> >& m); // input from stream

  template void swap(mat4< complex<double> >& a, mat4< complex<double> >& b); // swap m1 & m2
  template mat4< complex<double> > conj(const mat4< complex<double> >& a);
  template mat4< complex<double> > diagonal(const vec4< complex<double> >& v);
  template mat4< complex<double> > diagonal(const complex<double> x0, const complex<double> y1, const complex<double> z2, const complex<double> w3);
  template complex<double> trace(const mat4< complex<double> >& a);

} /* namespace aux */



/* 2D functions and 3D functions of < complex<double> > */



namespace aux {

  template mat2< complex<double> > identity1D< complex<double> >(void); // identity 1D
  template mat2< complex<double> > translation1D(const complex<double> & v); // translation 1D
  template mat2< complex<double> > scaling1D(const complex<double> & scaleVal); // scaling 1D
  template mat3< complex<double> > identity2D< complex<double> >(void); // identity 2D
  template mat3< complex<double> > translation2D(const vec2< complex<double> >& v); // translation 2D
#if __FIX_MAKROSCHROTT__ == 0
  template mat3< complex<double> > rotation2D(const vec2< complex<double> >& Center, const rtti< complex<double> >::value_type angleDeg); // rotation 2D
#endif /* __FIX_MAKROSCHROTT__ */
  template mat3< complex<double> > scaling2D(const vec2< complex<double> >& scaleVec); // scaling 2D
  template mat4< complex<double> > identity3D< complex<double> >(void); // identity 3D
  template mat4< complex<double> > translation3D(const vec3< complex<double> >& v); // translation 3D
#if __FIX_MAKROSCHROTT__ == 0
  template mat4< complex<double> > rotation3D(vec3< complex<double> > Axis, const rtti< complex<double> >::value_type angleDeg); // rotation 3D
#endif /* __FIX_MAKROSCHROTT__ */
  template mat4< complex<double> > scaling3D(const vec3< complex<double> >& scaleVec); // scaling 3D
  template mat4< complex<double> > perspective3D(const complex<double> d); // perspective 3D

} /* namespace aux */



/* AUX functions of < complex<double> > */



namespace aux {

  template vec2< complex<double> > MultiplyElementwise(const vec2< complex<double> >& v1, const vec2< complex<double> >& v2); // element by element multiplication
  template vec3< complex<double> > MultiplyElementwise(const vec3< complex<double> >& v1, const vec3< complex<double> >& v2); // element by element multiplication
  template vec4< complex<double> > MultiplyElementwise(const vec4< complex<double> >& v1, const vec4< complex<double> >& v2); // element by element multiplication

  template mat3< complex<double> > rotation2D(const mat2< complex<double> >& m); // convert 2x2 rotation matrix to 3x3
  template vec2< complex<double> > ExtractTranslation(const mat3< complex<double> >& pose); // extract translation vector
  template vec2< complex<double> > ExtractScaling(const mat3< complex<double> >& mat);
  template mat2< complex<double> > ExtractRotation(const mat3< complex<double> >& pose); // extract rotation matrix from transformation matrix
#if __FIX_MAKROSCHROTT__ == 0
  template rtti< complex<double> >::value_type PointToLine(const vec2< complex<double> >& point, const vec3< complex<double> >& line); // unsigned distance from a point to a line (2D)
#endif /* __FIX_MAKROSCHROTT__ */
  template mat4< complex<double> > rotation3D(const mat3< complex<double> >& m); // convert 3x3 rotation matrix to 4x4
  template vec3< complex<double> > ExtractTranslation(const mat4< complex<double> >& pose); // extract translation vector
  template vec3< complex<double> > ExtractScaling(const mat4< complex<double> >& mat);
  template mat3< complex<double> > ExtractRotation(const mat4< complex<double> >& pose); // extract rotation matrix from transformation matrix
#if __FIX_MAKROSCHROTT__ == 0
  template rtti< complex<double> >::value_type PointToPlane(const vec3< complex<double> >& point, const vec4< complex<double> >& plane); // unsigned distance from a point to a plane (3D)
#endif /* __FIX_MAKROSCHROTT__ */

  template complex<double> fmin(complex<double> x, complex<double> y);
  template complex<double> fmax(complex<double> x, complex<double> y);

  template vec2< complex<double> > fmin(const vec2< complex<double> >& v1, const vec2< complex<double> >& v2);
  template vec2< complex<double> > fmax(const vec2< complex<double> >& v1, const vec2< complex<double> >& v2);

  template vec3< complex<double> > fmin(const vec3< complex<double> >& v1, const vec3< complex<double> >& v2);
  template vec3< complex<double> > fmax(const vec3< complex<double> >& v1, const vec3< complex<double> >& v2);

  template vec4< complex<double> > fmin(const vec4< complex<double> >& v1, const vec4< complex<double> >& v2);
  template vec4< complex<double> > fmax(const vec4< complex<double> >& v1, const vec4< complex<double> >& v2);

} /* namespace aux */



/* RTTI class of < complex<long double> > */



namespace aux {

  template class rtti< complex<long double> >; // RTTI of < complex<long double> >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< complex<long double> >+;

#endif /* __MAKECINT__ */



/* RTTI class of < vec2< complex<long double> > > */



namespace aux {

  template class rtti< vec2< complex<long double> > >; // RTTI of < vec2< complex<long double> > >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< aux::vec2< complex<long double> > >+;

#endif /* __MAKECINT__ */



/* 2D Vector class and friends of < complex<long double> > */



namespace aux {

  template class vec2< complex<long double> >; // 2D Vector of < complex<long double> >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::vec2< complex<long double> >+;

#endif /* __MAKECINT__ */

namespace aux {

  template vec2< complex<long double> > operator - (const vec2< complex<long double> >& v); // -v1
  template vec2< complex<long double> > operator + (const vec2< complex<long double> >& a, const vec2< complex<long double> >& b); // v1 + v2
  template vec2< complex<long double> > operator - (const vec2< complex<long double> >& a, const vec2< complex<long double> >& b); // v1 - v2
  template vec2< complex<long double> > operator * (const vec2< complex<long double> >& a, const complex<long double> d); // v1 * 3.0
  template vec2< complex<long double> > operator * (const complex<long double> d, const vec2< complex<long double> >& a); // 3.0 * v1
  template vec2< complex<long double> > operator / (const vec2< complex<long double> >& a, const complex<long double> d); // v1 / 3.0
  template vec2< complex<long double> > operator * (const mat2< complex<long double> >& a, const vec2< complex<long double> >& v); // linear transform
  template vec2< complex<long double> > operator * (const mat3< complex<long double> >& a, const vec2< complex<long double> >& v); // M . v
  template vec2< complex<long double> > operator * (const vec2< complex<long double> >& v, const mat3< complex<long double> >& a); // v . M
  template vec3< complex<long double> > operator ^ (const vec2< complex<long double> >& a, const vec2< complex<long double> >& b); // cross product
  template complex<long double> operator * (const vec2< complex<long double> >& a, const vec2< complex<long double> >& b); // dot product
  template bool operator == (const vec2< complex<long double> >& a, const vec2< complex<long double> >& b); // v1 == v2 ?
  template bool operator != (const vec2< complex<long double> >& a, const vec2< complex<long double> >& b); // v1 != v2 ?

  template ostream& operator << (ostream& s, const vec2< complex<long double> >& v); // output to stream
  template istream& operator >> (istream& s, vec2< complex<long double> >& v); // input from stream

  template void swap(vec2< complex<long double> >& a, vec2< complex<long double> >& b); // swap v1 & v2
  template vec2< complex<long double> > min(const vec2< complex<long double> >& a, const vec2< complex<long double> >& b); // min(v1, v2)
  template vec2< complex<long double> > max(const vec2< complex<long double> >& a, const vec2< complex<long double> >& b); // max(v1, v2)
  template vec2< complex<long double> > prod(const vec2< complex<long double> >& a, const vec2< complex<long double> >& b); // term by term *
  template vec2< complex<long double> > conj(const vec2< complex<long double> >& a);

} /* namespace aux */



/* RTTI class of < vec3< complex<long double> > > */



namespace aux {

  template class rtti< vec3< complex<long double> > >; // RTTI of < vec3< complex<long double> > >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< aux::vec3< complex<long double> > >+;

#endif /* __MAKECINT__ */



/* 3D Vector class and friends of < complex<long double> > */



namespace aux {

  template class vec3< complex<long double> >; // 3D Vector of < complex<long double> >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::vec3< complex<long double> >+;

#endif /* __MAKECINT__ */

namespace aux {

  template vec3< complex<long double> > operator - (const vec3< complex<long double> >& v); // -v1
  template vec3< complex<long double> > operator + (const vec3< complex<long double> >& a, const vec3< complex<long double> >& b); // v1 + v2
  template vec3< complex<long double> > operator - (const vec3< complex<long double> >& a, const vec3< complex<long double> >& b); // v1 - v2
  template vec3< complex<long double> > operator * (const vec3< complex<long double> >& a, const complex<long double> d); // v1 * 3.0
  template vec3< complex<long double> > operator * (const complex<long double> d, const vec3< complex<long double> >& a); // 3.0 * v1
  template vec3< complex<long double> > operator / (const vec3< complex<long double> >& a, const complex<long double> d); // v1 / 3.0
  template vec3< complex<long double> > operator * (const mat3< complex<long double> >& a, const vec3< complex<long double> >& v); // linear transform
  template vec3< complex<long double> > operator * (const mat4< complex<long double> >& a, const vec3< complex<long double> >& v); // M . v
  template vec3< complex<long double> > operator * (const vec3< complex<long double> >& v, const mat4< complex<long double> >& a); // v . M
  template vec3< complex<long double> > operator ^ (const vec3< complex<long double> >& a, const vec3< complex<long double> >& b); // cross product
  template complex<long double> operator * (const vec3< complex<long double> >& a, const vec3< complex<long double> >& b); // dot product
  template bool operator == (const vec3< complex<long double> >& a, const vec3< complex<long double> >& b); // v1 == v2 ?
  template bool operator != (const vec3< complex<long double> >& a, const vec3< complex<long double> >& b); // v1 != v2 ?

  template ostream& operator << (ostream& s, const vec3< complex<long double> >& v); // output to stream
  template istream& operator >> (istream& s, vec3< complex<long double> >& v); // input from stream

  template void swap(vec3< complex<long double> >& a, vec3< complex<long double> >& b); // swap v1 & v2
  template vec3< complex<long double> > min(const vec3< complex<long double> >& a, const vec3< complex<long double> >& b); // min(v1, v2)
  template vec3< complex<long double> > max(const vec3< complex<long double> >& a, const vec3< complex<long double> >& b); // max(v1, v2)
  template vec3< complex<long double> > prod(const vec3< complex<long double> >& a, const vec3< complex<long double> >& b); // term by term *
  template vec3< complex<long double> > conj(const vec3< complex<long double> >& a);

} /* namespace aux */



/* RTTI class of < vec4< complex<long double> > > */



namespace aux {

  template class rtti< vec4< complex<long double> > >; // RTTI of < vec4< complex<long double> > >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< aux::vec4< complex<long double> > >+;

#endif /* __MAKECINT__ */



/* 4D Vector class and friends of < complex<long double> > */



namespace aux {

  template class vec4< complex<long double> >; // 4D Vector of < complex<long double> >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::vec4< complex<long double> >+;

#endif /* __MAKECINT__ */

namespace aux {

  template vec4< complex<long double> > operator - (const vec4< complex<long double> >& v); // -v1
  template vec4< complex<long double> > operator + (const vec4< complex<long double> >& a, const vec4< complex<long double> >& b); // v1 + v2
  template vec4< complex<long double> > operator - (const vec4< complex<long double> >& a, const vec4< complex<long double> >& b); // v1 - v2
  template vec4< complex<long double> > operator * (const vec4< complex<long double> >& a, const complex<long double> d); // v1 * 3.0
  template vec4< complex<long double> > operator * (const complex<long double> d, const vec4< complex<long double> >& a); // 3.0 * v1
  template vec4< complex<long double> > operator / (const vec4< complex<long double> >& a, const complex<long double> d); // v1 / 3.0
  template vec4< complex<long double> > operator * (const mat4< complex<long double> >& a, const vec4< complex<long double> >& v); // M . v
  template vec4< complex<long double> > operator * (const vec4< complex<long double> >& v, const mat4< complex<long double> >& a); // v . M
  template complex<long double> operator * (const vec4< complex<long double> >& a, const vec4< complex<long double> >& b); // dot product
  template bool operator == (const vec4< complex<long double> >& a, const vec4< complex<long double> >& b); // v1 == v2 ?
  template bool operator != (const vec4< complex<long double> >& a, const vec4< complex<long double> >& b); // v1 != v2 ?

  template ostream& operator << (ostream& s, const vec4< complex<long double> >& v); // output to stream
  template istream& operator >> (istream& s, vec4< complex<long double> >& v); // input from stream

  template void swap(vec4< complex<long double> >& a, vec4< complex<long double> >& b); // swap v1 & v2
  template vec4< complex<long double> > min(const vec4< complex<long double> >& a, const vec4< complex<long double> >& b); // min(v1, v2)
  template vec4< complex<long double> > max(const vec4< complex<long double> >& a, const vec4< complex<long double> >& b); // max(v1, v2)
  template vec4< complex<long double> > prod(const vec4< complex<long double> >& a, const vec4< complex<long double> >& b); // term by term *
  template vec4< complex<long double> > conj(const vec4< complex<long double> >& a);

} /* namespace aux */



/* RTTI class of < mat2< complex<long double> > > */



namespace aux {

  template class rtti< mat2< complex<long double> > >; // RTTI of < mat2< complex<long double> > >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< aux::mat2< complex<long double> > >+;

#endif /* __MAKECINT__ */



/* 2x2 Matrix class and friends of < complex<long double> > */



namespace aux {

  template class mat2< complex<long double> >; // 2x2 Matrix of < complex<long double> >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::mat2< complex<long double> >+;

#endif /* __MAKECINT__ */

namespace aux {

  template mat2< complex<long double> > operator - (const mat2< complex<long double> >& a); // -m1
  template mat2< complex<long double> > operator + (const mat2< complex<long double> >& a, const mat2< complex<long double> >& b); // m1 + m2
  template mat2< complex<long double> > operator - (const mat2< complex<long double> >& a, const mat2< complex<long double> >& b); // m1 - m2
  template mat2< complex<long double> > operator * (const mat2< complex<long double> >& a, const mat2< complex<long double> >& b); // m1 * m2
  template mat2< complex<long double> > operator * (const mat2< complex<long double> >& a, const complex<long double> d); // m1 * 3.0
  template mat2< complex<long double> > operator * (const complex<long double> d, const mat2< complex<long double> >& a); // 3.0 * m1
  template mat2< complex<long double> > operator / (const mat2< complex<long double> >& a, const complex<long double> d); // m1 / 3.0
  template bool operator == (const mat2< complex<long double> >& a, const mat2< complex<long double> >& b); // m1 == m2 ?
  template bool operator != (const mat2< complex<long double> >& a, const mat2< complex<long double> >& b); // m1 != m2 ?

  template ostream& operator << (ostream& s, const mat2< complex<long double> >& m); // output to stream
  template istream& operator >> (istream& s, mat2< complex<long double> >& m); // input from stream

  template void swap(mat2< complex<long double> >& a, mat2< complex<long double> >& b); // swap m1 & m2
  template mat2< complex<long double> > conj(const mat2< complex<long double> >& a);
  template mat2< complex<long double> > diagonal(const vec2< complex<long double> >& v);
  template mat2< complex<long double> > diagonal(const complex<long double> x0, const complex<long double> y1);
  template complex<long double> trace(const mat2< complex<long double> >& a);

} /* namespace aux */



/* RTTI class of < mat3< complex<long double> > > */



namespace aux {

  template class rtti< mat3< complex<long double> > >; // RTTI of < mat3< complex<long double> > >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< aux::mat3< complex<long double> > >+;

#endif /* __MAKECINT__ */



/* 3x3 Matrix class and friends of < complex<long double> > */



namespace aux {

  template class mat3< complex<long double> >; // 3x3 Matrix of < complex<long double> >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::mat3< complex<long double> >+;

#endif /* __MAKECINT__ */

namespace aux {

  template mat3< complex<long double> > operator - (const mat3< complex<long double> >& a); // -m1
  template mat3< complex<long double> > operator + (const mat3< complex<long double> >& a, const mat3< complex<long double> >& b); // m1 + m2
  template mat3< complex<long double> > operator - (const mat3< complex<long double> >& a, const mat3< complex<long double> >& b); // m1 - m2
  template mat3< complex<long double> > operator * (const mat3< complex<long double> >& a, const mat3< complex<long double> >& b); // m1 * m2
  template mat3< complex<long double> > operator * (const mat3< complex<long double> >& a, const complex<long double> d); // m1 * 3.0
  template mat3< complex<long double> > operator * (const complex<long double> d, const mat3< complex<long double> >& a); // 3.0 * m1
  template mat3< complex<long double> > operator / (const mat3< complex<long double> >& a, const complex<long double> d); // m1 / 3.0
  template bool operator == (const mat3< complex<long double> >& a, const mat3< complex<long double> >& b); // m1 == m2 ?
  template bool operator != (const mat3< complex<long double> >& a, const mat3< complex<long double> >& b); // m1 != m2 ?

  template ostream& operator << (ostream& s, const mat3< complex<long double> >& m); // output to stream
  template istream& operator >> (istream& s, mat3< complex<long double> >& m); // input from stream

  template void swap(mat3< complex<long double> >& a, mat3< complex<long double> >& b); // swap m1 & m2
  template mat3< complex<long double> > conj(const mat3< complex<long double> >& a);
  template mat3< complex<long double> > diagonal(const vec3< complex<long double> >& v);
  template mat3< complex<long double> > diagonal(const complex<long double> x0, const complex<long double> y1, const complex<long double> z2);
  template complex<long double> trace(const mat3< complex<long double> >& a);

} /* namespace aux */



/* RTTI class of < mat4< complex<long double> > > */



namespace aux {

  template class rtti< mat4< complex<long double> > >; // RTTI of < mat4< complex<long double> > >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::rtti< aux::mat4< complex<long double> > >+;

#endif /* __MAKECINT__ */



/* 4x4 Matrix class and friends of < complex<long double> > */



namespace aux {

  template class mat4< complex<long double> >; // 4x4 Matrix of < complex<long double> >

} /* namespace aux */

#ifdef __MAKECINT__

#pragma link C++ class aux::mat4< complex<long double> >+;

#endif /* __MAKECINT__ */

namespace aux {

  template mat4< complex<long double> > operator - (const mat4< complex<long double> >& a); // -m1
  template mat4< complex<long double> > operator + (const mat4< complex<long double> >& a, const mat4< complex<long double> >& b); // m1 + m2
  template mat4< complex<long double> > operator - (const mat4< complex<long double> >& a, const mat4< complex<long double> >& b); // m1 - m2
  template mat4< complex<long double> > operator * (const mat4< complex<long double> >& a, const mat4< complex<long double> >& b); // m1 * m2
  template mat4< complex<long double> > operator * (const mat4< complex<long double> >& a, const complex<long double> d); // m1 * 3.0
  template mat4< complex<long double> > operator * (const complex<long double> d, const mat4< complex<long double> >& a); // 3.0 * m1
  template mat4< complex<long double> > operator / (const mat4< complex<long double> >& a, const complex<long double> d); // m1 / 3.0
  template bool operator == (const mat4< complex<long double> >& a, const mat4< complex<long double> >& b); // m1 == m2 ?
  template bool operator != (const mat4< complex<long double> >& a, const mat4< complex<long double> >& b); // m1 != m2 ?

  template ostream& operator << (ostream& s, const mat4< complex<long double> >& m); // output to stream
  template istream& operator >> (istream& s, mat4< complex<long double> >& m); // input from stream

  template void swap(mat4< complex<long double> >& a, mat4< complex<long double> >& b); // swap m1 & m2
  template mat4< complex<long double> > conj(const mat4< complex<long double> >& a);
  template mat4< complex<long double> > diagonal(const vec4< complex<long double> >& v);
  template mat4< complex<long double> > diagonal(const complex<long double> x0, const complex<long double> y1, const complex<long double> z2, const complex<long double> w3);
  template complex<long double> trace(const mat4< complex<long double> >& a);

} /* namespace aux */



/* 2D functions and 3D functions of < complex<long double> > */



namespace aux {

  template mat2< complex<long double> > identity1D< complex<long double> >(void); // identity 1D
  template mat2< complex<long double> > translation1D(const complex<long double> & v); // translation 1D
  template mat2< complex<long double> > scaling1D(const complex<long double> & scaleVal); // scaling 1D
  template mat3< complex<long double> > identity2D< complex<long double> >(void); // identity 2D
  template mat3< complex<long double> > translation2D(const vec2< complex<long double> >& v); // translation 2D
#if __FIX_MAKROSCHROTT__ == 0
  template mat3< complex<long double> > rotation2D(const vec2< complex<long double> >& Center, const rtti< complex<long double> >::value_type angleDeg); // rotation 2D
#endif /* __FIX_MAKROSCHROTT__ */
  template mat3< complex<long double> > scaling2D(const vec2< complex<long double> >& scaleVec); // scaling 2D
  template mat4< complex<long double> > identity3D< complex<long double> >(void); // identity 3D
  template mat4< complex<long double> > translation3D(const vec3< complex<long double> >& v); // translation 3D
#if __FIX_MAKROSCHROTT__ == 0
  template mat4< complex<long double> > rotation3D(vec3< complex<long double> > Axis, const rtti< complex<long double> >::value_type angleDeg); // rotation 3D
#endif /* __FIX_MAKROSCHROTT__ */
  template mat4< complex<long double> > scaling3D(const vec3< complex<long double> >& scaleVec); // scaling 3D
  template mat4< complex<long double> > perspective3D(const complex<long double> d); // perspective 3D

} /* namespace aux */



/* AUX functions of < complex<long double> > */



namespace aux {

  template vec2< complex<long double> > MultiplyElementwise(const vec2< complex<long double> >& v1, const vec2< complex<long double> >& v2); // element by element multiplication
  template vec3< complex<long double> > MultiplyElementwise(const vec3< complex<long double> >& v1, const vec3< complex<long double> >& v2); // element by element multiplication
  template vec4< complex<long double> > MultiplyElementwise(const vec4< complex<long double> >& v1, const vec4< complex<long double> >& v2); // element by element multiplication

  template mat3< complex<long double> > rotation2D(const mat2< complex<long double> >& m); // convert 2x2 rotation matrix to 3x3
  template vec2< complex<long double> > ExtractTranslation(const mat3< complex<long double> >& pose); // extract translation vector
  template vec2< complex<long double> > ExtractScaling(const mat3< complex<long double> >& mat);
  template mat2< complex<long double> > ExtractRotation(const mat3< complex<long double> >& pose); // extract rotation matrix from transformation matrix
#if __FIX_MAKROSCHROTT__ == 0
  template rtti< complex<long double> >::value_type PointToLine(const vec2< complex<long double> >& point, const vec3< complex<long double> >& line); // unsigned distance from a point to a line (2D)
#endif /* __FIX_MAKROSCHROTT__ */
  template mat4< complex<long double> > rotation3D(const mat3< complex<long double> >& m); // convert 3x3 rotation matrix to 4x4
  template vec3< complex<long double> > ExtractTranslation(const mat4< complex<long double> >& pose); // extract translation vector
  template vec3< complex<long double> > ExtractScaling(const mat4< complex<long double> >& mat);
  template mat3< complex<long double> > ExtractRotation(const mat4< complex<long double> >& pose); // extract rotation matrix from transformation matrix
#if __FIX_MAKROSCHROTT__ == 0
  template rtti< complex<long double> >::value_type PointToPlane(const vec3< complex<long double> >& point, const vec4< complex<long double> >& plane); // unsigned distance from a point to a plane (3D)
#endif /* __FIX_MAKROSCHROTT__ */

  template complex<long double> fmin(complex<long double> x, complex<long double> y);
  template complex<long double> fmax(complex<long double> x, complex<long double> y);

  template vec2< complex<long double> > fmin(const vec2< complex<long double> >& v1, const vec2< complex<long double> >& v2);
  template vec2< complex<long double> > fmax(const vec2< complex<long double> >& v1, const vec2< complex<long double> >& v2);

  template vec3< complex<long double> > fmin(const vec3< complex<long double> >& v1, const vec3< complex<long double> >& v2);
  template vec3< complex<long double> > fmax(const vec3< complex<long double> >& v1, const vec3< complex<long double> >& v2);

  template vec4< complex<long double> > fmin(const vec4< complex<long double> >& v1, const vec4< complex<long double> >& v2);
  template vec4< complex<long double> > fmax(const vec4< complex<long double> >& v1, const vec4< complex<long double> >& v2);

} /* namespace aux */



#endif /* __MAKECINT__ || __ALGEBRA3_CXX_DEBUG__ */

#endif /* __ALGEBRA3_CXX__ */

/* End of file algebra3.cxx */

