/*
  File: roottest/python/cpp/Template.C
  Author: Wim Lavrijsen@lbl.gov
  Created: 01/07/08
  Last: 11/07/08
*/

#include <vector>

template< class T >
class MyTemplatedClass {
public:
   T m_b;
};

template< class T >
T MyTemplatedFunction( T t ) { return t; }

#ifdef __CINT__
#pragma link C++ class MyTemplatedClass< vector< float > >;
#pragma link C++ function MyTemplatedFunction< int >( int );
#pragma link C++ function MyTemplatedFunction< double >( double );
#endif

template class MyTemplatedClass< vector< float > >;
template int MyTemplatedFunction< int >( int );
template double MyTemplatedFunction< double >( double );

class MyTemplatedMethodClass {
public:
   template< class B > long GetSize();

   long GetCharSize()   { return sizeof(char); }
   long GetIntSize()    { return sizeof(int); }
   long GetLongSize()   { return sizeof(long); }
   long GetFloatSize()  { return sizeof(float); }
   long GetDoubleSize() { return sizeof(double); }

   long GetVectorOfDoubleSize() { return sizeof( std::vector< double > ); }
};

template< class B >
long MyTemplatedMethodClass::GetSize() {
   return sizeof(B);
}

template long MyTemplatedMethodClass::GetSize< char >();
template long MyTemplatedMethodClass::GetSize< int >();
template long MyTemplatedMethodClass::GetSize< long >();
template long MyTemplatedMethodClass::GetSize< float >();
template long MyTemplatedMethodClass::GetSize< double >();

typedef std::vector< double > MyDoubleVector_t;
template long MyTemplatedMethodClass::GetSize< MyDoubleVector_t >();

#ifdef __CINT__
#pragma link C++ class MyTemplatedMethodClass;
#endif
