/*
  File: roottest/python/cpp/Template.C
  Author: Wim Lavrijsen@lbl.gov
  Created: 01/07/08
  Last: 03/13/15
*/

#include <string>
#include <vector>


template< class T >
class MyTemplatedClass {
public:
   T m_b;
};

template< class T >
struct MyTemplatedClass2 {
   void set( const T& /* v */) {}
};    
typedef MyTemplatedClass2< std::string > MyTemplateTypedef;

template< class T >
T MyTemplatedFunction( T t ) { return t; }

namespace MyNamespace {
   template <typename T>
   T MyTemplatedFunctionNamespace(T t) { return t; }
};

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
// not overloaded
   template< class B > long GetSize();

   long GetCharSize()   { return sizeof(char); }
   long GetIntSize()    { return sizeof(int); }
   long GetLongSize()   { return sizeof(long); }
   long GetFloatSize()  { return sizeof(float); }
   long GetDoubleSize() { return sizeof(double); }

   long GetVectorOfDoubleSize() { return sizeof( std::vector< double > ); }

// cross-args
   template< class A, class B > long GetSize2( const B&, const A& );

// overloaded (note limitations as types must be distinguishable)
   template< class B > long GetSizeOL( const B& );
   long GetSizeOL( const long& ) { return -sizeof(long); }

// the following actually means that it is not possible to specify template
// arguments in python with a string, as it will match this overload ...
   long GetSizeOL( const std::string& s ) { return -s.size(); }

   // Not explicitly instantiated below
   template<class A> long GetSizeNEI(const A&);
   // Non-templated overload
   long GetSizeNEI() { return 1L; }
};

template< class B >
long MyTemplatedMethodClass::GetSize() {
   return sizeof(B);
}

template< class A, class B >
long MyTemplatedMethodClass::GetSize2( const B&, const A& ) {
   return sizeof(A) - sizeof(B);
}

template< class B >
long MyTemplatedMethodClass::GetSizeOL( const B& ) {
   return sizeof(B);
}

template<class A>
long MyTemplatedMethodClass::GetSizeNEI(const A&)
{
   return sizeof(A);
}

template long MyTemplatedMethodClass::GetSize< char >();
template long MyTemplatedMethodClass::GetSize< int >();
template long MyTemplatedMethodClass::GetSize< long >();
template long MyTemplatedMethodClass::GetSize< float >();
template long MyTemplatedMethodClass::GetSize< double >();

template long MyTemplatedMethodClass::GetSize2< char, char >( const char&, const char& );

template long MyTemplatedMethodClass::GetSizeOL< float >( const float& );

typedef std::vector< double > MyDoubleVector_t;
template long MyTemplatedMethodClass::GetSize< MyDoubleVector_t >();

#ifdef __CINT__
#pragma link C++ class MyTemplatedMethodClass;
#endif
