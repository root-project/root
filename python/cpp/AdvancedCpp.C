/*
  File: roottest/python/cpp/AdvancedCpp.C
  Author: WLavrijsen@lbl.gov
  Created: 06/04/05
  Last: 07/19/10
*/

#include <vector>

class A {
public:
   A() { m_a = 1; m_da = 1.1; }
   virtual ~A() {}
   virtual int GetValue() = 0;

public:
   int m_a;
   double m_da;
};

class B : public virtual A {
public:
   B() { m_b = 2; m_db = 2.2;}
   virtual int GetValue() { return m_b; }

public:
   int m_b;
   double m_db;
};

// NOTE: class C : public virtual A, public virtual B
//  causes gROOT->GetClass() to fail ...
class C : public virtual B, public virtual A {
public:
   C() { m_c = 3; }
   virtual int GetValue() { return m_c; }

public:
   int m_c;
};

class D : public virtual C, public virtual A {
public:
   D() { m_d = 4; }
   virtual int GetValue() { return m_d; }

public:
   int m_d;
};

int GetA( A& a ) { return a.m_a; }
int GetB( B& b ) { return b.m_b; }
int GetC( C& c ) { return c.m_c; }
int GetD( D& d ) { return d.m_d; }

template< typename T >
class T1 {
public:
   T1( T t = T(0) ) : m_t1( t ) {}
   T value() { return m_t1; }

public:
   T m_t1;
};

template< typename T >
class T2 {
public:
   T m_t2;
};

namespace {
   T1< int > tt1;
   T2< T1< int > > tt2;
}

// helpers for checking pass-by-ref
void SetIntThroughRef( Int_t& i, Int_t val ) { i = val; }
Int_t PassIntThroughConstRef( const Int_t& i ) { return i; }
void SetLongThroughRef( Long_t& l, Long_t val ) { l = val; }
Long_t PassLongThroughConstRef( const Long_t& l ) { return l; }
void SetDoubleThroughRef( Double_t& d, Double_t val ) { d = val; }
Double_t PassDoubleThroughConstRef( const Double_t& d ) { return d; }

// abstract class should not be instantiatable
class MyAbstractClass {
public:
   virtual void MyVirtualMethod() = 0;
};

class MyConcreteClass : public MyAbstractClass {
public:
   virtual void MyVirtualMethod() {}
};


// helpers for assignment by-ref
class RefTester {
public:
   RefTester() : m_i( -99 ) {}
   RefTester( int i ) : m_i( i ) {}
   RefTester( const RefTester& s ) : m_i( s.m_i ) {}
   RefTester& operator=( const RefTester& s ) {
      if ( &s != this ) m_i = s.m_i;
      return *this;
   }
   ~RefTester() {}

public:
   int m_i;
};

#ifdef __CINT__
#pragma link C++ class std::vector< RefTester >;
#endif

template class std::vector< RefTester >;


// helper for math conversionts
class Convertible {
public:
   Convertible() : m_i( -99 ), m_d( -99. ) {}

   operator Int_t() { return m_i; }
   operator long() { return m_i; }
   operator double() { return m_d; }

public:
   int m_i;
   double m_d;
};


class Comparable {
};

bool operator==( const Comparable& c1, const Comparable& c2 )
{
// does the opposite of a (default PyROOT) pointer comparison
   return &c1 != &c2;
}

bool operator!=( const Comparable& c1, const Comparable& c2 )
{
// does the opposite of a (default PyROOT) pointer comparison
   return &c1 == &c2;
}

#ifdef __CINT__
#pragma link C++ function operator==( const Comparable&, const Comparable& );
#pragma link C++ function operator!=( const Comparable&, const Comparable& );
#endif
