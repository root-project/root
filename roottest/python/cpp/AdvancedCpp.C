/*
  File: roottest/python/cpp/AdvancedCpp.C
  Author: WLavrijsen@lbl.gov
  Created: 06/04/05
  Last: 05/01/15
*/

#include "TString.h"
#include <sstream>
#include <string>
#include <vector>


class PR_A {
public:
   PR_A() { m_a = 1; m_da = 1.1; }
   virtual ~PR_A() {}
   virtual int GetValue() = 0;

public:
   int m_a;
   double m_da;
};

class PR_B : public virtual PR_A {
public:
   PR_B() { m_b = 2; m_db = 2.2;}
   virtual int GetValue() { return m_b; }

public:
   int m_b;
   double m_db;
};

// NOTE: class PR_C : public virtual PR_A, public virtual PR_B
//  causes gROOT->GetClass() to fail ...
class PR_C : public virtual PR_B, public virtual PR_A {
public:
   PR_C() { m_c = 3; }
   virtual int GetValue() { return m_c; }

public:
   int m_c;
};

class PR_D : public virtual PR_C, public virtual PR_A {
public:
   PR_D() { m_d = 4; }
   virtual int GetValue() { return m_d; }

public:
   int m_d;
};

int GetA( PR_A& a ) { return a.m_a; }
int GetB( PR_B& b ) { return b.m_b; }
int GetC( PR_C& c ) { return c.m_c; }
int GetD( PR_D& d ) { return d.m_d; }

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

template class T1<int>;
template class T2< T1< int > >;

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
   virtual ~MyAbstractClass() {}
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

class RefTesterNoAssign {
public:
   RefTesterNoAssign& operator[]( int ) { return *this; }

private:
   RefTesterNoAssign& operator=( const RefTesterNoAssign& ) {
      return *this;
   }
};

template class std::vector< RefTester >;


// helper for math conversions
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

namespace ComparableSpace {
   class NSComparable {
   };

   bool operator==( const NSComparable& c1, const NSComparable& c2 )
   {
   // does the opposite of a (default PyROOT) pointer comparison
      return &c1 != &c2;
   }

   bool operator!=( const NSComparable& c1, const NSComparable& c2 )
   {
   // does the opposite of a (default PyROOT) pointer comparison
      return &c1 == &c2;
   }
}

// a couple of globals
double myGlobalDouble = 12.;
double myGlobalArray[500];


// helper class for life-line testing
class SomeClassWithData {
public:
   class SomeData {
   public:
      SomeData()                  { ++s_numData; }
      SomeData( const SomeData& ) { ++s_numData; }
      ~SomeData()                 { --s_numData; }

      static int s_numData;
   };

   SomeClassWithData GimeCopy() {
      return *this;
   }

   const SomeData& GimeData() const {
      return m_data;
   }

   SomeData m_data;
};

int SomeClassWithData::SomeData::s_numData = 0;


// helpers for containers of pointers testing
class PR_ValClass {
public:
   std::string m_val;
};

size_t fillVect( std::vector< PR_ValClass* >& v ) {
   v.clear();
   PR_ValClass* m = new PR_ValClass();
   m->m_val = "aap";
   v.push_back(m);
   return v.size();
}


// helpers for custom new/delete
template <class T>
class PR_StaticStuff {
private:
   static T s_data;

public:
   static void set(const T& v) {
      s_data = v;
   }
   static std::string describe() {
      std::ostringstream s;
      s << "StaticStuff::s_data -> " << s_data;
      return s.str();
   }
};

template <class T> T PR_StaticStuff<T>::s_data = T(999);

struct PR_CustomNewDeleteClass {
   PR_CustomNewDeleteClass() : m_data(314) {}
   ~PR_CustomNewDeleteClass() {}
   static void* operator new(size_t size) {
      void *p = ::operator new(size);
      PR_StaticStuff<int>::set(123);
      return p;
   }
   static void operator delete(void* p) {
      PR_StaticStuff<int>::set(321);
      ::operator delete(p);
   }
   int m_data;
};

// more globals
TString PR_Lumi1 = "::1 C++ global lumi";
TString PR_GetLumi1() { return PR_Lumi1; }
TString PR_Lumi2 = "::2 C++ global lumi";
TString PR_GetLumi2() { return PR_Lumi2; }

namespace NS_PR_Lumi {
   TString PR_Lumi1 = "NS::1 C++ global lumi";
   TString PR_GetLumi1() { return PR_Lumi1; }
   TString PR_Lumi2 = "NS::2 C++ global lumi";
   TString PR_GetLumi2() { return PR_Lumi2; }
}


// copy constructor orderings
class MyCopyingClass1 {
public:
   MyCopyingClass1() : m_d1(-1.), m_d2(-1) {}
   MyCopyingClass1(const double d1, const double d2 = 42.) : m_d1(d1), m_d2(d2) {}
// no user-provided cctor
   operator double() { return m_d1+m_d2; }
   double m_d1, m_d2;
};

class MyCopyingClass2 {
public:
   MyCopyingClass2() : m_d1(-1.), m_d2(-1) {}
// cctor after ctor with default
   MyCopyingClass2(const double d1, const double d2 = 42.) : m_d1(d1), m_d2(d2) {}
   MyCopyingClass2(const MyCopyingClass2& s) : m_d1(s.m_d1), m_d2(s.m_d2) {}
   operator double() { return m_d1+m_d2; }
   double m_d1, m_d2;
};

class MyCopyingClass3 {
public:
   MyCopyingClass3() : m_d1(-1.), m_d2(-1) {}
// cctor before ctor with default
   MyCopyingClass3(const MyCopyingClass3& s) : m_d1(s.m_d1), m_d2(s.m_d2) {}
   MyCopyingClass3(const double d1, const double d2 = 42.) : m_d1(d1), m_d2(d2) {}
   operator double() { return m_d1+m_d2; }
   double m_d1, m_d2;
};
