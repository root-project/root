//--------------------------------------------------------------------*- C++ -*-
// file:   DataModelV2.h
// author: Lukasz Janyst <ljanyst@cern.ch>
//------------------------------------------------------------------------------

#include <utility>
#include <vector>
#include <Riostream.h>

class ClassAIns
{
public:
   ClassAIns(): m_a( 12 ), m_b( 32.23 ), m_unit(-1) {}
private:
   int    m_a;
   double m_b;
   int    m_unit;
public:
   void SetUnit(int unit) { m_unit = unit; }
};

class ClassABase
{
   public:
      ClassABase(): m_a( 3 ), m_b( 123.33 ) {}
      virtual ~ClassABase() {};
   private:
      int    m_a;
      double m_b;
};

class ClassA: public ClassABase
{
public:
   ClassA(): m_c( 65.22 ), m_e( 8 ), m_unit(2), m_md_set(false) {}
   virtual ~ClassA() {}
private:
   float      m_c;
   ClassAIns  m_d;
   int        m_e;
   int        m_unit;
   bool       m_md_set; //! To insure proper I/O transfer of m_unit into m_d
public:
   void SetMdUnit(int unit) {
      m_d.SetUnit(unit);
   }
   void Print() {
      cout << "ClassA: \n";
      cout << "   m_md_set: " << m_md_set << ' ' << (void*)this << ' ' << (char*)(&m_md_set)-(char*)this << ' ' << (void*)&m_md_set << '\n';
   }
};

class ClassB: public ClassA
{
   public:
      ClassB(): m_f( 34 ), m_g( 12.22 ) {}
      virtual ~ClassB() {};
   private:
      short  m_f;
      float  m_g;
};

class ClassC: public ClassABase
{
   public:
      ClassC(): m_f( 74.22 ), m_g( 199.22 ) {}
      virtual ~ClassC() {};
   private:
      double m_f;
      float  m_g;
//      std::vector<std::pair<float, int> > m_h;
};

class ClassD 
{
public:
   ClassD(): m_c( 65.22 ), m_e( 8 ) {}
   virtual ~ClassD() {}
private:
   float      m_c;
   ClassAIns  m_d;
   int        m_e;
};      

struct _dummy
{
   std::vector<float>                   _a1;
   std::pair<int,float>                 _a2;
   std::vector<ClassA>                  _a3;
   std::vector<std::pair<int, float> >  _a4;
   std::vector<ClassA*>                 _a5;
   std::vector<double>                  a1;
   std::pair<int,double>                a2;
   std::vector<std::pair<int, double> > a3;
   std::vector<ClassA>                  a4;
   std::vector<ClassA*>                 a5;
   std::vector<ClassB>                  a6;
   std::vector<ClassB*>                 a7;
   std::vector<ClassC>                  a8;
   std::vector<ClassC*>                 a9;
   std::vector<ClassD>                  a10;
   std::vector<ClassD*>                 a11;
};


