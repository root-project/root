//--------------------------------------------------------------------*- C++ -*-
// file:   DataModelV2.h
// author: Lukasz Janyst <ljanyst@cern.ch>
//------------------------------------------------------------------------------

#ifndef DataModelV2_h
#define DataModelV2_h

#include <utility>
#include <vector>

class ClassAIns2
{
   public:
      ClassAIns2(): m_a( 12 ), m_b( 32.23 ) {}
   private:
      int    m_a;
      double m_b;
};

class ClassABase2
{
   public:
      ClassABase2(): m_a( 3 ), m_b( 123.33 ) {}
      virtual ~ClassABase2() {};
   private:
      int    m_a;
      double m_b;
};

class ClassA2: public ClassABase2
{
   public:
      ClassA2(): m_c( 65.22 ), m_e( 8 ) {}
      virtual ~ClassA2() {}
   private:
      float      m_c;
      ClassAIns2 m_d;
      int        m_e;
};

class ClassB: public ClassA2
{
   public:
      ClassB(): m_f( 34 ), m_g( 12.22 ) {}
      virtual ~ClassB() {};
   private:
      short  m_f;
      float  m_g;
};

class ClassC: public ClassABase2
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
   ClassAIns2 m_d;
   int        m_e;
};

struct _dummy
{
   std::vector<float>                   a1;
   std::pair<int,float>                 a2;
   std::vector<ClassA2>                 a3;
   std::vector<std::pair<int, float> >  a4;
   std::vector<ClassA2*>                a5;
};

#endif
