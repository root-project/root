#ifndef TEMPLATE_H
#define TEMPLATE_H

#include <TObject.h>

template <class GeometricalType,
          class EnergeticType,
          class CompositeType>
class Template : public GeometricalType,
                 public EnergeticType,
                 public CompositeType
{
public:
  virtual ~Template() {}
  ClassDef(Template,1);
};

template <class T> class MyTest {
public:
   virtual ~MyTest() {}
   ClassDef(MyTest,1);
};

template <> class MyTest<int> {
public:
   virtual ~MyTest() {}
   ClassDef(MyTest,1);
};

#endif

#ifndef CLASSA_H
#define CLASSA_H

class ClassA
{
public:
  virtual ~ClassA() {}
  ClassDef(ClassA,1)
};

#endif 
#ifndef CLASSB_H 
#define CLASSB_H

class ClassB
{
public:
  virtual ~ClassB() {}
  ClassDef(ClassB,1);
};

#endif 
#ifndef CLASSC_H
#define CLASSC_H

class ClassC
{
public:
   virtual ~ClassC() {}
  ClassDef(ClassC,1);
};

#endif



#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ nestedclass;
#pragma link C++ nestedtypedef;

#pragma link C++ class Template<ClassA,ClassB,ClassC>+;

#pragma link C++ class MyTest<double>+;
#pragma link C++ class MyTest<int>+;

#pragma link C++ class ClassA+;
#pragma link C++ class ClassB+;
#pragma link C++ class ClassC+;

#endif
