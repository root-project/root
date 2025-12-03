#ifndef new_noclassdef_template_h
#define new_noclassdef_template_h

#include "TObject.h"

#include <vector>

template <class T> class MyTemplate : public TObject {
 public:
  T variable;
  std::vector<int> vars;

  MyTemplate(T a) { variable = a; }
  MyTemplate() {}

  ClassDefOverride(MyTemplate,1)
};

template <>
class MyTemplate <const double*> : public TObject {
 public:
  double variable;
  std::vector<int> vars;

  MyTemplate(const double* a) { variable = *a; }
  MyTemplate() {}

  ClassDefOverride(MyTemplate<const double*>, 2)
};

template <class T1, class T2> class MyPairTemplate : public TObject {
 public:
  T1 var1;
  T2 var2;

  MyPairTemplate(T1 a, T2 b) : var1(a), var2(b) {}
  MyPairTemplate() {}
  ~MyPairTemplate() override {}

  ClassDefOverride(MyPairTemplate,1)
};

template <>
class MyPairTemplate<int, double> : public TObject {
 public:
  float var1;
  float var2;

  MyPairTemplate(int a, double b) : var1(a), var2(b) {}
  MyPairTemplate() {}
  ~MyPairTemplate() override {}

  typedef MyPairTemplate<int, double> type;
  ClassDefOverride(type, 2)
};

void template_driver();

#endif
