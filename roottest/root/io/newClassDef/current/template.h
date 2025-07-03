#include "TObject.h"
#include "vector"

template <class T> class MyTemplate : public TObject {
 public:
  T variable;
  std::vector<int> vars;

  MyTemplate(T a) { variable = a; }
  MyTemplate() {}

  ClassDefT(MyTemplate,1)
};

ClassDefT2(MyTemplate,T)

template <>
class MyTemplate <const double*> : public TObject {
 public:
  double variable;
  std::vector<int> vars;

  MyTemplate(const double* a) { variable = *a; }
  MyTemplate() {}

#ifdef R__WIN32
  typedef MyTemplate<const double*> type;
  ClassDefOverride(type,2)
#else
  ClassDefOverride(MyTemplate<const double*>,2)
#endif
};

template <class T1, class T2> class MyPairTemplate : public TObject {
 public:
  T1 var1;
  T2 var2;

  MyPairTemplate(T1 a, T2 b) : var1(a), var2(b) {}
  MyPairTemplate() {}
  ~MyPairTemplate() {}

  ClassDefT(MyPairTemplate,1)
};

ClassDef2T2(MyPairTemplate,T1,T2)


void template_driver();

