#include "TObject.h"
#include "vector"

template <class T> class MyTemplate : public TObject {
 public:
  T variable; 
#ifdef R__GLOBALSTL
  vector<int> vars;
#else
  std::vector<int> vars;
#endif
  
  MyTemplate(T a) { variable = a; };
  MyTemplate() {};

  ClassDef(MyTemplate,1)
};

template <>
class MyTemplate <const double*> : public TObject {
 public:
  double variable;
#ifdef R__GLOBALSTL
  vector<int> vars;
#else
  std::vector<int> vars;
#endif
  
  MyTemplate<const double*>(const double* a) { variable = *a; };
  MyTemplate<const double*>() {};
  
  ClassDefT(MyTemplate<const double*>,2)
};

template <class T1, class T2> class MyPairTemplate : public TObject {
 public:
  T1 var1;
  T2 var2;
  
  MyPairTemplate(T1 a, T2 b) : var1(a), var2(b) {};
  MyPairTemplate() {};
  ~MyPairTemplate() {};

  ClassDef(MyPairTemplate,1)
};

template <> 
class MyPairTemplate<int, double> : public TObject {
 public:
  float var1;
  float var2;
  
  MyPairTemplate<int,double>(int a, double b) : var1(a), var2(b) {};
  MyPairTemplate<int,double>() {};
#if (__GNUC__>=3 || __GNUC_MINOR__>=95)
  ~MyPairTemplate<int,double>() {};
#endif

  typedef MyPairTemplate<int, double> type;
  ClassDef(type,2)
};

void template_driver();

