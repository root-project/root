#include "TObject.h"
#include "vector"

template <class T> class MyTemplate : public TObject {
 public:
  T variable; //!
  std::vector<int> vars;
  
  MyTemplate(T a) { variable = a; };
  MyTemplate() {};

  ClassDefT(MyTemplate,1)
};

// ClassDefT2(MyTemplate,T)

template <>
class MyTemplate <const double*> : public TObject {
 public:
  const double* variable; //!
  std::vector<int> vars;
  
  MyTemplate<const double*>(const double* a) { variable = a; };
  MyTemplate<const double*>() {};
  
  ClassDefT(MyTemplate<const double*>,2)
};



template <class T1, class T2> class MyPairTemplate : public TObject {
 public:
  T1 var1;
  T2 var2;
  
  MyPairTemplate(T1 a, T2 b) : var1(a), var2(b) {};
  ~MyPairTemplate() {};

  ClassDefT(MyPairTemplate,1)
};

// ClassDef2T2(MyPairTemplate,T1,T2)


template <> 
class MyPairTemplate<int, double> : public TObject {
 public:
  float var1;
  float var2;
  
  MyPairTemplate<int,double>(int a, double b) : var1(a), var2(b) {};
#if (__GNUC__>=3 || __GNUC_MINOR__>=95)
  ~MyPairTemplate<int,double>() {};
#endif

  typedef MyPairTemplate<int, double> type;

  ClassDefT(type,2)
};



