#include "TObject.h"
#include "TNamed.h"
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
  
  MyTemplate(const double* a) { variable = *a; };
  MyTemplate() {};
  
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
  
  MyPairTemplate(int a, double b) : var1(a), var2(b) {};
  MyPairTemplate() {};
  ~MyPairTemplate() {};

  typedef MyPairTemplate<int, double> type;
  ClassDef(type,2)
};     


// le tableau abstrait de base
class RtbVArray : public TNamed
{ 
  //       ...     
 private:     
  ClassDef(RtbVArray,1);
} ;


// la variante template pour eviter les casts a l'utilisateur
template <class T>
class RtbVTArray : public RtbVArray
{ 
  //...
 private:      
  ClassDef(RtbVTArray,1);
} ;

// une implementation concrete
template <class T>
class RtbCArray : public RtbVTArray<T>
{ 
  // ...
private:
  typedef T value_type;
  value_type a;
  ClassDef(RtbCArray,1);
} ;

// une classe etrangere
class RtbLorentzVector
{ 
  //...     
};


void template_driver();

