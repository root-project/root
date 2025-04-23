// #include "mytypes.h"

#include "TObject.h"
#include "vector"

namespace MySpace {

  template <class T> class MyTemplate : public TObject {
  public:
    T variable;
    std::vector<int> vars;
  
    MyTemplate(T a) { variable = a; };
    MyTemplate() {};
  
    ClassDefT(MyTemplate,1)
  };

  template <> class MyTemplate <const double*> : public TObject {
  public:
    double variable;
    double variable2;
    std::vector<int> vars;
  
    MyTemplate<const double*>(const double* a) { 
      variable = *a; variable2 = 2* *a; };
    MyTemplate<const double*>() {};
  
#ifdef R__WIN32
    typedef MyTemplate<const double*> type;
    ClassDef(type,1)
#else
    ClassDefT(MyTemplate<const double*>,1)
#endif
  };

}

