// #include "mytypes.h"

#include "TObject.h"
#include "vector"

namespace MySpace {

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

  template <> class MyTemplate <const double*> : public TObject {
  public:
    double variable;
    double variable2;
#ifdef R__GLOBALSTL
    vector<int> vars;
#else
    std::vector<int> vars;
#endif
  
    MyTemplate(const double* a) { 
      variable = *a; variable2 = 2* *a; };
    MyTemplate() {};
  
    ClassDef(MyTemplate<const double*>,1)
  };

}

void nstemplate_driver();
