// #include "mytypes.h"

#include "TObject.h"
#include "vector"

namespace MySpace {

  template <class T> class MyTemplate : public TObject {
  public:
    T variable; //!
    vector<int> vars;
  
    MyTemplate(T a) { variable = a; };
    MyTemplate() {};
  
    ClassDefT(MyTemplate,1)
  };

  class MyTemplate <const double*> : public TObject {
  public:
    const double* variable; //!
    vector<int> vars;
  
    MyTemplate<const double*>(const double* a) { variable = a; };
    MyTemplate<const double*>() {};
  
    ClassDefT(MyTemplate<const double*>,1)
  };

}

