#ifndef CLASS_HH_
#define CLASS_HH_

#include <TObject.h>

class TH1F;

namespace Name
{  
  class MyClass : public TObject{
  public:
    MyClass() 
    {}

    template<class T>
    T & func(T * t  )
    { return *t; }
    void Print(Option_t *);
  protected:
    ClassDef(MyClass,1);
  };
}


#endif
