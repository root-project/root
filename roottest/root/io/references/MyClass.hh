#ifndef CLASS_HH_
#define CLASS_HH_

#include <TObject.h>


namespace Name
{  
  class MyClass : public TObject{
  public:
    MyClass(const TObject  & s) :
    source(s)
    { }
  protected:
    const TObject  & source;
    ClassDef(MyClass,1);
  };
}


#endif
