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
    void Print(Option_t *) const;
  protected:
    ClassDef(MyClass,1);
  };
}


// Test veto of fwd decl of funny template args:
namespace TEST_N {
   enum E { kVal_TEST = 12 };

   template <typename U, int I = 1, int J = (I & kVal_TEST) ? kVal_TEST : I>
   class X {};
}

#endif
