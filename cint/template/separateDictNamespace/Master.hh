#ifndef CLASS_HH_
#define CLASS_HH_

#include <TObject.h>

namespace Master
{  
  class Container : public TObject{
  public:
    Container() 
    {}

    template<class T>
    T & func(T * t  )
    { return *t; }
    void Print(Option_t *) const;
  protected:
    ClassDef(Master::Container,1);
  };

  class Object : public TObject {
  public:
    Object() {}
    
  protected:
    ClassDef(Master::Object,1);
  };
}


#endif
