#ifndef SLAVE2_HH_
#define SLAVE2_HH_

#include <TObject.h>


namespace Slave2
{  

  class Object : public TObject {
  public:
    Object() {}
    
  protected:
    ClassDef(Slave2::Object,1);
  };
}


#endif
