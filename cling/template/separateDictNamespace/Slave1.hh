#ifndef SLAVE1_HH_
#define SLAVE1_HH_

#include <TObject.h>


namespace Slave1
{  

  class Object : public TObject {
  public:
    Object() {}
    
  protected:
    ClassDef(Slave1::Object,1);
  };
}


#endif
