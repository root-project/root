#ifndef CLASS_H
#define CLASS_H

#include "Riostream.h"

class MyClass {
 protected:
  bool GetProtected() ;
 public:
  bool GetPublic() {return GetProtected();};
};

#endif
