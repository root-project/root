#ifndef PUREABSTRACTINTERFACE_H
#define PUREABSTRACTINTERFACE_H
 
#include "TObject.h"
 
class PureAbstractInterface {
 
 public:
 
  virtual Short_t  GetXyzzy() const = 0;
  virtual Short_t  GetAbc()   const = 0;
 
 private:

  // because this is an interface provide no real ctor/dtors
  // PureAbstractInterface();
  // virtual ~PureAbstractInterface();

   ClassDef(PureAbstractInterface,0)
};

#endif
