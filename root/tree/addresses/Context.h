#ifndef CONTEXT_H
#define CONTEXT_H

#include "TObject.h"

class Context : public TObject {

 public:

   Context():fDetector(-1),fSimFlag(-1),fTimeStamp(-1) {}
   Context(Int_t detector,Int_t mcflag,Int_t time) : fDetector(detector),
     fSimFlag(mcflag),fTimeStamp(time) {}
   virtual ~Context() {}
   void                     Print(Option_t *option = "") const;

 private:

   Int_t       fDetector;
   Int_t       fSimFlag;
   Int_t       fTimeStamp;


   ClassDef(Context,1)  

};

#endif
