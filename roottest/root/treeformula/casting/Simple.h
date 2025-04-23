#ifndef SIMPLE_H
#define SIMPLE_H
//
//
// Simple class
//
#include "TShape.h"

class Simple : public TObject {

private:
   Int_t   fID;         // id number
   TShape* fShape;      // pointer to base class shape

public:

   Simple() : fID(0), fShape(0) { }
   Simple(Int_t id, TShape* shape): fID(id), fShape(shape) { }
   ~Simple() override;
   void Print(Option_t *option = "") const override;

   ClassDefOverride(Simple,1)  //Simple class
};

#endif                         // SIMPLE_H
