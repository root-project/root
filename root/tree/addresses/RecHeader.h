#ifndef RECHEADER_H
#define RECHEADER_H

#include "Context.h"

class RecHeader : public TObject {

 public:

   RecHeader() {}                 // necessary for streamer io
   RecHeader(const Context& vld) : fContext(vld) {}
   virtual ~RecHeader() {}

   virtual void Print(Option_t* option = "") const;
     
 private:
   
   Context  fContext;  // Detector_t, SimFlag_t, VldTimeStamp

ClassDef(RecHeader,1)
};

#endif
