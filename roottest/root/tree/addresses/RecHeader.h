#ifndef RECHEADER_H
#define RECHEADER_H

#include "Context.h"

class RecHeader : public TObject {

 public:

   RecHeader() {}                 // necessary for streamer io
   RecHeader(const Context& vld) : fContext(vld) {}
   ~RecHeader() override {}

   void Print(Option_t* option = "") const override;
     
 private:
   
   Context  fContext;  // Detector_t, SimFlag_t, VldTimeStamp

ClassDefOverride(RecHeader,1)
};

#endif
