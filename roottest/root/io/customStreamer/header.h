#include "TBuffer.h"
#include "TClass.h"
class Hard2Stream {
private:
#ifndef __CINT__
   double val;
#endif
public:
   Hard2Stream() : val(-1) {};
   Hard2Stream(double v) : val(v) {};

   double getVal() { return val; }
   void setVal(double v) { val = v; }

   void print();
};

// Various streamers
void hard2StreamStreamer(TBuffer &b, void *objadd) {
   TClass *R__cl = TClass::GetClass("Hard2Stream");

   Hard2Stream *obj = (Hard2Stream*)objadd;
   UInt_t R__s, R__c;

   if (b.IsReading()) {
      /* Version_t v =  */ b.ReadVersion(&R__s, &R__c);

      double val;
      b >> val;
      obj->setVal(val);

      b.CheckByteCount(R__s, R__c,R__cl);

   } else {
      R__c = b.WriteVersion(R__cl, kTRUE);

      b << obj->getVal();

      b.SetByteCount(R__c, kTRUE);

   }

}

#include "TStreamer.h"
#include "TClass.h"
void setStreamer() {
   TClass *cl = TClass::GetClass("Hard2Stream");
   cl->AdoptStreamer(new TClassStreamer(hard2StreamStreamer));
}
