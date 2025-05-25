#include "StreamerFieldXML.h"

#include <TBuffer.h>

// A custom streamer that does what the automatic streamer does
void ForceNativeXML::Streamer(TBuffer &R__b)
{
   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      R__b.ReadClassBuffer(ForceNativeXML::Class(), this, R__v, R__s, R__c);
   } else {
      R__b.WriteClassBuffer(ForceNativeXML::Class(), this);
   }
}
