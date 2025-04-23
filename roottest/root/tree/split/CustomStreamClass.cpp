#include "CustomStreamClass.h"
#include "TBuffer.h"

ClassImp(MyClass)

#ifdef CUSTOM_STREAMER
void MyClass::Streamer(TBuffer &R__b)
{
   // Stream an object of class MyClass.

   Info("Streamer","In the custom streamer");
   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(MyClass::Class(),this);
   } else {
      R__b.WriteClassBuffer(MyClass::Class(),this);
   }
}
#endif
