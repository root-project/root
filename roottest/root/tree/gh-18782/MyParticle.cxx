#include "MyParticle.hxx"
#include <TBuffer.h>

void UnsplittableBase::Streamer(TBuffer &R__b)
{
   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(UnsplittableBase::Class(), this);
   } else {
      R__b.WriteClassBuffer(UnsplittableBase::Class(), this);
   }
}
