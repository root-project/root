#include <ROOT/RError.hxx>
#include "RXTuple.hxx"

#include <TBuffer.h>
#include <TError.h>
#include <TFile.h>

#include <xxhash.h>

void ROOT::RXTuple::Streamer(TBuffer &buf)
{
   if (buf.IsReading()) {
      assert(!"This class should never be read!");
   } else {
      auto offCkData = buf.Length() + sizeof(UInt_t) + sizeof(Version_t);
      buf.WriteClassBuffer(RXTuple::Class(), this);
      std::uint64_t checksum = XXH3_64bits(buf.Buffer() + offCkData, buf.Length() - offCkData);
      buf << checksum;
   }
}
