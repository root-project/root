#include <ROOT/RError.hxx>
#include "RXTuple.hxx"

#include <TBuffer.h>
#include <TError.h>
#include <TFile.h>

#include <xxhash.h>

void ROOT::Experimental::RXTuple::Streamer(TBuffer &buf)
{
   if (buf.IsReading()) {
      // Skip byte count and class version
      auto offCkData = sizeof(UInt_t) + sizeof(Version_t);
      auto expectedChecksum = XXH3_64bits(buf.GetCurrent() + offCkData, sizeof(RXTuple));

      std::uint64_t onDiskChecksum;
      buf.ReadClassBuffer(RXTuple::Class(), this);
      if (static_cast<std::size_t>(buf.BufferSize()) < buf.Length() + sizeof(onDiskChecksum))
         throw RException(R__FAIL("the buffer containing RXTuple is too small to contain the checksum!"));
      buf >> onDiskChecksum;

      if (expectedChecksum != onDiskChecksum)
         throw RException(R__FAIL("checksum mismatch in RXTuple"));
   } else {
      auto offCkData = buf.Length() + sizeof(UInt_t) + sizeof(Version_t);
      buf.WriteClassBuffer(RXTuple::Class(), this);
      std::uint64_t checksum = XXH3_64bits(buf.Buffer() + offCkData, sizeof(RXTuple));
      buf << checksum; 
   }
}
