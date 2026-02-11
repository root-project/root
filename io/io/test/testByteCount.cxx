#include "TBufferFile.h"
#include "TExMap.h"

#include <iostream>
#include <vector>


int unittestByteCount()
{
   int errors = 0;
   unsigned int expectedByteCounts = 1;

   // TBufferFile currently reject size larger than 2GB.
   // SetBufferOffset does not check against the size,
   // so we can provide and use a larger buffer.
   std::vector<char> databuffer{};
   databuffer.reserve(4 * 1024 * 1024 * 1024ll);
   TBufferFile b(TBuffer::kWrite, 2 * 1024 * 1024 * 1024ll - 100, databuffer.data(), false /* don't adopt */);
   {
      // Regular object at offset 0
      UInt_t R__c = b.WriteVersion(TExMap::Class(), kTRUE);
      b.SetBufferOffset(1000);
      b.SetByteCount(R__c, kTRUE);
   }
   {
      // Regular object
      UInt_t R__c = b.WriteVersion(TExMap::Class(), kTRUE);
      b.SetBufferOffset(2000);
      b.SetByteCount(R__c, kTRUE);
   }
   {
      // Object larger than 1GB
      UInt_t R__c = b.WriteVersion(TExMap::Class(), kTRUE);
      b.SetBufferOffset(4000 + 1 * 1024 * 1024 * 1024ll);
      b.SetByteCount(R__c, kTRUE);
   }
   {
      // Regular object located past 1GB
      UInt_t R__c = b.WriteVersion(TExMap::Class(), kTRUE);
      b.SetBufferOffset(8000 + 1 * 1024 * 1024 * 1024ll);
      b.SetByteCount(R__c, kTRUE);
   }
   {
      ++expectedByteCounts;
      // Object larger than 1GB start after 1GB
      // NOTE: this does not yet fit, we are writing past the end.
      // Need to lift the 2GB limit for TBuffer first.
      // However the lifting might be temporary, so this might need to be
      // moved to a test that stored objects in a TFile.
      UInt_t R__c = b.WriteVersion(TExMap::Class(), kTRUE);
      b.SetBufferOffset(12000 + 2 * 1024 * 1024 * 1024ll);
      b.SetByteCount(R__c, kTRUE);
   }

   // To make a copy instead of using the const references:
   auto bytecounts{b.GetByteCounts()};
   if (bytecounts.size() != expectedByteCounts) {
      ++errors;
      std::cerr << "The number of bytecount is not as expected (1), it is " << bytecounts.size() << '\n';
      std::cerr << "The full list is:\n";
      for (auto bc : bytecounts)
         std::cerr << "values: " << bc.first << " , " << bc.second << '\n';
   }

   // Rewind.  Other code use Reset instead of SetBufferOffset
   b.SetReadMode();
   b.Reset();
   b.SetByteCounts(std::move(bytecounts));

   UInt_t R__s = 0;
   UInt_t R__c = 0;
   {
      // Regular object at offset 0
      auto version = b.ReadVersion(&R__s, &R__c, TExMap::Class());
      b.SetBufferOffset(1000);
      auto res = b.CheckByteCount(R__s, R__c, TExMap::Class());
      if (res != 0) {
         ++errors;
         // We can assume there as already an error message in CheckByCount
      }
   }
   {
      // Regular object
      auto version = b.ReadVersion(&R__s, &R__c, TExMap::Class());
      b.SetBufferOffset(2000);
      auto res = b.CheckByteCount(R__s, R__c, TExMap::Class());
      if (res != 0) {
         ++errors;
         // We can assume there as already an error message in CheckByCount
      }
   }
   {
      // Object larger than 1GB
      auto version = b.ReadVersion(&R__s, &R__c, TExMap::Class());
      b.SetBufferOffset(4000 + 1 * 1024 * 1024 * 1024ll);
      auto res = b.CheckByteCount(R__s, R__c, TExMap::Class());
      if (res != 0) {
         ++errors;
         // We can assume there as already an error message in CheckByCount
      }
   }
   {
      // Regular object located past 1GB
      auto version = b.ReadVersion(&R__s, &R__c, TExMap::Class());
      b.SetBufferOffset(8000 + 1 * 1024 * 1024 * 1024ll);
      auto res = b.CheckByteCount(R__s, R__c, TExMap::Class());
      if (res != 0) {
         ++errors;
         // We can assume there as already an error message in CheckByCount
      }
   }
   {
      // Object larger than 1GB start after 1GB
      // NOTE: this does not yet fit.
      auto version = b.ReadVersion(&R__s, &R__c, TExMap::Class());
      b.SetBufferOffset(12000 + 2 * 1024 * 1024 * 1024ll);
      auto res = b.CheckByteCount(R__s, R__c, TExMap::Class());
      if (res != 0) {
         ++errors;
         // We can assume there as already an error message in CheckByCount
      }
   }

   std::cerr << "The end.\n";
   return errors;
}

struct LargeByteCountsFixture {
   std::size_t fSize{0};
   void resize(size_t size) {
      fSize = size;
   }
   void Streamer(TBuffer &b) {
      // Bare minimum to trigger the large byte count mechanism,
      // we don't care about the content of the data.
      if (b.IsReading()) {
         b >> fSize;
         b.SetBufferOffset(b.GetCurrent() - b.Buffer() + fSize);
      } else {
         b << fSize;
         b.SetBufferOffset(b.GetCurrent() - b.Buffer() + fSize);
      }
   }
};

#ifdef __ROOTCLING__
#pragma link C++ class LargeByteCountsFixture-;
#endif

int readAndCheck(const std::string &msg, TBufferFile &b, size_t expected_size)
{
   // Same as b >> ptr;
   auto obj = b.ReadObjectAny(TClass::GetClass(typeid(LargeByteCountsFixture)));
   if (!obj) {
      std::cerr << msg << ": Failed to read back the object\n";
      return 1;
   } else {
      auto* readFixture = static_cast<LargeByteCountsFixture*>(obj);
      if (readFixture->fSize != expected_size ) {
         std::cerr << msg << ": The size of the data vectors do not match the original ones\n";
         delete readFixture;
         return 1;
      }
      delete readFixture;
   }
   return 0;
}


char *DoNothingAllocator(char* input, size_t, size_t)
{
   // We 'could' check that the requested memory in under what we
   // preallocated.
   return input;
}

int testReadWriteObjectAny()
{
   int errors = 0;

   // TBufferFile currently reject size larger than 2GB.
   // SetBufferOffset does not check against the size,
   // so we can provide and use a larger buffer.
   std::vector<char> databuffer{};
   databuffer.reserve(8 * 1024 * 1024 * 1024ll);
   TBufferFile b(TBuffer::kWrite, 8 * 1024 * 1024 * 1024ll - 100, databuffer.data(), false /* don't adopt */, DoNothingAllocator);

   LargeByteCountsFixture fixture;
   fixture.resize(100); // Small object, should be written with the regular byte count mechanism.

   auto startPos = b.GetCurrent() - b.Buffer();
   b.WriteObject(&fixture, false /* cacheReuse */);
   b.SetReadMode();
   b.SetBufferOffset(startPos);

   errors += readAndCheck("Small object written in regular section", b, 100);

   // Large object, should be written with the large byte count mechanism
   b.SetWriteMode();
   startPos = b.GetCurrent() - b.Buffer();
   fixture.resize(1024 * 1024 * 256); // 1GB of data
   b.WriteObject(&fixture, false /* cacheReuse */);
   b.SetReadMode();
   b.SetBufferOffset(startPos);
   errors += readAndCheck("Large object written in regular section", b, 1024 * 1024 * 256);

   // Large object written in long range section, should be written with
   // the large byte count mechanism
   b.SetWriteMode();
   b.SetBufferOffset(4 * 1024 * 1024 * 1024ll + 100);
   startPos = b.GetCurrent() - b.Buffer();
   fixture.resize(1024 * 1024 * 256); // 1GB of data
   b.WriteObject(&fixture, false /* cacheReuse */);
   b.SetReadMode();
   b.SetBufferOffset(startPos);
   errors += readAndCheck("Large object written in long range section", b, 1024 * 1024 * 256);

   return errors;
}


int testByteCount()
{
   int res = unittestByteCount();
   res += testReadWriteObjectAny();
   return res;
}