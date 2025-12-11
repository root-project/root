{
   int errors = 0;
   int expectedByteCounts = 1;

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
