int hadd_check_cms(const char *merged, const char *reference)
{
   auto mergedNtpl = ROOT::RNTupleReader::Open("Events", merged);
   auto refNtpl = ROOT::RNTupleReader::Open("Events", reference);

   const auto &mergedDesc = mergedNtpl->GetDescriptor();
   const auto &refDesc = refNtpl->GetDescriptor();

   struct Defer {
      ~Defer()
      {
         gSystem->Unlink(merg);
         gSystem->Unlink(ref);
      }
      const char *merg, *ref;
  } defer { merged, reference };

  if (mergedDesc.GetNEntries() != refDesc.GetNEntries()) {
    printf("Merged RNTuple has a different number of fields from reference: %" PRIu64 " (merged) vs %" PRIu64 " (ref)\n",
           mergedDesc.GetNEntries(), refDesc.GetNEntries());
    return 1;
  }

  for (const auto &fdesc : refDesc.GetTopLevelFields()) {
    const auto fid = mergedDesc.FindFieldId(fdesc.GetFieldName());
    if (fid == ROOT::kInvalidDescriptorId) {
      printf("Merged RNTuple is missing field '%s'\n", fdesc.GetFieldName().c_str());
      return 1;
    }
  }
  for (const auto &fdesc : mergedDesc.GetTopLevelFields()) {
    const auto fid = refDesc.FindFieldId(fdesc.GetFieldName());
    if (fid == ROOT::kInvalidDescriptorId) {
      printf("Merged RNTuple has extra field '%s' that reference doesn't have\n", fdesc.GetFieldName().c_str());
      return 1;
    }
  }

  return 0;
}
