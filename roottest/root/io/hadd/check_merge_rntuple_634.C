int check_merge_rntuple_634()
{
  // Verify that the given RNTuple is readable
  auto reader = ROOT::RNTupleReader::Open("Events", "test_hadd_merge_rntuple_634_merged.root");
  const auto &model = reader->GetModel();
  const auto &desc = reader->GetDescriptor();
  for (const auto &fdesc : desc.GetFieldIterable(desc.GetFieldZeroId()))  {
    if (fdesc.GetTypeName() != ROOT::Internal::GetRenormalizedTypeName(fdesc.GetTypeName())) {
      std::cerr << "Type name is not renormalized! " << fdesc.GetTypeName() << " vs " << 
        ROOT::Internal::GetRenormalizedTypeName(fdesc.GetTypeName()) << "\n";
      return 1;
    }
  }

  return 0;
}
