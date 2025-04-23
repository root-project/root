int execParamPack() {
  // Check that the generated dictionary works:
  if (!TClass::GetClass("edm::Table<42, float,int,std::string>")) {
    std::cerr << "Cannot find TClass for edm::Table<42, float,int,std::string>!\n";
    exit(1);
  }


  // Check that we can handle the rootmap entry:
  TInterpreter::EErrorCode err;
  gInterpreter->ProcessLine("namespace edm2 { template <class T> class Table2; }", &err);
  if (err == TInterpreter::kNoError) {
    // The real Table2 should have been available through another.rootmap
    std::cerr << "Should not be able to re-declare Table as different template!\n";
    exit(1);
  }

  return 0;
}
