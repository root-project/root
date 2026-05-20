{
  std::string name("TestMissingETBase::Types::jetlink_t");
  auto cl = TClass::GetClass(name.c_str());
  if (cl)
    return 0;
  else {
    cerr << "Error the first call to TClass::GetClass for '" << name << "' returned : "
         << cl << '\n';

    auto cci = gInterpreter->CheckClassInfo(name.c_str(), kTRUE, kTRUE);
    std::cerr << "cl was " << cl << " " << "CheckClassInfo returned    : " << cci << '\n';

    std::string normalizedName;
    {
      TInterpreter::SuspendAutoLoadingRAII autoloadOff(gInterpreter);
      TClassEdit::GetNormalizedName(normalizedName, name);
    }
    cerr << "Normalized named is  : " << normalizedName << '\n';
    cerr << "Second attempt: " << TClass::GetClass(name.c_str())
         << '\n';
    return 1;
  }
}
