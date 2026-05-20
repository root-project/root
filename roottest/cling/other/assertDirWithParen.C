bool check(const char* arg, int res)
{
  int interpError = 0;
  std::string exec = ".x dir-with(paren)//this_is_the_test.C";
  if (gROOT->ProcessLine((exec + arg).c_str(), &interpError) != res
      || interpError != 0) {
    std::cerr << arg << '\n';
    return false;
  }
  return true;
}

int assertDirWithParen()
{
  bool haveError = false;
  haveError |= !check("", 17);
  // Unsupported: haveError |= !check(";", 17);
  haveError |= !check("()", 17);
  haveError |= !check("();", 17);

  haveError |= !check("(42)", 42);
  haveError |= !check("(43);", 43);
  haveError |= !check("(17 + 42)", 59);
  haveError |= !check("((12) + (43))", 55);
      
  haveError |= !check("(42, \")\")", 42);
  haveError |= !check("(17 + 42, \"()\")", 59);
  haveError |= !check("((12) + (43), \"(\")", 55);

  if (haveError)
    exit(1);
  return 0;
}
