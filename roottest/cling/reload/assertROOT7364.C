int assertROOT7364() {
  gROOT->ProcessLine("#include <map>\n\n");
  gROOT->ProcessLine("std::map<int,std::string> p;");
  gROOT->ProcessLine("#include <string>");
  return 0;
}
