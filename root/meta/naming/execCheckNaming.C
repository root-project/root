/*
A test to verify the correct naming adopted by ROOT.
It is intended to collect all the issues encountered with naming
during the development, testing and production phase of ROOT6.

"Stat rosa pristina nomine, nomina nuda tenemus"
*/

#include <algorithm>

void checkTypedef(const std::string& name)
{
   std::cout << "@" << name << "@ --> @" << TClassEdit::ResolveTypedef(name.c_str()) << "@" <<  std::endl;
}

void checkShortType(const std::string& name)
{
   std::cout << "@" << name << "@ --> @" << TClassEdit::ShortType(name.c_str(), 1186) << "@" <<  std::endl;
}

int execCheckNaming(){
  using namespace std::placeholders;

  // TClassEdit
  const std::vector<const char*> tceNames={"const std::string&",
                                           "const std::string &",
                                           "const std::string    &",
                                           "A<B>[2]",
                                           "X<A<B>[2]>"};  

  std::cout << "Check TClassEdit::ResolveTypedef\n";
  for (auto& name : tceNames)
     checkTypedef(name);
     
  std::cout << "Check TClassEdit::ShortType\n";
  for (auto& name : tceNames)  
     checkShortType(name);


  // GetNormalizedName
  // Here tests for Norm Name


  return 0;
}
