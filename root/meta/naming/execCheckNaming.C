/*
A test to verify the correct naming adopted by ROOT.
It is intended to collect all the issues encountered with naming
during the development, testing and production phase of ROOT6.

"Stat rosa pristina nomine, nomina nuda tenemus"
*/

#include <algorithm>

namespace std {
   class Something {};
   typedef Something Something_t;
}

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

  // The role of ResolveTypedef is to remove typedef and should be given
  // an almost normalized name.  The main purpose of this test is to
  // insure that nothing is removed and no stray space is added. 
  // However, (at least for now), it is allowed to remove some spaces
  // but does not have to remove them (this is the job of ShortType
  // or the name normalization routine). 
  const std::vector<const char*> tceTypedefNames={"const Something_t&",
                                           "const std::Something&",
                                           "const string&",
                                           "A<B>[2]",
                                           "X<A<B>[2]>"};


  const std::vector<const char*> tceNames={"const std::Something&",
                                           "const std::Something  &",
                                           "const std::string&",
                                           "const std::string &",
                                           "const std::string    &",
                                           "A<B>[2]",
                                           "X<A<B>[2]>"};  

  std::cout << "Check TClassEdit::ResolveTypedef\n";
  for (auto& name : tceTypedefNames)
     checkTypedef(name);
     
  std::cout << "Check TClassEdit::ShortType\n";
  for (auto& name : tceNames)  
     checkShortType(name);


  // GetNormalizedName
  // Here tests for Norm Name


  return 0;
}
