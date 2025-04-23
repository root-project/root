//A test for ROOT-6704
#include <algorithm>
void execstlPatternSelection(){

   gSystem->Load("libstlPatternSelection_dictrflx");

   // Stat rosa pristina nomine
   // nomina nuda tenemus
   const std::vector<std::string> names {"std::map<string, bool>",
                                        "std::map<char, bool>",
                                        "std::map<const char*, bool>",
                                        "std::map<int, bool>",
                                        "std::map<int, const double*>",
                                        "std::map<int, unsigned long int>",
                                        "std::map<double, bool>",
                                        "std::map<double, const double*>",
                                        "std::map<double, unsigned long int>",
                                        "std::multimap<string, bool>",
                                        "std::multimap<char, bool>",
                                        "std::multimap<const char*, bool>",
                                        "std::multimap<int, bool>",
                                        "std::multimap<int, const double*>",
                                        "std::multimap<int, unsigned long int>",
                                        "std::multimap<double, bool>",
                                        "std::multimap<double, const double*>",
                                        "std::multimap<double, unsigned long int>",
                                        "std::multimap<float, float>"};

   std::for_each(names.begin(),
                 names.end(),
                 [](const std::string& name)
                  {
                     if (!TClass::GetDict(name.c_str())) std::cerr << name
                                                                   << " was not selected !Note that std::multimap<float, float> is expected not to be found.\n";
                  });

}
