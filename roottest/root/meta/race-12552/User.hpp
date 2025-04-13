#ifndef USER_CLASS_HPP
#define USER_CLASS_HPP

#include <vector>
#include <TString.h>

struct UserClass {
   // The dictionary for this class is generated using genreflex
   // so the following is not enough to trigger a GenerateInitInstance
   std::vector<TString> fValues;
};

#endif
