#ifndef RooFit_VariableGroups_h
#define RooFit_VariableGroups_h

#include <TNamed.h>

#include <iostream>
#include <unordered_map>
#include <vector>

class TNamed;

namespace RooFit {

struct VariableGroups {

   std::unordered_map<TNamed const*, std::vector<int>> groups;

   inline void print() {
      for (auto const& item : groups) {
          std::cout << item.first->GetName() << " :";
          for (int n : item.second) {
             std::cout << " " << n;
          }
          std::cout << std::endl;
      }
   }

   int currentIndex = 0;
};

}

#endif
