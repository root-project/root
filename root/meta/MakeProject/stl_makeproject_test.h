
#ifndef _stl_makeproject_test_h
#define _stl_makeproject_test_h

#include <bitset>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "TH1D.h"

#include "TObject.h"

class SillyStlEvent : public TObject {

private:
   std::bitset<16> foo = 0xfa2;
   std::vector<int> bar = {1,2};
   std::unordered_set<double> spam = {1,3,5,6,8};
   std::unordered_map<int,std::string> eggs = {{2,"two"},{4,"four"}};
   std::unordered_multimap<std::string,TH1D> strange = {{"one",TH1D("","one",11,0,10)},
                                                   {"one",TH1D("","oneBis",100,0,10)},
                                                   {"two",TH1D("","two",123,0,10)}};


public:
   SillyStlEvent() {}
   //{bar.push_back(1); bar.push_back(2);}

   virtual ~SillyStlEvent() {}
   unsigned long get_bitset() const {return foo.to_ulong();}
   bool correct_bar() const {return (bar.size() == 2) && (bar[0] == 1) && (bar[1] == 2);}

   ClassDef(SillyStlEvent,1)  //Event structure
};

#endif

