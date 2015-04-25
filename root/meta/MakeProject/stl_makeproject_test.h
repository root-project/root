
#ifndef _stl_makeproject_test_h
#define _stl_makeproject_test_h

#include <bitset>
#include <vector>

#include "TObject.h"

class SillyStlEvent : public TObject {

private:
   std::bitset<16> foo = 0xfa2;
   std::vector<int> bar = {1,2};

public:
   SillyStlEvent() {}
   //{bar.push_back(1); bar.push_back(2);}

   virtual ~SillyStlEvent() {}
   unsigned long get_bitset() const {return foo.to_ulong();}
   bool correct_bar() const {return (bar.size() == 2) && (bar[0] == 1) && (bar[1] == 2);}

   ClassDef(SillyStlEvent,1)  //Event structure
};

#endif

