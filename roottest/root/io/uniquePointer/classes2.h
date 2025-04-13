#include "TGraph.h"
#include "TH1F.h"

#include <memory>
#include <vector>
#include <list>
#include <forward_list>
#include <set>
#include <map>

using tgup = std::unique_ptr<TGraph>;
using thup = std::unique_ptr<TH1F>;
class B {
   std::vector<tgup> v;
   std::list<tgup> l;
   std::forward_list<tgup> fl;
   std::set<tgup> s;
   std::map<int,tgup> m1;
   std::map<tgup, int> m2;
   std::map<tgup, tgup> m3;
   std::map<thup, tgup> m4;
};