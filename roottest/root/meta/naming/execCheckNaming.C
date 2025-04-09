/*
A test to verify the correct naming adopted by ROOT.
It is intended to collect all the issues encountered with naming
during the development, testing and production phase of ROOT6.

"Stat rosa pristina nomine, nomina nuda tenemus"
*/

#include <algorithm>
#include <vector>
#include <list>
#include <set>
#include <map>
#include <unordered_map>
#include <unordered_set>

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

template <typename coll>
void checkTypeidShortType(const char *collname)
{
   int err__ = 0;
   std::string demTI = TClassEdit::DemangleTypeIdName(typeid(coll), err__);
   TClassEdit::TSplitType splitname(demTI.c_str(), (TClassEdit::EModType)(TClassEdit::kLong64 | TClassEdit::kDropStd) );
   splitname.ShortType(demTI, TClassEdit::kDropStlDefault | TClassEdit::kDropStd);
   std::cout << "@typeid(" << collname << ")@ --> @" << demTI << "@\n";
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
                                           "X<A<B>[2]>",
                                           "__shared_ptr<TObject>",
     "map<int,int>",
     "map<int,int>, less<int>>",
     "map<Int_t,int>, less<int>>",
     "map<Int_t,Int_t>, less<int>>",
     "map<int,int,less<int>,allocator<pair<int const,int> > >",
     "map<Int_t,Int_t,less<Int_t>,allocator<pair<Int_t const,Int_t> > >",
     "map<const TObject*,int,less<TObject const*>,allocator<pair<TObject* const,int> > >",
     "map<TObject const*,int,less<TObject const*>,allocator<pair<TObject const* const,int> > >",
     "map<TObject*const,int,less<TObject*const>,allocator<pair<const TObject* const,int> > >",
     "map<TObject*,int,less<TObject*>,allocator<pair<TObject* const,int> > >",
     "map<TObject*,int,less<TObject*>,allocator<pair<TObject*const,int> > >",
     "list<int,allocator<int> >"
  };

  std::cout << "Check TClassEdit::ResolveTypedef\n";
  for (auto& name : tceTypedefNames)
     checkTypedef(name);
     
  std::cout << "Check TClassEdit::ShortType\n";
  for (auto& name : tceNames)  
     checkShortType(name);

  checkTypeidShortType<set<int>>("set<int>");
  checkTypeidShortType<list<int>>("list<int>");
  checkTypeidShortType<deque<int>>("deque<int>");
  checkTypeidShortType<vector<int>>("vector<int>");
  checkTypeidShortType<map<int,float>>("map<int,float>");
  checkTypeidShortType<multimap<int,float>>("multimap<int,float>");
  checkTypeidShortType<unordered_set<int>>("unordered_set<int>");
  checkTypeidShortType<unordered_map<int,float>>("unordered_map<int,float>");
  checkTypeidShortType<unordered_multimap<int,float>>("unordered_multimap<int,float>");

  // GetNormalizedName
  // Here tests for Norm Name


  return 0;
}

#if 0

std::string demTI
demTI = "std::map<const TObject *, int, std::less<TObject *const>, std::allocator<std::pair<TObject*const, int> > >"
s = new TClassEdit::TSplitType (demTI.c_str(), (TClassEdit::EModType)(TClassEdit::kLong64 | TClassEdit::kDropStd) );
s->ShortType(demTI, TClassEdit::kDropStlDefault | TClassEdit::kDropStd);
demTI


#endif
