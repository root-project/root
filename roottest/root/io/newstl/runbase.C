#include <map>
#include <string>
#include <utility>
#include "TFile.h"
#include "TBranch.h"
#include "TTree.h"
#include "exception"
#include "algorithm"
#include "set"

#ifndef ClingWorkAroundStripDefaultArg
namespace std {
#endif
  template <typename key, typename value, 
    typename compare_operation = less<key>, 
    typename alloc = allocator<pair<const key, value> > >
  class cmap: public map<key, value, compare_operation, alloc> {
  public:
    typedef map<key,  value, compare_operation, alloc> _map;
    typedef cmap<key, value, compare_operation, alloc> _cmap;
    std::string map_name;
  public:
    typedef typename _map::const_iterator const_iterator;
    typedef typename _map::iterator iterator;
    // Some default ctors
    cmap (const std::string& name="unknown"): _map(), map_name(name) {}
    cmap (const _map& x, const std::string& name = "unknown"): _map(x), map_name(name) {}
    cmap (const _cmap& x): _map(x), map_name(x.map_name) {}
    value& operator[] (const key& k) { return (dynamic_cast<_map &>(*this))[k]; }
    const value& operator[] (const key& k) const {
      const_iterator item = this->find (k);
      if (item == _map::end ()) {
         throw ("Element not fount!");
      }
      return item->second;
    }
  };
#ifndef ClingWorkAroundStripDefaultArg
}
#endif

class foo: public TObject { 
public: 
#ifndef ClingWorkAroundStripDefaultArg
  std::cmap<int,int> & get_table () { return m_map; } 
#else
  cmap<int,int> & get_table () { return m_map; }
#endif
  foo (): TObject (), m_map("foo") {} 
  virtual ~foo() {}
private: 
#ifndef ClingWorkAroundStripDefaultArg
  std::cmap<int,int> m_map; 
#else
  cmap<int,int> m_map;
#endif
   std::set<std::string> m_string;
  ClassDef(foo,1);
}; 

ClassImp(foo);

#ifdef __CINT__
// pragma link C++ class std::map<int, int>;
#endif


void runbase()  {
  foo *e = new foo;
  TFile *file = new TFile("test.root","RECREATE");
  TTree *tree = new TTree("T","T");
  tree->Branch("test","foo",&e);
  e->get_table().insert(std::make_pair(1,1));
  e->get_table().insert(std::make_pair(2,2));
  tree->Fill();
  e->get_table().insert(std::make_pair(3,3));
  e->get_table().insert(std::make_pair(4,4));
  tree->Fill();
  file->Write();
  file->Close();
  delete file;
}
