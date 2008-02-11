#ifndef ROOT_TBENCH
#define ROOT_TBENCH
   
#include "TClonesArray.h"
namespace stdext {}
#include <vector>
#include <deque>
#include <list>
#include <set>
#include <map>

#ifndef R__GLOBALSTL
#ifndef WIN32
using std::vector;
using std::list;
using std::deque;
using std::set;
using std::multiset;
using std::map;
using std::multimap;
#else
using namespace std;
using namespace stdext;
#endif
#endif
#ifdef R__HPUX
namespace std {
  using ::make_pair;
  using ::pair;
}
#endif

#ifdef __CINT__
template<class a,class b,class c> class hash_map : public map<a,b,c> {};
template<class a,class b> class hash_set : public set<a,b> {};
template<class a,class b,class c> class hash_multimap : public multimap<a,b,c> {};
template<class a,class b> class hash_multiset : public multiset<a,b> {};
#else
//#include <hash_map>
//#include <hash_set>
#endif

//-------------------------------------------------------------
class THit {
protected:
  float     fX;         //x position at center
  float     fY;         //y position at center
  float     fZ;         //z position at center
  int       fNpulses;   //Number of pulses
  int      *fPulses;    //[fNpulses]
  int       fTime[10];  //time at the 10 layers
public:

  THit();
  THit(const THit &);
  THit(int time);
  virtual ~THit();

  void  Set (int time);
  inline int Get(int i) { return fTime[i]; }
  bool operator==(const THit& c) const { return this==&c;}
  bool operator<(const THit& c) const { return this<&c;}
  THit& operator=(const THit& c);
  friend TBuffer &operator<<(TBuffer &b, const THit *hit);

  ClassDef(THit,1) // the hit class
};

namespace stdext {
  template<class T>  inline size_t __gnu_cxx_hash_obj(const T& __o) {
    unsigned long __h = 0;
    const char* s = (const char*)&__o;
    for (size_t i=0; i<sizeof(T); ++s, ++i)
      __h = 5*__h + *s;
    return size_t(__h);
  }

  template <class _Key> struct hash { };
  inline size_t hash_value(const THit& s)  {
    return __gnu_cxx_hash_obj(s);
  }
}
#if defined R__TEMPLATE_OVERLOAD_BUG
template <> 
#endif
inline TBuffer &operator>>(TBuffer &buf,THit *&obj)
{
   obj = new THit();
   obj->Streamer(buf);
   return buf;
}

//-------------------------------------------------------------
class TObjHit : public TObject, public THit {

public:

  TObjHit();
  TObjHit(int time);
  virtual ~TObjHit(){;}

  ClassDef(TObjHit,1) // the hit class
};

//-------------------------------------------------------------
class TSTLhit {
protected:
  Int_t            fNhits;
  vector <THit>    fList1;

public:

  TSTLhit();
  TSTLhit(int nmax);
  void Clear(Option_t *option="");
  virtual ~TSTLhit();
  void MakeEvent(int ievent);
  Int_t MakeTree(int mode, int nevents, int compression, int split, float &cx);
  Int_t ReadTree();

  ClassDef(TSTLhit,1) // STL vector of THit
};
//-------------------------------------------------------------
class TSTLhitList {
protected:
  Int_t            fNhits;
  list   <THit>    fList1;

public:

  TSTLhitList();
  TSTLhitList(int nmax);
  void Clear(Option_t *option="");
  virtual ~TSTLhitList();
  void MakeEvent(int ievent);
  Int_t MakeTree(int mode, int nevents, int compression, int split, float &cx);
  Int_t ReadTree();

  ClassDef(TSTLhitList,1) // STL vector of THit
};

//-------------------------------------------------------------
class TSTLhitDeque {
protected:
  Int_t            fNhits;
  deque  <THit>    fList1;

public:

  TSTLhitDeque();
  TSTLhitDeque(int nmax);
  void Clear(Option_t *option="");
  virtual ~TSTLhitDeque();
  void MakeEvent(int ievent);
  Int_t MakeTree(int mode, int nevents, int compression, int split, float &cx);
  Int_t ReadTree();

  ClassDef(TSTLhitDeque,1) // STL vector of THit
};
      
//-------------------------------------------------------------
class TSTLhitMultiset {
protected:
  Int_t            fNhits;
  multiset  <THit>    fList1;

public:

  TSTLhitMultiset();
  TSTLhitMultiset(int nmax);
  void Clear(Option_t *option="");
  virtual ~TSTLhitMultiset();
  void MakeEvent(int ievent);
  Int_t MakeTree(int mode, int nevents, int compression, int split, float &cx);
  Int_t ReadTree();

  ClassDef(TSTLhitMultiset,1) // STL vector of THit
};

//-------------------------------------------------------------
class TSTLhitSet {
protected:
  Int_t            fNhits;
  set  <THit>    fList1;

public:

  TSTLhitSet();
  TSTLhitSet(int nmax);
  void Clear(Option_t *option="");
  virtual ~TSTLhitSet();
  void MakeEvent(int ievent);
  Int_t MakeTree(int mode, int nevents, int compression, int split, float &cx);
  Int_t ReadTree();

  ClassDef(TSTLhitSet,1) // STL vector of THit
};
//-------------------------------------------------------------
class TSTLhitMap {
protected:
  Int_t            fNhits;
  map  <int,THit>    fList1;

public:

  TSTLhitMap();
  TSTLhitMap(int nmax);
  void Clear(Option_t *option="");
  virtual ~TSTLhitMap();
  void MakeEvent(int ievent);
  Int_t MakeTree(int mode, int nevents, int compression, int split, float &cx);
  Int_t ReadTree();

  ClassDef(TSTLhitMap,1) // STL vector of THit
};
//-------------------------------------------------------------
class TSTLhitMultiMap {
protected:
  Int_t                fNhits;
  multimap  <int,THit> fList1;

public:

  TSTLhitMultiMap();
  TSTLhitMultiMap(int nmax);
  void Clear(Option_t *option="");
  virtual ~TSTLhitMultiMap();
  void MakeEvent(int ievent);
  Int_t MakeTree(int mode, int nevents, int compression, int split, float &cx);
  Int_t ReadTree();

  ClassDef(TSTLhitMultiMap,1) // STL vector of THit
};
#if 0
//-------------------------------------------------------------
class TSTLhitHashSet {
protected:
  Int_t            fNhits;
  hash_set  <THit> fList1;

public:

  TSTLhitHashSet();
  TSTLhitHashSet(int nmax);
  void Clear(Option_t *option="");
  virtual ~TSTLhitHashSet();
  void MakeEvent(int ievent);
  Int_t MakeTree(int mode, int nevents, int compression, int split, float &cx);
  Int_t ReadTree();

  ClassDef(TSTLhitHashSet,1) // STL vector of THit
};
//-------------------------------------------------------------
class TSTLhitHashMultiSet {
protected:
  Int_t            fNhits;
  hash_multiset  <THit>    fList1;

public:

  TSTLhitHashMultiSet();
  TSTLhitHashMultiSet(int nmax);
  void Clear(Option_t *option="");
  virtual ~TSTLhitHashMultiSet();
  void MakeEvent(int ievent);
  Int_t MakeTree(int mode, int nevents, int compression, int split, float &cx);
  Int_t ReadTree();

  ClassDef(TSTLhitHashMultiSet,1) // STL vector of THit
};
#endif
//-------------------------------------------------------------
class TSTLhitStar {
protected:
  Int_t            fNhits;
  vector <THit*>   fList2; //

public:

  TSTLhitStar();
  TSTLhitStar(int nmax);
  void Clear(Option_t *option="");
  virtual ~TSTLhitStar();
  void MakeEvent(int ievent);
  Int_t MakeTree(int mode, int nevents, int compression, int split, float &cx);
  Int_t ReadTree();

  ClassDef(TSTLhitStar,1) // STL vector of pointers to THit
};
//-------------------------------------------------------------
class TSTLhitStarList {
protected:
  Int_t            fNhits;
  list <THit*>   fList2; //

public:

  TSTLhitStarList();
  TSTLhitStarList(int nmax);
  void Clear(Option_t *option="");
  virtual ~TSTLhitStarList();
  void MakeEvent(int ievent);
  Int_t MakeTree(int mode, int nevents, int compression, int split, float &cx);
  Int_t ReadTree();

  ClassDef(TSTLhitStarList,1) // STL vector of pointers to THit
};
//-------------------------------------------------------------
class TSTLhitStarDeque {
protected:
  Int_t            fNhits;
  deque <THit*>   fList2; //

public:

  TSTLhitStarDeque();
  TSTLhitStarDeque(int nmax);
  void Clear(Option_t *option="");
  virtual ~TSTLhitStarDeque();
  void MakeEvent(int ievent);
  Int_t MakeTree(int mode, int nevents, int compression, int split, float &cx);
  Int_t ReadTree();

  ClassDef(TSTLhitStarDeque,1) // STL vector of pointers to THit
};

//-------------------------------------------------------------
class TSTLhitStarSet {
protected:
  Int_t            fNhits;
  set <THit*>   fList2; //

public:

  TSTLhitStarSet();
  TSTLhitStarSet(int nmax);
  void Clear(Option_t *option="");
  virtual ~TSTLhitStarSet();
  void MakeEvent(int ievent);
  Int_t MakeTree(int mode, int nevents, int compression, int split, float &cx);
  Int_t ReadTree();

  ClassDef(TSTLhitStarSet,1) // STL vector of pointers to THit
};
      
//-------------------------------------------------------------
class TSTLhitStarMultiSet {
protected:
  Int_t            fNhits;
  multiset <THit*>   fList2; //

public:

  TSTLhitStarMultiSet();
  TSTLhitStarMultiSet(int nmax);
  void Clear(Option_t *option="");
  virtual ~TSTLhitStarMultiSet();
  void MakeEvent(int ievent);
  Int_t MakeTree(int mode, int nevents, int compression, int split, float &cx);
  Int_t ReadTree();

  ClassDef(TSTLhitStarMultiSet,1) // STL vector of pointers to THit
};
      
//-------------------------------------------------------------
class TSTLhitStarMap {
protected:
  Int_t            fNhits;
  map <int,THit*>   fList2; //

public:

  TSTLhitStarMap();
  TSTLhitStarMap(int nmax);
  void Clear(Option_t *option="");
  virtual ~TSTLhitStarMap();
  void MakeEvent(int ievent);
  Int_t MakeTree(int mode, int nevents, int compression, int split, float &cx);
  Int_t ReadTree();

  ClassDef(TSTLhitStarMap,1) // STL vector of pointers to THit
};
      
//-------------------------------------------------------------
class TSTLhitStarMultiMap {
protected:
  Int_t            fNhits;
  multimap<int,THit*>   fList2; //

public:

  TSTLhitStarMultiMap();
  TSTLhitStarMultiMap(int nmax);
  void Clear(Option_t *option="");
  virtual ~TSTLhitStarMultiMap();
  void MakeEvent(int ievent);
  Int_t MakeTree(int mode, int nevents, int compression, int split, float &cx);
  Int_t ReadTree();

  ClassDef(TSTLhitStarMultiMap,1) // STL vector of pointers to THit
};
      
//-------------------------------------------------------------
class TCloneshit {
protected:
  Int_t            fNhits;
  TClonesArray    *fList3; //->

public:

  TCloneshit();
  TCloneshit(int nmax);
  void Clear(Option_t *option="");
  virtual ~TCloneshit();
  void MakeEvent(int ievent);
  Int_t MakeTree(int mode, int nevents, int compression, int split, float &cx);
  Int_t ReadTree();

  ClassDef(TCloneshit,1) // TClonesArray of TObjHit
};

#endif
