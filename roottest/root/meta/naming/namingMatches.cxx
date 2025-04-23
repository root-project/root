#ifndef namingMatches_cxx
#define namingMatches_cxx

#include <vector>
#include <list>
#include <atomic>

class Object {};

template <typename T, typename O = Object> class Wrapper {};

template <typename T> class Container
{
public:
   vector<Wrapper<T> > fValues;
};

class HolderAuto {
public:
   list<Wrapper<int> > fValues;
};

class Holder {
public:
   typedef vector<Wrapper<int> > value_t;
   value_t fContainer;
   vector<Wrapper<int> > fValues;
   //vector<Wrapper<int, Object> > fValues2;
   std::atomic<vector<Wrapper<int> >*> fAtom; //!
};

namespace Geant {
   inline namespace cxx {
      class GeantTrack {
         ClassDef(GeantTrack,1);
      };
   }
}

#endif

#ifdef __ROOTCLING__
#pragma link C++ class Wrapper<int>+;
#pragma link C++ class Object+;
#pragma link C++ class vector<Wrapper<int> >+;
#pragma link C++ class atomic<vector<Wrapper<int> >*>;
#pragma link C++ class Container<int>+;
#pragma link C++ class Geant::cxx::GeantTrack+;
#endif

