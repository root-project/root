#ifndef STLMAPPOLICY_H
#define STLMAPPOLICY_H

#include <string>
#include <map>

using namespace std;
template <class ElementType> class STLMapStoragePolicy 
{
  public:
  typedef string KeyType;
  typedef map<KeyType,ElementType> map_type;

  private:
  map_type map_;
};

#endif
// #include "STLMapStoragePolicy.h"
#include <TObject.h>

class Track : public STLMapStoragePolicy<int>
{
  public:
  Track() {;}
  virtual ~Track() {;}

  ClassDef(Track,1);
};
#ifdef __CINT__

#pragma link C++ class Track+;
#pragma link C++ class STLMapStoragePolicy<int>+;

#endif


