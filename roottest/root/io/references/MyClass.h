#ifndef SKYMAP_HH_
#define SKYMAP_HH_

#include <Rtypes.h>
#include <vector>
#include <map>
#include <TObject.h>


namespace Name
{  

  template <class T>
  class TT : public TObject
  {
  public:
    std::map<float, T> mymap;
    ClassDef(Name::TT<T>,1)      
  };


  class SkyMap
  {
  public:
    std::vector<Name::TT<float> > vssb; //!
  public:
    ClassDef(Name::SkyMap,2)
    
  };
}


#endif
