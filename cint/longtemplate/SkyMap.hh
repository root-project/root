#ifndef SKYMAP_HH_
#define SKYMAP_HH_

#include <TObject.h>
#include <vector>

namespace Name
{

  class Sk
  {
  public:  
    ClassDef(Name::Sk,1)
  };
}

namespace Name
{  
  class SkyMap
  {
  private:
     //std::vector<std::pair<Name::Sk,float> > vssb; //!
     std::vector<float > vssb; //
  public:
    ClassDef(Name::SkyMap,2)
    
  };
}


#endif

