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
     std::vector<std::pair<Name::Sk,float> > vssb1; //!
     std::vector<float > vssb2; //
     std::vector<std::vector<float>::iterator > vssb3; 
  public:
    ClassDef(Name::SkyMap,2)
    
  };
}


#endif

