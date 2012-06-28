/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#if defined(interp) && defined(makecint)
#pragma include "test.dll"
#else
#include "t1193.h"
#endif

namespace testscript0001 {
  class MyScript : public SystemScript
  {
  private :
    float mValue;
  public :
    MyScript() : SystemScript("MyScript", "MyObject")
    {
      System::Print("alive!");
      
      // this causes "Warning: Automatic variable mValue is allocated
      mValue = 1.0f;

      // working version
      // this->mValue = 1.0f;
    }
  } MyObject;
  //MyScript MyObject2;
}

// -- script start --
namespace testscript0002
{
  class Noise : public EditorScript
  {
  public :
    Noise() : EditorScript("Noise", "NoiseObject")
    {
      const int Size = 3;
      unsigned short *Map = new unsigned short[Size];
      for(int i=0; i<Size; i++)
        Map[i]=Math::rand()%1024;
                        
      BeginTerrainPaint();
      //[..(snipped)..]
    }
  };

  Noise NoiseObject;
};

int main() {
  return 0;
}
