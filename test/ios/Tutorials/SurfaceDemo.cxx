#include "TFrame.h"
#include "IOSPad.h"
#include "TF2.h"

#include "SurfaceDemo.h"

namespace ROOT {
namespace iOS {
namespace Demos {

//______________________________________________________________________________
SurfaceDemo::SurfaceDemo()
               : fSurface(new TF2("fun1","1000*((sin(x)/x)*(sin(y)/y))+200", -6., 6., -6., 6.))
{
}

//______________________________________________________________________________
SurfaceDemo::~SurfaceDemo()
{
   //For auto_ptr dtor only.
}

//______________________________________________________________________________
void SurfaceDemo::AdjustPad(Pad *pad)
{
   pad->SetFillColor(38);
}

//______________________________________________________________________________
void SurfaceDemo::PresentDemo()
{
   fSurface->Draw("surf1");
}


}
}
}
