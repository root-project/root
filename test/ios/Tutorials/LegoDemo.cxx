#include "TFrame.h"
#include "TF2.h"

#include "LegoDemo.h"
#include "IOSPad.h"

namespace ROOT {
namespace iOS {
namespace Demos {

//______________________________________________________________________________
LegoDemo::LegoDemo()
            : fLego(new TF2("fun1","1000*((sin(x)/x)*(sin(y)/y))+200", -6., 6., -6., 6.))
{
   //Ctor.
   fLego->SetFillColor(kOrange);
}

//______________________________________________________________________________
LegoDemo::~LegoDemo()
{
   //Just for std::auto_ptr's dtor.
}

//______________________________________________________________________________
void LegoDemo::AdjustPad(Pad *pad)
{
   pad->SetFillColor(0);
}

//______________________________________________________________________________
void LegoDemo::PresentDemo()
{
   //Draw fun of x and y as lego.
   fLego->Draw("lego");
}

}
}
}
