#include "IOSPad.h"
#include "TFrame.h"
#include "TF2.h"

#include "LegoDemo.h"

namespace ROOT {
namespace iOS {
namespace Demos {

////////////////////////////////////////////////////////////////////////////////
///Ctor.

LegoDemo::LegoDemo()
            : fLego(new TF2("fun1","1000*((sin(x)/x)*(sin(y)/y))+200", -6., 6., -6., 6.))
{
   fLego->SetFillColor(kOrange);
}

////////////////////////////////////////////////////////////////////////////////
///Just for std::auto_ptr's dtor.

LegoDemo::~LegoDemo()
{
}

////////////////////////////////////////////////////////////////////////////////

void LegoDemo::AdjustPad(Pad *pad)
{
   pad->SetFillColor(0);
}

////////////////////////////////////////////////////////////////////////////////
///Draw fun of x and y as lego.

void LegoDemo::PresentDemo()
{
   fLego->Draw("lego");
}

}
}
}
