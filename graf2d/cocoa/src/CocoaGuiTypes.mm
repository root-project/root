#include "CocoaGuiTypes.h"

namespace ROOT {
namespace MacOSX {
namespace X11 {

//______________________________________________________________________________
Point::Point()
         : fX(0), fY(0)
{
}

//______________________________________________________________________________
Point::Point(int x, int y)
         : fX(x), fY(y)
{
}

//______________________________________________________________________________
Rectangle::Rectangle()
             : fX(0), fY(0), fWidth(0), fHeight(0)
{
}

//______________________________________________________________________________
Rectangle::Rectangle(int x, int y, unsigned w, unsigned h)
             : fX(x), fY(y), fWidth(w), fHeight(h)
{
}

}//X11
}//MacOSX
}//ROOT
