//Author: Timur Pocheptsov.
#include "CocoaUtils.h"

namespace ROOT {
namespace MacOSX {
namespace Util {

//______________________________________________________________________________
AutoreleasePool::AutoreleasePool()
                  : fPool([[NSAutoreleasePool alloc] init])
{
}

//______________________________________________________________________________
AutoreleasePool::~AutoreleasePool()
{
   [fPool release];
}

}
}
}
