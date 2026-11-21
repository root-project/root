// For the licensing terms see $ROOTSYS/LICENSE.
// For the list of contributors see $ROOTSYS/README/CREDITS.

#include "TVirtualImageUtils.h"
#include "TPluginManager.h"

////////////////////////////////////////////////////////////////////////////////
/// Create an image utility plugin. The caller owns the returned object.
/// Returns nullptr if no implementation can be loaded.

TVirtualImageUtils *TVirtualImageUtils::Create()
{
   auto handler = gPluginMgr->FindHandler("TVirtualImageUtils");
   if (!handler || handler->LoadPlugin() == -1)
      return nullptr;

   return reinterpret_cast<TVirtualImageUtils *>(handler->ExecPlugin(0));
}
