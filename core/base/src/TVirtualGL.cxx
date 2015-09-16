// @(#)root/base:$Id$
// Author: Valery Fine   05/03/97

/** \class TVirtualGL

The TVirtualGL class is an abstract base class defining the
OpenGL interface protocol. All interactions with OpenGL should be
done via the global pointer gVirtualGL. If the OpenGL library is
available this pointer is pointing to an instance of the TGLKernel
class which provides the actual interface to OpenGL. Using this
scheme of ABC we can use OpenGL in other parts of the framework
without having to link with the OpenGL library in case we don't
use the classes using OpenGL.
*/

#include "TVirtualGL.h"
#include "TROOT.h"
#include "TGlobal.h"


ClassImp(TGLManager)

TGLManager * (*gPtr2GLManager)() = 0;

namespace {
static struct AddPseudoGlobals {
AddPseudoGlobals() {
  // User "gCling" as synonym for "libCore static initialization has happened".
   // This code here must not trigger it.
   TGlobalMappedFunction::Add(new TGlobalMappedFunction("gGLManager", "TVirtualGL*",
                                 (TGlobalMappedFunction::GlobalFunc_t)&gGLManager));
}
} gAddPseudoGlobals;
}

////////////////////////////////////////////////////////////////////////////////

TGLManager::TGLManager() : TNamed("gGLManager", "")
{
}

////////////////////////////////////////////////////////////////////////////////
/// Return the global GL Manager.

TGLManager *&TGLManager::Instance()
{
   static TGLManager *instance = 0;

   if(gPtr2GLManager) {
      instance = gPtr2GLManager();
   }

   return instance;
}

ClassImp(TVirtualGLPainter)


ClassImp(TVirtualGLManip)

ClassImp(TGLPaintDevice)
