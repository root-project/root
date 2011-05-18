// @(#)root/base:$Id$
// Author: Valery Fine(fine@vxcern.cern.ch)   05/03/97

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualGL                                                           //
//                                                                      //
// The TVirtualGL class is an abstract base class defining the          //
// OpenGL interface protocol. All interactions with OpenGL should be    //
// done via the global pointer gVirtualGL. If the OpenGL library is     //
// available this pointer is pointing to an instance of the TGLKernel   //
// class which provides the actual interface to OpenGL. Using this      //
// scheme of ABC we can use OpenGL in other parts of the framework      //
// without having to link with the OpenGL library in case we don't      //
// use the classes using OpenGL.                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TVirtualGL.h"
#include "TROOT.h"


ClassImp(TGLManager)

TGLManager * (*gPtr2GLManager)() = 0;

//____________________________________________________________________________
TGLManager::TGLManager() : TNamed("gGLManager", "")
{
}

//____________________________________________________________________________
TGLManager *&TGLManager::Instance()
{
   // Return the global GL Manager.

   static TGLManager *instance = 0;

   if(gPtr2GLManager) {
      instance = gPtr2GLManager();
   }

   return instance;
}

ClassImp(TVirtualGLPainter)


ClassImp(TVirtualGLManip)

ClassImp(TGLPaintDevice)
