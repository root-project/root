// @(#)root/base:$Name:  $:$Id: TVirtualGL.cxx,v 1.2 2004/08/09 15:35:51 brun Exp $
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


TVirtualGL * (*gPtr2VirtualGL)() = 0;

//____________________________________________________________________________
TVirtualGL::TVirtualGL(TVirtualGLImp *imp) : TNamed("gVirtualGL", "")
{
   // ctor

   fImp = imp;
}

//____________________________________________________________________________
TVirtualGL::TVirtualGL(const char *name) : TNamed("gVirtualGL", name)
{
   // ctor.

   TString cmd = "new ";
   cmd += name;
   fImp = (TVirtualGLImp*)gROOT->ProcessLineFast(cmd.Data());
}

//____________________________________________________________________________
TVirtualGL *& TVirtualGL::Instance()
{
   //

   static TVirtualGL * instance = 0;

   if(gPtr2VirtualGL) {
      instance = gPtr2VirtualGL();
   }

   return instance;
}
