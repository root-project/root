// @(#)root/base:$Name:  $:$Id: TVirtualGL.cxx,v 1.1.1.1 2000/05/16 17:00:39 rdm Exp $
// Author: Valery Fine(fine@vxcern.cern.ch)   05/03/97

//______________________________________________________________________________
//*-*-*-*-*-*-*-*-*-*-*-* TVirtualGL class *-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                     ================
//*-*
//*-*   TGLKernel class defines the interface for OpenGL commands and utilities
//*-*   Those are defined with GL/gl and GL/glu include directories
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

#include "TVirtualGL.h"
#include "TROOT.h"


TVirtualGL * (*gPtr2VirtualGL)() = 0;

//____________________________________________________________________________
TVirtualGL::TVirtualGL(TVirtualGLimp *imp) : TNamed("gVirtualGL", "")
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
   fImp = (TVirtualGLimp*)gROOT->ProcessLineFast(cmd.Data());
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
