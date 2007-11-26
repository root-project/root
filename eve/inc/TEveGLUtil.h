// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef Reve_TEveGLUtil
#define Reve_TEveGLUtil

#ifndef __CINT__
#include <TGLIncludes.h>
#endif

#include <TObject.h>

class TAttMarker;
class TAttLine;

class TEveGLUtil
{
public:
   virtual ~TEveGLUtil() {}

#ifndef __CINT__

   class GL_Capability_Switch {
      GLenum    fWhat;
      GLboolean fState;
      bool      fFlip;

      void set_state(GLboolean s)
      { if(s) glEnable(fWhat); else glDisable(fWhat); }

   public:
      GL_Capability_Switch(GLenum what, GLboolean state) :
         fWhat(what), fState(kFALSE), fFlip(kFALSE)
      {
         fState = glIsEnabled(fWhat);
         fFlip  = (fState != state);
         if(fFlip)        set_state(state);
      }
      ~GL_Capability_Switch()
      { if(fFlip) set_state(fState); }
   };

   class GL_Float_Holder
   {
      GL_Float_Holder(const GL_Float_Holder&);            // Not implemented
      GL_Float_Holder& operator=(const GL_Float_Holder&); // Not implemented

      GLenum    fWhat;
      GLfloat   fState;
      bool      fFlip;
      void    (*fFoo)(GLfloat);

   public:
      GL_Float_Holder(GLenum what, GLfloat state, void (*foo)(GLfloat)) :
         fWhat(what), fState(kFALSE), fFlip(kFALSE), fFoo(foo)
      {
         glGetFloatv(fWhat, &fState);
         fFlip = (fState != state);
         if(fFlip) fFoo(state);
      }
      ~GL_Float_Holder()
      { if(fFlip) fFoo(fState); }
   };

#endif

   // Commonly used rendering primitives.

   static void RenderLine(const TAttLine& al, Float_t* p, Int_t n,
                          Bool_t selection=kFALSE, Bool_t sec_selection=kFALSE);

   static void RenderPolyMarkers(const TAttMarker& marker, Float_t* p, Int_t n,
                                 Bool_t selection=kFALSE, Bool_t sec_selection=kFALSE);

   static void RenderPoints(const TAttMarker& marker, Float_t* p, Int_t n,
                            Bool_t selection=kFALSE, Bool_t sec_selection=kFALSE);

   static void RenderCrosses(const TAttMarker& marker, Float_t* p, Int_t n, Bool_t sec_selection=kFALSE);

   ClassDef(TEveGLUtil, 0); // Commonly used utilities for GL rendering.
};

#endif
