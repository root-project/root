// @(#)root/eve:$Id$
// Author: Matevz Tadel, Aug 2009

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLFBO
#define ROOT_TGLFBO

#include "Rtypes.h"

class TGLFBO
{
private:
   TGLFBO(const TGLFBO&);            // Not implemented
   TGLFBO& operator=(const TGLFBO&); // Not implemented

protected:
   UInt_t  fFrameBuffer;
   UInt_t  fColorTexture;
   UInt_t  fDepthBuffer;
   // UInt_t  fStencilBuffer;

   Int_t   fW, fH;

   Bool_t  fIsRescaled;
   Float_t fWScale, fHScale;

   static Bool_t fgRescaleToPow2;

public:
   TGLFBO();
   virtual ~TGLFBO();

   void Init(int w, int h);
   void Release();

   void Bind();
   void Unbind();

   void BindTexture();
   void UnbindTexture();

   ClassDef(TGLFBO, 0); // Frame-buffer object.
};

#endif
