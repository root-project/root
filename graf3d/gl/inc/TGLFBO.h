// @(#)root/gl:$Id$
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
   UInt_t  fMSFrameBuffer;
   UInt_t  fMSColorBuffer;

   Int_t   fW, fH, fReqW, fReqH, fMSSamples, fMSCoverageSamples;

   Float_t fWScale, fHScale;
   Bool_t  fIsRescaled;

   static Bool_t fgRescaleToPow2;
   static Bool_t fgMultiSampleNAWarned;

   void InitStandard();
   void InitMultiSample();

   UInt_t CreateAndAttachRenderBuffer(Int_t format, Int_t type);
   UInt_t CreateAndAttachColorTexture();

public:
   TGLFBO();
   virtual ~TGLFBO();

   void Init(int w, int h, int ms_samples=0);
   void Release();

   void Bind();
   void Unbind();

   void BindTexture();
   void UnbindTexture();

   void SetAsReadBuffer();

   Int_t   GetW()    const { return fW; }
   Int_t   GetH()    const { return fH; }
   Int_t   GetReqW() const { return fReqW; }
   Int_t   GetReqH() const { return fReqH; }
   Int_t   GetMSSamples()         const { return fMSSamples; }
   Int_t   GetMSCoverageSamples() const { return fMSCoverageSamples; }

   Float_t GetWScale() const { return fWScale; }
   Float_t GetHScale() const { return fHScale; }

   Bool_t  GetIsRescaled() const { return fIsRescaled; }

   static  Bool_t GetRescaleToPow2();
   static  void   SetRescaleToPow2(Bool_t r);

   ClassDef(TGLFBO, 0); // Frame-buffer object.
};

#endif
