// @(#)root/eve:$Id$
// Authors: Alja & Matevz Tadel 2008

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveText
#define ROOT_TEveText

#include "TNamed.h"
#include "TAtt3D.h"
#include "TAttBBox.h"

#include "TEveElement.h"

class TEveText : public TEveElement,
                 public TNamed,
                 public TAtt3D,
                 public TAttBBox
{
private:
   TEveText(const TEveText&);            // Not implemented
   TEveText& operator=(const TEveText&); // Not implemented

protected:
   TString   fText;       // text
   Color_t   fTextColor;  // text color

   Int_t     fFontSize;   // FTFont face size
   Int_t     fFontFile;   // FTFont file name
   Int_t     fFontMode;   // FTFont FTGL class id

   Float_t   fExtrude;    // extrude depth

   Bool_t    fAutoLighting; // use default lighting
   Bool_t    fLighting;     // force lighting

   Float_t   fPolygonOffset[2]; // depth test

public:
   TEveText(const char* txt="");
   virtual ~TEveText() {}

   Int_t   GetFontSize() const { return fFontSize; }
   Int_t   GetFontFile() const { return fFontFile; }
   Int_t   GetFontMode() const { return fFontMode; }
   void    SetFontSize(Int_t size, Bool_t validate = kTRUE);
   void    SetFontFile(Int_t file){ fFontFile = file; }
   void    SetFontFile(const char* name);
   void    SetFontMode(Int_t mode);

   const   char* GetText() const  { return fText.Data(); }
   void    SetText(const char* t) { fText = t; }

   Bool_t  GetLighting() const      { return fLighting; }
   void    SetLighting(Bool_t isOn) { fLighting = isOn; }

   Bool_t  GetAutoLighting() const      { return fAutoLighting; }
   void    SetAutoLighting(Bool_t isOn) { fAutoLighting = isOn; }

   Float_t GetExtrude() const    { return fExtrude; }
   void    SetExtrude(Float_t x) { fExtrude = x;    }

   Float_t  GetPolygonOffset(Int_t i) const { return fPolygonOffset[i]; }
   void     SetPolygonOffset(Float_t factor, Float_t units);

   virtual void   Paint(Option_t* option="");
   virtual void   ComputeBBox();

   virtual const TGPicture* GetListTreeIcon(Bool_t open=kFALSE);

   ClassDef(TEveText, 0); // Class for visualisation of text with FTGL font.
};

#endif
