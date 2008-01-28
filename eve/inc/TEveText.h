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

#include "TAtt3D.h"
#include "TAttBBox.h"
#include "TNamed.h"

#include "TEveElement.h"
#include "TEveTrans.h"

class TEveText : public TEveElement,
                 public TNamed,
                 public TAtt3D,
                 public TAttBBox
{
private:
   TEveText(const TEveText&);            // Not implemented
   TEveText& operator=(const TEveText&); // Not implemented

protected:
   TString           fText;         // text
   Color_t           fTextColor;    // text color

   Int_t             fSize;         // face size
   Int_t             fFile;         // font file name
   Int_t             fMode;         // FTGL class
   Float_t           fExtrude;      // extrude depth

   Bool_t            fAutoBehave;   // use defaults
   Bool_t            fLighting;     // enable GL lighting

   TEveTrans         fHMTrans;      // overall transformation

public:
   TEveText(const Text_t* txt="");
   virtual ~TEveText() {}

   virtual Bool_t CanEditMainColor() { return kTRUE; }
   virtual void   Paint(Option_t* option="");
   virtual void   ComputeBBox();

   Int_t   GetSize() const { return fSize; }
   Int_t   GetFile() const { return fFile; }
   Int_t   GetMode() const { return fMode; }
   void    SetFontSize(Int_t size, Bool_t validate = kTRUE);
   void    SetFontFile(Int_t file){ fFile = file; }
   void    SetFontFile(const char* name);
   void    SetFontMode(Int_t mode);

   const   Text_t* GetText() const  { return fText.Data(); }
   void    SetText(const Text_t* t) { fText = t; }

   Bool_t  GetLighting() const      { return fLighting; }
   void    SetLighting(Bool_t isOn) { fLighting = isOn; }

   Bool_t  GetAutoBehave() const      { return fAutoBehave; }
   void    SetAutoBehave(Bool_t isOn) { fAutoBehave = isOn; }

   Float_t GetExtrude() const    { return fExtrude; }
   void    SetExtrude(Float_t x) { fExtrude = x;    }

   virtual Bool_t     CanEditMainHMTrans() { return kTRUE; }
   virtual TEveTrans* PtrMainHMTrans()     { return &fHMTrans; }

   virtual const TGPicture* GetListTreeIcon();

   ClassDef(TEveText, 0); // Class for visualisation of text with FTGL font.
};

#endif
