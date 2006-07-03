// @(#)root/gui:$Name:  $:$Id: TGFont.cxx,v 1.6 2006/05/28 20:07:59 brun Exp $
// Author: Fons Rademakers   20/5/2003

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
/**************************************************************************

    This source is based on Xclass95, a Win95-looking GUI toolkit.
    Copyright (C) 1996, 1997 David Barth, Ricky Ralston, Hector Peraza.

    Xclass95 is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

**************************************************************************/


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGFont and TGFontPool                                                //
//                                                                      //
// Encapsulate fonts used in the GUI system.                            //
// TGFontPool provides a pool of fonts.                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGFont.h"
#include "TGClient.h"
#include "THashTable.h"
#include "TVirtualX.h"

#include "Riostream.h"
#include "TROOT.h"

ClassImp(TGFont)

//______________________________________________________________________________
TGFont::~TGFont()
{
   // Delete font.

   if (fFontStruct)
      gVirtualX->DeleteFont(fFontStruct);
}

//______________________________________________________________________________
void TGFont::GetFontMetrics(FontMetrics_t *m) const
{
   // Get font metrics.

   if (!m) {
      Error("GetFontMetrics", "argument may not be 0");
      return;
   }

   *m = fFM;
}

//______________________________________________________________________________
FontStruct_t TGFont::operator()() const
{
   // Not inline due to a bug in g++ 2.96 20000731 (Red Hat Linux 7.0)

   return fFontStruct;
}

//______________________________________________________________________________
void TGFont::Print(Option_t *) const
{
   // Print font info.

   Printf("TGFont: %s, %s, ref cnt = %u", fName.Data(),
          fFM.fFixed ? "fixed" : "prop", References());
}


ClassImp(TGFontPool)

//______________________________________________________________________________
TGFontPool::TGFontPool(TGClient *client)
{
   // Create a font pool.

   fClient = client;
   fList   = new THashTable(50);
   fList->SetOwner();
}

//______________________________________________________________________________
TGFontPool::~TGFontPool()
{
   // Cleanup font pool.

   delete fList;
}

//______________________________________________________________________________
TGFont *TGFontPool::GetFont(const char *font, Bool_t fixedDefault)
{
   // Get the specified font. Returns 0 if error or no font can be found.
   // If fixedDefault is false the "fixed" font will not be substituted
   // as fallback when the asked for font does not exist.

   if (!font || !*font) {
      Error("GetFont", "argument may not be 0 or empty");
      return 0;
   }

   TGFont *f = (TGFont*) fList->FindObject(font);

   if (f) {
      f->AddReference();
      return f;
   }

   FontStruct_t fs = fClient->GetFontByName(font, fixedDefault);

   if (fs) {
      f = new TGFont(font);
      f->fFontStruct = fs;
      f->fFontH      = gVirtualX->GetFontHandle(fs);
      gVirtualX->GetFontProperties(fs, f->fFM.fAscent, f->fFM.fDescent);
      f->fFM.fLinespace = f->fFM.fAscent + f->fFM.fDescent;
      f->fFM.fMaxWidth = gVirtualX->TextWidth(fs, "w", 1);
      f->fFM.fFixed = (f->fFM.fMaxWidth == gVirtualX->TextWidth(fs, "i", 1)) ? kTRUE : kFALSE;
      fList->Add(f);
   }

   return f;
}

//______________________________________________________________________________
TGFont *TGFontPool::GetFont(const TGFont *font)
{
   // Use font, i.e. increases ref count of specified font. Returns 0
   // if font is not found.

   TGFont *f = (TGFont*) fList->FindObject(font);

   if (f) {
      f->AddReference();
      return f;
   }

   return 0;
}

//______________________________________________________________________________
TGFont *TGFontPool::GetFont(FontStruct_t fs)
{
   // Use font, i.e. increases ref count of specified font. 

   TGFont *f = FindFont(fs);

   if (f) {
      f->AddReference();
      return f;
   }

   static int i = 0;

   f = new TGFont(Form("unknown-%d", i));
   f->fFontStruct = fs;
   f->fFontH      = gVirtualX->GetFontHandle(fs);
   gVirtualX->GetFontProperties(fs, f->fFM.fAscent, f->fFM.fDescent);
   f->fFM.fLinespace = f->fFM.fAscent + f->fFM.fDescent;
   f->fFM.fMaxWidth = gVirtualX->TextWidth(fs, "w", 1);
   f->fFM.fFixed = (f->fFM.fMaxWidth == gVirtualX->TextWidth(fs, "i", 1)) ? kTRUE : kFALSE;
   fList->Add(f);
   i++;

   return f;
}

//______________________________________________________________________________
void TGFontPool::FreeFont(const TGFont *font)
{
   // Free font. If ref count is 0 delete font.

   TGFont *f = (TGFont*) fList->FindObject(font);
   if (f) {
      if (f->RemoveReference() == 0) {
         fList->Remove(f);
         delete font;
      }
   }
}

//______________________________________________________________________________
TGFont *TGFontPool::FindFont(FontStruct_t font) const
{
   // Find font based on its font struct. Returns 0 if font is not found.

   TIter next(fList);

   while (TGFont *f = (TGFont*) next())
      if (f->fFontStruct == font)
         return f;

   return 0;
}

//______________________________________________________________________________
TGFont *TGFontPool::FindFontByHandle(FontH_t font) const
{
   // Find font based on its font handle. Returns 0 if font is not found.

   TIter next(fList);

   while (TGFont *f = (TGFont*) next())
      if (f->fFontH == font)
         return f;

   return 0;
}

//______________________________________________________________________________
void TGFontPool::Print(Option_t *) const
{
   // List all fonts in the pool.

   fList->Print();
}

//______________________________________________________________________________
void TGFont::SavePrimitive(ostream &out, Option_t * /*= ""*/)
{
    // Save the used font as a C++ statement(s) on output stream out

   char quote = '"';

   if (gROOT->ClassSaved(TGFont::Class())) {
      out << endl;
   } else {
      //  declare a font object to reflect required user changes
      out << endl;
      out << "   TGFont *ufont;         // will reflect user font changes" << endl;
   }
   out << "   ufont = gClient->GetFont(" << quote << GetName() << quote << ");" << endl;
}
