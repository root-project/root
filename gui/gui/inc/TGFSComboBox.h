// @(#)root/gui:$Id$
// Author: Fons Rademakers   19/01/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGFSComboBox
#define ROOT_TGFSComboBox


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGFSComboBox, TGTreeLBEntry                                          //
//                                                                      //
// This is a combo box that is used in the File Selection dialog box.   //
// It will allow the file path selection.                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGComboBox
#include "TGComboBox.h"
#endif



class TGPicture;
class TGSelectedPicture;


class TGTreeLBEntry : public TGLBEntry {

protected:
   TGString           *fText;        // entry description
   TGString           *fPath;        // entry path
   const TGPicture    *fPic;         // entry picture
   TGSelectedPicture  *fSelPic;      // selected picture
   UInt_t              fTWidth;      // width of entry text
   UInt_t              fTHeight;     // height of entry text
   Bool_t              fActive;      // true if active
   GContext_t          fNormGC;      // entry drawing context
   FontStruct_t        fFontStruct;  // font

   virtual void DoRedraw();

   static const TGFont *fgDefaultFont;
   static TGGC         *fgDefaultGC;

public:
   static FontStruct_t  GetDefaultFontStruct();
   static const TGGC   &GetDefaultGC();

   TGTreeLBEntry(const TGWindow *p = 0, TGString *text = 0, const TGPicture *pic = 0,
                 Int_t id = -1, TGString *path = 0, GContext_t norm = GetDefaultGC()(),
                 FontStruct_t font = GetDefaultFontStruct(),
                 UInt_t options = kHorizontalFrame, Pixel_t back = GetWhitePixel());
   virtual ~TGTreeLBEntry();

   const TGString  *GetText() const { return fText; }
   const TGPicture *GetPicture() const { return fPic; }
   const TGString  *GetPath() const { return fPath; }

   virtual TGDimension GetDefaultSize() const;

   virtual void Activate(Bool_t a);
   virtual void Update(TGLBEntry *e);
   virtual void DrawCopy(Handle_t id, Int_t x, Int_t y);

   ClassDef(TGTreeLBEntry,0)  // TGFSComboBox entry
};


class TGFSComboBox : public TGComboBox {

public:
   TGFSComboBox(const TGWindow *p = 0, Int_t id = -1,
                UInt_t options = kHorizontalFrame | kSunkenFrame |
                kDoubleBorder, Pixel_t back = GetWhitePixel());

   virtual void Update(const char *path);
   virtual void SavePrimitive(std::ostream &out, Option_t *option = "");

   ClassDef(TGFSComboBox,0)  // Combo box widget for file system path
};

#endif
