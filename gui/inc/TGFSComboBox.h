// @(#)root/gui:$Name:  $:$Id: TGFSComboBox.h,v 1.2 2000/09/29 08:57:05 rdm Exp $
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

friend class TGClient;

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

   static ULong_t        fgSelPixel;
   static FontStruct_t   fgDefaultFontStruct;
#ifdef R__SUNCCBUG
public:
#endif
   static TGGC           fgDefaultGC;

public:
   TGTreeLBEntry(const TGWindow *p, TGString *text, const TGPicture *pic,
                 Int_t id, TGString *path = 0, GContext_t norm = fgDefaultGC(),
                 FontStruct_t font = fgDefaultFontStruct,
                 UInt_t options = kHorizontalFrame, ULong_t back = fgWhitePixel);
   virtual ~TGTreeLBEntry();

   const TGString  *GetText() const { return fText; }
   const TGPicture *GetPicture() const { return fPic; }
   const TGString  *GetPath() const { return fPath; }

   virtual TGDimension GetDefaultSize() const;

   virtual void Activate(Bool_t a);
   virtual void Update(TGLBEntry *e);

   static FontStruct_t  GetDefaultFontStruct();
   static const TGGC   &GetDefaultGC();

   ClassDef(TGTreeLBEntry,0)  // TGFSComboBox entry
};


class TGFSComboBox : public TGComboBox {

public:
   TGFSComboBox(const TGWindow *p, Int_t id,
                UInt_t options = kHorizontalFrame | kSunkenFrame |
                kDoubleBorder, ULong_t back = fgWhitePixel);

   virtual void Update(const char *path);

   ClassDef(TGFSComboBox,0)  // Combo box widget for file system path
};

#endif
