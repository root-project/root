// @(#)root/gui:$Name:  $:$Id: TGFont.h,v 1.6 2006/05/15 11:01:14 rdm Exp $
// Author: Fons Rademakers   20/5/2003

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGFont
#define ROOT_TGFont


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGFont and TGFontPool                                                //
//                                                                      //
// Encapsulate fonts used in the GUI system.                            //
// TGFontPool provides a pool of fonts.                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif
#ifndef ROOT_TGObject
#include "TGObject.h"
#endif
#ifndef ROOT_TRefCnt
#include "TRefCnt.h"
#endif

class THashTable;


struct FontMetrics_t {
   Int_t   fAscent;          // from baseline to top of font
   Int_t   fDescent;         // from baseline to bottom of font
   Int_t   fLinespace;       // the sum of the ascent and descent
   Int_t   fMaxWidth;        // width of widest character in font
   Bool_t  fFixed;           // true if monospace, false otherwise
};



class TGFont : public TNamed, public TRefCnt {

friend class TGFontPool;

private:
   FontStruct_t    fFontStruct;      // low level graphics fontstruct
   FontH_t         fFontH;           // font handle (derived from fontstruct)
   FontMetrics_t   fFM;              // cached font metrics

protected:
   TGFont(const char *name)
      : TNamed(name,""), TRefCnt(), fFontStruct(0), fFontH(0), fFM()
   {
      SetRefCount(1);
   }

   TGFont(const TGFont &font);          // not implemented
   void operator=(const TGFont &rhs);   // use TGFontPool to get fonts

public:
   virtual ~TGFont();

   FontH_t      GetFontHandle() const { return fFontH; }
   FontStruct_t GetFontStruct() const { return fFontStruct; }
   FontStruct_t operator()() const;

   void         GetFontMetrics(FontMetrics_t *m) const;

   void         Print(Option_t *option="") const;

   virtual void SavePrimitive(ofstream &out, Option_t *);

   ClassDef(TGFont,0)   // GUI font description
};


class TGFontPool : public TGObject {

private:
   THashTable    *fList;

protected:

   TGFontPool(const TGFontPool& fp) 
     : TGObject(fp), fList(fp.fList) { }
   TGFontPool& operator=(const TGFontPool& fp)
     {if(this!=&fp) {TGObject::operator=(fp); fList=fp.fList;}
     return *this;}

public:
   TGFontPool(TGClient *client);
   virtual ~TGFontPool();

   TGFont  *GetFont(const char *font, Bool_t fixedDefault = kTRUE);
   TGFont  *GetFont(const TGFont *font);
   TGFont  *GetFont(FontStruct_t font);

   void     FreeFont(const TGFont *font);

   TGFont  *FindFont(FontStruct_t font) const;
   TGFont  *FindFontByHandle(FontH_t font) const;

   void     Print(Option_t *option="") const;

   ClassDef(TGFontPool,0)  // Font pool
};

#endif
