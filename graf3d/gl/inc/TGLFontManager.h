// @(#)root/gl:$Id$
// Author: Alja Mrak-Tadel 2008

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLFontManager
#define ROOT_TGLFontManager

#include "TObjArray.h"
#include <list>
#include <vector>
#include <map>

class FTFont;
class TGLFontManager;

class TGLFont
{
public:
   enum EMode
   {
      kUndef = -1,
      kBitmap, kPixmap,
      kTexture, kOutline, kPolygon, kExtrude
   }; // Font-types of FTGL.

   enum ETextAlignH_e { kLeft, kRight, kCenterH };
   enum ETextAlignV_e   { kBottom, kTop, kCenterV };

private:
   TGLFont& operator=(const TGLFont& o); // Not implemented.

   FTFont          *fFont;     // FTGL font.
   TGLFontManager  *fManager;  // Font manager.

   Float_t          fDepth;  // depth of extruded fonts, enforced at render time.

   template<class Char>
   void RenderHelper(const Char *txt, Double_t x, Double_t y, Double_t angle, Double_t /*mgn*/) const;

protected:
   Int_t            fSize;   // free-type face size
   Int_t            fFile;   // free-type file name
   EMode            fMode;   // free-type FTGL class id

   mutable Int_t    fTrashCount;
public:
   TGLFont();
   TGLFont(Int_t size, Int_t font, EMode mode, FTFont *f=0, TGLFontManager *mng=0);
   TGLFont(const TGLFont& o);            // Not implemented.
   virtual ~TGLFont();

   void CopyAttributes(const TGLFont &o);

   Int_t GetSize() const { return fSize;}
   Int_t GetFile() const { return fFile;}
   EMode GetMode() const { return fMode;}

   Int_t GetTrashCount()        const { return fTrashCount;   }
   void  SetTrashCount(Int_t c) const { fTrashCount = c;      }
   Int_t IncTrashCount()        const { return ++fTrashCount; }

   void  SetFont(FTFont *f) { fFont =f;}
   const FTFont* GetFont() const { return fFont; }
   void  SetManager(TGLFontManager *mng)    { fManager = mng;  }
   const TGLFontManager* GetManager() const { return fManager; }

   Float_t GetDepth()    const { return fDepth; }
   void    SetDepth(Float_t d) { fDepth = d;    }

   // FTGL wrapper functions
   Float_t GetAscent() const;
   Float_t GetDescent() const;
   Float_t GetLineHeight() const;
   void    MeasureBaseLineParams(Float_t& ascent, Float_t& descent, Float_t& line_height,
                                 const char* txt="Xj") const;

   void  BBox(const char* txt,
              Float_t& llx, Float_t& lly, Float_t& llz,
              Float_t& urx, Float_t& ury, Float_t& urz) const;
   void  BBox(const wchar_t* txt,
              Float_t& llx, Float_t& lly, Float_t& llz,
              Float_t& urx, Float_t& ury, Float_t& urz) const;

   void  Render(const char* txt, Double_t x, Double_t y, Double_t angle, Double_t mgn) const;
   void  Render(const wchar_t* txt, Double_t x, Double_t y, Double_t angle, Double_t mgn) const;
   void  Render(const TString &txt) const;
   void  Render(const TString &txt, Float_t x, Float_t y, Float_t z, ETextAlignH_e alignH, ETextAlignV_e alignV) const;

   // helper gl draw functions
   virtual void PreRender(Bool_t autoLight=kTRUE, Bool_t lightOn=kFALSE) const;
   virtual void PostRender() const;

   Bool_t operator< (const TGLFont& o) const
   {
      if (fSize == o.fSize)
      {
         if(fFile == o.fFile)
         {
            return fMode < o.fMode;
         }
         return fFile < o.fFile;
      }
      return fSize < o.fSize;
   }

   ClassDef(TGLFont, 0); // A wrapper class for FTFont.
};

/******************************************************************************/
/******************************************************************************/

class TGLFontManager
{
public:
   typedef std::vector<Int_t> FontSizeVec_t;

private:
   TGLFontManager(const TGLFontManager&);            // Not implemented
   TGLFontManager& operator=(const TGLFontManager&); // Not implemented

protected:
   typedef std::map<TGLFont, Int_t>           FontMap_t;
   typedef std::map<TGLFont, Int_t>::iterator FontMap_i;

   typedef std::list<const TGLFont*>                  FontList_t;
   typedef std::list<const TGLFont*>::iterator        FontList_i;
   typedef std::list<const TGLFont*>::const_iterator  FontList_ci;

   FontMap_t            fFontMap;        // map of created fonts
   FontList_t           fFontTrash;      // fonts to purge

   static TObjArray     fgFontFileArray;      // map font-id to ttf-font-file
   // Default fonts - for gl/eve, "extended" - for gl-pad
   static Int_t         fgExtendedFontStart;

   static FontSizeVec_t fgFontSizeArray;      // map of valid font-size
   static Bool_t        fgStaticInitDone;     // global initialization flag
   static void          InitStatics();

public:
   TGLFontManager() : fFontMap(), fFontTrash() {}
   virtual ~TGLFontManager();

   void   RegisterFont(Int_t size, Int_t file, TGLFont::EMode mode, TGLFont& out);
   void   RegisterFont(Int_t size, const char* name, TGLFont::EMode mode, TGLFont& out);
   void   ReleaseFont(TGLFont& font);

   static TObjArray*        GetFontFileArray();
   static FontSizeVec_t*    GetFontSizeArray();

   static Int_t             GetExtendedFontStartIndex();
   static Int_t             GetFontSize(Int_t ds);
   static Int_t             GetFontSize(Int_t ds, Int_t min, Int_t max);
   static const char*       GetFontNameFromId(Int_t);

   void   ClearFontTrash();

   ClassDef(TGLFontManager, 0); // A FreeType GL font manager.
};

#endif

