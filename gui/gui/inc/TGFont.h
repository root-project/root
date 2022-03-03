// @(#)root/gui:$Id$
// Author: Fons Rademakers   20/5/2003

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGFont
#define ROOT_TGFont


#include "TNamed.h"
#include "TGObject.h"
#include "TRefCnt.h"

class THashTable;
class TObjString;
class TGFont;

// Flags passed to TGFont::MeasureChars and TGFont::ComputeTextLayout

enum ETextLayoutFlags {
   kTextWholeWords = BIT(0),
   kTextAtLeastOne = BIT(1),
   kTextPartialOK  = BIT(2),
   kTextIgnoreTabs = BIT(3),
   kTextIgnoreNewlines = BIT(4)
};

enum EFontWeight {
   kFontWeightNormal = 0,
   kFontWeightMedium = 0,
   kFontWeightBold = 1,
   kFontWeightLight = 2,
   kFontWeightDemibold = 3,
   kFontWeightBlack = 4,
   kFontWeightUnknown = -1
};

enum EFontSlant {
   kFontSlantRoman = 0,
   kFontSlantItalic = 1,
   kFontSlantOblique = 2,
   kFontSlantUnknown = -1
};


struct FontMetrics_t {
   Int_t   fAscent;          // from baseline to top of font
   Int_t   fDescent;         // from baseline to bottom of font
   Int_t   fLinespace;       // the sum of the ascent and descent
   Int_t   fMaxWidth;        // width of widest character in font
   Bool_t  fFixed;           // true if monospace, false otherwise
};


struct FontAttributes_t {

   const char *fFamily; // Font family. The most important field.
   Int_t fPointsize;    // Pointsize of font, 0 for default size, or negative number meaning pixel size.
   Int_t fWeight;       // Weight flag; see below for def'n.
   Int_t fSlant;        // Slant flag; see below for def'n.
   Int_t fUnderline;    // Non-zero for underline font.
   Int_t fOverstrike;   // Non-zero for overstrike font.

   FontAttributes_t():  // default constructor
      fFamily    (0),
      fPointsize (0),
      fWeight    (kFontWeightNormal),
      fSlant     (kFontSlantRoman),
      fUnderline (0),
      fOverstrike(0) { }

   FontAttributes_t(const FontAttributes_t& f): // copy constructor
      fFamily    (f.fFamily),
      fPointsize (f.fPointsize),
      fWeight    (f.fWeight),
      fSlant     (f.fSlant),
      fUnderline (f.fUnderline),
      fOverstrike(f.fOverstrike) { }

   FontAttributes_t& operator=(const FontAttributes_t& f) // assignment operator
   {
      if (this != &f) {
         fFamily     = f.fFamily;
         fPointsize  = f.fPointsize;
         fWeight     = f.fWeight;
         fSlant      = f.fSlant;
         fUnderline  = f.fUnderline;
         fOverstrike = f.fOverstrike;
      }
      return *this;
   }

};



struct LayoutChunk_t;


class TGTextLayout : public TObject {

friend class TGFont;

protected:
   const TGFont  *fFont;         ///< The font used when laying out the text.
   const char    *fString;       ///< The string that was laid out.
   Int_t          fWidth;        ///< The maximum width of all lines in the text layout.
   Int_t          fNumChunks;    ///< Number of chunks actually used in following array.
   LayoutChunk_t *fChunks;       ///< Array of chunks. The actual size will be maxChunks.

   TGTextLayout(const TGTextLayout &tlayout) = delete;
   void operator=(const TGTextLayout &tlayout) = delete;

public:
   TGTextLayout(): fFont(nullptr), fString(""), fWidth(0), fNumChunks(0), fChunks(NULL) {}
   virtual ~TGTextLayout();

   void   DrawText(Drawable_t dst, GContext_t gc, Int_t x, Int_t y,
                   Int_t firstChar, Int_t lastChar) const;
   void   UnderlineChar(Drawable_t dst, GContext_t gc,
                        Int_t x, Int_t y, Int_t underline) const;
   Int_t  PointToChar(Int_t x, Int_t y) const;
   Int_t  CharBbox(Int_t index, Int_t *x, Int_t *y, Int_t *w, Int_t *h) const;
   Int_t  DistanceToText(Int_t x, Int_t y) const;
   Int_t  IntersectText(Int_t x, Int_t y, Int_t w, Int_t h) const;
   void   ToPostscript(TString *dst) const;

   ClassDef(TGTextLayout,0)   // Keep track of string  measurement information.
};


// The following class is used to keep track of the generic information about a font.

class TGFont : public TNamed, public TRefCnt {

friend class TGFontPool;
friend class TGTextLayout;

private:
   FontStruct_t     fFontStruct;      ///< Low level graphics fontstruct
   FontH_t          fFontH;           ///< Font handle (derived from fontstruct)
   FontMetrics_t    fFM;              ///< Cached font metrics
   FontAttributes_t fFA;              ///< Actual font attributes obtained when the font was created
   TObjString      *fNamedHash;       ///< Pointer to the named object TGFont was based on
   Int_t            fTabWidth;        ///< Width of tabs in this font (pixels).
   Int_t            fUnderlinePos;    ///< Offset from baseline to origin of underline bar
                                      ///< (used for drawing underlines on a non-underlined font).
   Int_t            fUnderlineHeight; ///< Height of underline bar (used for drawing
                                      ///< underlines on a non-underlined font).
   char             fTypes[256];      ///< Array giving types of all characters in
                                      ///< the font, used when displaying control characters.
   Int_t            fWidths[256];     ///< Array giving widths of all possible characters in the font.
   Int_t            fBarHeight;       ///< Height of underline or overstrike bar
                                      ///< (used for simulating a native underlined or strikeout font).

protected:
   TGFont(const char *name)
     : TNamed(name,""), TRefCnt(), fFontStruct(0), fFontH(0), fFM(),
     fFA(), fNamedHash(0), fTabWidth(0), fUnderlinePos(0), fUnderlineHeight(0), fBarHeight(0)
   {
      SetRefCount(1);
      for (Int_t i=0; i<256; i++) {
         fWidths[i] = 0;
         fTypes[i]  = ' ';
      }
   }

   TGFont(const TGFont &) = delete;
   void operator=(const TGFont &) = delete;

   LayoutChunk_t *NewChunk(TGTextLayout *layout, int *maxPtr,
                           const char *start, int numChars,
                           int curX, int newX, int y) const;
public:
   virtual ~TGFont();

   FontH_t      GetFontHandle() const { return fFontH; }
   FontStruct_t GetFontStruct() const { return fFontStruct; }
   FontStruct_t operator()() const;
   void         GetFontMetrics(FontMetrics_t *m) const;
   FontAttributes_t GetFontAttributes() const { return fFA; }

   Int_t  PostscriptFontName(TString *dst) const;
   Int_t  TextWidth(const char *string, Int_t numChars = -1) const;
   Int_t  XTextWidth(const char *string, Int_t numChars = -1) const;
   Int_t  TextHeight() const { return fFM.fLinespace; }
   void   UnderlineChars(Drawable_t dst, GContext_t gc,
                        const char *string, Int_t x, Int_t y,
                        Int_t firstChar, Int_t lastChar) const;
   TGTextLayout *ComputeTextLayout(const char *string, Int_t numChars,
                                  Int_t wrapLength, Int_t justify, Int_t flags,
                                  UInt_t *width, UInt_t *height) const;
   Int_t  MeasureChars(const char *source, Int_t numChars, Int_t maxLength,
                      Int_t flags, Int_t *length) const;
   void   DrawCharsExp(Drawable_t dst, GContext_t gc, const char *source,
                      Int_t numChars, Int_t x, Int_t y) const;
   void   DrawChars(Drawable_t dst, GContext_t gc, const char *source,
                   Int_t numChars, Int_t x, Int_t y) const;

   void  Print(Option_t *option="") const;
   virtual void SavePrimitive(std::ostream &out, Option_t * = "");

   ClassDef(TGFont,0)   // GUI font description
};


struct FontStateMap_t;
struct XLFDAttributes_t;


class TGFontPool : public TGObject {

private:
   THashTable    *fList;       // TGFont objects pool
   THashTable    *fUidTable;   // Hash table for some used string values like family names, etc.
   THashTable    *fNamedTable; // Map a name to a set of attributes for a font

   TGFontPool(const TGFontPool& fp) = delete;
   TGFontPool& operator=(const TGFontPool& fp) = delete;

protected:
   const char *GetUid(const char *string);
   Bool_t      ParseXLFD(const char *string, XLFDAttributes_t *xa);
   TGFont     *GetFontFromAttributes(FontAttributes_t *fa, TGFont *fontPtr);
   int         FindStateNum(const FontStateMap_t *map, const char *strKey);
   const char *FindStateString(const FontStateMap_t *map, int numKey);
   Bool_t      FieldSpecified(const char *field);
   TGFont     *GetNativeFont(const char *name, Bool_t fixedDefault = kTRUE);
   TGFont     *MakeFont(TGFont *font, FontStruct_t fontStruct, const char *fontName);

public:
   TGFontPool(TGClient *client);
   virtual ~TGFontPool();

   TGFont  *GetFont(const char *font, Bool_t fixedDefault = kTRUE);
   TGFont  *GetFont(const TGFont *font);
   TGFont  *GetFont(FontStruct_t font);
   TGFont  *GetFont(const char *family, Int_t ptsize, Int_t weight, Int_t slant);

   void     FreeFont(const TGFont *font);

   TGFont  *FindFont(FontStruct_t font) const;
   TGFont  *FindFontByHandle(FontH_t font) const;

   char   **GetAttributeInfo(const FontAttributes_t *fa);
   void     FreeAttributeInfo(char **info);
   char   **GetFontFamilies();
   void     FreeFontFamilies(char **f);
   Bool_t   ParseFontName(const char *string, FontAttributes_t *fa);
   const char *NameOfFont(TGFont *font);

   void     Print(Option_t *option="") const;

   ClassDef(TGFontPool,0)  // Font pool
};

#endif
