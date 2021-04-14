// Author: Roel Aaij   14/08/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGTableCell
#define ROOT_TGTableCell

#include "TGFrame.h"

class TGTable;
class TGString;
class TGTooltip;
class TGPicture;
class TObjArray;
class TGWindow;
class TGToolTip;

class TGTableCell : public TGFrame {

friend class TGTable;

protected:
   TGString      *fLabel;      ///< Text as shown in the cell
   TGToolTip     *fTip;        ///< Possible Tooltip
   Bool_t         fReadOnly;   ///< Cell readonly state
   Bool_t         fEnabled;    ///< Cell enabled state
   Int_t          fTMode;      ///< Text justify mode
   TGPicture     *fImage;      ///< Image or icon
   UInt_t         fTWidth;     ///< Label width
   UInt_t         fTHeight;    ///< Label height
   FontStruct_t   fFontStruct; ///< Font of the label
   Bool_t         fHasOwnFont; ///< Does the cell have it's own font
   GContext_t     fNormGC;     ///< graphics context used to draw the cell
   UInt_t         fColumn;     ///< Column this cell belongs to
   UInt_t         fRow;        ///< Row this cell belongs to
   TGTable       *fTable;      ///< TGTable that a cell belongs to

   static const TGGC   *fgDefaultGC;   ///< Default graphics context
   static const TGFont *fgDefaultFont; ///< Default font

   virtual void DoRedraw();
   virtual void DrawBorder();
   virtual void DrawBorder(Handle_t id, Int_t x, Int_t y);
   virtual void MoveDraw(Int_t x, Int_t y);
   virtual void Resize(UInt_t width, UInt_t height);
   virtual void Resize(TGDimension newsize);

   virtual void Highlight();
   void         Init(Bool_t resize);

public:
   static FontStruct_t GetDefaultFontStruct();
   static const TGGC   &GetDefaultGC();

   TGTableCell(const TGWindow *p = 0, TGTable *table = 0, TGString *label = 0,
               UInt_t row = 0, UInt_t column = 0, UInt_t width = 80,
               UInt_t height = 25, GContext_t norm = GetDefaultGC()(),
               FontStruct_t font = GetDefaultFontStruct(), UInt_t option = 0,
               Bool_t resize = kTRUE);
   TGTableCell(const TGWindow *p, TGTable *table, const char *label,
               UInt_t row = 0, UInt_t column = 0, UInt_t width = 80,
               UInt_t height = 25, GContext_t norm = GetDefaultGC()(),
               FontStruct_t font = GetDefaultFontStruct(),
               UInt_t option = 0, Bool_t resize =  kTRUE);

   virtual ~TGTableCell();

   virtual void DrawCopy(Handle_t id, Int_t x, Int_t y);

   virtual void SetLabel(const char *label);

   virtual void SetImage(TGPicture *image);
   //   virtual void SetBckgndGC(TGGC *gc);

   virtual void SetTextJustify(Int_t tmode);
   virtual void SetFont(FontStruct_t font);
   virtual void SetFont(const char *fontName);

   virtual void Select();
   virtual void SelectRow();
   virtual void SelectColumn();

   virtual UInt_t      GetColumn() const { return fColumn; }
   virtual UInt_t      GetRow() const { return fRow; };
   virtual TGString*   GetLabel() const { return fLabel; }
   virtual TGPicture*  GetImage() const { return fImage; }
   virtual UInt_t      GetWidth() const { return fWidth; }
   virtual UInt_t      GetHeight() const {return fHeight; }
   virtual TGDimension GetSize() const { return TGDimension(fWidth, fHeight); }
   virtual Int_t       GetTextJustify() const { return fTMode; }

   ClassDef(TGTableCell, 0) // A single cell in a TGTable.
} ;

#endif
