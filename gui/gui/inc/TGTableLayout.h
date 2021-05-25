// @(#)root/gui:$Id$
// Author: Brett Viren   04/15/2001

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGTableLayout
#define ROOT_TGTableLayout

#include "TGLayout.h"

// extension of ELayoutHints
enum ETableLayoutHints {
   kLHintsShrinkX = BIT(8),
   kLHintsShrinkY = BIT(9),
   kLHintsFillX   = BIT(10),
   kLHintsFillY   = BIT(11)
};




class TGTableLayoutHints : public TGLayoutHints {

private:
   TGTableLayoutHints(const TGTableLayoutHints&) = delete;
   TGTableLayoutHints& operator=(const TGTableLayoutHints&) = delete;

protected:
   UInt_t fAttachLeft;         ///< Column/row division number on which
   UInt_t fAttachRight;        ///< to attach the frame.  Starts at 0
   UInt_t fAttachTop;          ///< and goes to # columns / # rows
   UInt_t fAttachBottom;       ///< respectively

public:
   TGTableLayoutHints(UInt_t attach_left, UInt_t attach_right,
                      UInt_t attach_top, UInt_t attach_bottom,
                      ULong_t hints = kLHintsNormal,
                      UInt_t padleft = 0, UInt_t padright = 0,
                      UInt_t padtop = 0, UInt_t padbottom = 0)
      : TGLayoutHints(hints,padleft,padright,padtop,padbottom),
         fAttachLeft(attach_left),
         fAttachRight(attach_right),
         fAttachTop(attach_top),
         fAttachBottom(attach_bottom) { }
   virtual ~TGTableLayoutHints() { }

   UInt_t GetAttachLeft() const { return fAttachLeft; }
   UInt_t GetAttachRight() const { return fAttachRight; }
   UInt_t GetAttachTop() const { return fAttachTop; }
   UInt_t GetAttachBottom() const { return fAttachBottom; }
   virtual void SavePrimitive(std::ostream &out, Option_t * = "");

   ClassDef(TGTableLayoutHints,0)  // Class describing GUI table layout hints
};




class TGTableLayout : public TGLayoutManager {

private:
   TGTableLayout(const TGTableLayout&) = delete;
   TGTableLayout& operator=(const TGTableLayout&) = delete;

protected:
   struct TableData_t {
      UInt_t fDefSize;        ///< Default size of col/rows
      UInt_t fRealSize;       ///< Real size of col/rows (eg, if table resize)
      Bool_t fNeedExpand;
      Bool_t fNeedShrink;
      Bool_t fExpand;
      Bool_t fShrink;
      Bool_t fEmpty;
   };
   TableData_t        *fRow;          ///< info about each row
   TableData_t        *fCol;          ///< info about each column
   TGCompositeFrame   *fMain;         ///< container frame
   TList              *fList;         ///< list of frames to arrange
   Bool_t              fHomogeneous;  ///< all cols/rows same size

   void FindRowColSizes();
   void FindRowColSizesInit();
   void FindRowColSizesHomogeneous();
   void FindRowColSizesSinglyAttached();
   void FindRowColSizesMultiplyAttached();

   void SetRowColSizes();
   void SetRowColSizesInit();

   void CheckSanity();

   static void SetRowColResize(UInt_t real_size, UInt_t nthings,
                               TableData_t *thing, Bool_t homogeneous);

public:
   // these are public in TGMatrixLayout ???  Perpetuate it.
   Int_t   fSep;               ///< interval between frames
   Int_t   fHints;             ///< layout hints (currently not used)
   UInt_t  fNrows;             ///< number of rows
   UInt_t  fNcols;             ///< number of columns

   TGTableLayout(TGCompositeFrame *main, UInt_t nrows, UInt_t ncols,
                 Bool_t homogeneous = kFALSE, Int_t sep = 0, Int_t hints = 0);
   virtual ~TGTableLayout();

   virtual void Layout();
   virtual TGDimension GetDefaultSize() const; // return sum of all child sizes
   virtual void SavePrimitive(std::ostream &out, Option_t * = "");

   ClassDef(TGTableLayout,0)  // Table layout manager
};

#endif
