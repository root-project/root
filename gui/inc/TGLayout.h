// @(#)root/gui:$Name:  $:$Id: TGLayout.h,v 1.2 2000/10/20 12:22:08 rdm Exp $
// Author: Fons Rademakers   02/01/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLayout
#define ROOT_TGLayout


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// A number of different layout classes (TGLayoutManager,               //
// TGVerticalLayout, TGHorizontalLayout, TGLayoutHints, etc.).          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TGDimension
#include "TGDimension.h"
#endif


//---- layout hints

enum ELayoutHints {
   kLHintsNoHints = 0,
   kLHintsLeft    = BIT(0),
   kLHintsCenterX = BIT(1),
   kLHintsRight   = BIT(2),
   kLHintsTop     = BIT(3),
   kLHintsCenterY = BIT(4),
   kLHintsBottom  = BIT(5),
   kLHintsExpandX = BIT(6),
   kLHintsExpandY = BIT(7),
   kLHintsNormal  = (kLHintsLeft | kLHintsTop)
   // bits 8-11 used by ETableLayoutHints
};

class TGFrame;
class TGCompositeFrame;
class TGLayoutHints;
class TList;


// Temporarily public as we need to share this class definition
// with the frame manager class

class TGFrameElement : public TObject {
public:
   TGFrame        *fFrame;
   Int_t           fState;
   TGLayoutHints  *fLayout;
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLayoutHints                                                        //
//                                                                      //
// This class describes layout hints used by the layout classes.        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGLayoutHints : public TObject {

protected:
   ULong_t  fLayoutHints;     // layout hints (combination of ELayoutHints)
   UInt_t   fPadtop;          // amount of top padding
   UInt_t   fPadbottom;       // amount of bottom padding
   UInt_t   fPadleft;         // amount of left padding
   UInt_t   fPadright;        // amount of right padding

public:
   TGLayoutHints(ULong_t hints = kLHintsNormal,
                 UInt_t padleft = 0, UInt_t padright = 0,
                 UInt_t padtop = 0, UInt_t padbottom = 0)
       { fPadleft = padleft; fPadright = padright;
         fPadtop  = padtop;  fPadbottom = padbottom;
         fLayoutHints = hints; }
   virtual ~TGLayoutHints() { }

   ULong_t GetLayoutHints() const { return fLayoutHints; }
   UInt_t  GetPadTop() const { return fPadtop; }
   UInt_t  GetPadBottom() const { return fPadbottom; }
   UInt_t  GetPadLeft() const { return fPadleft; }
   UInt_t  GetPadRight() const { return fPadright; }

   ClassDef(TGLayoutHints,0)  // Class describing GUI layout hints
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLayoutManager                                                      //
//                                                                      //
// Frame layout manager. This is an abstract class.                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGLayoutManager : public TObject {
public:
   virtual void Layout() = 0;
   virtual TGDimension GetDefaultSize() const = 0;

   ClassDef(TGLayoutManager,0)  // Layout manager abstract base class
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGVerticalLayout and TGHorizontalLayout managers.                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGVerticalLayout : public TGLayoutManager {

protected:
   TGCompositeFrame  *fMain;     // container frame
   TList             *fList;     // list of frames to arrange

public:
   TGVerticalLayout(TGCompositeFrame *main);

   virtual void Layout();
   virtual TGDimension GetDefaultSize() const;

   ClassDef(TGVerticalLayout,0)  // Vertical layout manager
};

class TGHorizontalLayout : public TGVerticalLayout {
public:
   TGHorizontalLayout(TGCompositeFrame *main) : TGVerticalLayout(main) { }

   virtual void Layout();
   virtual TGDimension GetDefaultSize() const;

   ClassDef(TGHorizontalLayout,0)  // Horizontal layout manager
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGRowLayout and TGColumnLayout managers.                             //
//                                                                      //
// The follwing two layout managers do not make use of TGLayoutHints.   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGRowLayout : public TGVerticalLayout {
public:
   Int_t   fSep;             // interval between frames

   TGRowLayout(TGCompositeFrame *main, Int_t s = 0) :
      TGVerticalLayout(main) { fSep = s; }

   virtual void Layout();
   virtual TGDimension GetDefaultSize() const;

   ClassDef(TGRowLayout,0)  // Row layout manager
};

class TGColumnLayout : public TGRowLayout {
public:
   TGColumnLayout(TGCompositeFrame *main, Int_t s = 0) : TGRowLayout(main, s) { }

   virtual void Layout();
   virtual TGDimension GetDefaultSize() const;

   ClassDef(TGColumnLayout,0)  // Column layout manager
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGMatrixLayout manager.                                              //
//                                                                      //
// This layout managers does not make use of TGLayoutHints.             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGMatrixLayout : public TGLayoutManager {
protected:
   TGCompositeFrame *fMain;      // container frame
   TList            *fList;      // list of frames to arrange

public:
   Int_t   fSep;                      // interval between frames
   Int_t   fHints;                    // layout hints (currently not used)
   UInt_t  fRows;                     // number of rows
   UInt_t  fColumns;                  // number of columns

   TGMatrixLayout(TGCompositeFrame *main, UInt_t r, UInt_t c, Int_t s=0, Int_t h=0);

   virtual void Layout();
   virtual TGDimension GetDefaultSize() const;

   ClassDef(TGMatrixLayout,0)  // Matrix layout manager
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGTileLayout, TGListLayout and TGListDetailsLayout managers.         //
//                                                                      //
// This are layout managers for the TGListView widget.                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGTileLayout : public TGLayoutManager {

protected:
   Int_t             fSep;    // separation between tiles
   TGCompositeFrame *fMain;   // container frame
   TList            *fList;   // list of frames to arrange

public:
   TGTileLayout(TGCompositeFrame *main, Int_t sep = 0);

   virtual void Layout();
   virtual TGDimension GetDefaultSize() const;

   ClassDef(TGTileLayout,0)  // Tile layout manager
};

class TGListLayout : public TGTileLayout {
public:
   TGListLayout(TGCompositeFrame *main, Int_t sep = 0) :
      TGTileLayout(main, sep) { }

   virtual void Layout();
   virtual TGDimension GetDefaultSize() const;

   ClassDef(TGListLayout,0)  // Layout manager for TGListView widget
};

class TGListDetailsLayout : public TGTileLayout {
public:
   TGListDetailsLayout(TGCompositeFrame *main, Int_t sep = 0) :
      TGTileLayout(main, sep) { }

   virtual void Layout();
   virtual TGDimension GetDefaultSize() const;

   ClassDef(TGListDetailsLayout,0)  // Layout manager for TGListView details
};

#endif
