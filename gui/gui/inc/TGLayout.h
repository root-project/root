// @(#)root/gui:$Id$
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


#include "TObject.h"
#include "TGDimension.h"
#include "TRefCnt.h"

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
class TGFrameElement;

/** \class TGLayoutHints
    \ingroup guiwidgets

This class describes layout hints used by the layout classes.

*/


class TGLayoutHints : public TObject, public TRefCnt {

friend class TGFrameElement;
friend class TGCompositeFrame;

private:
   TGFrameElement *fFE;       // back pointer to the last frame element
   TGFrameElement *fPrev;     // previous element sharing this layout_hints

   TGLayoutHints& operator=(const TGLayoutHints&) = delete;

protected:
   ULong_t  fLayoutHints;     // layout hints (combination of ELayoutHints)
   Int_t    fPadtop;          // amount of top padding
   Int_t    fPadbottom;       // amount of bottom padding
   Int_t    fPadleft;         // amount of left padding
   Int_t    fPadright;        // amount of right padding

   void UpdateFrameElements(TGLayoutHints *l);

public:
   TGLayoutHints(ULong_t hints = kLHintsNormal,
                 Int_t padleft = 0, Int_t padright = 0,
                 Int_t padtop = 0, Int_t padbottom = 0):
     fFE(nullptr), fPrev(nullptr), fLayoutHints(hints), fPadtop(padtop), fPadbottom(padbottom),
     fPadleft(padleft), fPadright(padright)
     { SetRefCount(0); }

   TGLayoutHints(const TGLayoutHints &lh);

   virtual ~TGLayoutHints();

   ULong_t GetLayoutHints() const { return fLayoutHints; }
   Int_t   GetPadTop() const { return fPadtop; }
   Int_t   GetPadBottom() const { return fPadbottom; }
   Int_t   GetPadLeft() const { return fPadleft; }
   Int_t   GetPadRight() const { return fPadright; }

   virtual void SetLayoutHints(ULong_t lh) { fLayoutHints = lh; }
   virtual void SetPadTop(Int_t v)  {  fPadtop = v; }
   virtual void SetPadBottom(Int_t v)  {  fPadbottom = v; }
   virtual void SetPadLeft(Int_t v)  {  fPadleft = v; }
   virtual void SetPadRight(Int_t v)  {  fPadright = v; }

   void Print(Option_t* option = "") const override;
   void ls(Option_t* option = "") const override { Print(option); }

   void SavePrimitive(std::ostream &out, Option_t *option = "") override;

   ClassDefOverride(TGLayoutHints,0)  // Class describing GUI layout hints
};

// Temporarily public as we need to share this class definition
// with the frame manager class

class TGFrameElement : public TObject {

private:
   TGFrameElement(const TGFrameElement&);
   TGFrameElement& operator=(const TGFrameElement&);

public:
   TGFrame        *fFrame;    // frame used in layout
   Int_t           fState;    // EFrameState defined in TGFrame.h
   TGLayoutHints  *fLayout;   // layout hints used in layout

   TGFrameElement() : fFrame(nullptr), fState(0), fLayout(nullptr) { }
   TGFrameElement(TGFrame *f, TGLayoutHints *l);
   ~TGFrameElement();

   void Print(Option_t* option = "") const override;
   void ls(Option_t* option = "") const override { Print(option); }

   ClassDefOverride(TGFrameElement, 0); // Base class used in GUI containers
};


/** \class TGLayoutManager
    \ingroup guiwidgets

Frame layout manager. This is an abstract class.

*/


class TGLayoutManager : public TObject {
protected:
   Bool_t            fModified;// kTRUE if positions of subframes changed after layout

public:
   TGLayoutManager() : fModified(kTRUE) {}

   virtual void Layout() = 0;
   virtual TGDimension GetDefaultSize() const = 0;
   virtual void SetDefaultWidth(UInt_t /* w */) {}
   virtual void SetDefaultHeight(UInt_t /* h */) {}
   virtual Bool_t IsModified() const { return fModified; }
   virtual void SetModified(Bool_t flag = kTRUE) { fModified = flag; }

   ClassDefOverride(TGLayoutManager,0)  // Layout manager abstract base class
};


/** \class TGVerticalLayout
    \ingroup guiwidgets

*/


class TGVerticalLayout : public TGLayoutManager {

protected:
   TGCompositeFrame  *fMain;     // container frame
   TList             *fList;     // list of frames to arrange

   TGVerticalLayout(const TGVerticalLayout& gvl) :
     TGLayoutManager(gvl), fMain(gvl.fMain), fList(gvl.fList) { }
   TGVerticalLayout& operator=(const TGVerticalLayout& gvl)
     {if(this!=&gvl) { TGLayoutManager::operator=(gvl);
     fMain=gvl.fMain; fList=gvl.fList;} return *this;}

public:
   TGVerticalLayout(TGCompositeFrame *main);

   void Layout() override;
   TGDimension GetDefaultSize() const override;
   void SavePrimitive(std::ostream &out, Option_t * = "") override;

   ClassDefOverride(TGVerticalLayout,0)  // Vertical layout manager
};

/** \class TGHorizontalLayout
    \ingroup guiwidgets

*/


class TGHorizontalLayout : public TGVerticalLayout {
public:
   TGHorizontalLayout(TGCompositeFrame *main) : TGVerticalLayout(main) { }

   void Layout() override;
   TGDimension GetDefaultSize() const override;
   void SavePrimitive(std::ostream &out, Option_t * = "") override;

   ClassDefOverride(TGHorizontalLayout,0)  // Horizontal layout manager
};


/** \class TGRowLayout
    \ingroup guiwidgets

The following two layout managers do not make use of TGLayoutHints.

*/


class TGRowLayout : public TGVerticalLayout {
public:
   Int_t   fSep;             // interval between frames

   TGRowLayout(TGCompositeFrame *main, Int_t s = 0) :
      TGVerticalLayout(main), fSep(s) { }

   void Layout() override;
   TGDimension GetDefaultSize() const override;
   void SavePrimitive(std::ostream &out, Option_t * = "") override;

   ClassDefOverride(TGRowLayout,0)  // Row layout manager
};

/** \class TGColumnLayout
    \ingroup guiwidgets

The following layout manager do not make use of TGLayoutHints.

*/


class TGColumnLayout : public TGRowLayout {
public:
   TGColumnLayout(TGCompositeFrame *main, Int_t s = 0) : TGRowLayout(main, s) { }

   void Layout() override;
   TGDimension GetDefaultSize() const override;
   void SavePrimitive(std::ostream &out, Option_t * = "") override;

   ClassDefOverride(TGColumnLayout,0)  // Column layout manager
};


/** \class TGMatrixLayout
    \ingroup guiwidgets

This layout managers does not make use of TGLayoutHints.


It arranges frames in a matrix-like way.
This manager provides :
- a column number (0 means unlimited)
- a row number (0 means unlimited)
- horizontal & vertical separators

Notes : If both column and row are fixed values, any remaining
        frames outside the count won't be managed.
        Unlimited rows means the frame can expand downward
        (the default behaviour in most UI).
        Both unlimited rows and columns is undefined (read: will
        crash the algorithm ;-).
        With fixed dimensions, frames are always arranged in rows.
        That is: 1st frame is at position (0,0), next one is at
        row(0), column(1) and so on...
        When specifying one dimension as unlimited (i.e. row=0 or
        column=0) the frames are arranged according to the direction
        of the fixed dimension. This layout manager does not make
        use of TGLayoutHints.
*/


class TGMatrixLayout : public TGLayoutManager {

private:
   TGMatrixLayout(const TGMatrixLayout&) = delete;
   TGMatrixLayout& operator=(const TGMatrixLayout&) = delete;

protected:
   TGCompositeFrame *fMain;           ///< container frame
   TList            *fList;           ///< list of frames to arrange

public:
   Int_t   fSep;                      ///< interval between frames
   Int_t   fHints;                    ///< layout hints (currently not used)
   UInt_t  fRows;                     ///< number of rows
   UInt_t  fColumns;                  ///< number of columns

   TGMatrixLayout(TGCompositeFrame *main, UInt_t r, UInt_t c, Int_t s=0, Int_t h=0);

   void Layout() override;
   TGDimension GetDefaultSize() const override;
   void SavePrimitive(std::ostream &out, Option_t * = "") override;

   ClassDefOverride(TGMatrixLayout,0)  // Matrix layout manager
};


/** \class TGTileLayout
    \ingroup guiwidgets

This is a layout manager for the TGListView widget.

*/


class TGTileLayout : public TGLayoutManager {

private:
   TGTileLayout(const TGTileLayout&) = delete;
   TGTileLayout& operator=(const TGTileLayout&) = delete;

protected:
   Int_t             fSep;     ///< separation between tiles
   TGCompositeFrame *fMain;    ///< container frame
   TList            *fList;     ///< list of frames to arrange
   Bool_t            fModified; ///< layout changed


public:
   TGTileLayout(TGCompositeFrame *main, Int_t sep = 0);

   void Layout() override;
   TGDimension GetDefaultSize() const override;
   Bool_t IsModified() const override { return fModified; }
   void SavePrimitive(std::ostream &out, Option_t * = "") override;

   ClassDefOverride(TGTileLayout,0)  // Tile layout manager
};

/** \class TGListLayout
    \ingroup guiwidgets

This is a layout manager for the TGListView widget.

*/


class TGListLayout : public TGTileLayout {
public:
   TGListLayout(TGCompositeFrame *main, Int_t sep = 0) :
      TGTileLayout(main, sep) { }

   void Layout() override;
   TGDimension GetDefaultSize() const override;
   void SavePrimitive(std::ostream &out, Option_t * = "") override;

   ClassDefOverride(TGListLayout,0)  // Layout manager for TGListView widget
};

/** \class TGListDetailsLayout
    \ingroup guiwidgets

This is a layout manager for the TGListView widget.

*/


class TGListDetailsLayout : public TGTileLayout {
private:
   UInt_t fWidth; // width of listview container

public:
   TGListDetailsLayout(TGCompositeFrame *main, Int_t sep = 0, UInt_t w = 0) :
      TGTileLayout(main, sep), fWidth(w) { }

   void Layout() override;
   TGDimension GetDefaultSize() const override;
   void SetDefaultWidth(UInt_t w) override { fWidth = w; }
   void SavePrimitive(std::ostream &out, Option_t * = "") override;

   ClassDefOverride(TGListDetailsLayout,0)  // Layout manager for TGListView details
};

#endif
