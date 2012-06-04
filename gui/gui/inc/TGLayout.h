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
#ifndef ROOT_TRefCnt
#include "TRefCnt.h"
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
class TGFrameElement;

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLayoutHints                                                        //
//                                                                      //
// This class describes layout hints used by the layout classes.        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGLayoutHints : public TObject, public TRefCnt {

friend class TGFrameElement;
friend class TGCompositeFrame;

private:
   TGFrameElement *fFE;       // back pointer to the last frame element
   TGFrameElement *fPrev;     // previous element sharing this layout_hints

   TGLayoutHints& operator=(const TGLayoutHints&);

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
     fFE(0), fPrev(0), fLayoutHints(hints), fPadtop(padtop), fPadbottom(padbottom),
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

   void Print(Option_t* option = "") const;
   void ls(Option_t* option = "") const { Print(option); }

   virtual void SavePrimitive(std::ostream &out, Option_t *option = "");

   ClassDef(TGLayoutHints,0)  // Class describing GUI layout hints
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

   TGFrameElement() : fFrame(0), fState(0), fLayout(0) { }
   TGFrameElement(TGFrame *f, TGLayoutHints *l);
   ~TGFrameElement();

   void Print(Option_t* option = "") const;
   void ls(Option_t* option = "") const { Print(option); }

   ClassDef(TGFrameElement, 0); // Base class used in GUI containers
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLayoutManager                                                      //
//                                                                      //
// Frame layout manager. This is an abstract class.                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

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
   virtual void   SetModified(Bool_t flag = kTRUE) { fModified = flag; }

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

   TGVerticalLayout(const TGVerticalLayout& gvl) :
     TGLayoutManager(gvl), fMain(gvl.fMain), fList(gvl.fList) { }
   TGVerticalLayout& operator=(const TGVerticalLayout& gvl)
     {if(this!=&gvl) { TGLayoutManager::operator=(gvl);
     fMain=gvl.fMain; fList=gvl.fList;} return *this;}

public:
   TGVerticalLayout(TGCompositeFrame *main);

   virtual void Layout();
   virtual TGDimension GetDefaultSize() const;
   virtual void SavePrimitive(std::ostream &out, Option_t * = "");

   ClassDef(TGVerticalLayout,0)  // Vertical layout manager
};

class TGHorizontalLayout : public TGVerticalLayout {
public:
   TGHorizontalLayout(TGCompositeFrame *main) : TGVerticalLayout(main) { }

   virtual void Layout();
   virtual TGDimension GetDefaultSize() const;
   virtual void SavePrimitive(std::ostream &out, Option_t * = "");

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
      TGVerticalLayout(main), fSep(s) { }

   virtual void Layout();
   virtual TGDimension GetDefaultSize() const;
   virtual void SavePrimitive(std::ostream &out, Option_t * = "");

   ClassDef(TGRowLayout,0)  // Row layout manager
};

class TGColumnLayout : public TGRowLayout {
public:
   TGColumnLayout(TGCompositeFrame *main, Int_t s = 0) : TGRowLayout(main, s) { }

   virtual void Layout();
   virtual TGDimension GetDefaultSize() const;
   virtual void SavePrimitive(std::ostream &out, Option_t * = "");

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

private:
   TGMatrixLayout(const TGMatrixLayout&);
   TGMatrixLayout& operator=(const TGMatrixLayout&);

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
   virtual void SavePrimitive(std::ostream &out, Option_t * = "");

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

private:
   TGTileLayout(const TGTileLayout&);
   TGTileLayout& operator=(const TGTileLayout&);

protected:
   Int_t             fSep;    // separation between tiles
   TGCompositeFrame *fMain;   // container frame
   TList            *fList;   // list of frames to arrange
   Bool_t            fModified;// layout changed


public:
   TGTileLayout(TGCompositeFrame *main, Int_t sep = 0);

   virtual void Layout();
   virtual TGDimension GetDefaultSize() const;
   virtual Bool_t IsModified() const { return fModified; }
   virtual void SavePrimitive(std::ostream &out, Option_t * = "");

   ClassDef(TGTileLayout,0)  // Tile layout manager
};

class TGListLayout : public TGTileLayout {
public:
   TGListLayout(TGCompositeFrame *main, Int_t sep = 0) :
      TGTileLayout(main, sep) { }

   virtual void Layout();
   virtual TGDimension GetDefaultSize() const;
   virtual void SavePrimitive(std::ostream &out, Option_t * = "");

   ClassDef(TGListLayout,0)  // Layout manager for TGListView widget
};

class TGListDetailsLayout : public TGTileLayout {
private:
   UInt_t fWidth; // width of listview container

public:
   TGListDetailsLayout(TGCompositeFrame *main, Int_t sep = 0, UInt_t w = 0) :
      TGTileLayout(main, sep), fWidth(w) { }

   virtual void Layout();
   virtual TGDimension GetDefaultSize() const;
   virtual void SetDefaultWidth(UInt_t w) { fWidth = w; }
   virtual void SavePrimitive(std::ostream &out, Option_t * = "");

   ClassDef(TGListDetailsLayout,0)  // Layout manager for TGListView details
};

#endif
