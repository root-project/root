// @(#)root/gui:$Name:$:$Id:$
// Author: Reiner Rohlfs   24/03/2002

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun, Fons Rademakers and Reiner Rohlfs *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGXYLayout                                                           //
//                                                                      //
// Is a layout manager where the position and the size of each widget   //
// in the frame are defined by X / Y - coordinates. The coordinates     //
// for each widget are defined by the TGXYLayoutHints. Therefore it     //
// is not possible to share a layout hint for several widgets.          //
//                                                                      //
// The coordinates (X, Y) and the size (W, H) are defined in units      //
// of the size of a typical character. Also the size of the             //
// TGCompositeFrame for which a TGXYLayout manager is used has to be    //
// defined in its constructor in units of the size of a character!      //
//                                                                      //
// It is not possible to use any other layout hint than the             //
// TGXYLayoutHints for this layout manager!                             //
//                                                                      //
// The rubberFlag in the constructor of the TGLXYLayoutHins defines     //
// how the position and the size of a widget is recalculated if the     //
// size of the frame is increased:                                      //
// - kLRubberX: The X - position (left edge) is increased by the same   //
//              factor as the width of the frame increases.             //
// - kLRubberY: The Y - position (upper edge) is increased by the same  //
//              factor as the height of the frame increases.            //
// - kLRubberW: The width of the widget is increased by the same        //
//              factor as the width of the frame increases.             //
// - kLRubberY: The height of the widget is increased by the same       //
//              factor as the height of the frame increases.            //
// But the size never becomes smaller than defined by the               //
// TGXYLayoutHints and the X and Y coordinates becomes never smaller    //
// than defined by the layout hints.                                    //
//                                                                      //
// TGXYLayoutHints                                                      //
//                                                                      //
// This layout hint must be used for the TGXYLouyout manager!           //
//                                                                      //
//                                                                      //
// Example how to use this layout manager:                              //
//                                                                      //
// TGMyFrame::TGMyFrame()                                               //
//    : TGMainFrame(gClient->GetRoot(), 30, 12)                         //
//    // frame is 30 character long and 12 character heigh              //
// {                                                                    //
//    SetLayoutManager(new TGXYLayout(this));                           //
//                                                                      //
//    // create a button of size 8 X 1.8 at position 20 / 1             //
//    TGTextButton * button;                                            //
//    button = new TGTextButton(this, "&Apply", 1);                     //
//    AddFrame(button, new TGXYLayoutHints(20, 1, 8, 1.8));             //
//                                                                      //
//    // create a listbox of size 18 X 10 at position 1 / 1.            //
//    // The height will increase if the frame height increases         //
//    TGListBox * listBox;                                              //
//    listBox = new TGListBox(this, 2);                                 //
//    AddFrame(listBox, new TGXYLayoutHints(1, 1, 18, 10,               //
//             TGXYLayoutHints::kLRubberX |                             //
//             TGXYLayoutHints::kLRubberY |                             //
//             TGXYLayoutHints::kLRubberH ));                           //
//    .                                                                 //
//    .                                                                 //
//    .                                                                 //
// }                                                                    //
//                                                                      //
// Normaly there is one layout hint per widget. Therefore these         //
// can be deleted like in the following example in the desctuctor       //
// of the frame:                                                        //
//                                                                      //
// TGMyFrame::~TGMyFrame()                                              //
// {                                                                    //
//    // Destructor, deletes all frames and their layout hints.         //
//                                                                      //
//    TGFrameElement *ptr;                                              //
//                                                                      //
//    // delete all frames and layout hints                             //
//    if (fList) {                                                      //
//       TIter next(fList);                                             //
//       while ((ptr = (TGFrameElement *) next())) {                    //
//          if (ptr->fLayout)                                           //
//             delete ptr->fLayout;                                     //
//          if (ptr->fFrame)                                            //
//             delete ptr->fFrame;                                      //
//       }                                                              //
//    }                                                                 //
// }                                                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGXYLayout
#define ROOT_TGXYLayout

#ifndef ROOT_TGLayout
#include "TGLayout.h"
#endif


class TGXYLayoutHints : public TGLayoutHints {

protected:
   Double_t   fX;    // x - position of widget
   Double_t   fY;    // y - position of widget
   Double_t   fW;    // width of widget
   Double_t   fH;    // height of widget
   UInt_t     fFlag; // rubber flag

public:

   enum ERubberFlag {
      kLRubberX   = BIT(0),
      kLRubberY   = BIT(1),
      kLRubberW   = BIT(2),
      kLRubberH   = BIT(3)
   };

   TGXYLayoutHints(Double_t x, Double_t y, Double_t w, Double_t h,
                   UInt_t rubberFlag = kLRubberX | kLRubberY);

   Double_t  GetX() const { return fX; };
   Double_t  GetY() const { return fY; };
   Double_t  GetW() const { return fW; };
   Double_t  GetH() const { return fH; };
   UInt_t    GetFlag() const { return fFlag; };

   void      SetX(Double_t x) { fX = x; }
   void      SetY(Double_t y) { fY = y; }
   void      SetW(Double_t w) { fW = w; }
   void      SetH(Double_t h) { fH = h; }
   void      SetFlag(UInt_t flag) { fFlag = flag; }

   ClassDef(TGXYLayoutHints,0)  // Hits for the X / Y - layout manager
};


class TGXYLayout : public TGLayoutManager {

protected:
   TList            *fList;           // list of frames to arrange
   TGCompositeFrame *fMain;           // container frame

   Bool_t            fFirst;          // flag to determine the first call of Layout()
   UInt_t            fFirstWidth;     // original width of the frame fMain
   UInt_t            fFirstHeight;    // original height of the fram fMain

   Int_t             fTWidth;         // text width of a default character "1234567890" / 10
   Int_t             fTHeight;        // text height

public:
   TGXYLayout(TGCompositeFrame *main);

   virtual void Layout();
   virtual TGDimension GetDefaultSize() const;

   void NewSize() { fFirst = kTRUE; }

   ClassDef(TGXYLayout,0)  // X / Y - layout manager
};

#endif
