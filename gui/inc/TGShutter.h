// @(#)root/gui:$Name:$:$Id:$
// Author: Fons Rademakers   18/9/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGShutter
#define ROOT_TGShutter


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGShutter, TGShutterItem                                             //
//                                                                      //
// A shutter widget contains a set of shutter items that can be         //
// open and closed ilike a shutter.                                     //
// This widget is usefull to group a large number of options in         //
// a number of categories.                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif
#ifndef ROOT_TGCanvas
#include "TGCanvas.h"
#endif
#ifndef ROOT_TGWidget
#include "TGWidget.h"
#endif


class TGButton;
class TGCanvas;
class TTimer;
class TList;



class TGShutterItem : public TGVerticalFrame, public TGWidget {

friend class TGShutter;

protected:
   TGButton      *fButton;     // shutter item button
   TGCanvas      *fCanvas;     // canvas of shutter item
   TGFrame       *fContainer;  // container in canvas containing shutter items
   TGLayoutHints *fL1, *fL2;   // positioning hints

public:
   TGShutterItem(const TGWindow *p, TGHotString *s = 0, Int_t id = -1,
                 UInt_t options = 0);
   virtual ~TGShutterItem();

   TGFrame  *GetContainer() const { return fCanvas->GetContainer(); }

   ClassDef(TGShutterItem,0)  // Shutter widget item
};



class TGShutter : public TGCompositeFrame {

friend class OXShutterItem;

protected:
   TTimer         *fTimer;                  // Timer for animation
   TGShutterItem  *fSelectedItem;           // Item currently open
   TGShutterItem  *fClosingItem;            // Item closing down
   TList          *fTrash;                  // Items that need to be cleaned up
   Int_t           fHeightIncrement;        // Height delta
   Int_t           fClosingHeight;          // Closing items current height
   Int_t           fClosingHadScrollbar;    // Closing item had a scroll bar

public:
   TGShutter(const TGWindow *p, UInt_t options = kSunkenFrame);
   virtual ~TGShutter();

   virtual void   AddItem(TGShutterItem *item);
   virtual Bool_t HandleTimer(TTimer *t);
   virtual void   Layout();

   virtual Bool_t ProcessMessage(Long_t cmd, Long_t parm1, Long_t parm2);

   ClassDef(TGShutter,0)  // Shutter widget
};

#endif
