// @(#)root/guibuilder:$Id$
// Author: Valeriy Onuchin   12/09/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGuiBldNameFrame
#define ROOT_TGuiBldNameFrame

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGuiBldNameFrame - frame sdisplaying the class name of frame         //
//                    and the name  of frame                            //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif

class TGLabel;
class TGTextEntry;
class TGuiBldEditor;
class TGuiBldEditor;
class TRootGuiBuilder;
class TGListTree;
class TGFrame;
class TGCanvas;
class TGListTreeItem;
class TGuiBldDragManager;


//////////////////////////////////////////////////////////////////////////
class TGuiBldNameFrame : public TGCompositeFrame {

private:
   TGLabel              *fLabel;       // label of frame class name
   TGTextEntry          *fFrameName;   // name of the frame
   TGuiBldEditor        *fEditor;      // pointer to main editor
   TGCompositeFrame     *fTitleFrame;  // frame saying that it's "Name Frame"
   TRootGuiBuilder      *fBuilder;     // pointer to builder
   TGuiBldDragManager   *fManager;     // main manager
   TGListTree           *fListTree;    // list tree containing frames hierarchy
   TGCanvas             *fCanvas;

protected:
   void DoRedraw();

public:
   TGuiBldNameFrame(const TGWindow *p, TGuiBldEditor *editor);
   virtual ~TGuiBldNameFrame() { }

   void              ChangeSelected(TGFrame *frame);
   Bool_t            CheckItems(TGCompositeFrame *main);
   TGListTreeItem   *FindItemByName(TGListTree *tree, const char* name, TGListTreeItem *item = 0);
   TGCompositeFrame *GetMdi(TGFrame *frame);
   void              MapItems(TGCompositeFrame *main);
   void              RemoveFrame(TGFrame *frame);
   void              Reset();
   void              SelectFrameByItem(TGListTreeItem* item, Int_t i = 0);
   void              UpdateName();

   ClassDef(TGuiBldNameFrame, 0) // frame name editor
};


#endif
