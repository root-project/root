// @(#)root/gui:$Name$:$Id$
// Author: Fons Rademakers   13/01/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
// @(#)root/gui:$Name$:$Id$
// Author: Fons Rademakers   13/01/98

#ifndef ROOT_TGTab
#define ROOT_TGTab


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGTab, TGTabElement, TGTabLayout                                     //
//                                                                      //
// A tab widget contains a set of composite frames each with a little   //
// tab with a name (like a set of folders with tabs).                   //
//                                                                      //
// The TGTab is user callable. The TGTabElement and TGTabLayout are     //
// is a service classes of the tab widget.                              //
//                                                                      //
// Clicking on a tab will bring the associated composite frame to the   //
// front and generate the following event:                              //
// kC_COMMAND, kCM_TAB, tab id, 0.                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif
#ifndef ROOT_TGWidget
#include "TGWidget.h"
#endif

class TList;
class TGTab;


class TGTabElement : public TGFrame {

protected:
   TGString        *fText;            // text on tab
   GContext_t       fNormGC;          // graphics context for drawing tab
   FontStruct_t     fFontStruct;      // font used for tab
   UInt_t           fTWidth;          // width of tab text
   UInt_t           fTHeight;         // height of tab text

public:
   TGTabElement(const TGWindow *p, TGString *text, UInt_t w, UInt_t h,
                GContext_t norm, FontStruct_t font,
                UInt_t options = kRaisedFrame,
                ULong_t back = fgDefaultFrameBackground);
   virtual ~TGTabElement();

   virtual void        DrawBorder();
   virtual TGDimension GetDefaultSize() const;
   const TGString     *GetText() const { return fText; }
   const char         *GetString() const { return fText->GetString(); }
   void                SetText(TGString *text);

   ClassDef(TGTabElement,0)  // Little tab on tab widget
};


class TGTabLayout : public TGLayoutManager {

protected:
   TGTab    *fMain;      // container frame
   TList    *fList;      // list of frames to arrange

public:
   TGTabLayout(TGTab *main);

   virtual void Layout();
   virtual TGDimension GetDefaultSize() const;

   ClassDef(TGTabLayout,0)  // Layout manager for TGTab widget
};



class TGTab : public TGCompositeFrame, public TGWidget {

friend class TGClient;

protected:
   Int_t               fCurrent;        // index of current tab
   UInt_t              fTabh;           // tab height
   TGCompositeFrame   *fContainer;      // main container
   TList              *fRemoved;        // list of removed tabs
   FontStruct_t        fFontStruct;     // font
   GContext_t          fNormGC;         // drawing context

   static FontStruct_t    fgDefaultFontStruct;
   static GContext_t      fgDefaultGC;

   void ChangeTab(Int_t tabIndex);

public:
   TGTab(const TGWindow *p, UInt_t w, UInt_t h,
         GContext_t norm = fgDefaultGC, FontStruct_t font = fgDefaultFontStruct,
         UInt_t options = kChildFrame,
         ULong_t back = fgDefaultFrameBackground);
   virtual ~TGTab();

   virtual TGCompositeFrame *AddTab(TGString *text);
   virtual TGCompositeFrame *AddTab(const char *text);
   virtual void              RemoveTab(Int_t tabIndex);
   virtual Bool_t            SetTab(Int_t tabIndex);
   virtual void              DrawBorder() { }
   virtual Bool_t            HandleButton(Event_t *event);

   TGCompositeFrame *GetContainer() const { return fContainer; }
   Int_t             GetCurrent() const { return fCurrent; }
   TGCompositeFrame *GetTabContainer(Int_t tabIndex) const;
   TGTabElement     *GetTabTab(Int_t tabIndex) const;
   TGCompositeFrame *GetCurrentContainer() const { return GetTabContainer(fCurrent); }
   TGTabElement     *GetCurrentTab() const { return GetTabTab(fCurrent); }
   UInt_t            GetTabHeight() const { return fTabh; }
   Int_t             GetNumberOfTabs() const;

   ClassDef(TGTab,0)  // Tab widget
};

#endif
