// @(#)root/gui:$Name$:$Id$
// Author: Fons Rademakers   12/01/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGListBox
#define ROOT_TGListBox


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGListBox, TGLBContainer, TGLBEntry and TGTextLBEntry                //
//                                                                      //
// A listbox is a box, possibly with scrollbar, containing entries.     //
// Currently entries are simple text strings (TGTextLBEntry).           //
// A TGListBox looks a lot like a TGCanvas. It has a TGViewPort         //
// containing a TGLBContainer which contains the entries and it also    //
// has a vertical scrollbar which becomes visible if there are more     //
// items than fit in the visible part of the container.                 //
//                                                                      //
// The TGListBox is user callable. The other classes are service        //
// classes of the listbox.                                              //
//                                                                      //
// Selecting an item in the listbox will generate the event:            //
// kC_COMMAND, kCM_LISTBOX, listbox id, item id.                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif
#ifndef ROOT_TGCanvas
#include "TGCanvas.h"
#endif

class TGViewPort;


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLBEntry                                                            //
//                                                                      //
// Basic listbox entries. Listbox entries are created by a TGListBox    //
// and not by the user.                                                 //                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGLBEntry : public TGFrame {

friend class TGClient;

protected:
   Int_t      fEntryId;          // message id of listbox entry
   ULong_t    fBkcolor;          // entry background color
   Bool_t     fActive;           // true if entry is active

   virtual void DoRedraw() { }

public:
   TGLBEntry(const TGWindow *p, Int_t id, UInt_t options = kHorizontalFrame,
             ULong_t back = fgWhitePixel) : TGFrame(p, 10, 10, options, back)
      { fActive = kFALSE; fEntryId = id; fBkcolor = back; }

   virtual void Activate(Bool_t a);
   virtual void Toggle();
   virtual void Update(TGLBEntry *) { }  // this is needed on TGComboBoxes :(
   Int_t  EntryId() const { return fEntryId; }
   Bool_t IsActive() const { return fActive;  }

   ClassDef(TGLBEntry,0)  // Basic listbox entry
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGTextLBEntry                                                        //
//                                                                      //
// Text string listbox entries.                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGTextLBEntry : public TGLBEntry {

friend class TGClient;

protected:
   TGString     *fText;           // entry text string
   UInt_t        fTWidth;         // text width
   UInt_t        fTHeight;        // text height
   Bool_t        fTextChanged;    // true if text has been changed
   GContext_t    fNormGC;         // text drawing graphics context
   FontStruct_t  fFontStruct;     // font used to draw string

   static ULong_t        fgSelPixel;
   static GContext_t     fgDefaultGC;
   static FontStruct_t   fgDefaultFontStruct;

   virtual void DoRedraw();

public:
   TGTextLBEntry(const TGWindow *p, TGString *s, Int_t id,
                 GContext_t norm = fgDefaultGC, FontStruct_t font = fgDefaultFontStruct,
                 UInt_t options = kHorizontalFrame, ULong_t back = fgWhitePixel);
   virtual ~TGTextLBEntry();

   virtual TGDimension GetDefaultSize() const { return TGDimension(fTWidth, fTHeight+1); }
   const TGString *GetText() const { return fText; }
   void SetText(TGString *new_text);
   virtual void Update(TGLBEntry *e)
       { SetText(new TGString(((TGTextLBEntry *)e)->GetText())); }

   ClassDef(TGTextLBEntry,0)  // Text listbox entry
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLBContainer                                                        //
//                                                                      //
// A Composite frame that contains a list of TGLBEnties.                //
// A TGLBContainer is created by the TGListBox and not by the user.     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGLBContainer : public TGCompositeFrame {

protected:
   TGLBEntry       *fLastActive;   // last active listbox entry in single selection listbox
   const TGWindow  *fMsgWindow;    // window handling container messages
   Bool_t          fMultiSelect;   // true if multi selection is switched on
   Int_t           fChangeStatus;  // defines the changes (select or unselect) while the mouse
                                   // moves over a multi selectable list box

public:
   TGLBContainer(const TGWindow *p, UInt_t w, UInt_t h,
                 UInt_t options = kSunkenFrame,
                 ULong_t back = fgDefaultFrameBackground);
   virtual ~TGLBContainer();

   virtual void AddEntry(TGLBEntry *lbe, TGLayoutHints *lhints);
   virtual void AddEntrySort(TGLBEntry *lbe, TGLayoutHints *lhints);
   virtual void InsertEntry(TGLBEntry *lbe, TGLayoutHints *lhints, Int_t afterID);
   virtual void RemoveEntry(Int_t id);
   virtual void RemoveEntries(Int_t from_ID, Int_t to_ID);

   virtual void   Associate(const TGWindow *w) { fMsgWindow = w; }
   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t HandleMotion(Event_t *event);
   virtual Int_t  GetSelected() const;
   virtual Bool_t GetSelection(Int_t id);
   virtual Int_t  GetPos(Int_t id);
   virtual TGLBEntry *GetSelectedEntry() const { return fLastActive; }
   virtual void   GetSelectedEntries(TList *selected);
   virtual TGLBEntry *Select(Int_t id, Bool_t sel);
   virtual TGLBEntry *Select(Int_t id);

   virtual void   SetMultipleSelections(Bool_t multi);
   virtual Bool_t GetMultipleSelections() const { return fMultiSelect; }

   ClassDef(TGLBContainer,0)  // Listbox container
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGListBox                                                            //
//                                                                      //
// A TGListBox widget.                                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGListBox : public TGCompositeFrame {

protected:

  Int_t            fListBoxId;       // listbox widget id
  UInt_t           fItemVsize;       // maximum height of single entry
  Bool_t           fIntegralHeight;  // true if height should be multiple of fItemVsize
  TGLBContainer   *fLbc;             // listbox container
  TGViewPort      *fVport;           // listbox viewport (see TGCanvas.h)
  TGVScrollBar    *fVScrollbar;      // vertical scrollbar
  const TGWindow  *fMsgWindow;       // window handling listbox messages

   void SetContainer(TGFrame *f) { fVport->SetContainer(f); }

   virtual void InitListBox();

public:
   TGListBox(const TGWindow *p, Int_t id,
             UInt_t options = kSunkenFrame | kDoubleBorder,
             ULong_t back = fgWhitePixel);
   virtual ~TGListBox();

   virtual void AddEntry(TGString *s, Int_t id);
   virtual void AddEntry(const char *s, Int_t id);
   virtual void AddEntry(TGLBEntry *lbe, TGLayoutHints *lhints);
   virtual void AddEntrySort(TGString *s, Int_t id);
   virtual void AddEntrySort(const char *s, Int_t id);
   virtual void AddEntrySort(TGLBEntry *lbe, TGLayoutHints *lhints);
   virtual void InsertEntry(TGString *s, Int_t id, Int_t afterID);
   virtual void InsertEntry(const char *s , Int_t id, Int_t afterID);
   virtual void InsertEntry(TGLBEntry *lbe, TGLayoutHints *lhints, Int_t afterID);
   virtual void RemoveEntry(Int_t id) { fLbc->RemoveEntry(id); }
   virtual void RemoveEntries(Int_t from_ID, Int_t to_ID)
                                  { fLbc->RemoveEntries(from_ID, to_ID); }
   virtual void SetTopEntry(Int_t id);
   virtual void SetMultipleSelections(Bool_t multi)
                                  { fLbc->SetMultipleSelections(multi); }
   virtual Bool_t GetMultipleSelections() const
                                  { return fLbc->GetMultipleSelections(); }

   TGFrame *GetContainer() const { return fVport->GetContainer(); }
   TGFrame *GetViewPort() const { return fVport; }
   virtual void DrawBorder();
   virtual void Resize(UInt_t w, UInt_t h);
   virtual void Resize(TGDimension size) { Resize(size.fWidth, size.fHeight); }
   virtual void MoveResize(Int_t x, Int_t y, UInt_t w, UInt_t h);
   virtual void Layout();
   virtual void IntegralHeight(Bool_t mode) { fIntegralHeight = mode; }
   virtual TGDimension GetDefaultSize() const;

   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);

   virtual void Associate(const TGWindow *w) { fMsgWindow = w; }
   virtual TGLBEntry *Select(Int_t id, Bool_t sel = kTRUE)
                                       { return fLbc->Select(id, sel); }
   virtual Int_t GetSelected() const;
   virtual Bool_t GetSelection(Int_t id) { return fLbc->GetSelection(id); }
   virtual TGLBEntry *GetSelectedEntry() const { return fLbc->GetSelectedEntry(); }
   virtual void GetSelectedEntries(TList *selected);

   ClassDef(TGListBox,0)  // Listbox widget
};

#endif
