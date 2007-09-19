// @(#)root/treeviewer:$Id$
//Author : Andrei Gheata   16/08/00

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTVLVContainer
#define ROOT_TTVLVContainer


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TTVLVEntry                                                           //
//                                                                      //
// This class represent entries that goes into the TreeViewer           //
// listview container. It subclasses TGLVEntry and adds 2               //
// data members: the item true name and the alias                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGListView
#include "TGListView.h"
#endif


class TGLabel;
class TGTextEntry;
class TTreeViewer;
class TGToolTip;
class TTVLVEntry;
class TTVLVContainer;

class TGItemContext : public TObject {

protected:
   TTVLVEntry *fItem;       // pointer to associated item
public:
   TGItemContext();
   virtual     ~TGItemContext() { }
   void         Associate(TTVLVEntry *item) { fItem = item; }
   virtual void Delete(Option_t *) { }           // *MENU*
   void         Draw(Option_t *option="");       // *MENU*
   void         EditExpression();                // *MENU*
   void         Empty();                         // *MENU*
   void         RemoveItem();                    // *MENU*
   void         Scan();                          // *MENU*
   void         SetExpression(const char *name="", const char *alias="-empty-", Bool_t cut=kFALSE); // *MENU*

   ClassDef(TGItemContext, 0)  // Context menu for TTVLVEntry
};


class TTVLVEntry : public TGLVEntry {

protected:
   TTVLVContainer *fContainer;  // container to whom this item belongs
   TString         fTrueName;   // name for this entry
   TString         fAlias;      // alias for this entry
   TString         fConvName;   // name converted into true expressions
   TGToolTip      *fTip;        // tool tip associated with item
   Bool_t          fIsCut;      // flag for cut type items
   TGItemContext  *fContext;    // associated context menu

protected:
   Bool_t         FullConverted();

public:
   TTVLVEntry(const TGWindow *p,
              const TGPicture *bigpic, const TGPicture *smallpic,
              TGString *name, TGString **subnames, EListViewMode ViewMode);
   virtual ~TTVLVEntry();
   const char     *ConvertAliases();
   void            CopyItem(TTVLVEntry *dest);
   const char     *GetAlias() {return fAlias.Data();}
   TTVLVContainer *GetContainer() {return fContainer;}
   TGItemContext  *GetContext() {return fContext;}
   const char     *GetConvName() {return fConvName;}
   const char     *GetTrueName() {return fTrueName.Data();}
   TGToolTip      *GetTip() {return fTip;}
   virtual Bool_t  HandleCrossing(Event_t *event);
   Bool_t          HasAlias();
   Bool_t          IsCut() {return fIsCut;}
   void            PrependTilde();
   void            SetCutType(Bool_t type=kFALSE);
   void            SetItemName(const char* name);
   void            SetAlias(const char* alias) {fAlias = alias;}
   void            SetExpression(const char* name, const char* alias, Bool_t cutType=kFALSE);
   void            SetTrueName(const char* name) {fTrueName = name;}
   void            SetToolTipText(const char *text, Long_t delayms = 1000);
   void            SetSmallPic(const TGPicture *spic);
   void            Empty();

   ClassDef(TTVLVEntry,0)  // Item that goes into the tree list view widget
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
//   TTVLVContainer                                                     //
//                                                                      //
// This class represent the list view container for the                 //
// TreeView class. It is a TGLVContainer with item dragging             //
// capabilities for the TTVLVEntry objects inside                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TTVLVContainer : public TGLVContainer {

friend class TGClient;

private:
   Cursor_t     fCursor;             // current cursor
   Cursor_t     fDefaultCursor;      // default cursor
   TGListView  *fListView;           // associated list view
   TTreeViewer *fViewer;             // pointer to tree viewer
   TList       *fExpressionList;     // list of user defined expression widgets
public:
   TTVLVContainer(const TGWindow *p, UInt_t w, UInt_t h, UInt_t options=kSunkenFrame);
   virtual ~TTVLVContainer();
   virtual void AddThisItem(TTVLVEntry *item)
                  { AddFrame(item, fItemLayout); item->SetColumns(fCpos, fJmode); }
   const char    *Cut();
   void           EmptyAll();     // empty all items of expression type
   TTVLVEntry    *ExpressionItem(Int_t index);
   TList         *ExpressionList();
   const char    *Ex();
   const char    *Ey();
   const char    *Ez();
   TTreeViewer   *GetViewer() {return fViewer;}
   void           SetListView(TGListView *lv) {fListView = lv;}
   void           SetViewer(TTreeViewer *viewer) {fViewer = viewer;}
   void           RemoveNonStatic();
   const char    *ScanList();
   void           SelectItem(const char* name);
   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t HandleMotion(Event_t *event);

   ClassDef(TTVLVContainer,0)  // A dragging-capable LVContainer
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGSelectBox                                                          //
//                                                                      //
// This class represent a specialized expression editor for             //
// TTVLVEntry 'true name' and 'alias' data members.                     //
// It is a singleton in order to be able to use it for several          //
// expressions.                                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGSelectBox : public TGTransientFrame {

private:
   TTreeViewer       *fViewer;        // pointer to tree viewer
   TGLabel           *fLabel;         // label
   TTVLVEntry        *fEntry;         // edited expression entry
   TGTextEntry       *fTe;            // text entry box
   TGLabel           *fLabelAlias;    // alias label
   TGTextEntry       *fTeAlias;       // alias text entry
   TString            fOldAlias;      // old alias for edited entry
   TGLayoutHints     *fLayout;        // layout hints for widgets inside
   TGLayoutHints     *fBLayout;       // layout for cancel button
   TGLayoutHints     *fBLayout1;      // layout for close button
   TGHorizontalFrame *fBf;            // buttons frame
   TGTextButton      *fDONE;          // close button
   TGTextButton      *fCANCEL;        // cancel button

protected:
   static TGSelectBox *fgInstance;// pointer to this select box

public:
   TGSelectBox(const TGWindow *p, const TGWindow *main, UInt_t w = 10, UInt_t h = 10);
   virtual ~TGSelectBox();
   virtual void   CloseWindow();
   TTVLVEntry    *EditedEntry() {return fEntry;}
   void           GrabPointer();
   void           SetLabel(const char* title);
   void           SetEntry(TTVLVEntry *entry);
   void           SaveText();
   void           InsertText(const char* text);
   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);
   Bool_t         ValidateAlias();

   static TGSelectBox *GetInstance();

   ClassDef(TGSelectBox,0)  // TreeView dialog widget
};

#endif
