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

#include "TGListView.h"


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
       ~TGItemContext() override { }
   void         Associate(TTVLVEntry *item) { fItem = item; }
   void Delete(Option_t *) override { }           // *MENU*
   void         Draw(Option_t *option="") override;       // *MENU*
   void         EditExpression();                // *MENU*
   void         Empty();                         // *MENU*
   void         RemoveItem();                    // *MENU*
   void         Scan();                          // *MENU*
   void         SetExpression(const char *name="", const char *alias="-empty-", bool cut=false); // *MENU*

   ClassDefOverride(TGItemContext, 0)  // Context menu for TTVLVEntry
};


class TTVLVEntry : public TGLVEntry {

protected:
   TTVLVContainer *fContainer;  ///< Container to whom this item belongs
   TString         fTrueName;   ///< Name for this entry
   TString         fAlias;      ///< Alias for this entry
   TString         fConvName;   ///< Name converted into true expressions
   TGToolTip      *fTip;        ///< Tool tip associated with item
   bool            fIsCut;      ///< Flag for cut type items
   TGItemContext  *fContext;    ///< Associated context menu

protected:
   bool           FullConverted();

public:
   TTVLVEntry(const TGWindow *p,
              const TGPicture *bigpic, const TGPicture *smallpic,
              TGString *name, TGString **subnames, EListViewMode ViewMode);
   ~TTVLVEntry() override;
   const char     *ConvertAliases();
   void            CopyItem(TTVLVEntry *dest);
   const char     *GetAlias() {return fAlias.Data();}
   TTVLVContainer *GetContainer() {return fContainer;}
   TGItemContext  *GetContext() {return fContext;}
   const char     *GetConvName() {return fConvName;}
   const char     *GetTrueName() {return fTrueName.Data();}
   TGToolTip      *GetTip() {return fTip;}
   bool            HandleCrossing(Event_t *event) override;
   bool            HasAlias();
   bool            IsCut() {return fIsCut;}
   void            PrependTilde();
   void            SetCutType(bool type=false);
   void            SetItemName(const char* name);
   void            SetAlias(const char* alias) {fAlias = alias;}
   void            SetExpression(const char* name, const char* alias, bool cutType=false);
   void            SetTrueName(const char* name) {fTrueName = name;}
   void            SetToolTipText(const char *text, Long_t delayms = 1000);
   void            SetSmallPic(const TGPicture *spic);
   void            Empty();

   ClassDefOverride(TTVLVEntry,0)  // Item that goes into the tree list view widget
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
   Cursor_t     fCursor;             ///< Current cursor
   Cursor_t     fDefaultCursor;      ///< Default cursor
   TGListView  *fListView;           ///< Associated list view
   TTreeViewer *fViewer;             ///< Pointer to tree viewer
   TList       *fExpressionList;     ///< List of user defined expression widgets

public:
   TTVLVContainer(const TGWindow *p, UInt_t w, UInt_t h, UInt_t options=kSunkenFrame);
   ~TTVLVContainer() override;
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
   void           SetListView(TGListView *lv) override {fListView = lv;}
   void           SetViewer(TTreeViewer *viewer) {fViewer = viewer;}
   void           RemoveNonStatic();
   const char    *ScanList();
   void           SelectItem(const char* name);
   bool           HandleButton(Event_t *event) override;
   bool           HandleMotion(Event_t *event) override;

   ClassDefOverride(TTVLVContainer,0)  // A dragging-capable LVContainer
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
   TTreeViewer       *fViewer;        ///< Pointer to tree viewer
   TGLabel           *fLabel;         ///< Label
   TTVLVEntry        *fEntry;         ///< Edited expression entry
   TGTextEntry       *fTe;            ///< Text entry box
   TGLabel           *fLabelAlias;    ///< Alias label
   TGTextEntry       *fTeAlias;       ///< Alias text entry
   TString            fOldAlias;      ///< Old alias for edited entry
   TGLayoutHints     *fLayout;        ///< Layout hints for widgets inside
   TGLayoutHints     *fBLayout;       ///< Layout for cancel button
   TGLayoutHints     *fBLayout1;      ///< Layout for close button
   TGHorizontalFrame *fBf;            ///< Buttons frame
   TGTextButton      *fDONE;          ///< Close button
   TGTextButton      *fCANCEL;        ///< Cancel button

protected:
   static TGSelectBox *fgInstance;// pointer to this select box

public:
   TGSelectBox(const TGWindow *p, const TGWindow *main, UInt_t w = 10, UInt_t h = 10);
   ~TGSelectBox() override;
   void   CloseWindow() override;
   TTVLVEntry    *EditedEntry() {return fEntry;}
   void           GrabPointer();
   void           SetLabel(const char* title);
   void           SetEntry(TTVLVEntry *entry);
   void           SaveText();
   void           InsertText(const char* text);
   bool ProcessMessage(Longptr_t msg, Longptr_t parm1, Longptr_t parm2) override;
   bool           ValidateAlias();

   static TGSelectBox *GetInstance();

   ClassDefOverride(TGSelectBox,0)  // TreeView dialog widget
};

#endif
