// @(#)root/treeviewer:$Name:  $:$Id: TGTreeLVC.h,v 1.4 2000/11/22 16:27:44 rdm Exp $
//Author : Andrei Gheata   16/08/00

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGTreeLVC
#define ROOT_TGTreeLVC


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLVTreeEntry                                                        //
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


class TGLVTreeEntry : public TGLVEntry {

protected:
   TString      fTrueName;   // name for this entry
   TString      fAlias;      // alias for this entry

public:
   TGLVTreeEntry(const TGWindow *p,
                 const TGPicture *bigpic, const TGPicture *smallpic,
                 TGString *name, TGString **subnames, EListViewMode ViewMode);
   virtual      ~TGLVTreeEntry() {}
   void         CopyItem(TGLVTreeEntry *dest);
   const char*  GetAlias() {return fAlias.Data();}
   const char*  GetTrueName() {return fTrueName.Data();}
   Bool_t       HasAlias();
   void         SetItemName(const char* name);
   void         SetAlias(const char* alias) {fAlias = alias;}
   void         SetTrueName(const char* name) {fTrueName = name;}
   void         SetSmallPic(const TGPicture *spic);
   void         Empty();

   ClassDef(TGLVTreeEntry,0)  // Item that goes into the tree list view widget
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//   TGTreeLVC                                                          //
//                                                                      //
// This class represent the list view container for the                 //
// TreeView class. It is a TGLVContainer with item dragging             //
// capabilities for the TGLVTreeEntry objects inside                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGTreeLVC : public TGLVContainer {

friend class TGClient;

private:
   Cursor_t     fCursor;             // current cursor
   Cursor_t     fDefaultCursor;      // default cursor
   TGListView   *fListView;          // associated list view
   TTreeViewer  *fViewer;            // pointer to tree viewer
public:
   TGTreeLVC(const TGWindow *p, UInt_t w, UInt_t h, UInt_t options=kSunkenFrame);
   virtual ~TGTreeLVC() {}
   virtual void AddThisItem(TGLVTreeEntry *item)
                { AddFrame(item, fItemLayout); item->SetColumns(fCpos, fJmode);}
   const char*  Cut();
   void         EmptyAll();     // empty all items of expression type
   const char*  Ex();
   const char*  Ey();
   const char*  Ez();
   void         SetListView(TGListView *lv) {fListView = lv;}
   void         SetViewer(TTreeViewer *viewer) {fViewer = viewer;}
   void         RemoveNonStatic();
   const char*  ScanList();
   void         SelectItem(const char* name);
   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t HandleMotion(Event_t *event);

   ClassDef(TGTreeLVC,0)  // a dragging-capable LVContainer
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGSelectBox                                                          //
//                                                                      //
// This class represent a specialized expression editor for             //
// TGLVTreeEntry 'true name' and 'alias' data members.                  //
// It is a singleton in order to be able to use it for several          //
// expressions.                                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGSelectBox : public TGTransientFrame {

private:
   TGLabel       *fLabel;         // label
   TGLVTreeEntry *fEntry;         // edited expression entry
   TGTextEntry   *fTe;            // text entry box
   TGLabel       *fLabelAlias;    // alias label
   TGTextEntry   *fTeAlias;       // alias text entry
   TGLayoutHints *fLayout;        // layout hints for widgets inside
   TGLayoutHints *fbLayout;       // layout for close button
   TGTextButton  *fbDone;         // close button

protected:
   static TGSelectBox *fpInstance;// pointer to this select box

public:
   TGSelectBox(const TGWindow *p, const TGWindow *main, UInt_t w = 10, UInt_t h = 10);
   virtual       ~TGSelectBox();
   virtual void  CloseWindow();
   void          GrabPointer();
   void          SetLabel(const char* title);
   void          SetEntry(TGLVTreeEntry *entry);
   void          SaveText();
   void          InsertText(const char* text);
   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);

   static TGSelectBox* GetInstance();

   ClassDef(TGSelectBox,0)  // TreeView dialog widget
};

#endif
