// @(#)root/treeviewer:$Id$
//Author : Andrei Gheata   16/08/00

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TTVLVContainer.h"
#include "TTreeViewer.h"
#include "TGPicture.h"
#include "TGLabel.h"
#include "TGButton.h"
#include "TGTextEntry.h"
#include "TGToolTip.h"
#include "TList.h"


ClassImp(TGItemContext)

//////////////////////////////////////////////////////////////////////////
// TGItemContext  -  empty object used as context menu support for      //
// TGLVTreeEntries                                                      //
//////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
TGItemContext::TGItemContext()
{
   // Constructor

   fItem = 0;
}
//______________________________________________________________________________
void TGItemContext::Draw(Option_t *)
{
   // Draw item

   fItem->GetContainer()->GetViewer()->ProcessMessage(MK_MSG(kC_CONTAINER, kCT_ITEMDBLCLICK), kButton1, 0);
}
//______________________________________________________________________________
void TGItemContext::EditExpression()
{
   // Edit expression

   fItem->GetContainer()->GetViewer()->EditExpression();
}
//______________________________________________________________________________
void TGItemContext::Empty()
{
   // Empty item

   fItem->Empty();
}
//______________________________________________________________________________
void TGItemContext::RemoveItem()
{
   // Remove item

   fItem->GetContainer()->GetViewer()->RemoveItem();
}
//______________________________________________________________________________
void TGItemContext::Scan()
{
   // Scan item

   fItem->GetContainer()->GetViewer()->SetScanMode();
   fItem->GetContainer()->GetViewer()->ProcessMessage(MK_MSG(kC_CONTAINER, kCT_ITEMDBLCLICK), kButton1, 0);
}
//______________________________________________________________________________
void TGItemContext::SetExpression(const char *name, const char *alias, Bool_t cut)
{
   // Set item expression

   fItem->SetExpression(name, alias, cut);
}

ClassImp(TTVLVEntry)


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGTreeLVEntry is a TGLVEntry that has a name of a variable to be     //
// draw by the TTreeViewer GUI, and an alias for it.                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
TTVLVEntry::TTVLVEntry(const TGWindow *p,
                             const TGPicture *bigpic, const TGPicture *smallpic,
                             TGString *name, TGString **subnames,
                             EListViewMode ViewMode)
              :TGLVEntry(p, bigpic, smallpic, name, subnames, ViewMode)
{
   // TGTreeLVEntry constructor.

   // both alias and true name are initialized to name
   fTrueName.InitialCapacity(1000);
   fContainer = (TTVLVContainer *) p;

   fTip = 0;
   fIsCut = kFALSE;
   fTrueName = name->GetString();
   fContext = new TGItemContext();
   fContext->Associate(this);

   AddInput(kEnterWindowMask | kLeaveWindowMask);
}
//______________________________________________________________________________
TTVLVEntry::~TTVLVEntry()
{
   // TGTreeLVEntry destructor

   if (fTip) delete fTip;
   delete fContext;
}
//______________________________________________________________________________
const char *TTVLVEntry::ConvertAliases()
{
   // Convert all aliases into true names

   TList *list = GetContainer()->GetViewer()->ExpressionList();
   fConvName = fTrueName;
   TString start(fConvName);
   TIter next(list);
   TTVLVEntry* item;
   while (!FullConverted()) {
      next.Reset();
      start = fConvName;
      while ((item=(TTVLVEntry*)next())) {
         if (item != this)
            fConvName.ReplaceAll(item->GetAlias(), item->GetTrueName());
      }
      if (fConvName == start) {
         //the following line is deadcode reported by coverity because item=0
         //if (item) Warning(item->GetAlias(), "Cannot convert aliases for this expression.");
         return(fConvName.Data());
      }
   }
   return(fConvName.Data());
}
//______________________________________________________________________________
Bool_t TTVLVEntry::FullConverted()
{
   // Return true if converted name is alias free

   TList *list = GetContainer()->GetViewer()->ExpressionList();
   TIter next(list);
   TTVLVEntry* item;
   while ((item=(TTVLVEntry*)next())) {
      if (item != this) {
         if (fConvName.Contains(item->GetAlias())) return kFALSE;
      }
   }
   return kTRUE;
}
//______________________________________________________________________________
void TTVLVEntry::CopyItem(TTVLVEntry *dest)
{
   // Copy this item's name and alias to an other.

   if (!dest) return;
   dest->SetExpression(fTrueName.Data(), fAlias.Data(), fIsCut);
   TString alias = dest->GetAlias();
   if (!alias.BeginsWith("~") && !alias.Contains("empty")) dest->PrependTilde();
}
//______________________________________________________________________________
Bool_t TTVLVEntry::HandleCrossing(Event_t *event)
{
   // Handle mouse crossing event.

   if (fTip) {
      if (event->fType == kEnterNotify)
         fTip->Reset();
      else
         fTip->Hide();
   }
   return kTRUE;
}
//______________________________________________________________________________
Bool_t TTVLVEntry::HasAlias()
{
   // Check if alias name is not empty.

   if (fAlias.Length()) return kTRUE;
   return kFALSE;
}
//______________________________________________________________________________
void TTVLVEntry::PrependTilde()
{
   // Prepend a ~ to item alias

   fAlias = "~" + fAlias;
   SetItemName(fAlias.Data());
}
//______________________________________________________________________________
void TTVLVEntry::SetItemName(const char* name)
{
   // redraw this entry with new name

   if (fItemName) delete fItemName;
   fItemName = new TGString(name);
   Int_t max_ascent, max_descent;
   fTWidth = gVirtualX->TextWidth(fFontStruct, fItemName->GetString(), fItemName->GetLength());
   gVirtualX->GetFontProperties(fFontStruct, max_ascent, max_descent);
   fTHeight = max_ascent + max_descent;
   gVirtualX->ClearWindow(fId);
   Resize(GetDefaultSize());
   fClient->NeedRedraw(this);
}
//______________________________________________________________________________
void TTVLVEntry::SetCutType(Bool_t type)
{
   // Set cut type

   if (fIsCut && type) return;
   if (!fIsCut && !type) return;
   if (type) {
      SetSmallPic(fClient->GetPicture("selection_t.xpm"));
      SetToolTipText("Selection expression. Drag to scissors to activate");
   } else
      SetSmallPic(fClient->GetPicture("expression_t.xpm"));
   fIsCut = type;
}
//______________________________________________________________________________
void TTVLVEntry::SetExpression(const char* name, const char* alias, Bool_t cutType)
{
   // Set the true name, alias and type of the expression, then refresh it

   SetItemName(alias);
   SetAlias(alias);
   SetTrueName(name);
   ULong_t *itemType = (ULong_t *) GetUserData();
   if (*itemType & TTreeViewer::kLTPackType) {
      if (strlen(name))
         SetSmallPic(fClient->GetPicture("pack_t.xpm"));
      else
         SetSmallPic(fClient->GetPicture("pack-empty_t.xpm"));
   }
   if ((*itemType & TTreeViewer::kLTDragType) && strlen(name) && !fIsCut)
      SetToolTipText("Double-click to draw. Drag and drop. Use Edit/Expression or context menu to edit.");
   if (*itemType & TTreeViewer::kLTDragType) SetCutType(cutType);
}
//______________________________________________________________________________
void TTVLVEntry::Empty()
{
   // clear all names and alias

   SetExpression("","-empty-");
   ULong_t *itemType = (ULong_t *) GetUserData();
   if (itemType && (*itemType & TTreeViewer::kLTDragType))
      SetToolTipText("User-defined expression/cut. Double-click to edit");
}
//______________________________________________________________________________
void TTVLVEntry::SetToolTipText(const char *text, Long_t delayms)
{
   // Set tool tip text associated with this item. The delay is in
   // milliseconds (minimum 250). To remove tool tip call method with text = 0

   if (fTip) {
      delete fTip;
      fTip = 0;
   }

   if (text && strlen(text))
      fTip = new TGToolTip(fClient->GetRoot(), this, text, delayms);
}
//______________________________________________________________________________
void TTVLVEntry::SetSmallPic(const TGPicture *spic)
{
   // Set small picture

   const TGPicture *cspic = fSmallPic;
   fSmallPic = spic;
   fCurrent = fSmallPic;
   if (fSelPic) delete fSelPic;
   fSelPic = 0;
   if (fActive) {
      fSelPic = new TGSelectedPicture(fClient, fCurrent);
   }
   DoRedraw();
   fClient->FreePicture(cspic);
}

ClassImp(TTVLVContainer)

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//   TTVLVContainer                                                     //
//                                                                      //
// This class represent the list view container for the.                //
// TreeView class. It is a TGLVContainer with item dragging             //
// capabilities for the TTVLVEntry objects inside.                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
TTVLVContainer::TTVLVContainer(const TGWindow *p, UInt_t w, UInt_t h, UInt_t options)
          :TGLVContainer(p, w, h,options | kSunkenFrame)
{
   // TGLVContainer constructor

   fListView = 0;
   fViewer = 0;
   fExpressionList = new TList;
   fCursor = gVirtualX->CreateCursor(kMove);
   fDefaultCursor = gVirtualX->CreateCursor(kPointer);
   fMapSubwindows = kTRUE;
}

//______________________________________________________________________________
TTVLVContainer::~TTVLVContainer()
{
   // TGLVContainer destructor

   delete fExpressionList;
}

//______________________________________________________________________________
const char* TTVLVContainer::Cut()
{
   // return the cut entry

   TGFrameElement *el = (TGFrameElement *) fList->At(3);
   if (el) {
      TTVLVEntry *f = (TTVLVEntry *) el->fFrame;
      if (f) return f->ConvertAliases();
      return 0;
   }
   return 0;
}

//______________________________________________________________________________
TTVLVEntry * TTVLVContainer::ExpressionItem(Int_t index)
{
   // return the expression item at specific position

   TGFrameElement *el = (TGFrameElement *) fList->At(index);
   if (el) {
      TTVLVEntry *item = (TTVLVEntry *) el->fFrame;
      return item;
   }
   return 0;
}

//______________________________________________________________________________
TList* TTVLVContainer::ExpressionList()
{
   // return the list of user-defined expressions

   fExpressionList->Clear();
   TIter next(fList);
   TGFrameElement *el;
   while ((el = (TGFrameElement*)next())) {
      TTVLVEntry *item = (TTVLVEntry *)el->fFrame;
      if (item) {
         ULong_t *itemType = (ULong_t *) item->GetUserData();
         if ((*itemType & TTreeViewer::kLTExpressionType) &&
            (*itemType & TTreeViewer::kLTDragType)) fExpressionList->Add(item);
      }
   }
   return fExpressionList;
}

//______________________________________________________________________________
const char* TTVLVContainer::Ex()
{
   // return the expression on X

   TGFrameElement *el = (TGFrameElement *) fList->At(0);
   if (el) {
      TTVLVEntry *f = (TTVLVEntry *) el->fFrame;
      if (f) return f->ConvertAliases();
      return 0;
   }
   return 0;
}

//______________________________________________________________________________
const char* TTVLVContainer::Ey()
{
   // return the expression on Y

   TGFrameElement *el = (TGFrameElement *) fList->At(1);
   if (el) {
      TTVLVEntry *f = (TTVLVEntry *) el->fFrame;
      if (f) return f->ConvertAliases();
      return 0;
   }
   return 0;
}

//______________________________________________________________________________
const char* TTVLVContainer::Ez()
{
   // return the expression on Z

   TGFrameElement *el = (TGFrameElement *) fList->At(2);
   if (el) {
      TTVLVEntry *f = (TTVLVEntry *) el->fFrame;
      if (f) return f->ConvertAliases();
      return 0;
   }
   return 0;
}

//______________________________________________________________________________
const char* TTVLVContainer::ScanList()
{
   // return the cut entry

   TGFrameElement *el = (TGFrameElement *) fList->At(4);
   if (el) {
      TTVLVEntry *f = (TTVLVEntry *) el->fFrame;
      if (f) return f->GetTrueName();
      return 0;
   }
   return 0;
}

//______________________________________________________________________________
Bool_t TTVLVContainer::HandleButton(Event_t *event)
{
   // Handle mouse button event in container.

   int total, selected;

   if (event->fType == kButtonPress) {
      fXp = event->fX;
      fYp = event->fY;
      if (fLastActive) {
         fLastActive->Activate(kFALSE);
         fLastActive = 0;
      }
      total = selected = 0;

      TGFrameElement *el;
      TIter next(fList);
      while ((el = (TGFrameElement *) next())) {
         TTVLVEntry *f = (TTVLVEntry *) el->fFrame;
         ++total;
         if (f->GetId() == (Window_t)event->fUser[0]) {  // fUser[0] = subwindow
            f->Activate(kTRUE);
            if (f->GetTip()) (f->GetTip())->Hide();
            fX0 = f->GetX();
            fY0 = f->GetY();
            ++selected;
            fLastActive = f;
         } else {
            f->Activate(kFALSE);
         }
      }

      if (fTotal != total || fSelected != selected) {
         fTotal = total;
         fSelected = selected;
         SendMessage(fMsgWindow, MK_MSG(kC_CONTAINER, kCT_SELCHANGED),
                     fTotal, fSelected);
      }

      if (selected == 1 && event->fCode == 1) {
         ULong_t *itemType = (ULong_t *) fLastActive->GetUserData();
         if (*itemType & TTreeViewer::kLTDragType) {
            fDragging = kTRUE;
            gVirtualX->SetCursor(fId,fCursor);
            fXp = event->fX;
            fYp = event->fY;
         }
      }
   }

   if (event->fType == kButtonRelease) {
      if (fDragging) {
         fDragging = kFALSE;
         gVirtualX->SetCursor(fId,fDefaultCursor);
         fLastActive->Move(fX0,fY0);
         TGFrameElement *el;
         TIter next(fList);
         while ((el = (TGFrameElement *) next())) {
            TTVLVEntry *f = (TTVLVEntry *) el->fFrame;
            if ((f == fLastActive) || !f->IsActive()) continue;
            ULong_t *itemType = (ULong_t *) f->GetUserData();
            fLastActive->Activate(kFALSE);
            if (!(*itemType & TTreeViewer::kLTPackType)) {
               // dragging items to expressions
               ((TTVLVEntry *) fLastActive)->CopyItem(f);
               if (*itemType & TTreeViewer::kLTDragType)
                  f->SetToolTipText("Double-click to draw. Drag and drop. Use Edit/Expression or context menu to edit.");
            } else {
               if (strlen(((TTVLVEntry *) fLastActive)->GetTrueName())) {
                  // dragging to scan box
                  if (!strlen(f->GetTrueName())) {
                     f->SetTrueName(((TTVLVEntry *)fLastActive)->GetTrueName());
                     f->SetSmallPic(fClient->GetPicture("pack_t.xpm"));
                  } else {
                     TString name(2000);
                     TString dragged = ((TTVLVEntry *)fLastActive)->ConvertAliases();
                     name  = f->GetTrueName();
                     if ((name.Length()+dragged.Length()) < 228) {
                        name += ":";
                        name += dragged;
                        f->SetTrueName(name.Data());
                     } else {
                        Warning("HandleButton",
                                "Name too long. Can not add any more items to scan box.");
                     }
                  }
               }
            }
            fLastActive = f;
            if (fViewer) {
               char msg[2000];
               msg[0] = 0;
               snprintf(msg,2000, "Content : %s", f->GetTrueName());
               fViewer->Message(msg);
            }
         }
         if ((TMath::Abs(event->fX - fXp) < 2) && (TMath::Abs(event->fY - fYp) < 2)) {
            SendMessage(fMsgWindow, MK_MSG(kC_CONTAINER, kCT_ITEMCLICK),
                        event->fCode, (event->fYRoot << 16) | event->fXRoot);
         }
      } else {
         SendMessage(fMsgWindow, MK_MSG(kC_CONTAINER, kCT_ITEMCLICK),
                     event->fCode, (event->fYRoot << 16) | event->fXRoot);
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TTVLVContainer::HandleMotion(Event_t *event)
{
   // Handle mouse motion events.

   Int_t xf0, xff, yf0, yff;
   Int_t xpos = event->fX - (fXp-fX0);
   Int_t ypos = event->fY - (fYp-fY0);

   if (fDragging) {
      TGFrameElement *el;
      ULong_t *itemType;
      TIter next(fList);
      while ((el = (TGFrameElement *) next())) {
         TTVLVEntry *f = (TTVLVEntry *) el->fFrame;
         if (f == fLastActive) {
            if (f->GetTip()) (f->GetTip())->Hide();
            continue;
         }
         xf0 = f->GetX();
         yf0 = f->GetY();
         xff = f->GetX() + f->GetWidth();
         yff = f->GetY() + f->GetHeight();
         itemType = (ULong_t *) f->GetUserData();
         if (*itemType & TTreeViewer::kLTExpressionType) {
            if (xpos>xf0 && xpos<xff && ypos>yf0 && ypos<yff) {
               f->Activate(kTRUE);
            } else {
               f->Activate(kFALSE);
            }
         }
      }
      if ((fXp - event->fX) > 10) {
         fListView->SetHsbPosition(0);
         fListView->SetVsbPosition(0);
      }
      fLastActive->Move(xpos, ypos);
      gVirtualX->RaiseWindow(fLastActive->GetId());
      SendMessage(fMsgWindow, MK_MSG(kC_CONTAINER,(EWidgetMessageTypes)4),event->fX, event->fY);
   }
   return kTRUE;
}

//______________________________________________________________________________
void TTVLVContainer::EmptyAll()
{
   // Clear all names and aliases for expression type items

   TGFrameElement *el;
   TIter next(fList);
   while ((el = (TGFrameElement *) next())) {
      TTVLVEntry *f = (TTVLVEntry *) el->fFrame;
      UInt_t *userData = (UInt_t *) f->GetUserData();
      if (*userData & TTreeViewer::kLTExpressionType) {
         if (*userData & TTreeViewer::kLTPackType) {
            f->SetSmallPic(fClient->GetPicture("pack-empty_t.xpm"));
            f->SetTrueName("");
         } else {
            f->Empty();
         }
      }
   }
}

//______________________________________________________________________________
void TTVLVContainer::RemoveNonStatic()
{
   // remove all non-static items from the list view, except expressions

   TGFrameElement *el;
   TIter next(fList);
   while ((el = (TGFrameElement *) next())) {
      TTVLVEntry *f = (TTVLVEntry *) el->fFrame;
      UInt_t *userData = (UInt_t *) f->GetUserData();
      if (!((*userData) & TTreeViewer::kLTExpressionType)) {
         RemoveItem(f);
      }
   }
   fLastActive = 0;
}

//______________________________________________________________________________
void TTVLVContainer::SelectItem(const char* name)
{
   // select an item

   if (fLastActive) {
      fLastActive->Activate(kFALSE);
      fLastActive = 0;
   }
   TGFrameElement *el;
   fSelected = 0;
   TIter next(fList);
   while ((el = (TGFrameElement *) next())) {
      TTVLVEntry *f = (TTVLVEntry *) el->fFrame;
      if (!strcmp(f->GetItemName()->GetString(),name)) {
         f->Activate(kTRUE);
         fLastActive = (TGLVEntry *) f;
         fSelected++;
      } else {
         f->Activate(kFALSE);
      }
   }
}

ClassImp(TGSelectBox)

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//   TGSelectBox                                                        //
//                                                                      //
// This class represent a specialized expression editor for             //
// TTVLVEntry 'true name' and 'alias' data members.                     //
// It is a singleton in order to be able to use it for several          //
// expressions.                                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

enum ETransientFrameCommands {
   kTFDone,
   kTFCancel
};

TGSelectBox* TGSelectBox::fgInstance = 0;

//______________________________________________________________________________
TGSelectBox::TGSelectBox(const TGWindow *p, const TGWindow *main,
                         UInt_t w, UInt_t h)
            :TGTransientFrame(p, main, w, h)
{
   // TGSelectBox constructor

   if (!fgInstance) {
      fgInstance = this;
      fViewer = (TTreeViewer *)fMain;
      if (!fViewer) Error("TGSelectBox", "Must be started from viewer");
      fEntry = 0;
      fLayout = new TGLayoutHints(kLHintsTop | kLHintsCenterY | kLHintsExpandX, 0, 0, 0, 2);
      fBLayout = new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 2, 2, 2);
      fBLayout1= new TGLayoutHints(kLHintsTop | kLHintsRight, 2, 0, 2, 2);

      fLabel = new TGLabel(this, "");
      AddFrame(fLabel,fLayout);

      fTe = new TGTextEntry(this, new TGTextBuffer(2000));
      fTe->SetToolTipText("Type an expression using C++ syntax. Click other expression/leaves to paste them here.");
      AddFrame(fTe, fLayout);

      fLabelAlias = new TGLabel(this, "Alias");
      AddFrame(fLabelAlias,fLayout);

      fTeAlias = new TGTextEntry(this, new TGTextBuffer(100));
      fTeAlias->SetToolTipText("Define an alias for this expression. Do NOT use leading strings of other aliases.");
      AddFrame(fTeAlias, fLayout);

      fBf = new TGHorizontalFrame(this, 10, 10);

      fCANCEL = new TGTextButton(fBf, "&Cancel", kTFCancel);
      fCANCEL->Associate(this);
      fBf->AddFrame(fCANCEL, fBLayout);

      fDONE = new TGTextButton(fBf, "&Done", kTFDone);
      fDONE->Associate(this);
      fBf->AddFrame(fDONE, fBLayout1);

      AddFrame(fBf, fLayout);

      MapSubwindows();
      Resize(GetDefaultSize());

//      SetBackgroundColor(color);
      Window_t wdum;
      Int_t ax, ay;
      gVirtualX->TranslateCoordinates(main->GetId(), GetParent()->GetId(), 25,
                        (Int_t)(((TGFrame *) main)->GetHeight() - fHeight) >> 1,
                        ax, ay, wdum);
      MoveResize(ax, ay, w, GetDefaultHeight());
      MapWindow();
   }
}

//______________________________________________________________________________
TGSelectBox::~TGSelectBox()
{
   // TGSelectBox destructor

   fgInstance = 0;
   delete fLabel;
   delete fTe;
   delete fLabelAlias;
   delete fTeAlias;
   delete fDONE;
   delete fCANCEL;
   delete fBf;
   delete fLayout;
   delete fBLayout;
   delete fBLayout1;
}

//______________________________________________________________________________
void TGSelectBox::CloseWindow()
{
   // close the select box

   gVirtualX->UnmapWindow(GetId());
   delete this;
}

//______________________________________________________________________________
TGSelectBox * TGSelectBox::GetInstance()
{
   // return the pointer to the instantiated singleton

   return fgInstance;
}

//______________________________________________________________________________
void TGSelectBox::GrabPointer()
{
   // just focus the cursor inside

   Event_t event;
   event.fType = kButtonPress;
   event.fCode = kButton1;
   Int_t position = fTe->GetCursorPosition();
   fTe->HandleButton(&event);
   fTe->SetCursorPosition(position);
}

//______________________________________________________________________________
void TGSelectBox::SetLabel(const char* title)
{
   // Set label of selection box

   fLabel->SetText(new TGString(title));
}

//______________________________________________________________________________
void TGSelectBox::SaveText()
{
   // save the edited entry true name and alias

   if (fEntry) {

      Bool_t cutType;
      TString name(fTe->GetText());
      if (name.Length())
         fEntry->SetToolTipText("Double-click to draw. Drag and drop. Use Edit/Expression or context menu to edit.");
      else
         fEntry->SetToolTipText("User-defined expression/cut. Double-click to edit");
      // Set type of item to "cut" if containing boolean operators
      cutType = name.Contains("<") || name.Contains(">") || name.Contains("=") ||
                name.Contains("!") || name.Contains("&") || name.Contains("|");
      TString alias(fTeAlias->GetText());
      if (!alias.BeginsWith("~") && !alias.Contains("empty")) fTeAlias->InsertText("~", 0);
      fEntry->SetExpression(fTe->GetText(), fTeAlias->GetText(), cutType);

      if (fOldAlias.Contains("empty")) {
         fOldAlias = fTeAlias->GetText();
         return;
      }
      TList *list = fViewer->ExpressionList();
      TIter next(list);
      TTVLVEntry* item;
      while ((item=(TTVLVEntry*)next())) {
         if (item != fEntry) {
            name = item->GetTrueName();
            name.ReplaceAll(fOldAlias.Data(), fTeAlias->GetText());
            item->SetTrueName(name.Data());
         }
      }
   }
}

//______________________________________________________________________________
void TGSelectBox::SetEntry(TTVLVEntry *entry)
{
   // connect one entry

   fEntry = entry;
   fTe->SetText(entry->GetTrueName());
   fTeAlias->SetText(entry->GetAlias());
   fOldAlias = entry->GetAlias();
}

//______________________________________________________________________________
void TGSelectBox::InsertText(const char* text)
{
   // Insert text in text entry

   Int_t start = fTe->GetCursorPosition();
   fTe->InsertText(text, fTe->GetCursorPosition());
   fTe->SetCursorPosition(start+strlen(text));
}

//______________________________________________________________________________
Bool_t TGSelectBox::ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2)
{
   // Message interpreter

   switch (GET_MSG(msg)) {
      case kC_TEXTENTRY:
         switch (GET_SUBMSG(msg)) {
            case kTE_ENTER:
               if (ValidateAlias()) SaveText();
               break;
            default:
               break;
         }
         break;
      case kC_COMMAND:
         switch (GET_SUBMSG(msg)) {
            case kCM_BUTTON:
               switch (parm1) {
                  case kTFDone:
                     if (!ValidateAlias()) break;
                     SaveText();
                     CloseWindow();
                     break;
                  case kTFCancel:
                     CloseWindow();
                     break;
                  default:
                     break;
               }
               break;
            default:
               break;
         }
         break;
      default:
         if (parm2) break;       // just to avoid warning on CC compiler
         break;
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGSelectBox::ValidateAlias()
{
   // return true if edited alias is not a leading string of other expression aliases

   if (!strcmp(fTeAlias->GetText(), "-empty-") || !strlen(fTeAlias->GetText())) {
      fViewer->Warning("ValidateAlias", "You should define the alias first.");
      return kFALSE;
   }
   TList *list = fViewer->ExpressionList();
   TIter next(list);
   TTVLVEntry* item;
   while ((item=(TTVLVEntry*)next())) {
      if (item != fEntry) {
         TString itemalias(item->GetAlias());
         if (itemalias.Contains(fTeAlias->GetText())) {
            fViewer->Warning("ValidAlias", "Alias can not be the leading string of other alias.");
            return kFALSE;
         }
      }
   }
   return kTRUE;
}
