// Author : Andrei Gheata 02/11/00
#include "TGTreeLVC.h"

ClassImp(TGLVTreeEntry)

//______________________________________________________________________________
//*-*   TGTreeLVEntry is a TGLVEntry that has a name of a variable to be draw
//*-*   by the TTreeView GUI, and an alias for it
//______________________________________________________________________________
TGLVTreeEntry::TGLVTreeEntry(const TGWindow *p,
                             const TGPicture *bigpic, const TGPicture *smallpic,
                             TGString *name, TGString **subnames,
                             EListViewMode ViewMode)
              :TGLVEntry(p, bigpic, smallpic, name, subnames, ViewMode)
{
//*-*-*-*-*-*-*-*-*-*-*-*TGTreeLVEntry constructor*-*-*-*-*-*-*-*-*-*-*
//*-*                    =========================
// both alias and true name are initialized to name
   fAlias = name->GetString();
   fTrueName = name->GetString();
}
//______________________________________________________________________________
void TGLVTreeEntry::Copy(TGLVTreeEntry *dest)
{
//*-*-*-*-*-*-*-*-*-*-*-*Copy this item's name and alias to an other*-*-*-*-*-*-*-*-*-*-*
//*-*                    ===========================================
   if (!dest) return;
   dest->SetItemName(fName->GetString());
   dest->SetAlias(fAlias);
   dest->SetTrueName(fTrueName);
}
//______________________________________________________________________________
Bool_t TGLVTreeEntry::HasAlias()
{
// check if alias name is not empty
   if (fAlias.Length()) return kTRUE;
   return kFALSE;
}
//______________________________________________________________________________
void TGLVTreeEntry::SetItemName(const char* name)
{
// redraw this entry with new name
   if (fName) delete fName;
   fName = new TGString(name);
   Int_t max_ascent, max_descent;
   fTWidth = gVirtualX->TextWidth(fFontStruct, fName->GetString(), fName->GetLength());
   gVirtualX->GetFontProperties(fFontStruct, max_ascent, max_descent);
   fTHeight = max_ascent + max_descent;
   gVirtualX->ClearWindow(fId);
   Resize(GetDefaultSize());
   fClient->NeedRedraw(this);
}
//______________________________________________________________________________
void TGLVTreeEntry::Empty()
{
// clear all names and alias
   SetItemName("");
   SetAlias("");
   SetTrueName("");
}

ClassImp(TGTreeLVC)

//______________________________________________________________________________
//*-*   TGTreeLVC is a container having TGLVTreeEntry items that can be dragged
//
//

//______________________________________________________________________________
TGTreeLVC::TGTreeLVC(const TGWindow *p, UInt_t w, UInt_t h, UInt_t options)
          :TGLVContainer(p, w, h,options | kSunkenFrame)
{
//*-*-*-*-*-*-*-*-*-*-*-*TGLVContainer constructor*-*-*-*-*-*-*-*-*-*-*
//*-*                    =========================
   fListView = 0;
   fCursor = gVirtualX->CreateCursor(kMove);
   fDefaultCursor = gVirtualX->CreateCursor(kPointer);
}   
//______________________________________________________________________________
const char* TGTreeLVC::Cut()
{
// return the cut entry
   TGFrameElement *el = (TGFrameElement *) fList->At(3);
   if (el) {
      TGLVTreeEntry *f = (TGLVTreeEntry *) el->fFrame;
      if (f) return f->GetTrueName(); 	
      return 0;
   }	
   return 0;
}
//______________________________________________________________________________
const char* TGTreeLVC::Ex()
{
// return the expression on X
   TGFrameElement *el = (TGFrameElement *) fList->At(0);
   if (el) {
      TGLVTreeEntry *f = (TGLVTreeEntry *) el->fFrame;
      if (f) return f->GetTrueName(); 	
      return 0;
   }	
   return 0;
}
//______________________________________________________________________________
const char* TGTreeLVC::Ey()
{
// return the expression on Y
   TGFrameElement *el = (TGFrameElement *) fList->At(1);
   if (el) {
      TGLVTreeEntry *f = (TGLVTreeEntry *) el->fFrame;
      if (f) return f->GetTrueName(); 	
      return 0;
   }	
   return 0;
}
//______________________________________________________________________________
const char* TGTreeLVC::Ez()
{
// return the expression on Z
   TGFrameElement *el = (TGFrameElement *) fList->At(2);
   if (el) {
      TGLVTreeEntry *f = (TGLVTreeEntry *) el->fFrame;
      if (f) return f->GetTrueName(); 	
      return 0;
   }	
   return 0;
}
//______________________________________________________________________________
Bool_t TGTreeLVC::HandleButton(Event_t *event)
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
         TGLVTreeEntry *f = (TGLVTreeEntry *) el->fFrame;
         ++total;
         if (f->GetId() == (Window_t)event->fUser[0]) {  // fUser[0] = subwindow
            f->Activate(kTRUE);
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
         if (*itemType & kLTDragType) {
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
              TGLVTreeEntry *f = (TGLVTreeEntry *) el->fFrame;
              if ((f == fLastActive) || !f->IsActive()) continue;
                 f->Activate(kFALSE);
                 ((TGLVTreeEntry *) fLastActive)->Copy(f);
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
Bool_t TGTreeLVC::HandleMotion(Event_t *event)
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
         TGLVTreeEntry *f = (TGLVTreeEntry *) el->fFrame;
	 if (f == fLastActive) continue;
         xf0 = f->GetX();
         yf0 = f->GetY();
         xff = f->GetX() + f->GetWidth();
         yff = f->GetY() + f->GetHeight();
         itemType = (ULong_t *) f->GetUserData();	   
         if (*itemType & kLTExpressionType) {
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
void TGTreeLVC::ClearAll()
{
// Clear all names and aliases for expression type items
   TGFrameElement *el;
   TIter next(fList);
   while ((el = (TGFrameElement *) next())) {
      TGLVTreeEntry *f = (TGLVTreeEntry *) el->fFrame;
      UInt_t *userData = (UInt_t *) f->GetUserData();
      if (((*userData) & kLTExpressionType)) {
         f->Empty();
      }	   
   }
}
//______________________________________________________________________________
void TGTreeLVC::RemoveNonStatic()
{
   // remove all non-static items from the list view, except expressions
   TGFrameElement *el;
   TIter next(fList);
   while ((el = (TGFrameElement *) next())) {
      TGLVTreeEntry *f = (TGLVTreeEntry *) el->fFrame;
      UInt_t *userData = (UInt_t *) f->GetUserData();
      if (!((*userData) & kLTExpressionType)) {
         RemoveItem(f);
      }	   
   }
}
//______________________________________________________________________________
void TGTreeLVC::SelectItem(const char* name)
{
 // select an item
   if (fLastActive) {
      fLastActive->Activate(kFALSE);
      fLastActive = 0;
   }
   TGFrameElement *el;
   TIter next(fList);
   while ((el = (TGFrameElement *) next())) {
      TGLVTreeEntry *f = (TGLVTreeEntry *) el->fFrame;
      if (!strcmp(f->GetItemName()->GetString(),name)) {
         f->Activate(kTRUE);
         fLastActive = f;
      } else {
         f->Activate(kFALSE);
      }
   }
}


ClassImp(TGSelectBox)

//______________________________________________________________________________
//*-*   TGSelectBox is a transient frame with 2 text entries and it can
//*-*   edit TGLVTreeEntries
//

enum ETransientFrameCommands {
   kTFDone
};

TGSelectBox* TGSelectBox::fpInstance = 0;

//______________________________________________________________________________
//*-*-*-*-*-*-*-*-*-*-*-*TGSelectBox constructor*-*-*-*-*-*-*-*-*-*-*
//*-*                    =========================
TGSelectBox::TGSelectBox(const TGWindow *p, const TGWindow *main,
                         UInt_t w, UInt_t h)
            :TGTransientFrame(p, main, w, h)
{
   if (!fpInstance) {
      fpInstance = this;
      ULong_t color;
      if (!gClient->GetColorByName("#808080",color))
      gClient->GetColorByName("gray",color);
      fEntry = 0;
      fLayout = new TGLayoutHints(kLHintsTop | kLHintsCenterY | kLHintsExpandX, 0, 0, 0, 2);
      fbLayout = new TGLayoutHints(kLHintsBottom | kLHintsCenterY | kLHintsExpandX, 0, 0, 0, 2);

      fLabel = new TGLabel(this, "");
      AddFrame(fLabel,fLayout);

      fTe = new TGTextEntry(this, new TGTextBuffer(256));
      AddFrame(fTe, fLayout);

      fLabelAlias = new TGLabel(this, "Alias");
      AddFrame(fLabelAlias,fLayout);

      fTeAlias = new TGTextEntry(this, new TGTextBuffer(100));
      AddFrame(fTeAlias, fLayout);
   
      fbDone = new TGTextButton(this, "&Done", kTFDone);
      AddFrame(fbDone, fbLayout);   
   
      MapSubwindows();
      Resize(GetDefaultSize());
   
      SetBackgroundColor(color);
      Window_t wdum;
      Int_t ax, ay;
      gVirtualX->TranslateCoordinates(main->GetId(), GetParent()->GetId(), 0,
                                      (((TGFrame *) main)->GetHeight() - fHeight) >> 1,
                                      ax, ay, wdum);
      MoveResize(ax, ay, w, GetDefaultHeight());   
      MapWindow();
   }
}
//______________________________________________________________________________
TGSelectBox::~TGSelectBox()
{
//*-*-*-*-*-*-*-*-*-*-*-*TGSelectBox destructor*-*-*-*-*-*-*-*-*-*-*
//*-*                    ======================
   fpInstance = 0;
   delete fLabel;
   delete fTe;
   delete fLabelAlias;
   delete fTeAlias;
   delete fLayout;
   delete fbLayout;
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
   return fpInstance;
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
   fLabel->SetText(new TGString(title));
}
//______________________________________________________________________________
void TGSelectBox::SaveText()
{
// save the edited entry true name and alias
   if (fEntry) {
      fEntry->SetTrueName(fTe->GetText());
      fEntry->SetAlias(fTeAlias->GetText());
      if (strlen(fTeAlias->GetText())) {
         fEntry->SetItemName(fTeAlias->GetText());
      } else {
         fEntry->SetItemName(fTe->GetText());	
      }
   }
}
//______________________________________________________________________________
void TGSelectBox::SetEntry(TGLVTreeEntry *entry)
{
	// connect one entry
   fEntry = entry;
   fTe->SetText(entry->GetTrueName());
   fTeAlias->SetText(entry->GetAlias());
}
//______________________________________________________________________________
void TGSelectBox::InsertText(const char* text)
{
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
               SaveText();
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
                     SaveText();
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
         break;
   }
   return kTRUE;
}
