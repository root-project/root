// @(#)root/gui:$Id$
// Author: Fons Rademakers   12/02/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRootContextMenu                                                     //
//                                                                      //
// This class provides an interface to context sensitive popup menus.   //
// These menus pop up when the user hits the right mouse button, and    //
// are destroyed when the menu pops downs.                              //
// The picture below shows a canvas with a pop-up menu.                 //
//                                                                      //
//Begin_Html <img src="gif/hsumMenu.gif"> End_Html                      //
//                                                                      //
// The picture below shows a canvas with a pop-up menu and a dialog box.//
//                                                                      //
//Begin_Html <img src="gif/hsumDialog.gif"> End_Html                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TRootContextMenu.h"
#include "TROOT.h"
#include "TGClient.h"
#include "TEnv.h"
#include "TList.h"
#include "TContextMenu.h"
#include "TMethod.h"
#include "TMethodArg.h"
#include "TClass.h"
#include "TVirtualX.h"
#include "TCanvas.h"
#include "TDataMember.h"
#include "TToggle.h"
#include "TRootDialog.h"
#include "TDataType.h"
#include "TCanvas.h"
#include "TBrowser.h"
#include "TRootCanvas.h"
#include "TRootBrowser.h"
#include "TClassMenuItem.h"
#include "TObjectSpy.h"
#include "KeySymbols.h"

enum EContextMenu {
   kToggleStart       = 1000, // first id of toggle menu items
   kToggleListStart   = 2000, // first id of toggle list menu items
   kUserFunctionStart = 3000  // first id of user added functions/methods, etc...
};


ClassImp(TRootContextMenu)

//______________________________________________________________________________
TRootContextMenu::TRootContextMenu(TContextMenu *c, const char *)
    : TGPopupMenu(gClient->GetDefaultRoot()), TContextMenuImp(c)
{
   // Create context menu.

   fDialog  = 0;
   fTrash = new TList;

   AddInput(kButtonPressMask | kButtonReleaseMask);
   gROOT->GetListOfCleanups()->Add(this);
   // Context menu handles its own messages
   Associate(this);
}

//______________________________________________________________________________
TRootContextMenu::~TRootContextMenu()
{
   // Delete a context menu.

   gROOT->GetListOfCleanups()->Remove(this);
   delete fDialog;
   if (fTrash) fTrash->Delete();
   delete fTrash;
}

//______________________________________________________________________________
void TRootContextMenu::DisplayPopup(Int_t x, Int_t y)
{
   // Display context popup menu for currently selected object.

   if (fClient->IsEditable()) return;

   // delete menu items releated to previous object and reset menu size
   if (fEntryList) fEntryList->Delete();
   fCurrent = 0;
   if (fTrash)   fTrash->Delete();
   fMenuHeight = 6;
   fMenuWidth  = 8;

   // delete previous dialog
   if (fDialog) {
      delete fDialog;
      fDialog = 0;
   }

   // add menu items to popup menu
   CreateMenu(fContextMenu->GetSelectedObject());

   int    xx, yy, topx = 0, topy = 0;
   UInt_t w, h;

   if (fContextMenu->GetSelectedCanvas())
      gVirtualX->GetGeometry(fContextMenu->GetSelectedCanvas()->GetCanvasID(),
                        topx, topy, w, h);

   xx = topx + x + 1;
   yy = topy + y + 1;

   PlaceMenu(xx, yy, kTRUE, kTRUE);
   // add some space for the right-side '?' (help)
   fMenuWidth += 5;
   Resize(GetDefaultWidth()+5, GetDefaultHeight());
}

//______________________________________________________________________________
TGPopupMenu * TRootContextMenu::FindHierarchy(const char *commentstring, TString & last_component)
{
  // Decodes the Hierarchy="Level0/Level1/Level2/..." statement from the comment field
  // and returns the - if needed - created sub menu "Level0/Level1"
  // Returns the last component in last_component.

   TString cmd(commentstring);
   TString option;
   TString hierarchy;
   TGPopupMenu *currentMenu = 0;

   // search for arguments to the MENU statement
   // strcpy(cmd,commentstring);
   Ssiz_t opt_ptr;
   if ((opt_ptr=cmd.Index("*MENU={"))   != kNPOS ||
       (opt_ptr=cmd.Index("*SUBMENU={"))!= kNPOS ||
       (opt_ptr=cmd.Index("*TOGGLE={")) != kNPOS ) {

      Ssiz_t start = cmd.Index("{",opt_ptr) + 1;
      Ssiz_t end = cmd.Index("}",start);
      option = cmd(start,end - start);

      // Look for Hierarchy token
      TObjArray * array = option.Tokenize(";");
      TIter iter(array);
      TObject *obj;
      while((obj = iter())) {
         TString token(obj->GetName());
         if (token.Index("Hierarchy=\"") != kNPOS) {
            Ssiz_t tstart = token.Index("\"") + 1;
            Ssiz_t tend = token.Index("\"",tstart+1);
            if (tend == kNPOS) continue;
            hierarchy = token(tstart,tend - tstart);
         }
      }
      delete array;
   }

   // Build Hierarchy
   currentMenu = this;
   TObjArray * array = hierarchy.Tokenize("/");
   TIter iter(array);
   TObject *obj = iter();
   while(obj) {
      last_component = obj->GetName();
      obj = iter();
      if (obj) {
         TGMenuEntry *ptr;
         TIter next(currentMenu->GetListOfEntries());
         // Search for popup with corresponding name
         while ((ptr = (TGMenuEntry *) next()) &&
                (ptr->GetType() != kMenuPopup ||
                last_component.CompareTo(ptr->GetName()))) { }
         if (ptr)
            currentMenu = ptr->GetPopup();
         else {
            TGPopupMenu *r = new TGPopupMenu(gClient->GetDefaultRoot());
            // Alphabetical ordering
            TGMenuEntry *ptr2;
            TIter next2(currentMenu->GetListOfEntries());
            // Search for popup with corresponding name
            while ((ptr2 = (TGMenuEntry *) next2()) &&
                   (ptr2->GetType() != kMenuPopup  ||
                   last_component.CompareTo(ptr2->GetName()) > 0 )) { }

            currentMenu->AddPopup(last_component, r,ptr2);
            currentMenu = r;
            fTrash->Add(r);
            last_component = obj->GetName();
         }
      }
   }

   delete array;
   return currentMenu;
}

//______________________________________________________________________________
void TRootContextMenu::AddEntrySorted(TGPopupMenu *currentMenu, const char *s, Int_t id, void *ud,
                                         const TGPicture *p , Bool_t sorted)
{
   // Add a entry to current menu with alphabetical ordering.

   TGMenuEntry *ptr2 = 0;
   if (sorted) {
      TIter next(currentMenu->GetListOfEntries());
      // Search for popup with corresponding name
      while ((ptr2 = (TGMenuEntry *) next()) &&
             (ptr2->GetType() != kMenuEntry ||
             strcmp(ptr2->GetName(), s)<0 )) { }
   }
   currentMenu->AddEntry(s,id,ud,p,ptr2);
}

//______________________________________________________________________________
void TRootContextMenu::CreateMenu(TObject *object)
{
   // Create the context menu depending on the selected object.

   if (!object || fClient->IsEditable()) return;

   int entry = 0, toggle = kToggleStart, togglelist = kToggleListStart;
   int userfunction = kUserFunctionStart;

   // Add a title
   AddLabel(fContextMenu->CreatePopupTitle(object));
   AddSeparator();

   // Get list of menu items from the selected object's class
   TList *menuItemList = object->IsA()->GetMenuList();

   TClassMenuItem *menuItem;
   TIter nextItem(menuItemList);

   while ((menuItem = (TClassMenuItem*) nextItem())) {
      switch (menuItem->GetType()) {
         case TClassMenuItem::kPopupSeparator:
            {
            TGMenuEntry *last = (TGMenuEntry *)GetListOfEntries()->Last();
            if (last && last->GetType() != kMenuSeparator)
               AddSeparator();
            break;
            }
         case TClassMenuItem::kPopupStandardList:
            {
               // Standard list of class methods. Rebuild from scratch.
               // Get linked list of objects menu items (i.e. member functions
               // with the token *MENU in their comment fields.
               TList *methodList = new TList;
               object->IsA()->GetMenuItems(methodList);

               TMethod *method;
               TClass  *classPtr = 0;
               TIter next(methodList);
               Bool_t needSep = kFALSE;

               while ((method = (TMethod*) next())) {
                  if (classPtr != method->GetClass()) {
                     needSep = kTRUE;
                     classPtr = method->GetClass();
                  }

                  TDataMember *m;
                  EMenuItemKind menuKind = method->IsMenuItem();
                  TGPopupMenu *currentMenu = 0;
                  TString last_component;

                  switch (menuKind) {
                     case kMenuDialog:
                        // search for arguments to the MENU statement
                        currentMenu = FindHierarchy(method->GetCommentString(),last_component);
                        if (needSep && currentMenu == this) {
                           AddSeparator();
                           needSep = kFALSE;
                        }
                        AddEntrySorted(currentMenu,last_component.Length() ? last_component.Data() : method->GetName(), entry++, method,0,currentMenu != this);
                        break;
                     case kMenuSubMenu:
                        if ((m = method->FindDataMember())) {

                           // search for arguments to the MENU statement
                           currentMenu = FindHierarchy(method->GetCommentString(),last_component);

                           if (m->GetterMethod()) {
                              TGPopupMenu *r = new TGPopupMenu(gClient->GetDefaultRoot());
                              if (needSep && currentMenu == this) {
                                 AddSeparator();
                                 needSep = kFALSE;
                              }
                              if (last_component.Length()) {
                                 currentMenu->AddPopup(last_component, r);
                              } else {
                                 currentMenu->AddPopup(method->GetName(), r);
                              }
                              fTrash->Add(r);
                              TIter nxt(m->GetOptions());
                              TOptionListItem *it;
                              while ((it = (TOptionListItem*) nxt())) {
                                 char *name = it->fOptName;
                                 Long_t val = it->fValue;

                                 TToggle *t = new TToggle;
                                 t->SetToggledObject(object, method);
                                 t->SetOnValue(val);
                                 fTrash->Add(t);

                                 r->AddEntry(name, togglelist++, t);
                                 if (t->GetState())
                                    r->CheckEntry(togglelist-1);
                              }
                           } else {
                              if (needSep && currentMenu == this) {
                                 AddSeparator();
                                 needSep = kFALSE;
                              }
                              AddEntrySorted(currentMenu,last_component.Length() ? last_component.Data() : method->GetName(), entry++, method,0,currentMenu != this);
                           }
                        }
                        break;

                     case kMenuToggle:
                        {
                           TToggle *t = new TToggle;
                           t->SetToggledObject(object, method);
                           t->SetOnValue(1);
                           fTrash->Add(t);
                           // search for arguments to the MENU statement
                           currentMenu = FindHierarchy(method->GetCommentString(),last_component);
                           if (needSep && currentMenu == this) {
                              AddSeparator();
                              needSep = kFALSE;
                           }
                           AddEntrySorted(currentMenu,last_component.Length() ? last_component.Data() : method->GetName(), toggle++, t,0,currentMenu != this);
                           if (t->GetState()) currentMenu->CheckEntry(toggle-1);
                        }
                        break;

                     default:
                        break;
                  }
               }
               delete methodList;
            }
            break;
         case TClassMenuItem::kPopupUserFunction:
            {
               if (menuItem->IsToggle()) {
                  TMethod* method =
                        object->IsA()->GetMethodWithPrototype(menuItem->GetFunctionName(),menuItem->GetArgs());
                  if (method) {
                     TToggle *t = new TToggle;
                     t->SetToggledObject(object, method);
                     t->SetOnValue(1);
                     fTrash->Add(t);

                     AddEntry(method->GetName(), toggle++, t);
                     if (t->GetState()) CheckEntry(toggle-1);
                  }
               } else {
                  const char* menuItemTitle = menuItem->GetTitle();
                  if (strlen(menuItemTitle)==0) menuItemTitle = menuItem->GetFunctionName();
                  AddEntry(menuItemTitle,userfunction++,menuItem);
               }
            }
            break;
         default:
            break;
      }
   }
}

//______________________________________________________________________________
void TRootContextMenu::Dialog(TObject *object, TMethod *method)
{
   // Create dialog object with OK and Cancel buttons. This dialog
   // prompts for the arguments of "method".

   Dialog(object,(TFunction*)method);
}

//______________________________________________________________________________
void TRootContextMenu::Dialog(TObject *object, TFunction *function)
{
   // Create dialog object with OK and Cancel buttons. This dialog
   // prompts for the arguments of "function".
   // function may be a global function or a method

   Int_t selfobjpos;

   if (!function) return;

   // Position, if it exists, of the argument that correspond to the object itself
   if (fContextMenu->GetSelectedMenuItem())
      selfobjpos =  fContextMenu->GetSelectedMenuItem()->GetSelfObjectPos();
   else selfobjpos = -1;

   const TGWindow *w;
   if (fContextMenu->GetSelectedCanvas()) {
      TCanvas *c = (TCanvas *) fContextMenu->GetSelectedCanvas();
      // Embedded canvas has no canvasimp that is a TGFrame
      if (c->GetCanvasImp()->IsA()->InheritsFrom(TGFrame::Class())) {
         w = fClient->GetWindowById(gVirtualX->GetWindowID(c->GetCanvasID()));
         if (!w) w = (TRootCanvas *) c->GetCanvasImp();
      } else {
         w = gClient->GetDefaultRoot();
      }
   } else if (fContextMenu->GetBrowser()) {
      TBrowser *b = (TBrowser *) fContextMenu->GetBrowser();
      w = (TRootBrowser *) b->GetBrowserImp();
   } else {
      w = gClient->GetDefaultRoot();
   }
   fDialog = new TRootDialog(this, w, fContextMenu->CreateDialogTitle(object, function));

   // iterate through all arguments and create apropriate input-data objects:
   // inputlines, option menus...
   TMethodArg *argument = 0;

   TIter next(function->GetListOfMethodArgs());
   Int_t argpos = 0;

   while ((argument = (TMethodArg *) next())) {
      // Do not input argument for self object
      if (selfobjpos != argpos) {
         const char *argname    = fContextMenu->CreateArgumentTitle(argument);
         const char *type       = argument->GetTypeName();
         TDataType  *datatype   = gROOT->GetType(type);
         const char *charstar   = "char*";
         char        basictype[32];

         if (datatype) {
            strlcpy(basictype, datatype->GetTypeName(), 32);
         } else {
            TClass *cl = TClass::GetClass(type);
            if (strncmp(type, "enum", 4) && (cl && !(cl->Property() & kIsEnum)))
               Warning("Dialog", "data type is not basic type, assuming (int)");
            strlcpy(basictype, "int", 32);
         }

         if (strchr(argname, '*')) {
            strlcat(basictype, "*",32);
            if (!strncmp(type, "char", 4))
               type = charstar;
            else if (strstr(argname, "[default:")) {
               // skip arguments that are pointers (but not char *)
               // and have a default value
               argpos++;
               continue;
            }
         }

         TDataMember *m = argument->GetDataMember();
         if (m && object && m->GetterMethod(object->IsA())) {

            // Get the current value and form it as a text:

            char val[256];

            if (!strncmp(basictype, "char*", 5)) {
               char *tdefval;
               m->GetterMethod()->Execute(object, "", &tdefval);
               strlcpy(val, tdefval, sizeof(val));
            } else if (!strncmp(basictype, "float", 5) ||
                       !strncmp(basictype, "double", 6)) {
               Double_t ddefval;
               m->GetterMethod()->Execute(object, "", ddefval);
               snprintf(val,256, "%g", ddefval);
            } else if (!strncmp(basictype, "char", 4) ||
                       !strncmp(basictype, "bool", 4) ||
                       !strncmp(basictype, "int", 3)  ||
                       !strncmp(basictype, "long", 4) ||
                       !strncmp(basictype, "short", 5)) {
               Long_t ldefval;
               m->GetterMethod()->Execute(object, "", ldefval);
               snprintf(val,256, "%li", ldefval);
            }

            // Find out whether we have options ...

            TList *opt;
            // coverity[returned_pointer]: keep for later use
            if ((opt = m->GetOptions())) {
               Warning("Dialog", "option menu not yet implemented");
#if 0
               TMotifOptionMenu *o= new TMotifOptionMenu(argname);
               TIter nextopt(opt);
               TOptionListItem *it = 0;
               while ((it = (TOptionListItem*) nextopt())) {
                  char *name  = it->fOptName;
                  char *label = it->fOptLabel;
                  Long_t value  = it->fValue;
                  if (value != -9999) {
                     char val[256];
                     snprintf(val,256, "%li", value);
                     o->AddItem(name, val);
                  }else
                     o->AddItem(name, label);
               }
               o->SetData(val);
               fDialog->Add(o);
#endif
            } else {
               // we haven't got options - textfield ...
               fDialog->Add(argname, val, type);
            }
         } else {    // if m not found ...

            char val[256] = "";
            const char *tval = argument->GetDefault();
            if (tval && strlen(tval)) {
               // Remove leading and trailing quotes
               strlcpy(val, tval + (tval[0] == '"' ? 1 : 0), sizeof(val));
               if (val[strlen(val)-1] == '"')
                  val[strlen(val)-1]= 0;
            }
            fDialog->Add(argname, val, type);
         }
      }
      argpos++;
   }

   fDialog->Popup();
}

//______________________________________________________________________________
void TRootContextMenu::DrawEntry(TGMenuEntry *entry)
{
   // Draw context menu entry.

   int ty, offset;
   static int max_ascent = 0, max_descent = 0;

   TGPopupMenu::DrawEntry(entry);
   // draw the ? (help) in the right side when highlighting a menu entry
   if (entry->GetType() == kMenuEntry && (entry->GetStatus() & kMenuActiveMask)) {
      if (max_ascent == 0) {
         gVirtualX->GetFontProperties(fFontStruct, max_ascent, max_descent);
      }
      offset = (entry->GetEh() - (max_ascent + max_descent)) / 2;
      ty = entry->GetEy() + max_ascent + offset - 1;
      TGHotString s("&?");
      s.Draw(fId, fSelGC, fMenuWidth-12, ty);
   }
}

//______________________________________________________________________________
Bool_t TRootContextMenu::HandleButton(Event_t *event)
{
   // Handle button event in the context menu.

   int   id;
   void *ud;
   if ((event->fType == kButtonRelease) && (event->fX >= (Int_t)(fMenuWidth-15)) &&
       (event->fX <= (Int_t)fMenuWidth)) {
      id = EndMenu(ud);
      if (fHasGrab) gVirtualX->GrabPointer(0, 0, 0, 0, kFALSE);  // ungrab
      if (ud) {
         // retrieve the highlighted function
         TFunction *function = 0;
         if (id < kToggleStart) {
            TMethod *m = (TMethod *)ud;
            function = (TFunction *)m;
         } else if (id >= kToggleStart && id < kUserFunctionStart) {
            TToggle *t = (TToggle *)ud;
            TMethodCall *mc = (TMethodCall *)t->GetSetter();
            function = (TFunction *)mc->GetMethod();
         } else {
            TClassMenuItem *mi = (TClassMenuItem *)ud;
            function = gROOT->GetGlobalFunctionWithPrototype(mi->GetFunctionName());
         }
         if (function)
            fContextMenu->SetMethod(function);
      }
      OnlineHelp();
      return kTRUE;
   }
   return TGPopupMenu::HandleButton(event);
}

//______________________________________________________________________________
Bool_t TRootContextMenu::HandleCrossing(Event_t *event)
{
   // Handle pointer crossing event in context menu.

   if (event->fType == kLeaveNotify) {
      // just to reset the mouse pointer...
      HandleMotion(event);
   }
   return TGPopupMenu::HandleCrossing(event);
}

//______________________________________________________________________________
Bool_t TRootContextMenu::HandleMotion(Event_t *event)
{
   // Handle pointer motion event in context menu.

   static int toggle = 0;
   static Cursor_t handCur = kNone, rightCur = kNone;
   static UInt_t mask = kButtonPressMask | kButtonReleaseMask | kPointerMotionMask;

   if (handCur == kNone)
      handCur    = gVirtualX->CreateCursor(kHand);
   if (rightCur == kNone)
      rightCur   = gVirtualX->CreateCursor(kArrowRight);

   if (event->fType == kLeaveNotify) {
      gVirtualX->ChangeActivePointerGrab(fId, mask, rightCur);
      toggle = 0;
      return kTRUE;
   }
   // change the cursot to a small hand when over the ? (help)
   if ((event->fX >= (Int_t)(fMenuWidth-15)) && (event->fX <= (Int_t)fMenuWidth) &&
       fCurrent && (fCurrent->GetType() == kMenuEntry)) {
      if (toggle == 0) {
         gVirtualX->ChangeActivePointerGrab(fId, mask, handCur);
         toggle = 1;
      }
   }
   else {
      if (toggle == 1) {
         gVirtualX->ChangeActivePointerGrab(fId, mask, rightCur);
         toggle = 0;
      }
   }
   return TGPopupMenu::HandleMotion(event);
}

//______________________________________________________________________________
void TRootContextMenu::OnlineHelp()
{
   // Open the online help matching the actual class/method.

   TString clname;
   TString cmd;
   TString url = gEnv->GetValue("Browser.StartUrl", "http://root.cern.ch/root/html/");
   if (url.EndsWith(".html", TString::kIgnoreCase)) {
      if (url.Last('/') != kNPOS)
         url.Remove(url.Last('/'));
   }
   if (!url.EndsWith("/")) {
      url += '/';
   }
   TObject *obj = fContextMenu->GetSelectedObject();
   if (obj) {
      clname = obj->ClassName();
      if (fContextMenu->GetSelectedMethod()) {
         TString smeth = fContextMenu->GetSelectedMethod()->GetName();
         TMethod *method = obj->IsA()->GetMethodAllAny(smeth.Data());
         if (method) clname = method->GetClass()->GetName();
         url += clname;
         url += ".html";
         url += "#";
         url += clname;
         url += ":";
         url += smeth.Data();
      }
      else {
         url += clname;
         url += ".html";
      }
      if (fDialog) delete fDialog;
      fDialog = 0;
      cmd = TString::Format("new TGHtmlBrowser(\"%s\", 0, 900, 300);", url.Data());
      gROOT->ProcessLine(cmd.Data());
   }
}

//______________________________________________________________________________
Bool_t TRootContextMenu::ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2)
{
   // Handle context menu messages.

   TObjectSpy savedPad;
   if (GetContextMenu()->GetSelectedPad()) {
      savedPad.SetObject(gPad);
      gPad = GetContextMenu()->GetSelectedPad();
   }

   switch (GET_MSG(msg)) {

      case kC_COMMAND:

         switch (GET_SUBMSG(msg)) {

            case kCM_MENU:

               if (parm1 < kToggleStart) {
                  TMethod *m = (TMethod *) parm2;
                  GetContextMenu()->Action(m);
               } else if (parm1 >= kToggleStart && parm1 < kToggleListStart) {
                  TToggle *t = (TToggle *) parm2;
                  GetContextMenu()->Action(t);
               } else if (parm1 >= kToggleListStart && parm1<kUserFunctionStart) {
                  TToggle *t = (TToggle *) parm2;
                  if (t->GetState() == 0)
                     t->SetState(1);
               } else {
                  TClassMenuItem *mi = (TClassMenuItem*)parm2;
                  GetContextMenu()->Action(mi);
               }
               break;

            case kCM_BUTTON:
               if (parm1 == 1) {
                  const char *args = fDialog->GetParameters();
                  GetContextMenu()->Execute((char *)args);
                  delete fDialog;
                  fDialog = 0;
               }
               if (parm1 == 2) {
                  const char *args = fDialog->GetParameters();
                  GetContextMenu()->Execute((char *)args);
               }
               if (parm1 == 3) {
                  delete fDialog;
                  fDialog = 0;
               }
               if (parm1 == 4) {
                  OnlineHelp();
               }
               break;

            default:
               break;
         }
         break;

      case kC_TEXTENTRY:

         switch (GET_SUBMSG(msg)) {

            case kTE_ENTER:
               {
                  const char *args = fDialog->GetParameters();
                  GetContextMenu()->Execute((char *)args);
                  delete fDialog;
                  fDialog = 0;
               }
               break;

            default:
               break;
         }
         break;

      default:
         break;
   }

   if (savedPad.GetObject()) gPad = (TVirtualPad*) savedPad.GetObject();

   return kTRUE;
}

//______________________________________________________________________________
void TRootContextMenu::RecursiveRemove(TObject *obj)
{
   // Close the context menu if the object is deleted in the
   // RecursiveRemove() operation.

   void *ud;
   if (obj == fContextMenu->GetSelectedCanvas())
      fContextMenu->SetCanvas(0);
   if (obj == fContextMenu->GetSelectedPad())
      fContextMenu->SetPad(0);
   if (obj == fContextMenu->GetSelectedObject()) {
      // if the object being deleted is the one selected,
      // ungrab the mouse pointer and terminate (close) the menu
      fContextMenu->SetObject(0);
      if (fHasGrab) 
         gVirtualX->GrabPointer(0, 0, 0, 0, kFALSE);
      EndMenu(ud);
   }
}

