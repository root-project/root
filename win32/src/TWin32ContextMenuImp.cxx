// @(#)root/win32:$Name$:$Id$
// Author: Valery Fine   22/02/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TWin32ContextMenuImp.h"

#include "TContextMenuItem.h"

#include "TCanvas.h"
#include "TClass.h"
#include "TWin32Canvas.h"
#include "TContextMenu.h"
#include "TMethod.h"
#include "TMethodArg.h"
#include "TDataType.h"
#include "TMethodCall.h"
#include "TROOT.h"
#include "TGWin32WindowsObject.h"

#include "TContextMenuImp.h"
#include "TObjArray.h"
#include "TWin32Dialog.h"
#include "TApplication.h"
#include "TWin32Application.h"
#include "TBrowser.h"
#include "TWin32BrowserImp.h"

typedef struct  {
            char   *Title;
            WORD    Ident;
            WNDPROC ButtonCallBack;
            SIZE    size;
            } KW_BUTTONS;

// ClassImp(TWin32ContextMenuImp)

//______________________________________________________________________________
TWin32ContextMenuImp::TWin32ContextMenuImp(TContextMenu *c) :
                       TContextMenuImp(c),
                       TWin32Menu("ContextMenu","ContextMenu",kPopUpMenu),
                       fTitle("Title","Context Menu",MF_STRING, MF_GRAYED,-1)
{
    fDialog       = 0;
    fProperties   = 0;
    fPopupCreated = 0;
//    if (c)
//       CreatePopup();
}

//______________________________________________________________________________
TWin32ContextMenuImp::~TWin32ContextMenuImp(){
  //    cout << "TWin32ContextMenuImp::~TWin32ContextMenuImp() : this="<< this << endl;
     ClearProperties();
}

//______________________________________________________________________________
void TWin32ContextMenuImp::CreatePopup ( TObject *object ) {
    ;
}
//______________________________________________________________________________
void TWin32ContextMenuImp::CreatePopup () {
TContextMenu *c;

//*-*   Find the parent canvas window

    if (c=GetContextMenu()) {
        TCanvas *canvas = (TCanvas*)c->GetSelectedCanvas();
        if (canvas)
            fWindowObj   = (TGWin32WindowsObject *)((TWin32Canvas *)(canvas->GetCanvasImp()));
        else
        {
            TBrowser *browser = c->GetBrowser();
            if (browser)
                 fWindowObj   = (TGWin32WindowsObject *)((TWin32BrowserImp *)(browser->GetBrowserImp()));
            else
                 {Error("CreatePopup"," Attention !!! Wrong behavior !!! \n"); return;}
        }
 //*-*
//*-*  Create a title
//*-*  Since WIN32 menus have no title we use a zero position of menu + separartor instead
//*-*

//*-*  Add a title.
        fWindowObj->RegisterMenuItem(&fTitle);
        Add(&fTitle);
                                   Add(fWindowObj->GetStaticItem(kS1));
                                   Add(fWindowObj->GetStaticItem(kS1));

//*-*  Include the standard static item into the context menu

        Add(fWindowObj->GetStaticItem(kMNew));
                                   Add(fWindowObj->GetStaticItem(kS1));
        Add(fWindowObj->GetStaticItem(kMSave));
        Add(fWindowObj->GetStaticItem(kMSaveAs));
                                   Add(fWindowObj->GetStaticItem(kS1));
//        Add(cImp->GetStaticItem(kMCut));
//        Add(cImp->GetStaticItem(kMCopy));
//        Add(cImp->GetStaticItem(kMPaste));
//        Add(cImp->GetStaticItem(kMDelete));
                                   Add(fWindowObj->GetStaticItem(kS1));
        Add(fWindowObj->GetStaticItem(kMPrint));
                                   Add(fWindowObj->GetStaticItem(kS1));
        fPopupCreated = 1;

   }
}
//______________________________________________________________________________
void TWin32ContextMenuImp::ClearProperties()
{
//*-*   Delete the obsolete properties
       if (fProperties) {
         RemoveTheItem(fProperties);
         delete fProperties;
         fProperties = 0;
       }
}
//______________________________________________________________________________
void       TWin32ContextMenuImp::Dialog( TObject *object, TMethod *method )
{
    if ( !( object && method ) ) return;

#ifndef WIN32


    static Char_t argName[128];
    Arg args[20];

    const Int_t xPos         = 10;
    const Int_t yPos         = 10;
    const Int_t lineOffset   = 50;
    const Int_t buttonWidth  = 80;
    const Int_t buttonHeight = 30;

    Int_t xCurrPos           = xPos;
    Int_t yCurrPos           = yPos;

    if ( fDialog ) {
        XtDestroyWidget( fDialog );
        fDialog = NULL;
    }

#endif


    HWND win = fWindowObj->GetWindow();
    fDialog = new TWin32Dialog(win,"ContextMenuDialog",fContextMenu->CreateDialogTitle( object, method ));

    if ( fDialog ) {

        Int_t   x0 = 1;
        Int_t   y0 = 2*x0;
        Int_t MaxSize = 150;

        TMethodArg *argument = NULL;
        TIter next( method->GetListOfMethodArgs() );

        Int_t LeftMargin = fDialog->GetLeftMargin();
        Int_t VertStep   = fDialog->GetVertStep();
        Int_t yDefSize   = fDialog->GetyDefSize();
        Int_t wId        = fDialog->GetFirstID();

        POINT ControlPosition = { 0,0};
        SIZE  ControlSize     = { 0,0};
        UINT  lStyle;

//*-*   ControlPosition.x and ControlSize.cy are the constant for all controls

        ControlPosition.x   = LeftMargin;
        ControlSize.cy      = yDefSize;

        while ( argument = (TMethodArg *) next() ) {

            // Create a label gadget.
//*-* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//*-*          Label      as STATIC control

          char *argname  = fContextMenu->CreateArgumentTitle( argument );

//------------------ begin of copy / paste from void TRootContextMenu::Dialog(TObject *object, TMethod *method)

         const Text_t *type       = argument->GetTypeName();
         TDataType    *datatype   = gROOT->GetType(type);
         const Text_t *charstar   = "char*";
         Text_t        basictype [32];

         if (datatype) {
           strcpy(basictype, datatype->GetTypeName());
         } else {
            if (strncmp(type, "enum", 4) != 0)
              Warning("Dialog", "data type is not basic type, assuming (int)");
           strcpy(basictype, "int");
        }

        if (strchr(argname, '*')) {
           strcat(basictype, "*");
           type = charstar;
        }

        TDataMember *m = argument->GetDataMember();
        Text_t val[256]= "";
        if (m && m->GetterMethod()) {

         // WARNING !!!!!!!!
         // MUST "reset" getter method!!! otherwise TAxis methods doesn't work!!!
         Text_t gettername[256] = "";
         strcpy(gettername, m->GetterMethod()->GetMethodName());
         m->GetterMethod()->Init(object->IsA(), gettername, "");

         // Get the current value and form it as a text:

         if (!strncmp(basictype, "char*", 5)) {
            Text_t *tdefval;
            m->GetterMethod()->Execute(object, "", &tdefval);
            strncpy(val, tdefval, 255);
         } else if (!strncmp(basictype, "float", 5) ||
                    !strncmp(basictype, "double", 6)) {
            Double_t ddefval;
            m->GetterMethod()->Execute(object, "", ddefval);
            sprintf(val, "%g", ddefval);
         } else if (!strncmp(basictype, "char", 4) ||
                    !strncmp(basictype, "int", 3)  ||
                    !strncmp(basictype, "long", 4) ||
                    !strncmp(basictype, "short", 5)) {
            Long_t ldefval;
            m->GetterMethod()->Execute(object, "", ldefval);
            sprintf(val, "%li", ldefval);
         }

         // Find out whether we have options ...

         TList *opt;
         if ((opt = m->GetOptions())) {
            Warning("Dialog", "option menu not yet implemented", opt);
         }

       } else {    // if m not found ...
         const char *tval = argument->GetDefault();
         if (tval) strncpy(val, tval, 255);
      }
//------------------ end of copy / paste from void TRootContextMenu::Dialog(TObject *object, TMethod *method)

          lStyle              = WS_VISIBLE | WS_CHILD;
          ControlPosition.y  += VertStep;
          ControlSize.cx      = 4*strlen(argname);
          MaxSize = TMath::Max(MaxSize,(Int_t)ControlSize.cx);

          fDialog->AttachControlItem(&ControlPosition,
                                     &ControlSize,
                                     lStyle,0,
                                     argname,
                                     wId++,kWStatic);
//          yCurrPos += lineOffset/2;


            // Create a text field widget.

//*-*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//*-*                          text window
//*-*                Input TEXT window as EDIT control

          lStyle              = WS_VISIBLE | WS_CHILD |
                                ES_AUTOHSCROLL | WS_BORDER | WS_TABSTOP;
          ControlPosition.y  += VertStep;
          ControlSize.cx      = TMath::Max(MaxSize,(Int_t)(fDialog->GetWidth()-ControlPosition.x-LeftMargin));
          MaxSize = TMath::Max(MaxSize,(Int_t)ControlSize.cx);

//*-*     Save  an "active" wId
// ???          lpwId[i] = wId;

          fDialog->AttachControlItem(&ControlPosition,
                                     &ControlSize,
                                     lStyle,0,
                                     val,            // (char *) argument->GetDefault(),
                                     wId++,kWEdit);

            // remember the first widget to put initial focus on it
        }

//*-*    Set full dialog width

        fDialog->SetWidth(MaxSize+2*LeftMargin);

//*-*  - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - - - -
//*-*                 Create buttons OK, Cancel

       {
          int xStep, NumBut;
          KW_BUTTONS lpwButtons[2] =
                {"OK",       IDOK,          NULL,  0, 0,
                 "&Cancel",  IDCANCEL,      NULL,  0, 0
                };
          ControlPosition.y += VertStep+2;
          ControlSize.cy    = yDefSize;
          ControlSize.cx    = 5*ControlSize.cy/2;
          xStep             = (fDialog->GetWidth()-2*x0)/2;
          ControlPosition.x = x0+(xStep-ControlSize.cx)/2+2;
          for (NumBut=0;NumBut<2; NumBut++) {

            lStyle              = WS_VISIBLE | WS_CHILD | WS_TABSTOP;

            if (NumBut == 0) lStyle |= BS_DEFPUSHBUTTON;
            else             lStyle |= BS_PUSHBUTTON;

            fDialog->AttachControlItem(&ControlPosition,
                                       &ControlSize,
                                       lStyle,0,lpwButtons[NumBut].Title,
                                       lpwButtons[NumBut].Ident,kWButton);
            wId++;

            ControlPosition.x += xStep ;
          }
        }

        ControlPosition.y += yDefSize;
//* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


        fDialog->SetHeight(ControlPosition.y+y0);
        fDialog->SetNumberOfControls(wId-fDialog->GetFirstID());  // number of controls

//*-*   Dialog template has been created - Just draw it

        fDialog->Draw();

//*-*   Get result of dialog and execute command

//=       char *params =
//=         new char((char *)fDialog->GetDialogResult());
//=       TGWin32MenuExecute *CodeOp =
//=           new TGWin32MenuExecute (object, method, GetContextMenu(), params );
        ExecCommandThread();

//        GetContextMenu()->Execute( object, method, (char *)fDialog->GetDialogResult());

//*-*  Delete dialog

//        delete fDialog;
//        fDialog = 0;

    }
}
//______________________________________________________________________________
void  TWin32ContextMenuImp::DisplayPopup ( Int_t x, Int_t y)
{
    if (!GetContextMenu()) return;
//*-*

//   DetachItems();

//*-*
//*-*  In fact we ought call CreatePopup from ctor but ....
//
    if (!fPopupCreated) CreatePopup();

//*-*   Update a popup

    UpdateProperties();

//*-*   Display Popup

   HWND win = fWindowObj->GetWindow();
   if (win) {
//*-*  We have to convert these coord to the screen ones
       POINT mp = { x, y };
       ClientToScreen(win,&mp);
       TrackPopupMenu(fMenu, TPM_LEFTALIGN | TPM_RIGHTBUTTON, mp.x, mp.y, 0, win, NULL);
   }
}


//______________________________________________________________________________
void TWin32ContextMenuImp::ExecThreadCB(TWin32SendClass *code){
   TContextMenu *menu = GetContextMenu();
   TObjArray  *params      = (TObjArray  *)(fDialog->GetDialogResult());

   if (params) {
       menu->Execute(menu->GetSelectedObject(),
                     menu->GetSelectedMethod(),
                     params);
      params->Delete();
      delete params;
   }


//*-*  Delete dialog

     if (fDialog) {
         delete fDialog;
         fDialog = 0;
   }
   if (code) {
     delete code;
     code = 0;
   }
}

//______________________________________________________________________________
void TWin32ContextMenuImp::UpdateProperties()
{
       ClearProperties();
       TContextMenu *c = GetContextMenu();
       TObject *object = 0;

       if (c) {
         object    = c->GetSelectedObject();
       }

       if (object)
       {
//*-*   Change title

        fTitle.ModifyTitle(fContextMenu->CreatePopupTitle( object ));
        fTitle.Grayed();

//*-*  Include the "Properties" item "by canvases"

         fProperties = new TWin32MenuItem("Properties","&Properties",kSubMenu),
         fWindowObj->JoinMenu(fProperties);
         Add(fProperties);

//*-*  Create Menu "Properties"

         TClass *classPtr = NULL;
         TMethod *method  = NULL;

//*-*  Create a linked list
         TList *methodList = new TList();
         object->IsA()->GetMenuItems( methodList );
         TIter next( methodList );

         TWin32Menu *MenuProperties = fProperties->GetPopUpItem();

         while ( method = (TMethod *) next () ) {

           if ( classPtr != method->GetClass() ) {
//*-*  Add a separator.
             if (classPtr)
                   MenuProperties->Add(fWindowObj->GetStaticItem(kSMenuBarBreak));
//                 MenuProperties->Add(fWindowObj->GetStaticItem(kS1));
             classPtr = method->GetClass();
           }

//*-*  Create a popup item.
         TContextMenuItem *item = new TContextMenuItem(c, object, method,
                              "ContextItem",(Char_t *) method->GetName());
         fWindowObj->RegisterMenuItem(item);
         MenuProperties->Add(item);
       }

       // Delete linked list of methods.
       delete methodList;
     }
}


