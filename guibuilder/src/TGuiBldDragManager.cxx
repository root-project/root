// @(#)root/guibuilder:$Name:  $:$Id: TGuiBldDragManager.cxx,v 1.42 2006/04/13 15:33:03 brun Exp $
// Author: Valeriy Onuchin   12/09/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include "TGuiBldDragManager.h"
#include "TGuiBldEditor.h"
#include "TRootGuiBuilder.h"
#include "TGuiBldQuickHandler.h"

#include "TTimer.h"
#include "TList.h"
#include "TSystem.h"
#include "TMath.h"
#include "TGResourcePool.h"
#include "TROOT.h"
#include "TColor.h"
#include "KeySymbols.h"
#include "TGMenu.h"
#include "TGFileDialog.h"
#include "TGMsgBox.h"
#include "TRandom.h"
#include "TGButton.h"
#include "TGMdi.h"
#include "TError.h"

#include "TClassMenuItem.h"
#include "TMethod.h"
#include "TMethodArg.h"
#include "TToggle.h"
#include "TDataType.h"
#include "TObjString.h"


#undef DEBUG_LOCAL

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGuiBldDragManager                                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


ClassImp(TGuiBldDragManager)


static UInt_t gGridStep = 10;
static TGuiBldDragManager *gGuiBldDragManager = 0;


////////////////////////////////////////////////////////////////////////////////
class TGuiBldMenuDialog : public TGTransientFrame {

public:
   TGButton    *fOK;       // OK button
   //TGButton    *fApply;    // apply button
   TGButton    *fCancel;   // cancel button
   TObject     *fObject;   // selected object/frame
   TMethod     *fMethod;   // method to be applied 
   TGLayoutHints *fL1;     // internally used layout hints 
   TGLayoutHints *fL2;     // internally used layout hints 
   TList         *fWidgets;

public:
   virtual ~TGuiBldMenuDialog();
   TGuiBldMenuDialog(const TGWindow *main, TObject *obj, TMethod *method);

   const char *GetParameters();
   void CloseWindow();
   void ConnectButtonSignals();
   void Build();
   void Popup();
   void ApplyMethod();
   void Add(const char *argname, const char *value, const char *type);
};

static TGuiBldMenuDialog *gMenuDialog = 0;

//______________________________________________________________________________
TGuiBldMenuDialog::TGuiBldMenuDialog(const TGWindow *main, TObject *obj, TMethod *method) : 
    TGTransientFrame(gClient->GetDefaultRoot(), main, 200, 100)
{
   // ctor.

   fObject = obj;
   fMethod = method;
   if (!obj) return; // zombie

   fWidgets = new TList();

   fL1 = new TGLayoutHints(kLHintsTop | kLHintsCenterX, 0, 0, 5, 0);
   fL2 = new TGLayoutHints(kLHintsTop | kLHintsLeft, 5, 5, 5, 5);

   TString title = obj->ClassName();
   title += "::";
   title += method->GetName();

   Build();
   ConnectButtonSignals();

   SetWindowName(title);
   SetIconName(title);
   SetEditDisabled(kEditDisable);

   //TRootGuiBuilder::PropagateBgndColor(this, TRootGuiBuilder::GetBgnd());
}

//______________________________________________________________________________
TGuiBldMenuDialog::~TGuiBldMenuDialog()
{
   // dtor.

   fWidgets->Delete();
   delete fWidgets;
   delete fL1;
   delete fL2;
}

//______________________________________________________________________________
void TGuiBldMenuDialog::ConnectButtonSignals()
{
   // connect signals

   fOK->Connect("Pressed()", "TGuiBldDragManager", gGuiBldDragManager, "DoDialogOK()");
//   fApply->Connect("Pressed()", "TGuiBldDragManager", gGuiBldDragManager, "DoDialogApply()");
   fCancel->Connect("Pressed()", "TGuiBldDragManager", gGuiBldDragManager, "DoDialogCancel()");
}

//______________________________________________________________________________
void TGuiBldMenuDialog::ApplyMethod()
{
   // execute method for obejct with input args

   const char *params =  GetParameters();
   fObject->Execute(fMethod->GetName(), params);
}

//______________________________________________________________________________
const char *TGuiBldMenuDialog::GetParameters()
{
   // return input parameters as single string

   static char params[1024];
   char param[256];

   TObjString   *str;
   TObject      *obj;

   Int_t selfobjpos;
//   if (fMenu->GetContextMenu()->GetSelectedMenuItem())
//      selfobjpos = fMenu->GetContextMenu()->GetSelectedMenuItem()->GetSelfObjectPos();
//   else
      selfobjpos = -1;

   params[0] = 0;
   TIter next(fWidgets);
   Int_t nparam = 0;

   while ((obj = next())) {        // first element is label, skip...
      if (obj->IsA() != TGLabel::Class()) break;
      obj = next();                // get either TGTextEntry or TGComboBox
      str = (TObjString *) next(); // get type string

      nparam++;

      const char *type = str->GetString().Data();
      const char *data = 0;

      if (obj->IsA() == TGTextEntry::Class())
         data = ((TGTextEntry *) obj)->GetBuffer()->GetString();

      // TODO: Combobox...

      // if necessary, replace the selected object by it's address
      if (selfobjpos == nparam-1) {
         if (params[0]) strcat(params, ",");
         sprintf(param, "(TObject*)0x%lx", (Long_t)fObject);
         strcat(params, param);
      }

      if (params[0]) strcat(params, ",");
      if (data) {
         if (!strncmp(type, "char*", 5))
            sprintf(param, "\"%s\"", data);
         else
            strcpy(param, data);
      } else
         strcpy(param, "0");

      strcat(params, param);
   }

   // if selected object is the last argument, have to insert it here
   if (selfobjpos == nparam) {
      if (params[0]) strcat(params, ",");
      sprintf(param, "(TObject*)0x%lx", (Long_t)fObject);
      strcat(params, param);
   }

   return params;
}

//______________________________________________________________________________
static TString CreateArgumentTitle(TMethodArg *argument)
{
   // Create string describing argument 

   static TString ret;

   if (argument) {
      ret.Form("(%s)  %s", argument->GetTitle(), argument->GetName());
      if (argument->GetDefault() && *(argument->GetDefault())) {
         ret += "  [default: ";
         ret += argument->GetDefault();
         ret += "]";
      }
   }

   return ret;
}

//______________________________________________________________________________
void TGuiBldMenuDialog::Add(const char *argname, const char *value, const char *type)
{
   // Add a label and text input field.

   TGLabel      *l = new TGLabel(this, argname);
   TGTextBuffer *b = new TGTextBuffer(20);
   b->AddText(0, value);
   TGTextEntry  *t = new TGTextEntry(this, b);

   t->Connect("ReturnPressed", "TGuiBldDragManager", gGuiBldDragManager, "DoDialogOK()");
   t->Resize(260, t->GetDefaultHeight());
   AddFrame(l, fL1);
   AddFrame(t, fL2);

   fWidgets->Add(l);
   fWidgets->Add(t);
   fWidgets->Add(new TObjString(type));
}

//______________________________________________________________________________
void TGuiBldMenuDialog::CloseWindow()
{
   // close window

   gGuiBldDragManager->DoDialogCancel();
}

//______________________________________________________________________________
void TGuiBldMenuDialog::Build()
{
   // build dialog

   TMethodArg *argument = 0;
   Int_t selfobjpos = -1;

   TIter next(fMethod->GetListOfMethodArgs());
   Int_t argpos = 0;

   while ((argument = (TMethodArg *) next())) {
      // Do not input argument for self object
      if (selfobjpos != argpos) {
         const char *argname    = CreateArgumentTitle(argument).Data();
         const char *type       = argument->GetTypeName();
         TDataType  *datatype   = gROOT->GetType(type);
         const char *charstar   = "char*";
         char        basictype[32];

         if (datatype) {
            strcpy(basictype, datatype->GetTypeName());
         } else {
            TClass *cl = gROOT->GetClass(type);
            if (strncmp(type, "enum", 4) && (cl && !(cl->Property() & kIsEnum)))
               Warning("Dialog", "data type is not basic type, assuming (int)");
            strcpy(basictype, "int");
         }

         if (strchr(argname, '*')) {
            strcat(basictype, "*");
            type = charstar;
         }

         TDataMember *m = argument->GetDataMember();
         if (m && m->GetterMethod(fObject->IsA())) {

            // Get the current value and form it as a text:
            char val[256];

            if (!strncmp(basictype, "char*", 5)) {
               char *tdefval;
               m->GetterMethod()->Execute(fObject, "", &tdefval);
               strncpy(val, tdefval, 255);
            } else if (!strncmp(basictype, "float", 5) ||
                       !strncmp(basictype, "double", 6)) {
               Double_t ddefval;
               m->GetterMethod()->Execute(fObject, "", ddefval);
               sprintf(val, "%g", ddefval);
            } else if (!strncmp(basictype, "char", 4) ||
                       !strncmp(basictype, "bool", 4) ||
                       !strncmp(basictype, "int", 3)  ||
                       !strncmp(basictype, "long", 4) ||
                       !strncmp(basictype, "short", 5)) {
               Long_t ldefval;
               m->GetterMethod()->Execute(fObject, "", ldefval);
               sprintf(val, "%li", ldefval);
            }

            // Find out whether we have options ...

            TList *opt;
            if ((opt = m->GetOptions())) {
               Warning("Dialog", "option menu not yet implemented", opt);

            } else {
               // we haven't got options - textfield ...
               Add(argname, val, type);
            }
         } else {    // if m not found ...

            char val[256] = "";
            const char *tval = argument->GetDefault();
            if (tval) strncpy(val, tval, 255);
            Add(argname, val, type);
         }
      }
      argpos++;
   }

   // add OK, Apply, Cancel buttons
   TGHorizontalFrame *hf = new TGHorizontalFrame(this, 60, 20, kFixedWidth);
   TGLayoutHints     *l1 = new TGLayoutHints(kLHintsCenterY | kLHintsExpandX, 5, 5, 0, 0);
   UInt_t  width = 0, height = 0;

   fWidgets->Add(l1);

   fOK = new TGTextButton(hf, "&OK", 1);
   hf->AddFrame(fOK, l1);
   fWidgets->Add(fOK);
   height = fOK->GetDefaultHeight();
   width  = TMath::Max(width, fOK->GetDefaultWidth());

/*
   fApply = new TGTextButton(hf, "&Apply", 2);
   hf->AddFrame(fApply, l1);
   fWidgets->Add(fApply);
   height = fApply->GetDefaultHeight();
   width  = TMath::Max(width, fApply->GetDefaultWidth());
*/

   fCancel = new TGTextButton(hf, "&Cancel", 3);
   hf->AddFrame(fCancel, l1);
   fWidgets->Add(fCancel);
   height = fCancel->GetDefaultHeight();
   width  = TMath::Max(width, fCancel->GetDefaultWidth());

   // place buttons at the bottom
   l1 = new TGLayoutHints(kLHintsBottom | kLHintsCenterX, 0, 0, 5, 5);
   AddFrame(hf, l1);
   fWidgets->Add(l1);
   fWidgets->Add(hf);

   hf->Resize((width + 20) * 3, height);

   // map all widgets and calculate size of dialog
   MapSubwindows();
}

//______________________________________________________________________________
void TGuiBldMenuDialog::Popup()
{
   // popup dialog

   UInt_t width  = GetDefaultWidth();
   UInt_t height = GetDefaultHeight();

   Resize(width, height);

   Window_t wdummy;
   Int_t x = (Int_t)((TGFrame*)fMain)->GetWidth();
   Int_t y = (Int_t)((TGFrame*)fMain)->GetHeight();
   gVirtualX->TranslateCoordinates(fMain->GetId(), fClient->GetDefaultRoot()->GetId(),
                                   x, y, x, y, wdummy);

   x += 10;
   y += 10;

   // make the message box non-resizable
   SetWMSize(width, height);
   SetWMSizeHints(width, height, width, height, 0, 0);

   SetMWMHints(kMWMDecorAll | kMWMDecorResizeH  | kMWMDecorMaximize |
                              kMWMDecorMinimize | kMWMDecorMenu,
               kMWMFuncAll  | kMWMFuncResize    | kMWMFuncMaximize |
                              kMWMFuncMinimize,
               kMWMInputModeless);

   Move(x, y);
   SetWMPosition(x, y);
   MapRaised();
   fClient->WaitFor(this);
}


///////////////////////// auxilary static functions ///////////////////////////
//______________________________________________________________________________
static Window_t GetWindowFromPoint(Int_t x, Int_t y)
{
   // returns a window located at point x,y (in screen coordinates)

   Window_t src, dst, child;
   Window_t ret = 0;
   Int_t xx = x;
   Int_t yy = y;

   if (!gGuiBldDragManager || gGuiBldDragManager->IsStopped() ||
       !gClient->IsEditable()) return 0;

   dst = src = child = gVirtualX->GetDefaultRootWindow();

   while (child && dst) {
      src = dst;
      dst = child;
      gVirtualX->TranslateCoordinates(src, dst, xx, yy, xx, yy, child);
      ret = dst;
   }
   return ret;
}

//______________________________________________________________________________
static void GuiBldErrorHandler(Int_t /*level*/, Bool_t /*abort*/, 
                                 const char * /*location*/, const char * /*msg*/)
{
   // our own error handler (not used yet)

}

////////////////////////////////////////////////////////////////////////////////
class TGuiBldDragManagerGrid {

public:
   static UInt_t   fgStep;
   static ULong_t  fgPixel;
   static TGGC    *fgBgnd;

   Pixmap_t    fPixmap;
   TGWindow   *fWindow;
   Int_t       fWinId;

   TGuiBldDragManagerGrid();
   ~TGuiBldDragManagerGrid();
   void  Draw();
   void  SetStep(UInt_t step);
   void  InitPixmap();
   void  InitBgnd();
};

UInt_t   TGuiBldDragManagerGrid::fgStep = gGridStep;
ULong_t  TGuiBldDragManagerGrid::fgPixel = 0;
TGGC    *TGuiBldDragManagerGrid::fgBgnd = 0;

//______________________________________________________________________________
TGuiBldDragManagerGrid::TGuiBldDragManagerGrid()
{
   // creates a grid background over a window

   fPixmap = 0;
   fWindow = 0;
   fWinId = 0;

   if (!fgBgnd) InitBgnd();
   SetStep(fgStep);
}
 

//______________________________________________________________________________
TGuiBldDragManagerGrid::~TGuiBldDragManagerGrid()
{
   // ctor.

   fWindow = gClient->GetWindowById(fWinId);

   if (fWindow) {
      fWindow->SetBackgroundPixmap(0);
      fWindow->SetBackgroundColor(((TGFrame*)fWindow)->GetBackground());
      gClient->NeedRedraw(fWindow, kTRUE);
   }
   if (fPixmap) {
      gVirtualX->DeletePixmap(fPixmap);
   }

   fPixmap = 0;
   fWindow = 0;
   fWinId = 0;
}
 
//______________________________________________________________________________
void TGuiBldDragManagerGrid::SetStep(UInt_t step)
{
   // set the grid step

   if (!gClient || !gClient->IsEditable()) return;

   fWindow = (TGWindow*)gClient->GetRoot();
   fWinId = fWindow->GetId();
   fgStep = step;
   InitPixmap();

}

//______________________________________________________________________________
void TGuiBldDragManagerGrid::InitBgnd()
{
   // initiate background

   if (fgBgnd) return;

   fgBgnd = new TGGC(TGFrame::GetBckgndGC());

   Float_t r, g, b;

   r = 220./255;
   g = 219./255;
   b = 217./255;

   fgPixel = TColor::RGB2Pixel(r, g, b);
   fgBgnd->SetForeground(fgPixel);
}

//______________________________________________________________________________
void TGuiBldDragManagerGrid::InitPixmap()
{
   // creates background pixmap

   if (fPixmap) {
      gVirtualX->DeletePixmap(fPixmap);
   }

   fPixmap = gVirtualX->CreatePixmap(gClient->GetDefaultRoot()->GetId(), fgStep, fgStep);
   gVirtualX->FillRectangle(fPixmap, fgBgnd->GetGC(), 0, 0, fgStep, fgStep);

   if(fgStep > 2) {
      gVirtualX->FillRectangle(fPixmap, TGFrame::GetShadowGC()(),
                               fgStep - 1, fgStep - 1, 1, 1);
   }
}

//______________________________________________________________________________
void TGuiBldDragManagerGrid::Draw()
{
   // draw grid over frame

   if (!gClient || !gClient->IsEditable()) return;

   fWindow = gClient->GetWindowById(fWinId);

   if (fWindow && (fWindow != gClient->GetRoot())) {
      fWindow->SetBackgroundPixmap(0);
      fWindow->SetBackgroundColor(((TGFrame*)fWindow)->GetBackground());
      gClient->NeedRedraw(fWindow);
   }

   if (!fPixmap) InitPixmap();

   fWindow = (TGWindow*)gClient->GetRoot();
   fWinId = fWindow->GetId();
   fWindow->SetBackgroundPixmap(fPixmap);

   gClient->NeedRedraw(fWindow);
}


////////////////////////////////////////////////////////////////////////////////
class TGuiBldDragManagerRepeatTimer : public TTimer {

private:
   TGuiBldDragManager *fManager;   // back pointer

public:
   TGuiBldDragManagerRepeatTimer(TGuiBldDragManager *m, Long_t ms) :
                                 TTimer(ms, kTRUE) { fManager = m; }
   Bool_t Notify() { fManager->HandleTimer(this); Reset(); return kFALSE; }
};


////////////////////////////////////////////////////////////////////////////////
class TGGrabRect : public TGFrame {

private:
   Pixmap_t    fPixmap;
   ECursor     fType;

public:
   TGGrabRect(Int_t type);
   ~TGGrabRect() {}

   Bool_t HandleButton(Event_t *ev);
   ECursor GetType() const { return fType; }
};

//______________________________________________________________________________
TGGrabRect::TGGrabRect(Int_t type) : TGFrame(gClient->GetDefaultRoot(), 7, 7, kTempFrame)
{
   // ctor.

   switch (type) {
      case 0:
         fType = kTopLeft;
         break;
      case 1:
         fType = kTopSide;
         break;
      case 2:
         fType = kTopRight;
         break;
      case 3:
         fType = kBottomLeft;
         break;
      case 4:
         fType = kLeftSide;
         break;
      case 5:
         fType = kRightSide;
         break;
      case 6:
         fType = kBottomSide;
         break;
      case 7:
         fType = kBottomRight;
         break;
   }

   SetWindowAttributes_t attr;
   attr.fMask             = kWAOverrideRedirect | kWASaveUnder;
   attr.fOverrideRedirect = kTRUE;
   attr.fSaveUnder        = kTRUE;

   gVirtualX->ChangeWindowAttributes(fId, &attr);

   fPixmap = gVirtualX->CreatePixmap(gVirtualX->GetDefaultRootWindow(), 7, 7);
   const TGGC *bgc = fClient->GetResourcePool()->GetSelectedBckgndGC();
   const TGGC *gc = fClient->GetResourcePool()->GetSelectedGC();

   gVirtualX->FillRectangle(fPixmap, bgc->GetGC(), 0, 0, 6, 6);
   gVirtualX->DrawRectangle(fPixmap, gc->GetGC(), 0, 0, 6, 6);

   AddInput(kButtonPressMask);
   SetBackgroundPixmap(fPixmap);

   gVirtualX->SetCursor(fId, gVirtualX->CreateCursor(fType));
}

//______________________________________________________________________________
Bool_t TGGrabRect::HandleButton(Event_t *ev)
{
   // handle button press

   gGuiBldDragManager->CheckDragResize(ev);
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
class TGAroundFrame : public TGFrame {

public:
   TGAroundFrame();
   ~TGAroundFrame() {}
};

//______________________________________________________________________________
TGAroundFrame::TGAroundFrame() : TGFrame(gClient->GetDefaultRoot(), 1, 1,
                                         kTempFrame | kOwnBackground)
{
   // ctor.

   SetWindowAttributes_t attr;
   attr.fMask             = kWAOverrideRedirect | kWASaveUnder;
   attr.fOverrideRedirect = kTRUE;
   attr.fSaveUnder        = kTRUE;

   gVirtualX->ChangeWindowAttributes(fId, &attr);
   ULong_t red;
   fClient->GetColorByName("red", red);
   SetBackgroundColor(red); 

}


////////////////////////////////////////////////////////////////////////////////
class TGuiBldDragManagerPimpl {

friend class TGuiBldDragManager;

private:
   TGuiBldDragManager     *fManager;   // drag and drop manager
   TTimer            *fRepeatTimer;    // repeat rate timer (when mouse stays pressed)
   TGFrame           *fGrab;           // grabbed/selected frame
   TGLayoutHints     *fGrabLayout;     // layout of grabbed frame
   TGFrame           *fSaveGrab;       // used during context menu handling
   TGFrame           *fClickFrame;     // last clicked frame
   TGuiBldDragManagerGrid *fGrid;      //
   ECursor            fResizeType;     // defines resize type 
   Int_t              fX0, fY0;        // initial drag position in pixels
   Int_t              fX, fY;          // current drag position in pixels
   Int_t              fXf, fYf;        // offset of inititial position inside frame
   Int_t              fGrabX, fGrabY;  // 
   const TGWindow    *fGrabParent;     // parent of the grabbed/seleceted frame
   Int_t              fLastPopupAction;//
   Bool_t             fReplaceOn;
   TGGrabRect        *fGrabRect[8];    // small rectangles drawn over grabbed/selected frame
   TGFrame           *fAroundFrame[4]; // red lines drawn over layouted frame
   Bool_t             fGrabRectHidden;
   TGFrameElement    *fGrabListPosition;
   Bool_t             fButtonPressed;  //
   Bool_t             fCompacted;   //
   TGFrame           *fPlane; // highlighted plain composite frame when mose is moving
   TGFrame           *fSpacePressedFrame; // frame which was grabbed via spacebar pressed
   Bool_t             fPlacePopup; // kTRUE is menu fo frame was placed
   TList             *fFrameMenuTrash;

public:
   TGuiBldDragManagerPimpl(TGuiBldDragManager *m) {
      fManager = m;
      fRepeatTimer = new TGuiBldDragManagerRepeatTimer(m, 100);

      int i = 0;
      for (i = 0; i <8; i++) {
         fGrabRect[i] = new TGGrabRect(i);
      }
      for (i = 0; i <4; i++) {
         fAroundFrame[i] = new TGAroundFrame();
      }

      fFrameMenuTrash = new TList();

      ResetParams();
   }
   void ResetParams() {
      fGrab = 0;
      fSaveGrab = 0;
      fClickFrame = 0;
      fGrid = 0;
      fX0 = fY0 = fX = fY = fXf = fYf = fGrabX = fGrabY = 0;
      fGrabParent = 0;
      fResizeType = kPointer;
      fLastPopupAction = kNoneAct;
      fReplaceOn = kFALSE;
      fGrabLayout = 0;
      fGrabRectHidden = kFALSE;
      fGrabListPosition = 0;
      fButtonPressed = kFALSE;
      fCompacted = kFALSE;
      fPlane = 0;
      fSpacePressedFrame = 0;
      fPlacePopup = kFALSE;
      fFrameMenuTrash->Delete();
   }

   ~TGuiBldDragManagerPimpl() {
      int i;
      for (i = 0; i <8; i++) {
         delete fGrabRect[i];
      }
      for (i = 0; i <4; i++) {
         delete fAroundFrame[i];
      }

      delete fRepeatTimer;
      delete fGrab;
      fFrameMenuTrash->Delete();
      delete fFrameMenuTrash;

      if (fPlane) {
         fPlane->ChangeOptions(fPlane->GetOptions() & ~kRaisedFrame);
         gClient->NeedRedraw(fPlane, kTRUE);
         fPlane = 0;
      }
   }
};


////////////////////////////////////////////////////////////////////////////////
TGuiBldDragManager::TGuiBldDragManager() : TVirtualDragManager() ,
                    TGFrame(gClient->GetDefaultRoot(), 1, 1)
{
   // Constructor. Create "fantom window".

   SetWindowAttributes_t attr;
   attr.fMask             = kWAOverrideRedirect | kWASaveUnder;
   attr.fOverrideRedirect = kTRUE;
   attr.fSaveUnder        = kTRUE;

   gVirtualX->ChangeWindowAttributes(fId, &attr);

   gGuiBldDragManager = this;
   fPimpl = new TGuiBldDragManagerPimpl(this);

   Reset1();
   fSelectionIsOn = kFALSE;
   fFrameMenu     = 0;
   fLassoMenu     = 0;
   fEditor        = 0;
   fBuilder       = 0;
   fQuickHandler  = 0;
   fLassoDrawn    = kFALSE;
   fDropStatus    = kFALSE;
   fStop          = kTRUE;

   TString tmpfile = gSystem->TempDirectory();
   fPasteFileName = gSystem->ConcatFileName(tmpfile.Data(),
                             Form("RootGuiBldClipboard%d.C", gSystem->GetPid()));

   fName = "Gui Builder Drag Manager";
   SetWindowName(fName.Data());

   // let's try to solve the problems by myself
   SetErrorHandler(GuiBldErrorHandler);

   fClient->UnregisterWindow(this);
}

//______________________________________________________________________________
TGuiBldDragManager::~TGuiBldDragManager()
{
   // Destructor

   SetEditable(kFALSE);

   delete fPimpl;

   delete fBuilder;
   fBuilder = 0;

//   delete fEditor;
//   fEditor = 0;

   delete fFrameMenu;
   fFrameMenu =0;

   delete fLassoMenu;
   fLassoMenu = 0;

   if (!gSystem->AccessPathName(fPasteFileName.Data())) {
      gSystem->Unlink(fPasteFileName.Data());
   }

   gGuiBldDragManager = 0;
}

//______________________________________________________________________________
void TGuiBldDragManager::Reset1()
{
   //  reset some parameters 

   TVirtualDragManager::Init();
   fTargetId = 0;
   fPimpl->fPlacePopup = kFALSE;
   SetCursorType(kPointer);
}

//______________________________________________________________________________
void TGuiBldDragManager::Snap2Grid()
{
   // draw grid on editable frame and restore background on previuosly editted one

   if (fStop) return;

   delete fPimpl->fGrid;

   fPimpl->fGrid = new TGuiBldDragManagerGrid();
   fPimpl->fGrid->Draw();
}

//______________________________________________________________________________
UInt_t TGuiBldDragManager::GetGridStep()
{
   // returns grid step

   return fPimpl->fGrid ? fPimpl->fGrid->fgStep : 1;
}

//______________________________________________________________________________
void TGuiBldDragManager::SetGridStep(UInt_t step)
{
   // sets a grid step

   fPimpl->fGrid->SetStep(step);
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::IgnoreEvent(Event_t *event)
{
   // returns kTRUE if event is rejected for processing by drag manager

   if (fStop || !fClient || !fClient->IsEditable()) return kTRUE;
   if (event->fType == kClientMessage) return kFALSE;

   TGWindow *w = fClient->GetWindowById(event->fWindow);

   if (w) {
      if (IsEditDisabled(w)) {
         w = GetEditableParent((TGFrame*)w);
         return !w;
      }
   } else {
      return kTRUE;
   }
   return kFALSE;
}

//______________________________________________________________________________
TGFrame* TGuiBldDragManager::InEditable(Window_t id)
{
   // returns a pointer to parent window which is being editted

   if (fStop || !id) return 0;

   Window_t preparent = id;
   Window_t parent = (Window_t)gVirtualX->GetParent(id);

   while (!parent || (parent != fClient->GetDefaultRoot()->GetId())) {
      if (parent == fClient->GetRoot()->GetId()) {
         TGWindow *w = fClient->GetWindowById(preparent);
         return (w ? (TGFrame*)w : 0);
      }
      preparent = parent;
      parent = gVirtualX->GetParent(parent);
   }
   return 0;
}

//______________________________________________________________________________
TGCompositeFrame *TGuiBldDragManager::FindCompositeFrame(Window_t id)
{
   // finds first composite parent

   if (fStop || !id) return 0;

   Window_t parent = id;

   while (!parent || (parent != fClient->GetDefaultRoot()->GetId())) {
      TGWindow *w = fClient->GetWindowById(parent);
      if (w) {
         if (w->InheritsFrom(TGCompositeFrame::Class())) {
            return (TGCompositeFrame*)w;
         }
      }
      parent = gVirtualX->GetParent(parent);
   }
   return 0;
}

//______________________________________________________________________________
void TGuiBldDragManager::SetCursorType(Int_t cur)
{
   // sets cursor

   if (fStop) return;

   static UInt_t gid = 0;
   static UInt_t rid = 0;

   if (fPimpl->fGrab && (gid != fPimpl->fGrab->GetId())) {
      gVirtualX->SetCursor(fPimpl->fGrab->GetId(),
                           gVirtualX->CreateCursor((ECursor)cur));
      gid = fPimpl->fGrab->GetId();
   }
   if (fClient->IsEditable() && (rid != fClient->GetRoot()->GetId())) {
      gVirtualX->SetCursor(fClient->GetRoot()->GetId(),
                           gVirtualX->CreateCursor((ECursor)cur));
      rid = fClient->GetRoot()->GetId();
   }
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::CheckDragResize(Event_t *event)
{
   // check resize type event

   if (fStop) return kFALSE;

   Bool_t ret = kFALSE;
   fPimpl->fResizeType = kPointer;

   for (int i = 0; i < 8; i++) {
      if (fPimpl->fGrabRect[i]->GetId() == event->fWindow) {
         fPimpl->fResizeType = fPimpl->fGrabRect[i]->GetType();
         ret = kTRUE;
      }
   }

   if ((event->fType == kButtonPress) && (fPimpl->fResizeType != kPointer)) {
      fDragType = kDragResize;
      ret = kTRUE;
   }

   SetCursorType(ret ? fPimpl->fResizeType : kPointer);
   return ret;
}

//______________________________________________________________________________
void TGuiBldDragManager::DoRedraw()
{
   // redraw content of the editted window

   if (fStop || !fClient || !fClient->IsEditable()) return;

   TGWindow *root = (TGWindow*)fClient->GetRoot();

   fClient->NeedRedraw(root, kTRUE);

   if (fBuilder) {
      fClient->NeedRedraw(fBuilder, kTRUE);
   }
}

//______________________________________________________________________________
void TGuiBldDragManager::SwitchEditable(TGFrame *frame)
{
   // switch editable

   if (fStop || !frame) return;

   TGCompositeFrame *comp = 0;

   if (frame->InheritsFrom(TGCompositeFrame::Class())) {
      comp = (TGCompositeFrame *)frame;
   } else if (frame->GetParent()->InheritsFrom(TGCompositeFrame::Class())) {
      comp = (TGCompositeFrame *)frame->GetParent();
   }

   if (!comp) {
      return;
   }

   TString str = comp->ClassName();
   str += "::";
   str += comp->GetName();

   if (IsEditDisabled(comp)) {
      if (fBuilder) {
         str += " cannot be editted.";
         fBuilder->UpdateStatusBar(str.Data());
      }
      return;
   }

   if (frame != comp) {
      SelectFrame(frame);
   }

   if (comp->IsEditable()) {
      return;
   }

   RaiseMdiFrame(comp);
   comp->SetEditable(kTRUE);

   if (fBuilder) {
      str += " set editable";
      fBuilder->UpdateStatusBar(str.Data());
   }
}

//______________________________________________________________________________
void TGuiBldDragManager::SelectFrame(TGFrame *frame, Bool_t add)
{
   // make a frame grabbed/selected

   if (fStop || !frame || (frame->GetParent() == fClient->GetDefaultRoot()) ||
       !fClient->IsEditable()) return;

   TString str = frame->ClassName();
   str += "::";
   str += frame->GetName();

   if (IsGrabDisabled(frame)) {
      if (fBuilder) {
         str += "can not be selected";
         fBuilder->UpdateStatusBar(str.Data());
      }
      return;
   }

   // do not grab mdi frames (quick hack)
   if (fBuilder && frame->InheritsFrom(TGMdiFrame::Class())) return;

   static Int_t x, x0, y, y0, xx, yy;
   Window_t c;

   RaiseMdiFrame(FindMdiFrame(frame));
   frame->MapRaised();

   if (!add) {
      fDragType = (fDragType != kDragCopy ) ?  kDragMove : fDragType;

      gVirtualX->TranslateCoordinates(frame->GetId(),
                                      fClient->GetDefaultRoot()->GetId(),
                                      0, 0, x0, y0, c);

      x = x0 + frame->GetWidth();
      y = y0 + frame->GetHeight();

      if (fBuilder) {
         str += " selected";
         str += (IsEditDisabled(frame) || IsFixedLayout(frame)  ? ". This frame cannot be editted" : 
                 ". Press spacebar to start editting.");
         fBuilder->UpdateStatusBar(str.Data());
         if (IsFixedSize(frame)) str += " This frame cannot be resized (resize the parent frame).";
      }

   } else {

      gVirtualX->TranslateCoordinates(frame->GetId(),
                                      fClient->GetDefaultRoot()->GetId(),
                                      0, 0, xx, yy, c);

      fDragType = kDragLasso;
      fPimpl->fX0 = x0 = TMath::Min(x0, xx);
      fPimpl->fX = x = TMath::Max(x, xx + (Int_t)frame->GetWidth());
      fPimpl->fY0 = y0 = TMath::Min(y0, yy);
      fPimpl->fY = y = TMath::Max(y, yy + (Int_t)frame->GetHeight());

      DrawLasso();
   }

   fFrameUnder = fPimpl->fGrab = frame;
   SetCursorType(kMove);

   fLassoDrawn = kFALSE;
   ChangeSelected(fPimpl->fGrab);
   DrawGrabRectangles(fPimpl->fGrab);
}

//______________________________________________________________________________
void TGuiBldDragManager::ChangeSelected(TGFrame *fr)
{
   // inform outside wold the selected frame was changed

   if (fStop) return;

   TGFrame *sel = fr;

   if (fBuilder && (sel == fBuilder->GetMdiMain()->GetCurrent())) {
      sel = 0;
   }

   if (!fr) UngrabFrame();
   if (fEditor) fEditor->ChangeSelected(sel);
   if (fBuilder) {
      fBuilder->ChangeSelected(sel);
      fBuilder->Update();
   }
}

//______________________________________________________________________________
void TGuiBldDragManager::GrabFrame(TGFrame *frame)
{
   // grab frame (see SelectFrame)

   if (fStop || !frame || !fClient->IsEditable()) return;

   fPimpl->fGrabParent = frame->GetParent();
   fPimpl->fGrabX = frame->GetX();
   fPimpl->fGrabY = frame->GetY();

   Window_t c;

#ifdef DEBUG_LOCAL
printf("GrabFrame1\n");
#endif

   gVirtualX->TranslateCoordinates(frame->GetId(),
                                   fClient->GetDefaultRoot()->GetId(),
                                   0, 0, fPimpl->fX0, fPimpl->fY0, c);

#ifdef DEBUG_LOCAL
printf("GrabFrame2\n");
#endif

   fPimpl->fX = fPimpl->fX0;
   fPimpl->fY = fPimpl->fY0;

   if (frame->GetFrameElement() && frame->GetFrameElement()->fLayout) {
      fPimpl->fGrabLayout = frame->GetFrameElement()->fLayout;
   }

   if (fPimpl->fGrabParent && frame->GetFrameElement() &&
       fPimpl->fGrabParent->InheritsFrom(TGCompositeFrame::Class())) {
      TList *li = ((TGCompositeFrame*)fPimpl->fGrabParent)->GetList();
      fPimpl->fGrabListPosition = (TGFrameElement*)li->Before(frame->GetFrameElement());
      ((TGCompositeFrame*)fPimpl->fGrabParent)->RemoveFrame(frame);
   }

   SetWindowAttributes_t attr;
   attr.fMask             = kWAOverrideRedirect | kWASaveUnder;
   attr.fOverrideRedirect = kTRUE;
   attr.fSaveUnder        = kTRUE;

   gVirtualX->ChangeWindowAttributes(frame->GetId(), &attr);

   frame->UnmapWindow();
   frame->ReparentWindow(fClient->GetDefaultRoot(), fPimpl->fX0, fPimpl->fY0);
   gVirtualX->Update(1);
   frame->Move(fPimpl->fX0, fPimpl->fY0);
   frame->MapRaised();

   if (fBuilder) {
      fBuilder->Update();
      TString str = frame->ClassName();
      str += "::";
      str += frame->GetName();
      str += " is being dragged";
      fBuilder->UpdateStatusBar(str.Data());
   }
}

//______________________________________________________________________________
void TGuiBldDragManager::UngrabFrame()
{
   // ungrab/unselect frame

   if (fStop || !fPimpl->fGrab) return;

   SetCursorType(kPointer);
   HideGrabRectangles();

   DoRedraw();

   if (fBuilder) {
      fBuilder->Update();
      TString str = fPimpl->fGrab->ClassName();
      str += "::";
      str += fPimpl->fGrab->GetName();
      str += " ungrabbed";
      fBuilder->UpdateStatusBar(str.Data());
   }
   fPimpl->fGrab = 0;
}

//______________________________________________________________________________
static Bool_t IsParentOfGrab(Window_t id, const TGWindow *grab)
{
   // helper for IsPointVisible

   const TGWindow *parent = grab;

   while (parent && (parent != gClient->GetDefaultRoot())) {
      if (parent->GetId() == id) {
         return kTRUE;
      }
      parent = parent->GetParent();
   }
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::IsPointVisible(Int_t xi, Int_t yi)
{
   // helper function for IsSelectedWindow method

   Window_t w = gVirtualX->GetDefaultRootWindow();
   Window_t src, dst, child;
   Int_t x = xi;
   Int_t y = yi;
   Bool_t ret = kFALSE;

#ifdef DEBUG_LOCAL
printf("IsPointVisible1\n");
#endif

   gVirtualX->TranslateCoordinates(fPimpl->fGrab->GetId(), w, x, y, x, y, child);

#ifdef DEBUG_LOCAL
printf("IsPointVisible2\n");
#endif

   dst = src = child = w;

   while (child) {
      src = dst;
      dst = child;
      gVirtualX->TranslateCoordinates(src, dst, x, y, x, y, child);

      if (IsParentOfGrab(child, fPimpl->fGrab)) {
         return kTRUE;
      }
   }

#ifdef DEBUG_LOCAL
printf("IsPointVisible3\n");
#endif

   return ret;
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::IsSelectedVisible()
{
   // return kTRUE if grabbed/selected frame is not overlapped

   if (fStop || !fPimpl->fGrab || !fClient->IsEditable()) return kFALSE;

   // popup menu was placed
   if (fPimpl->fPlacePopup) {
      return kTRUE;
   }

   static Long_t was = gSystem->Now();
   static Bool_t visible = kFALSE;

   Long_t now = (long)gSystem->Now();

   if (now-was < 200) {
      return visible;
   }
   was = now;

   visible = kFALSE;

   if (!IsPointVisible(2, 2)) {
      return visible;
   }

   if (!IsPointVisible(2, fPimpl->fGrab->GetHeight()-2)) {
      return visible;
   }

   if (!IsPointVisible(fPimpl->fGrab->GetWidth()-2, 2)) {
      return visible;
   }

   if (!IsPointVisible(fPimpl->fGrab->GetWidth()-2, 
                       fPimpl->fGrab->GetHeight()-2)) {
      return visible;
   }

   visible = kTRUE;

   return visible;
}

//______________________________________________________________________________
void TGuiBldDragManager::DrawGrabRectangles(TGWindow *win)
{
   // draw grab small rectangles around grabbed/selected/frame

   if (fStop) return;

   TGFrame *frame = win ? (TGFrame *)win : fPimpl->fGrab;

   if (!frame || !fClient->IsEditable() || fPimpl->fPlacePopup) return;

   Window_t w = gVirtualX->GetDefaultRootWindow();
   Window_t c; Int_t x, y;

   gVirtualX->TranslateCoordinates(frame->GetId(), w,  0, 0, x, y, c);

   if (frame->InheritsFrom(TGCompositeFrame::Class()) && 
       CanChangeLayout(frame) && !frame->IsLayoutBroken()) {
      fPimpl->fAroundFrame[0]->MoveResize(x-3, y-3, frame->GetWidth()+6, 2);
      fPimpl->fAroundFrame[0]->MapRaised();
      fPimpl->fAroundFrame[1]->MoveResize(x+frame->GetWidth()+3, y-3, 2, frame->GetHeight()+6);
      fPimpl->fAroundFrame[1]->MapRaised();
      fPimpl->fAroundFrame[2]->MoveResize(x-3, y+frame->GetHeight()+2, frame->GetWidth()+6, 2);
      fPimpl->fAroundFrame[2]->MapRaised();
      fPimpl->fAroundFrame[3]->MoveResize(x-3, y-3, 2, frame->GetHeight()+6);
      fPimpl->fAroundFrame[3]->MapRaised();
   } else {
      for (int i = 0; i < 4; i++) fPimpl->fAroundFrame[i]->UnmapWindow();
   }

   // draw rectangles
   DrawGrabRect(0, x - 6, y - 6);
   DrawGrabRect(1, x + frame->GetWidth()/2 - 3, y - 6);
   DrawGrabRect(2, x + frame->GetWidth(), y - 6);
   DrawGrabRect(3, x - 6, y + frame->GetHeight());
   DrawGrabRect(4, x - 6, y + frame->GetHeight()/2 - 3);
   DrawGrabRect(5, x + frame->GetWidth(), y + frame->GetHeight()/2 - 3);
   DrawGrabRect(6, x + frame->GetWidth()/2 - 3, y + frame->GetHeight());
   DrawGrabRect(7, x + frame->GetWidth(), y + frame->GetHeight());

   fPimpl->fGrabRectHidden = kFALSE;
}

//______________________________________________________________________________
void TGuiBldDragManager::DrawGrabRect(Int_t i, Int_t x, Int_t y)
{
   // Helper method to draw grab rectangle at position x,y

   if (fStop) return;

   fPimpl->fGrabRect[i]->Move(x, y);
   fPimpl->fGrabRect[i]->MapRaised();
}

//______________________________________________________________________________
void TGuiBldDragManager::HighlightCompositeFrame(Window_t win)
{
   // raise composite frame whne mouse is moving 

   static Window_t gw = 0;

   if (fStop || !win || (win == gw)) return;

   TGWindow *w = fClient->GetWindowById(win);

   if (!w || (w == fPimpl->fPlane) || w->GetEditDisabled() || w->IsEditable() ||
       !w->InheritsFrom(TGCompositeFrame::Class())) return;

   TGFrame *frame = (TGFrame*)w;
   UInt_t opt = frame->GetOptions();

   if ((opt & kRaisedFrame) || (opt & kSunkenFrame)) return;

   gw = win;
   if (fPimpl->fPlane) {
      fPimpl->fPlane->ChangeOptions(fPimpl->fPlane->GetOptions() & ~kRaisedFrame);
      fClient->NeedRedraw(fPimpl->fPlane, kTRUE);
   }
   fPimpl->fPlane = frame;
   fPimpl->fPlane->ChangeOptions(opt | kRaisedFrame);
   fClient->NeedRedraw(fPimpl->fPlane, kTRUE);

   if (fBuilder) {
      TString str = frame->ClassName();
      str += "::";
      str += frame->GetName();
      fBuilder->UpdateStatusBar(str.Data());
   }
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::HandleTimer(TTimer *t)
{
   // The main event loop is  originated here
   // It repeadeatly queries pointer state and position on the screen. 
   // From this info an Event_t structure is built. 

   // if nothing is editted stop timer and reset everything
   if (!fClient || !fClient->IsEditable()) { 
      SetEditable(kFALSE);
      return kFALSE;
   }

   if (!IsSelectedVisible()) {
      UngrabFrame();
   }

   Window_t dum;
   Event_t ev;
   ev.fCode = kButton1;
   ev.fType = kMotionNotify;
   ev.fState = 0;

   gVirtualX->QueryPointer(gVirtualX->GetDefaultRootWindow(), dum, dum,
                           ev.fXRoot, ev.fYRoot, ev.fX, ev.fY, ev.fState);

   static Int_t gy = 0;
   static Int_t gx = 0;
   static UInt_t gstate = 0;
   static Window_t gw = 0;

   ev.fWindow = GetWindowFromPoint(ev.fXRoot, ev.fYRoot);

   if (ev.fWindow && (gw == ev.fWindow) && (gstate == ev.fState) &&
       (ev.fYRoot == gy) && (ev.fXRoot == gx)) {
      return kFALSE;
   }

   gw = ev.fWindow;
   gstate = ev.fState;
   ev.fState &= ~16;    // ignore "num lock" pressed
   ev.fState &= ~2;     // ignore "caps lock" pressed

   if (!fDragging && !fMoveWaiting && !fPimpl->fButtonPressed &&
       ((ev.fState == kButton1Mask) || (ev.fState == kButton3Mask) ||
        (ev.fState == (kButton1Mask | kKeyShiftMask)) ||
        (ev.fState == (kButton1Mask | kKeyControlMask)))) {

      if (ev.fState & kButton1Mask) ev.fCode = kButton1;
      if (ev.fState & kButton3Mask) ev.fCode = kButton3;

      ev.fType = kButtonPress;
      t->SetTime(40);

      if (fPimpl->fPlane && fClient->GetWindowById(fPimpl->fPlane->GetId())) {
         fPimpl->fPlane->ChangeOptions(fPimpl->fPlane->GetOptions() & ~kRaisedFrame);
         fClient->NeedRedraw(fPimpl->fPlane, kTRUE);
      } else {
         fPimpl->fPlane = 0;
      }

      Bool_t ret = HandleButtonPress(&ev);
      return ret;
   }

   if ((fDragging || fMoveWaiting) && (!ev.fState || (ev.fState == kKeyShiftMask)) &&
       fPimpl->fButtonPressed) {

      ev.fType = kButtonRelease;
      t->SetTime(100);

      return HandleButtonRelease(&ev);
   }

   fPimpl->fButtonPressed = (ev.fState & kButton1Mask) ||
                            (ev.fState & kButton2Mask) ||
                            (ev.fState & kButton3Mask);

   if ((ev.fYRoot == gy) && (ev.fXRoot == gx)) return kFALSE;

   gy = ev.fYRoot;
   gx = ev.fXRoot;

   if (!fMoveWaiting && !fDragging && !ev.fState) {
      if (!CheckDragResize(&ev) && fClient->GetWindowById(ev.fWindow)) {
         HighlightCompositeFrame(ev.fWindow);
      }
   } else if (ev.fState & kButton1Mask) {
      HandleMotion(&ev);
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::RecognizeGesture(Event_t *event, TGFrame *frame)
{
   // recognize what is done after mouse button pressed

   if (fStop) return kFALSE;

   if (((event->fCode != kButton1) && (event->fCode != kButton3)) ||
       !frame || !fClient->IsEditable()) {
      return kFALSE;
   }

   // handle context menu
   if (event->fCode == kButton3 ) {
      TGFrame *context_fr = fPimpl->fSpacePressedFrame ? fPimpl->fSpacePressedFrame : frame;

      if (!fPimpl->fSpacePressedFrame) SelectFrame(frame);
      HandleButon3Pressed(event, context_fr);
      return kTRUE;
   }

   // Ctrl with pointer switches editable frame 
   if (((event->fCode == kButton1) && (event->fState & kKeyControlMask)) ||
             ((event->fCode == kButton1) && fBuilder && fBuilder->IsSelectMode())) {
      SwitchEditable(frame);
      return kTRUE;
   }

   fDragType = kDragNone;

   if (!fSelectionIsOn) {
      fPimpl->fX0 = event->fXRoot;
      fPimpl->fY0 = event->fYRoot;
   }

   //HideGrabRectangles();
   fPimpl->fClickFrame = frame;

   if (fBuilder && fBuilder->IsExecutable() &&
       frame->InheritsFrom(TGCompositeFrame::Class())) {
      UngrabFrame();
      frame->SetEditable(kTRUE);
      fSource = 0;
      fDragType = kDragLasso;
      goto out;
   }

   if (event->fState & kKeyShiftMask) {
      // drag grabbed frame with shift key pressed should create a copy of grabbed frame
      // move a copy of editable and selected frame
      if (frame == fPimpl->fGrab) {
         fSource = frame;
         fDragType = kDragCopy;
         gVirtualX->SetCursor(frame->GetId(), gVirtualX->CreateCursor(kMove));
         goto out;
      }

      // otherwise do lasso selection
      if (!fSelectionIsOn) {
         fSelectionIsOn = kTRUE;
      } else {
         fPimpl->fX = event->fXRoot;
         fPimpl->fY = event->fYRoot;
         fDragType = kDragLasso;

         DrawLasso();
         return kTRUE;
      }
   }
  
   CheckDragResize(event);

   if (frame->IsEditable()) {
      fSource = 0;

      if (fDragType != kDragResize) {

         // move editable and selected frame
         if (frame == fPimpl->fGrab) {
            fSource = frame;
            fDragType = kDragMove;
            gVirtualX->SetCursor(frame->GetId(), gVirtualX->CreateCursor(kMove));
            goto out;
         }

         fDragType = kDragLasso;
      }
   } else if ((fDragType != kDragResize) && !fPimpl->fSpacePressedFrame) {

      fSource = frame;
      SelectFrame(frame, event->fState & kKeyShiftMask);
   }

   if ((fDragType == kDragNone) && !fPimpl->fSpacePressedFrame) {
      SwitchEditable(frame);
      fSource = 0;

      // try again
      CheckDragResize(event);

      if (fDragType == kDragNone) {
         return kFALSE;
      }
   }

out:

   Window_t c;

   gVirtualX->TranslateCoordinates(fClient->GetDefaultRoot()->GetId(),
                                   frame->GetId(),
                                   event->fXRoot, event->fYRoot,
                                   fPimpl->fXf, fPimpl->fYf, c);
   fPimpl->fX = event->fXRoot;
   fPimpl->fY = event->fYRoot;

   fMoveWaiting = kTRUE;
   DoRedraw();

   return kTRUE;
}

//______________________________________________________________________________
void TGuiBldDragManager::HandleButon3Pressed(Event_t *event, TGFrame *frame)
{
   // handle 3d mouse pressed (show context menu)

   if (fStop) return;

   if (fClient->GetWaitForEvent() == kUnmapNotify) {
      return;
   }

   if (frame == fPimpl->fGrab) {
      Menu4Frame(frame, event->fXRoot, event->fYRoot);
   } else if (frame->IsEditable())  {
      if (fLassoDrawn) {
         Menu4Lasso(event->fXRoot, event->fYRoot);
      } else {
         Menu4Frame(frame, event->fXRoot, event->fYRoot);
      }
   } else {
      TGFrame *base = InEditable(frame->GetId());
      if (base) {
         //SelectFrame(base);
         Menu4Frame(base, event->fXRoot, event->fYRoot);
      } else {
         Menu4Frame(frame, event->fXRoot, event->fYRoot);
      }
   }
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::HandleButton(Event_t *event)
{
   // handle button event occured in some ROOT frame

   if (fStop) return kFALSE;

   if (event->fCode != kButton3) CloseMenus();

   if (event->fType == kButtonPress) return HandleButtonPress(event);
   else return HandleButtonRelease(event);
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::HandleConfigureNotify(Event_t *event)
{
   // resize events

   if (fStop) return kFALSE;

   TGWindow *w = fClient->GetWindowById(event->fWindow);

   if (!w) return kFALSE;

   fPimpl->fCompacted = kFALSE;
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::HandleExpose(Event_t *event)
{
   // repaint events

   if (fStop) return kFALSE;

   static Long_t was = gSystem->Now();
   static Window_t win = 0;
   Long_t now = (long)gSystem->Now();

   if (event->fCount || (win == event->fWindow) || (now-was < 40) || fDragging) {
      if (fDragging) HideGrabRectangles();
      return kFALSE;
   }

   if (gMenuDialog) {
      HideGrabRectangles();
      gMenuDialog->RaiseWindow();
      return kTRUE;
   }

   if (fLassoDrawn) {
      DrawLasso();
   } else {
      DrawGrabRectangles();
   }
   win = event->fWindow;
   was = now;

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::HandleEvent(Event_t *event)
{
   // Handle all events.

   if (fStop) return kFALSE;

   if (IgnoreEvent(event)) return kFALSE;

   switch (event->fType) {

      case kExpose:
         return HandleExpose(event);

      case kConfigureNotify:
         while (gVirtualX->CheckEvent(fId, kConfigureNotify, *event))
            ;
         return HandleConfigureNotify(event);

      case kGKeyPress:
      case kKeyRelease:
         return HandleKey(event);

      case kFocusIn:
      case kFocusOut:
         //HandleFocusChange(event);
         break;

      case kButtonPress:
         {
            Int_t dbl_clk = kFALSE;

            static Window_t gDbw = 0;
            static Long_t gLastClick = 0;
            static UInt_t gLastButton = 0;
            static Int_t gDbx = 0;
            static Int_t gDby = 0;

            if ((event->fTime - gLastClick < 350) &&
                (event->fCode == gLastButton) &&
                (TMath::Abs(event->fXRoot - gDbx) < 6) &&
                (TMath::Abs(event->fYRoot - gDby) < 6) &&
                (event->fWindow == gDbw)) {
               dbl_clk = kTRUE;
            }

            if (dbl_clk) {
               if (event->fState & kKeyControlMask) {
                  HandleAction(kEndEditAct);
                  return kTRUE;
               } else if (!(event->fState & 0xFF)) {
                  ExecuteQuickAction(event);
                  return kTRUE;
               }
            } else {
               gDbw = event->fWindow;
               gLastClick = event->fTime;
               gLastButton = event->fCode;
               gDbx = event->fXRoot;
               gDby = event->fYRoot;

               Bool_t ret = HandleButtonPress(event);
               return ret;
            }

            return kFALSE;
         }

      case kButtonRelease:
         return HandleButtonRelease(event);

      case kEnterNotify:
      case kLeaveNotify:
         //HandleCrossing(event);
         break;

      case kMotionNotify:
         while (gVirtualX->CheckEvent(fId, kMotionNotify, *event))
            ;
         return HandleMotion(event);

      case kClientMessage:
         HandleClientMessage(event);
         break;

      case kSelectionNotify:
         //HandleSelection(event);
         break;

      case kSelectionRequest:
         //HandleSelectionRequest(event);
         break;

      case kSelectionClear:
         //HandleSelectionClear(event);
         break;

      case kColormapNotify:
         //HandleColormapChange(event);
         break;

      default:
         //Warning("HandleEvent", "unknown event (%#x) for (%#x)", event->fType, fId);
         break;
   }

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::HandleDoubleClick(Event_t *)
{
   // double clicked is handled here (never should happen)

   if (fStop) return kFALSE;

   return kFALSE;
}

//______________________________________________________________________________
TGFrame *TGuiBldDragManager::GetBtnEnableParent(TGFrame *fr)
{
   //

   TGWindow *parent = fr;

   while (parent && (parent != fClient->GetDefaultRoot())) {
      if (parent->GetEditDisabled() & kEditDisableBtnEnable) {
         return (TGFrame*)parent;
      }
      parent = (TGWindow*)parent->GetParent();
   }
   return 0;
}

//______________________________________________________________________________
void TGuiBldDragManager::UnmapAllPopups()
{
   // unmap all popups 

   TList *li = fClient->GetListOfPopups();
   if (!li->GetEntries()) return;

   TGPopupMenu *pup; 
   TIter next(li);

   while ((pup = (TGPopupMenu*)next())) {
      pup->UnmapWindow();
      fClient->ResetWaitFor(pup);
   }
   gVirtualX->GrabPointer(0, 0, 0, 0, kFALSE);
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::HandleButtonPress(Event_t *event)
{
   // handle button press event

   if (fStop) return kFALSE;

   fPimpl->fButtonPressed = kTRUE;
   fPimpl->fPlacePopup = kFALSE;

   if (fPimpl->fPlane) {
      fPimpl->fPlane->ChangeOptions(fPimpl->fPlane->GetOptions() & ~kRaisedFrame);
      fClient->NeedRedraw(fPimpl->fPlane, kTRUE);
   }

   if (fQuickHandler && fQuickHandler->fEditor) { // keep quick editor on the top 
      fQuickHandler->fEditor->RaiseWindow();
      fQuickHandler->fEditor->RequestFocus();
   }

   if ( ((event->fCode != kButton1) && (event->fCode != kButton3)) ||
        (event->fType != kButtonPress) || IgnoreEvent(event)) {
      return kFALSE;
   }

   Reset1();
   //HideGrabRectangles();

   Window_t w = GetWindowFromPoint(event->fXRoot, event->fYRoot);
   TGFrame *fr = 0;

   if (w) {
      fr = (TGFrame*)fClient->GetWindowById(w);
      if (!fr) {
         return kFALSE;
      }

      //fr->HandleButton(event);
      if (!IsEventsDisabled(fr)) {
         TGFrame *btnframe = GetBtnEnableParent(fr);
         if (btnframe) {
            event->fUser[0] = fr->GetId();
            btnframe->HandleButton(event);
         }
      }
      if (IsGrabDisabled(fr)) {
         fr = GetEditableParent(fr);
      } 
      if (!fr) {
         return kFALSE;
      }

   } else {
      return kFALSE;
   }

   return RecognizeGesture(event, fr);
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::HandleButtonRelease(Event_t *event)
{
   // handle button release event

   if (fStop) return kFALSE;

   // unmap all waiting popups
   if (fClient->GetWaitForEvent() == kUnmapNotify) {
     UnmapAllPopups();
   }

   TGWindow *w = fClient->GetWindowById(event->fWindow);

   if (w && !IsEventsDisabled(w)) {
      TGFrame *btnframe = GetBtnEnableParent((TGFrame*)w);
      if (btnframe) {
         event->fUser[0] = w->GetId();
         btnframe->HandleButton(event);
      }
   }

   fPimpl->fButtonPressed = kFALSE;
   gVirtualX->SetCursor(fClient->GetRoot()->GetId(), gVirtualX->CreateCursor(kPointer));
   EndDrag();
   fSelectionIsOn &= (event->fState & kKeyShiftMask);

   if (fLassoDrawn) {
      DrawLasso();
      return kTRUE;
   }

   if (fPimpl->fClickFrame && !fSelectionIsOn) {

      // make editable the clicked frame if no lasso was drawn
      if ((fPimpl->fClickFrame == fPimpl->fGrab) && 
           !fPimpl->fGrab->IsEditable()) {
         SwitchEditable(fPimpl->fClickFrame);
         return kTRUE;

      // select/grab clicked frame if there s no grab frame
      } else if (!fPimpl->fGrab || (fPimpl->fClickFrame != fPimpl->fGrab)) {

         SelectFrame(fPimpl->fClickFrame);
         return kTRUE;
      }

   }

   SelectFrame(fPimpl->fGrab);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::HandleKey(Event_t *event)
{
   // handle key event

   if (fStop) return kFALSE;

   static const char *gSaveMacroTypes[] = { "Macro files", "*.C",
                                            "All files",   "*",
                                            0,             0 };
   char   tmp[10];
   UInt_t keysym;
   Bool_t ret = kFALSE;
   TGFileInfo fi;
   static TString dir(".");
   static Bool_t overwr = kFALSE;
   const char *fname;

   TGWindow *w = fClient->GetWindowById(GetWindowFromPoint(event->fXRoot, event->fYRoot));

   if ((event->fType != kGKeyPress) || !w) return kFALSE;

   if (IsEditDisabled(w) && !((TGFrame*)w)->HandleKey(event)) {
      TGFrame *parent = GetEditableParent((TGFrame*)w);
      if (!parent) {
         event->fWindow = parent->GetId();
         parent->HandleKey(event);
         return kTRUE;
      }
   }

   fPimpl->fSpacePressedFrame = 0;

   if (fPimpl->fPlane) {
      fPimpl->fPlane->ChangeOptions(fPimpl->fPlane->GetOptions() & ~kRaisedFrame);
      fClient->NeedRedraw(fPimpl->fPlane, kTRUE);
   }

   CloseMenus();

   fi.fFileTypes = gSaveMacroTypes;
   fi.fIniDir    = StrDup(dir);
   fi.fOverwrite = overwr;

   gVirtualX->LookupString(event, tmp, sizeof(tmp), keysym);

   if (event->fState & kKeyControlMask) {

      switch ((EKeySym)keysym & ~0x20) {
         case kKey_Return:
         case kKey_Enter:
            HandleReturn(kTRUE);
            ret = kTRUE;
            break;
         case kKey_X:
            HandleCut();
            ret = kTRUE;
            break;
         case kKey_C:
            HandleCopy();
            ret = kTRUE;
            break;
         case kKey_V:
            if (fPimpl->fClickFrame && !fPimpl->fClickFrame->IsEditable()) {
               fPimpl->fClickFrame->SetEditable(kTRUE);
            }
            HandlePaste();
            ret = kTRUE;
            break;
         case kKey_B:
         {
            if (fPimpl->fGrab && (fPimpl->fClickFrame != fClient->GetRoot())) {
               if (fPimpl->fGrab->InheritsFrom(TGCompositeFrame::Class())) {
                  BreakLayout();
               }
            }
            ret = kTRUE;
            break;
         }
         case kKey_L:
         {
            if (fPimpl->fGrab && (fPimpl->fClickFrame != fClient->GetRoot())) {
               Compact(kFALSE);
            } else {
               Compact(kTRUE);
            }
            ret = kTRUE;
            break;
         }
         case kKey_R:
            HandleReplace();
            ret = kTRUE;
            break;
         case kKey_S:
            Save();
            ret = kTRUE;
            break;
         case kKey_G:
            HandleGrid();
            ret = kTRUE;
            break;
         case kKey_H:
            SwitchLayout();
            ret = kTRUE;
            break;
         case kKey_N:
            if (fBuilder) {
               fBuilder->NewProject();
            } else {
               TGMainFrame *main = new TGMainFrame(fClient->GetDefaultRoot(), 300, 300);
               main->MapRaised();
               main->SetEditable(kTRUE);
            }
            ret = kTRUE;
            break;
         case kKey_O:
            if (fBuilder) {
               fBuilder->NewProject();
            } else {
               TGMainFrame *main = new TGMainFrame(fClient->GetDefaultRoot(), 300, 300);
               main->MapRaised();
               main->SetEditable(kTRUE);
            }
            new TGFileDialog(fClient->GetDefaultRoot(), this, kFDSave, &fi);

            if (!fi.fFilename) return kTRUE;
            dir = fi.fIniDir;
            overwr = fi.fOverwrite;
            fname = gSystem->BaseName(gSystem->UnixPathName(fi.fFilename));

            if (strstr(fname, ".C")) {
               gROOT->Macro(fname);
            } else {
               Int_t retval;
               new TGMsgBox(fClient->GetDefaultRoot(), this, "Error...",
                            Form("file (%s) must have extension .C", fname),
                            kMBIconExclamation, kMBRetry | kMBCancel, &retval);
               if (retval == kMBRetry) {
                  HandleKey(event);
               }
            }
            ret = kTRUE;
            break;
         default:
            break;
      }
   } else {
      switch ((EKeySym)keysym ) {
         case kKey_Delete:
         case kKey_Backspace:
            HandleDelete(event->fState & kKeyShiftMask);
            ret = kTRUE;
            break;
         case kKey_Return:
         case kKey_Enter:
            HandleReturn(kFALSE);
            ret = kTRUE;
            break;
         case kKey_Left:
         case kKey_Right:
         case kKey_Up:
         case kKey_Down:
            if (fLassoDrawn) {
               HandleAlignment(keysym, event->fState & kKeyShiftMask);
            } else if (fPimpl->fGrab) {
               HandleLayoutOrder((keysym == kKey_Right) || (keysym == kKey_Down));
            }
            ret = kTRUE;
            break;
         case kKey_Space:
            if (fPimpl && fPimpl->fGrab) {
               SwitchEditable(fPimpl->fGrab);

               if (fPimpl->fGrab->InheritsFrom(TGCompositeFrame::Class())) {
                  UngrabFrame();
               } else {
                  TGFrame *p = (TGFrame*)fPimpl->fGrab->GetParent();
                  SelectFrame(p);
                  fSource = fPimpl->fSpacePressedFrame = p;
               }
            }
            ret = kTRUE;
            break;
         default:
            break;
      }
   }
   if (fBuilder) {
      fBuilder->SetAction(0);
      fBuilder->Update();
   }

   if (fLassoDrawn) {
      DrawLasso();
   }

   return ret;
}

//______________________________________________________________________________
void TGuiBldDragManager::ReparentFrames(TGFrame *newfr, TGCompositeFrame *oldfr)
{
   // reparent frames

   if (fStop || !fClient->IsEditable() || (newfr == fClient->GetDefaultRoot())) { 
      return;
   }

   Int_t x0, y0, xx, yy;
   Window_t c;
   static TGLayoutHints *hints = new TGLayoutHints(kLHintsNormal, 2, 2, 2, 2);

   if (!newfr || !newfr->GetId() || !oldfr || !oldfr->GetId()) return;

   gVirtualX->TranslateCoordinates(newfr->GetId(), oldfr->GetId(),
                                   0, 0, x0, y0, c);

   x0 = x0 < 0 ? 0 : x0;
   y0 = y0 < 0 ? 0 : y0;
   Int_t x = x0 + newfr->GetWidth();
   Int_t y = y0 + newfr->GetHeight();

   TGCompositeFrame *comp = 0;

   if (newfr->InheritsFrom(TGCompositeFrame::Class())) {
      comp = (TGCompositeFrame*)newfr;
      comp->SetLayoutBroken();
   }

   TIter next(oldfr->GetList());
   TGFrameElement *el;

   while ((el = (TGFrameElement*)next())) {
      TGFrame *frame = el->fFrame;

      if ((frame->GetX() >= x0) && (frame->GetY() >= y0) &&
          (frame->GetX() + (Int_t)frame->GetWidth() <= x) &&
          (frame->GetY() + (Int_t)frame->GetHeight() <= y)) {

         if (frame == fPimpl->fGrab) UngrabFrame();

         oldfr->RemoveFrame(frame);

         gVirtualX->TranslateCoordinates(oldfr->GetId(), newfr->GetId(),
                                        frame->GetX(), frame->GetY(), xx, yy, c);

         frame->ReparentWindow(newfr, xx, yy);

         if (comp) {
            comp->AddFrame(frame, hints); // el->fLayout);
         }
      }
   }
#ifdef DEBUG_LOCAL
printf("ReparentFrames3\n");
#endif
}

//______________________________________________________________________________
TList *TGuiBldDragManager::GetFramesInside(Int_t x0, Int_t y0, Int_t x, Int_t y)
{
   // return a list of frames inside of some area

   if (fStop) return 0;

   Int_t xx, yy;

   if (!fClient->GetRoot()->InheritsFrom(TGCompositeFrame::Class())) return 0;
   TList *list = new TList();


#ifdef DEBUG_LOCAL
printf("GetFramesInside1\n");
#endif

   xx = x0; yy = y0;
   x0 = TMath::Min(xx, x); x = TMath::Max(xx, x);
   y0 = TMath::Min(yy, y); y = TMath::Max(yy, y);

   TIter next(((TGCompositeFrame*)fClient->GetRoot())->GetList());
   TGFrameElement *el;

   while ((el = (TGFrameElement*)next())) {
      if ((el->fFrame->GetX() >= x0) && (el->fFrame->GetY() >= y0) &&
          (el->fFrame->GetX() + (Int_t)el->fFrame->GetWidth() <= x) &&
          (el->fFrame->GetY() + (Int_t)el->fFrame->GetHeight() <= y)) {
         list->Add(el->fFrame);
      }
   }
   if (list->IsEmpty()) {
      delete list;
      return 0;
   }
   return list;
}

//______________________________________________________________________________
void TGuiBldDragManager::HandleReturn(Bool_t on)
{
   // Handling of  return/enter key pressing
   //
   // If on is kFALSE: 
   //    If Return or Enter key was pressed - Grab Act
   //    If lasso is  drawn - new composite frame is created and 
   //    all frames inside lasso became it's childrens.
   //
   // If on is kTRUE: 
   //    If Return or Enter key was pressed with Control Key - Drop Act,
   //    The opposite action to the Grab Act.
   //    All frames inside the grabbed frame are "dropped" into
   //    the underlying frame and the grabbed frame is deleted 

   if (fStop) return;

   Int_t x0, y0, x, y, xx, yy;
   Window_t c;
   TGCompositeFrame *parent = 0;
   TList *li = 0;

   if (!fClient->GetRoot()->InheritsFrom(TGCompositeFrame::Class()) ||
       !fClient->IsEditable()) return;

   // if grabbed frame is editable - we need to switch edit to parent 
   if (fPimpl->fGrab && fPimpl->fGrab->IsEditable()) {
      ((TGFrame*)fPimpl->fGrab->GetParent())->SetEditable(kTRUE);
   }

   TGCompositeFrame *comp = (TGCompositeFrame*)fClient->GetRoot();

   if (fLassoDrawn) {

#ifdef DEBUG_LOCAL
printf("HandleReturn1\n");
#endif

      gVirtualX->TranslateCoordinates(fClient->GetDefaultRoot()->GetId(),
                                      fClient->GetRoot()->GetId(),
                                      fPimpl->fX, fPimpl->fY, x, y, c);
      gVirtualX->TranslateCoordinates(fClient->GetDefaultRoot()->GetId(),
                                      fClient->GetRoot()->GetId(),
                                      fPimpl->fX0, fPimpl->fY0, x0, y0, c);

#ifdef DEBUG_LOCAL
printf("HandleReturn2\n");
#endif

      xx = x0; yy = y0;
      x0 = TMath::Min(xx, x); x = TMath::Max(xx, x);
      y0 = TMath::Min(yy, y); y = TMath::Max(yy, y);

      li = GetFramesInside(x0, y0, x, y);

      if (!on && li) {
         parent = new TGCompositeFrame(comp, x - x0, y - y0);
         parent->MoveResize(x0, y0, x - x0, y - y0);
         ReparentFrames(parent, comp);

#ifdef DEBUG_LOCAL
printf("HandleReturn3\n");
#endif

         comp->AddFrame(parent);
         parent->MapWindow();
         fLassoDrawn = kFALSE;
         SelectFrame(parent);

         if (fBuilder) {
            fBuilder->UpdateStatusBar("Grab action performed");
         }
      }
   } else if (on && fPimpl->fGrab) {

      // check if it is forbidden
      if (!CanChangeLayout(fPimpl->fGrab) || 
          !CanChangeLayout((TGWindow*)fPimpl->fGrab->GetParent())) {
         if (fBuilder) {
            fBuilder->UpdateStatusBar("Drop action disabled");
         }
         return;
      }

#ifdef DEBUG_LOCAL
printf("HandleReturn4\n");
#endif

      if (fPimpl->fGrab->InheritsFrom(TGCompositeFrame::Class())) {
         parent = (TGCompositeFrame*)fPimpl->fGrab;
      } else {
         //parent = (TGCompositeFrame*)fPimpl->fGrab->GetParent();
      }
      if (parent) {
         ReparentFrames(comp, parent);
         DeleteFrame(fPimpl->fGrab);
         UngrabFrame();
         ChangeSelected(0);   //update editors

#ifdef DEBUG_LOCAL
printf("HandleReturn6\n");
#endif

         if (fBuilder) {
            fBuilder->UpdateStatusBar("Drop action performed");
         }
      }
   }
   delete li;
}

//______________________________________________________________________________
void TGuiBldDragManager::HandleAlignment(Int_t to, Bool_t lineup)
{
   // align frames located inside lasso

   if (fStop) return;

   Int_t x0, y0, x, y, xx, yy;
   Window_t c;
   TGCompositeFrame *comp = 0;

   if (!fClient->GetRoot()->InheritsFrom(TGCompositeFrame::Class()) ||
       !fClient->IsEditable()) return;

   if (fLassoDrawn) {
#ifdef DEBUG_LOCAL
printf("HandleAlignment1\n");
#endif
      gVirtualX->TranslateCoordinates(fClient->GetDefaultRoot()->GetId(),
                                      fClient->GetRoot()->GetId(),
                                      fPimpl->fX, fPimpl->fY, x, y, c);
      gVirtualX->TranslateCoordinates(fClient->GetDefaultRoot()->GetId(),
                                      fClient->GetRoot()->GetId(),
                                      fPimpl->fX0, fPimpl->fY0, x0, y0, c);
#ifdef DEBUG_LOCAL
printf("HandleAlignment2\n");
#endif
      xx = x0; yy = y0;
      x0 = TMath::Min(xx, x); x = TMath::Max(xx, x);
      y0 = TMath::Min(yy, y); y = TMath::Max(yy, y);

      comp = (TGCompositeFrame*)fClient->GetRoot();

      ToGrid(x, y);
      ToGrid(x0, y0);

      TIter next(comp->GetList());
      TGFrameElement *el;
      TGFrame *prev = 0;

      while ((el = (TGFrameElement*)next())) {
         TGFrame *fr = el->fFrame;

         if ((fr->GetX() >= x0) && (fr->GetY() >= y0) &&
             (fr->GetX() + (Int_t)fr->GetWidth() <= x) &&
             (fr->GetY() + (Int_t)fr->GetHeight() <= y)) {

            switch ((EKeySym)to) {
               case kKey_Left:
                  fr->Move(x0, fr->GetY());
                  if (lineup) {
                     if (prev) fr->Move(fr->GetX(), prev->GetY() + prev->GetHeight());
                     else fr->Move(x0, y0);
                  }
                  break;
               case kKey_Right:
                  fr->Move(x - fr->GetWidth(), fr->GetY());
                  if (lineup) {
                     if (prev) fr->Move(fr->GetX(), prev->GetY() + prev->GetHeight());
                     else fr->Move(x - fr->GetWidth(), y0);
                  }
                  break;
               case kKey_Up:
                  fr->Move(fr->GetX(), y0);
                  if (lineup) {
                     if (prev) fr->Move(prev->GetX() + prev->GetWidth(), fr->GetY());
                     else fr->Move(x0, y0);
                  }
                  break;
               case kKey_Down:
                  fr->Move(fr->GetX(), y - fr->GetHeight());
                  if (lineup) {
                     if (prev) fr->Move(prev->GetX() + prev->GetWidth(), fr->GetY());
                     else fr->Move(x0, y - fr->GetHeight());
                  }
                  break;
               default:
                  break;
            }
            prev = fr;
         }
      }
   }
   if (fLassoDrawn) {
      DrawLasso();
   }
}

//______________________________________________________________________________
void TGuiBldDragManager::HandleDelete(Bool_t crop)
{
   // Handle delete or crop action
   //
   // crop is kFALSE - delete action
   //   - if lasso is drawn -> all frames inside lasso area are deleted
   //   - if frame is grabbed/selected -> the frame is deleted
   // crop is kTRUE - crop action
   //   - if lasso is drawn -> all frames outside of lasso area are deleted
   //   - if frame is grabbed/selected -> all frames except the grabbed frame are deleted
   //     In both cases the main frame is shrinked to the size of crop area.

   if (fStop) return;

   Int_t x0, y0, x, y, xx, yy, w, h;
   Window_t c;

   if (!fClient->GetRoot()->InheritsFrom(TGCompositeFrame::Class()) ||
       !fClient->IsEditable()) {
      return;
   }

   TGCompositeFrame *comp = 0;
   Bool_t fromGrab = kFALSE;
   TGFrame *frame = fPimpl->fGrab;

   if (fBuilder && crop) {
      comp = fBuilder->FindEditableMdiFrame(fClient->GetRoot());
   } else {
      comp = (TGCompositeFrame*)fClient->GetRoot();
   }

   if (frame && !CanChangeLayout((TGWindow*)frame->GetParent())) {
      frame = GetMovableParent(frame);

      if (!frame) {
         TString str = fPimpl->fGrab->ClassName();
         str += "::";
         str += fPimpl->fGrab->GetName();
         str += " cannot be deleted";

         if (fBuilder) {
            fBuilder->UpdateStatusBar(str.Data());
         }
         return;
      }
   }

   // prepare to crop grabbed frame
   if (frame && !fLassoDrawn && crop) {
      gVirtualX->TranslateCoordinates(frame->GetId(),
                                      fClient->GetDefaultRoot()->GetId(),
                                      -2, -2,
                                      fPimpl->fX0, fPimpl->fY0, c);

      fPimpl->fX = fPimpl->fX0 + frame->GetWidth()+4;
      fPimpl->fY = fPimpl->fY0 + frame->GetHeight()+4;
      fromGrab = kTRUE;
   }

   gVirtualX->TranslateCoordinates(fClient->GetDefaultRoot()->GetId(),
                                   comp->GetId(),
                                   fPimpl->fX, fPimpl->fY, x, y, c);
   gVirtualX->TranslateCoordinates(fClient->GetDefaultRoot()->GetId(),
                                   comp->GetId(),
                                   fPimpl->fX0, fPimpl->fY0, x0, y0, c);

   xx = x0; yy = y0;
   x0 = TMath::Min(xx, x); x = TMath::Max(xx, x);
   y0 = TMath::Min(yy, y); y = TMath::Max(yy, y);
   w = x - x0;
   h = y - y0;

   if (fLassoDrawn || fromGrab) {
      TIter next(comp->GetList());
      TGFrameElement *el;

      while ((el = (TGFrameElement*)next())) {
         TGFrame *fr = el->fFrame;

         if ((fr->GetX() >= x0) && (fr->GetY() >= y0) &&
             (fr->GetX() + (Int_t)fr->GetWidth() <= x) &&
             (fr->GetY() + (Int_t)fr->GetHeight() <= y)) {
            if (!crop) {
               DeleteFrame(fr);
            } else {
               fr->Move(fr->GetX() - x0, fr->GetY() - y0);
            }
         } else {
            if (crop) {
               DeleteFrame(fr);
            }
         }
      }
      if (crop && comp) {
         gVirtualX->TranslateCoordinates(comp->GetId(), comp->GetParent()->GetId(),
                                          x0, y0, xx, yy, c);

         comp->MoveResize(xx, yy, w, h);

         if (comp->GetParent()->InheritsFrom(TGMdiDecorFrame::Class())) {
            TGMdiDecorFrame *decor = (TGMdiDecorFrame *)comp->GetParent();


            gVirtualX->TranslateCoordinates(decor->GetId(), decor->GetParent()->GetId(),
                                            xx, yy, xx, yy, c);

            Int_t b = 2 * decor->GetBorderWidth();
            decor->MoveResize(xx, yy, comp->GetWidth() + b,
                              comp->GetHeight() + b + decor->GetTitleBar()->GetDefaultHeight());
         }
      }
   } else { //  no lasso drawn -> delete selected frame
      DeleteFrame(frame);
      UngrabFrame();
      ChangeSelected(0);   //update editors
   }
   fLassoDrawn = kFALSE;

   if (fBuilder) {
      fBuilder->Update();
      fBuilder->UpdateStatusBar(crop ? "Crop action performed" : "Delete action performed");
   }
}

//______________________________________________________________________________
void TGuiBldDragManager::DeleteFrame(TGFrame *frame)
{
   // delete frame

   if (fStop |!frame) {
      return;
   }

   frame->UnmapWindow();

   TGCompositeFrame *comp = 0;

   if (frame->GetParent()->InheritsFrom(TGCompositeFrame::Class())) {
      comp = (TGCompositeFrame*)frame->GetParent();
   }

   if (comp) comp->RemoveFrame(frame);

   if (frame == fPimpl->fGrab) UngrabFrame();

   fClient->UnregisterWindow(frame);
   //frame->DeleteWindow();
}

//______________________________________________________________________________
void TGuiBldDragManager::HandleCut()
{
   // handle cut

   if (fStop || !fPimpl->fGrab) return;

   // 
   fPimpl->fGrab = GetMovableParent(fPimpl->fGrab);
   HandleCopy();
   DeleteFrame(fPimpl->fGrab);
   ChangeSelected(0);   //update editors
}

//______________________________________________________________________________
void TGuiBldDragManager::HandleCopy()
{
   // handle copy

   if (fStop || !fPimpl->fGrab) return;

   TGMainFrame *tmp = new TGMainFrame(fClient->GetDefaultRoot(),
                                      fPimpl->fGrab->GetWidth(),
                                      fPimpl->fGrab->GetHeight());

   // save coordinates
   Int_t x0 = fPimpl->fGrab->GetX();
   Int_t y0 = fPimpl->fGrab->GetY();

   // save parent name
   TString name = fPimpl->fGrab->GetParent()->GetName();

   ((TGWindow*)fPimpl->fGrab->GetParent())->SetName(tmp->GetName());

   fPimpl->fGrab->SetX(0);
   fPimpl->fGrab->SetY(0);

   TGFrameElement *fe = fPimpl->fGrab->GetFrameElement();

   tmp->GetList()->Add(fe);
   tmp->SetLayoutBroken(kTRUE);

   tmp->SaveSource(fPasteFileName.Data(), "quiet");
   tmp->GetList()->Remove(fe);

   fPimpl->fGrab->SetX(x0);
   fPimpl->fGrab->SetY(y0);

   ((TGWindow*)fPimpl->fGrab->GetParent())->SetName(name.Data());

   if (fBuilder) {
      TString str = fPimpl->fGrab->ClassName();
      str += "::";
      str += fPimpl->fGrab->GetName();
      str += " copied to clipboard";
      fBuilder->UpdateStatusBar(str.Data());
   }

   delete tmp;
}

//______________________________________________________________________________
void TGuiBldDragManager::HandlePaste()
{
   // handle paste

   if (fStop) return;

   Int_t xp = 0;
   Int_t yp = 0;

   if (gSystem->AccessPathName(fPasteFileName.Data())) return;

   fPasting = kTRUE;
   gROOT->Macro(fPasteFileName.Data());

   Window_t c;

   if (!fPimpl->fReplaceOn) {
      gVirtualX->TranslateCoordinates(fClient->GetDefaultRoot()->GetId(),
                                      fClient->GetRoot()->GetId(),
                                      fPimpl->fX0, fPimpl->fY0, xp, yp, c);
      ToGrid(xp, yp);

      // fPasteFrame is a TGMainFrame consisting "the frame to paste" 
      // into the editable frame (aka fClient->GetRoot())

      if (fPasteFrame) {
         fPasteFrame->Move(xp, yp);
         fPimpl->fGrab = fPasteFrame;
         HandleReturn(1);  // drop
      }
   }

   fPasting = kFALSE;

   if (fBuilder) {
      fBuilder->UpdateStatusBar("Paste action performed");
   }
}

//______________________________________________________________________________
void TGuiBldDragManager::DoReplace(TGFrame *frame)
{
   // replace frame (doesn't work yet properly)

   if (fStop || !frame || !fPimpl->fGrab || !fPimpl->fReplaceOn) {
      return;
   }

   Int_t w = fPimpl->fGrab->GetWidth();
   Int_t h = fPimpl->fGrab->GetHeight();
   Int_t x = fPimpl->fGrab->GetX();
   Int_t y = fPimpl->fGrab->GetY();

   if (fBuilder) {
      TString str = fPimpl->fGrab->ClassName();
      str += "::";
      str += fPimpl->fGrab->GetName();
      str += " replaced by ";
      str += frame->ClassName();
      str += "::";
      str += frame->GetName();
      fBuilder->UpdateStatusBar(str.Data());
   }

   TGFrameElement *fe = fPimpl->fGrab->GetFrameElement();

   fe->fFrame = 0;
   fPimpl->fGrab->DestroyWindow();
   delete fPimpl->fGrab;
   fPimpl->fGrab = 0;

   fe->fFrame = frame;
   frame->MoveResize(x, y, w, h);
   frame->MapRaised();
   frame->SetFrameElement(fe);

   SelectFrame(frame);
   fPimpl->fReplaceOn = kFALSE;

   TGWindow *root = (TGWindow *)fClient->GetRoot();
   root->SetEditable(kFALSE);
   DoRedraw();
   root->SetEditable(kTRUE);
}

//______________________________________________________________________________
void TGuiBldDragManager::HandleReplace()
{
   // handle replace 

   if (fStop || !fPimpl->fGrab) return;

   fPimpl->fReplaceOn = kTRUE;
   TGFrame *frame = 0;

   if (fBuilder && fBuilder->IsExecutable())  {
      frame = (TGFrame *)fBuilder->ExecuteAction();
   } else {
      HandlePaste();
      frame = fPasteFrame;
   }
   DoReplace(frame);
   fPimpl->fReplaceOn = kFALSE;
}

//______________________________________________________________________________
void TGuiBldDragManager::CloneEditable()
{
   // create a frame which is the same as currently editted frame

   if (fStop) return;

   TString tmpfile = gSystem->TempDirectory();
   tmpfile = gSystem->ConcatFileName(tmpfile.Data(),
                                     Form("tmp%d.C", gRandom->Integer(100)));
   Save(tmpfile.Data());
   gROOT->Macro(tmpfile.Data());
   gSystem->Unlink(tmpfile.Data());

   if (fClient->GetRoot()->InheritsFrom(TGFrame::Class())) {
      TGFrame *f = (TGFrame *)fClient->GetRoot();
      f->Resize(f->GetWidth() + 10, f->GetHeight() + 10);
   }
}

//______________________________________________________________________________
void TGuiBldDragManager::Save(const char *file)
{
   // save a editted frame to the file

   if (fStop || !fClient->GetRoot() || !fClient->IsEditable()) return;

   TGMainFrame *main = (TGMainFrame*)fClient->GetRoot()->GetMainFrame();
   Bool_t lbrk = main->IsLayoutBroken();
   main->SetLayoutBroken();

   TGWindow *root = (TGWindow*)fClient->GetRoot();
   TString fname = file;

   root->SetEditable(kFALSE);

   if (!file || !strlen(file)) {
      static TString dir(".");
      static Bool_t overwr = kFALSE;
      static const char *gSaveMacroTypes[] = { "Macro files", "*.C",
                                               "All files",   "*",
                                               0,             0 };
      TGFileInfo fi;

      fi.fFileTypes = gSaveMacroTypes;
      fi.fIniDir    = StrDup(dir);
      fi.fOverwrite = overwr;
      new TGFileDialog(fClient->GetDefaultRoot(), this, kFDSave, &fi);

      if (!fi.fFilename) goto out;
      dir = fi.fIniDir;
      overwr = fi.fOverwrite;
      fname = gSystem->BaseName(gSystem->UnixPathName(fi.fFilename));
   }

   if (strstr(fname.Data(), ".C")) {
      main->SaveSource(fname.Data(), file ? "quiet" : "");
   } else {
      Int_t retval;
      TString msg = Form("file (%s) must have extension .C", fname.Data());

      new TGMsgBox(fClient->GetDefaultRoot(), main, "Error...", msg.Data(),
                   kMBIconExclamation, kMBRetry | kMBCancel, &retval);

      if (retval == kMBRetry) {
         Save();
      }
   }

out:
   main->RaiseWindow();
   main->SetLayoutBroken(lbrk);
//   root->SetEditable(kTRUE);
}

//______________________________________________________________________________
void TGuiBldDragManager::DoResize()
{
   // handle resize

   if (fStop) return;

   TGFrame *fr = fPimpl->fGrab;

   if (!fr || IsFixedSize(fr) || 
        IsFixedLayout((TGWindow*)fr->GetParent())) {
      fr = (TGFrame*)GetResizableParent(fr);

      if (!fr || !fClient->IsEditable()) {
         return;
      }
   }

   TGCompositeFrame *comp = 0;

   if (fr->InheritsFrom(TGCompositeFrame::Class())) {
      comp = (TGCompositeFrame*)fr;
   }

   Window_t c;
   Int_t x = fPimpl->fX;
   Int_t y = fPimpl->fY;
   UInt_t w = 0;
   UInt_t h = 0;

   gVirtualX->TranslateCoordinates(fClient->GetDefaultRoot()->GetId(),
                                   fr->GetId(), x, y, x, y, c);

   ToGrid(x, y);

   switch (fPimpl->fResizeType) {
      case kTopLeft:
         if ((((int)fr->GetWidth() > x) || (x < 0)) &&
             (((int)fr->GetHeight() > y) || (y < 0))) {

            if (!IsFixedH(fr) && !IsFixedW(fr)) {
               fr->MoveResize(fr->GetX() + x, fr->GetY() + y,
                              fr->GetWidth() - x, fr->GetHeight() - y);
               break;
            }
            if (IsFixedH(fr)) {
               fr->MoveResize(fr->GetX() + x, fr->GetY(),
                              fr->GetWidth() - x, fr->GetDefaultHeight());
               break;
            }
            if (IsFixedW(fr)) {
               fr->MoveResize(fr->GetX(), fr->GetY() + y,
                              fr->GetDefaultWidth(), fr->GetHeight() - y);
               break;
            }
         }
         break;
      case kTopRight:
         if ((x > 0) && (((int)fr->GetHeight() > y) || (y < 0))) {
            w = !IsFixedW(fr) ? x : fr->GetDefaultWidth();
            if (!IsFixedH(fr)) { 
               fr->MoveResize(fr->GetX(), fr->GetY() + y,
                              w, fr->GetHeight() - y);
            } else {
               fr->Resize(x, fr->GetDefaultHeight());
            }
         }
         break;
      case kTopSide:
         if (((int)fr->GetHeight() > y) || (y < 0)) {
            if (IsFixedH(fr)) {
               break;
            }
            fr->MoveResize(fr->GetX(), fr->GetY() + y,
                           fr->GetWidth(), fr->GetHeight() - y);
         }
         break;
      case kBottomLeft:
         if ((((int)fr->GetWidth() > x) || (x < 0)) && (y > 0)) {
            if (!IsFixedH(fr) && !IsFixedW(fr)) {
               fr->MoveResize(fr->GetX() + x, fr->GetY(),
                              fr->GetWidth() - x, y);
               break;
            }
            if (IsFixedH(fr)) {
               fr->MoveResize(fr->GetX() + x, fr->GetY(),
                              fr->GetWidth() - x, fr->GetDefaultHeight());
               break;
            }
            if (IsFixedW(fr)) {
               fr->MoveResize(fr->GetX(), fr->GetY(),
                              fr->GetDefaultWidth(), y);
               break;
            }
         }
         break;
      case kBottomRight:
         if ((x > 0) && (y > 0)) {
            w = !IsFixedW(fr) ? x : fr->GetDefaultWidth();
            h = !IsFixedH(fr) ? y : fr->GetDefaultHeight();
            fr->Resize(w, h);
         }
         break;
      case kBottomSide:
         if (y > 0) {
            if (IsFixedH(fr)) {
               break;
            }
            fr->Resize(fr->GetWidth(), y);
         }
         break;
      case kLeftSide:
         if ((int)fr->GetWidth() > x) {
            if (IsFixedW(fr)) {
               break;
            }
            fr->MoveResize(fr->GetX() + x, fr->GetY(),
                           fr->GetWidth() - x, fr->GetHeight());
         }
         break;
      case kRightSide:
         if (x > 0) {
            if (IsFixedW(fr)) {
               break;
            }
            fr->Resize(x, fr->GetHeight());
         }
         break;
      default:
         break;
   }
   if (comp && (!comp->IsLayoutBroken() || IsFixedLayout(comp))) {
      comp->GetLayoutManager()->Layout();
   }
   gVirtualX->SetCursor(fClient->GetRoot()->GetId(),
                        gVirtualX->CreateCursor(fPimpl->fResizeType));
   fClient->NeedRedraw(fr, kTRUE);
   DoRedraw();
}

//______________________________________________________________________________
void TGuiBldDragManager::DoMove()
{
   // handle move

   if (fStop || !fPimpl->fGrab || !fClient->IsEditable()) return;

   TGWindow *parent = (TGWindow*)fPimpl->fGrab->GetParent();

   // do not remove frame from fixed layout or non-editable parent  
   if (IsFixedLayout(parent) || IsEditDisabled(parent)) {
      return;
   }

   Int_t x = fPimpl->fX - fPimpl->fXf;
   Int_t y = fPimpl->fY - fPimpl->fYf;

   fPimpl->fGrab->Move(x, y);
   CheckTargetUnderGrab();
}

//______________________________________________________________________________
TGFrame *TGuiBldDragManager::FindMdiFrame(TGFrame *in)
{
   // returns a pointer to the parent mdi frame

   if (fStop || !in) {
      return 0;
   }

   TGFrame *p = in;

   while (p && (p != fClient->GetDefaultRoot()) && 
         !p->InheritsFrom(TGMainFrame::Class())) {
      if (p->InheritsFrom(TGMdiFrame::Class())) {
         return p;
      }
      p = (TGFrame*)p->GetParent();
   }
   return 0;
}

//______________________________________________________________________________
void TGuiBldDragManager::RaiseMdiFrame(TGFrame *comp)
{
   // resize edittable mdi frame

   if (fStop || !comp) {
      return;
   }

   if (comp && comp->InheritsFrom(TGMdiFrame::Class()) && fBuilder) {
      if (fBuilder->GetMdiMain()->GetCurrent() != comp) {
         fBuilder->GetMdiMain()->SetCurrent((TGMdiFrame*)comp);
      }
   }
}

//______________________________________________________________________________
void TGuiBldDragManager::CheckTargetUnderGrab()
{
   // looks for the drop target under grabbed/selected frame while moving 

   if (fStop || !fPimpl->fGrab ) return;

   Int_t x = fPimpl->fGrab->GetX();
   Int_t y = fPimpl->fGrab->GetY();
   UInt_t w = fPimpl->fGrab->GetWidth();
   UInt_t h = fPimpl->fGrab->GetHeight();

   Bool_t ok = CheckTargetAtPoint(x - 1, y - 1);

   if (!ok) {
      CheckTargetAtPoint(x + w + 1, y + h + 1);
   }
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::CheckTargetAtPoint(Int_t x, Int_t y)
{
   // Helper. Looks for the drop target under grabbed/selected frame while moving.

   if (fStop || !fPimpl->fGrab ) return kFALSE;

   UInt_t ww = fPimpl->fGrab->GetWidth();
   UInt_t hh = fPimpl->fGrab->GetHeight();
   Bool_t ret = kFALSE;
   Window_t c;
   TGWindow *win = 0;

   Window_t w = GetWindowFromPoint(x, y);

   if (w && (w != gVirtualX->GetDefaultRootWindow())) {
      win = fClient->GetWindowById(w);
      TGCompositeFrame *comp = 0;

      if (!win) goto out;

      if (win->InheritsFrom(TGCompositeFrame::Class())) {
         comp = (TGCompositeFrame *)win;
      } else if (win->GetParent() != fClient->GetDefaultRoot()) {
         comp = (TGCompositeFrame *)win->GetParent();
      }

      if (comp) {

#ifdef DEBUG_LOCAL
printf("CheckTarget1\n");
#endif
         gVirtualX->TranslateCoordinates(fClient->GetDefaultRoot()->GetId(),
                                         comp->GetId(), x, y, x, y, c);

#ifdef DEBUG_LOCAL
printf("CheckTarget2\n");
#endif

         RaiseMdiFrame(comp);

         if ((comp != fPimpl->fGrab) && (x >= 0) && (y >= 0) &&
             (x + ww <= comp->GetWidth()) &&
             (y + hh <= comp->GetHeight())) {

            if (comp != fTarget) {
               comp->HandleDragEnter(fPimpl->fGrab);

               if (fTarget) fTarget->HandleDragLeave(fPimpl->fGrab);

               else Snap2Grid();
            } else {
               if (fTarget) {
                  fTarget->HandleDragMotion(fPimpl->fGrab);
               }
            }

            fTarget = comp;
            fTargetId = comp->GetId();
            ret = kTRUE;
            return ret;

         } else {
            if (fTarget) {
               fTarget->HandleDragLeave(fPimpl->fGrab);
            }
            fTarget = 0;
            fTargetId = 0;
         }
      }
   }

out:
   if (fTarget) fTarget->HandleDragLeave(fPimpl->fGrab);

   if (!w || !win) {
      fTarget = 0;
      fTargetId = 0;
   }
   return ret;
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::HandleMotion(Event_t *event)
{
   // handle motion event

   if (fStop) return kFALSE;

   static Long_t was = gSystem->Now();
   static Int_t gy = event->fYRoot;
   static Int_t gx = event->fXRoot;

   Long_t now = (long)gSystem->Now();

   if ((now-was < 100) || !(event->fState & kButton1Mask) ||
       ((event->fYRoot == gy) && (event->fXRoot == gx))) {
      return kFALSE;
   }
   was = now;
   gy = event->fYRoot;
   gx = event->fXRoot;

   if (!fDragging) {
      if (fMoveWaiting && ((TMath::Abs(fPimpl->fX - event->fXRoot) > 10) ||
          (TMath::Abs(fPimpl->fY - event->fYRoot) > 10))) {

         return StartDrag(fSource, event->fXRoot, event->fYRoot);
      }
   } else {

      fPimpl->fX = event->fXRoot;
      fPimpl->fY = event->fYRoot;

      switch (fDragType) {
         case kDragLasso:
            DrawLasso();
            fSelectionIsOn = event->fState & kKeyShiftMask;
            break;
         case kDragMove:
         case kDragCopy:
         case kDragLink:
            DoMove();
            break;
         case kDragResize:
            DoResize();
            break;
         default:
            break;
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
void TGuiBldDragManager::PlaceFrame(TGFrame *frame, TGLayoutHints *hints)
{
   // Put just created frame at position of the last click

   Int_t x0, y0, x, y;
   Window_t c;

   if (fStop || !frame || !fClient->IsEditable()) return;

   gVirtualX->TranslateCoordinates(fClient->GetDefaultRoot()->GetId(),
                                   fClient->GetRoot()->GetId(),
                                   fPimpl->fX0 , fPimpl->fY0, x0, y0, c);
   gVirtualX->TranslateCoordinates(fClient->GetDefaultRoot()->GetId(),
                                   fClient->GetRoot()->GetId(),
                                   fPimpl->fX , fPimpl->fY, x, y, c);

   ToGrid(x, y);
   ToGrid(x0, y0);

   UInt_t w = TMath::Abs(x - x0);
   UInt_t h = TMath::Abs(y - y0);

   w = w < frame->GetDefaultWidth() + 2 ? frame->GetDefaultWidth() + 2 : w;
   h = h < frame->GetDefaultHeight() + 2 ? frame->GetDefaultHeight() + 2: h;

   frame->Move(x > x0 ? x0 : x, y > y0 ? y0 : y);

   if (IsFixedW(frame) || IsFixedH(frame) || IsFixedSize(frame)) {
      frame->Resize(IsFixedW(frame) ? frame->GetDefaultWidth() : w,
                    IsFixedH(frame) ? frame->GetDefaultHeight() : h);
   } else {
      frame->Resize(w, h);
   }

   frame->MapRaised();
   frame->SetCleanup(kDeepCleanup);
   frame->AddInput(kButtonPressMask);

   if (fClient->GetRoot()->InheritsFrom(TGCompositeFrame::Class())) {
      TGCompositeFrame *edit = (TGCompositeFrame*)fClient->GetRoot();
      edit->SetCleanup(kDeepCleanup);
      ReparentFrames(frame, edit);
      frame->MapRaised();
      edit->SetLayoutBroken();
      UInt_t g = 2;
      edit->AddFrame(frame, hints ? hints : new TGLayoutHints(kLHintsNormal, g, g, g, g));
      if (hints) {
         edit->GetLayoutManager()->Layout();
      }
   }
   if (fBuilder) {
      TString str = frame->ClassName();
      str += "::";
      str += frame->GetName();
      str += " created";
      fBuilder->UpdateStatusBar(str.Data());
   }
}

//______________________________________________________________________________
void TGuiBldDragManager::DrawLasso()
{
   // draw lasso for allocation new object

   if (fStop || !fClient->IsEditable()) return;

   UngrabFrame();

   Int_t x0, y0, x, y;
   Window_t c;

#ifdef DEBUG_LOCAL
printf("DrawLasso1\n");
#endif

   gVirtualX->TranslateCoordinates(fClient->GetDefaultRoot()->GetId(),
                                   fClient->GetRoot()->GetId(),
                                   fPimpl->fX0 , fPimpl->fY0, x0, y0, c);
   gVirtualX->TranslateCoordinates(fClient->GetDefaultRoot()->GetId(),
                                   fClient->GetRoot()->GetId(),
                                   fPimpl->fX , fPimpl->fY, x, y, c);
#ifdef DEBUG_LOCAL
printf("DrawLasso2\n");
#endif

   ToGrid(x, y);
   ToGrid(x0, y0);
   Int_t w = TMath::Abs(x - x0);
   Int_t h = TMath::Abs(y - y0);

   DoRedraw();

   gVirtualX->DrawRectangle(fClient->GetRoot()->GetId(),
                            GetBlackGC()(), x > x0 ? x0 : x,
                            y > y0 ? y0 : y, w, h);
   gVirtualX->DrawRectangle(fClient->GetRoot()->GetId(),
                            GetBlackGC()(), x > x0 ? x0-1 : x-1,
                            y > y0 ? y0-1 : y-1, w+2, h+2);

   gVirtualX->SetCursor(fId, gVirtualX->CreateCursor(kCross));
   gVirtualX->SetCursor(fClient->GetRoot()->GetId(), gVirtualX->CreateCursor(kCross));

   fLassoDrawn = kTRUE;
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::HandleClientMessage(Event_t *event)
{
   // handle client message

   if (fStop) return kFALSE;

   if ((event->fFormat == 32) && ((Atom_t)event->fUser[0] == gWM_DELETE_WINDOW) &&
       (event->fHandle != gROOT_MESSAGE)) {

      TGMainFrame *main = (TGMainFrame*)fClient->GetRoot()->GetMainFrame();

      if (event->fWindow == main->GetId()) {

         if (main != fBuilder) {
            if (fEditor && !fEditor->IsEmbedded()) {
               delete fEditor;
               fEditor = 0;
            }
            return kFALSE;
         }

         delete fFrameMenu;
         fFrameMenu =0;

         delete fLassoMenu;
         fLassoMenu = 0;

         delete fPimpl->fGrid;
         fPimpl->fGrid = 0;
         Reset1();

      } else if (fBuilder && (event->fWindow == fBuilder->GetId())) {
         fBuilder->CloseWindow();

      } else if (fEditor && (event->fWindow == fEditor->GetMainFrame()->GetId())) {
         TQObject::Disconnect(fEditor);
         fEditor = 0;
      }
   }
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::HandleSelection(Event_t *)
{
   // not used yet.

   if (fStop) return kFALSE;

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::HandleSelectionRequest(Event_t *)
{
   // not used yet.

   if (fStop) return kFALSE;

   return kFALSE;
}

//______________________________________________________________________________
TGFrame *TGuiBldDragManager::GetMovableParent(TGWindow *p)
{
   // find parent frame which can be dragged

   if (fStop) return 0;

   TGFrame *ret = (TGFrame*)p;
   TGWindow *parent = (TGWindow*)ret->GetParent();

   while (parent && (parent != fClient->GetDefaultRoot())) {
      if (!IsFixedLayout(parent) && !IsEditDisabled(parent)) {
         return ret;
      }
      ret = (TGFrame*)parent;
      parent = (TGWindow*)ret->GetParent();
   }

   return 0;
}

//______________________________________________________________________________
TGWindow *TGuiBldDragManager::GetResizableParent(TGWindow *p)
{
   // find parent frame which can be resized

   if (fStop) return 0;

   TGWindow *parent = p;

   while (parent && (parent != fClient->GetDefaultRoot())) {
      if (!IsFixedSize(parent) && 
          !IsFixedLayout((TGWindow*)parent->GetParent()) &&
          !IsEditDisabled((TGWindow*)parent->GetParent())) {
         return parent;
      }
      parent = (TGWindow*)parent->GetParent();
   }

   return 0;
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::StartDrag(TGFrame *src, Int_t x, Int_t y)
{
   // start dragging

   if (fStop || fDragging) return kFALSE;

   TGFrame *mov = src;

   // special case when frame was grabbed via spacebar pressing
   if (fPimpl->fSpacePressedFrame) {
      if (fDragType == kDragNone) {
         fDragType = kDragMove;
         mov = fPimpl->fSpacePressedFrame;
      } else {
         fPimpl->fSpacePressedFrame = 0;
      }
   }

   TGWindow *parent = (TGWindow*)(mov ? mov->GetParent() : 0);

   // do not remove frame from fixed layout or non-editable parent  
   // try to drag "draggable parent"
   if (parent && (IsFixedLayout(parent) || IsEditDisabled(parent))) {
      mov = GetMovableParent(parent);
      if (!mov) {
         return kFALSE;
      }
   }

   SetEditable(kTRUE);  // grab server

   fPimpl->fX = x;
   fPimpl->fY = y;
   fSelectionIsOn = kFALSE;

   fPimpl->fRepeatTimer->Reset();
   gSystem->AddTimer(fPimpl->fRepeatTimer);

   fMoveWaiting = kFALSE;
   fDragging = kTRUE;
   if (src) gVirtualX->SetCursor(src->GetId(), gVirtualX->CreateCursor(kMove));

   switch (fDragType) {
      case kDragCopy:
         HandleCopy();
         HandlePaste();
         GrabFrame(fPimpl->fGrab);
         break;
      case kDragMove:
         fPimpl->fGrab = mov;
         GrabFrame(fPimpl->fGrab);
         break;
      default:
         //fPimpl->fGrab = 0;
         break;
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::EndDrag()
{
   // end dragging

   TGFrame *frame = 0;
   Bool_t ret = kFALSE;

   if (fStop) return kFALSE;

   if (fPimpl->fGrab && (fDragType >= kDragMove) && (fDragType <= kDragLink)) {

      ret = Drop();
   } else if (fBuilder && fBuilder->IsExecutable() &&
              (fDragType == kDragLasso) && !fSelectionIsOn) {

      frame = (TGFrame*)fBuilder->ExecuteAction();
      PlaceFrame(frame, fBuilder->GetAction()->fHints);
      fLassoDrawn = kFALSE;
      ret = kTRUE;
      //return ret;
   } else if ((fDragType == kDragLasso) && fSelectionIsOn) {

      HandleReturn(kFALSE);
      ret = kTRUE;
   }

   if (!fLassoDrawn) DoRedraw();

   Reset1();
   fPimpl->fSpacePressedFrame = 0;

   if (fBuilder) {
      fBuilder->SetAction(0);
      fBuilder->Update();
      if (fLassoDrawn) {
         fBuilder->UpdateStatusBar("Lasso Drawn");
      }
   }

   return ret;
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::Cancel(Bool_t /*delSrc*/)
{
   // do cancel

   if (fStop) return kFALSE;

   fTarget = 0;
   EndDrag();
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::Drop()
{
   // drop grabbed frame

   if (fStop || !fPimpl->fGrab && !fSelectionIsOn && 
       !(fDragType >= kDragMove) && (fDragType <= kDragLink)) return kFALSE;

   fDropStatus = kFALSE;
   TGFrame *frame = 0;
   TGFrame *parent = 0;
   Int_t x, y;
   Window_t c;

   switch (fDragType) {
      case kDragCopy:
      case kDragMove:
         frame = (TGFrame*)fPimpl->fGrab;
         break;
      default:
         break;
   }

   TGWindow *w = fClient->GetWindowById(fTargetId);

   if (fTarget && fPimpl->fGrab && (w == fTarget) &&  w &&
       (w != fClient->GetDefaultRoot())) {
      parent = fTarget;

      gVirtualX->TranslateCoordinates(fClient->GetDefaultRoot()->GetId(),
                                      fTarget->GetId(),
                                      fPimpl->fGrab->GetX(),
                                      fPimpl->fGrab->GetY(), x, y, c);

      fTarget->HandleDragLeave(fPimpl->fGrab);
   } else {
      parent = (TGFrame*)fPimpl->fGrabParent;
      x = fPimpl->fGrabX;
      y = fPimpl->fGrabY;
   }

   if (parent && frame && (parent != fClient->GetDefaultRoot())) {
      ToGrid(x, y);
      fDropStatus = parent->HandleDragDrop(frame, x, y, fPimpl->fGrabLayout);

      // drop was rejected
      if (!fDropStatus) {
         if (fDragType == kDragMove) {  // return dragged frame to initial position 
            parent = (TGFrame*)fPimpl->fGrabParent;
            x = fPimpl->fGrabX;
            y = fPimpl->fGrabY;
            frame = fPimpl->fGrab;

            if (parent && frame && (parent != fClient->GetDefaultRoot())) {
               fDropStatus = parent->HandleDragDrop(frame, x, y, fPimpl->fGrabLayout);
            }
         } else { // (fDragType == kDragCopy) - delete it
            DeleteFrame(frame);
         }
      }
   }

   if (fDropStatus) {
      if (fBuilder) {
         TString str = frame->ClassName();
         str += "::";
         str += frame->GetName();
         str += " dropped into ";
         str += parent->ClassName();
         str += "::";
         str += parent->GetName();
         fBuilder->UpdateStatusBar(str.Data());
      }
      fTarget = 0;
      fTargetId = 0;

      if (parent && (parent == fPimpl->fGrabParent) && fPimpl->fGrabListPosition &&
          frame && parent->InheritsFrom(TGCompositeFrame::Class())) {

         TList *li = ((TGCompositeFrame*)parent)->GetList();
         li->Remove(frame->GetFrameElement());
         li->AddAfter(fPimpl->fGrabListPosition, frame->GetFrameElement());
      }
   } else { // grab frame cannot be dropped
      if (fDragType == kDragCopy) { // dosn't work (point is not reached ???)
         HandleDelete(fPimpl->fGrab);
      } 
   }

   fPimpl->fGrabParent = 0;
   fPimpl->fGrabX = 0;
   fPimpl->fGrabY = 0;
   fPimpl->fGrabListPosition = 0;

   return fDropStatus;
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::IsMoveWaiting() const
{
   // Waits for either the mouse move from the given initial ButtonPress location
   // or for the mouse button to be released. If mouse moves away from the initial
   // ButtonPress location before the mouse button is released "IsMoveWaiting"
   // returns kTRUE. If the mouse button released before the mose moved from the
   // initial ButtonPress location, "IsMoveWaiting" returns kFALSE.

   return fMoveWaiting;
}

//______________________________________________________________________________
void TGuiBldDragManager::Compact(Bool_t global)
{
   // Layout and Resize frame.
   // If global is kFALSE - compact selected frame
   // If global is kFALSE - compact main frame of selected frame

   TGCompositeFrame *comp = 0;
   TGFrameElement *fe;

   if (fStop || !fClient || !fClient->IsEditable() || !fPimpl->fGrab) return;

   TGWindow *parent = (TGWindow*)fPimpl->fGrab->GetParent();

   if (global) {
      if (!fBuilder) {
         comp = (TGCompositeFrame*)fClient->GetRoot()->GetMainFrame();
      } else {
         comp = fBuilder->FindEditableMdiFrame(fClient->GetRoot());
         if (!comp) {
            comp = (TGCompositeFrame*)fClient->GetRoot()->GetMainFrame();
         }
      }
   } else {
      if (fPimpl->fGrab &&
          fPimpl->fGrab->InheritsFrom(TGCompositeFrame::Class())) {
         comp = (TGCompositeFrame*)fPimpl->fGrab;
      } else {
         comp = (TGCompositeFrame*)parent;
      }
   }

   if (!comp || IsFixedLayout(comp)  || IsFixedLayout(parent) || 
       IsFixedSize(comp) || IsFixedH(comp) || IsFixedW(comp)) return;

   comp->SetLayoutBroken(kFALSE);

   TIter next(comp->GetList());

   TGFrame *root = (TGFrame *)fClient->GetRoot();
   root->SetEditable(kFALSE);

   TGDimension d;

   if (global) {
      while ((fe = (TGFrameElement*)next())) {
         if (IsFixedLayout(fe->fFrame) || IsFixedSize(fe->fFrame) ||
             IsFixedH(fe->fFrame) || IsFixedW(fe->fFrame)) continue;

         fe->fFrame->SetLayoutBroken(kFALSE);
         d = fe->fFrame->GetDefaultSize();

         // avoid "to point" resizing
         if ((d.fWidth > 10) && (d.fHeight > 10)) {
            fe->fFrame->Resize();
         } else if (d.fWidth > 10) {
            fe->fFrame->Resize(d.fWidth, 10);
         } else if (d.fHeight > 10) {
            fe->fFrame->Resize(10, d.fHeight);
         } else {
            fe->fFrame->Resize(10, 10);
         }
      }
      if (!IsFixedLayout(root)) root->SetLayoutBroken(kFALSE);
      fPimpl->fCompacted = kTRUE;
   }

   if (!IsFixedLayout(comp)) {
      comp->SetLayoutBroken(kFALSE);
      d = comp->GetDefaultSize();

      // avoid "to point" resizing
      if ((d.fWidth > 10) && (d.fHeight > 10)) {
         comp->Resize();
      } else if (d.fWidth > 10) {
         comp->Resize(d.fWidth, 10);
      } else if (d.fHeight > 10) {
         comp->Resize(10, d.fHeight);
      } else {
         comp->Resize(10, 10);
      }
   }

   if (comp->GetParent()->InheritsFrom(TGMdiDecorFrame::Class())) {
      TGMdiDecorFrame *decor = (TGMdiDecorFrame *)comp->GetParent();
      Int_t b = 2 * decor->GetBorderWidth();
      decor->MoveResize(decor->GetX(), decor->GetY(), comp->GetDefaultWidth() + b,
                        comp->GetDefaultHeight() + b + decor->GetTitleBar()->GetDefaultHeight());
   }

   root->SetEditable(kTRUE);
   SelectFrame(comp);
   DoRedraw();
}

//______________________________________________________________________________
void TGuiBldDragManager::SetEditable(Bool_t on)
{
   // grab server

   static Bool_t gon = kFALSE;
   static const TGWindow *gw = 0;

   if ((gon == on) && (fClient->GetRoot() == gw)) return;
   gon = on;  gw = fClient->GetRoot();

   if (on) {
      fStop = kFALSE;

      if (fPimpl->fRepeatTimer) {
         fPimpl->fRepeatTimer->Reset();
      } else {
         fPimpl->fRepeatTimer = new TGuiBldDragManagerRepeatTimer(this, 100);
      }
      gSystem->AddTimer(fPimpl->fRepeatTimer);
      ((TGFrame*)fClient->GetRoot())->AddInput(kKeyPressMask | kButtonPressMask);

      Snap2Grid();
   } else {
      HideGrabRectangles();

      if (fPimpl->fRepeatTimer) {
         fPimpl->fRepeatTimer->Remove();
      }
      fPimpl->fGrab = 0;
      
      delete fPimpl->fGrid;
      fPimpl->fGrid = 0;

      fPimpl->ResetParams();

      TGWindow *root = (TGWindow*)fClient->GetRoot();
      if (root) {
         fClient->SetRoot(0);
      }

      if (!gSystem->AccessPathName(fPasteFileName.Data())) {
         gSystem->Unlink(fPasteFileName.Data());
      }

      if (fBuilder) {
         fBuilder->Update();
      }
      CloseMenus();

      fStop = kTRUE;
   }

   if (on && fClient->IsEditable()) {
      gVirtualX->SetCursor(fClient->GetRoot()->GetId(),
                           gVirtualX->CreateCursor(kPointer));
   }
}

//______________________________________________________________________________
void TGuiBldDragManager::ToGrid(Int_t &x, Int_t &y)
{
   // return grid coordinates which are close to given

   UInt_t step = GetGridStep();
   x = x - x%step;
   y = y - y%step;
}

//______________________________________________________________________________
void TGuiBldDragManager::HandleAction(Int_t act)
{
   // main handler of actions

   fPimpl->fLastPopupAction = act;

   switch ((EActionType)act) {
      case kPropertyAct:
         CreatePropertyEditor();
         break;
      case kEditableAct:
         if (fPimpl->fSaveGrab) fPimpl->fSaveGrab->SetEditable(kTRUE);
         if (fBuilder) {
            fBuilder->HandleMenu(kGUIBLD_FILE_START);
         }
         break;
      case kCutAct:
         HandleCut();
         break;
      case kCopyAct:
         HandleCopy();
         break;
      case kPasteAct:
         HandlePaste();
         break;
      case kCropAct:
         HandleDelete(kTRUE);
         break;
      case kCompactAct:
         Compact(kFALSE);
         break;
      case kCompactGlobalAct:
         Compact(kTRUE);
         break;
      case kDropAct:
         HandleReturn(kTRUE);
         break;
      case kLayUpAct:
         HandleLayoutOrder(kFALSE);
         break;
      case kLayDownAct:
         HandleLayoutOrder(kTRUE);
         break;
      case kCloneAct:
         CloneEditable();
         break;
      case kGrabAct:
         HandleReturn(kFALSE);
         break;
      case kDeleteAct:
         HandleDelete(kFALSE);
         break;
      case kLeftAct:
         HandleAlignment(kKey_Left);
         break;
      case kRightAct:
         HandleAlignment(kKey_Right);
         break;
      case kUpAct:
         HandleAlignment(kKey_Up);
         break;
      case kDownAct:
         HandleAlignment(kKey_Down);
         break;
      case kEndEditAct:
         SetEditable(kFALSE);
         ((TGWindow*)fClient->GetRoot())->SetEditable(kFALSE);
         if (fBuilder) {
            fBuilder->HandleMenu(kGUIBLD_FILE_STOP);
         }
         break;
      case kReplaceAct:
         HandleReplace();
         break;
      case kGridAct:
         HandleGrid();
         break;
      case kBreakLayoutAct:
         BreakLayout();
         break;
      case kSwitchLayoutAct:
      case kLayoutVAct:
      case kLayoutHAct:
         SwitchLayout();
         break;
      case kNewAct:
         if (fBuilder) {
            fBuilder->NewProject();
         } else {
            TGMainFrame *main = new TGMainFrame(fClient->GetDefaultRoot(), 300, 300);
            main->MapRaised();
            main->SetEditable(kTRUE);
         }
         break;
      case kOpenAct:
         if (fBuilder) {
            fBuilder->OpenProject();
         } else {
            TGMainFrame *main = new TGMainFrame(fClient->GetDefaultRoot(), 300, 300);
            main->MapRaised();
            main->SetEditable(kTRUE);
         }
         break;
      case kSaveAct:
         if (fBuilder) {
            if (fBuilder->FindEditableMdiFrame(fClient->GetRoot()) || 
                (!fClient->IsEditable() && fBuilder->GetMdiMain()->GetCurrent())) {
               fBuilder->SaveProject();
            } else {
               Save();
            }
         } else {
            Save();
         }
         break;
      default:
         break;
   }

   fPimpl->fPlacePopup = kFALSE;

   if (fBuilder) {
      fBuilder->SetAction(0);
      fBuilder->Update();
   }

   if (fPimpl->fSaveGrab) {
      fClient->NeedRedraw(fPimpl->fSaveGrab, kTRUE);
   }

   DoRedraw();
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::CanChangeLayout(TGWindow *w) const
{
   //  kTRUE - if it's possible to switch disable/enable layout

   return (!(w->GetEditDisabled() & kEditDisable) &&
           !IsFixedLayout(w) && w->InheritsFrom(TGCompositeFrame::Class()));
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::CanChangeLayoutOrder(TGWindow *w) const
{
   // kTRUE - if it's possible to change layout order in the parent's layout of window w

   return (w->GetParent()->InheritsFrom(TGCompositeFrame::Class()) &&
           !((TGCompositeFrame*)w->GetParent())->IsLayoutBroken() && 
           !IsFixedLayout((TGWindow*)w->GetParent()));
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::CanCompact(TGWindow *w) const
{
   // kTRUE is frame could be compacted/"layouted"

   return CanChangeLayout(w);
/*
   return (!IsFixedLayout(w) && 
           w->InheritsFrom(TGCompositeFrame::Class()) && 
           ((TGCompositeFrame*)w)->IsLayoutBroken() &&
           !IsEditDisabled((TGWindow*)w->GetParent()) &&
           !IsFixedLayout((TGWindow*)w->GetParent()));
*/
}

//______________________________________________________________________________
void TGuiBldDragManager::CreatePropertyEditor()
{
   // create widget property editor (it could be done outside of guibuilder)

//   if (!fPimpl->fClickFrame) return;

   TGWindow *root = (TGWindow*)fClient->GetRoot();
   root->SetEditable(kFALSE);

   fBuilder = (TRootGuiBuilder*)TRootGuiBuilder::Instance();

   fBuilder->Move(fPimpl->fX0, fPimpl->fY0);
   fBuilder->SetWMPosition(fPimpl->fX0, fPimpl->fY0);
   SetPropertyEditor(fBuilder->GetEditor());

   root->SetEditable(kTRUE);
}

//______________________________________________________________________________
void TGuiBldDragManager::SetPropertyEditor(TGuiBldEditor *e)
{
   // helper method

   fEditor = e;

   if (!fEditor) return;

   ChangeSelected(fPimpl->fClickFrame);
   fEditor->Connect("UpdateSelected(TGFrame*)", "TGuiBldDragManager", this,
                    "HandleUpdateSelected(TGFrame*)");
}

//______________________________________________________________________________
void TGuiBldDragManager::HandleLayoutOrder(Bool_t forward)
{
   // change layout order

   if (fStop || !fPimpl->fGrab || !fPimpl->fGrab->GetFrameElement() || 
       !CanChangeLayoutOrder(fPimpl->fGrab)) {
      return;
   }

   TGCompositeFrame *comp = (TGCompositeFrame*)fPimpl->fGrab->GetParent();
   TList *li = comp->GetList();
   TGFrameElement *fe = fPimpl->fGrab->GetFrameElement();

   if (!fe) return;

   TGFrame *frame;
   TGFrameElement *el;

   if (forward) {
      el = (TGFrameElement *)li->After(fe);
      if (!el) return;
      frame = el->fFrame;

      el->fFrame = fPimpl->fGrab;
      fPimpl->fGrab->SetFrameElement(el);
      fe->fFrame = frame;
      frame->SetFrameElement(fe);
   } else {
      el = (TGFrameElement *)li->Before(fe);
      if (!el) return;
      frame = el->fFrame;

      el->fFrame = fPimpl->fGrab;
      fPimpl->fGrab->SetFrameElement(el);
      fe->fFrame = frame;
      frame->SetFrameElement(fe);
   }

   Bool_t sav = comp->IsLayoutBroken();
   comp->SetLayoutBroken(kFALSE);
   TGWindow *root = (TGWindow *)fClient->GetRoot();
   root->SetEditable(kFALSE);
   comp->Layout();
   DoRedraw();
   root->SetEditable(kTRUE);
   if (sav) comp->SetLayoutBroken(kTRUE);

   SelectFrame(el->fFrame);
}

//______________________________________________________________________________
void TGuiBldDragManager::HandleGrid()
{
   // switch on/of grid drawn on the editted frame

   if (fStop) return;

   TGWindow *root = (TGWindow*)fClient->GetRoot();

   if (fPimpl->fGrid->fgStep > 1) {
      fPimpl->fGrid->SetStep(1);
      if (fBuilder) {
         fBuilder->UpdateStatusBar("Grid switched OFF");
      }
   } else {
      fPimpl->fGrid->SetStep(gGridStep);

      if (fBuilder) {
         fBuilder->UpdateStatusBar("Grid switched ON");
      }

      if (root->InheritsFrom(TGCompositeFrame::Class())) {
         TGCompositeFrame *comp = (TGCompositeFrame*)root;
         TIter next(comp->GetList());
         TGFrameElement *fe;
         Int_t x, y, w, h;

         while ((fe = (TGFrameElement*)next())) {
            x = fe->fFrame->GetX();
            y = fe->fFrame->GetY();
            w = fe->fFrame->GetWidth();
            h = fe->fFrame->GetHeight();
            ToGrid(x, y);
            ToGrid(w, h);
            fe->fFrame->MoveResize(x, y, w, h);
         }
      }
   }

   root->SetEditable(kFALSE);
   DoRedraw();
   root->SetEditable(kTRUE);
   DrawGrabRectangles();
}

//______________________________________________________________________________
TGCompositeFrame *TGuiBldDragManager::FindLayoutFrame(TGFrame *f)
{
   // helper to find a frame which can be layouted

   if (fStop || !f) return 0;

   const TGWindow *parent = f->GetParent();
   TGCompositeFrame *ret = 0;

   while (parent && (parent != fClient->GetDefaultRoot())) {
      ret = (TGCompositeFrame*)parent;
      if (parent->InheritsFrom(TGMdiFrame::Class())) return ret;
      parent = parent->GetParent();
   }
   return ret;
}

//______________________________________________________________________________
void TGuiBldDragManager::HandleUpdateSelected(TGFrame *f)
{
   // if selected frame was changed by guibuilder editor -> update its app[earence

   if (fStop || !f) return;

   TGCompositeFrame *parent = 0;
   if (f->GetParent()->InheritsFrom(TGCompositeFrame::Class())) {
      parent = (TGCompositeFrame*)f->GetParent();
   }

   if (!CanChangeLayout(parent)) {
      return;
   }

   Bool_t sav = parent->IsLayoutBroken();
   parent->SetLayoutBroken(kFALSE);

   if ((parent->GetWidth() < parent->GetDefaultWidth()) ||
        (parent->GetHeight() < parent->GetDefaultHeight())) {
      parent->Resize(parent->GetDefaultSize());
   } else {
      if (f->InheritsFrom(TGCompositeFrame::Class())) {
         ((TGCompositeFrame*)f)->GetLayoutManager()->Layout();
      }
      parent->Layout();
   }
   fClient->NeedRedraw(parent, kTRUE);
   fClient->NeedRedraw(f);

   if (sav) parent->SetLayoutBroken(kTRUE);

   SelectFrame(f);
}

//______________________________________________________________________________
void TGuiBldDragManager::HideGrabRectangles()
{
   // hide/unmap grab rectangles

   if (fPimpl->fGrabRectHidden) {
      return;
   }
   int i = 0;
   for (i = 0; i < 8; i++) fPimpl->fGrabRect[i]->UnmapWindow();
   for (i = 0; i < 4; i++) fPimpl->fAroundFrame[i]->UnmapWindow();
   fPimpl->fGrabRectHidden = kTRUE;
}

//______________________________________________________________________________
void TGuiBldDragManager::DeletePropertyEditor()
{
   // delete widget property editor

   if (fStop || !fEditor) return;

   TQObject::Disconnect(fEditor);

   delete fEditor;
   fEditor = 0;
}

//______________________________________________________________________________
Int_t TGuiBldDragManager::GetStrartDragX() const
{
   // returns the X coordinate when dragging begins

   return fPimpl->fX0;
}

//______________________________________________________________________________
Int_t TGuiBldDragManager::GetStrartDragY() const
{
   // returns the Y coordinate when dragging begins

   return fPimpl->fY0;
}

//______________________________________________________________________________
Int_t TGuiBldDragManager::GetEndDragX() const
{
   // returns the X coordinate of the dragged frame

   return fPimpl->fY;
}

//______________________________________________________________________________
Int_t TGuiBldDragManager::GetEndDragY() const
{
   // returns the Y coordinate of the dragged frame

   return fPimpl->fY;
}

//______________________________________________________________________________
void TGuiBldDragManager::ExecuteQuickAction(Event_t *event)
{
   // create dialog on the double click on selected frame

   if (fStop) return;

   TGWindow *win = gClient->GetWindowById(event->fWindow);
   if (!win) return;

   if (!fQuickHandler) fQuickHandler = new TGuiBldQuickHandler();

   if (fQuickHandler) {
      fQuickHandler->HandleEvent(win);
   }
}

//______________________________________________________________________________
void TGuiBldDragManager::BreakLayout()
{
   // disable/enable layouting

   if (fStop) return;

   TGFrame *frame = fPimpl->fGrab;

   if (!frame) {
      return;
   }

   TString str = frame->ClassName();
   str += "::";
   str += frame->GetName();

   if (IsFixedLayout(frame)) {
      if (fBuilder) {
         str += " layout cannot be broken";
         fBuilder->UpdateStatusBar(str.Data());
      }
      return;
   }
   ChangeSelected(frame);   // update editors

   frame->SetLayoutBroken(!frame->IsLayoutBroken());
   //if (!fPimpl->fGrab->IsLayoutBroken()) Compact();

   if (fBuilder) {
      str += (frame->IsLayoutBroken() ? " Disable Layout" : " Enable Layout");
      fBuilder->UpdateStatusBar(str.Data());
   }
   DrawGrabRectangles();
}

//______________________________________________________________________________
void TGuiBldDragManager::SwitchLayout()
{
   // switch horizontal/vertical layout

   if (fStop || !fPimpl->fGrab) {
      return;
   }

   TGCompositeFrame *comp = (TGCompositeFrame*)fPimpl->fGrab;

   comp->SetLayoutBroken(kFALSE);

   UInt_t opt = comp->GetOptions();
   TGLayoutManager *m = comp->GetLayoutManager();

   if (m->InheritsFrom(TGHorizontalLayout::Class())) {
      opt &= ~kHorizontalFrame;
      opt |= kVerticalFrame;

      if (fBuilder) {
         TString str = comp->ClassName();
         str += "::";
         str += comp->GetName();
         str += " Vertical Layout ON";
         fBuilder->UpdateStatusBar(str.Data());
      }
   } else if (m->InheritsFrom(TGVerticalLayout::Class())) {
      opt &= ~kVerticalFrame;
      opt |= kHorizontalFrame;

      if (fBuilder) {
         TString str = comp->ClassName();
         str += "::";
         str += comp->GetName();
         str += " Horizontal Layout ON";
         fBuilder->UpdateStatusBar(str.Data());
      }
   }

   comp->ChangeOptions(opt);
   if (!IsFixedSize(comp)) comp->Resize();
   SelectFrame(comp);
}

//______________________________________________________________________________
TGFrame *TGuiBldDragManager::GetSelected() const
{
   // returns grabbed/selected frame

   return (fPimpl ?  fPimpl->fGrab : 0);
}

//______________________________________________________________________________
void TGuiBldDragManager::CloseMenus()
{
   // helper to close all menus

   void *ud;

   if (fFrameMenu) {
      fFrameMenu->EndMenu(ud);
   }
   if (fLassoMenu) {
      fLassoMenu->EndMenu(ud);
   }
   UnmapAllPopups();
}

//______________________________________________________________________________
TGFrame *TGuiBldDragManager::GetEditableParent(TGFrame *fr)
{
   // returns the parent frame which could be editted

   TGWindow *parent = (TGWindow*)fr->GetParent();

   while (parent && (parent != fClient->GetDefaultRoot())) {
      if (!IsEditDisabled(parent) && !IsGrabDisabled(parent)) {
         return (TGFrame*)parent;
      }
      parent = (TGWindow*)parent->GetParent();
   }
   return 0;
}

//______________________________________________________________________________
static TString FindMenuIconName(TString &in)
{
   // returns a name of icon

   Int_t p1 = in.Index("*", 1);
   if (p1 == kNPOS) return "";
   p1++;
   Int_t p2 = in.Index("*", p1);

   if (p2 == kNPOS) return "";

   return in(p1, p2-p1);
}

//______________________________________________________________________________
void TGuiBldDragManager::AddClassMenuMethods(TGPopupMenu *menu, TObject *object)
{
   // add entries with class //*MENU* methods

   if (!menu || !object) {
      return;
   }

   TList *menuItemList = object->IsA()->GetMenuList();

   TClassMenuItem *menuItem;
   TIter nextItem(menuItemList);

   while ((menuItem = (TClassMenuItem*) nextItem())) {
      switch (menuItem->GetType()) {
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

               while ((method = (TMethod*) next())) {
                  if (classPtr != method->GetClass()) {
//                     menu->AddSeparator();
                     classPtr = method->GetClass();
                  }

                  TDataMember *m;
                  EMenuItemKind menuKind = method->IsMenuItem();
                  switch (menuKind) {
                     case kMenuDialog:
                     {  
                        TString str = method->GetCommentString();
                        str = FindMenuIconName(str);
                        const TGPicture *pic = fClient->GetPicture(str.Data());
                        menu->AddEntry(method->GetName(), kMethodMenuAct, method, pic);
                        break;
                     }

                     case kMenuSubMenu:
                        if ((m = method->FindDataMember())) {
                           if (m->GetterMethod()) {
                              TGPopupMenu *r = TRootGuiBuilder::CreatePopup();
                              menu->AddPopup(method->GetName(), r);
                              fPimpl->fFrameMenuTrash->Add(r);
                              TIter nxt(m->GetOptions());
                              TOptionListItem *it;

                              while ((it = (TOptionListItem*) nxt())) {
                                 char  *name  = it->fOptName;
                                 Long_t val   = it->fValue;

                                 TToggle *t = new TToggle;
                                 t->SetToggledObject(object, method);
                                 t->SetOnValue(val);
                                 fPimpl->fFrameMenuTrash->Add(t);

                                 //r->AddSeparator();
                                 r->AddEntry(name, kToggleMenuAct, t);
                                 if (t->GetState()) r->CheckEntryByData(t);
                              }
                           } else {
                              menu->AddEntry(method->GetName(), kMethodMenuAct, method);
                           }
                        }
                        break;

                     case kMenuToggle:
                        {
                           TToggle *t = new TToggle;
                           t->SetToggledObject(object, method);
                           t->SetOnValue(1);
                           fPimpl->fFrameMenuTrash->Add(t);
                           menu->AddEntry(method->GetName(), kToggleMenuAct, t);
                           if (t->GetState()) menu->CheckEntryByData(t);
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
                  if (object) {
                     TMethod* method =
                           object->IsA()->GetMethodWithPrototype(menuItem->GetFunctionName(),
                                                                 menuItem->GetArgs());
                     TToggle *t = new TToggle;
                     t->SetToggledObject(object, method);
                     t->SetOnValue(1);
                     fPimpl->fFrameMenuTrash->Add(t);

                     menu->AddEntry(method->GetName(), kToggleMenuAct, t);
                     if (t->GetState()) menu->CheckEntryByData(t);
                  } else {
                     Warning("Dialog","Cannot use toggle for a global function");
                  }
               } else {
                  const char* menuItemTitle = menuItem->GetTitle();
                  if (strlen(menuItemTitle)==0) menuItemTitle = menuItem->GetFunctionName();
                  menu->AddEntry(menuItemTitle, kMethodMenuAct, menuItem);
               }
            }

            break;
         default:
            break;
      }
   }
}

//______________________________________________________________________________
void TGuiBldDragManager::DoClassMenu(Int_t id)
{
   // processing of method choosen via frame context menu 

   if (!fFrameMenu || ((id != kMethodMenuAct) && (id != kToggleMenuAct))) {
      return;
   }

   if (id == kMethodMenuAct) {
      delete gMenuDialog;
      TMethod *method = (TMethod*)fFrameMenu->GetCurrent()->GetUserData();
      gMenuDialog = new TGuiBldMenuDialog(fPimpl->fGrab, fPimpl->fGrab, method);
      gMenuDialog->Popup();

   } else if (id == kToggleMenuAct) {
      TGPopupMenu *menu = fFrameMenu->GetCurrent()->GetPopup();
      TToggle *toggle = 0;

      if (menu) {    //process submenu
         toggle = (TToggle*)menu->GetCurrent()->GetUserData();
      } else {    //process check entry
         toggle = (TToggle*)fFrameMenu->GetCurrent()->GetUserData();
      }
      if (toggle) toggle->Toggle();
   }
}

//______________________________________________________________________________
void TGuiBldDragManager::DeleteMenuDialog()
{
   // delete dialog and trash

   fPimpl->fFrameMenuTrash->Delete();
   gMenuDialog->DeleteWindow();
}

//______________________________________________________________________________
void TGuiBldDragManager::DoDialogOK()
{
   // process dialog OK button pressed

   gMenuDialog->ApplyMethod();
   DoRedraw();
   DeleteMenuDialog();
   gMenuDialog = 0;
}

//______________________________________________________________________________
void TGuiBldDragManager::DoDialogApply()
{
   // process dialog Apply button pressed

   gMenuDialog->ApplyMethod();
}

//______________________________________________________________________________
void TGuiBldDragManager::DoDialogCancel()
{
   // process dialog Cancel button pressed

   DeleteMenuDialog();
   gMenuDialog = 0;
}

//______________________________________________________________________________
void TGuiBldDragManager::Menu4Frame(TGFrame *frame, Int_t x, Int_t y)
{
   // create and  place context menu for selected frame

   if (fStop) return;

   fPimpl->fSaveGrab = fPimpl->fGrab;
   fPimpl->fX0 = x;
   fPimpl->fY0 = y;

   fPimpl->fClickFrame = frame;

   Bool_t composite = frame->InheritsFrom(TGCompositeFrame::Class());
   Bool_t compar = frame->GetParent()->InheritsFrom(TGCompositeFrame::Class());

   TGCompositeFrame *cfr = 0;
   TGCompositeFrame *cfrp = 0;
   TGLayoutManager *lm = 0;

   if (composite)  {
      cfr = (TGCompositeFrame *)frame;
      lm = cfr->GetLayoutManager();
   }
   if (compar)  {
      cfrp = (TGCompositeFrame *)frame->GetParent();
   }

   delete fFrameMenu;

   fFrameMenu = TRootGuiBuilder::CreatePopup();
   fFrameMenu->Connect("Activated(Int_t)", "TGuiBldDragManager", this, "DoClassMenu(Int_t)");

   TString title = frame->ClassName();
   title += "::";
   title += frame->GetName();
   fFrameMenu->AddLabel(title.Data());
   fFrameMenu->AddSeparator();

   // special case - menu for editable Mdi frame
   if (fBuilder && (frame == fBuilder->GetMdiMain()->GetCurrent())) {
      if (!gSystem->AccessPathName(fPasteFileName.Data())) {
         fFrameMenu->AddEntry("Paste               Ctrl+V", kPasteAct,
                               0, fClient->GetPicture("bld_paste.png"));
      }
      fFrameMenu->AddEntry("Compact          Ctrl+L", kCompactAct,
                               0, fClient->GetPicture("bld_compact.png"));
      fFrameMenu->AddEntry("Grid On/Off       Ctrl+G", kGridAct,
                              0, fClient->GetPicture("bld_grid.png"));
      fFrameMenu->AddEntry("Save As ...        Ctrl+S", kSaveAct,
                              0, fClient->GetPicture("bld_save.png"));
      fFrameMenu->AddEntry("End Edit         Ctrl+DblClick", kEndEditAct,
                              0, fClient->GetPicture("bld_stop.png"));
      goto out;
   }

   AddClassMenuMethods(fFrameMenu, frame);

   if (!fBuilder) {
      fFrameMenu->AddEntry("Gui Builder", kPropertyAct);
      fFrameMenu->AddSeparator();
   }
/*
   if (!frame->IsEditable() && !InEditable(frame->GetId())) {
      fPimpl->fSaveGrab = frame;
      goto out;
   }
*/
   if (!IsEditDisabled(cfrp)) {
      fFrameMenu->AddSeparator();

      if (composite && !IsFixedLayout(frame)) {
         fFrameMenu->AddEntry("Drop               Ctrl+Return", kDropAct);
      }

      if (!IsFixedLayout(cfrp)) {
         fFrameMenu->AddEntry("Cut                 Ctrl+X", kCutAct,
                               0, fClient->GetPicture("bld_cut.png"));
      }
      //
      fFrameMenu->AddEntry("Copy              Ctrl+C", kCopyAct,
                            0, fClient->GetPicture("bld_copy.png"));

      if (frame->IsEditable() && !IsFixedLayout(frame) && 
          !gSystem->AccessPathName(fPasteFileName.Data())) {
         fFrameMenu->AddEntry("Paste              Ctrl+V", kPasteAct,
                               0, fClient->GetPicture("bld_paste.png"));
      }

      if (!IsFixedLayout(cfrp)) {
         fFrameMenu->AddEntry("Delete             Del", kDeleteAct,
                              0, fClient->GetPicture("bld_delete.png"));
      }

      if (!IsFixedLayout(cfrp)) {
         fFrameMenu->AddEntry("Crop               Shift+Del", kCropAct,
                               0, fClient->GetPicture("bld_crop.png"));
      }

//      if (!IsFixedLayout(cfrp) && !gSystem->AccessPathName(fPasteFileName.Data())) {
//         fFrameMenu->AddEntry("Replace          Ctrl+R", kReplaceAct, 
//                               0, fClient->GetPicture("bld_paste_into.png"));
//      }

      fFrameMenu->AddSeparator();
   } else {
      if (!gSystem->AccessPathName(fPasteFileName.Data()) && !IsFixedLayout(frame)) {
         fFrameMenu->AddEntry("Paste               Ctrl+V", kPasteAct,
                               0, fClient->GetPicture("bld_paste.png"));
      }
      if (frame->GetMainFrame() == frame) {
         fFrameMenu->AddEntry("Clone               Ctrl+A", kCloneAct);
      }
      fFrameMenu->AddSeparator();
   }

   if (CanChangeLayout(frame)) {
      const char *label = (frame->IsLayoutBroken() ? "Allow Layout   Ctrl+B" : 
                                                     "Break Layout   Ctrl+B");
      fFrameMenu->AddEntry(label, kBreakLayoutAct,
                            0, fClient->GetPicture("bld_break.png"));
   }

   if (composite && !cfr->GetList()->IsEmpty()) {
      if (CanCompact(frame)) {
         if (!frame->IsEditable()) {
            fFrameMenu->AddEntry("Compact          Ctrl+L", kCompactAct,
                                  0, fClient->GetPicture("bld_compact.png"));
         } else {
            fFrameMenu->AddEntry("Compact          Ctrl+L", kCompactGlobalAct,
                                  0, fClient->GetPicture("bld_compact.png"));
         }
      }

      if (((lm->IsA() == TGVerticalLayout::Class()) || 
           (lm->IsA() == TGHorizontalLayout::Class())) && !IsFixedLayout(frame)) {

         if (lm->IsA() == TGVerticalLayout::Class()) {
            fFrameMenu->AddEntry("Horizontal        Ctrl+H", kSwitchLayoutAct,
                                 0, fClient->GetPicture("bld_hbox.png"));
         } else if (lm->IsA() == TGHorizontalLayout::Class()) {
            fFrameMenu->AddEntry("Vertical          Ctrl+H", kSwitchLayoutAct,
                                 0, fClient->GetPicture("bld_vbox.png"));
         }
      }
   }

   if (compar && (cfrp->GetList()->GetSize() > 1) && CanChangeLayoutOrder(frame)) {
      if (cfrp->GetList()->First() != frame->GetFrameElement()) {
         fFrameMenu->AddEntry("Lay Up             Up/Left", kLayUpAct);
      }
      if (cfrp->GetList()->Last() != frame->GetFrameElement()) {
         fFrameMenu->AddEntry("Lay Down         Down/Right", kLayDownAct);
      }
      fFrameMenu->AddSeparator();
   }

   if (frame->IsEditable()) {
      fFrameMenu->AddEntry("Grid On/Off       Ctrl+G", kGridAct,
                            0, fClient->GetPicture("bld_grid.png"));
      fFrameMenu->AddEntry("Save As ...        Ctrl+S", kSaveAct,
                            0, fClient->GetPicture("bld_save.png"));
      //fFrameMenu->AddEntry("End Edit         Ctrl+DblClick", kEndEditAct);
   } else if (composite && !IsFixedLayout(frame)) {
      //fFrameMenu->AddEntry("Edit              Ctrl+DblClick", kEditableAct);
   }

out:
   fFrameMenu->Connect("Activated(Int_t)", "TGuiBldDragManager", this, "HandleAction(Int_t)");

   fPimpl->fLastPopupAction = kNoneAct;
   fPimpl->fPlacePopup = kTRUE;

   HideGrabRectangles();
   fFrameMenu->PlaceMenu(x, y, kFALSE, kTRUE);
}

//______________________________________________________________________________
void TGuiBldDragManager::Menu4Lasso(Int_t x, Int_t y)
{
   // creates menu when lasso is drawn

   if (fStop || !fLassoDrawn) return;

   DrawLasso();

   delete fLassoMenu;

   fLassoMenu = TRootGuiBuilder::CreatePopup();
   fLassoMenu->AddLabel("Edit actions");
   fLassoMenu->AddSeparator();
   fLassoMenu->AddEntry("Grab              Return", kGrabAct);
   fLassoMenu->AddSeparator();
   fLassoMenu->AddEntry("Delete            Delete", kDeleteAct, 
                        0, fClient->GetPicture("bld_delete.png"));
   fLassoMenu->AddEntry("Crop              Shift+Delete", kCropAct, 
                        0, fClient->GetPicture("bld_crop.png"));
   fLassoMenu->AddSeparator();
   fLassoMenu->AddEntry("Align Left        Left Key", kLeftAct, 
                        0, fClient->GetPicture("bld_AlignLeft.png"));
   fLassoMenu->AddEntry("Align Right      Right Key", kRightAct, 
                        0, fClient->GetPicture("bld_AlignRight.png"));
   fLassoMenu->AddEntry("Align Up         Up Key", kUpAct, 
                        0, fClient->GetPicture("bld_AlignTop.png"));
   fLassoMenu->AddEntry("Align Down     Down Key", kDownAct, 
                        0, fClient->GetPicture("bld_AlignBtm.png"));

   fLassoMenu->Connect("Activated(Int_t)", "TGuiBldDragManager", this, "HandleAction(Int_t)");

   fPimpl->fLastPopupAction = kNoneAct;
   fPimpl->fPlacePopup = kTRUE;
   fLassoMenu->PlaceMenu(x, y, kFALSE, kTRUE);
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::IsPasteFrameExist()
{
   // returns kTRUE if paste frame exist

   return !gSystem->AccessPathName(fPasteFileName.Data());
}

