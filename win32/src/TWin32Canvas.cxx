// @(#)root/win32:$Name:  $:$Id: TWin32Canvas.cxx,v 1.2 2001/03/14 21:43:06 brun Exp $
// Author: Valery Fine   05/01/96

#include "TWin32Canvas.h"
#include "TGWin32WindowsObject.h"
#include "TWin32MenuItem.h"
#include "TROOT.h"
#include "TApplication.h"
#include "TError.h"
#include "TMarker.h"
#include "TStyle.h"
#include "TBrowser.h"
#include "TClassTree.h"
#include "TInterpreter.h"
#include "TSystem.h"


// ClassImp(TWin32Canvas)

//______________________________________________________________________________
TWin32Canvas::TWin32Canvas() {
    fCanvasImpID = -1;
}
//______________________________________________________________________________
TWin32Canvas::TWin32Canvas(TCanvas *c, const char *name, UInt_t width, UInt_t height)
{
    fCanvasImpID = -1;
    fCanvas = c;
    SetCanvas(0,0,width,height);
    SetCanvas(name);

    fMenu = new TWin32Menu("CanvasMenu",name);
    MakeMenu();
 }
//______________________________________________________________________________
TWin32Canvas::TWin32Canvas(TCanvas *c, const char *name, Int_t x, Int_t y, UInt_t width, UInt_t height)
{

    fCanvasImpID = -1;
    fCanvas = c;
    SetCanvas(x,y,width,height);
    SetCanvas(name);

    fMenu = new TWin32Menu("CanvasMenu",name);
    MakeMenu();
}

//______________________________________________________________________________
TWin32Canvas::~TWin32Canvas() {
    if ( fCanvasImpID == -1) return;
//    gVirtualX->SelectWindow(fCanvasImpID);    //select current canvas
//    gVirtualX->CloseWindow();
    ((TGWin32 *)gVirtualX)->RemoveWindow((TGWin32Switch *)fCanvasImpID);
    fCanvasImpID = -1;

#ifdef UUU
   TCanvas *canvas = obj->Canvas();
   if(canvas) { // delete canvas;
      gROOT->ProcessLineAsynch(Form("TCanvas *c=(TCanvas *)0x%lx; delete c;",(Long_t)canvas));
   }
#endif

}


//______________________________________________________________________________
void TWin32Canvas::MakeMenu()
{
Int_t iMenuLength = sizeof(fStaticMenuItems) / sizeof(fStaticMenuItems[0]);
Int_t i = 0;
TWin32Menu *PopUpMenu;

//*-*   Static data member to create menus for all canvases
 TWin32MenuItem *item = 0;
//  ATTENTION !!!!  To change menu one MUST adjust kEndOfMenu constant in $TGWIN32WINDOWSOBJECT first !!!
//  =====================================================================================================
 fStaticMenuItems  =  new TWin32MenuItem *[kEndOfMenu+2]; // We have to estimate the total size of the
                                                          // menu in advance (not very good so!)

 //*-*  simple  separator
 fStaticMenuItems[i++] = new                                   TWin32MenuItem(kSeparator);
 //*-*  Some other type of separators
 fStaticMenuItems[i++] = new                                   TWin32MenuItem(kMenuBreak);
 fStaticMenuItems[i++] = new                                   TWin32MenuItem(kMenuBarBreak);

//*-*  Main Canvas menu items

 Int_t iMainMenuStart = i;
 fStaticMenuItems[i++] = new TWin32MenuItem("File","&File",kSubMenu);
 fStaticMenuItems[i++] = new TWin32MenuItem("Edit","&Edit",kSubMenu);
 fStaticMenuItems[i++] = new TWin32MenuItem("View","&View",kSubMenu);
 fStaticMenuItems[i++] = new TWin32MenuItem("Options","&Options",kSubMenu);
 fStaticMenuItems[i++] = new TWin32MenuItem("Inspector","&Inspector",kSubMenu);
 fStaticMenuItems[i++] = new TWin32MenuItem("Classes","&Classes",kSubMenu);

 fStaticMenuItems[i++] = new TWin32MenuItem("Help","&Help",HelpCB,kString | kRightJustify);
 Int_t iMainMenuEnd = i-1;

//*-*   Items for the File Menu

 Int_t iFileMenuStart = i;
 fStaticMenuItems[i++] = new TWin32MenuItem("New","&New",NewCB);
 fStaticMenuItems[i++] = new TWin32MenuItem("Open","&Open",OpenCB);
 fStaticMenuItems[i++] = new                     TWin32MenuItem(kSeparator);
 fStaticMenuItems[i++] = new TWin32MenuItem("Save","&Save",SaveCB);
 fStaticMenuItems[i++] = new TWin32MenuItem("SaveAs","Save &As",SaveAsCB);
 fStaticMenuItems[i++] = new                                  TWin32MenuItem(kSeparator);
 fStaticMenuItems[i++] = new TWin32MenuItem("Print","&Print",PrintCB);
 fStaticMenuItems[i++] = new                                  TWin32MenuItem(kSeparator);
 fStaticMenuItems[i++] = new TWin32MenuItem("Close","&Close",CloseCB);
 fStaticMenuItems[i++] = new                                  TWin32MenuItem(kSeparator);
 fStaticMenuItems[i++] = new TWin32MenuItem("Quit","&Quit",QuitCB);
 fStaticMenuItems[i++] = new TWin32MenuItem("Exit","E&xit",QuitCB);
 Int_t iFileMenuEnd = i-1;

//*-*   Items for the Edit Menu

 Int_t iEditMenuStart = i;
 fStaticMenuItems[i++] = new TWin32MenuItem("Undo","&Undo",UnDoCB);
 fStaticMenuItems[i++] = new                                   TWin32MenuItem(kSeparator);
 fStaticMenuItems[i++] = new TWin32MenuItem("Editor","&Editor",EditorCB);
 fStaticMenuItems[i++] = new                                   TWin32MenuItem(kSeparator);
 fStaticMenuItems[i++] = new TWin32MenuItem("ClearPad","Clear&Pad",ClearPadCB);
 fStaticMenuItems[i++] = new TWin32MenuItem("ClearCanvas","Clear &Window",ClearCanvasCB);
 Int_t iEditMenuEnd = i-1;

//*-*   Items for the View

 Int_t iViewMenuStart = i;
 fStaticMenuItems[i++] = new TWin32MenuItem("Browser","&Explore",BrowserCB);
 fStaticMenuItems[i++] = new                                   TWin32MenuItem(kSeparator);
 fStaticMenuItems[i++] = new TWin32MenuItem("Colors","&Color Box",ColorsCB);
 fStaticMenuItems[i++] = new TWin32MenuItem("Fonts","&Font Box", FontsCB);
 fStaticMenuItems[i++] = new TWin32MenuItem("Markers","&Markers",MarkersCB);
 fStaticMenuItems[i++] = new TWin32MenuItem("Iconify","&Iconify",IconifyCB);
 fStaticMenuItems[i++] = new TWin32MenuItem("X3D","&3D View",X3DViewCB);
 fStaticMenuItems[i++] = new                                   TWin32MenuItem(kSeparator);
 fStaticMenuItems[i++] = new TWin32MenuItem("Interrupt","No&tify CINT",InterruptCB);
 Int_t iViewMenuEnd = i-1;

//*-*   Items for the Options Menu

 Int_t iOptionMenuStart = i;
 fStaticMenuItems[i++] = new TWin32MenuItem("EventStatus","&Status Bar",EventStatusCB);
 fStaticMenuItems[i++] = new                                   TWin32MenuItem(kSeparator);
 fStaticMenuItems[i++] = new TWin32MenuItem("AutoFit","&Auto Resize Canvas",AutoFitCanvasCB);
 fStaticMenuItems[i++] = new TWin32MenuItem("FitCanvas","&Resize Canvas",FitCanvasCB);
 fStaticMenuItems[i++] = new                                   TWin32MenuItem(kSeparator);
 fStaticMenuItems[i++] = new TWin32MenuItem("Refresh","Re&Fresh",RefreshCB);
 fStaticMenuItems[i++] = new                                   TWin32MenuItem(kSeparator);
 int iOptStat = i;
 fStaticMenuItems[i++] = new TWin32MenuItem("Statistics","&Statistics",OptStatCB);
 int iOptTitle = i;
 fStaticMenuItems[i++] = new TWin32MenuItem("Htitle","&Histogram title",OptTitleCB);
 int iOptFit = i;
 fStaticMenuItems[i++] = new TWin32MenuItem("FitParams","Fit &Params",OptFitCB);
 int iCanEditHist = i;
 fStaticMenuItems[i++] = new TWin32MenuItem("CanEditHistograms","&Can Edit Histograms",CanEditHistogramsCB);
 Int_t iOptionMenuEnd = i-1;

//*-*   Items for the Inspect Menu

 Int_t iInspectMenuStart = i;
 fStaticMenuItems[i++] = new TWin32MenuItem("ROOT","&ROOT",ROOTInspectCB);
 fStaticMenuItems[i++] = new TWin32MenuItem("StarBrowser","&Start Browser",BrowserCB);
 Int_t iInspectMenuEnd = i-1;

//*-*   Items for the Class Menu

 Int_t iClassesMenuStart = i;
 fStaticMenuItems[i++] = new TWin32MenuItem("ClassTree","&Class Tree",FullTreeCB);
 Int_t iClassesMenuEnd = i-1;

 Int_t iEndOfMenu =  i-1;

 iMenuLength = i;
//*-* Create full list of the items

 for (i=0;i<=iEndOfMenu;i++)
    RegisterMenuItem(fStaticMenuItems[i]);

 if (gStyle->GetOptStat())       fStaticMenuItems[iOptStat]     ->Checked();
 if (gStyle->GetOptTitle())      fStaticMenuItems[iOptTitle]    ->Checked();
 if (gStyle->GetOptFit())        fStaticMenuItems[iOptFit]      ->Checked();
 if (gROOT->GetEditHistograms()) fStaticMenuItems[iCanEditHist] ->Checked();

//*-*  Create static menues (one times for all Canvas ctor)


//*-* File
   PopUpMenu = fStaticMenuItems[kMFile]->GetPopUpItem();
      for (i=iFileMenuStart;i<=iFileMenuEnd; i++)
        PopUpMenu->Add(fStaticMenuItems[i]);

//*-* Edit
   PopUpMenu = fStaticMenuItems[kMEdit]->GetPopUpItem();
     for (i=iEditMenuStart;i<=iEditMenuEnd; i++)
       PopUpMenu->Add(fStaticMenuItems[i]);

//*-* View
   PopUpMenu = fStaticMenuItems[kMView]->GetPopUpItem();
     for (i=iViewMenuStart;i<=iViewMenuEnd; i++)
       PopUpMenu->Add(fStaticMenuItems[i]);

//*-* Options
   PopUpMenu = fStaticMenuItems[kMOptions]->GetPopUpItem();
     for (i=iOptionMenuStart;i<=iOptionMenuEnd; i++)
       PopUpMenu->Add(fStaticMenuItems[i]);

//*-* Inspector
   PopUpMenu = fStaticMenuItems[kMInspector]->GetPopUpItem();
     for (i=iInspectMenuStart;i<=iInspectMenuEnd; i++)
       PopUpMenu->Add(fStaticMenuItems[i]);

//*-* Inspector
   PopUpMenu = fStaticMenuItems[kMClasses]->GetPopUpItem();
     for (i=iClassesMenuStart;i<=iClassesMenuEnd; i++)
       PopUpMenu->Add(fStaticMenuItems[i]);

//*-*  Create main menu
     for (i=iMainMenuStart;i<=iMainMenuEnd; i++)
       fMenu->Add(fStaticMenuItems[i]);
}

//______________________________________________________________________________
void TWin32Canvas::FitCanvas()
{
#ifndef WIN32
   Dimension w, h, s;
   XtVaGetValues(fScrolledWindow, XmNwidth,   &w,
                                  XmNheight,  &h,
                                  XmNspacing, &s,
                                  NULL);
   fCwidth  = w-s;
   fCheight = h-s;
   fUnits   = kPixels;

   ResizeCanvas();
#endif
}
//______________________________________________________________________________
void   TWin32Canvas::ForceUpdate(){};
//______________________________________________________________________________
void   TWin32Canvas::GetWindowGeometry(Int_t &x, Int_t &y, UInt_t &w, UInt_t &h)
{
   W32_GetGeometry(x, y, w, h);
}
//______________________________________________________________________________
void   TWin32Canvas::Iconify(){};
//______________________________________________________________________________
Int_t  TWin32Canvas::InitWindow(){
        return fCanvasImpID;}

//______________________________________________________________________________
void   TWin32Canvas::SetCanvasSize(UInt_t w, UInt_t h){
   if (fCanvasImpID==-1)
     SetCanvas(0,0,w,h);
   else
     gVirtualX->RescaleWindow(fCanvasImpID, w, h);
}
//______________________________________________________________________________
void   TWin32Canvas::ShowMenuBar(Bool_t show)
{
   if (show) W32_SetMenu(fMenu->GetMenuHandle());
}
//______________________________________________________________________________
void   TWin32Canvas::ShowStatusBar(Bool_t show)
{
   W32_ShowStatusBar(show);
}

//______________________________________________________________________________
void   TWin32Canvas::Show(){;}

//______________________________________________________________________________
void   TWin32Canvas::CreateStatusBar(Int_t nparts)
{
     // Creates the StatusBar object with <nparts> "single-size" parts
  W32_CreateStatusBar(nparts);
}

//______________________________________________________________________________
void   TWin32Canvas::CreateStatusBar(Int_t *parts, Int_t nparts)
{
  // parts  - an interger array of the relative sizes of each parts (in percents) //
  // nParts - number of parts                                                     //

  W32_CreateStatusBar(parts,nparts);
}

//______________________________________________________________________________
void   TWin32Canvas::SetCanvas(Int_t x, Int_t y,UInt_t w, UInt_t h){
   fCanvasImp = this;
   if (fCanvasImpID==-1) {
      CreateWindowsObject((TGWin32 *)gVirtualX, x, y, w, h);
      TGWin32WindowsObject *winobj = (TGWin32WindowsObject *)this;
      fCanvasImpID = gVirtualX->InitWindow((ULong_t)winobj);
   }
   W32_Set(x, y, w, h);
}

//______________________________________________________________________________
void   TWin32Canvas::SetCanvas(const char *title){
   W32_SetTitle(title);
}

//______________________________________________________________________________
void   TWin32Canvas::SetStatusText(const char *text, Int_t partidx)
{ // Set Text into the 'npart'-th part of the status bar
   W32_SetStatusText(text,partidx);
}

//______________________________________________________________________________
void   TWin32Canvas::SetWindowPosition(Int_t x, Int_t y)
{
   Int_t  dum1, dum2;
   UInt_t w, h;

   W32_GetGeometry(dum1, dum2, w, h);
   W32_Set(x, y, w, h);
}

//______________________________________________________________________________
void TWin32Canvas::RootExec(const char *cmd)
{
  printf(" %s \n" , cmd );
}



//______________________________________________________________________________
void   TWin32Canvas::UpdateCanvasImp(){
   W32_Update(0);
}

//______________________________________________________________________________
void TWin32Canvas::NewCanvas()
{
  printf("NewCanvas() \n");
}

// Menu Callbacks

//______________________________________________________________________________
void TWin32Canvas::ClearCanvasCB(TWin32Canvas *obj, TVirtualMenuItem *item)
{
   obj->Canvas()->Clear();
   obj->Canvas()->Modified();
   obj->Canvas()->Update();
}

//______________________________________________________________________________
void TWin32Canvas::ClearPadCB(TWin32Canvas *obj, TVirtualMenuItem *item){
  gPad->Clear();
  gPad->Modified();
  gPad->Update();
}

//______________________________________________________________________________
void TWin32Canvas::CloseCB(TWin32Canvas *obj, TVirtualMenuItem *item)
{

   // delete the TCanvas, which in turn will delete its TCanvasImp

   TVirtualPad *savepad = gPad;
   gPad = 0;                         // hide gPad from CINT
   gInterpreter->DeleteGlobal(obj->Canvas());
   gPad = savepad;                   // restore gPad for ROOT
   // delete the TCanvas, which in turn will delete its TCanvasImp
   delete obj->Canvas();
}
//______________________________________________________________________________
void TWin32Canvas::EditorCB(TWin32Canvas *obj, TVirtualMenuItem *item)
{
   obj->Canvas()->EditorBar();
}

//______________________________________________________________________________
void TWin32Canvas::HelpCB(TWin32Canvas *obj, TVirtualMenuItem *item)
{
   gSystem->Exec("explorer http://root.cern.ch/root/html/ClassIndex.html");
}

//______________________________________________________________________________
void TWin32Canvas::NewCB(TWin32Canvas *obj, TVirtualMenuItem *item)
{
   gROOT->GetMakeDefCanvas()();
}

//______________________________________________________________________________
void TWin32Canvas::OpenCB(TWin32Canvas *obj, TVirtualMenuItem *item)
{
   obj->RootExec("Exec Open");
}

//______________________________________________________________________________
void TWin32Canvas::PrintCB(TWin32Canvas *obj, TVirtualMenuItem *item)
{
   obj->Canvas()->Print();
}

//______________________________________________________________________________
void TWin32Canvas::QuitCB(TWin32Canvas *obj, TVirtualMenuItem *item)
{
   gApplication->Terminate(0);
}

#define MAX_file_name 512

//______________________________________________________________________________
static const Char_t *Extension(char *file, const char *ex)
{
// Append a standard extenstion to the file name if any
  if (!strchr(file,'.')) {
     strncat(file,".",MAX_file_name);
     strncat(file,ex,MAX_file_name);
  }
  return file;
}

//______________________________________________________________________________
void TWin32Canvas::SaveAsCB(TWin32Canvas *obj, TVirtualMenuItem *item)
{
//--->     obj->Canvas()->Print("EPS");
//   obj->RootExec("Exec Save As...");


   OPENFILENAME *lpOpenBuffer = new OPENFILENAME;

   lpOpenBuffer->lStructSize = sizeof(OPENFILENAME);
   lpOpenBuffer->hwndOwner         = obj->GetWindow();

   lpOpenBuffer->hInstance         = NULL;


     static const char filter[] = {
 //
 //    File format    \0           acceptable extension
 //                         ( the last extension is used as default )
     "C++ macro (*.cpp,*.cxx,*.C)\0*.cpp,*.cxx,*.C\0"
     "Postscript (*.ps)\0*.ps\0"
     "Encapsulated Postscript (*.eps)\0*.eps\0"
     "ROOT file (*.root)\0*.root\0"
     "\0"};

   static const Char_t **extensions = 0;
   // Create array of the default file extensions
   if (!extensions) {
      Int_t i = 0;
      Int_t lFilter = sizeof(filter);
      // Count the number of different files
      Int_t lExts = 0;
      for (i=0;i<lFilter;i++) if (!filter[i]) lExts++;

        // NOTE: The next operations do nothing.
        // This is odd ...
      lExts >> 1;
      // create extensions
      extensions = new const Char_t *[lExts];
      lFilter--;
      Int_t s = 1;
      lExts = 0;
      for (i=0;i<lFilter;i++) {
         if (filter[i]) continue;
         s = -s;
         if (s > 0) continue;
         extensions[lExts] = strrchr(&filter[i]+1,'.');
         if (extensions[lExts]) extensions[lExts]++;
         lExts++;
      }
   }

   lpOpenBuffer->lpstrFilter       = filter;

   lpOpenBuffer->lpstrCustomFilter = NULL;
   lpOpenBuffer->nMaxCustFilter    = NULL;
   lpOpenBuffer->nFilterIndex      = 1;
   lpOpenBuffer->lpstrFile         = new char[MAX_file_name];
   strcpy(lpOpenBuffer->lpstrFile,(obj->Canvas())->GetName());
   strcat(lpOpenBuffer->lpstrFile,".*");
   lpOpenBuffer->nMaxFile          = MAX_file_name;
/*
 *  Take the initial file name from EDIT control
 */
//  GetDlgItemText(hwnd,Id+1,lpOpenBuffer->lpstrFile,MAX_string);


   lpOpenBuffer->lpstrFileTitle    = NULL;
   lpOpenBuffer->nMaxFileTitle     = NULL;
   lpOpenBuffer->lpstrInitialDir   = NULL;
   lpOpenBuffer->lpstrTitle        = "Save the selected Canvas/Pad as";
   lpOpenBuffer->Flags             = OFN_HIDEREADONLY    | OFN_LONGNAMES |
                                     OFN_OVERWRITEPROMPT ;
   lpOpenBuffer->nFileOffset       = NULL;
   lpOpenBuffer->nFileExtension    = NULL;
   lpOpenBuffer->lpstrDefExt       = NULL;
   lpOpenBuffer->lCustData         = NULL;
   lpOpenBuffer->lpfnHook          = NULL;
   lpOpenBuffer->lpTemplateName    = NULL;

   if ( GetSaveFileName(lpOpenBuffer)) ;
 //      SetDlgItemText(hwnd,Id+1,lpOpenBuffer->lpstrFile);

   Int_t exIndx = lpOpenBuffer->nFilterIndex;
   const char *option = extensions[exIndx-1];

   if (exIndx==1) option = "cxx";
   obj->Canvas()->Print(Extension(lpOpenBuffer->lpstrFile,extensions[exIndx-1]),option);

 delete [] lpOpenBuffer->lpstrFile;
 delete lpOpenBuffer;
}

//______________________________________________________________________________
void TWin32Canvas::SaveCB(TWin32Canvas *obj, TVirtualMenuItem *item)
{
   obj->Canvas()->Print();
}


//______________________________________________________________________________
void TWin32Canvas::SaveSourceCB(TWin32Canvas *obj, TVirtualMenuItem *item)
{
      obj->Canvas()->SaveSource();
}

//______________________________________________________________________________
void TWin32Canvas::ColorsCB(TWin32Canvas *obj, TVirtualMenuItem *item)
{
   TVirtualPad *padsav = gPad->GetCanvas();
   char defcanvas[32];
   strcpy(defcanvas,gROOT->GetDefCanvasName());
   gROOT->SetDefCanvasName("DisplayColors");
   (gROOT->GetMakeDefCanvas())();
   gROOT->SetDefCanvasName(defcanvas);
   TPad::DrawColorTable();
   gPad->Update();
   padsav->cd();
}

//______________________________________________________________________________
void TWin32Canvas::FontsCB(TWin32Canvas *obj, TVirtualMenuItem *item)
{
       obj->RootExec("Exec Fonts");
}

//______________________________________________________________________________
void TWin32Canvas::IconifyCB(TWin32Canvas *obj, TVirtualMenuItem *item)
{
   obj->Iconify();
}
//______________________________________________________________________________
void TWin32Canvas::MarkersCB(TWin32Canvas *obj, TVirtualMenuItem *item)
{
   TVirtualPad *padsav = gPad->GetCanvas();
   char defcanvas[32];
   TCanvas *m = new TCanvas("markers","MarkersTypes",600,200);
   TMarker::DisplayMarkerTypes();
   m->Update();
   padsav->cd();
}

//______________________________________________________________________________
void TWin32Canvas::X3DViewCB(TWin32Canvas *obj, TVirtualMenuItem *item)
{
   gPad->x3d("OPENGL");
}
//______________________________________________________________________________
void TWin32Canvas::InterruptCB(TWin32Canvas *obj, TVirtualMenuItem *item)
{
 gROOT->SetInterrupt();
}
//______________________________________________________________________________
void TWin32Canvas::EventStatusCB(TWin32Canvas *obj, TVirtualMenuItem *item)
{
   obj->fCanvas->ToggleEventStatus();
   if (obj->fCanvas->GetShowEventStatus())
   {
    obj->ShowStatusBar();
    item->Checked();
   }
   else {
    item->UnChecked();
    obj->ShowStatusBar(kFALSE);
   }

}

//______________________________________________________________________________
void TWin32Canvas::AutoFitCanvasCB(TWin32Canvas *obj, TVirtualMenuItem *item)
{
//   obj->fAutoFit = obj->fAutoFit ? kFALSE : kTRUE;
}

//______________________________________________________________________________
void TWin32Canvas::FitCanvasCB(TWin32Canvas *obj, TVirtualMenuItem *item)
{
   obj->FitCanvas();
   obj->Canvas()->Update();
}

//______________________________________________________________________________
void TWin32Canvas::RefreshCB(TWin32Canvas *obj, TVirtualMenuItem *item)
{
   obj->Canvas()->Paint();
   obj->Canvas()->Update();
}
//______________________________________________________________________________
void TWin32Canvas::OptStatCB(TWin32Canvas *obj, TVirtualMenuItem *item)
{
   if (gStyle->GetOptStat()) {gStyle->SetOptStat(0); item->UnChecked(); }
   else                      {gStyle->SetOptStat(1); item->Checked();   }
   gPad->Modified();
   obj->Canvas()->Update();
}
//______________________________________________________________________________
void TWin32Canvas::OptTitleCB(TWin32Canvas *obj, TVirtualMenuItem *item)
{
   if (gStyle->GetOptTitle()) {gStyle->SetOptTitle(0); item->UnChecked(); }
   else                       {gStyle->SetOptTitle(1); item->Checked();   }
   gPad->Modified();
   obj->Canvas()->Update();
}
//______________________________________________________________________________
void TWin32Canvas::OptFitCB(TWin32Canvas *obj, TVirtualMenuItem *item)
{
   if (gStyle->GetOptFit()) {gStyle->SetOptFit(0); item->UnChecked(); }
   else                     {gStyle->SetOptFit(1); item->Checked();   }
   gPad->Modified();
   obj->Canvas()->Update();

}
//______________________________________________________________________________
void TWin32Canvas::CanEditHistogramsCB(TWin32Canvas *obj, TVirtualMenuItem *item)
{
   Bool_t oldedith = gROOT->GetEditHistograms();
   Bool_t newedith = oldedith ? kFALSE : kTRUE;
   gROOT->SetEditHistograms(newedith);
   if (newedith == kTRUE) item->Checked();
   else                   item->UnChecked();
}

//______________________________________________________________________________
void TWin32Canvas::ROOTInspectCB(TWin32Canvas *obj, TVirtualMenuItem *item)
{
   obj->Canvas()->cd();
   gROOT->Inspect();
//   obj->Canvas()->Update();
}
//______________________________________________________________________________
void TWin32Canvas::UnDoCB(TWin32Canvas *obj, TVirtualMenuItem *item)
{
//   TEditor *editor = obj->Canvas()->GetEditor();
//   if (editor) editor->SetUndo(1);
}

//______________________________________________________________________________
void TWin32Canvas::BrowserCB(TWin32Canvas *obj, TVirtualMenuItem *item)
{
   TBrowser *browser = new TBrowser("browser");
}

//______________________________________________________________________________
void TWin32Canvas::FullTreeCB(TWin32Canvas *obj, TVirtualMenuItem *item)
{
  char cdef[64];
  TList *lc = (TList*)gROOT->GetListOfCanvases();
  if (lc->FindObject("ClassTree"))
     sprintf(cdef,"ClassTree_%d",lc->GetSize()+1);
   else
     sprintf(cdef,"%s","ClassTree");

   new TClassTree(cdef,"TObject");
   obj->Canvas()->Update();
}

//______________________________________________________________________________
void TWin32Canvas::PartialTreeCB(TWin32Canvas *obj, TVirtualMenuItem *item)
{ ;
//   cout<<"Displaying Partial tree in canvas:"<<obj->GetTitle()<<endl;
}
