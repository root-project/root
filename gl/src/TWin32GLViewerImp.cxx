// @(#)root/gl:$Name$:$Id$
// Author: Valery Fine(fine@vxcern.cern.ch)   29/12/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TWin32GLViewer                                                       //
//                                                                      //
// This class creates a main window with menubar, scrollbars and an     //
// OpenGL capable drawing area.                                         //
//                                                                      //
//                                                                      //
// Basic interface between the WIN32 graphics system and OpenGL package //
//                                                                      //
//                                                                      //
//   -----------------------------------------------------------        //
//   OpenGL for Windows 95 can be found as follows:                     //
//                                                                      //
//   ftp://ftp.microsoft.com/Softlib/Mslfiles/OGLFIX.EXE                //
//                                                                      //
//   This release includes:                                             //
//                                                                      //
//    - DLL contains the runtime dynamic-link libraries for OpenGL and  //
//      GLU including a new OPENGL32.DLL which fixes the memory         //
//      DC/bitmap resizing bug.                                         //
//   -----------------------------------------------------------        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <iostream.h>
#include "TWin32GLViewerImp.h"
#include "TROOT.h"
#include "TSeqCollection.h"
#include "TMap.h"
#include "TObjString.h"
#include "TError.h"

#include "Buttons.h"
#include "TPadOpenGLView.h"
#include "TGWin32StatusBar.h"

// TWin32GLViewer* TWin32GLViewerImp::fgCurrent = 0;

// ClassImp(TWin32GLViewerImp)

//______________________________________________________________________________
TWin32GLViewerImp::TWin32GLViewerImp()
{
    fGLView      = 0;
    fhOpenGLRC    = 0;
    fWin32Object = 0;
    fThreadId    = 0; // No current context
}
//______________________________________________________________________________
TWin32GLViewerImp::TWin32GLViewerImp(TPadOpenGLView *padview,const char *title, UInt_t width, UInt_t height)
{
   // Create browser with a specified width and height.
   fGLView  = padview;
   CreateViewer(title,0,0,width,height);
   SetDrawList(0);
}

//______________________________________________________________________________
TWin32GLViewerImp::TWin32GLViewerImp(TPadOpenGLView *padview,const char *title, Int_t x, Int_t y,
                             UInt_t width, UInt_t height)
{
   // Create browser with a specified width and height and at position x, y.

    fGLView  = padview;
    CreateViewer(title,x,y,width,height);
    SetDrawList(0);
}

//______________________________________________________________________________
TWin32GLViewerImp::~TWin32GLViewerImp()
{
   // Win32 GLViewer destructor.
    DeleteContext();
    return;

//*-* make the rendering context not current
//   fgCurrent = 0;
   wglMakeCurrent (NULL, NULL) ;

//*-* delete the rendering context
   if (fhOpenGLRC) wglDeleteContext (fhOpenGLRC);


}
//______________________________________________________________________________
void TWin32GLViewerImp::CreateContext()
{
  if(!fhOpenGLRC) CreateViewer("OpenGL Viewer");
  return;

  if ( ((TGWin32 *)gVirtualX)->IsCmdThread())
      CreateContextCB();
  else
  {
      TWin32SendWaitClass code(this,(UInt_t)kCreateContext,0,0,0);
      ExecWindowThread(&code);
      code.Wait();
  }
}

//______________________________________________________________________________
void TWin32GLViewerImp::CreateContextCB()
{
  if (fhOpenGLRC)
              DeleteContextCB(); //  Delete previous Rendering context first
  fhOpenGLRC = 0;

//  if (!SetupPixelFormat()) return;

  printf(" fPixelFormatIndex = %d \n", fPixelFormatIndex);

  if (fObjectDC)
  {
//      fhOpenGLRC = wglCreateContext(fWin32Object->GetWin32DC());
      fhOpenGLRC = wglCreateContext(fObjectDC);
      if (!fhOpenGLRC)
          printf(" Error: OpenGL was not created. Error code = %x \n", GetLastError());
  }

}

//______________________________________________________________________________
void TWin32GLViewerImp::CreateViewer( const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height)
{

    fThreadId    = 0; // No current context
    CreateWindowsObject((TGWin32 *)gVirtualX,x,y,width,height);
//   W32_SetTitle(title && strlen(title) ? title : b->GetName());
    W32_SetTitle(title && strlen(title) ? title : "OpenGl Viewer");

    fMenu = new TWin32Menu("OpenGLMenu",title);

    MakeMenu();
    // Create th Status bar
    int parts[] = {9,21,6,14,50};
    CreateStatusBar(parts,5);
    ShowStatusBar();
    SetStatusText("Projection:",0,TGLViewerImp::kStatusNoBorders);
    SetStatusText("Light:",2,TGLViewerImp::kStatusNoBorders);
}


//______________________________________________________________________________
void TWin32GLViewerImp::CreateStatusBar(Int_t nparts)
{
  // Creates the StatusBar object with <nparts> "single-size" parts
  W32_CreateStatusBar(nparts);
}

//______________________________________________________________________________
void TWin32GLViewerImp::CreateStatusBar(Int_t *parts, Int_t nparts)
{
  // parts  - an interger array of the relative sizes of each parts (in percents) //
  // nParts - number of parts                                                     //

  W32_CreateStatusBar(parts,nparts);
}

//______________________________________________________________________________
void TWin32GLViewerImp::DeleteContext()
{
    if (!fhOpenGLRC) return;

    if ( ((TGWin32 *)gVirtualX)->IsCmdThread())
        DeleteContextCB();
    else
    {
      TWin32SendWaitClass code(this,(UInt_t)kDeleteContext,0,0,0);
      ExecWindowThread(&code);
      code.Wait();
    }

}
//______________________________________________________________________________
void TWin32GLViewerImp::DeleteContextCB()
{
// Delete current OpenGL RC
  if (fhOpenGLRC)
  {
      MakeCurrentCB(kFALSE);
      wglDeleteContext(fhOpenGLRC);
      fhOpenGLRC = 0;
  }
}

//______________________________________________________________________________
void TWin32GLViewerImp::ExecThreadCB(TWin32SendClass *command)
{
    EGLCommand cmd = (EGLCommand)(command->GetData(0));
    switch (cmd)
    {
    case kCreateContext:
        {
            CreateContextCB();
            break;
        }

    case kDeleteContext:
        {
            DeleteContextCB();
            break;
        }
    case kMakeCurrent:
        {
            Bool_t flag = (Bool_t)(command->GetData(1));
            MakeCurrentCB(flag);
            break;
        }
    case kSwapBuffers:
        {
            SwapBuffersCB();
            break;
        }
    default:
        break;
    }

    if (LOWORD(command->GetCOP()) == kSendWaitClass)
        ((TWin32SendWaitClass *)command)->Release();
    else
        delete command;
}


//______________________________________________________________________________
void TWin32GLViewerImp::MakeCurrent(Bool_t flag)
{
//  if (flag && fThreadId) return; // The current thread is occupied by GL context already

  if ( ((TGWin32 *)gVirtualX)->IsCmdThread())
      MakeCurrentCB(flag);
  else
  {
      TWin32SendWaitClass code(this,(UInt_t)kMakeCurrent,(UInt_t)flag,0,0);
      ExecWindowThread(&code);
      code.Wait();
  }
}
//______________________________________________________________________________
void TWin32GLViewerImp::MakeCurrentCB(Bool_t flag)
{
// Make this object current
  if (fObjectDC)
  {
      if (!fThreadId) fThreadId  = GetCurrentThreadId(); // ID of this thread
      HGLRC curctx = wglGetCurrentContext();
      if (curctx != fhOpenGLRC)
      {
          Bool_t res = wglMakeCurrent(fObjectDC,fhOpenGLRC);
          Int_t ierr = GetLastError();
          if (!res)
          {
            printf(" Error: TWin32GLViewerImp::MakeCurrent Error code =  %d, Thread id = %d \n",
              ierr,GetCurrentThreadId());
            fThreadId            = 0;
          }
      }
  }
  else
  {
      wglMakeCurrent(NULL,NULL); // Disable any Rendering context for the current thread
      fThreadId = 0;
  }
}

//______________________________________________________________________________
void TWin32GLViewerImp::MakeMenu(){

#ifdef draft
Int_t iMenuLength = sizeof(fStaticMenuItems) / sizeof(fStaticMenuItems[0]);
Int_t i = 0;
TWin32Menu *PopUpMenu;

//*-*   Static data member to create menus for all canvases

 fStaticMenuItems  =  new TMenuItem *[kEndOfMenu+2];

 //*-*  simple  separator
 fStaticMenuItems[i++] = new                     TMenuItem(kSeparator);
 //*-*  Some other type of separators
 fStaticMenuItems[i++] = new                     TMenuItem(kMenuBreak);
 fStaticMenuItems[i++] = new                     TMenuItem(kMenuBarBreak);


//*-*  Main Canvas menu items
 Int_t iMainMenuStart = i;
 fStaticMenuItems[i++] = new TMenuItem("File","&File",MF_POPUP);
 fStaticMenuItems[i++] = new TMenuItem("Edit","&Edit",MF_POPUP);
 fStaticMenuItems[i++] = new TMenuItem("View","&View",MF_POPUP);
 Int_t iMainMenuEnd = i-1;


//*-*   Items for the File Menu

 Int_t iFileMenuStart = i;
 fStaticMenuItems[i++] = new TMenuItem("New","&New",NewCB);
 fStaticMenuItems[i++] = new TMenuItem("Open","&Open",OpenCB);
 fStaticMenuItems[i++] = new                     TMenuItem(kSeparator);
 fStaticMenuItems[i++] = new TMenuItem("Save","&Save",SaveCB);
 fStaticMenuItems[i++] = new TMenuItem("SaveAs","Save &As",SaveAsCB);
 fStaticMenuItems[i++] = new                                  TMenuItem(kSeparator);
 fStaticMenuItems[i++] = new TMenuItem("Print","&Print",PrintCB);
 fStaticMenuItems[i++] = new                                  TMenuItem(kSeparator);
 fStaticMenuItems[i++] = new TMenuItem("Close","&Close",CloseCB);
 Int_t iFileMenuEnd = i-1;


//*-*   Items for the Edit Menu

 Int_t iEditMenuStart = i;
 fStaticMenuItems[i++] = new TMenuItem("Cut","Cu&t",CutCB);
 fStaticMenuItems[i++] = new TMenuItem("Copy","&Copy",CopyCB);
 fStaticMenuItems[i++] = new TMenuItem("Paste","&Paste",PasteCB);
 fStaticMenuItems[i++] = new                                   TMenuItem(kSeparator);
 fStaticMenuItems[i++] = new TMenuItem("SelectAll","Select &All",SelectAllCB);
 fStaticMenuItems[i++] = new TMenuItem("InvertSelection","&Invert Selection",InvertSelectionCB);
 Int_t iEditMenuEnd = i-1;

//*-*   Items for the View

 Int_t iViewMenuStart = i;
 fStaticMenuItems[i++] = new TMenuItem("ToolBar","&Tool Bar",ToolBarCB);
 fStaticMenuItems[i++] = new TMenuItem("StatusBar","&Status Bar", StatusBarCB);
 fStaticMenuItems[i++] = new                                   TMenuItem(kSeparator);
 fStaticMenuItems[i++] = new TMenuItem("LargeIcons","&Large Icons",LargeIconsCB);
 fStaticMenuItems[i++] = new TMenuItem("SmallIcons","&Small Icons",SmallIconsCB);
 fStaticMenuItems[i++] = new TMenuItem("Details","&Details",DetailsCB);
 fStaticMenuItems[i++] = new                                   TMenuItem(kSeparator);
 fStaticMenuItems[i++] = new TMenuItem("ArrangeIcons","&Arrange Icons",MF_POPUP);
 fStaticMenuItems[i++] = new                                   TMenuItem(kSeparator);
 fStaticMenuItems[i++] = new TMenuItem("Refresh","&Refresh",RefreshCB);
 fStaticMenuItems[i++] = new TMenuItem("Options","&Options",MF_POPUP);
 Int_t iViewMenuEnd = i-1;

 Int_t iEndOfMenu =  i-1;

 iMenuLength = i;
//*-* Create full list of the items

 for (i=0;i<=iEndOfMenu;i++)
    RegisterMenuItem(fStaticMenuItems[i]);

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

//*-*  Create main menu
     for (i=iMainMenuStart;i<=iMainMenuEnd; i++)
       fMenu->Add(fStaticMenuItems[i]);

//*-*  Glue this menu onto the canvas window

   // W32_SetMenu(fMenu->GetMenuHandle());

#endif
}


//______________________________________________________________________________
void TWin32GLViewerImp::MakeStatusBar()
{
  // fStatusBar = new TGWin32StatusBar(this);
}

//______________________________________________________________________________
void TWin32GLViewerImp::MakeToolBar()
{
}

//______________________________________________________________________________
void TWin32GLViewerImp::SetStatusText(const Text_t *text, Int_t partidx,Int_t stype)
{
  // Set Text into the 'npart'-th part of the status bar
  //   enum {kStatusPopIn, kStatusNoBorders, kStatusOwn, kStatusPopOut};
  // 0              - The text is drawn with a border to appear lower than the plane of the window.
  // SBT_NOBORDERS  - The text is drawn without borders.
  // SBT_OWNERDRAW  - The text is drawn by the parent window.
  // SBT_POPOUT     - The text is drawn with a border to appear higher than the plane of the window.

   static Int_t opt[] = {0,SBT_NOBORDERS,SBT_OWNERDRAW,SBT_POPOUT} ;
   W32_SetStatusText(text,partidx,opt[stype]);
}

//______________________________________________________________________________
Bool_t TWin32GLViewerImp::SetupPixelFormat()
{
    PIXELFORMATDESCRIPTOR *ppfd = &fPixelFormat;
    HDC hdc = fObjectDC;

    ZeroMemory(ppfd,sizeof(PIXELFORMATDESCRIPTOR));

    ppfd->nSize = sizeof(PIXELFORMATDESCRIPTOR);
    ppfd->nVersion = 1;
    ppfd->dwFlags =  PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
//                        PFD_DRAW_TO_BITMAP | PFD_SUPPORT_GDI ;

   if (fWin32Mother->fhdCommonPalette)
   {
       /* Set to color-index mode and use the default color palette. */
       ppfd->iPixelType =  PFD_TYPE_COLORINDEX;
       ppfd->cColorBits = 8;
       gVirtualGL->SetTrueColorMode(kFALSE);
   }
   else
   {
       ppfd->iPixelType =  PFD_TYPE_RGBA;
       ppfd->cColorBits = 24;
       gVirtualGL->SetTrueColorMode();
   }
 //*-*  Get the device context's best available pixel format

    if ( (fPixelFormatIndex = ChoosePixelFormat(hdc, ppfd)) == 0 )
    {
        printf("Error:  ChoosePixelFormat failed %d \n",GetLastError());
        return kFALSE;
    }

//*-*  Make that match the device context's current pixel format

    if (SetPixelFormat(hdc, fPixelFormatIndex, ppfd) == FALSE)
    {
        printf("Error: SetPixelFormat failed %d \n", GetLastError());
        return kFALSE;
    }

    return kTRUE;
}

//______________________________________________________________________________
void TWin32GLViewerImp::ShowHelp()
{
    char buffer[1024] = "";
    char *title = "Open GL Viewer controls:";
    char *help[28] = {
     "PRESS \tu\t--- to Move down \n",
     "\ti\t--- to Move up\n",
     "\th\t--- to Shift right\n",
     "\tl\t--- to Shift left\n",
     "\tj\t--- to Poll the object backward\n",
     "\tk\t--- to Push the object forward\n",
     "\n",
     "\t+\t--- to Increase speed to move\n",
     "\t-\t--- to Decrease speed to move\n",
     "",
     "\tn\t--- to turn \"SMOOTH\" color mode on\n",
     "\tm\t--- to turn \"SMOOTH\" color mode off\n",
     "\n",
     "\tt\t--- to toggle Light model: \"Real\"/\"Pseudo\"\n",
     "\n",
     "\tp\t--- to toggle Perspective/Orthographic projection\n",
     "\tr\t--- to Hidden surface mode\n",
     "\tw\t--- to wireframe mode\n",
     "\tc\t--- to cull-face mode\n",
     "\n",
     "\ts\t--- to increase the scale factor (clip cube borders)\n",
     "\ta\t--- to decrease the scale factor (clip cube borders)\n",
     "\n",
     "\txX\n",
     "\tyY\t--- and any \"arrow\" keys to rotate object along x,y,z axis \n",
     "\tzZ\n",
     "\n",
     "HOLD the left mouse button and MOVE mouse to ROTATE object"};

     for (int i=0;i<28; i++) strcat(buffer,help[i]);

     MessageBox( NULL, buffer ,title, MB_ICONINFORMATION | MB_OK | MB_TOPMOST);
  }

//______________________________________________________________________________
void TWin32GLViewerImp::ShowStatusBar(Bool_t show)
{
   W32_ShowStatusBar(show);
}

//______________________________________________________________________________
void TWin32GLViewerImp::SwapBuffers()
{
  if ( ((TGWin32 *)gVirtualX)->IsCmdThread())
      SwapBuffersCB();
  else
  {
      TWin32SendWaitClass code(this,(UInt_t)kSwapBuffers,0,0,0);
      ExecWindowThread(&code);
      code.Wait();
  }

}
//______________________________________________________________________________
void TWin32GLViewerImp::SwapBuffersCB()
{
// Spaw the OpenGL double buffer
    if (fhOpenGLRC) {
      ::SwapBuffers(fObjectDC);
      SetStatusText("Done",4);
    }

}

//______________________________________________________________________________
void TWin32GLViewerImp::Update()
{
//*-* Allow paint
    TGLViewerImp::Update();
//*-* Update the OpenGL view on the screen
    W32_Update();
}

//______________________________________________________________________________
LRESULT APIENTRY TWin32GLViewerImp::OnChar
                 (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
//*-*
//*-*  Windows Procedure to manage: WM_CHAR
//*-*
  Int_t help = (TCHAR) wParam ;
  if (!strchr("uihljk+-pnmtrwcasxyzXYZ",help)) ShowHelp();
  HandleInput(kKeyPress,Int_t(wParam),Int_t(lParam & 0xff));
  return 0;
}

//______________________________________________________________________________
LRESULT APIENTRY TWin32GLViewerImp::OnClose(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
//*-*    Message ID: WM_CLOSE
//                   ========
    CloseCB(this,NULL);
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}


//______________________________________________________________________________
LRESULT APIENTRY TWin32GLViewerImp::OnCreate(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{

//*-*  Create a toolbar that the user can customize and that has a tooltip
//*-*  associated with it.
//*-*  Check system error

//    Float_t w = 1;
//    SetWindow(hwnd);
//    TGWin32WindowsObject::OnCreate(hwnd, uMsg, wParam, lParam);

//
// Initial fill the data structure in
//
  fObjectDC            = GetDC(hwnd);
  fObjectClipRegion    = (HRGN)NULL;
  fROOTCursor          = FALSE;
  fSystemCursorVisible = TRUE;
  fMouseInit           = 0;
  fSetTextInput        = FALSE;

  fLoc.x = fLoc.y = fLocp.x = fLocp.y=0;

  BOOL mc;
//*-* create a rendering context
  SetupPixelFormat();
  if (fWin32Mother->fhdCommonPalette) {
      HPALETTE hPal = SelectPalette(fObjectDC,fWin32Mother->fhdCommonPalette,FALSE);
      if (hPal != fWin32Mother->fhdCommonPalette) DeleteObject(hPal);
      Int_t n = RealizePalette(fObjectDC);
  }
#if 0
  fhOpenGLRC = wglCreateContext(fObjectDC);
  if (!(mc=wglMakeCurrent(fObjectDC,fhOpenGLRC)))
  {
      int ierr = GetLastError();
      printf(" Error %d RC=%x DC=%x, MC=%x nPal=%d\n", ierr, fhOpenGLRC, fObjectDC,mc,n);
  }
#endif

    CreateContextCB();
//    fObjectGLRC = wglCreateContext (fObjectDC);
    MakeCurrentCB();
    return 0;
}

//______________________________________________________________________________
LRESULT APIENTRY TWin32GLViewerImp::OnKeyDown
                 (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
//*-*
//*-*  Windows Procedure to manage: WM_KEYDOWN
//*-*
// Map "arrows" to the "ASCII" and handle it

  Int_t nVirtKey = (int) wParam;
  Int_t asciikeymap[] = {'z','x','Z','X' };
  Int_t indx = nVirtKey - VK_LEFT;
  if (indx >= 0 && nVirtKey <= VK_DOWN)
  {
      HandleInput(kKeyPress,asciikeymap[indx],1);
      return 0;
  }
  else
      return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

//______________________________________________________________________________
LRESULT APIENTRY TWin32GLViewerImp::OnMouseButton
                 (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
//
//  Windows Procedure to manage: WM_LBUTTONDOWN WM_MBUTTONDOWN WM_RBUTTONDOWN
//                               WM_LBUTTONUP   WM_MBUTTONUP   WM_RBUTTONUP
//                               WM_MOUSEMOVE
//                               WM_CONTEXTMENU
//
 if (!fGLView)  return DefWindowProc(hwnd,uMsg, wParam, lParam);

 MakeCurrentCB();

#if 0
 TCanvas *canvas = 0;

 if (canvas=GetCanvas()) {

#endif
//  if (fMouseInit)
     switch (uMsg) {
       case WM_LBUTTONDOWN:
           HandleInput(kButton1Down,MAKEPOINTS(lParam).x,MAKEPOINTS(lParam).y);
//           SetCursor(fWin32Mother->fCursors[kPointer]);
           return 0;
           break;
       case WM_MBUTTONDOWN:
       case WM_RBUTTONDOWN:
           break;

//         fMouseInit = 0;
//         OnRootMouse(hwnd,uMsg,wParam,lParam);
       case WM_MOUSEMOVE:
           if (wParam & MK_LBUTTON)
               HandleInput(kButton1Motion,MAKEPOINTS(lParam).x,MAKEPOINTS(lParam).y);
           break;
//        SetCursor(fWin32Mother->fCursors[kPointer]);
       case WM_RBUTTONUP:
       case WM_MBUTTONUP:
           break;
       case WM_LBUTTONUP:
               HandleInput(kButton1Up,MAKEPOINTS(lParam).x,MAKEPOINTS(lParam).y);
//           SetCursor(fWin32Mother->fCursors[kPointer]);
//         OnRootMouse(hwnd,uMsg,wParam,lParam);
           return 0;
       default:
         break;
     };

   return DefWindowProc(hwnd,uMsg, wParam, lParam);
 }
//______________________________________________________________________________
LRESULT APIENTRY TWin32GLViewerImp::OnPaint      (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
//    Message ID: WM_PAINT
//                ========
   PAINTSTRUCT paint;
   if (!fGLView) return DefWindowProc(hwnd, uMsg, wParam, lParam);

   HDC hdc = BeginPaint(hwnd, &paint);
   {
       if (fWin32Mother->fhdCommonPalette)
       {
           HPALETTE hPal = SelectPalette(fObjectDC,fWin32Mother->fhdCommonPalette,FALSE);
           if (hPal != fWin32Mother->fhdCommonPalette) DeleteObject(hPal);
           RealizePalette(fObjectDC);
       }
       TGLViewerImp::Paint();
   }
   EndPaint(hwnd, &paint);
   return 0;
}

//______________________________________________________________________________
LRESULT APIENTRY TWin32GLViewerImp::OnSysCommand
                 (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
//*-*  WM_SYSCOMMAND  WM_DESTROY
//*-*  =============  ==========
//*-*    uCmdType = wParam;        - type of system command requested
//*-*    xPos = LOWORD(lParam);    - horizontal postion, in screen coordinates
//*-*    yPos = HIWORD(lParam);    - vertical postion, in screen coordinates

//*-*   By unknown reason this message is supplied with zero parameters
//*-*   but Windows sends a WM_CLOSE message itself to proceed

// if ((wParam & 0xFFF0) == SC_CLOSE) return OnClose(hwnd,uMsg,wParam,lParam);
 if ((wParam & 0xFFF0) == SC_CLOSE)
 {
    DeleteView();
#if 0
      TCanvas *canvas = GetCanvas();
      if(canvas) { // delete canvas;
        char *cmd;
        cmd = Form("TCanvas *c=(TCanvas *)0x%lx; delete c;", (Long_t)canvas);
        printf("OnSysCommand %s\n", cmd);
        gROOT->ProcessLine(cmd);
      }
#endif
      return DefWindowProc(hwnd,uMsg, wParam, lParam);
 }

#if 0
 if (fWin32Mother->fhdCommonPalette) {
   SetSystemPaletteUse(fObjectDC,SYSPAL_STATIC);
   while(!PostMessage(HWND_BROADCAST,WM_SYSCOLORCHANGE, 0, 0)){;}
 }
#endif

 return DefWindowProc(hwnd,uMsg, wParam, lParam);
}


//______________________________________________________________________________
LRESULT APIENTRY TWin32GLViewerImp::OnSize     (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
//    Message ID: WM_SIZE
//                =======
//    cout <<" TWin32ControlBarImp::OnSize" << endl;

    if (wParam == SIZE_MAXHIDE  ||
        wParam == SIZE_MAXSHOW  ||
        wParam == SIZE_MINIMIZED) return DefWindowProc(hwnd, uMsg, wParam, lParam);

    if (!fGLView)
        return DefWindowProc(hwnd, uMsg, wParam, lParam);

    Int_t nWidth  = LOWORD(lParam);  // width of client area
    Int_t nHeight = HIWORD(lParam);  // height of client area
    Int_t nBottom   = 0;             // The left-lower corner of the viewport

 //*-*  Resize the status bar window if any
    if (fStatusBar) {
        SendMessage(fStatusBar->GetWindow(),uMsg,wParam,lParam);
        fStatusBar->OnSize();
        nBottom = fStatusBar->GetHeight();
    }

    MakeCurrentCB();
    glViewport( 0, nBottom, (GLint) nWidth, (GLint)  nHeight );
    fGLView->Size(nWidth,nHeight);
    return 0;
}

//______________________________________________________________________________
void TWin32GLViewerImp::Iconify()
{
   // Iconify the browser.

   // TMenuWindow::Iconify();
}

//______________________________________________________________________________
void TWin32GLViewerImp::RootExec(const char *cmd)
{
   // Process ROOT commands.

   cout << "RootExec: " << cmd << endl;
}

//______________________________________________________________________________
void TWin32GLViewerImp::NewCB(TWin32GLViewerImp *obj, TVirtualMenuItem *item)
{
   // Static callback for menu item: New.

//   TWin32GLViewerImp *b = new TWin32GLViewerImp( "ROOT GLViewer" );
}

//______________________________________________________________________________
void TWin32GLViewerImp::SaveCB(TWin32GLViewerImp *obj, TVirtualMenuItem *item)
{
   // Static callback for menu item: Save.

   obj->RootExec("Exec Save");
}

//______________________________________________________________________________
void TWin32GLViewerImp::SaveAsCB(TWin32GLViewerImp *obj, TVirtualMenuItem *item)
{
   // Static callback for menu item: SaveAs.

   obj->RootExec("Exec Save As...");
}

//______________________________________________________________________________
void TWin32GLViewerImp::Show()
{
   // Show (i.e. popup) the browser.
}


//______________________________________________________________________________
void TWin32GLViewerImp::PrintCB(TWin32GLViewerImp *obj, TVirtualMenuItem *item)
{
   // Static callback for menu item: Print.

   obj->RootExec("Exec Print...");
}

//______________________________________________________________________________
void TWin32GLViewerImp::CloseCB(TWin32GLViewerImp *obj, TVirtualMenuItem *item)
{
   // Static callback for menu item: Close.


   // delete the TGLViewer
//   delete obj;
}
