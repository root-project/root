// @(#)root/win32:$Name:  $:$Id: TGWin32.cxx,v 1.13 2002/07/04 06:47:38 brun Exp $
// Author: Valery Fine   28/11/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


//______________________________________________________________________________
//*-*-*-*-*-*-*-*-*-*-*-*The   G  W  I  N  3  2  class*-*-*-*-*-*-*-*-*-*-*
//*-*                    =============================
//*-*
//*-*  Basic interface to the WIN32 graphics system
//*-*
//*-*  This code was initially developped in the context of HIGZ and PAW
//*-*  by Valery Fine to port the package X11INT (by Olivie Couet)
//*-*  to Windows NT.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

#include <process.h>

#ifndef ROOT_TGWin32WindowsObject
#include "TGWin32WindowsObject.h"
#endif

#ifndef ROOT_TGWin32Pen
#include "TGWin32Pen.h"
#endif

#include "TGWin32PixmapObject.h"

#ifndef ROOT_TMath
#include "TMath.h"
#endif

#include "TGWin32Brush.h"

#include "TWinNTSystem.h"

#ifndef ROOT_TError
#include "TError.h"
#endif


#define NoOperation (TGWin32Switch *)(-1)
#define SafeCallW32(_w)       if (##_w == NoOperation) return; if (##_w) ##_w

#define SafeCallWin32         SafeCallW32(fSelectedWindow)

//                              )TGWin32Object *_w = (TGWin32Object *)fWindows[fSelectedWindow]; \
//                              if (_w) _w



#define ReturnCallW32(_w)      if (_w == NoOperation) return 0; return !(_w) ? 0 : _w

#define ReturnCallWin32        ReturnCallW32(fSelectedWindow)

//                              SafeCallW32(fSelectedWindow)

//______________________________________________________________________________
LRESULT APIENTRY OnCreate
                 (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
//
//  Windows Procedure to manage: WM_CREATE
//

  TGWin32WindowsObject *lpTGWin32WindowsObject =
                        (TGWin32WindowsObject *)(((CREATESTRUCT *)lParam)->lpCreateParams);
  SetWindowLong(hwnd,GWL_USERDATA,(LONG)lpTGWin32WindowsObject);   // tie the window and ROOT object
  return lpTGWin32WindowsObject->OnCreate(hwnd,uMsg,wParam,lParam);
}

//______________________________________________________________________________
LRESULT APIENTRY WndROOT(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Main Universal Windows procedure to manage all dispatched events     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

 if (uMsg == WM_CREATE) return ::OnCreate(hwnd, uMsg, wParam, lParam);
 else {
     TGWin32Object* lpWin32Object = ((TGWin32Object*)GetWindowLong(hwnd,GWL_USERDATA));
     if (lpWin32Object) {
               ((TGWin32WindowsObject *)lpWin32Object)->StartPaint();
               LRESULT ret_value = lpWin32Object->CallCallback(hwnd, uMsg, wParam, lParam);
               ((TGWin32WindowsObject *)lpWin32Object)->FinishPaint();
               return ret_value;
     }
     else return ::DefWindowProc(hwnd, uMsg, wParam, lParam);
 }
}

//______________________________________________________________________________
// LPTHREAD_START_ROUTINE ROOT_MsgLoop(HANDLE ThrSem)
unsigned int _stdcall ROOT_MsgLoop(HANDLE ThrSem)
//*-*-*-*-*-*-*-*-*-*-*-*-* ROOT_MsgLoop*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                       ============
//*-*  Launch a separate thread to handle the ROOTCLASS messages
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
 {
   MSG msg;
   HWND hwndROOT;
   int value;
   TGWin32WindowsObject *lpWinThr;
   int erret;  // GetMessage result

   ReleaseSemaphore(ThrSem, 1, NULL);
 //  static i = 0;
   Bool_t EventLoopStop = kFALSE;
   while(!EventLoopStop)
    {
       if (EventLoopStop = (!(erret=GetMessage(&msg,NULL,0,0)) || erret == -1))
                                                                     continue;

       if (msg.hwnd == NULL)
       {
           switch (msg.message)
           {
           case IX11_ROOT_MSG:
               {
                   switch(HIWORD(msg.wParam))
                   {
                   case ROOT_Control:
                       switch (LOWORD(msg.wParam))
                       {
                       case IX_OPNWI:
                           {
                               ((TGWin32WindowsObject *)(msg.lParam))->Win32CreateObject();
                                continue;  /* IX_OPNWI */
                           }
                       default:
                           break;
                       } /* End of ROOT_Control */
                   case ROOT_Attribute:
                       switch (LOWORD(msg.wParam))
                       {
                       default:
                           break;
                       }
                   default:
                       break;
                   }
               }
           case ROOT_CMD:
           case ROOT_SYNCH_CMD:
               if (TWin32HookViaThread::ExecuteEvent(&msg,msg.message==ROOT_SYNCH_CMD)) continue;   // This command is synchronyzed with "cmd" thread
               break;
           default:
               break;
           }
       }
       if (msg.message != IX11_ROOT_MSG && msg.message != ROOT_CMD && msg.message != ROOT_HOOK )
                           TranslateMessage(&msg);
       DispatchMessage(&msg);
    }
    if (erret == -1)
    {
        erret = GetLastError();
        Error("MsgLoop", "Error in GetMessage");
        Printf(" %d \n", erret);
    }


    ((TGWin32 *) gVirtualX)->SetMsgThreadID(0);
    if (msg.wParam) ReleaseSemaphore((HANDLE) msg.wParam, 1, NULL);

    _endthreadex(0);
    return 0;
 } /* ROOT_MsgLoop */


ClassImp(TGWin32)

//______________________________________________________________________________
TGWin32::TGWin32(const char *name, const char *title) : TVirtualX(name,title)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*Normal Constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                        ==================                              *-*

// 2  fSelectedWindow = fPrevWindow = -1;
  fSelectedWindow = fPrevWindow = 0;
  fDisplayOpened  = kFALSE;
  Init();
}


//______________________________________________________________________________
Bool_t TGWin32::Init(void *display)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*WIN32 GUI initialization-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                        ========================                         *-*
  if(fDisplayOpened)   return fDisplayOpened;
  fTextAlignH      = 1;
  fTextAlignV      = 1;
  fTextMagnitude   = 1;
  fCharacterUpX    = 1;
  fCharacterUpY    = 1;
  fDrawMode        = kCopy;
  fTextFontModified = 1;
  fhdCommonPalette = 0;

  fGLKernel = 0;
 // fGLKernel = new TWin32GLKernel();

  //
  // Retrieve the applicaiton instance
  //

  fHInstance = GetModuleHandle(NULL);

  fROOTCLASS = "ROOT";


  //
  // Create cursors
  //

  fCursors[kBottomLeft]  = LoadCursor(NULL, IDC_SIZENESW);// (display, XC_bottom_left_corner);
  fCursors[kBottomRight] = LoadCursor(NULL, IDC_SIZENWSE);// (display, XC_bottom_right_corner);
  fCursors[kTopLeft]     = LoadCursor(NULL, IDC_SIZENWSE);// (display, XC_top_left_corner);
  fCursors[kTopRight]    = LoadCursor(NULL, IDC_SIZENESW);// (display, XC_top_right_corner);
  fCursors[kBottomSide]  = LoadCursor(NULL, IDC_SIZENS);  // (display, XC_bottom_side);
  fCursors[kLeftSide]    = LoadCursor(NULL, IDC_SIZEWE);  // (display, XC_left_side);
  fCursors[kTopSide]     = LoadCursor(NULL, IDC_SIZENS);  // (display, XC_top_side);
  fCursors[kRightSide]   = LoadCursor(NULL, IDC_SIZEWE);  // (display, XC_right_side);
  fCursors[kMove]        = LoadCursor(NULL, IDC_SIZEALL); // (display, XC_fleur);
//  fCursors[kCross]       = LoadCursor(NULL, IDC_CROSS);   // (display, XC_tcross);
  fCursors[kArrowHor]    = LoadCursor(NULL, IDC_SIZEWE);  // (display, XC_sb_h_double_arrow);
  fCursors[kArrowVer]    = LoadCursor(NULL, IDC_SIZENS);  // (display, XC_sb_v_double_arrow);
  fCursors[kHand]        = LoadCursor(NULL, IDC_NO);      // (display, XC_hand2);
  fCursors[kRotate]      = LoadCursor(NULL, IDC_ARROW);    // (display, XC_exchange);
//  fCursors[kRotate]      = LoadCursor(NULL, IDC_SIZEALL); // (display, XC_exchange);
  fCursors[kPointer]     = LoadCursor(NULL, IDC_ARROW);   // (display, XC_left_ptr);
  fCursors[kArrowRight]  = LoadCursor(NULL, IDC_ARROW);   // XC_arrow
  fCursors[kCaret]       = LoadCursor(NULL, IDC_IBEAM);   // XC_xterm



// Yin cursor AND bitmask

  BYTE ANDmaskCursor[] =
                       {0xff, 0xfc, 0x7f, 0xff,   /* line 1 */
                        0xff, 0xfc, 0x7f, 0xff,   /* line 2 */
                        0xff, 0xfc, 0x7f, 0xff,   /* line 3 */
                        0xff, 0xfc, 0x7f, 0xff,   /* line 4 */

                        0xff, 0xfc, 0x7f, 0xff,   /* line 5 */

                        0xff, 0xfc, 0x7f, 0xff,   /* line 6 */
                        0xff, 0xfc, 0x7f, 0xff,   /* line 7 */
                        0xff, 0xfc, 0x7f, 0xff,   /* line 8 */

                        0xff, 0xfc, 0x7f, 0xff,   /* line 9 */
                        0xff, 0xfc, 0x7f, 0xff,   /* line 10 */
                        0xff, 0xfc, 0x7f, 0xff,   /* line 11 */
                        0xff, 0xfc, 0x7f, 0xff,   /* line 12 */

                        0xff, 0xfc, 0x7f, 0xff,   /* line 13 */
                        0xff, 0xfc, 0x7f, 0xff,   /* line 14 */

                        0x00, 0x02, 0x80, 0x01,   /* line 15 */
                        0x00, 0x01, 0x00, 0x01,   /* line 16 */
                        0x00, 0x02, 0x80, 0x01,   /* line 17 */

                        0xff, 0xfc, 0x7f, 0xff,   /* line 18 */
                        0xff, 0xfc, 0x7f, 0xff,   /* line 19 */
                        0xff, 0xfc, 0x7f, 0xff,   /* line 20 */

                        0xff, 0xfc, 0x7f, 0xff,   /* line 21 */
                        0xff, 0xfc, 0x7f, 0xff,   /* line 22 */

                        0xff, 0xfc, 0x7f, 0xff,   /* line 23 */
                        0xff, 0xfc, 0x7f, 0xff,   /* line 24 */

                        0xff, 0xfc, 0x7f, 0xff,   /* line 25 */
                        0xff, 0xfc, 0x7f, 0xff,   /* line 26 */
                        0xff, 0xfc, 0x7f, 0xff,   /* line 27 */
                        0xff, 0xfc, 0x7f, 0xff,   /* line 28 */

                        0xff, 0xfc, 0x7f, 0xff,   /* line 29 */
                        0xff, 0xfc, 0x7f, 0xff,   /* line 30 */
                        0xff, 0xfc, 0x7f, 0xff,   /* line 31 */

                        0xff, 0xff, 0xff, 0xff};  /* line 32 */

// Yin cursor XOR bitmask

  BYTE XORmaskCursor[] =
                       {0x00, 0x02, 0x80, 0x00,   /* line 1 */
                        0x00, 0x02, 0x80, 0x00,   /* line 2 */
                        0x00, 0x02, 0x80, 0x00,   /* line 3 */
                        0x00, 0x02, 0x80, 0x00,   /* line 4 */

                        0x00, 0x02, 0x80, 0x00,   /* line 5 */

                        0x00, 0x02, 0x80, 0x00,   /* line 6 */
                        0x00, 0x02, 0x80, 0x00,   /* line 7 */
                        0x00, 0x02, 0x80, 0x00,   /* line 8 */

                        0x00, 0x02, 0x80, 0x00,   /* line 9 */
                        0x00, 0x02, 0x80, 0x00,   /* line 10 */
                        0x00, 0x02, 0x80, 0x00,   /* line 11 */
                        0x00, 0x02, 0x80, 0x00,   /* line 12 */

                        0x00, 0x02, 0x80, 0x00,   /* line 13 */
                        0x00, 0x02, 0x80, 0x00,   /* line 14 */

                        0xFF, 0xFC, 0x7F, 0xFE,   /* line 15 white border */
                        0x00, 0x00, 0x00, 0x00,   /* line 16 black cross  */
                        0xFF, 0xFC, 0x7F, 0xFE,   /* line 17 white border */

                        0x00, 0x02, 0x80, 0x00,   /* line 18 */
                        0x00, 0x02, 0x80, 0x00,   /* line 19 */
                        0x00, 0x02, 0x80, 0x00,   /* line 20 */

                        0x00, 0x02, 0x80, 0x00,   /* line 21 */
                        0x00, 0x02, 0x80, 0x00,   /* line 22 */

                        0x00, 0x02, 0x80, 0x00,   /* line 23 */
                        0x00, 0x02, 0x80, 0x00,   /* line 24 */

                        0x00, 0x02, 0x80, 0x00,   /* line 25 */
                        0x00, 0x02, 0x80, 0x00,   /* line 26 */
                        0x00, 0x02, 0x80, 0x00,   /* line 27 */
                        0x00, 0x02, 0x80, 0x00,   /* line 28 */

                        0x00, 0x02, 0x80, 0x00,   /* line 29 */
                        0x00, 0x02, 0x80, 0x00,   /* line 30 */
                        0x00, 0x02, 0x80, 0x00,   /* line 31 */

                        0x00, 0x00, 0x00, 0x00};  /* line 32 */

/* Create a custom cursor at run time. */

  fCursors[kCross] = ::CreateCursor(fHInstance,
                                15,     /* horiz pos of hot spot */
                                15,     /* vert pos of hot spot  */

                                32,     /* cursor width          */
                                32,     /* cursor height         */
                      ANDmaskCursor,    /* AND bitmask           */
                      XORmaskCursor);   /* XOR bitmask           */

/*
 *
 * AND  XOR     Display
 *
 * 0     0       Black
 * 0     1       White
 * 1     0       Screen
 * 1     1       Reverse screen
 *
 */

  fCursor = kCross;

  fhdCursorPen   = (HPEN)  GetStockObject(BLACK_PEN);
  fhdCursorBrush = (HBRUSH)GetStockObject(HOLLOW_BRUSH);  // Brush to draw ROOT locator 3 or 5

//  fhdCommonBrush   = NULL;
//   fhdCommonPen     = NULL;
  fWin32Pen    = new TGWin32Pen;
  fWin32Brush  = new TGWin32Brush;
  fWin32Marker = new TGWin32Marker;
  fhdCommonFont    = NULL;

//
//*-*-Text management
//

  fROOTFont.lfHeight       =   0;
  fROOTFont.lfWidth        =   0;
  fROOTFont.lfEscapement   =   0; // Specifies the angle, in tenths of degrees,
                                  // of each line of text written in the font
                                  // (relative to the bottom of the page).
  fROOTFont.lfOrientation  =   0;   //  (doesn't used by ROOT)
  fROOTFont.lfWeight       = 400;   //   (Normal = 400, BOLD =700)
  fROOTFont.lfItalic       = FALSE;
  fROOTFont.lfUnderline    = FALSE;
  fROOTFont.lfStrikeOut    = FALSE;

  printf(" the current keyboard layout is %d \n", GetOEMCP());

  OSVERSIONINFO OsVersionInfo;

  OsVersionInfo.dwOSVersionInfoSize=sizeof(OSVERSIONINFO);
  GetVersionEx(&OsVersionInfo);
  if (OsVersionInfo.dwMajorVersion >= 4)
  {
      switch (GetOEMCP()) {
         case 708:
         case 709:
         case 710:
         case 720:
         case 864:
              fROOTFont.lfCharSet      = ARABIC_CHARSET;
              break;
         case 737:
              fROOTFont.lfCharSet      = GREEK_CHARSET;
              break;
         case 775:
              fROOTFont.lfCharSet      = BALTIC_CHARSET;
              break;
         case 850:
              fROOTFont.lfCharSet      = ANSI_CHARSET;
              break;
         case 852:
              fROOTFont.lfCharSet      = EASTEUROPE_CHARSET;
              break;
         case 857:
              fROOTFont.lfCharSet      = TURKISH_CHARSET;
              break;
         case 862:
              fROOTFont.lfCharSet      = HEBREW_CHARSET;
              break;
         case 855:
         case 866:
              fROOTFont.lfCharSet      = RUSSIAN_CHARSET;
              break;
         case 874:
              fROOTFont.lfCharSet      = THAI_CHARSET;
              break;
         case 1361:
              fROOTFont.lfCharSet      = JOHAB_CHARSET;
              break;
         default:
              fROOTFont.lfCharSet      = DEFAULT_CHARSET;
              break;
      }
  }
  else
      fROOTFont.lfCharSet      = ANSI_CHARSET;

  fROOTFont.lfOutPrecision = OUT_DEFAULT_PRECIS;
  fROOTFont.lfClipPrecision= CLIP_DEFAULT_PRECIS;
  fROOTFont.lfQuality      = DEFAULT_QUALITY;
  fROOTFont.lfPitchAndFamily=FF_DONTCARE;

  fdwCommonTextAlign = TA_LEFT | TA_BASELINE;


//*-* Allocate enough memory for a logical palette with
//*-* PALETTESIZE entries and set the size and version fields
//*-* of the logical palette structure.
//*-*

  HDC hDC = CreateCompatibleDC(NULL);

//*-*  Check whether we can use palette

  flpPalette = 0;
  fMaxCol    = 0;
//*-*
//*-*   Create a critical section object to synchronize threads
//*-*

  flpCriticalSection = new CRITICAL_SECTION;
  fSectionCount = 0;
  InitializeCriticalSection(flpCriticalSection);
  fWriteLock = CreateEvent(NULL,TRUE,FALSE,NULL);

//*-*  Create event to build a semaphore object

  XW_CreateSemaphore();

  CreatROOTThread();

  return fDisplayOpened;
}

//______________________________________________________________________________
Int_t TGWin32::CreatROOTThread()
{
//*-*-*-*-*Open the display. Return -1 if the opening fails*-*-*-*-*-*-*-*-*
//*-*      ================================================

  //  Make sure that this window hasn't been registered yet

  if (GetClassInfo(fHInstance,fROOTCLASS,&fRoot_Display))
      return 0;

  //
  // Set the common wndClass information. This is common for all windows
  // of this application.
  //

  fRoot_Display.style      = CS_OWNDC | CS_HREDRAW | CS_VREDRAW | CS_DBLCLKS;
  fRoot_Display.cbClsExtra = 0;

  fRoot_Display.cbWndExtra = 0; // sizeof(LONG)+LastExtraMember*sizeof(HANDLE);

  fRoot_Display.hCursor    = NULL;
  fRoot_Display.hInstance  = fHInstance;

  //
  //  Register the main top-level window
  //

  fRoot_Display.lpfnWndProc   = &WndROOT;


  fRoot_Display.hIcon = ((TWinNTSystem *)gSystem)->GetSmallIcon(kMainROOTIcon);
  if (!fRoot_Display.hIcon) fRoot_Display.hIcon = LoadIcon(NULL, IDI_APPLICATION);

//  fRoot_Display.hbrBackground = GetStockObject(WHITE_BRUSH);
  fRoot_Display.hbrBackground = NULL;
  fRoot_Display.lpszMenuName  = NULL;
  fRoot_Display.lpszClassName = fROOTCLASS;


  if (!RegisterClass(&fRoot_Display)) {DWORD l_err = GetLastError();
                                       printf(" Last Error is %d \n", l_err);
                                       return -1;}
  else {

     HANDLE ThrSem;

  //
  //  Create thread to do the msg loop
  //

     ThrSem = CreateSemaphore(NULL, 0, 1, NULL);

//     CreateThread(NULL,0, (LPTHREAD_START_ROUTINE) ROOT_MsgLoop,
//                  (LPVOID) ThrSem, 0,  &fIDThread);

#ifdef _SC_
     _beginthreadex(NULL, 0, (void *) ROOT_MsgLoop,
                  (LPVOID) ThrSem, 0, (void *)&fIDThread);
#else
     _beginthreadex(NULL, 0, ROOT_MsgLoop,
                  (LPVOID) ThrSem, 0,(unsigned int *) &fIDThread);
#endif

     WaitForSingleObject(ThrSem, INFINITE);
     CloseHandle(ThrSem);

     fDisplayOpened = kTRUE;
     return 0;
  }
}
//______________________________________________________________________________
TGWin32::TGWin32() : TVirtualX()
{
//*-*-*-*-*-*-*-*-*-*-*-*Default Constructor *-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ===================
  fSelectedWindow = fPrevWindow = (TGWin32Switch *)(-1);
//  fWindows        = new TObjArray;
}

//______________________________________________________________________________
TGWin32::~TGWin32(){
//*-*-*-*-*-*-*-*-*-*-*-*Default Destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ==================

  if ((int) fSelectedWindow == -1 || !fDisplayOpened) return;

  fWindows.Delete();

  DeleteObject(fhdCommonPalette);
  DeleteObject(fhdCommonFont);

  if (fWin32Pen)    delete fWin32Pen;
  if (fWin32Pen)    delete fWin32Brush;
  if (fWin32Marker) delete fWin32Marker;

//*-*  Cancel the windows message loop

 if (fIDThread) {
    HANDLE ThrSem = CreateSemaphore(NULL, 0, 1, NULL);
    BOOL postresult;
    while (!(postresult = PostThreadMessage(fIDThread,
                                            WM_QUIT,
                                            (WPARAM)0,      // exit code
                                            (LPARAM)ThrSem))
          );
    WaitForSingleObject(ThrSem, INFINITE);
    CloseHandle(ThrSem);
 }

//*-*  Delete the critial section object

  DeleteCriticalSection(flpCriticalSection);
  delete flpCriticalSection;

}

//______________________________________________________________________________
Int_t TGWin32::InitWindow(ULong_t window){

//*-*
//*-*  window must be casted to  TGWin32WindowsObject *winobj
//*-*  if window == 0 InitWindow creates his own instance of  TGWin32WindowsObject object
//*-*
//*-*  Create a new windows
//*-*  Note: All "real" windows go ahead of all "pixmap" object in the 'fWindows' list
//*-*
    TGWin32WindowsObject *winobj = (TGWin32WindowsObject *)window;
    TGWin32Switch *obj = 0;
    if (!winobj) {
        winobj = new TGWin32WindowsObject(this,0,0);
        obj    = new TGWin32Switch(winobj);
        }
    else
        obj =  new TGWin32Switch(winobj, kFALSE); // kFALSE means winobj is an external object

    if (obj) {
        fWindows.AddFirst(obj);
        int parts[] = {43,7,10,39};
        winobj->W32_CreateStatusBar(parts,4);
        winobj->CreateDoubleBuffer();
        winobj->W32_Show();
    }
    else
       Printf("TGWin32::InitWindow error *** \n");
    return (Int_t) obj;
}

//______________________________________________________________________________
Int_t TGWin32::OpenPixmap(UInt_t w, UInt_t h){

//*-*  Create a new pixmap object
//*-*  Note: All "pixmap" windows go behind of all "real window" objects in the 'fWindows' list

    TGWin32Switch *obj =  new TGWin32Switch(new TGWin32PixmapObject(this,w,h));
    if (obj) fWindows.Add(obj);
    return (Int_t) obj;
}

//______________________________________________________________________________
COLORREF TGWin32::ColorIndex(Color_t ic) {
  COLORREF c;
  if (fhdCommonPalette)
       c =  PALETTEINDEX(ic+ColorOffset);
  else {
    PALETTEENTRY palentr = flpPalette->palPalEntry[ic];
    c = *(DWORD *)(&palentr) & 0x00FFFFFF;
  }
  return c;
}

//______________________________________________________________________________
void TGWin32::GetPlanes(Int_t &nplanes)
{
   nplanes = GetDepth();
}

//______________________________________________________________________________
Int_t TGWin32::GetDepth() const
{
   // Get maximum number of planes.
   // nplanes returns the number of bit planes.

   HDC hDCGlobal= CreateCompatibleDC(NULL);

   int nplanes;
   if (GetDeviceCaps(hDCGlobal,RASTERCAPS) & RC_PALETTE != 0)
      nplanes = GetDeviceCaps(hDCGlobal,COLORRES);
   else {
      nplanes=GetDeviceCaps(hDCGlobal,PLANES);
      int nBitsPixel=GetDeviceCaps(hDCGlobal,BITSPIXEL);
      nplanes = nplanes*nBitsPixel;
   }
   ReleaseDC(NULL,hDCGlobal);
   return nplanes;
}

//______________________________________________________________________________
TGWin32Switch *TGWin32::GetSwitchObjectbyId(Int_t ID){
//*-*-*-*Return a pointer to TGWin32 object associated with the present ID *-*-*
//*-*    =================================================================
//*-*  ID     : index of the GWin32SwitchObject object;
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

        return ID ? (TGWin32Switch *)ID : (TGWin32Switch *)(fWindows.First());
}

//______________________________________________________________________________
TGWin32Object *TGWin32::GetMasterObjectbyId(Int_t ID){
//*-*-*Return a pointer to TGWin32 master object associated with the present ID *
//*-*    =================================================================
//*-*  ID  : index of the GWin32SwitchObject object containing a master object
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    TGWin32Switch *swObj = ID ? (TGWin32Switch *)ID : (TGWin32Switch *)(fWindows.First());
    if (swObj) return swObj->GetMasterObject();
    return 0;
}

//______________________________________________________________________________
void  TGWin32::ClearWindow(){
     SafeCallWin32->W32_Clear();
//     fSelectedWindow = 0;
}
//______________________________________________________________________________
void  TGWin32::ClosePixmap(){
     DeleteSelectedObj();
}
//______________________________________________________________________________
void  TGWin32::CloseWindow(){
     DeleteSelectedObj();
}

//______________________________________________________________________________
void  TGWin32::DeleteSelectedObj(){
    if(fSelectedWindow)
    {
      RemoveWindow(fSelectedWindow);
      delete  fSelectedWindow;
      fSelectedWindow = 0;
    }
}

//______________________________________________________________________________
void  TGWin32::CopyPixmap(int wid, int xpos, int ypos){
     SafeCallWin32
       ->W32_CopyTo(GetMasterObjectbyId((Int_t)wid),xpos,ypos);
}
//______________________________________________________________________________
void TGWin32::CreateOpenGLContext(int wid)
{
 // Create OpenGL context for win windows (for "selected" Window by default)
 // printf(" TGWin32::CreateOpenGLContext for wid = %x fSelected= %x, threadID= %d \n",wid,fSelectedWindow,
 //    GetCurrentThreadId());
    if (!wid)
    {
      SafeCallWin32
         ->W32_CreateOpenGL();
    }
    else
    {
      SafeCallW32(((TGWin32Switch *)wid))
         ->W32_CreateOpenGL();
    }

}

//______________________________________________________________________________
void TGWin32::DeleteOpenGLContext(int wid)
{
  // Delete OpenGL context for win windows (for "selected" Window by default)
    if (!wid)
    {
      SafeCallWin32
         ->W32_DeleteOpenGL();
    }
    else
    {
      SafeCallW32(((TGWin32Switch *)wid))
         ->W32_DeleteOpenGL();
    }
}

//______________________________________________________________________________
void  TGWin32::DrawBox(int x1, int y1, int x2, int y2, EBoxMode mode){
     SafeCallWin32
      ->W32_DrawBox(x1, y1, x2, y2, mode);
}
//______________________________________________________________________________
void  TGWin32::DrawCellArray(int x1, int y1, int x2, int y2, int nx, int ny, int *ic){
     SafeCallWin32
      ->W32_DrawCellArray( x1, y1, x2, y2,  nx,  ny,  ic);
}
//______________________________________________________________________________
void  TGWin32::DrawFillArea(int n, TPoint *xy){
     SafeCallWin32
      ->W32_DrawFillArea( n, xy);
}
//______________________________________________________________________________
void  TGWin32::DrawLine(int x1, int y1, int x2, int y2){
     SafeCallWin32
       ->W32_DrawLine( x1, y1, x2, y2);
}
//______________________________________________________________________________
void  TGWin32::DrawPolyLine(int n, TPoint *xy){
     SafeCallWin32
      ->W32_DrawPolyLine(n, xy);
}
//______________________________________________________________________________
void  TGWin32::DrawPolyMarker(int n, TPoint *xy){
     SafeCallWin32
      ->W32_DrawPolyMarker(n, xy);
}
//______________________________________________________________________________
void  TGWin32::DrawText(int x, int y, float angle, float mgn, const char *text, TVirtualX::ETextMode mode){

//*-*-*-*-*-*-*-*-*-*-*Draw a text string using current font*-*-*-*-*-*-*-*-*-*
//*-*                  =====================================
//*-*  mode       : drawing mode
//*-*  mode=0     : the background is not drawn (kClear)
//*-*  mode=1     : the background is drawn (kSolid)
//*-*  x,y        : text position
//*-*  angle      : text angle
//*-*  mgn        : magnification factor
//*-*  text       : text string
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//     char *tr_mess = (char *)malloc(strlen(text));

//     OemToChar(text,tr_mess);       // since Windows code table and Console code table

//*-* We have to check angle to make sure we are setting the right font
     if (fROOTFont.lfEscapement != (LONG) fTextAngle*10)  {
        fTextFontModified=1;
        fROOTFont.lfEscapement   = (LONG) fTextAngle*10;
     }

     if (fTextFontModified) {
        SetWin32Font();
        fTextFontModified = 0;
     }

     SafeCallWin32
      ->W32_DrawText(x, y, angle, mgn, text, mode);

//      free(tr_mess);
}
//______________________________________________________________________________
void  TGWin32::GetCharacterUp(Float_t &chupx, Float_t &chupy){
//*-*-*-*-*-*-*-*-*-*-*-*Return character up vectors*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ===========================
   chupx = fCharacterUpX;
   chupy = fCharacterUpY;
}
//______________________________________________________________________________
Int_t  TGWin32::GetDoubleBuffer(Int_t wid){
   ReturnCallW32(((TGWin32Switch *)wid))
              ->W32_GetDoubleBuffer();
}
//______________________________________________________________________________
void  TGWin32::GetGeometry(int wid, int &x, int &y, unsigned int &w, unsigned int &h){
     if( wid < 0 ) {

      HDC  hDCGlobal= CreateCompatibleDC(NULL);

      x = 0;
      y = 0;
      w = GetDeviceCaps(hDCGlobal,HORZRES);
      h = GetDeviceCaps(hDCGlobal,VERTRES);}

     else {
      SafeCallW32(((TGWin32Switch *)wid))
       ->W32_GetGeometry(x, y, w, h);
     }
}
//______________________________________________________________________________
const char *TGWin32::DisplayName(const char *){ return "localhost"; }

//______________________________________________________________________________
void  TGWin32::GetPixel(int y, int width, Byte_t *scline){
     SafeCallWin32
      ->W32_GetPixel(y, width, scline);
}
//______________________________________________________________________________
void  TGWin32::GetRGB(int index, float &r, float &g, float &b){
const BIGGEST_RGB_VALUE=255;

  if (fSelectedWindow == NoOperation) {  r = g = b = 0; return;}

  if (index >= 0 && index < fMaxCol) {
   if (fhdCommonPalette) {
      SafeCallWin32
       ->W32_GetRGB(index, r, g, b);
   }
   else if (flpPalette) {
     r = (Float_t)(flpPalette->palPalEntry[index].peRed)/BIGGEST_RGB_VALUE;
     g = (Float_t)(flpPalette->palPalEntry[index].peGreen)/BIGGEST_RGB_VALUE;
     b = (Float_t)(flpPalette->palPalEntry[index].peBlue)/BIGGEST_RGB_VALUE;
   }
   else {
      Warning("GetRGB", " There is no color");
      r = g = b = 0;
   }
  }
  else {
      Warning("GetRGB", " There is no color");
      r = g = b = 0;
  }
}
//______________________________________________________________________________
void  TGWin32::GetTextExtent(unsigned int &w, unsigned int &h, char *mess){
//     char *tr_mess = (char *)malloc(strlen(mess));
//     OemToChar(mess,tr_mess);
     SafeCallWin32
      ->W32_GetTextExtent(w, h, mess);
//      free(tr_mess);
}
//______________________________________________________________________________
void  TGWin32::MoveWindow(Int_t wid, Int_t x, Int_t y){
     SafeCallWin32
      ->W32_Move(x, y);
}
//______________________________________________________________________________
void  TGWin32::PutByte(Byte_t b){
     SafeCallWin32
      ->W32_PutByte(b);
}
//______________________________________________________________________________
void  TGWin32::QueryPointer(int &ix, int &iy){;
     SafeCallWin32
      ->W32_QueryPointer(ix, iy);
}
//______________________________________________________________________________
Int_t  TGWin32::RequestLocator(Int_t mode, Int_t ctyp, Int_t &x, Int_t &y){
      ReturnCallWin32
              ->W32_RequestLocator(mode, ctyp, x, y);
}
//______________________________________________________________________________
Int_t  TGWin32::RequestString(int x, int y, char *text){
      ReturnCallWin32
              ->W32_RequestString(x, y, text);
}

//______________________________________________________________________________
void  TGWin32::RescaleWindow(int wid, UInt_t w, UInt_t h){
    SafeCallW32(((TGWin32Switch *)wid))
             ->W32_Rescale(wid,w, h);
}
//______________________________________________________________________________
Int_t  TGWin32::ResizePixmap(int wid, UInt_t w, UInt_t h){
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*Resize a pixmap*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                          ===============
//*-*  wid : pixmap to be resized
//*-*  w,h : Width and height of the pixmap.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
      ReturnCallW32(((TGWin32Switch *)wid))
          ->W32_Rescale(wid, w, h);
}
//______________________________________________________________________________
void  TGWin32::ResizeWindow(int wid){
     SafeCallW32(((TGWin32Switch *)wid))
       ->W32_Resize();
}
//______________________________________________________________________________
void  TGWin32::SelectWindow(int wid){
     if (wid == (int) fSelectedWindow)
                                         return;
     fPrevWindow     = fSelectedWindow;
     if (wid == -1) fSelectedWindow = 0;
     else           fSelectedWindow = (TGWin32Switch *)wid;
     SafeCallWin32
       ->W32_Select();
}
//______________________________________________________________________________
void  TGWin32::SetCharacterUp(Float_t chupx, Float_t chupy){
//*-*-*-*-*-*-*-*-*-*-*-*Set character up vectors*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ========================
   if (chupx == fCharacterUpX  && chupy == fCharacterUpY) return;

   if      (chupx == 0  && chupy == 0)  fTextAngle = 0;
   else if (chupx == 0  && chupy == 1)  fTextAngle = 0;
   else if (chupx == -1 && chupy == 0)  fTextAngle = 90;
   else if (chupx == 0  && chupy == -1) fTextAngle = 180;
   else if (chupx == 1  && chupy ==  0) fTextAngle = 270;
   else {
      fTextAngle = ((TMath::ACos(chupx/TMath::Sqrt(chupx*chupx +chupy*chupy))*180.)/3.14159)-90;
      if (chupy < 0) fTextAngle = 180 - fTextAngle;
      if (TMath::Abs(fTextAngle) < 0.01) fTextAngle = 0;
   }

   fCharacterUpX = chupx;
   fCharacterUpY = chupy;
}
//______________________________________________________________________________
void  TGWin32::SetClipOFF(Int_t wid){
     SafeCallWin32
      ->W32_SetClipOFF();
}
//______________________________________________________________________________
void  TGWin32::SetClipRegion(int wid, int x, int y, UInt_t w, UInt_t h){
     SafeCallW32(((TGWin32Switch *)wid))
      ->W32_SetClipRegion(x, y, w, h);
}
//______________________________________________________________________________
void  TGWin32::SetCursor(Int_t wid, ECursor cursor){
      fCursor = cursor;
}
//______________________________________________________________________________
void  TGWin32::SetDoubleBuffer(int wid, int mode){
//     TGWin32Switch *obj =  new TGWin32Switch(new TGWin32DoubleBufferObject(this));
     SafeCallW32(((TGWin32Switch *)wid))
             ->W32_SetDoubleBuffer(mode);
}
//______________________________________________________________________________
void  TGWin32::SetDoubleBufferOFF(){
     SafeCallWin32
      ->W32_SetDoubleBufferOFF();
}
//______________________________________________________________________________
void  TGWin32::SetDoubleBufferON(){
     SafeCallWin32
      ->W32_SetDoubleBufferON();
}
//______________________________________________________________________________
void  TGWin32::SetDrawMode(TVirtualX::EDrawMode mode){
     SafeCallWin32
      ->W32_SetDrawMode(mode);
}
//______________________________________________________________________________
void  TGWin32::SetFillColor(Color_t cindex){
//*-*-*-*-*-*-*-*-*-*-*Set color index for fill areas*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ==============================
//*-*  cindex : color index defined my IXSETCOL
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

  if (fFillColor == cindex) return;
  fFillColor = cindex;

  if (cindex >=0) fWin32Brush->SetColor(ColorIndex(cindex));
}
//______________________________________________________________________________
void  TGWin32::SetFillStyle(Style_t fstyle){
//*-*-*-*-*-*-*-*-*-*-*Set fill area style*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ===================
//*-*  fstyle   : compound fill area interior style
//*-*
//*-*  fstyle = 1000*interiorstyle + styleindex
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

  if (fFillStyle == fstyle) return;
  fFillStyle = fstyle;

  Int_t style = fstyle/1000;
  Int_t fasi  = fstyle%1000;

  fWin32Brush->SetStyle(style,fasi);
}

//______________________________________________________________________________
void TGWin32::SetFillStyleIndex( Int_t style, Int_t fasi )
{
//*-*-*-*-*-*-*-*-*-*-*Set fill area style index*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =========================
//*-*  style   : fill area interior style hollow or solid
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
  SetFillStyle(1000*style + fasi);
}
//______________________________________________________________________________
void  TGWin32::SetLineColor(Color_t cindex){
//*-*-*-*-*-*-*-*-*-*-*Set color index for lines*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =========================
//*-*  cindex    : color index
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

  if (fLineColor == cindex) return;
  if (cindex < 0) return;
  fLineColor = cindex;

//  fWin32Pen->SetColor(ROOTColorIndex(cindex));

  fWin32Pen->SetColor(ColorIndex(cindex));
}
//______________________________________________________________________________
void  TGWin32::SetLineType(int n, int *dash){
    fWin32Pen->SetType(n,dash);
}
//______________________________________________________________________________
void  TGWin32::SetLineStyle(Style_t linestyle){
//*-*-*-*-*-*-*-*-*-*-*Set line style-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ==============
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   if (fLineStyle != linestyle) { //set style index only if different
      fLineStyle = linestyle;

     fWin32Pen->SetType(-linestyle, NULL);
   }
}
//______________________________________________________________________________
void  TGWin32::SetLineWidth(Width_t width){
//*-*-*-*-*-*-*-*-*-*-*Set line width*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ==============
//*-*  width   : line width in pixels
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

  if (fLineWidth == width) return;
  if( width == 1) fLineWidth = 0;
  else            fLineWidth = width;

  if (fLineWidth < 0) return;

  fWin32Pen->SetWidth(fLineWidth);
}
//______________________________________________________________________________
void  TGWin32::SetMarkerColor( Color_t cindex){
//*-*-*-*-*-*-*-*-*-*-*Set color index for markers*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ===========================
//*-*  cindex : color index defined my IXSETCOL
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

  if (fMarkerColor == cindex) return;
  fMarkerColor = cindex;
  if (cindex < 0) return;
}

//______________________________________________________________________________
void  TGWin32::SetMarkerSize(Float_t markersize){
//*-*-*-*-*-*-*-*-*-*-*Set marker size index for markers*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =================================
//*-*  msize  : marker scale factor
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

  if (markersize == fMarkerSize) return;

  fMarkerSize = markersize;
  if (markersize < 0) return;

  SetMarkerStyle(-fMarkerStyle);
}

//______________________________________________________________________________
void  TGWin32::SetMarkerStyle(Style_t markerstyle){
//*-*-*-*-*-*-*-*-*-*-*Set marker style*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ================

  if (fMarkerStyle == markerstyle) return;
  TPoint shape[15];
  if (markerstyle >= 31) return;
  markerstyle  = TMath::Abs(markerstyle);
  fMarkerStyle = markerstyle;
  Int_t im = Int_t(4*fMarkerSize + 0.5);
  switch (markerstyle) {

case 2:
//*-*--- + shaped marker
     shape[0].SetX(-im); shape[0].SetY( 0);
     shape[1].SetX(im);  shape[1].SetY( 0);
     shape[2].SetX(0) ;  shape[2].SetY( -im);
     shape[3].SetX(0) ;  shape[3].SetY( im);
     SetMarkerType(4,4,shape);
     break;

case 3:
//*-*--- * shaped marker
     shape[0].SetX(-im);  shape[0].SetY(  0);
     shape[1].SetX( im);  shape[1].SetY(  0);
     shape[2].SetX(  0);  shape[2].SetY(-im);
     shape[3].SetX(  0);  shape[3].SetY( im);
     im = Int_t(0.707*Float_t(im) + 0.5);
     shape[4].SetX(-im);  shape[4].SetY(-im);
     shape[5].SetX( im);  shape[5].SetY( im);
     shape[6].SetX(-im);  shape[6].SetY( im);
     shape[7].SetX( im);  shape[7].SetY(-im);
     SetMarkerType(4,8,shape);
     break;

case 4:
case 24:
//*-*--- O shaped marker
     SetMarkerType(0,im*2,shape);
     break;

case 5:
//*-*--- X shaped marker
     im = Int_t(0.707*Float_t(im) + 0.5);
     shape[0].SetX(-im);  shape[0].SetY(-im);
     shape[1].SetX( im);  shape[1].SetY( im);
     shape[2].SetX(-im);  shape[2].SetY( im);
     shape[3].SetX( im);  shape[3].SetY(-im);
     SetMarkerType(4,4,shape);
     break;

case  6:
//*-*--- + shaped marker (with 1 pixel)
     shape[0].SetX(-1);  shape[0].SetY( 0);
     shape[1].SetX( 1);  shape[1].SetY( 0);
     shape[2].SetX( 0);  shape[2].SetY(-1);
     shape[3].SetX( 0);  shape[3].SetY( 1);
     SetMarkerType(4,4,shape);
     break;

case 7:
//*-*--- . shaped marker (with 9 pixel)
     shape[0].SetX(-1);  shape[0].SetY( 1);
     shape[1].SetX( 1);  shape[1].SetY( 1);
     shape[2].SetX(-1);  shape[2].SetY( 0);
     shape[3].SetX( 1);  shape[3].SetY( 0);
     shape[4].SetX(-1);  shape[4].SetY(-1);
     shape[5].SetX( 1);  shape[5].SetY(-1);
     SetMarkerType(4,6,shape);
     break;
case  8:
case 20:
//*-*--- O shaped marker (filled)
     SetMarkerType(1,im*2,shape);
     break;
case 21:      //*-*- here start the old HIGZ symbols
//*-*--- HIGZ full square
     shape[0].SetX(-im);  shape[0].SetY(-im);
     shape[1].SetX( im);  shape[1].SetY(-im);
     shape[2].SetX( im);  shape[2].SetY( im);
     shape[3].SetX(-im);  shape[3].SetY( im);
//     shape[4].SetX(-im);  shape[4].SetY(-im);
     SetMarkerType(3,4,shape);
     break;
case 22:
//*-*--- HIGZ full triangle up
     shape[0].SetX(-im);  shape[0].SetY( im);
     shape[1].SetX( im);  shape[1].SetY( im);
     shape[2].SetX(  0);  shape[2].SetY(-im);
//     shape[3].SetX(-im);  shape[3].SetY( im);
     SetMarkerType(3,3,shape);
     break;
case 23:
//*-*--- HIGZ full triangle down
     shape[0].SetX(  0);  shape[0].SetY( im);
     shape[1].SetX( im);  shape[1].SetY(-im);
     shape[2].SetX(-im);  shape[2].SetY(-im);
//     shape[3].SetX(  0);  shape[3].SetY( im);
     SetMarkerType(3,3,shape);
     break;
case 25:
//*-*--- HIGZ open square
     shape[0].SetX(-im);  shape[0].SetY(-im);
     shape[1].SetX( im);  shape[1].SetY(-im);
     shape[2].SetX( im);  shape[2].SetY( im);
     shape[3].SetX(-im);  shape[3].SetY( im);
//     shape[4].SetX(-im);  shape[4].SetY(-im);
     SetMarkerType(2,4,shape);
     break;
case 26:
//*-*--- HIGZ open triangle up
     shape[0].SetX(-im);  shape[0].SetY( im);
     shape[1].SetX( im);  shape[1].SetY( im);
     shape[2].SetX(  0);  shape[2].SetY(-im);
//     shape[3].SetX(-im);  shape[3].SetY( im);
     SetMarkerType(2,3,shape);
     break;
case 27: {
//*-*--- HIGZ open losange
     Int_t imx = Int_t(2.66*fMarkerSize + 0.5);
     shape[0].SetX(-imx); shape[0].SetY( 0);
     shape[1].SetX(  0);  shape[1].SetY(-im);
     shape[2].SetX(imx);  shape[2].SetY( 0);
     shape[3].SetX(  0);  shape[3].SetY( im);
//     shape[4].SetX(-imx); shape[4].SetY( 0);
     SetMarkerType(2,4,shape);
     break;
}
case 28: {
//*-*--- HIGZ open cross
     Int_t imx = Int_t(1.33*fMarkerSize + 0.5);
     shape[0].SetX(-im);  shape[0].SetY(-imx);
     shape[1].SetX(-imx); shape[1].SetY(-imx);
     shape[2].SetX(-imx); shape[2].SetY( -im);
     shape[3].SetX(imx);  shape[3].SetY( -im);
     shape[4].SetX(imx);  shape[4].SetY(-imx);
     shape[5].SetX( im);  shape[5].SetY(-imx);
     shape[6].SetX( im);  shape[6].SetY( imx);
     shape[7].SetX(imx);  shape[7].SetY( imx);
     shape[8].SetX(imx);  shape[8].SetY( im);
     shape[9].SetX(-imx); shape[9].SetY( im);
     shape[10].SetX(-imx);shape[10].SetY(imx);
     shape[11].SetX(-im); shape[11].SetY(imx);
//     shape[12].SetX(-im); shape[12].SetY(-imx);
     SetMarkerType(2,12,shape);
     break;
    }
case 29: {
//*-*--- HIGZ full star pentagone
     Int_t im1 = Int_t(0.66*fMarkerSize + 0.5);
     Int_t im2 = Int_t(2.00*fMarkerSize + 0.5);
     Int_t im3 = Int_t(2.66*fMarkerSize + 0.5);
     Int_t im4 = Int_t(1.33*fMarkerSize + 0.5);
     shape[0].SetX(-im);  shape[0].SetY( im4);
     shape[1].SetX(-im2); shape[1].SetY(-im1);
     shape[2].SetX(-im3); shape[2].SetY( -im);
     shape[3].SetX(  0);  shape[3].SetY(-im2);
     shape[4].SetX(im3);  shape[4].SetY( -im);
     shape[5].SetX(im2);  shape[5].SetY(-im1);
     shape[6].SetX( im);  shape[6].SetY( im4);
     shape[7].SetX(im4);  shape[7].SetY( im4);
     shape[8].SetX(  0);  shape[8].SetY( im);
     shape[9].SetX(-im4); shape[9].SetY( im4);
//     shape[10].SetX(-im); shape[10].SetY( im4);
     SetMarkerType(3,10,shape);
     break;
    }

case 30: {
//*-*--- HIGZ open star pentagone
     Int_t im1 = Int_t(0.66*fMarkerSize + 0.5);
     Int_t im2 = Int_t(2.00*fMarkerSize + 0.5);
     Int_t im3 = Int_t(2.66*fMarkerSize + 0.5);
     Int_t im4 = Int_t(1.33*fMarkerSize + 0.5);
     shape[0].SetX(-im);  shape[0].SetY( im4);
     shape[1].SetX(-im2); shape[1].SetY(-im1);
     shape[2].SetX(-im3); shape[2].SetY( -im);
     shape[3].SetX(  0);  shape[3].SetY(-im2);
     shape[4].SetX(im3);  shape[4].SetY( -im);
     shape[5].SetX(im2);  shape[5].SetY(-im1);
     shape[6].SetX( im);  shape[6].SetY( im4);
     shape[7].SetX(im4);  shape[7].SetY( im4);
     shape[8].SetX(  0);  shape[8].SetY( im);
     shape[9].SetX(-im4); shape[9].SetY( im4);
     SetMarkerType(2,10,shape);
     break;
}

case 31:
//*-*--- HIGZ +&&x (kind of star)
     SetMarkerType(1,im*2,shape);
     break;
 default:
//*-*--- single dot
     SetMarkerType(0,0,shape);
  }
}

//______________________________________________________________________________
void  TGWin32::SetMarkerType( int type, int n, TPoint *xy ){
//*-*-*-*-*-*-*-*-*-*-*Set marker type*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ===============
//*-*  type      : marker type
//*-*  n         : length of marker description
//*-*  xy        : list of points describing marker shape
//*-*
//*-*     if N.EQ.0 marker is a single point
//*-*     if TYPE.EQ.0 marker is hollow circle of diameter N
//*-*     if TYPE.EQ.1 marker is filled circle of diameter N
//*-*     if TYPE.EQ.2 marker is a hollow polygon describe by line XY
//*-*     if TYPE.EQ.3 marker is a filled polygon describe by line XY
//*-*     if TYPE.EQ.4 marker is described by segmented line XY
//*-*     e.g. TYPE=4,N=4,XY=(-3,0,3,0,0,-3,0,3) sets a plus shape of 7x7 pixels
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
     fWin32Marker->SetMarker(n,xy,type);
}

//______________________________________________________________________________
void  TGWin32::MakePallete(HDC objectDC)
{
  if (!flpPalette)
  {
    int depth;
    HDC oDC = objectDC;
    if (oDC == 0) oDC =  CreateCompatibleDC(NULL);

    int iPalExist = GetDeviceCaps(oDC,RASTERCAPS) & RC_PALETTE ;
    if (iPalExist) {
        depth=GetDeviceCaps(oDC,COLORRES);
    }
    else{
        int nPlanes=GetDeviceCaps(oDC,PLANES);
        int nBitsPixel=GetDeviceCaps(oDC,BITSPIXEL);
        depth = (nPlanes*nBitsPixel);
    }
    if(depth<=8){
       fMaxCol = 256-20;
    }
    else {
       if(depth > 24) depth = 24;
       fMaxCol = 1 << depth;
    }

//*-*  Create palette

      flpPalette = (LPLOGPALETTE) malloc((sizeof (LOGPALETTE) +
                 (sizeof (PALETTEENTRY) * (fMaxCol))));

     if(!flpPalette){
        MessageBox(NULL, "<WM_CREATE> Not enough memory for palette.", NULL,
                   MB_OK | MB_ICONHAND);
        PostQuitMessage (0) ;
     }

     flpPalette->palVersion    = 0x300;
     flpPalette->palNumEntries = fMaxCol;

//*-*  fill in intensities for all palette entry colors

    if (iPalExist) {
//*-*
//*-*  create a logical color palette according the information
//*-*  in the LOGPALETTE structure.
//*-*

      fhdCommonPalette = CreatePalette ((LPLOGPALETTE) flpPalette);
    }
    if (objectDC == 0) DeleteDC(oDC);
  }
}

//______________________________________________________________________________
void  TGWin32::SetRGB(int cindex, float r, float g, float b){
#define BIGGEST_RGB_VALUE 255  // 65535

//  static PALETTEENTRY ChColor;
  if (fSelectedWindow == NoOperation) return;
  if (cindex < 0 ) return;
  else {
   MakePallete();
   if  (cindex < fMaxCol) {

      if (fhdCommonPalette) {
        PALETTEENTRY ChColor;

        ChColor.peRed   = (BYTE) (r*BIGGEST_RGB_VALUE);
        ChColor.peGreen = (BYTE) (g*BIGGEST_RGB_VALUE);
        ChColor.peBlue  = (BYTE) (b*BIGGEST_RGB_VALUE);

        ChColor.peFlags = PC_NOCOLLAPSE;

        SetPaletteEntries(fhdCommonPalette,cindex+ColorOffset,1,&ChColor);
      }
      else if (flpPalette) {
        flpPalette->palPalEntry[cindex].peRed   = (BYTE) (r*BIGGEST_RGB_VALUE);;
        flpPalette->palPalEntry[cindex].peGreen = (BYTE) (g*BIGGEST_RGB_VALUE);;
        flpPalette->palPalEntry[cindex].peBlue  = (BYTE) (b*BIGGEST_RGB_VALUE);;
      }
    }
    else {
        printf(" Error the current Video card doesn't provide %d, Only %d colors are supplied \n",
                 cindex, fMaxCol);
    }
  }
}
//______________________________________________________________________________
void  TGWin32::SetTextAlign(Short_t talign){
//*-*-*-*-*-*-*-*-*-*-*Set text alignment*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ==================
//*-*  txalh   : horizontal text alignment
//*-*  txalv   : vertical text alignment
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

  Int_t txalh = talign/10;
  Int_t txalv = talign%10;

  fTextAlignH = txalh;
  fTextAlignV = txalv;

  fdwCommonTextAlign = 0;
  switch( txalh ) {

  case 2:
    fdwCommonTextAlign |= TA_CENTER;
    break;

  case 3:
    fdwCommonTextAlign |= TA_RIGHT;
    break;

  default:
    fdwCommonTextAlign |= TA_LEFT;
  }


  switch( txalv ) {

  case 1:
    fdwCommonTextAlign |= TA_BASELINE;
    break;

  case 2:
    fdwCommonTextAlign |= TA_TOP;
    break;

  case 3:
    fdwCommonTextAlign |= TA_TOP;
    break;

  default:
    fdwCommonTextAlign |= TA_BASELINE;
  }

}

//______________________________________________________________________________
void  TGWin32::SetTextColor(Color_t cindex){
//*-*-*-*-*-*-*-*-*-*-*Set color index for text*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ========================
//*-*  cindex    : color index defined my IXSETCOL
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

  if (fTextColor == cindex) return;
  fTextColor = cindex;
  if (cindex < 0) return;

}
//______________________________________________________________________________
Int_t  TGWin32::SetTextFont(char *fontname, TVirtualX::ETextSetMode mode){
  return 1;
}
//______________________________________________________________________________
void  TGWin32::SetTextFont(char *fontname, int italic, int bold){

//*-*    mode              : Option message
//*-*    italic   : Italic attribut of the TTF font
//*-*    bold     : Weight attribute of the TTF font
//*-*    fontname : the name of True Type Font (TTF) to draw text.
//*-*
//*-*    Set text font to specified name. This function returns 0 if
//*-*    the specified font is found, 1 if not.

   fROOTFont.lfItalic = (BYTE) italic;
   fROOTFont.lfWeight = (LONG) bold*100;
   fROOTFont.lfHeight = (LONG)(fTextSize*1.1); //mode[2]*1.1; // To account "tail"
 //  fROOTFont.lfWidth  = (LONG) mode[2]*1.2
   fROOTFont.lfEscapement   = (LONG) fTextAngle*10; //(LONG)mode[3];
   fROOTFont.lfOutPrecision = 0;    // (LONG)mode[4];
   strcpy(fROOTFont.lfFaceName,fontname);

   fTextFontModified = 1;
//   SetWin32Font();
}

//______________________________________________________________________________
void  TGWin32::SetTextFont(Font_t fontnumber)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Set current text font number*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ===========================

  fTextFont = fontnumber;

//*-*  List of the currently supported fonts (screen and PostScript)
//*-*  =============================================================
//*-*   Font ID       X11                       Win32 TTF       lfItalic  lfWeight x 10
//*-*        1 : times-medium-i-normal      "Times New Roman"      1           4
//*-*        2 : times-bold-r-normal        "Times New Roman"      0           7
//*-*        3 : times-bold-i-normal        "Times New Roman"      1           7
//*-*        4 : helvetica-medium-r-normal  "Arial"                0           4
//*-*        5 : helvetica-medium-o-normal  "Arial"                1           4
//*-*        6 : helvetica-bold-r-normal    "Arial"                0           7
//*-*        7 : helvetica-bold-o-normal    "Arial"                1           7
//*-*        8 : courier-medium-r-normal    "Courier New"          0           4
//*-*        9 : courier-medium-o-normal    "Courier New"          1           4
//*-*       10 : courier-bold-r-normal      "Courier New"          0           7
//*-*       11 : courier-bold-o-normal      "Courier New"          1           7
//*-*       12 : symbol-medium-r-normal     "Symbol"               0           6
//*-*       13 : times-medium-r-normal      "Times New Roman"      0           4
//*-*       14 :                            "Wingdings"            0           4

 int italic, bold;

 switch(fontnumber/10) {

  case  1:
          italic = 1;
          bold   = 4;
              SetTextFont("Times New Roman", italic, bold);
          break;
  case  2:
          italic = 0;
          bold   = 7;
              SetTextFont("Times New Roman", italic, bold);
          break;
  case  3:
          italic = 1;
          bold   = 7;
              SetTextFont("Times New Roman", italic, bold);
          break;
  case  4:
          italic = 0;
          bold   = 4;
              SetTextFont("Arial", italic, bold);
          break;
  case  5:
          italic = 1;
          bold   = 4;
                  SetTextFont("Arial", italic, bold);
          break;
  case  6:
          italic = 0;
          bold   = 7;
                  SetTextFont("Arial", italic, bold);
          break;
  case  7:
          italic = 1;
          bold   = 7;
                  SetTextFont("Arial", italic, bold);
          break;
  case  8:
          italic = 0;
          bold   = 4;
                  SetTextFont("Courier New", italic, bold);
          break;
  case  9:
          italic = 1;
          bold   = 4;
                  SetTextFont("Courier New", italic, bold);
          break;
  case 10:
          italic = 0;
          bold   = 7;
                  SetTextFont("Courier New", italic, bold);
          break;
  case 11:
          italic = 1;
          bold   = 7;
                  SetTextFont("Courier New", italic, bold);
          break;
  case 12:
          italic = 0;
          bold   = 6;
                  SetTextFont("Symbol", italic, bold);
          break;
  case 13:
          italic = 0;
          bold   = 4;
                  SetTextFont("Times New Roman", italic, bold);
          break;
  case 14:
          italic = 0;
          bold   = 4;
                  SetTextFont("Wingdings", italic, bold);
          break;
  default:
          italic = 0;
          bold   = 4;
                  SetTextFont("Times New Roman", italic, bold);
          break;

 }

}
//______________________________________________________________________________
void  TGWin32::SetTextSize(Float_t textsize){
//*-*-*-*-*-*-*-*-*-*-*-*-*Set current text size*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =====================

  fTextSize = textsize;
  fROOTFont.lfHeight = (LONG)(fTextSize*1.1); //*1.1; // To account "tail"

  fTextFontModified = 1;
//  SetWin32Font();
}

//______________________________________________________________________________
void  TGWin32::SetTitle(const char *title){
//*-*-*-*-*-*-*-*-*-*-*-*-*Set title of the object*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =======================
     SafeCallWin32
      ->W32_SetTitle(title);
}

//______________________________________________________________________________
void  TGWin32::UpdateWindow(int mode){
     SafeCallWin32
      ->W32_Update(mode);
}
//______________________________________________________________________________
void  TGWin32::SetWin32Font(){

  if (fhdCommonFont != NULL) DeleteObject(fhdCommonFont);
  fhdCommonFont = CreateFontIndirect(&fROOTFont);
}
//______________________________________________________________________________
void  TGWin32::Warp(int ix, int iy){
     SafeCallWin32
      ->W32_Warp(ix, iy);
}
//______________________________________________________________________________
Int_t  TGWin32::WriteGIF(char *name){
     //SafeCallWin32
     fSelectedWindow->W32_WriteGIF(name);
     return 0;
}
//______________________________________________________________________________
void  TGWin32::WritePixmap(int wid, UInt_t w, UInt_t h, char *pxname){
     SafeCallWin32
      ->W32_WritePixmap(w, h, pxname);
}

//______________________________________________________________________________
Bool_t TGWin32::IsCmdThread()
{
    return fIDThread == GetCurrentThreadId() ? kTRUE : kFALSE;
}
//______________________________________________________________________________
UInt_t TGWin32::ExecCommand(TGWin32Command  *command)
{
// To exec a command coming from the other threads
 BOOL postresult;
 if (IsCmdThread())
     Warning("TGWin32::ExecCommand","The dead lock danger");

 while (!(postresult = PostThreadMessage(fIDThread,
                             ROOT_CMD,                     // IX11_ROOT_MSG,
                             (WPARAM)command->GetCOP(),
                             (LPARAM)command))
       ){ printf(" TGWin32::ExecCommand Error %d %d \n",postresult, GetLastError()); }
 return (UInt_t) postresult;
}

//______________________________________________________________________________
void TGWin32::EnterCrSection(){
//*-* In fact this is a LOCK function
  return;
  write_lock();
}

//______________________________________________________________________________
void TGWin32::LeaveCrSection(){
    return;
    release_write_lock();
}

//______________________________________________________________________________
void TGWin32::write_lock (){
   return;
 top:
    EnterCriticalSection(flpCriticalSection);
    if (fSectionCount) {
        LeaveCriticalSection(flpCriticalSection);
        WaitForSingleObject(fWriteLock,INFINITE);
        goto top;
    }
}

//______________________________________________________________________________
void TGWin32::release_write_lock(){
      return;

    LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::read_lock(){
      return;

    EnterCriticalSection(flpCriticalSection);
    fSectionCount++;
    ResetEvent(fWriteLock);
    LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::release_read_lock(){
      return;

    EnterCriticalSection(flpCriticalSection);
    if (--fSectionCount == 0)
           SetEvent(fWriteLock);
    LeaveCriticalSection(flpCriticalSection);
}


//______________________________________________________________________________
void TGWin32::XW_CreateSemaphore(){
        fhEvent = CreateEvent(NULL,TRUE,FALSE,NULL);
}
//______________________________________________________________________________
void TGWin32::XW_OpenSemaphore(){
        SetEvent(fhEvent);
}
//______________________________________________________________________________
void TGWin32::XW_CloseSemaphore(){
        ResetEvent(fhEvent);
}
//______________________________________________________________________________
void TGWin32::XW_WaitSemaphore(){
    WaitForSingleObject(fhEvent, INFINITE);
}
