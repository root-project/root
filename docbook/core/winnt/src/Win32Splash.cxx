// @(#)root/winnt:$Id$
// Author: Bertrand Bellenot   30/07/02

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifdef WIN32
#include "Windows4Root.h"
#include "RVersion.h"
#include "strlcpy.h"
#include <stdlib.h>
#include <stdio.h>
#include <ocidl.h>
#include <olectl.h>

#define ID_SPLASHSCREEN      25

static const char *gConception[] = {
   "Rene Brun",
   "Fons Rademakers",
   0
};

static const char *gLeadDevelopers[] = {
   "Rene Brun",
   "Philippe Canal",
   "Fons Rademakers",
   0
};

static const char *gRootDevelopers[] = {
   "Ilka Antcheva",
   "Maarten Ballintijn",
   "Bertrand Bellenot",
   "Olivier Couet",
   "Valery Fine",
   "Gerardo Ganis",
   "Eddy Offermann",
   "Valeriy Onuchin",
   0
};

static const char *gCintDevelopers[] = {
   "Masaharu Goto",
   0
};

static const char *gRootDocumentation[] = {
   "Ilka Antcheva",
   "Suzanne Panacek",
   0
};

static char **gContributors = 0;

typedef struct tagImgInfo {
   IPicture *Ipic;
   SIZE sizeInHiMetric;
   SIZE sizeInPix;
   char *Path;
} IMG_INFO;

static IMG_INFO gImageInfo;

///////////////////////////////////////////////////////////////////////////////
// Global Variables:
static HINSTANCE    gInst; // Current instance
static HWND         gSplashWnd = 0; // Splash screen
static BOOL         gShow = FALSE;
static DWORD        gDelayVal = 0;
static HDC          gDCScreen = 0, gDCCredits = 0;
static HBITMAP      gBmpScreen = 0, gBmpOldScreen = 0;
static HBITMAP      gBmpCredits = 0, gBmpOldCredits = 0;
static HRGN         gRgnScreen = 0;
static int          gCreditsBmpWidth;
static int          gCreditsBmpHeight;

static bool         gStayUp        = true;
static bool         gAbout         = false;
static RECT         gCreditsRect   = { 15, 155, 305, 285 }; // clip rect in logo
static unsigned int gCreditsWidth  = gCreditsRect.right - gCreditsRect.left; // credits pixmap size
static unsigned int gCreditsHeight = gCreditsRect.bottom - gCreditsRect.top; // credits rect height

static void ReadContributors()
{
   // Read the file $ROOTSYS/README/CREDITS for the names of the
   // contributors.

   char buf[2048];
#ifdef ROOTDOCDIR
   sprintf(buf, "%s/CREDITS", ROOTDOCDIR);
#else
   sprintf(buf, "%s/README/CREDITS", getenv("ROOTSYS"));
#endif

   gContributors = 0;

   FILE *f = fopen(buf, "r");
   if (!f) return;

   int cnt = 0;
   while (fgets(buf, sizeof(buf), f)) {
      if (!strncmp(buf, "N: ", 3)) {
         cnt++;
      }
   }
   gContributors = new char*[cnt+1];

   cnt = 0;
   rewind(f);
   while (fgets(buf, sizeof(buf), f)) {
      if (!strncmp(buf, "N: ", 3)) {
         int len = strlen(buf);
         buf[len-1] = 0;    // remove \n
         len -= 3;          // remove "N: "
         gContributors[cnt] = new char[len];
         strlcpy(gContributors[cnt], buf+3, len);
         cnt++;
      }
   }
   gContributors[cnt] = 0;

   fclose(f);
}

static void DrawVersion(HDC hDC)
{
   // Draw version string.
   RECT drawRect;
   SIZE lpSize;
   char version[80];
   int  Height;
   sprintf(version, "Version %s", ROOT_RELEASE);

   Height = gImageInfo.sizeInPix.cy;

   GetTextExtentPoint32(hDC, version, strlen(version), &lpSize);

   drawRect.left = 15;
   drawRect.top = Height - 25;
   drawRect.right = 15 + lpSize.cx;
   drawRect.bottom = drawRect.top + lpSize.cy;
   DrawTextEx(hDC, version, strlen(version), &drawRect, DT_LEFT, 0);
}

static int DrawCreditItem(HDC hDC, const char *creditItem, const char **members,
                          int y, bool draw)
{
   // Draw credit item.

   char credit[1024];
   SIZE lpSize1, lpSize2;
   RECT drawRect;
   TEXTMETRIC lptm;
   int i;
   int lineSpacing;

   GetTextMetrics(hDC, &lptm);

   lineSpacing = lptm.tmAscent + lptm.tmDescent;

   strcpy(credit, creditItem);
   for (i = 0; members && members[i]; i++) {
      if (i) strcat(credit, ", ");
      GetTextExtentPoint32(hDC, credit, strlen(credit), &lpSize1);
      GetTextExtentPoint32(hDC, members[i], strlen(members[i]), &lpSize2);
      if((lpSize1.cx + lpSize2.cx) > (int) gCreditsWidth) {
         drawRect.left = 0;
         drawRect.top = y;
         drawRect.right = gCreditsRect.right;
         drawRect.bottom = y + lineSpacing;
         if (draw)
            DrawTextEx(hDC, credit, strlen(credit), &drawRect, DT_LEFT, 0);
         y += lineSpacing;
         strcpy(credit, "   ");
      }
      strcat(credit, members[i]);
   }
   drawRect.left = 0;
   drawRect.top = y;
   drawRect.right = gCreditsRect.right;
   drawRect.bottom = y + lineSpacing;
   if (draw)
      DrawTextEx(hDC, credit, strlen(credit), &drawRect, DT_LEFT, 0);

   return y;
}

static int DrawCredits(HDC hDC, bool draw, bool extended)
{
   // Draw credits. If draw is true draw credits,
   // otherwise just return size of all credit text.
   // If extended is true draw or returns size for extended full
   // credits list.

   char user_name[256];
   TEXTMETRIC lptm;
   DWORD length = sizeof (user_name);
   int lineSpacing, y;

   GetTextMetrics(hDC, &lptm);

   lineSpacing = lptm.tmAscent + lptm.tmDescent;
   y = 0; // 140
   y = DrawCreditItem(hDC, "Conception: ", gConception, y, draw);
   y += 2 * lineSpacing - 3;
   y = DrawCreditItem(hDC, "Lead Developers: ", gLeadDevelopers, y, draw);
   y += 2 * lineSpacing - 3;  // special layout tweak
   y = DrawCreditItem(hDC, "Core Engineering: ", gRootDevelopers, y, draw);
   y += 2 * lineSpacing - 3;  // to just not cut the bottom of the "p"
   y = DrawCreditItem(hDC, "CINT C/C++ Intepreter: ", gCintDevelopers, y, draw);
   y += 2 * lineSpacing - 3;
   y = DrawCreditItem(hDC, "Documentation: ", gRootDocumentation, y, draw);

   if (extended && gContributors) {
      y += 2 * lineSpacing;
      y = DrawCreditItem(hDC, "Contributors: ", (const char **)gContributors, y, draw);

      y += 2 * lineSpacing;
      y = DrawCreditItem(hDC, "Our sincere thanks and apologies to anyone who deserves", 0, y, draw);
      y += lineSpacing;
      y = DrawCreditItem(hDC, "credit but fails to appear in this list.", 0, y, draw);

      if (GetUserName (user_name, &length)) {
         char *name = new char [strlen(user_name)+1];
         strcpy(name, user_name);
         char *s = strchr(name, ',');
         if (s) *s = 0;
         char line[1024];
         sprintf(line, "Extra special thanks go to %s,", name);
         delete [] name;
         y += 2 * lineSpacing;
         y = DrawCreditItem(hDC, line, 0, y, draw);
         y += lineSpacing;
         y = DrawCreditItem(hDC, "one of our favorite users.", 0, y, draw);
      }
   }
   return y;
}

void CreateCredits(HDC hDC, bool extended)
{
   HFONT   hFont, hOldFont;
   HBRUSH  hBr;
   LOGFONT lf;
   RECT    fillRect;

   gRgnScreen = CreateRectRgnIndirect(&gCreditsRect);
   SelectClipRgn(hDC, gRgnScreen);

   gDCScreen = CreateCompatibleDC(hDC);
   gBmpScreen = CreateCompatibleBitmap(hDC, (gCreditsRect.right - gCreditsRect.left),
                                      (gCreditsRect.bottom - gCreditsRect.top) );
   gBmpOldScreen = (HBITMAP)SelectObject(gDCScreen, gBmpScreen);

   gDCCredits = CreateCompatibleDC(hDC);

   gCreditsBmpWidth = (gCreditsRect.right - gCreditsRect.left);
   gCreditsBmpHeight = DrawCredits(gDCCredits, false, extended);

   gBmpCredits = CreateCompatibleBitmap(gDCCredits, gCreditsBmpWidth, gCreditsBmpHeight);
   gBmpOldCredits = (HBITMAP)SelectObject(gDCCredits, gBmpCredits);

   hBr = CreateSolidBrush(RGB(255,255,255));
   fillRect.top = fillRect.left = 0;
   fillRect.bottom = gCreditsBmpHeight;
   fillRect.right = gCreditsBmpWidth;
   FillRect(gDCCredits, &fillRect, hBr);

   memset((void*)&lf, 0, sizeof(lf));
   lf.lfHeight = 14;
   lf.lfWeight = 400;
   lf.lfQuality = NONANTIALIASED_QUALITY;
   strcpy(lf.lfFaceName, "Arial");
   hFont = CreateFontIndirect(&lf);

   if(hFont)
      hOldFont = (HFONT)SelectObject(gDCCredits, hFont);

   SetBkMode(gDCCredits, TRANSPARENT);
   SetTextColor(gDCCredits, 0x00000000);

   DrawCredits(gDCCredits, true, extended);

   SetBkColor(gDCCredits, 0x00FFFFFF);
   SelectObject(gDCCredits, hOldFont);
}

void ScrollCredits(BOOL extended)
{
   // Track scroll position

   static int nScrollY = 0;

   if (!gShow)
      return;
   if (!IsWindow(gSplashWnd))
      return;
   HDC hDC = GetDC(gSplashWnd);

   if(gDCCredits == 0) {
      CreateCredits(hDC, extended);
      nScrollY = 0;
   }

   BitBlt(gDCScreen, 0, 0, gCreditsBmpWidth, gCreditsBmpHeight, gDCCredits,
          0, nScrollY, SRCCOPY);
   BitBlt(hDC, gCreditsRect.left, gCreditsRect.top,
          (gCreditsRect.right - gCreditsRect.left),
          (gCreditsRect.bottom - gCreditsRect.top),
          gDCScreen, 0, 0, SRCCOPY);

   GdiFlush();

   if (extended) {
      // delay scrolling by the specified time
      Sleep(100);

      if (nScrollY == 0)
         Sleep(2000);

      // continue scrolling
      nScrollY += 1;
      if (nScrollY > (int) (gCreditsBmpHeight - 2*gCreditsHeight))
         nScrollY = -int(gCreditsHeight);
   }
}

///////////////////////////////////////////////////////////////////////////////
// Foward declarations of functions included in this code module:
ATOM                MyRegisterClass(HINSTANCE hInstance);
LRESULT CALLBACK    SplashWndProc(HWND, UINT, WPARAM, LPARAM);


void *OpenGraphic(char *name)
{
   IPicture *Ipic = 0;
   SIZE sizeInHiMetric,sizeInPix;
   const int HIMETRIC_PER_INCH = 2540;
   HDC hDCScreen = GetDC(0);
   HRESULT hr;
   int nPixelsPerInchX = GetDeviceCaps(hDCScreen, LOGPIXELSX);
   int nPixelsPerInchY = GetDeviceCaps(hDCScreen, LOGPIXELSY);
   wchar_t OlePathName[512];

   ReleaseDC(0, hDCScreen);
   mbstowcs(OlePathName,name,strlen(name)+1);
   hr = OleLoadPicturePath(OlePathName, 0, 0, 0, IID_IPicture,
                           (void **)(&Ipic));
   if (hr)
      return 0;
   if (Ipic) {
      // get width and height of picture
      hr = Ipic->get_Width(&sizeInHiMetric.cx);
      if (!SUCCEEDED(hr))
         goto err;
      Ipic->get_Height(&sizeInHiMetric.cy);
      if (!SUCCEEDED(hr))
         goto err;

      // convert himetric to pixels
      sizeInPix.cx = (nPixelsPerInchX * sizeInHiMetric.cx +
                      HIMETRIC_PER_INCH / 2) / HIMETRIC_PER_INCH;
      sizeInPix.cy = (nPixelsPerInchY * sizeInHiMetric.cy +
                      HIMETRIC_PER_INCH / 2) / HIMETRIC_PER_INCH;
      gImageInfo.sizeInPix = sizeInPix;
      gImageInfo.sizeInHiMetric = sizeInHiMetric;
      gImageInfo.Ipic = Ipic;
      gImageInfo.Path = name;
      return Ipic;
   }
err:
   return 0;
}

void DisplayGraphic(HWND hwnd,HDC pDC)
{
   IPicture *Ipic = gImageInfo.Ipic;
   DWORD dwAttr = 0;
   HBITMAP Bmp,BmpOld;
   RECT rc;
   HRESULT hr;
   HPALETTE pPalMemOld;

   if (Ipic != 0) {
      // get palette
      OLE_HANDLE hPal = 0;
      HPALETTE hPalOld=0, hPalMemOld=0;
      hr = Ipic->get_hPal(&hPal);

      if (!SUCCEEDED(hr))
         return;
      if (hPal != 0) {
         hPalOld = SelectPalette(pDC,(HPALETTE)hPal,FALSE);
         RealizePalette(pDC);
      }

      // Fit the image to the size of the client area. Change this
      // For more sophisticated scaling
      GetClientRect(hwnd,&rc);
      // transparent?
      if (SUCCEEDED(Ipic->get_Attributes(&dwAttr)) ||
          (dwAttr & PICTURE_TRANSPARENT)) {
         // use an off-screen DC to prevent flickering
         HDC MemDC = CreateCompatibleDC(pDC);
         Bmp = CreateCompatibleBitmap(pDC,gImageInfo.sizeInPix.cx,gImageInfo.sizeInPix.cy);

         BmpOld = (HBITMAP)SelectObject(MemDC,Bmp);
         pPalMemOld = 0;
         if (hPal != 0) {
            hPalMemOld = SelectPalette(MemDC,(HPALETTE)hPal, FALSE);
            RealizePalette(MemDC);
         }

         // display picture using IPicture::Render
         hr = Ipic->Render(MemDC, 0, 0, rc.right, rc.bottom, 0,
                           gImageInfo.sizeInHiMetric.cy,
                           gImageInfo.sizeInHiMetric.cx,
                           -gImageInfo.sizeInHiMetric.cy, &rc);

         BitBlt(pDC,0, 0, gImageInfo.sizeInPix.cx, gImageInfo.sizeInPix.cy,
                MemDC, 0, 0, SRCCOPY);

         SelectObject(MemDC,BmpOld);

         if (pPalMemOld)
            SelectPalette(MemDC,pPalMemOld, FALSE);
         DeleteObject(Bmp);
         DeleteDC(MemDC);

      }
      else {
         // display picture using IPicture::Render
         Ipic->Render(pDC, 0, 0, rc.right, rc.bottom, 0,
                      gImageInfo.sizeInHiMetric.cy,
                      gImageInfo.sizeInHiMetric.cx,
                      -gImageInfo.sizeInHiMetric.cy, &rc);
      }

      if (hPalOld != 0)
         SelectPalette(pDC,hPalOld, FALSE);
      if (hPal)
         DeleteObject((HPALETTE)hPal);
   }
}

void CloseImage(void *Ipict)
{
   IPicture *ip = (IPicture *)Ipict;

   if (ip == 0)
      ip = gImageInfo.Ipic;
   if (ip == 0)
      return;
   ip->Release();
   memset(&gImageInfo,0,sizeof(gImageInfo));
}

////////////////////////////////////////////////////////////////////////////////
// Splashscreen functions
////////////////////////////////////////////////////////////////////////////////
//
//
void DestroySplashScreen()
{
   // Destroy the window
   if (IsWindow(gSplashWnd)) {
      DestroyWindow(gSplashWnd);
      gSplashWnd = 0;
      UnregisterClass("RootSplashScreen", gInst);
   }
   if(gDCScreen != 0 && gBmpOldScreen != 0) {
      SelectObject(gDCScreen, gBmpOldScreen);
      DeleteObject(gBmpScreen);
   }
   if(gDCCredits != 0 && gBmpOldCredits != 0) {
      SelectObject(gDCCredits, gBmpOldCredits);
      DeleteObject(gBmpCredits);
   }
   DeleteDC(gDCCredits);
   gDCCredits = 0;
   DeleteDC(gDCScreen);
   gDCScreen = 0;
   CloseImage(gImageInfo.Ipic);
}

////////////////////////////////////////////////////////////////////////////////
//
//
BOOL CreateSplashScreen(HWND hParent)
{
   int xScreen;
   int yScreen;
   // Crenter the splashscreen
   xScreen = GetSystemMetrics(SM_CXFULLSCREEN);
   yScreen = GetSystemMetrics(SM_CYFULLSCREEN);

   gStayUp = true;

   gSplashWnd = CreateWindowEx(
            WS_EX_TOOLWINDOW,
            "RootSplashScreen",
            0,
            WS_POPUP | WS_VISIBLE,
            (xScreen - 360)/2,
            (yScreen - 240)/2,
            360, 240,
            hParent,
            0,
            gInst,
            0);

   return (gSplashWnd != 0);
}


////////////////////////////////////////////////////////////////////////////////
//
//
BOOL PreTranslateMessage(MSG* pMsg)
{
   if (!IsWindow(gSplashWnd))
      return FALSE;

   // If we get a keyboard or mouse message, hide the splash screen.
   if (pMsg->message == WM_KEYDOWN ||
       pMsg->message == WM_SYSKEYDOWN ||
       pMsg->message == WM_LBUTTONDOWN ||
       pMsg->message == WM_RBUTTONDOWN ||
       pMsg->message == WM_MBUTTONDOWN ||
       pMsg->message == WM_NCLBUTTONDOWN ||
       pMsg->message == WM_NCRBUTTONDOWN ||
       pMsg->message == WM_NCMBUTTONDOWN) {
      DestroySplashScreen();
      return TRUE;    // message handled here
   }
   return FALSE;   // message not handled
}

void CreateSplash(DWORD time, BOOL extended)
{
   MSG msg;
   MyRegisterClass(gInst);
   gShow = FALSE;
   if(extended)
      gAbout = true;
   if(time > 0) {
      gDelayVal = time * 1000;
   }
   else return;

   if (extended)
      ReadContributors();

   // Create the splash screen
   CreateSplashScreen(0);

   // Main message loop:
   while (gStayUp) {
      if(PeekMessage(&msg, 0, 0, 0, PM_REMOVE)) {
         PreTranslateMessage(&msg);
         TranslateMessage(&msg);
         DispatchMessage(&msg);
      }
      if(gShow) {
         if(extended) {
            ScrollCredits(extended);
         }
         else {
            ScrollCredits(extended);
            gShow = false;
         }
      }
   }

   DestroySplashScreen();
}

///////////////////////////////////////////////////////////////////////////////
//
//  FUNCTION: MyRegisterClass()
//
//  PURPOSE: Registers the window class.
//
//  COMMENTS:
//
//    This function and its usage is only necessary if you want this code
//    to be compatible with Win32 systems prior to the 'RegisterClassEx'
//    function that was added to Windows 95. It is important to call this function
//    so that the application will get 'well formed' small icons associated
//    with it.
//
ATOM MyRegisterClass(HINSTANCE hInstance)
{
   WNDCLASSEX wcex;

   wcex.cbSize = sizeof(WNDCLASSEX);

   wcex.style          = CS_HREDRAW | CS_VREDRAW;
   wcex.lpfnWndProc    = (WNDPROC)SplashWndProc;
   wcex.cbClsExtra     = 0;
   wcex.cbWndExtra     = 0;
   wcex.hInstance      = hInstance;
   wcex.hIcon          = 0;
   wcex.hCursor        = LoadCursor(0, IDC_ARROW);
   wcex.hbrBackground  = 0;
   wcex.lpszMenuName   = 0;
   wcex.lpszClassName  = "RootSplashScreen";
   wcex.hIconSm        = 0;
   return RegisterClassEx(&wcex);
}



////////////////////////////////////////////////////////////////////////////
// Message handler for splash screen.
//
LRESULT CALLBACK SplashWndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
   PAINTSTRUCT ps;
   HDC hDC;
   LOGFONT lf;
   static HFONT hFont;
   const char bmpDir[] = "\\icons\\Splash.gif";
   HWND hwndFound;         // this is what is returned to the caller
   char pszNewWindowTitle[1024]; // contains fabricated WindowTitle
   char pszOldWindowTitle[1024]; // contains original WindowTitle
   char FullBmpDir[256];
   char *RootSysDir;
   int xScreen;
   int yScreen;

   switch (message) {
      case WM_CREATE:
         if(!gAbout)
            SetTimer(hWnd, ID_SPLASHSCREEN, gDelayVal, 0);
         RootSysDir = getenv("ROOTSYS");
         sprintf(FullBmpDir,"%s%s",RootSysDir,bmpDir);
         // Retrieve a handle identifying the file.
         OpenGraphic(FullBmpDir);
         hDC = GetDC(hWnd);
         DisplayGraphic(hWnd, hDC);
         SetBkMode(hDC, TRANSPARENT);
         memset((void*)&lf, 0, sizeof(lf));
         lf.lfHeight = 14;
         lf.lfWeight = 400;
         lf.lfQuality = NONANTIALIASED_QUALITY;
         strcpy(lf.lfFaceName, "Arial");
         hFont = CreateFontIndirect(&lf);
         xScreen = GetSystemMetrics(SM_CXFULLSCREEN);
         yScreen = GetSystemMetrics(SM_CYFULLSCREEN);
         SetWindowPos(hWnd, HWND_TOPMOST, (xScreen - gImageInfo.sizeInPix.cx)/2,
                      (yScreen - gImageInfo.sizeInPix.cy)/2, gImageInfo.sizeInPix.cx,
                      gImageInfo.sizeInPix.cy, 0 );
         break;

      case WM_TIMER:
         if (wParam == ID_SPLASHSCREEN) {
            KillTimer (hWnd, ID_SPLASHSCREEN);
            DestroySplashScreen();
         }
         break;

      case WM_DESTROY:
         gStayUp = false;
         PostQuitMessage(0);

      case WM_PAINT:
         hDC = BeginPaint(hWnd, &ps);
         RECT rt;
         GetClientRect(hWnd, &rt);
         RootSysDir = getenv("ROOTSYS");
         sprintf(FullBmpDir,"%s%s",RootSysDir,bmpDir);
         OpenGraphic(FullBmpDir);
         hDC = GetDC(hWnd);
         DisplayGraphic(hWnd, hDC);
         SetBkMode(hDC, TRANSPARENT);
         if(hFont)
            SelectObject(hDC, hFont);
         DrawVersion(hDC);
         EndPaint(hWnd, &ps);
         gShow = TRUE;
         // fetch current window title
         GetConsoleTitle(pszOldWindowTitle, 1024);
         // format a "unique" NewWindowTitle
         wsprintf(pszNewWindowTitle,"%d/%d", GetTickCount(), GetCurrentProcessId());
         // change current window title
         SetConsoleTitle(pszNewWindowTitle);
         // ensure window title has been updated
         Sleep(40);
         // look for NewWindowTitle
         hwndFound=FindWindow(NULL, pszNewWindowTitle);
         // restore original window title
         ShowWindow(hwndFound, SW_RESTORE);
         SetForegroundWindow(hwndFound);
         SetConsoleTitle("ROOT session");
         break;

      default:
         return DefWindowProc(hWnd, message, wParam, lParam);
   }
   return 0;
}
#endif
