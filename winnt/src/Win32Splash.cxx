// @(#)root/winnt:$Name:  $:$Id: Win32Splash.cxx,v 1.6 2003/10/10 17:01:54 brun Exp $
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
#include "RConfig.h"
#include <stdlib.h>
#include <stdio.h>
#include <ocidl.h>
#include <olectl.h>

#define ID_SPLASHSCREEN      25
#define MY_BUFSIZE         1024 // buffer size for console window titles
// define mask color
#define MASK_RGB	(COLORREF)0xFFFFFF

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
   "Suzanne Panachek",
   0
};

static char **gContributors = 0;

typedef struct tagImgInfo {
    IPicture *Ipic;
    SIZE sizeInHiMetric;
    SIZE sizeInPix;
    char *Path;
} IMG_INFO;

static IMG_INFO ImageInfo;

///////////////////////////////////////////////////////////////////////////////
// Global Variables:
HINSTANCE hInst;                        // current instance
HWND    hSplashWnd = NULL;               // Splash screen
BOOL    bShow = FALSE;
DWORD   DelayVal = 0;
HDC     hdcScreen = NULL, hdcCredits = NULL;
HDC     hdcBk = NULL, hdcMask = NULL;
HBITMAP hbmpBk = NULL, hbmpOldBk = NULL;
HBITMAP hbmpScreen = NULL, hbmpOldScreen = NULL;
HBITMAP hbmpCredits = NULL, hbmpOldCredits = NULL;
HBITMAP hbmpMask = NULL, hbmpOldMask = NULL;
HRGN    hrgnScreen = NULL;
int     CreditsBmpWidth;
int     CreditsBmpHeight;

static bool         gStayUp        = true;
static bool         gAbout         = false;
static RECT         gCreditsRect   = { 15, 155, 300, 285 }; // clip rect in logo
static unsigned int gCreditsWidth  = gCreditsRect.right - gCreditsRect.left; // credits pixmap size
static unsigned int gCreditsHeight = 0;

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
         strncpy(gContributors[cnt], buf+3, len);
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

   Height = ImageInfo.sizeInPix.cy;

   GetTextExtentPoint32(hDC, version, strlen(version), &lpSize);

   drawRect.left = 15;
   drawRect.top = Height - 25;
   drawRect.right = 15 + lpSize.cx;
   drawRect.bottom = drawRect.top + lpSize.cy;
   DrawTextEx(hDC, version, strlen(version), &drawRect, DT_LEFT, NULL);
//   TextOut(hDC, 15, Height - 25, version, strlen(version));
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
             DrawTextEx(hDC, credit, strlen(credit), &drawRect, DT_LEFT, NULL);
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
      DrawTextEx(hDC, credit, strlen(credit), &drawRect, DT_LEFT, NULL);

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
      y = DrawCreditItem(hDC, "Special thanks to the neglected families and friends", 0, y, draw);
      y += lineSpacing;
      y = DrawCreditItem(hDC, "of the aforementioned persons.", 0, y, draw);

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
    SIZE    sizeFont;
	LOGFONT lf;
    RECT    fillRect;
    int     logpixelsy, nHeight;

    hrgnScreen = CreateRectRgnIndirect(&gCreditsRect);
    SelectClipRgn(hDC, hrgnScreen);

    hdcScreen = CreateCompatibleDC(hDC);
    hbmpScreen = CreateCompatibleBitmap(hDC, (gCreditsRect.right - gCreditsRect.left),
        (gCreditsRect.bottom - gCreditsRect.top) );
    hbmpOldScreen = (HBITMAP)SelectObject(hdcScreen, hbmpScreen);

    hdcCredits = CreateCompatibleDC(hDC);

    CreditsBmpWidth = (gCreditsRect.right - gCreditsRect.left);
    CreditsBmpHeight = DrawCredits(hdcCredits, false, extended);

    hbmpCredits = CreateCompatibleBitmap(hDC, CreditsBmpWidth, CreditsBmpHeight);
    hbmpOldCredits = (HBITMAP)SelectObject(hdcCredits, hbmpCredits);

    hBr = CreateSolidBrush(RGB(254,254,254)); //MASK_RGB);
    fillRect.top = fillRect.left = 0;
    fillRect.bottom = CreditsBmpHeight;
    fillRect.right = CreditsBmpWidth;
    FillRect(hdcCredits, &fillRect, hBr);

	memset((void*)&lf, 0, sizeof(lf));
	lf.lfHeight = 14;
	lf.lfWeight = 400;
	lf.lfQuality = NONANTIALIASED_QUALITY;
	strcpy(lf.lfFaceName, "Arial");
	hFont = CreateFontIndirect(&lf);

    if(hFont)
        hOldFont = (HFONT)SelectObject(hdcCredits, hFont);

    SetBkMode(hdcCredits, TRANSPARENT);
    SetTextColor(hdcCredits, 0x00000000);

    DrawCredits(hdcCredits, true, extended);

    SetBkColor(hdcCredits, MASK_RGB);
    SelectObject(hdcCredits, hOldFont);

    // create the mask bitmap
    hdcMask = CreateCompatibleDC(hdcScreen);
    hbmpMask = CreateBitmap(CreditsBmpWidth, CreditsBmpHeight, 1, 1, NULL);

    // select the mask bitmap into the appropriate dc
    hbmpOldMask = (HBITMAP)SelectObject(hdcMask, hbmpMask);

    // build mask based on transparent color
    BitBlt(hdcMask, 0, 0, CreditsBmpWidth, CreditsBmpHeight, hdcCredits, 0, 0, SRCCOPY);
}

void PaintBk(HDC hDC, HDC mDC)
{
	//save background the first time
	if (hdcBk == NULL)
	{
		hdcBk = CreateCompatibleDC(mDC);
		hbmpBk = CreateCompatibleBitmap(mDC, gCreditsRect.right-gCreditsRect.left, 
            gCreditsRect.bottom-gCreditsRect.top);
		hbmpOldBk = (HBITMAP)SelectObject(hdcBk, hbmpBk);
		BitBlt(hdcBk, 0, 0, gCreditsRect.right-gCreditsRect.left, 
            gCreditsRect.bottom-gCreditsRect.top, mDC, gCreditsRect.left, 
            gCreditsRect.top, SRCCOPY);
	}

	BitBlt(hDC, gCreditsRect.left, gCreditsRect.top, 
        gCreditsRect.right-gCreditsRect.left, 
        gCreditsRect.bottom-gCreditsRect.top, 
        hdcBk, 0, 0, SRCCOPY);
}

void ScrollCredits(BOOL extended)
{
	// track scroll position
	static int nScrollY = 0;
	int nTimeInMilliseconds;

    if (!bShow)
        return;
    if (!IsWindow(hSplashWnd))
        return;
    HDC hDC = GetDC(hSplashWnd);

    if(hdcCredits == NULL) {
		CreateCredits(hDC, extended);
        nScrollY = 0;
    }

    if(nScrollY == 1)
        Sleep(1000);

    PaintBk(hdcScreen, hDC);
	BitBlt(hdcScreen, 0, 0, CreditsBmpWidth, CreditsBmpHeight, hdcCredits, 0, nScrollY, SRCINVERT);
	BitBlt(hdcScreen, 0, 0, CreditsBmpWidth, CreditsBmpHeight, hdcMask, 0, nScrollY, SRCAND);
	BitBlt(hdcScreen, 0, 0, CreditsBmpWidth, CreditsBmpHeight, hdcCredits, 0, nScrollY, SRCINVERT);

	BitBlt(hDC, gCreditsRect.left, gCreditsRect.top, (gCreditsRect.right - gCreditsRect.left), 
           (gCreditsRect.bottom - gCreditsRect.top), hdcScreen, 0, 0, SRCCOPY);

	GdiFlush();

    // continue scrolling
	nScrollY += 1;
	if(nScrollY >= CreditsBmpHeight) nScrollY = 0;	// scrolling up
	if(nScrollY < 0) nScrollY = CreditsBmpHeight;	// scrolling down
    
    // delay scrolling by the specified time
	Sleep(25);
}

///////////////////////////////////////////////////////////////////////////////
// Foward declarations of functions included in this code module:
ATOM                MyRegisterClass(HINSTANCE hInstance);
LRESULT CALLBACK    SplashWndProc(HWND, UINT, WPARAM, LPARAM);


void *OpenGraphic(char *name)
{
    IPicture *Ipic = NULL;
    SIZE sizeInHiMetric,sizeInPix;
    const int HIMETRIC_PER_INCH = 2540;
    HDC hDCScreen = GetDC(NULL);
    HRESULT hr;
    int nPixelsPerInchX = GetDeviceCaps(hDCScreen, LOGPIXELSX);
    int nPixelsPerInchY = GetDeviceCaps(hDCScreen, LOGPIXELSY);
    unsigned short OlePathName[512];

    ReleaseDC(NULL,hDCScreen);
    mbstowcs(OlePathName,name,strlen(name)+1);
    hr = OleLoadPicturePath(OlePathName,
              NULL,
              0,
                 0,
              IID_IPicture,
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
        ImageInfo.sizeInPix = sizeInPix;
        ImageInfo.sizeInHiMetric = sizeInHiMetric;
        ImageInfo.Ipic = Ipic;
        ImageInfo.Path = name;
        return Ipic;

    }
err:
    return 0;
}

void DisplayGraphic(HWND hwnd,HDC pDC)
{
    IPicture *Ipic = ImageInfo.Ipic;
    DWORD dwAttr = 0;
    HBITMAP Bmp,BmpOld;
    RECT rc;
    HRESULT hr;
    HPALETTE pPalMemOld;

    if (Ipic != NULL) {
        // get palette
        OLE_HANDLE hPal = 0;
        HPALETTE hPalOld=NULL,hPalMemOld=NULL;
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
            Bmp = CreateCompatibleBitmap(pDC,ImageInfo.sizeInPix.cx,ImageInfo.sizeInPix.cy);

            BmpOld = (HBITMAP)SelectObject(MemDC,Bmp);
            pPalMemOld = NULL;
            if (hPal != 0) {
                hPalMemOld = SelectPalette(MemDC,(HPALETTE)hPal, FALSE);
                RealizePalette(MemDC);
            }

            // display picture using IPicture::Render
            hr = Ipic->Render(MemDC, 0, 0, rc.right, rc.bottom, 0,
                              ImageInfo.sizeInHiMetric.cy,
                              ImageInfo.sizeInHiMetric.cx,
                              -ImageInfo.sizeInHiMetric.cy, &rc);

            BitBlt(pDC,0, 0, ImageInfo.sizeInPix.cx, ImageInfo.sizeInPix.cy,
                   MemDC, 0, 0, SRCCOPY);

            SelectObject(MemDC,BmpOld);

            if (pPalMemOld) SelectPalette(MemDC,pPalMemOld, FALSE);
            DeleteObject(Bmp);
            DeleteDC(MemDC);

        }
        else {
            // display picture using IPicture::Render
            Ipic->Render(pDC, 0, 0, rc.right, rc.bottom, 0,
                         ImageInfo.sizeInHiMetric.cy,
                         ImageInfo.sizeInHiMetric.cx,
                         -ImageInfo.sizeInHiMetric.cy, &rc);
        }

        if (hPalOld != NULL) 
            SelectPalette(pDC,hPalOld, FALSE);
        if (hPal) 
            DeleteObject((HPALETTE)hPal);
    }
}

void CloseImage(void *Ipict)
{
    IPicture *ip = (IPicture *)Ipict;

    if (ip == NULL)
        ip = ImageInfo.Ipic;
    if (ip == NULL)
        return;
    ip->Release();
    memset(&ImageInfo,0,sizeof(ImageInfo));
}

////////////////////////////////////////////////////////////////////////////////
// Splashscreen functions
////////////////////////////////////////////////////////////////////////////////
//
//
void HideSplashScreen()
{
    // Destroy the window
    if (IsWindow(hSplashWnd)) {
        DestroyWindow(hSplashWnd);
        hSplashWnd = NULL;
        UnregisterClass("RootSplashScreen", hInst);
    }
	if(hdcBk != NULL && hbmpOldBk != NULL) {
		SelectObject(hdcBk, hbmpOldBk);
		DeleteObject(hbmpBk);
	}

	if(hdcScreen != NULL && hbmpOldScreen != NULL) {
		SelectObject(hdcScreen, hbmpOldScreen);
		DeleteObject(hbmpScreen);
	}

	if(hdcCredits != NULL && hbmpOldCredits != NULL) {
		SelectObject(hdcCredits, hbmpOldCredits);
		DeleteObject(hbmpCredits);
	}

	if(hdcMask != NULL && hbmpOldMask != NULL) {
		SelectObject(hdcMask, hbmpOldMask);
		DeleteObject(hbmpMask);
	}
	DeleteDC(hdcMask);
	hdcMask = NULL;
	DeleteDC(hdcCredits);
	hdcCredits = NULL;
	DeleteDC(hdcScreen);
	hdcScreen = NULL;
	DeleteDC(hdcBk);
	hdcBk = NULL;
    CloseImage(ImageInfo.Ipic);
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

    hSplashWnd = CreateWindowEx(
            WS_EX_TOOLWINDOW,
            "RootSplashScreen",
            NULL,
            WS_POPUP | WS_VISIBLE,
            (xScreen - 360)/2,
            (yScreen - 240)/2,
            360, 240,
            hParent,
            NULL,
            hInst,
            NULL);
    
    return (hSplashWnd != NULL);
}


////////////////////////////////////////////////////////////////////////////////
//
//
BOOL PreTranslateMessage(MSG* pMsg)
{
    if (!IsWindow(hSplashWnd))
        return FALSE;

    // If we get a keyboard or mouse message, hide the splash screen.
    if (pMsg->message == WM_KEYDOWN ||
        pMsg->message == WM_SYSKEYDOWN ||
        pMsg->message == WM_LBUTTONDOWN ||
        pMsg->message == WM_RBUTTONDOWN ||
        pMsg->message == WM_MBUTTONDOWN ||
        pMsg->message == WM_NCLBUTTONDOWN ||
        pMsg->message == WM_NCRBUTTONDOWN ||
        pMsg->message == WM_NCMBUTTONDOWN)
    {
        HideSplashScreen();
        return TRUE;    // message handled here
    }

    return FALSE;   // message not handled
}

void CreateSplash(DWORD time, BOOL extended)
{
    MSG msg;
    MyRegisterClass(hInst);
    bShow = FALSE;
    if(extended)
        gAbout = true;
    if(time > 0) {
        DelayVal = time * 1000;
    }
    else return;

    ReadContributors();
    
    // Create the splash screen
    CreateSplashScreen(NULL);

    // Main message loop:
    while (gStayUp) {
        if(PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
            PreTranslateMessage(&msg);
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        ScrollCredits(extended);
    }

    HideSplashScreen();
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
    wcex.hIcon          = NULL;
    wcex.hCursor        = LoadCursor(NULL, IDC_ARROW);
    wcex.hbrBackground  = NULL;
    wcex.lpszMenuName   = NULL;
    wcex.lpszClassName  = "RootSplashScreen";
    wcex.hIconSm        = NULL;
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
    HWND hwndFound;         // this is what is returned to the caller
    char pszNewWindowTitle[MY_BUFSIZE]; // contains fabricated WindowTitle
    char pszOldWindowTitle[MY_BUFSIZE]; // contains original WindowTitle
    const char bmpDir[] = "\\icons\\Splash.gif";
    char FullBmpDir[256];
    char *RootSysDir;
    int xScreen;
    int yScreen;
    int logpixelsy, nHeight;

    switch (message) {
        case WM_CREATE:
            if(!gAbout)
                SetTimer(hWnd, ID_SPLASHSCREEN, DelayVal, NULL);
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
            SetWindowPos(hWnd, HWND_TOPMOST, (xScreen - ImageInfo.sizeInPix.cx)/2, 
                (yScreen - ImageInfo.sizeInPix.cy)/2, ImageInfo.sizeInPix.cx, 
                ImageInfo.sizeInPix.cy, 0 );
            break;

        case WM_TIMER:
            if (wParam == ID_SPLASHSCREEN) {
                KillTimer (hWnd, ID_SPLASHSCREEN);
                HideSplashScreen();
            }
            break;

        case WM_DESTROY:
            gStayUp = false;
            PostQuitMessage(0);

        case WM_PAINT:
            hDC = BeginPaint(hWnd, &ps);
           // TODO: Add any drawing code here...
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
            if(!gAbout) {
                // fetch current window title
                GetConsoleTitle(pszOldWindowTitle, MY_BUFSIZE);
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
            }
            bShow = TRUE;
            break;

        default:
            return DefWindowProc(hWnd, message, wParam, lParam);
   }
   return 0;
}
#endif
