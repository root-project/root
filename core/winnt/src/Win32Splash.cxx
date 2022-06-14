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
#include "RVersion.h"
#include "strlcpy.h"
#include <wincodec.h>
#include <tchar.h>
#include <iostream>
#include <string>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#pragma comment(lib, "windowscodecs.lib")
#pragma comment(lib, "msimg32.lib")

#define ID_SPLASHSCREEN      25

static const char *gConception[] = {
   "Rene Brun",
   "Fons Rademakers",
   0
};

const char * gROOTCoreTeam[] = {
   "Rene Brun",
   "Fons Rademakers",
   "Philippe Canal",
   "Axel Naumann",
   "Olivier Couet",
   "Lorenzo Moneta",
   "Vassil Vassilev",
   "Gerardo Ganis",
   "Bertrand Bellenot",
   "Danilo Piparo",
   "Wouter Verkerke",
   "Timur Pocheptsov",
   "Matevz Tadel",
   "Pere Mato",
   "Wim Lavrijsen",
   "Ilka Antcheva",
   "Paul Russo",
   "Andrei Gheata",
   "Anirudha Bose",
   "Valeri Onuchine",
   0
};

///////////////////////////////////////////////////////////////////////////////
// Global Variables:
static HINSTANCE    gInst          = 0; // Current instance
static HWND         gSplashWnd     = 0; // Splash screen
static bool         gShow          = FALSE;
static DWORD        gDelayVal      = 0;
static bool         gAbout         = false;
static RECT         gCreditsRect   = { 115, 0, 580, 80 }; // clip rect in logo
static unsigned int gCreditsWidth  = gCreditsRect.right - gCreditsRect.left; // credits pixmap size

///////////////////////////////////////////////////////////////////////////
/// Create a bitmap and draw alpha blended text on it.

HBITMAP CreateAlphaTextBitmap(LPCSTR inText, HFONT inFont, COLORREF inColour)
{
   int TextLength = (int)strlen(inText);
   if (TextLength <= 0) return NULL;

   // Create DC and select font into it
   HDC hTextDC = CreateCompatibleDC(NULL);
   HFONT hOldFont = (HFONT)SelectObject(hTextDC, inFont);
   HBITMAP hMyDIB = NULL;

   // Get text area
   RECT TextArea = {0, 0, 0, 0};
   DrawText(hTextDC, inText, TextLength, &TextArea, DT_CALCRECT);
   if ((TextArea.right > TextArea.left) && (TextArea.bottom > TextArea.top)) {
      BITMAPINFOHEADER BMIH;
      memset(&BMIH, 0x0, sizeof(BITMAPINFOHEADER));
      void *pvBits = NULL;

      // Specify DIB setup
      BMIH.biSize = sizeof(BMIH);
      BMIH.biWidth = TextArea.right - TextArea.left;
      BMIH.biHeight = TextArea.bottom - TextArea.top;
      BMIH.biPlanes = 1;
      BMIH.biBitCount = 32;
      BMIH.biCompression = BI_RGB;

      // Create and select DIB into DC
      hMyDIB = CreateDIBSection(hTextDC, (LPBITMAPINFO)&BMIH, 0,
                                (LPVOID*)&pvBits, NULL, 0);
      HBITMAP hOldBMP = (HBITMAP)SelectObject(hTextDC, hMyDIB);
      if (hOldBMP != NULL) {
         // Set up DC properties
         SetTextColor(hTextDC, 0x00FFFFFF);
         SetBkColor(hTextDC, 0x00000000);
         SetBkMode(hTextDC, OPAQUE);

         // Draw text to buffer
         DrawText(hTextDC, inText, TextLength, &TextArea, DT_NOCLIP);
         BYTE* DataPtr = (BYTE*)pvBits;
         BYTE FillR = GetRValue(inColour);
         BYTE FillG = GetGValue(inColour);
         BYTE FillB = GetBValue(inColour);
         BYTE ThisA;
         for (int LoopY = 0; LoopY < BMIH.biHeight; LoopY++) {
            for (int LoopX = 0; LoopX < BMIH.biWidth; LoopX++) {
               ThisA = *DataPtr; // Move alpha and pre-multiply with RGB
               *DataPtr++ = (FillB * ThisA) >> 8;
               *DataPtr++ = (FillG * ThisA) >> 8;
               *DataPtr++ = (FillR * ThisA) >> 8;
               *DataPtr++ = ThisA; // Set Alpha
            }
         }
         // De-select bitmap
         SelectObject(hTextDC, hOldBMP);
      }
   }
   // De-select font and destroy temp DC
   SelectObject(hTextDC, hOldFont);
   DeleteDC(hTextDC);

   // Return DIBSection
   return hMyDIB;
}
///////////////////////////////////////////////////////////////////////////
/// Draw alpha blended text on the splash screen.

void DrawAlphaText(HDC inDC, HFONT inFont, COLORREF inColor,
                   const char *text, int inX, int inY)
{
   RECT TextArea = {0, 0, 0, 0};
   HBITMAP MyBMP = CreateAlphaTextBitmap(text, inFont, inColor);
   if (MyBMP) {
      // Create temporary DC and select new Bitmap into it
      HDC hTempDC = CreateCompatibleDC(inDC);
      HBITMAP hOldBMP = (HBITMAP)SelectObject(hTempDC, MyBMP);
      if (hOldBMP) {
         // Get Bitmap image size
         BITMAP BMInf;
         GetObject(MyBMP, sizeof(BITMAP), &BMInf);

         // Fill blend function and blend new text to window
         BLENDFUNCTION bf;
         bf.BlendOp = AC_SRC_OVER;
         bf.BlendFlags = 0;
         bf.SourceConstantAlpha = 0xFF;
         bf.AlphaFormat = AC_SRC_ALPHA;
         AlphaBlend(inDC, inX, inY, BMInf.bmWidth, BMInf.bmHeight, hTempDC,
                    0, 0, BMInf.bmWidth, BMInf.bmHeight, bf);

         // Clean up
         SelectObject(hTempDC, hOldBMP);
         DeleteObject(MyBMP);
         DeleteDC(hTempDC);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Draw the ROOT version on the bottom right of the splash screen.

static void DrawVersion(HDC hDC, HFONT inFont, COLORREF inColor)
{
   SIZE lpSize;
   char version[256];
   sprintf(version, "Version %s", ROOT_RELEASE);
   GetTextExtentPoint32(hDC, version, strlen(version), &lpSize);
   DrawAlphaText(hDC, inFont, inColor, version, 580-lpSize.cx, 400);
}

////////////////////////////////////////////////////////////////////////////////
/// Draw credit item.

static int DrawCreditItem(HDC hDC, HFONT inFont, COLORREF inColor,
                          const char *creditItem, const char **members, int y)
{
   char credit[1024];
   SIZE lpSize1, lpSize2;
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
         DrawAlphaText(hDC, inFont, inColor, credit, gCreditsRect.left, y);
         y += lineSpacing;
         strcpy(credit, "   ");
      }
      strcat(credit, members[i]);
   }
   DrawAlphaText(hDC, inFont, inColor, credit, gCreditsRect.left, y);
   return y;
}

////////////////////////////////////////////////////////////////////////////////
/// Draw the credits on the splah window.

void DrawCredits(HDC hDC, HFONT inFont, COLORREF inColor)
{
   TEXTMETRIC lptm;
   int lineSpacing, y;
   GetTextMetrics(hDC, &lptm);
   lineSpacing = lptm.tmAscent + lptm.tmDescent;
   y = 305;
   y = DrawCreditItem(hDC, inFont, inColor, "Conception: ", gConception, y);
   y += 2 * lineSpacing - 4;
   y = DrawCreditItem(hDC, inFont, inColor, "Core Engineering: ", gROOTCoreTeam, y);
}

////////////////////////////////////////////////////////////////////////////////
/// Get a stream from the specified file name (using Windows Imaging Component).

IStream *FromFile(LPCWSTR Filename)
{
   IWICStream *Stream = 0;
   IWICImagingFactory *Factory = 0;

#if(_WIN32_WINNT >= 0x0602) || defined(_WIN7_PLATFORM_UPDATE)
   // WIC2 is available on Windows 8 and Windows 7 SP1 with KB 2670838 installed
   HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory2, 0, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&Factory));
   if (FAILED(hr)) {
      hr = CoCreateInstance(CLSID_WICImagingFactory1, 0, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&Factory));
      if (FAILED(hr)) {
         return NULL;
      }
   }
#else
   HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, 0, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&Factory));
   if (FAILED(hr)) {
      return NULL;
   }
#endif
   if (SUCCEEDED(Factory->CreateStream(&Stream))) {
      Stream->InitializeFromFilename(Filename, GENERIC_READ);
   }
   Factory->Release();
   return Stream;
}

////////////////////////////////////////////////////////////////////////////////
/// Loads a PNG image from the specified stream (using Windows Imaging
/// Component).

IWICBitmapSource *LoadBitmapFromStream(IStream *ipImageStream)
{
   // initialize return value
   IWICBitmapSource *ipBitmap = NULL;

   // load WIC's PNG decoder
   IWICBitmapDecoder *ipDecoder = NULL;

#if(_WIN32_WINNT >= 0x0602) || defined(_WIN7_PLATFORM_UPDATE)
   // WIC2 is available on Windows 8 and Windows 7 SP1 with KB 2670838 installed
   HRESULT hr = CoCreateInstance(CLSID_WICPngDecoder2, NULL, CLSCTX_INPROC_SERVER, __uuidof(ipDecoder), reinterpret_cast<void **>(&ipDecoder));
   if (FAILED(hr)) {
      hr = CoCreateInstance(CLSID_WICPngDecoder1, NULL, CLSCTX_INPROC_SERVER, __uuidof(ipDecoder), reinterpret_cast<void **>(&ipDecoder));
      if (FAILED(hr)) {
         return NULL;
      }
   }
#else
   HRESULT hr = CoCreateInstance(CLSID_WICPngDecoder, NULL, CLSCTX_INPROC_SERVER, __uuidof(ipDecoder), reinterpret_cast<void **>(&ipDecoder));
   if (FAILED(hr)) {
      return NULL;
   }
#endif
   // load the PNG
   if (FAILED(ipDecoder->Initialize(ipImageStream, WICDecodeMetadataCacheOnLoad))) {
      ipDecoder->Release();
      return NULL;
   }
   // check for the presence of the first frame in the bitmap
   UINT nFrameCount = 0;

   if (FAILED(ipDecoder->GetFrameCount(&nFrameCount)) || nFrameCount != 1) {
      ipDecoder->Release();
      return NULL;
   }
   // load the first frame (i.e., the image)
   IWICBitmapFrameDecode *ipFrame = NULL;

   if (FAILED(ipDecoder->GetFrame(0, &ipFrame))) {
      ipDecoder->Release();
      return NULL;
   }
   // convert the image to 32bpp BGRA format with pre-multiplied alpha
   //   (it may not be stored in that format natively in the PNG resource,
   //   but we need this format to create the DIB to use on-screen)
   WICConvertBitmapSource(GUID_WICPixelFormat32bppPBGRA, ipFrame, &ipBitmap);
   ipFrame->Release();

   ipDecoder->Release();
   return ipBitmap;
}

////////////////////////////////////////////////////////////////////////////////
/// Create a 32-bit DIB from the specified WIC bitmap.

HBITMAP CreateHBITMAP(IWICBitmapSource *ipBitmap)
{
   // initialize return value
   HBITMAP hbmp = NULL;

   // get image attributes and check for valid image
   UINT width = 0;
   UINT height = 0;

   if (FAILED(ipBitmap->GetSize(&width, &height)) || width == 0 || height == 0) {
      return hbmp;
   }

   // prepare structure giving bitmap information (negative height indicates a top-down DIB)
   BITMAPINFO bminfo;
   ZeroMemory(&bminfo, sizeof(bminfo));
   bminfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
   bminfo.bmiHeader.biWidth = width;
   bminfo.bmiHeader.biHeight = -((LONG) height);
   bminfo.bmiHeader.biPlanes = 1;
   bminfo.bmiHeader.biBitCount = 32;
   bminfo.bmiHeader.biCompression = BI_RGB;

   // create a DIB section that can hold the image
   void *pvImageBits = NULL;
   HDC hdcScreen = GetDC(NULL);
   hbmp = CreateDIBSection(hdcScreen, &bminfo, DIB_RGB_COLORS, &pvImageBits, NULL, 0);
   ReleaseDC(NULL, hdcScreen);

   if (hbmp == NULL) {
      return NULL;
   }
   // extract the image into the HBITMAP
   const UINT cbStride = width * 4;
   const UINT cbImage = cbStride * height;

   if (FAILED(ipBitmap->CopyPixels(NULL, cbStride, cbImage, static_cast<BYTE *>(pvImageBits)))) {
      // couldn't extract image; delete HBITMAP
      DeleteObject(hbmp);
      hbmp = NULL;
   }
   return hbmp;
}

////////////////////////////////////////////////////////////////////////////////
/// Loads the PNG containing the splash image into a HBITMAP.

HBITMAP LoadSplashImage(LPCWSTR file_name)
{
   HBITMAP hbmpSplash = NULL;

   // load the PNG image data into a stream
   IStream *ipImageStream = FromFile(file_name);

   if (ipImageStream == NULL) {
      return hbmpSplash;
   }
   // load the bitmap with WIC
   IWICBitmapSource *ipBitmap = LoadBitmapFromStream(ipImageStream);

   if (ipBitmap == NULL) {
      ipImageStream->Release();
      return NULL;
   }
   // create a HBITMAP containing the image
   hbmpSplash = CreateHBITMAP(ipBitmap);
   ipBitmap->Release();

   ipImageStream->Release();
   return hbmpSplash;
}

////////////////////////////////////////////////////////////////////////////////
/// Destroy our splash screen window.

void DestroySplashScreen()
{
   if (IsWindow(gSplashWnd)) {
      DestroyWindow(gSplashWnd);
      gSplashWnd = 0;
      UnregisterClass("SplashWindow", gInst);
   }
}

///////////////////////////////////////////////////////////////////////////
/// Message handler for the splash screen window.

LRESULT CALLBACK SplashWndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
   switch (message) {
      case WM_CREATE:
         if(!gAbout)
            SetTimer(hWnd, ID_SPLASHSCREEN, gDelayVal, 0);
         break;

      case WM_TIMER:
         if (wParam == ID_SPLASHSCREEN) {
            KillTimer (hWnd, ID_SPLASHSCREEN);
            DestroySplashScreen();
         }
         break;

      case WM_DESTROY:
         PostQuitMessage(0);

      default:
         return DefWindowProc(hWnd, message, wParam, lParam);
   }
   return 0;
}

///////////////////////////////////////////////////////////////////////////
/// Registers a window class for the splash and splash owner windows.

void RegisterWindowClass(HINSTANCE g_hInstance)
{
   WNDCLASS wc = { 0 };
   wc.lpfnWndProc = (WNDPROC)SplashWndProc;//DefWindowProc;
   wc.hInstance = g_hInstance;
   //wc.hIcon = LoadIcon(g_hInstance, MAKEINTRESOURCE(_T("SPLASH")));
   wc.hCursor = LoadCursor(NULL, IDC_ARROW);
   wc.lpszClassName = _T("SplashWindow");
   RegisterClass(&wc);
}

///////////////////////////////////////////////////////////////////////////
/// Create the splash owner window and the splash window.

HWND CreateSplashWindow(HINSTANCE g_hInstance)
{
   return CreateWindowEx(WS_EX_LAYERED | WS_EX_TOOLWINDOW | WS_EX_TOPMOST,
                         _T("SplashWindow"), NULL, WS_POPUP | WS_VISIBLE,
                         0, 0, 0, 0, NULL, NULL, g_hInstance, NULL);
}

///////////////////////////////////////////////////////////////////////////
/// Call UpdateLayeredWindow to set a bitmap (with alpha) as the content of
/// the splash window.

void SetSplashImage(HWND hwndSplash, HBITMAP hbmpSplash)
{
   // get the size of the bitmap
   BITMAP bm;
   GetObject(hbmpSplash, sizeof(bm), &bm);
   SIZE sizeSplash = { bm.bmWidth, bm.bmHeight };

   // get the primary monitor's info
   POINT ptZero = { 0 };
   HMONITOR hmonPrimary = MonitorFromPoint(ptZero, MONITOR_DEFAULTTOPRIMARY);
   MONITORINFO monitorinfo = { 0 };
   monitorinfo.cbSize = sizeof(monitorinfo);
   GetMonitorInfo(hmonPrimary, &monitorinfo);

   // center the splash screen in the middle of the primary work area
   const RECT &rcWork = monitorinfo.rcWork;
   POINT ptOrigin;
   ptOrigin.x = rcWork.left + (rcWork.right - rcWork.left - sizeSplash.cx - 93) / 2;
   ptOrigin.y = rcWork.top + (rcWork.bottom - rcWork.top - sizeSplash.cy - 104) / 2;

   // create a memory DC holding the splash bitmap
   HDC hdcScreen = GetDC(NULL);
   HDC hdcMem = CreateCompatibleDC(hdcScreen);
   HBITMAP hbmpOld = (HBITMAP) SelectObject(hdcMem, hbmpSplash);

   // use the source image's alpha channel for blending
   BLENDFUNCTION blend = { 0 };
   blend.BlendOp = AC_SRC_OVER;
   blend.SourceConstantAlpha = 255;
   blend.AlphaFormat = AC_SRC_ALPHA;

   SetBkMode(hdcMem, TRANSPARENT);
   HFONT hFont = CreateFont(14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "Arial\0");
   HFONT hOldFont = (HFONT)SelectObject(hdcMem, hFont);
   DrawVersion(hdcMem, hFont, RGB(255,255,255));
   DrawCredits(hdcMem, hFont, RGB(176,210,249));
   SelectObject(hdcMem, hOldFont);
   DeleteObject(hFont);

   // paint the window (in the right location) with the alpha-blended bitmap
   UpdateLayeredWindow(hwndSplash, hdcScreen, &ptOrigin, &sizeSplash,
                       hdcMem, &ptZero, RGB(0, 0, 0), &blend, ULW_ALPHA);

   // delete temporary objects
   SelectObject(hdcMem, hbmpOld);
   DeleteDC(hdcMem);
   ReleaseDC(NULL, hdcScreen);
}

////////////////////////////////////////////////////////////////////////////////
/// check for keybord or mouse event and destroy the splash screen accordingly.

bool PreTranslateMessage(MSG* pMsg)
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

////////////////////////////////////////////////////////////////////////////////
/// Create our splash screen.

void CreateSplash(DWORD time, bool extended)
{
   MSG msg;
   gShow = FALSE;
   if (extended) gAbout = true;
   if (time > 0) gDelayVal = time * 1000;
   else return;

   RegisterWindowClass(gInst);

   if (!_wgetenv(L"ROOTSYS")) return;
   std::wstring RootSysDir = _wgetenv(L"ROOTSYS");
   std::wstring splash_picture = RootSysDir + L"\\icons\\Root6Splash.png";
   CoInitialize(0);
   HBITMAP bkg_img = LoadSplashImage(splash_picture.c_str());
   gSplashWnd = CreateSplashWindow(gInst);
   SetSplashImage(gSplashWnd, bkg_img);
   DeleteObject(bkg_img);
   CoUninitialize();
   // Main message loop:
   while (GetMessage(&msg, 0, 0, 0)) {
      PreTranslateMessage(&msg);
      TranslateMessage(&msg);
      DispatchMessage(&msg);
   }
   DestroySplashScreen();
}

#endif
