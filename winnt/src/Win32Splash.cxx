// @(#)root/winnt:$Name:$:$Id:$
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
#include <stdlib.h>
#include <stdio.h>

#define ID_SPLASHSCREEN      25

///////////////////////////////////////////////////////////////////////////////
// Global Variables:
HINSTANCE hInst;                        // current instance
HWND hMainWnd = NULL;
HWND hSplashWnd = NULL;               // Splash screen
HBITMAP hBmp = NULL;
BITMAP bm;
BOOL bSplash = TRUE;
DWORD DelayVal = 0;
BOOL bClassRegistered = FALSE;

///////////////////////////////////////////////////////////////////////////////
// Foward declarations of functions included in this code module:
ATOM                MyRegisterClass(HINSTANCE hInstance);
LRESULT CALLBACK    SplashWndProc(HWND, UINT, WPARAM, LPARAM);


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
        DeleteObject(hBmp);
        hSplashWnd = NULL;
        UnregisterClass("RootSplashScreen", hInst);
    }
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

void CreateSplash(DWORD time)
{
    MSG msg;
    if (MyRegisterClass(hInst) != 0)
        bClassRegistered = TRUE;
    if(time > 0) {
        bSplash = TRUE;
        DelayVal = time * 1000;
    }
    else return;
    // Create the splash screen
    if (bSplash)
        CreateSplashScreen(NULL);

    // Main message loop:
    while (GetMessage(&msg, NULL, 0, 0)) {
        PreTranslateMessage(&msg);
        TranslateMessage(&msg);
        DispatchMessage(&msg);
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
    HANDLE hfbm;
    BITMAPFILEHEADER bmfh;
    BITMAPINFOHEADER bmih;
    HGLOBAL hmem1,hmem2;
    LPBITMAPINFO lpbmi;
    LPVOID lpvBits;
    DWORD dwRead;
    const char bmpDir[] = "\\icons\\Splash.bmp";
    char FullBmpDir[128];
    char *RootSysDir;
    int xScreen;
    int yScreen;

    switch (message)
    {
        case WM_CREATE:
            SetTimer(hWnd, ID_SPLASHSCREEN, DelayVal, NULL);
            RootSysDir = getenv("ROOTSYS");
            sprintf(FullBmpDir,"%s%s",RootSysDir,bmpDir);
            // Retrieve a handle identifying the file. 
            hfbm = CreateFile(FullBmpDir, GENERIC_READ, 
                    FILE_SHARE_READ, (LPSECURITY_ATTRIBUTES) NULL, 
                    OPEN_EXISTING, FILE_ATTRIBUTE_READONLY, 
                    (HANDLE) NULL); 
            if(hfbm) {
                // Retrieve the BITMAPFILEHEADER structure. 
                ReadFile(hfbm, &bmfh, sizeof(BITMAPFILEHEADER), &dwRead, 
                        (LPOVERLAPPED)NULL); 
                // Retrieve the BITMAPFILEHEADER structure. 
                ReadFile(hfbm, &bmih, sizeof(BITMAPINFOHEADER), 
                        &dwRead, (LPOVERLAPPED)NULL); 
                // Allocate memory for the BITMAPINFO structure. 
                hmem1 = GlobalAlloc(GHND, sizeof(BITMAPINFOHEADER) + 
                        ((1<<bmih.biBitCount) * sizeof(RGBQUAD))); 
                lpbmi = (LPBITMAPINFO)GlobalLock(hmem1); 
                // Load BITMAPINFOHEADER into the BITMAPINFO structure. 
                lpbmi->bmiHeader.biSize = bmih.biSize; 
                lpbmi->bmiHeader.biWidth = bmih.biWidth; 
                lpbmi->bmiHeader.biHeight = bmih.biHeight; 
                lpbmi->bmiHeader.biPlanes = bmih.biPlanes; 
                lpbmi->bmiHeader.biBitCount = bmih.biBitCount; 
                lpbmi->bmiHeader.biCompression = bmih.biCompression; 
                lpbmi->bmiHeader.biSizeImage = bmih.biSizeImage; 
                lpbmi->bmiHeader.biXPelsPerMeter = bmih.biXPelsPerMeter; 
                lpbmi->bmiHeader.biYPelsPerMeter = bmih.biYPelsPerMeter; 
                lpbmi->bmiHeader.biClrUsed = bmih.biClrUsed; 
                lpbmi->bmiHeader.biClrImportant = bmih.biClrImportant; 
                // Retrieve the color table. 
                // 1 << bmih.biBitCount == 2 ^ bmih.biBitCount 
                ReadFile(hfbm, lpbmi->bmiColors, 
                        ((1<<bmih.biBitCount) * sizeof(RGBQUAD)), 
                        &dwRead, (LPOVERLAPPED) NULL); 
                // Allocate memory for the required number of bytes. 
                hmem2 = GlobalAlloc(GHND, (bmfh.bfSize - bmfh.bfOffBits)); 
                lpvBits = GlobalLock(hmem2); 
                // Retrieve the bitmap data. 
                ReadFile(hfbm, lpvBits, (bmfh.bfSize - bmfh.bfOffBits), 
                        &dwRead, (LPOVERLAPPED) NULL); 
                hDC = GetDC(hWnd);
                // Create a bitmap from the data stored in the .BMP file. 
                hBmp = CreateDIBitmap(hDC, &bmih, CBM_INIT, lpvBits, lpbmi, DIB_RGB_COLORS); 
                // Unlock the global memory objects and close the .BMP file. 
                GlobalUnlock(hmem1); 
                GlobalUnlock(hmem2); 
                CloseHandle(hfbm); 
                GetObjectA(hBmp, sizeof(bm), &bm);
                // Center the splashscreen
                xScreen = GetSystemMetrics(SM_CXFULLSCREEN);
                yScreen = GetSystemMetrics(SM_CYFULLSCREEN);
                SetWindowPos(hWnd, HWND_TOPMOST, (xScreen - bm.bmWidth)/2, 
                    (yScreen - bm.bmHeight)/2, bm.bmWidth, bm.bmHeight, 0 );
            } 
            break;

        case WM_TIMER:
            if (wParam == ID_SPLASHSCREEN) {
                KillTimer (hWnd, ID_SPLASHSCREEN);
                HideSplashScreen();
            }
            break;

        case WM_DESTROY:
            PostQuitMessage(0);

        case WM_PAINT:
            hDC = BeginPaint(hWnd, &ps);
            // TODO: Add any drawing code here...
            RECT rt;
            GetClientRect(hWnd, &rt);

            HDC hImageDC;
            hImageDC = CreateCompatibleDC(hDC);
            if (hImageDC == NULL)
                return FALSE;

            // Paint the image.
            HBITMAP hOldBitmap;
            hOldBitmap = (HBITMAP)SelectObject(hImageDC, hBmp);
            BitBlt(hDC, 0, 0, bm.bmWidth, bm.bmHeight, hImageDC, 0, 0, SRCCOPY);
            SelectObject(hImageDC, hOldBitmap);

            EndPaint(hWnd, &ps);
            break;

        default:
            return DefWindowProc(hWnd, message, wParam, lParam);
   }
   return 0;
}
#endif
