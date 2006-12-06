/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/* =======================================================
	Simple Win32API Application for Drawing Graphics.

	WndProc.c: Defines procedures of the main window
	Last modified Time-stamp: <02/01/13 09:48:12 hata>            
	Proposed by K.Hata <kazuhiko.hata@nifty.ne.jp>
========================================================== */

#define		STRICT
#include	<windows.h>
#include	<stdio.h>
#include    "resource.h"
#include	"WndProc.h"

#include "G__ci.h"          /* Cint header file */

extern HINSTANCE hInst;

// forward declaration of the functions in the module.
static	LRESULT	Wm_CommandProc(HWND hWnd,WORD wNotifyCode,WORD wID,HWND hwndCtl);
static	LRESULT	Wm_CloseProc(HWND);
static	LRESULT Wm_DestroyProc(void);
static	LRESULT	Wm_PaintProc(HWND);

static  LRESULT CALLBACK About( HWND hDlg, UINT, WPARAM, LPARAM);
int DrawGr(HDC);

//
//  FUNCTION: WndProc(HWND, unsigned, WORD, LONG)
//
//  PURPOSE:  Processes messages for the main window.
//
//  WM_COMMAND	- process the application menu
//  WM_PAINT	- Paint the main window
//  WM_DESTROY	- post a quit message and return
//
LRESULT CALLBACK WndProc(HWND hWnd,UINT message,WPARAM wparam,LPARAM lparam)
{
	switch(message)
	{
		case	WM_COMMAND:
			return	Wm_CommandProc(hWnd,HIWORD(wparam),LOWORD(wparam),(HWND)lparam);
        case WM_CLOSE:
			return	Wm_CloseProc(hWnd);
		case	WM_DESTROY:
			return	Wm_DestroyProc();
		case	WM_PAINT:
			return	Wm_PaintProc(hWnd);
	}
	return	DefWindowProc(hWnd,message,wparam,lparam);
}

static	LRESULT Wm_CloseProc(HWND hWnd)
{
	int id;
	id = MessageBox(hWnd,
         (LPCSTR)"Really exit ?",
         (LPCSTR)"Exit Message",
         MB_OKCANCEL | MB_ICONQUESTION);
	if(id == IDOK)
		DestroyWindow(hWnd);
	return (0L);
}

static	LRESULT Wm_DestroyProc(void)
{
	PostQuitMessage(0);
	return	0;
}

static	LRESULT	Wm_CommandProc(HWND hWnd,WORD wNotifyCode,WORD wID,HWND hwndCtl)
{
	// Parse the menu selections:
	switch(wID)
	{
		case IDM_ABOUT:
			DialogBox(hInst, (LPCTSTR)IDD_ABOUTBOX, hWnd, (DLGPROC)About);
			break;
		case IDM_EXIT:
           SendMessage(hWnd, WM_CLOSE, 0, 0L);
		   break;
	}
	return	0;
}

static	LRESULT	Wm_PaintProc(HWND hWnd)
{
	PAINTSTRUCT	ps;
	HDC			PaintDC;

	// TODO: Add any drawing code here...
	if(GetUpdateRect(hWnd,NULL,TRUE))
	{
	    char tmp[200];
	    PaintDC=BeginPaint(hWnd,&ps);
	    sprintf(tmp,"DrawGr((HDC)%ld",PaintDC);
	    G__calc(tmp); /* Call Cint parser */
	    /* DrawGr(PaintDC); */
	    EndPaint(hWnd, &ps);
	}
	return	0;
}

//  Drawing Function:
/*
int DrawGr(HDC hdc)
{
    HPEN hPen, hOldPen;
    HBRUSH hBrush, hOldBrush;

    hPen = CreatePen(PS_SOLID, 1, RGB(255, 0, 0));
    hOldPen = SelectObject(hdc, hPen);
    hBrush = CreateHatchBrush(HS_CROSS, RGB(0, 255, 0));
    hOldBrush = SelectObject(hdc, hBrush);
    Rectangle(hdc, 10, 10, 200, 100);
    SelectObject(hdc, hOldPen);
    SelectObject(hdc, hOldBrush);
    DeleteObject(hPen);
    DeleteObject(hBrush);

    hPen = CreatePen(PS_DASH, 1, RGB(255, 100, 10));
    hOldPen = SelectObject(hdc, hPen);
    hBrush = CreateHatchBrush(HS_BDIAGONAL, RGB(0, 0, 255));
    hOldBrush = SelectObject(hdc, hBrush);
    Rectangle(hdc, 40, 40, 240, 140);
    SelectObject(hdc, hOldPen);
    SelectObject(hdc, hOldBrush);
    DeleteObject(hPen);
    DeleteObject(hBrush);

    hPen = CreatePen(PS_DOT, 1, RGB(100, 100, 100));
    hOldPen = SelectObject(hdc, hPen);
    SelectObject(hdc, GetStockObject(NULL_BRUSH));
    Rectangle(hdc, 70, 70, 270, 270);
    SelectObject(hdc, hOldPen);
    DeleteObject(hPen);

    hPen = CreatePen(PS_DASHDOTDOT, 1, RGB(0, 255, 0));
    hOldPen = SelectObject(hdc, hPen);
    hBrush = CreateHatchBrush(HS_DIAGCROSS, RGB(255, 0, 255));
    hOldBrush = SelectObject(hdc, hBrush);
    Rectangle(hdc, 100, 100, 200, 160);
    SelectObject(hdc, hOldPen);
    SelectObject(hdc, hOldBrush);
    DeleteObject(hPen);
    DeleteObject(hBrush);
    return 0;
}
*/

// Mesage handler for about box.
LRESULT CALLBACK About( HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam )
{
	switch( message )
	{
		case WM_INITDIALOG:
				return TRUE;

		case WM_COMMAND:
			if( LOWORD(wParam) == IDOK || LOWORD(wParam) == IDCANCEL ) 
			{
				EndDialog(hDlg, LOWORD(wParam));
				return TRUE;
			}
			break;
	}
    return FALSE;
}
