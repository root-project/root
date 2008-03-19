/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/* =======================================================
	Simple Win32API Application for Drawing Graphics.

	Script.c: Source file that is interpreted by Cint.
	Last modified Time-stamp: <02/01/13 09:48:12 hata>            
	Proposed by K.Hata <kazuhiko.hata@nifty.ne.jp>
========================================================== */

#include <windows.h>

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

    DrawRect4(hdc); /* Call compiled function */
    return 0;
}

