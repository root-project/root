/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/********************************************************************
* CompiledLib.c
*  This file contains precompiled library that is compiled and embedded
*  into Cint interpreter. 
********************************************************************/

#include <windows.h>

void DrawRect4(HDC hdc) {
    HPEN hPen, hOldPen;
    HBRUSH hBrush, hOldBrush;

    hPen = CreatePen(PS_DASHDOTDOT, 1, RGB(0, 255, 0));
    hOldPen = SelectObject(hdc, hPen);
    hBrush = CreateHatchBrush(HS_DIAGCROSS, RGB(255, 0, 255));
    hOldBrush = SelectObject(hdc, hBrush);
    Rectangle(hdc, 100, 100, 200, 160);
    SelectObject(hdc, hOldPen);
    SelectObject(hdc, hOldBrush);
    DeleteObject(hPen);
    DeleteObject(hBrush);
}


