/* @(#)root/win32:$Name$:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TGWin32PixmapObject
#define ROOT_TGWin32PixmapObject

#include "TGWin32Object.h"

class TGWin32PixmapObject : public TGWin32Object  {

////////////////////////////////////////////////////////////////////
//                                                                //
//  TGWin32PixmapObject                                           //
//                                                                //
//  It defines behaviour of the BATCH objects of WIN32 GDI        //
//  For instance, Pixmaps, Bitmaps, Window MetaFiles and so on    //
//                                                                //
////////////////////////////////////////////////////////////////////

private:

    HBITMAP      fRootPixmap;
    Bool_t       fModified;        //Set to true when pixmap is modified

    static HBITMAP TGWin32PixmapObject::CreateDIBSurface(HDC hWndDC,Int_t w, Int_t h);


public:

    TGWin32PixmapObject();                                        // To implement OpenPixmap
    TGWin32PixmapObject(TGWin32 *lpTGWin32, UInt_t w, UInt_t h);  // To implement OpenPixmap
    virtual ~TGWin32PixmapObject();

    virtual void      W32_Clear();
    virtual void      W32_Close();
    virtual void      W32_CopyTo(TGWin32Object *obj, int xpos, int ypos);
    virtual void      W32_DrawBox(int x1, int y1, int x2, int y2, TVirtualX::EBoxMode mode);
    virtual void      W32_DrawCellArray(int x1, int y1, int x2, int y2, int nx, int ny, int *ic);
    virtual void      W32_DrawFillArea(int n, TPoint *xy);
    virtual void      W32_DrawLine(int x1, int y1, int x2, int y2);
    virtual void      W32_DrawPolyLine(int n, TPoint *xy);
    virtual void      W32_DrawPolyMarker(int n, TPoint *xy);
    virtual void      W32_DrawText(int x, int y, float angle, float mgn, const char *text, TVirtualX::ETextMode mode);
    virtual void      W32_GetCharacterUp(Float_t &chupx, Float_t &chupy);
    virtual Int_t     W32_GetDoubleBuffer();
    virtual void      W32_GetGeometry(int &x, int &y, unsigned int &w, unsigned int &h);
    virtual void      W32_GetPixel(int y, int width, Byte_t *scline);
    virtual void      W32_GetRGB(int index, float &r, float &g, float &b);
    virtual void      W32_GetTextExtent(unsigned int &w, unsigned int &h, char *mess);
    virtual void      W32_Move(Int_t x, Int_t y);
    virtual void      W32_PutByte(Byte_t b);
    virtual void      W32_QueryPointer(int &ix, int &iy);
    virtual Int_t     W32_RequestLocator(Int_t mode, Int_t ctyp, Int_t &x, Int_t &y);
    virtual Int_t     W32_RequestString(int x, int y, char *text);
    virtual TGWin32Object *Rescale(unsigned int w, unsigned int h);
    virtual void      W32_Resize();
    virtual void      W32_Select();
    virtual void      W32_SetCharacterUp(Float_t chupx, Float_t chupy);
    virtual void      W32_SetClipOFF();
    virtual void      W32_SetClipRegion(int x, int y, unsigned int w, unsigned int h);
    virtual void      W32_SetCursor(ECursor cursor);
    virtual void      W32_SetDoubleBuffer(int mode);
    virtual void      W32_SetDoubleBufferOFF();
    virtual void      W32_SetDoubleBufferON();
    virtual void      W32_SetDrawMode(TVirtualX::EDrawMode mode);
    virtual void      W32_SetFillColor(Color_t cindex);
    virtual void      W32_SetFillStyle(Style_t style);
    virtual void      W32_SetLineColor(Color_t cindex);
    virtual void      W32_SetLineType(int n, int *dash);
    virtual void      W32_SetLineStyle(Style_t linestyle);
    virtual void      W32_SetLineWidth(Width_t width);
    virtual void      W32_SetMarkerColor( Color_t cindex);
    virtual void      W32_SetMarkerSize(Float_t markersize);
    virtual void      W32_SetMarkerStyle(Style_t markerstyle);
    virtual void      W32_SetRGB(int cindex, float r, float g, float b);
    virtual void      W32_SetTextAlign(Short_t talign=11);
    virtual void      W32_SetTextColor(Color_t cindex);
    virtual Int_t     W32_SetTextFont(char *fontname, TVirtualX::ETextSetMode mode);
    virtual void      W32_SetTextFont(Int_t fontnumber);
    virtual void      W32_SetTextSize(Float_t textsize);
    virtual void      W32_SetTitle(const char *title);
    virtual void      W32_Update(int mode);
    virtual void      W32_Warp(int ix, int iy);
    virtual void      W32_WriteGIF(char *name);
    virtual void      W32_WritePixmap(unsigned int w, unsigned int h, char *pxname);

    Int_t   ExecCommand(TGWin32Command *servercommand);
    Bool_t            IsModified() {return fModified;}
    void              Modified(Bool_t flag=1) { fModified = flag; }
    virtual void      Win32CreateObject();
    HBITMAP           GetBitmap(); // Return the Handle of the BITMAP data structure

    // ClassDef(TGWin32PixmapObject, 0)  // Pixmap GDI objects for Win32
   };

#endif
