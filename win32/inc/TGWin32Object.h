// @(#)root/win32:$Name:  $:$Id: TGWin32Object.h,v 1.1.1.1 2000/05/16 17:00:47 rdm Exp $
// Author: Valery Fine   10/01/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGWin32Object                                                        //
//                                                                      //
// Interface to low level Windows32. This class gives access to basic   //
// Win32 graphics, pixmap, text and font handling routines.             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGWin32Object
#define ROOT_TGWin32Object

#include "TGWin32.h"
#include "TObject.h"
#include "TWin32CallBackList.h"

#ifndef ROOT_TCritSection
#include "TCritSection.h"
#endif

#ifndef ROOT_Win32Constants
#include "Win32Constants.h"
#endif

#include "TGWin32Command.h"

class TGWin32Object;

//______________________________________________________________________________
class TGWin32Switch  :  public TObject  {

private:

 Int_t          fMasterIsActive;  // = 0 Master object is passive
 TGWin32Object *flpMasterObject;  // Object to implement WIN32 direct opeartions
 TGWin32Switch *flpMirror;        // Object to implement double buffering and "mirror" operation
 Bool_t         fOwnMasterFlag;   // Whether this object "owns" the Master object and gas to delete it alone

public:

    TGWin32Switch():fMasterIsActive(-1),flpMasterObject(0),flpMirror(0),fOwnMasterFlag(kFALSE){}
    TGWin32Switch(TGWin32Object *master, TGWin32Switch *mirror = 0, Bool_t ownmaster = kTRUE );
    TGWin32Switch(TGWin32Object *master,  Bool_t ownmaster);
    virtual ~TGWin32Switch();

    virtual void      W32_Clear();
    virtual void      W32_Close();
    virtual void      W32_CopyTo(TGWin32Object *obj, int xpos, int ypos);
    virtual void      W32_CreateOpenGL();
    virtual void      W32_DeleteOpenGL();
    virtual void      W32_DrawBox(int x1, int y1, int x2, int y2, TVirtualX::EBoxMode mode);
    virtual void      W32_DrawCellArray(int x1, int y1, int x2, int y2, int nx, int ny, int *ic);
    virtual void      W32_DrawFillArea(int n, TPoint *xy);
    virtual void      W32_DrawLine(int x1, int y1, int x2, int y2);
    virtual void      W32_DrawPolyLine(int n, TPoint *xy);
    virtual void      W32_DrawPolyMarker(int n, TPoint *xy);
    virtual void      W32_DrawText(int x, int y, float angle, float mgn, const char *text, TVirtualX::ETextMode mode);
    virtual void      W32_GetCharacterUp(Float_t &chupx, Float_t &chupy);
    Int_t     W32_GetDoubleBuffer();
    virtual void      W32_GetGeometry(int &x, int &y, unsigned int &w, unsigned int &h);
    virtual void      W32_GetPixel(int y, int width, Byte_t *scline);
    virtual void      W32_GetRGB(int index, float &r, float &g, float &b);
    virtual void      W32_GetTextExtent(unsigned int &w, unsigned int &h, char *mess);
    virtual void      W32_Move(Int_t x, Int_t y);
    virtual void      W32_PutByte(Byte_t b);
    virtual void      W32_QueryPointer(int &ix, int &iy);
    virtual Int_t     W32_RequestLocator(Int_t mode, Int_t ctyp, Int_t &x, Int_t &y);
    virtual Int_t     W32_RequestString(int x, int y, char *text);
    Int_t     W32_Rescale(int wid, unsigned int w, unsigned int h);
    virtual void      W32_Resize();
    virtual void      W32_Select();
    virtual void      W32_SetCharacterUp(Float_t chupx, Float_t chupy);
    virtual void      W32_SetClipOFF();
    virtual void      W32_SetClipRegion(int x, int y, unsigned int w, unsigned int h);
    virtual void      W32_SetCursor(ECursor cursor);
    void      W32_SetDoubleBuffer(int mode);
    void      W32_SetDoubleBufferOFF();
    void      W32_SetDoubleBufferON();
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

            void      Delete();
    TGWin32Object    *GetMasterObject();
            void      SetMasterObject(TGWin32Object *master){flpMasterObject = master;}

    // ClassDef(TGWin32Switch,0)  // TGWin32Switch interface
};

class TVirtualXOpenGL;

//______________________________________________________________________________
class TGWin32Object  :  public TObject  {

protected:

     friend class TGWin32;

     Int_t    fIsPixmap;              // = 1 object contains NON windows object (pixmap, for instance)
     TGWin32 *fWin32Mother;           // Pointer to the mother object
     TWin32CallBackList fWinAction;   // List of the callback functions to manage the events

     TGWin32Object *fBufferObj; // Double buffering object (pixmap usually)
     TCritSection  fBufferObjSec; // Crit section to access fBufferObj data member

     Int_t    fDoubleBuffer;    // Double buffer on flag
     TCritSection fDoubleBufferSec; // Crit section to access fDoubleBuffer data member

     HANDLE   fhSemaphore;     // Win32 semaphore to synch events
     HDC      fObjectDC;       // WIN32 Device Context handle;
     HRGN     fObjectClipRegion;
     RECT     fWin32WindowSize;

     POINT    fPosition;     // X(x) and Y(y) coord's of the frame
     SIZE     fWinSize;      // width(cx)&height(cy) of the window
     SIZE     fSizeFull;     // width(cx)&height(cy) of the window (size of the bord is included)
     int      fClip;         // 1 if the clipping is on
     int      fXclip;        // x coordinate of the clipping rectangle
     int      fYclip;        // y coordinate of the clipping rectangle
     int      fWclip;        // width of the clipping rectangle
     int      fHclip;        // height of the clipping rectangle

     DWORD    fdwCharX;
     DWORD    fdwCharY;
     DWORD    fdwAscent;


     Int_t    fTypeFlag;     // = 2 means text window

//*-*
//*-*   Input echo Graphic Context global for all windows
//*-*

     int      fFill_hollow;                 // Flag if fill style is hollow
     int      fCurrent_fasi;                // Current fill area style index
     int      fAlign_hori; // = -1;         // Align text left, center, right
     int      fAlign_vert; // = -1;         // Align text bottom, middle, top

     int      fCurrent_font_number; //  = 0 // current font number in font[]
     LRESULT APIENTRY RootAct(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
     void SetPosition(int x, int y) {fPosition.x = x; fPosition.y = y;} //save the pixmap origin

public:

//    virtual TGWin32Object(TGWin32 *, Int_t doublebuffer, Int_t ispixmap); // To implement InitWindow
//    virtual TGWin32Object(TGWin32 *lpTGWin32, UInt_t w, UInt_t h);        // To implement OpenPixmap

    TGWin32Object();
    virtual ~TGWin32Object();

    void  Delete();

//    virtual void      W32_Clear() = 0;
    virtual void      W32_Close() = 0;
    virtual void      W32_CopyTo(TGWin32Object *obj, int xpos, int ypos) = 0;
    virtual void      W32_CreateOpenGL();
    virtual void      W32_DeleteOpenGL(){;}

//    virtual void      W32_DrawBox(int x1, int y1, int x2, int y2, TVirtualX::EBoxMode mode) = 0;
//    virtual void      W32_DrawCellArray(int x1, int y1, int x2, int y2, int nx, int ny, int *ic) = 0;
//    virtual void      W32_DrawFillArea(int n, TPoint *xy) = 0;
//    virtual void      W32_DrawLine(int x1, int y1, int x2, int y2) = 0;
//    virtual void      W32_DrawPolyLine(int n, TPoint *xy) = 0;
//    virtual void      W32_DrawPolyMarker(int n, TPoint *xy) = 0;
//    virtual void      W32_DrawText(int x, int y, float angle, float mgn, const char *text, TVirtualX::ETextMode mode) = 0;
    virtual void      W32_GetCharacterUp(Float_t &chupx, Float_t &chupy) = 0;
    virtual Int_t     W32_GetDoubleBuffer() = 0;
    virtual void      W32_GetGeometry(int &x, int &y, unsigned int &w, unsigned int &h) = 0;
    virtual void      W32_GetPixel(int y, int width, Byte_t *scline) = 0;
//    virtual void      W32_GetRGB(int index, float &r, float &g, float &b) = 0;
    virtual void      W32_GetTextExtent(unsigned int &w, unsigned int &h, char *mess) = 0;
    virtual void      W32_Move(Int_t x, Int_t y) = 0;
    virtual void      W32_PutByte(Byte_t b) = 0;
    virtual void      W32_QueryPointer(int &ix, int &iy) = 0;
    virtual Int_t     W32_RequestLocator(Int_t mode, Int_t ctyp, Int_t &x, Int_t &y) = 0;
    virtual Int_t     W32_RequestString(int x, int y, char *text) = 0;
    virtual TGWin32Object *Rescale(unsigned int w, unsigned int h) = 0;
    virtual void      W32_Resize() = 0;
    virtual void      W32_Select();
    virtual void      W32_SetCharacterUp(Float_t chupx, Float_t chupy) = 0;
//  virtual void      W32_SetClipOFF() = 0;
//  virtual void      W32_SetClipRegion(int x, int y, unsigned int w, unsigned int h) = 0;
    virtual void      W32_SetCursor(ECursor cursor) = 0;
    virtual void      W32_SetDoubleBuffer(int mode) = 0;
    virtual void      W32_SetDoubleBufferOFF() = 0;
    virtual void      W32_SetDoubleBufferON()  = 0;
//    virtual void      W32_SetDrawMode(TVirtualX::EDrawMode mode)  = 0;
    virtual void      W32_SetFillColor(Color_t cindex) = 0;
    virtual void      W32_SetFillStyle(Style_t style)  = 0;
    virtual void      W32_SetLineColor(Color_t cindex) = 0;
    virtual void      W32_SetLineType(int n, int *dash) = 0;
    virtual void      W32_SetLineStyle(Style_t linestyle) = 0;
    virtual void      W32_SetLineWidth(Width_t width);
    virtual void      W32_SetMarkerColor( Color_t cindex) = 0;
    virtual void      W32_SetMarkerSize(Float_t markersize) = 0;
    virtual void      W32_SetMarkerStyle(Style_t markerstyle) = 0;
    virtual void      W32_SetRGB(int cindex, float r, float g, float b) = 0;
    virtual void      W32_SetTextAlign(Short_t talign=11) = 0;
    virtual void      W32_SetTextColor(Color_t cindex) = 0;
    virtual Int_t     W32_SetTextFont(char *fontname, TVirtualX::ETextSetMode mode) = 0;
    virtual void      W32_SetTextFont(Int_t fontnumber) = 0;
    virtual void      W32_SetTextSize(Float_t textsize) = 0;
    virtual void      W32_SetTitle(const char *title) = 0;
    virtual void      W32_Update(int mode) = 0;
    virtual void      W32_Warp(int ix, int iy) = 0;
    virtual void      W32_WriteGIF(char *name) = 0;
    virtual void      W32_WritePixmap(unsigned int w, unsigned int h, char *pxname) = 0;


    virtual Int_t     ExecCommand(TGWin32Command *servercommand) = 0;
            Int_t     GetObjectType(){return fIsPixmap;}
            HDC       GetWin32DC(){return fObjectDC;} // returns WIN32 Device Context handle;
            TGWin32  *GetWin32ObjectMother(){ return fWin32Mother;}  // returns a pointer
                                                                     // to the mother object

    virtual void      Win32CreateObject() = 0;
    virtual void      Win32CreateCallbacks();

    LRESULT         CallCallback(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
    int             CharWidth(char ch); // retrieves the widthsof 'ch', in logical coordinates

    LRESULT APIENTRY   OnRootAct(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
    static LRESULT APIENTRY OnRootActCB(TGWin32Object *obj,HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
         return obj->OnRootAct(hwnd, uMsg, wParam, lParam);}

    LRESULT APIENTRY     Wnd_BOX(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
    LRESULT APIENTRY    Wnd_CPPX(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
    LRESULT APIENTRY   Wnd_FLARE(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
    LRESULT APIENTRY   Wnd_MARKE(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
    LRESULT APIENTRY      Wnd_CA(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);


    // ClassDef(TGWin32Object,0)  //Interface to Win32

 };

#endif
