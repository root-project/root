// @(#)root/win32:$Name$:$Id$
// Author: Valery Fine   11/03/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TGWin32WindowsObject
#define ROOT_TGWin32WindowsObject

#include "TGWin32Object.h"
#include "TWin32Command.h"
#include "TWin32Menu.h"
#include "TWin32CommCtrl.h"


class TCanvas;
class TWin32Canvas;
class TGWin32StatusBar;
class TWin32SimpleEditCtrl;

//*-*   Static data member to create menus for all windows

typedef enum { kS1,       kSMenuBreak,kSMenuBarBreak,
               kMFile,    kMEdit,     kMView, kMOptions,  kMInspector, kMClasses, kHelp,
               kMNew,     kMOpen,kS12,kMSave, kMSaveAs,    kS2,kMPrint,kS3, kMClose, kS4, kMQuit, kMExit,
               kMUnDo,kS5,kMEdtor,kS6,kMClearPad,  kMClearCanvas,
               kMColors, kMFonts,     kMMarkers,   kMIconify,     kMX3D, kS7, kMInterrupt,
               kMEventStatus, kS9,  kMAutoFit,kMFitCanvas,kS10,kMRefresh,kS11,kMOptStat,kMOptHTitle,kMOptFit,kMEditHist,
               kMROOTInspect, kMStartBrowser,
               kMClassTree,
               kEndOfMenu
               } EMenuItems;


class TGWin32WindowsObject  : public  TGWin32Object  {

////////////////////////////////////////////////////////////////////
//                                                                //
//  TGWin32WindowsObject                                          //
//                                                                //
//  It defines behaviour of the INTERACTIVE objects of WIN32 GDI  //
//  For instance "real" windows                                   //
//                                                                //
////////////////////////////////////////////////////////////////////

protected:

    friend class TGWin32;
    friend class TGWin32Object;
    friend class TWin32Canvas;
    friend class TWin32BrowserImp;
    friend class TWin32GLViewerImp;

//***    TCanvas           *fCanvas;
    TWin32Canvas      *fCanvasImp;
    int                fButton;       // = 0 button was released, =1,2,3 button was pressed
    DWORD              fDwStyle;      // window style
    DWORD              fDwExtStyle;   // extended window style
    TWin32Menu        *fMenu;         // Handle of the window menu
    TWin32Menu        *fContextMenu;
    TGWin32StatusBar  *fStatusBar;    // Pointer to the status bar object
    TWin32Command      fCommandArray; // TObjArray of the hold the ID numbers of the menu items
    TWin32MenuItem    **fStaticMenuItems; // Pointer to the permanent windows menus

    Bool_t             fMouseActive;  // Flag: whether this object is activated by Mouse clicking


    Int_t              fPaintFlag;      // Flag to synch WM_PAINT operations
    Int_t              fMouseInit;
    Int_t              fButton_Press;
    Int_t              fButton_Up;
    Int_t              fXMouse;         // X position of the last mouse event
    Int_t              fYMouse;         // Y position of the last mouse event
    BOOL               fSystemCursorVisible;
    BOOL               fROOTCursor;
    POINT              fLoc;             // Positions of ROOT graphics locator
    POINT              fLocp;            // Positions of ROOT graphics locator
    TGWin32GetLocator *flpROOTMouse;     // Class to hold the present mouse state
    TGWin32GetString  *flpROOTString;    // Class to hold the present text input state

   //   Data member to control the text input
    BOOL               fSetTextInput;    // Flag "the text input in progress"
    TWin32SimpleEditCtrl *fEditCtrl;     // Edit control object to enter text
    Int_t              fiXText, fiYText; // Coord the current caret
    Char_t             fch;              // entered symbol
    Char_t            *flpInstr;         // Input line buffer
    int                fnCur,            // Current text cursor postion
                       flStr,            // Current length of the entered string
                       fLenLine,         // Pixel length of the entered string
                       fix0,fiy0,
                       fInsert;          // "insert" mode flag

   static int insert;

   //
   //   Protected member functions
   //

   protected:
   HWND               fhwndRootWindow;
   virtual void CreateWindowsObject(TGWin32 *lpTGWin32, Int_t x, Int_t y, UInt_t w, UInt_t h);
   virtual void CreateDoubleBuffer();

   //
   // Windows procdure to manage "Windows" messages:
   //


  //*-*    Message ID: WM_ACTIVATE
  //                =============
  virtual LRESULT APIENTRY OnActivate(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
  static LRESULT APIENTRY OnActivateCB(TGWin32Object *obj,HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
      return ((TGWin32WindowsObject *)obj)->OnActivate(hwnd, uMsg, wParam, lParam);}

  //*-*    Message ID: WM_CLOSE
  //                =============
  virtual LRESULT APIENTRY OnClose(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
  static LRESULT APIENTRY OnCloseCB(TGWin32Object *obj,HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
      return ((TGWin32WindowsObject *)obj)->OnClose(hwnd, uMsg, wParam, lParam);}


  //    Message ID: WM_COMMAND
  //                =============
  virtual LRESULT APIENTRY OnCommand(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
  virtual LRESULT APIENTRY OnCommandForControl(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
  static LRESULT APIENTRY OnCommandCB(TGWin32Object *obj,HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
      return ((TGWin32WindowsObject *)obj)->OnCommand(hwnd, uMsg, wParam, lParam);}

  //    Message ID: WM_ERASEBKGND
  //                =============
  virtual LRESULT APIENTRY OnEraseBkgnd(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
  static LRESULT APIENTRY OnEraseBkgndCB(TGWin32Object *obj,HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
      return ((TGWin32WindowsObject *)obj)->OnEraseBkgnd(hwnd, uMsg, wParam, lParam);}

  //    Message ID: WM_EXITSIZEMOVE
  //                =============
  virtual LRESULT APIENTRY OnExitSizeMove(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
  static LRESULT APIENTRY OnExitSizeMoveCB(TGWin32Object *obj,HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
      return ((TGWin32WindowsObject *)obj)->OnExitSizeMove(hwnd, uMsg, wParam, lParam);}

  //    Message ID: WM_GETMINMAXINFO
  //                ===================
  virtual LRESULT APIENTRY OnGetMinMaxInfo(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
  static LRESULT APIENTRY OnGetMinMaxInfoCB(TGWin32Object *obj,HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
      return ((TGWin32WindowsObject *)obj)->OnGetMinMaxInfo(hwnd, uMsg, wParam, lParam);}

  //    Message ID: WM_MOUSEACTIVATE
  //                ================
  virtual LRESULT APIENTRY OnMouseActivate(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
  static LRESULT APIENTRY OnMouseActivateCB(TGWin32Object *obj,HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
      return ((TGWin32WindowsObject *)obj)->OnMouseActivate(hwnd, uMsg, wParam, lParam);};

  //    Message ID: WM_LBUTTONDOWN(UP) WM_MBUTTONDOWN(UP) WM_RBUTTONDOWN(UP)
  //                ================== ================== ==================
  //                WM_LBUTTONDBLCLK  WM_MBUTTONDBLCLK   WM_RBUTTONDBLCLK
  //                ================== ================== ==================
  //                WM_MOUSEMOVE
  //                ============
  virtual LRESULT APIENTRY OnMouseButton(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
  static LRESULT APIENTRY OnMouseButtonCB(TGWin32Object *obj,HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
      return ((TGWin32WindowsObject *)obj)->OnMouseButton(hwnd, uMsg, wParam, lParam);};
  //    Message ID: WM_PALETTECHANGED
  //                =================
  virtual LRESULT APIENTRY OnPaletteChanged(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
  static LRESULT APIENTRY OnPaletteChangedCB(TGWin32Object *obj,HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
      return ((TGWin32WindowsObject *)obj)->OnPaletteChanged(hwnd, uMsg, wParam, lParam);};

  //    Message ID: WM_SETFOCUS
  //                ===========
  virtual LRESULT APIENTRY OnSetFocus(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
  static LRESULT APIENTRY OnSetFocusCB(TGWin32Object *obj,HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
      return ((TGWin32WindowsObject *)obj)->OnSetFocus(hwnd, uMsg, wParam, lParam);};
  //    Message ID: WM_KILLFOCUS
  //                ===========
  virtual LRESULT APIENTRY OnKillFocus(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
  static LRESULT APIENTRY OnKillFocusCB(TGWin32Object *obj,HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
      return ((TGWin32WindowsObject *)obj)->OnKillFocus(hwnd, uMsg, wParam, lParam);};
  //    Message ID: WM_CHAR
  //                =======
  virtual LRESULT APIENTRY OnChar(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
  static LRESULT APIENTRY OnCharCB(TGWin32Object *obj,HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
      return ((TGWin32WindowsObject *)obj)->OnChar(hwnd, uMsg, wParam, lParam);};
  //    Message ID: WM_KEYDOWN
  //                ==========
  virtual LRESULT APIENTRY OnKeyDown(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
  static LRESULT APIENTRY OnKeyDownCB(TGWin32Object *obj,HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
      return ((TGWin32WindowsObject *)obj)->OnKeyDown(hwnd, uMsg, wParam, lParam);};


  //                WM_NOTIFY
  //                =========
  virtual LRESULT APIENTRY OnNotify(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
                                      return DefWindowProc(hwnd, uMsg, wParam, lParam);}
  static LRESULT APIENTRY OnNotifyCB(TGWin32Object *obj,HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
      return ((TGWin32WindowsObject *)obj)->OnNotify(hwnd, uMsg, wParam, lParam);};

  //    Message ID: WM_PAINT
  //                =======
  virtual LRESULT APIENTRY OnPaint      (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
  static LRESULT APIENTRY OnPaintCB    (TGWin32Object *obj,HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
      return ((TGWin32WindowsObject *)obj)->OnPaint(hwnd, uMsg, wParam, lParam);}

  //    Message ID: WM_SIZE
  //                =======
  virtual LRESULT APIENTRY OnSize     (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
  static LRESULT APIENTRY OnSizeCB   (TGWin32Object *obj,HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
      return ((TGWin32WindowsObject *)obj)->OnSize(hwnd, uMsg, wParam, lParam);}

  //    Message ID: WM_SIZING
  //                =======
  virtual LRESULT APIENTRY OnSizing   (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
  static LRESULT APIENTRY OnSizingCB (TGWin32Object *obj,HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
      return ((TGWin32WindowsObject *)obj)->OnSizing(hwnd, uMsg, wParam, lParam);}

  //    Message ID: WM_SYSCOMMAND
  //                =============
  virtual LRESULT APIENTRY OnSysCommand(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
  static LRESULT APIENTRY OnSysCommandCB(TGWin32Object *obj,HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
      return ((TGWin32WindowsObject *)obj)->OnSysCommand(hwnd, uMsg, wParam, lParam);}

  //    Message ID: WM_USER+10 OnRootInput
  //                ==========
  virtual LRESULT APIENTRY OnRootInput(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
  static LRESULT APIENTRY OnRootInputCB(TGWin32Object *obj,HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
      return ((TGWin32WindowsObject *)obj)->OnRootInput(hwnd, uMsg, wParam, lParam);}

  //   Message ID:  WM_USER+10 OnRootInput  LOWORD(wParam) = IX_REQLO
  virtual LRESULT APIENTRY OnRootMouse(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
  //                                        LOWORD(wParam) = IX_REQST
  virtual LRESULT APIENTRY OnRootTextInput(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
  virtual LRESULT APIENTRY OnRootEditInput(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

  //    Message ID: WM_USER+    OnRootHook
  //                ==========
  virtual LRESULT APIENTRY OnRootHook(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
  static LRESULT APIENTRY OnRootHookCB(TGWin32Object *obj,HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
      return ((TGWin32WindowsObject *)obj)->OnRootHook(hwnd, uMsg, wParam, lParam);}



  void DrawROOTCursor(int ctyp);
  void ROOTCursorInit(HWND hwnd,int ctyp);
  void RestoreROOT(int ctyp);
  BOOL IsMouseLeaveCanvas(Int_t x, Int_t y);
  TCanvas  *GetCanvas();  // Enter Critical section to access Canvas



public:

    TGWin32WindowsObject();

    TGWin32WindowsObject(TGWin32 *lpTGWin32, Int_t x=0, Int_t y=0, UInt_t w=512, UInt_t h=512);
    TGWin32WindowsObject(TGWin32 *lpTGWin32, UInt_t w=512, UInt_t h=512);
    virtual ~TGWin32WindowsObject();

    virtual void      W32_Clear();
    virtual void      W32_Close();
    virtual void      W32_CopyTo(TGWin32Object *obj, int xpos, int ypos);
    virtual void      W32_CreateStatusBar(Int_t nparts=1);
    virtual void      W32_CreateStatusBar(Int_t *parts, Int_t nparts=1);
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
            void      W32_Set(int x, int y, int w, int h);
    virtual void      W32_SetCharacterUp(Float_t chupx, Float_t chupy);
    virtual void      W32_SetClipOFF();
    virtual void      W32_SetClipRegion(int x, int y, int w, int h);
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
    virtual void      W32_SetStatusText(const Text_t *text, Int_t partidx=0,Int_t stype=0);
    virtual void      W32_SetTextAlign(Short_t talign=11);
    virtual void      W32_SetTextColor(Color_t cindex);
    virtual Int_t     W32_SetTextFont(char *fontname, TVirtualX::ETextSetMode mode);
    virtual void      W32_SetTextFont(Int_t fontnumber);
    virtual void      W32_SetTextSize(Float_t textsize);
    void  W32_SetMenu();
    virtual void      W32_SetTitle(const char *title);
    void W32_Show();
    void W32_ShowMenu();
    void W32_ShowMenu(Int_t x, Int_t y);
    void W32_SetMenu(HMENU menu);
    void W32_ShowStatusBar(Bool_t show = kTRUE);
    virtual void      W32_Update(int mode=1);
    virtual void      W32_Warp(int ix, int iy);
    virtual void      W32_WriteGIF(char *name);
    virtual void      W32_WritePixmap(unsigned int w, unsigned int h, char *pxname);

    virtual Int_t     ExecCommand(TGWin32Command *servercommand);
    virtual void      Win32CreateObject();
    TWin32MenuItem   *GetStaticItem(Int_t idx){ return fStaticMenuItems[idx]; }
    HWND              GetWindow(){return fhwndRootWindow;}   // return a window handle
    TWin32Menu       *GetWindowMenu(){ return fMenu;}
    TWin32Menu       *GetWindowContextMenu(){ return fContextMenu;}
    void              JoinMenu(TVirtualMenuItem *item){ fCommandArray.JoinMenuItem(item); }
    virtual void      MakeMenu(){;}  // To generate the array of the menu items
    void              RegisterMenuItem(TVirtualMenuItem *item){fCommandArray.JoinMenuItem(item);}
    void              RegisterControlItem(TWin32CommCtrl *ctrl){fCommandArray.JoinControlItem(ctrl);}
    virtual void      RunMenuItem(Int_t index);
    // void              SetCanvas(TCanvas *canvas){ fCanvas = canvas;}
    void              SetCanvas(TCanvas *canvas, TWin32Canvas *canvasimp){ /* SetCanvas(canvas); */ fCanvasImp = canvasimp;}
    void              SetWindow(HWND hwnd){fhwndRootWindow = hwnd;}   // set the window handle
    void              StartPaint(){ fPaintFlag++; }
    void              UnRegisterMenuItem(TVirtualMenuItem *item);    // Remove the menu item from the list
    void              UnRegisterControlItem(TWin32CommCtrl *ctrl);   // Remove the Common control from the list
    void              UnRegisterMenuItem(Int_t itemidx) {fCommandArray.RemoveAt(itemidx);} //Remove either menu or control item
    void              FinishPaint(){ fPaintFlag--;}
    Int_t             GetPaint(){return fPaintFlag;}
    void              LeaveCrSection(){if (fCanvasImp && !InSendMessage()) fWin32Mother->release_read_lock();}

//   void              LinkMenu(TWin32Menu *menu){HMENU m = menu->GetMenuHandle(); if (m) SetMenu(fhwndRootWindow,m);}
//   HMENU             UnLinkMenu(){HMENU m = GetMenu(fhwndRootWindow); if (m) SetMenu(fhwndRootWindow,NULL);
//                      return m; }


  //    Message ID: WM_CREATE
  //

  virtual LRESULT APIENTRY OnCreate(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
  static LRESULT APIENTRY OnCreateCB(TGWin32Object *obj,HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
      return ((TGWin32WindowsObject *)obj)->OnCreate(hwnd, uMsg, wParam, lParam);}


//    LPTHREAD_START_ROUTINE  ROOT_MsgLoop(HANDLE ThrSem);

    // ClassDef(TGWin32WindowsObject,0)  //Interface to Win32
 };

#endif
