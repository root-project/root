// @(#)root/winnt:$Name$:$Id$
// Author: Valery Fine   23/01/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TGWin32Command
#define ROOT_TGWin32Command

#ifndef ROOT_TVirtualX
#include "TVirtualX.h"
#endif

#include "Windows4Root.h"

#ifndef ROOT_Win32Constants
#include "Win32Constants.h"
#endif

#ifndef ROOT_TWin32Semaphore
#include "TWin32Semaphore.h"
#endif

class TGWin32Object;

//______________________________________________________________________________
class TGWin32Command {

private:
    int     fMasterFlag;   // Command suits the master object only
    int     fCodeOP;       // Code of the opertation
    int     fBuffered;     // Command does sense "Double biffer" mode
    UINT    fMessageID;    // ID of the windows message to perform this command;

public:
   TGWin32Command(int code, int type=ROOT_Primitive,int master=0);
   void SetMsgID(UINT uMsg=IX11_ROOT_MSG){fMessageID = uMsg;}
   virtual void SetCOP(int code);
   virtual int  GetCOP();
   UINT GetMsgID(){ return fMessageID; }
   void SetBuffered(int buffered=1){fBuffered=buffered;}
   int  GetBuffered(){ return fBuffered;}
};

//______________________________________________________________________________
class TGWin32GLCommand :  public TWin32Semaphore, public TGWin32Command
{

public:
  TGWin32GLCommand(int code=GL_MAKECURRENT) : TGWin32Command(code,ROOT_OpenGL){;}
};

//______________________________________________________________________________
class TGWin32Box : public TGWin32Command {

private:

  int fX1;  // Coordinate of the corners of the box to draw
  int fY1;
  int fX2;
  int fY2;
  int fMode;

public:

  TGWin32Box(int x1, int y1, int x2, int y2, TVirtualX::EBoxMode mode,int code=IX_BOX) : TGWin32Command(code){
    fX1 = min(x1,x2);
    fX2 = max(x1,x2);
    fY1 = min(y1,y2);
    fY2 = max(y1,y2);
    fMode = (int) mode;
  }

  int GetX1(){return fX1;}
  int GetY1(){return fY1;}
  int GetX2(){return fX2;}
  int GetY2(){return fY2;}
  int GetMode(){return fMode;}
};

//______________________________________________________________________________
class TGWin32SetDoubleBuffer : public TGWin32Command {
//*-*-*-*-*-*-*-*-*-*-*-*TGWin32DoubleBuffer*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*  This class changes the Double buffer mode of the affected window
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
private:
    int fDoubleBuffer;
public:
  TGWin32SetDoubleBuffer(int doublebuf=1,int code=IX_SETBUF):TGWin32Command(code,ROOT_Control)
    {fDoubleBuffer = doublebuf; SetBuffered(0);}
  int GetBuffer(){ return fDoubleBuffer;}
};

//______________________________________________________________________________
class TGWin32GetDoubleBuffer :  public TWin32Semaphore, public TGWin32Command {
//*-*-*-*-*-*-*-*-*-*-*-*TGWin32DoubleBuffer*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*  This class tests the Double buffer mode of the affected window
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
private:
    int fDoubleBuffer;
public:
  TGWin32GetDoubleBuffer(int code=IX_GETBUF):TGWin32Command(code,ROOT_Control) {SetBuffered(0);;}
  int GetBuffer(){ return fDoubleBuffer;}
  void SetBuffer(int mode){fDoubleBuffer = mode;}
};

//______________________________________________________________________________
class TGWin32Cell : public TGWin32Command {

private:

  int fX1;  // Coordinate of the corners of the box to draw
  int fY1;
  int fX2;
  int fY2;
  int fNx;         // Number of cell along X
  int fNy;         // Number of cell along Y
  int *fCellArray; // Array of the cells to draw

public:

  TGWin32Cell(int x1, int y1, int x2, int y2, int nx, int ny, int *cells,int code=IX_CA) : TGWin32Command(code){
    fX1 = x1;
    fX2 = x2;
    fY1 = y1;
    fY2 = y2;
    fNx = nx;
    fNy = ny;
    fCellArray = cells;
  }

  int  GetX1(){return fX1;}
  int  GetY1(){return fY1;}
  int  GetX2(){return fX2;}
  int  GetY2(){return fY2;}
  int  GetNx(){return fNx;}
  int  GetNy(){return fNy;}
  int *GetCells(){return fCellArray;}
};

//______________________________________________________________________________
class TGWin32Clear : public TGWin32Command {

public:
  TGWin32Clear(int code=IX_CLRWI) : TGWin32Command(code,ROOT_Control){ ; }
};

//______________________________________________________________________________
class TGWin32Clip : public TGWin32Command {

    RECT fRegion;  // The rectangle area to clip Win object

public:
  TGWin32Clip(int w = 0, int h = 0, int x = 0, int y =0,  int code=IX_CLIP) : TGWin32Command(code,ROOT_Control){
   fRegion.left   =  x;
   fRegion.top    =  y;
   fRegion.right  =  x+w;
   fRegion.bottom =  y+h;
 }
};

//______________________________________________________________________________
class TGWin32CopyTo : public TGWin32Command {

private:

   TGWin32Object *fSourceWinObject; // Poiter to the Win object where this will be copied to
   POINT          fPointFrom;         // List of the nodes
   POINT          fPointTo;         // List of the nodes

public:

   TGWin32CopyTo(TGWin32Object *, int xpost, int ypost, int xposf, int yposf, int code=IX_CPPX);

   TGWin32Object *GetSource();
   POINT *GetPointsFrom();
   POINT *GetPointsTo();
};

//______________________________________________________________________________
class TGWin32CreateStatusBar :  public TWin32Semaphore, public TGWin32Command {
//*-*-*-*-*-*-*-*-*-*-*-*TGWin32CreateStatusBar-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*  This class creates a child WIN32 status window
//*-*  It returns the HWND handle of the child status window
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
private:
    HWND fhwndWindow ;
public:
  TGWin32CreateStatusBar(int code=IX_SETSTATUS):TGWin32Command(code,ROOT_Control) {SetBuffered(0);}
  HWND GetWindow(){ return fhwndWindow;}
  void SetWindow(HWND hwnd){fhwndWindow = hwnd;}
};

//______________________________________________________________________________
class TGWin32UpdateWindow : public TGWin32CopyTo {
public:
   TGWin32UpdateWindow(int code=IX_UPDWI) :
                      TGWin32CopyTo(0,0,0,0,0,code){SetBuffered(0);}
};
//______________________________________________________________________________
class TGWin32DrawMode : public TGWin32Command {


private:
   int    fMode;
public:
  TGWin32DrawMode(TVirtualX::EDrawMode mode, int code=IX_DRMDE) : TGWin32Command(code,ROOT_Attribute){
     fMode = Win32DrawMode[mode-1];
  }
  int GetMode(){return fMode;}
};

//______________________________________________________________________________
class TGWin32DrawPolyLine : public TGWin32Command {

private:
   int    fNum;            // Number of the nodes in the polylines (=1 means just a single pixel)
   POINT *flpPoint;        // A position inside of the destination object to copy this

public:
   TGWin32DrawPolyLine(int n, POINT *lpp, int code=IX_LINE);
   int GetNumber();
   POINT *GetPoints();
};


//______________________________________________________________________________
class TGWin32DrawText : public TGWin32Command {

//*-*-*-*-*-*-*-*-*-*-*Draw a text string using current font*-*-*-*-*-*-*-*-*-*
//*-*                  =====================================
//*-*  mode       : drawing mode
//*-*  mode=0     : the background is not drawn (kClear)
//*-*  mode=1     : the background is drawn (kSolid)
//*-*  x,y        : text position
//*-*  angle      : text angle
//*-*  mgn        : magnification factor
//*-*  text       : text string
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

private:

   int   fX;       // Text position;
   int   fY;
   const char  *fText;  // text string
   TVirtualX::ETextMode fMode; // Draw mode

public:

    TGWin32DrawText(int x, int y, const char *text, TVirtualX::ETextMode mode=TVirtualX::kClear, int code=IX_TEXT)
                : TGWin32Command(code,ROOT_Text)
    {
      fX = x;
      fY = y;
      fText = text;
      fMode = mode;
    }

    int   GetX(){return fX;}
    int   GetY(){return fY;}
    void  SetSize(SIZE *lpSize){fX = lpSize->cx;fY = lpSize->cy;}
    TVirtualX::ETextMode GetMode(){ return fMode;}

    const char *GetText(){return fText;}
};

//______________________________________________________________________________
class TGWin32GetColor : public TGWin32Command {

private:

   int    fCindex;         // Color index
   PALETTEENTRY fPalette;  // Palette entry data structure

public:

   TGWin32GetColor(int ci, int code=IX_GETCOL) : TGWin32Command(code,ROOT_Inquiry){fCindex = ci;}
   PALETTEENTRY  *GetPalPointer(){return &fPalette;}
   int GetCIndex(){return fCindex;}
   int Red(){return (int) fPalette.peRed;}
   int Green(){return (int) fPalette.peGreen;}
   int Blue(){return (int) fPalette.peBlue;}


};

//______________________________________________________________________________
class TGWin32GetLocator : public TWin32Semaphore, public TGWin32Command {

private:

   LONG fX;      // x cursor position (initial and queried)
   LONG fY;      // y cursor postion
   int  fType ;  //  shape of the cursor:
                 //    =1 tracking cross
                 //    =2 cross-hair
                 //    =3 rubber circle
                 //    =4 rubber band
                 //    =5 rubber rectangle

   int  fButton; //  Number of the pressed button
   int  fMode;   //  Input mode:
                 //    =0 request
                 //    =1 sample

public:

  TGWin32GetLocator(int x, int y, int ctyp, int mode, int code = IX_REQLO) : TGWin32Command(code,ROOT_Input){
     fX = x;
     fY = y;
     fType = ctyp;
     fMode = mode;
  }

   int  GetX(){return fX;}
   int  GetY(){return fY;}
   int  GetButton(){return fButton;}
   int  GetMode(){return fMode;}
   int  GetType(){return fType;}
   void SetXY(POINT *xy){ fX = xy->x; fY = xy->y;}
   void SetButton(int button){ fButton = button;}

};

//______________________________________________________________________________
class TGWin32GetString : public TWin32Semaphore, public TGWin32Command {

private:

   LONG fX;      // x cursor position (initial and queried)
   LONG fY;      // y cursor postion
   Int_t fBreakKey; // Flag to mark whether user did inter line or cancel input
   const Text_t *fText;  // Text buffer to transfer init value and accept result


public:

  TGWin32GetString(int x, int y, const Text_t *string, int code = IX_REQST) : TGWin32Command(code,ROOT_Input){
     fX = x;
     fY = y;
     fBreakKey = -1;
     fText = string;
  }

   int  GetX(){return fX;}
   int  GetY(){return fY;}
   Int_t GetBreakKey(){ return fBreakKey;}
   const Text_t *GetTextPointer(){ return fText;}
   void IncBreakKey(){ fBreakKey++;}
   void SetXY(POINT *xy){ fX = xy->x; fY = xy->y;}
};

class TContextMenu;
class TMethod;

//______________________________________________________________________________
class TGWin32MenuExecute : public TGWin32Command {


private:

  TObject       *fObject;
  TMethod       *fMethod;
  TContextMenu  *fContextMenu;
  char          *fParams;

public:

  TGWin32MenuExecute(TObject *o, TMethod *m, TContextMenu *menu, char *params, int code = 0) : TGWin32Command(code,0) {
    fObject = o;
    fMethod  = m;
    fContextMenu = menu;
    fParams = 0;
    if (params) {fParams = new char[strlen(params)+1]; strcpy(fParams,params);}
  }
  ~TGWin32MenuExecute() { if (fParams) delete [] fParams; fParams = 0;}
  TContextMenu *GetContextMenu(){return fContextMenu;}
  char *GetMenuParams(){return fParams;}
  TObject *GetMenuObject(){return fObject;}
  TMethod *GetMenuMethod(){return fMethod;}
};

//______________________________________________________________________________
class TGWin32AddMenu: public TGWin32Command {

private:

  HMENU  fMenu;  // Menu handle to set

public:
  TGWin32AddMenu(HMENU menu, int code = IX_SETMENU): TGWin32Command(code,ROOT_Attribute) {fMenu = menu;}
  HMENU GetMenuHandle(){ return fMenu;}
};


class TWin32Dialog;
//______________________________________________________________________________
class TWin32AddDialog : public TGWin32Command {

private:

  TWin32Dialog *fDialog;  // pointer to the dialog to add
  Int_t          fType;    // type of the dialog to add

public:

  TWin32AddDialog(TWin32Dialog *dialog,int code = 0) : TGWin32Command(code,0){fDialog=dialog;}
 ~TWin32AddDialog(){;}
  TWin32Dialog *GetDialog(){return fDialog;}
  Int_t GetDialogType(){ return fType;}
};

//______________________________________________________________________________
class TWin32SendClass : public TGWin32Command {
private:
  void   *fPointer;
  UInt_t  fMessageData[4];

public:

  TWin32SendClass(void *sentclass, int code=kSendClass) : TGWin32Command(code,0)
  {
    fPointer = sentclass;
    SetMsgID(ROOT_CMD);
    fMessageData[0] = 0;
    fMessageData[1] = 0;
    fMessageData[2] = 0;
    fMessageData[3] = 0;
  }

  TWin32SendClass(void *sentclass, UInt_t hwnd, UInt_t message, UInt_t wParam, UInt_t lParam, int code=kSendClass) : TGWin32Command(code,0)
  {
    fPointer = sentclass;
    SetMsgID(ROOT_CMD);
    fMessageData[0] = hwnd;
    fMessageData[1] = message;
    fMessageData[2] = wParam;
    fMessageData[3] = lParam;
  }

  virtual void *GetPointer(){ return fPointer;}
  virtual UInt_t GetData(Int_t i){ return (i >= 0 && i <4) ? fMessageData[i] : 0;}

};

//______________________________________________________________________________
class TWin32SendWaitClass : public TWin32SendClass  {

private:
 TWin32Semaphore fSemaphore;

public:
  TWin32SendWaitClass(void *sentclass, int code=kSendWaitClass) :TWin32SendClass(sentclass,code){ ; }

  TWin32SendWaitClass(void *sentclass, UInt_t hwnd, UInt_t message, UInt_t wParam, UInt_t lParam, int code=kSendWaitClass):
        TWin32SendClass(sentclass, hwnd, message, wParam, lParam, code){ ; }
 void Wait()   { fSemaphore.Wait(); }
 void Release(){ fSemaphore.Release(); }

};

#endif
