/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>
#include <tcl.h>


main() {
  Win a;
  WildCard_MainLoop();
}

/***************************************************************************
* CintWindowBase
***************************************************************************/
class CintWindowBase {
 public:
  CintWindowBase(CintWindowBase *basewin,char *pathname="."
		 ,char *screen="unix:0.0") {
    Tk_Window *parentwin;
    if(basewin) parentwin = basewin->tkwin;
    else        parentwin = (Tk_Window*)NULL;
    interp = Tcl_CreateInterp();
    tkwin = Tk_CreateMainWindow(interp,screen,"Win","CintWindowBase");
  }
  ~CintWindowBase() {
    Tk_DestroyWindow(tkwin);
  }
 protected:
  Tk_Window tkwin;
  Display *display;
  Tcl_Interp *interp;
  int x,y;
  int size;
  int boarderWidth;
  Tk_3DBorder bgBorder;
  Tk_3DBorder fgBorder;
  int relief;
  GC gc;
  int updatePending;
};

/***************************************************************************
* Square
***************************************************************************/
class Win:  public CintWindowBase {
 public:
  Win(int x=100,int y=100,int xpos=0,int ypos=0
	 ,char *pathname="."
	 ,char *screen="unix:0.0")
    : CintWindowBase((CintWindowBase*)NULL,pathname,screen) {
  }
  ~Win() {}
};

SquareEventProc() {
}
