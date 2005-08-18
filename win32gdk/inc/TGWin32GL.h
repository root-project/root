// @(#)root/win32gdk:$Name:  $:$Id: TGWin32GL.h,v 1.4 2005/08/17 09:10:44 brun Exp $
// Author: Valeriy Onuchin  05/08/04

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGWin32GL
#define ROOT_TGWin32GL


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGWin32GL                                                            //
//                                                                      //
// The TGWin32GL is win32gdk implementation of TVirtualGLImp class.     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TVirtualGL
#include "TVirtualGL.h"
#endif


class TGWin32GL : public TVirtualGLImp {

public:
   TGWin32GL();
   ~TGWin32GL();

   Window_t CreateGLWindow(Window_t wind);
   ULong_t  CreateContext(Window_t wind);
   void     DeleteContext(ULong_t ctx);
   void     MakeCurrent(Window_t wind, ULong_t ctx);
   void     SwapBuffers(Window_t wind);

   ClassDef(TGWin32GL,0);
};

class TGWin32GLManager : public TGLManager {
private:
	class TGWin32GLImpl;
	TGWin32GLImpl *fPimpl;

public:
	TGWin32GLManager();
	~TGWin32GLManager();

	Int_t InitGLWindow(Window_t winId, Bool_t isOffScreen);
	Int_t CreateGLContext(Int_t winInd);
	Int_t OpenGLPixmap(Int_t winInd, Int_t x, Int_t y, UInt_t w, UInt_t h);	

	void ResizeGLPixmap(Int_t pixInd, Int_t x, Int_t y, UInt_t w, UInt_t h);
	void SelectGLPixmap(Int_t pixInd);
	Int_t GetVirtualXInd(Int_t pixInd);
	void MarkForDirectCopy(Int_t pixInd, Bool_t isDirect);

	Bool_t MakeCurrent(Int_t deviceInd);
	void Flush(Int_t deviceInd, Int_t x, Int_t y);
	void DeletePaintDevice(Int_t deviceInd);
	void ExtractViewport(Int_t deviceInd, Int_t *viewport);
   void DrawViewer(TVirtualViewer3D *v);
   TObject *Select(TVirtualViewer3D *v, Int_t x, Int_t y);
private:
	Bool_t CreateGLPixmap(Int_t wid, Int_t x, Int_t y, UInt_t w, 
                         UInt_t h, Int_t prevInd = -1);

	ClassDef(TGWin32GLManager, 0)
};

#endif
