// @(#)root/gl:$Name:  $:$Id: TViewerOpenGL.h
// Author:  Timur Pocheptsov  03/08/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TViewerOpenGL
#define ROOT_TViewerOpenGL

#include <utility>

#include <TVirtualViewer3D.h>
#include <TGLViewerImp.h>
#include <TGFrame.h>
#include <TPoint.h>
#include <TList.h>

#ifndef __MAKECINT__
#include <TRootGLU.h>
#ifdef GDK_WIN32
#include <TRootWGL.h>
#else
#include "TRootGLX.h"
#endif //GDK_WIN32
#endif //!__MAKECINT__

namespace TPGL
{
    class TGLWidget;
}

class TViewerOpenGL : public TVirtualViewer3D, public TGMainFrame
{
public:
    TViewerOpenGL(TVirtualPad * pad);
    ~TViewerOpenGL();

    //final overriders for TVirtualViewer3D
    void UpdateScene(Option_t *);
    void CreateScene(Option_t *);

        
    Bool_t HandleContainerButton(Event_t *ev);
    Bool_t HandleContainerConfigure(Event_t *ev);
    Bool_t HandleContainerKey(Event_t *ev);
    Bool_t HandleContainerMotion(Event_t *ev);
    Bool_t HandleContainerExpose(Event_t *ev);
    void HandleInput(int, int, int){}
private:
    class TGCanvas * fCanvasWindow;       
    TPGL::TGLWidget * fCanvasContainer;   
    class TGLayoutHints * fCanvasLayout;   
    TList fGLObjects;
    
    Double_t fRotationX;
    Double_t fRotationY;
    Double_t fRotationZ;
    Double_t fXc;
    Double_t fYc;
    Double_t fZc;

    typedef std::pair<Double_t, Double_t>PDD_t;

    PDD_t fRangeX;
    PDD_t fRangeY;
    PDD_t fRangeZ;
    Double_t fRad;
#ifndef __MAKECINT__
#ifdef GDK_WIN32
    HGLRC fCtx;
    GdkWindow * fGLWin;
#else
    Display * fDpy;
    XVisualInfo * fVisInfo;
    GLXContext fCtx;
    Window fGLWin;    
#endif //GDK_WIN32
#endif //__MAKE_CINT__
    TPoint fLastPos;
    Bool_t fPressed;
    mutable Int_t fDList;

    void CreateViewer();
    void InitGLWindow();
    void DeleteGLWindow();
    void DrawObjects()const;
    void CreateContext();
    void DeleteContext();
    void MakeCurrent()const;
    void SwapBuffers()const;
    void Show();
    void UpdateRange(const class TBuffer3D * buf);
    void BuildGLList()const;
    // final overriders from TGMainFrame
    void CloseWindow();
    //Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);

    //non-copyable class
    TViewerOpenGL(const TViewerOpenGL &);
    TViewerOpenGL & operator = (const TViewerOpenGL &);

    ClassDef(TViewerOpenGL, 0)
};

#endif //!ROOT_TViewerOpenGL
