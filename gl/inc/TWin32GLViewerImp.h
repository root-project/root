// @(#)root/gl:$Name$:$Id$
// Author: Valery Fine(fine@vxcern.cern.ch)   29/12/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TWin32GLViewerImp
#define ROOT_TWin32GLViewerImp

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TWin32GLViewer                                                       //
//                                                                      //
// This class creates a toplevel window with menubar, and an openGL     //
// drawing area and context.                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef GL_RootWGL
#include "TRootWGL.h"
#endif

#ifndef GL_RootGLU
#include "TRootGLU.h"
#endif

#ifndef ROOT_TGLViewerImp
#include "TGLViewerImp.h"
#endif

#ifndef ROOT_TgWin32WindowsObject
#include "TGWin32WindowsObject.h"
#endif

#ifndef ROOT_TWin32HookViaThread
#include "TWin32HookViaThread.h"
#endif

#ifndef ROOT_TContextMenu
#include "TContextMenu.h"
#endif

class TWin32GLViewerImp : protected TWin32HookViaThread,
                          public TGWin32WindowsObject, public TGLViewerImp {

private:
  typedef enum  {kCreateContext=1, kDeleteContext, kMakeCurrent, kSwapBuffers
                } EGLCommand;
   HGLRC            fhOpenGLRC;         //Handle of theOpenGL rendering context
   PIXELFORMATDESCRIPTOR fPixelFormat;  // Pixel format descriptor
   Int_t            fPixelFormatIndex;  // The selected pixel format index
   TGWin32Object   *fWin32Object;       // The Win32 object to display OpenGL object
   DWORD            fThreadId;          // Id of the working thread for the current GL context

protected:
   TWin32GLViewerImp();                 // used by Dictionary()
   void ExecThreadCB(TWin32SendClass *command);
   virtual void  CreateContextCB();     // Create OpenGL rendering context for wid TVirtualX Window
   void CreateViewer( const char *title="", Int_t x=0, Int_t y=0, UInt_t width=400, UInt_t height=300);


   // CallBack methods

   virtual void   DeleteContextCB();                // Delete current OpenGL RC
   virtual void   MakeCurrentCB(Bool_t flag=kTRUE); // Make this object current
   virtual Bool_t SetupPixelFormat();
   virtual void   SwapBuffersCB();                  // Swap the OpenGL double buffer
   virtual void   ShowHelp();                       // Show thenhelp message

   void MakeMenu();                                // Make the viewer menu
   void MakeStatusBar();
   void MakeToolBar();


public:
   TWin32GLViewerImp(TPadOpenGLView *padview,const char *title, UInt_t width = 600, UInt_t height = 600);
   TWin32GLViewerImp(TPadOpenGLView *padview,const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height);

   virtual ~TWin32GLViewerImp();

   virtual void  CreateContext();                // Create OpenGL rendering context for wid TVirtualX Window
   virtual void  CreateStatusBar(Int_t nparts=1);
   virtual void  CreateStatusBar(Int_t *parts, Int_t nparts=1);
   virtual void  DeleteContext();                // Delete current OpenGL RC
   HGLRC GetRC(){ return fhOpenGLRC;}            // Returns GL Rendering context
   virtual void  MakeCurrent(Bool_t flag=kTRUE); // Make this object current

   virtual void  SetStatusText(const Text_t *text, Int_t partidx=0, Int_t stype=0);
   virtual void  ShowStatusBar(Bool_t show = kTRUE);

   virtual void  SwapBuffers();
   virtual void  Update();                       // Paint the scene on the screen

   const char  *ClassName() const { return "TWin32GLViewerImp"; }

   virtual void         Iconify();
   virtual void         Show();


   LRESULT APIENTRY OnChar       (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
   LRESULT APIENTRY OnClose      (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
   LRESULT APIENTRY OnCreate     (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
   LRESULT APIENTRY OnKeyDown    (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
   LRESULT APIENTRY OnMouseButton(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
   LRESULT APIENTRY OnPaint      (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
   LRESULT APIENTRY OnSysCommand (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
   LRESULT APIENTRY OnSize       (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

   void RootExec(const char *cmd);
   void NewCB(TWin32GLViewerImp *obj, TVirtualMenuItem *item);
   void SaveCB(TWin32GLViewerImp *obj, TVirtualMenuItem *item);
   void SaveAsCB(TWin32GLViewerImp *obj, TVirtualMenuItem *item);
   void PrintCB(TWin32GLViewerImp *obj, TVirtualMenuItem *item);
   void CloseCB(TWin32GLViewerImp *obj, TVirtualMenuItem *item);

//   ClassDef(TWin32GLViewer,0)  //Win32 version of the ROOT GLViewer

};

#endif
