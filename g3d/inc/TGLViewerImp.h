// @(#)root/g3d:$Name$:$Id$
// Author: Valery Fine      23/05/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TGLViewerImp
#define ROOT_TGLViewerImp

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLViewerImp                                                         //
//                                                                      //
// Second ABC TGLViewerImp specifies Window system independent openGL   //
// interface. This class uses the GL includes and isn't passed to CINT  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif
#ifndef ROOT_Buttons
#include "Buttons.h"
#endif

class TPadOpenGLView;

class TGLViewerImp  {

protected:

   UInt_t               fDrawList;      // GL list used to redraw the contents
   TPadOpenGLView      *fGLView;        // Pointer to Pad GL View object
   Bool_t               fPaint;         // Allows "Refresh" OpenGL window

public:
   enum {kStatusPopIn, kStatusNoBorders, kStatusOwn, kStatusPopOut};
   TGLViewerImp();
   TGLViewerImp(TPadOpenGLView *padview, const char *title="OpenGL Viewer", UInt_t width=400, UInt_t height=300);
   TGLViewerImp(TPadOpenGLView *padview, const char *title, Int_t x, Int_t y,UInt_t width, UInt_t height);

   virtual ~TGLViewerImp();

   virtual void   CreateContext() { }
   virtual void   CreateStatusBar(Int_t nparts=1);
   virtual void   CreateStatusBar(Int_t *parts, Int_t nparts=1);
   virtual void   DeleteContext() { }
   virtual void   DeleteView();
   virtual void   HandleInput(EEventType button, Int_t x, Int_t y);
   virtual void   MakeCurrent() { };
   virtual void   Paint(Option_t *opt="");

   virtual void   SetStatusText(const char *text, Int_t partidx=0,Int_t stype=0);
   virtual void   ShowStatusBar(Bool_t show = kTRUE);

   virtual void   SwapBuffers() { };

   virtual UInt_t GetDrawList() { return fDrawList; }
   virtual void   SetDrawList(UInt_t list) { fDrawList = list; }

   virtual void   Iconify() { };
   virtual void   Show() { };
   virtual void   Update() { fPaint = kTRUE; }

   // ClassDef(TGLViewerImp,0)  //ROOT OpenGL viewer implementation
};

inline void TGLViewerImp::CreateStatusBar(Int_t) { }
inline void TGLViewerImp::CreateStatusBar(Int_t *, Int_t) { }
inline void TGLViewerImp::SetStatusText(const char *, Int_t, Int_t) { }
inline void TGLViewerImp::ShowStatusBar(Bool_t) { }

#endif
