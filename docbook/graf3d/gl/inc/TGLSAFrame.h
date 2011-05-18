// @(#)root/gl:$Id$
// Author:  Richard Maunder  10/08/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLSAFrame
#define ROOT_TGLSAFrame

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif
#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif

class TGLSAViewer;

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLSAFrame                                                           //
//                                                                      //
// Standalone GL Viewer GUI main frame. Is aggregated in TGLSAViewer -  //
// top level standalone viewer object.                                  //
//////////////////////////////////////////////////////////////////////////

class TGLSAFrame : public TGMainFrame
{
private:
   TGLSAViewer & fViewer;

   // non-copyable class
   TGLSAFrame(const TGLSAFrame &);
   TGLSAFrame & operator = (const TGLSAFrame &);

public:
   TGLSAFrame(TGLSAViewer &viewer);
   TGLSAFrame(const TGWindow *parent, TGLSAViewer &viewer);
   virtual ~TGLSAFrame();

   Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);
   void   CloseWindow();

   ClassDef(TGLSAFrame, 0) // GUI frame for standalone viewer
};

#endif
