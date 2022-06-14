// Author: Richard Maunder   04/08/05

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TX3DFrame
#define ROOT_TX3DFrame

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TViewerX3D                                                           //
//                                                                      //
// C++ interface to the X3D viewer                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGFrame.h"

class TViewerX3D;

class TX3DFrame : public TGMainFrame
{
private:
   TViewerX3D & fViewer;

   Bool_t ProcessMessage(Longptr_t msg, Longptr_t parm1, Longptr_t parm2);
   void   CloseWindow();

public:
   TX3DFrame(TViewerX3D & viewer, const TGWindow * win, UInt_t width, UInt_t height);
   ~TX3DFrame();
};

#endif
