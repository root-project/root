// @(#)root/guibuilder:$Id$
// Author: Valeriy Onuchin, Lucie Flekova   12/09/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGuiBldGeometryFrame
#define ROOT_TGuiBldGeometryFrame


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGuiBldGeometryFrame                                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif


class TGuiBldEditor;
class TGNumberEntry;
class TGFrame;
class TRootGuiBuilder;
class TGuiBldDragManager;


//////////////////////////////////////////////////////////////////////////
class TGuiBldGeometryFrame : public TGVerticalFrame {

friend class TGuiBldDragManager;

private:
   TGuiBldEditor        *fEditor;
   TRootGuiBuilder      *fBuilder;
   TGuiBldDragManager   *fDragManager;
   TGNumberEntry        *fNEWidth;
   TGNumberEntry        *fNEHeight;
   TGFrame              *fSelected;

public:
   TGuiBldGeometryFrame(const TGWindow *p, TGuiBldEditor *editor);
   virtual ~TGuiBldGeometryFrame() { }

   void ResizeSelected();
   void ChangeSelected(TGFrame *frame);

   ClassDef(TGuiBldGeometryFrame, 0) // frame geometry editor
};

#endif


