// @(#)root/guibuilder:$Id$
// Author: Valeriy Onuchin   12/09/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGuiBldNameFrame
#define ROOT_TGuiBldNameFrame

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGuiBldNameFrame - frame sdisplaying the class name of frame         //
//                    and the name  of frame                            //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif

class TGLabel;
class TGTextEntry;
class TGuiBldEditor;
class TGuiBldEditor;

////////////////////////////////////////////////////////////////////////////////
class TGuiBldNameFrame : public TGCompositeFrame {

private:
   TGLabel         *fLabel;      // label of frame class name 
   TGTextEntry     *fFrameName;  // name of the frame
   TGuiBldEditor   *fEditor;     // pointer to main editor
   TGCompositeFrame *fTitleFrame;      // frame saying that it's "Name Frame"

protected:
   void DoRedraw();

public:
   TGuiBldNameFrame(const TGWindow *p, TGuiBldEditor *editor);
   virtual ~TGuiBldNameFrame() { }

   void ChangeSelected(TGFrame *frame);
   void Reset();
};

#endif
