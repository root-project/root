// @(#)root/ged:$Name:  $:$Id: TLineEditor.h,v 1.1 2006/04/25 08:14:20 antcheva Exp $
// Author: Ilka  Antcheva 24/04/06

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TLineEditor
#define ROOT_TLineEditor

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TLineEditor                                                         //
//                                                                      //
//  Implements GUI for editing line attributes, start/end points.       //                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGButton
#include "TGWidget.h"
#endif
#ifndef ROOT_TGedFrame
#include "TGedFrame.h"
#endif

class TGNumberEntry;
class TGCheckButton;
class TLine;

class TLineEditor : public TGedFrame {

protected:
   TLine           *fLine;         //line object
   TGNumberEntry   *fStartPointX;  //start point x coordinate
   TGNumberEntry   *fStartPointY;  //start point y coordinate
   TGNumberEntry   *fEndPointX;    //end point x coordinate
   TGNumberEntry   *fEndPointY;    //end point y coordinate
   TGCheckButton   *fVertical;     //set the line vertical
   TGCheckButton   *fHorizontal;   //set the line horizontal

   virtual void   ConnectSignals2Slots();

public:
   TLineEditor(const TGWindow *p, Int_t id,
               Int_t width = 140, Int_t height = 30,
               UInt_t options = kChildFrame,
               Pixel_t back = GetDefaultFrameBackground());
   virtual ~TLineEditor();

   virtual void   SetModel(TVirtualPad *pad, TObject *obj, Int_t event);
   virtual void   DoStartPoint();
   virtual void   DoEndPoint();
   virtual void   DoLineVertical();
   virtual void   DoLineHorizontal();

   ClassDef(TLineEditor,0)  // GUI for editing Line attributes
};

#endif
