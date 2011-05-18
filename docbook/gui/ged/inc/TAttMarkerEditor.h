// @(#)root/ged:$Id$
// Author: Ilka  Antcheva 11/05/04

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAttMarkerEditor
#define ROOT_TAttMarkerEditor

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TAttMarkerEditor                                                    //
//                                                                      //
//  Implements GUI for editing marker attributes.                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGedFrame
#include "TGedFrame.h"
#endif

class TGNumberEntry;
class TGColorSelect;
class TGedMarkerSelect;
class TAttMarker;

class TAttMarkerEditor : public TGedFrame {

protected:
   TAttMarker          *fAttMarker;       // marker attribute object
   TGNumberEntry       *fMarkerSize;      // marker size combo box
   TGColorSelect       *fColorSelect;     // marker color
   TGedMarkerSelect    *fMarkerType;      // marker type
   Bool_t              fSizeForText;      // true if "text" draw option uses marker size
   
   virtual void        ConnectSignals2Slots();

public:
   TAttMarkerEditor(const TGWindow *p = 0,
                    Int_t width = 140, Int_t height = 30,
                    UInt_t options = kChildFrame,
                    Pixel_t back = GetDefaultFrameBackground());
   virtual ~TAttMarkerEditor();

   virtual void     SetModel(TObject* obj);
   virtual void     DoMarkerColor(Pixel_t color);
   virtual void     DoMarkerSize();
   virtual void     DoMarkerStyle(Style_t style);

   ClassDef(TAttMarkerEditor,0)  // GUI for editing marker attributes
};

#endif
