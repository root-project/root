// @(#)root/ged:$Id$
// Author: Ilka  Antcheva 10/05/04

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAttFillEditor
#define ROOT_TAttFillEditor

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TAttFillEditor                                                      //
//                                                                      //
//  Implements GUI for editing fill attributes.                         //                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGedFrame
#include "TGedFrame.h"
#endif

class TGColorSelect;
class TGedPatternSelect;
class TAttFill;


class TAttFillEditor : public TGedFrame {

protected:
   TAttFill            *fAttFill;          // fill attribute object
   TGColorSelect       *fColorSelect;      // fill color widget
   TGedPatternSelect   *fPatternSelect;    // fill pattern widget

   virtual void ConnectSignals2Slots();
 
public:
   TAttFillEditor(const TGWindow *p = 0, 
                  Int_t width = 140, Int_t height = 30,
                  UInt_t options = kChildFrame,
                  Pixel_t back = GetDefaultFrameBackground());
   virtual ~TAttFillEditor(); 

   virtual void   SetModel(TObject* obj);
   virtual void   DoFillColor(Pixel_t color);
   virtual void   DoFillPattern(Style_t color);
           
   ClassDef(TAttFillEditor,0)  //GUI for editing fill attributes
};

#endif
