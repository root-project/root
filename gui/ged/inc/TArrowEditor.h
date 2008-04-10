// @(#)root/ged:$Id$
// Author: Ilka  Antcheva 20/10/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TArrowEditor
#define ROOT_TArrowEditor

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TArrowEditor                                                        //
//                                                                      //
//  Implements GUI for editing arrow attributes: shape, size, angle.    //                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGedFrame
#include "TGedFrame.h"
#endif

class TGComboBox;
class TGNumberEntry;
class TArrow;

class TArrowEditor : public TGedFrame {

protected:
   TArrow               *fArrow;            // arrow object
   TGComboBox           *fOptionCombo;      // arrow shapes combo box
   TGNumberEntry        *fAngleEntry;       // opening angle entry
   TGNumberEntry        *fSizeEntry;        // size entry

   virtual void   ConnectSignals2Slots();
   TGComboBox    *BuildOptionComboBox(TGFrame* parent, Int_t id);
   Int_t          GetShapeEntry(Option_t *opt);

public:
   TArrowEditor(const TGWindow *p = 0, 
                Int_t width = 140, Int_t height = 30,
                UInt_t options = kChildFrame,
                Pixel_t back = GetDefaultFrameBackground());
   virtual ~TArrowEditor();

   virtual void   SetModel(TObject* obj);
   virtual void   DoAngle();
   virtual void   DoOption(Int_t id);
   virtual void   DoSize();

   ClassDef(TArrowEditor,0)  // GUI for editing arrow attributes
};

#endif
