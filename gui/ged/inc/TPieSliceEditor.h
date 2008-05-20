// Author: Guido Volpi 05/18/2008

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TPieSliceEditor                                                     //
//                                                                      //
//  Editor for changing pie-chart's slice attributes.                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TPieSliceEditor
#define ROOT_TPieSliceEditor
#ifndef ROOT_TGedFrame
#include "TGedFrame.h"
#endif

class TPieSlice;
class TGTextEntry;
class TGNumberEntry;

class TPieSliceEditor : public TGedFrame {

private:
   TPieSlice *fPieSlice;
   
protected:
   TGTextEntry *fTitle;          // Slice label
   TGNumberEntry *fValue;        // Value of the slice
   TGNumberEntry *fOffset;    // Grafical offset in the radial direction
   
   void ConnectSignals2Slots();
   
public:
   TPieSliceEditor(const TGWindow *p = 0, 
                  Int_t width = 140, Int_t height = 30,
                  UInt_t options = kChildFrame,
                  Pixel_t back = GetDefaultFrameBackground());
   ~TPieSliceEditor();
   
   void SetModel(TObject *);
   
   void DoTitle(const char*);
   void DoValue();
   void DoOffset();
      
   ClassDef(TPieSliceEditor,0)        // piechart' slice editor
};

#endif // ROOT_TPieSliceEditor

