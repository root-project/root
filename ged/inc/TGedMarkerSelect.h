// @(#)root/ged:$Name:  $:$Id: TGedMarkerSelect.h,v 1.1 2004/02/18 20:13:42 brun Exp $
// Author: Marek Biskup, Ilka Antcheva   24/07/03
// **** It needs more fixes ****
// 
/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGedMarkerSelect
#define ROOT_TGedMarkerSelect

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGedMarkerPopup and TGedMarkerSelect.                                //
//                                                                      //
// The TGedMarkerPopup is a popup containing all diferent styles of     //
// markers.                                                             //
//                                                                      //
// The TGedMarkerSelect widget is a button with marker drawn inside     //
// and a little down arrow. When clicked the TGMarkerPopup.             //
//                                                                      //
// Selecting a marker in this widget will generate the event:           //
// kC_MARKERSEL, kMAR_SELCHANGED, widget id, pixel.                     //
//                                                                      //
// and the signal:                                                      //
// MarkerSelected(Style_t marker)                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGButton
#include "TGButton.h"
#endif
#ifndef ROOT_TGToolTip
#include "TGToolTip.h"
#endif
#ifndef ROOT_TGedPatternSelect
#include "TGedPatternSelect.h"
#endif


class TGedMarkerPopup : public TGedPopup {

protected:
   Style_t  fCurrentStyle;

public:
   TGedMarkerPopup(const TGWindow *p, const TGWindow *m, Style_t markerStyle);
   virtual ~TGedMarkerPopup();

   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);

   ClassDef(TGedMarkerPopup,0)  //marker select popup
};


class TGedMarkerSelect : public TGedSelect {

protected:
   Style_t          fMarkerStyle;
   const TGPicture *fPicture;

   virtual void     DoRedraw();

public:
   TGedMarkerSelect(const TGWindow *p, Style_t markerStyle, Int_t id);
   virtual ~TGedMarkerSelect() { if(fPicture) gClient->FreePicture(fPicture);}

   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);
   void           SetMarkerStyle(Style_t pattern);
   Pixel_t        GetMarkerStyle() const { return fMarkerStyle; }
   virtual void   SavePrimitive(ofstream &out, Option_t *);

   virtual TGDimension GetDefaultSize() const { return TGDimension(38, 21); }

   virtual void MarkerSelected() { Emit("MarkerSelected(Style_t)", GetMarkerStyle()); }  // *SIGNAL*
  
   ClassDef(TGedMarkerSelect,0)  // Marker selection button
};

#endif
