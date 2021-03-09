// @(#)root/ged:$Id$
// Author: Marek Biskup, Ilka Antcheva   24/07/03

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGedMarkerSelect
#define ROOT_TGedMarkerSelect


#include "TGedPatternSelect.h"

class TGToolTip;
class TGPicture;


class TGedMarkerPopup : public TGedPopup {

protected:
   Style_t  fCurrentStyle;     //currently selected style

public:
   TGedMarkerPopup(const TGWindow *p, const TGWindow *m, Style_t markerStyle);
   virtual ~TGedMarkerPopup();

   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);

   ClassDef(TGedMarkerPopup,0)  //marker select popup
};


class TGedMarkerSelect : public TGedSelect {

protected:
   Style_t          fMarkerStyle;   ///< marker style
   const TGPicture *fPicture;       ///< image used for popup window

   virtual void     DoRedraw();

public:
   TGedMarkerSelect(const TGWindow *p, Style_t markerStyle, Int_t id);
   virtual ~TGedMarkerSelect() { if(fPicture) gClient->FreePicture(fPicture);}

   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);
   Style_t        GetMarkerStyle() const { return fMarkerStyle; }
   void           SetMarkerStyle(Style_t pattern);
   virtual void   MarkerSelected(Style_t marker = 0) { Emit("MarkerSelected(Style_t)", marker ? marker : GetMarkerStyle()); }  // *SIGNAL*
   virtual void   SavePrimitive(std::ostream &out, Option_t * = "");
   virtual TGDimension GetDefaultSize() const { return TGDimension(38, 21); }

   ClassDef(TGedMarkerSelect,0)  // Marker selection button
};

#endif
