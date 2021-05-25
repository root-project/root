// @(#)root/gpad:$Id$
// Author: Rene Brun   23/11/96

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSlider
#define ROOT_TSlider

#include "TPad.h"

class TSlider : public TPad {

protected:
   Double_t      fMinimum;      ///< Slider minimum value in [0,1]
   Double_t      fMaximum;      ///< Slider maximum value in [0,1]
   TObject      *fObject;       ///<!Pointer to associated object
   TString       fMethod;       ///< command to be executed when slider is changed

private:
   TSlider(const TSlider &) = delete;
   TSlider &operator=(const TSlider &) = delete;

public:
   TSlider();
   TSlider(const char *name, const char *title, Double_t x1, Double_t y1,Double_t x2 ,Double_t y2, Color_t color=16, Short_t bordersize=2, Short_t bordermode =-1);
   virtual ~TSlider();
   TObject      *GetObject()  const {return fObject;}
   Double_t      GetMinimum() const {return fMinimum;}
   Double_t      GetMaximum() const {return fMaximum;}
   virtual const char *GetMethod() const { return fMethod.Data(); }
   virtual void  Paint(Option_t *option="");
   virtual void  SavePrimitive(std::ostream &out, Option_t *option = "");
   virtual void  SetMethod(const char *method) { fMethod=method; } // *MENU*
   void          SetObject(TObject *obj=0) {fObject=obj;}
   virtual void  SetMinimum(Double_t min=0) {fMinimum=min;}
   virtual void  SetMaximum(Double_t max=1) {fMaximum=max;}
   virtual void  SetRange(Double_t xmin=0, Double_t xmax=1);

   ClassDef(TSlider,1)  //A user interface slider.
};

#endif

