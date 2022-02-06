// @(#)root/hist:$Id$
// Authors: Lorenzo Moneta, Arthur Tsang  16/08/17

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2017  ROOT  Team, CERN/PH-SFT                        *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_TF1AbsComposition__
#define ROOT_TF1AbsComposition__

#include "TObject.h"

class TF1AbsComposition : public TObject {

public:
   ~TF1AbsComposition() override {}

   virtual double operator()(const Double_t *x, const Double_t *p) = 0; // for Eval
   virtual void SetRange(Double_t a, Double_t b) = 0;
   virtual void SetParameters(const Double_t *params) = 0;
   virtual void Update() = 0;

   void Copy(TObject &obj) const override = 0;

   ClassDefOverride(TF1AbsComposition, 1);
};

#endif
