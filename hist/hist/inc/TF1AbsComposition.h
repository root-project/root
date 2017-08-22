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

#include <iostream>
#include <memory>

class TF1AbsComposition : public TObject {

public:
   virtual ~TF1AbsComposition() {
      std::cout << "Calling TF1AbsComposition destructor" << std::endl;
   }

   virtual double operator()(const Double_t *x, const Double_t *p) = 0; // for Eval
   virtual void   SetRange(Double_t a, Double_t b) = 0;
   virtual void   SetParameters(const Double_t *params) = 0;
   virtual void   Update() = 0;

   virtual void   Copy(TObject &obj) const = 0;
};

#endif
