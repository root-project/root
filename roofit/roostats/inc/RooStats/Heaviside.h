/*
 * Project: RooFit
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooStats_Heaviside
#define RooStats_Heaviside

#include "RooAbsReal.h"
#include "RooRealProxy.h"
#include "RooCategoryProxy.h"
#include "RooAbsCategory.h"

namespace RooStats {

   class Heaviside : public RooAbsReal {
   public:
      Heaviside() {} ;
      Heaviside(const char *name, const char *title,
            RooAbsReal& _x,
            RooAbsReal& _c);
      Heaviside(const Heaviside& other, const char* name=nullptr) ;
      TObject* clone(const char* newname=nullptr) const override { return new Heaviside(*this,newname); }

   protected:

      RooRealProxy x ;
      RooRealProxy c ;

      double evaluate() const override ;

   private:

      ClassDefOverride(Heaviside,1);
   };
}

#endif
