/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: RooNovosibirsk.rdl,v 1.1 2001/02/06 22:01:53 htanaka Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   29-Feb-2000 DK Created initial version
 *
 * Copyright (C) 2000 Stanford University
* 20 Jan 2001: Hirohisa A. Tanaka Novosibirsk PDF for RooFitTools
 *****************************************************************************/

#ifndef ROO_NOVOSIBIRSK
#define ROO_NOVOSIBIRSK

#include "RooFitCore/RooAbsPdf.hh"
#include "RooFitCore/RooRealProxy.hh"

class RooRealVar;
class RooAbsReal;

class RooNovosibirsk : public RooAbsPdf {
public:
  // Your constructor needs a name and title and then a list of the
  // dependent variables and parameters used by this PDF. Use an
  // underscore in the variable names to distinguish them from your
  // own local versions.
  RooNovosibirsk(const char *name, const char *title,
		 RooRealVar& _x,     RooRealVar& _peak,
		 RooRealVar& _width, RooRealVar& _tail);

  RooNovosibirsk(const RooNovosibirsk& other,const char* name=0) ;	

  virtual TObject* clone(const char* newname) const { return new RooNovosibirsk(*this,newname);	}

  // An empty constructor is usually ok
  inline virtual ~RooNovosibirsk() { }

protected:
  RooRealProxy x;
  RooRealProxy width;
  RooRealProxy peak;
  RooRealProxy tail;
  Double_t evaluate() const;

private:
  ClassDef(RooNovosibirsk,0) // Novosibirsk PDF
};

#endif
