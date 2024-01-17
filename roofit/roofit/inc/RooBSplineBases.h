/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOBSPLINEBASES
#define ROOBSPLINEBASES

#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooListProxy.h"

#include <sstream>


class RooRealVar;
class RooArgList ;

class RooBSplineBases : public RooAbsReal {
public:

    RooBSplineBases() ;
    RooBSplineBases(const char* name, const char* title, int order, std::vector<double>& tValues,
		    RooAbsReal& t, int nrClose=0);

    RooBSplineBases(const char *name, const char *title);
    RooBSplineBases(const RooBSplineBases&, const char*);

    virtual TObject* clone(const char* newname) const { return new RooBSplineBases(*this, newname); }
    virtual ~RooBSplineBases() ;
    
    int getOrder() const {return _n;}
    int getM() const {return _m;}
    int getNrClose() const {return _nrClose;}
    const RooAbsReal& getT() const { return _t.arg(); }
    Double_t getBasisVal(int n, int i, bool rebuild=true) const;
    const std::vector<double>& getTValues() const {return _tValues;}
    const std::vector<double>& getTAry() const {return _t_ary;}
    const std::vector<std::vector<double>>& getBin() const {return _bin;}    
    
  protected:
    
    void buildTAry() const;
    
    std::vector<double> _tValues;
    int _m;
    RooRealProxy _t;
    int _n;
    int _nrClose;
    mutable std::vector<double> _t_ary;
    mutable std::vector<std::vector<double> > _bin;
    
    Double_t evaluate() const;
    
    ClassDef(RooBSplineBases,1) // Uniform B-Spline
  };

#endif
