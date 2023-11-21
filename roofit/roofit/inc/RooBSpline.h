/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOBSPLINE
#define ROOBSPLINE

#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooListProxy.h"
#include "RooSetProxy.h"

#include "RooBSplineBases.h"

#include "RooObjCacheManager.h"
#include "RooNumIntConfig.h"


#include <sstream>


class RooRealVar;
class RooArgList ;

class RooBSpline : public RooAbsReal {
public:

    RooBSpline() ;
    RooBSpline(const char* name, const char* title,
	       const RooArgList& controlPoints, RooBSplineBases& bases, const RooArgSet& vars);
    //RooBSpline(const char* name, const char* title, int order, vector<double>& tValues,
    //       const RooArgList& controlPoints, RooAbsReal& t, const RooArgSet& vars, int nrClose=0);

    RooBSpline(const char *name, const char *title);
    RooBSpline(const RooBSpline&, const char*);

    virtual TObject* clone(const char* newname) const { return new RooBSpline(*this, newname); }
    virtual ~RooBSpline() ;

/*     Double_t getCurvature() const; */

//    RooBSplinePenalty* getRealPenalty(int k, RooRealVar* obs, RooRealVar* beta, const char* name = "") const;


    void setWeights(const RooArgList& weights);

    Bool_t setBinIntegrator(RooArgSet& allVars) ;
    Int_t getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet,const char* rangeName=0) const ;
    Double_t analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName=0) const ;

    const RooArgList& getControlPoints() const {return _controlPoints;}

    RooBSplineBases* getBases() const {return (RooBSplineBases*)&_bases.arg();}
    int getOrder() const {return _n;}

  protected:

    RooListProxy _controlPoints;
    //RooListProxy _t_ary;
    int _m;
/*     double* _t_ary; //[_m] */
/*     RooRealProxy _t; */
    int _n;
    RooListProxy _weights;
    RooRealProxy _bases;
    RooSetProxy _vars;


    // Cache the integrals   
    class CacheElem : public RooAbsCacheElement {
    public:
      virtual ~CacheElem();
      // Payload
      RooArgList _I ;
      virtual RooArgList containedArgs(Action) ;
    };
    mutable RooObjCacheManager _cacheMgr ; // The cache manager


    Double_t evaluate() const;

    ClassDef(RooBSpline,2) // Uniform B-Spline
   public:

    int getM() const {return _m;} 
    int getN() const { return _n;}
    const RooArgList& getWeights() const { return _weights; }
    const RooArgSet& getVariables() const { return _vars; }

    
};

#endif
