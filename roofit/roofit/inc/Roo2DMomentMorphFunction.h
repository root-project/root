/*****************************************************************************
 * Project: RooFit                                                           *
 * author: Max Baak (mbaak@cern.ch)                                          *
 *****************************************************************************/

// Written by Max Baak (mbaak@cern.ch)
// 2-dimensional morph function between a list of input functions (varlist) as a function of one input parameter (m).
// The vector mrefpoints assigns an m-number to each function in the function list.
// For example: varlist can contain MC histograms (or single numbers) of a reconstructed mass, for certain 
// true Higgs masses indicated in mrefpoints. the input parameter m is the true (continous) Higgs mass.
// Morphing can be set to be linear or non-linear, or mixture of the two.

#ifndef ROO2DMOMENTMORPHFUNCTION
#define ROO2DMOMENTMORPHFUNCTION

#include "RooAbsReal.h"
#include "RooRealProxy.h"
#include "RooCategoryProxy.h"
#include "RooAbsReal.h"
#include "RooAbsCategory.h"

#include "TMatrix.h"
#include "TVectorD.h"

#include <vector>
#include <string>

class Roo2DMomentMorphFunction : public RooAbsReal {

 public:

  enum Setting { Linear, LinearPosFractions } ;

  Roo2DMomentMorphFunction() {} ; 

  Roo2DMomentMorphFunction( const char *name, const char *title,
	                  RooAbsReal& _m1, RooAbsReal& _m2, 
                          const TMatrixD& mrefpoints, const Setting& setting = Linear, const Bool_t& verbose=false ) ;

  Roo2DMomentMorphFunction( const char *name, const char *title,
                          RooAbsReal& _m1, RooAbsReal& _m2,
                          const Int_t& nrows, const Double_t* dm1arr, const Double_t* dm2arr, const Double_t* dvalarr, 
			  const Setting& setting = Linear, const Bool_t& verbose=false ) ;

  Roo2DMomentMorphFunction( const Roo2DMomentMorphFunction& other, const char* name=0 );

  virtual TObject* clone( const char* newname ) const { return new Roo2DMomentMorphFunction(*this,newname); }
  virtual ~Roo2DMomentMorphFunction() ;

  void setMode( const Setting& setting ) { _setting=setting; }

  void Summary() const;

 protected:

  Double_t evaluate() const ;

  void     initialize();
  void     calculateFractions(Bool_t verbose=kTRUE) const;

  RooRealProxy m1 ;
  RooRealProxy m2 ;

  Setting _setting;
  Bool_t _verbose;
  Int_t _ixmin, _ixmax, _iymin, _iymax;
  Int_t	_npoints;

  mutable TMatrixD _mref;
  mutable TMatrixD _MSqr;
  mutable TVectorD _frac;

  mutable TMatrixD _squareVec;
  mutable int _squareIdx[4];

 private:

  Bool_t   findSquare(const double& x, const double& y) const;
  Bool_t   onSameSide(const double& p1x, const double& p1y, const double& p2x, const double& p2y, const double& ax, const double& ay, const double& bx, const double& by) const ;
  Bool_t   pointInSquare(const double& px, const double& py, const double& ax, const double& ay, const double& bx, const double& by, const double& cx, const double& cy, const double& dx, const double& dy) const;
  Bool_t   pointInTriangle(const double& px, const double& py, const double& ax, const double& ay, const double& bx, const double& by, const double& cx, const double& cy) const ;
  Double_t myCrossProduct(const double& ax, const double& ay, const double& bx, const double& by) const ;
  Bool_t   isAcceptableSquare(const double& ax, const double& ay, const double& bx, const double& by, const double& cx, const double& cy, const double& dx, const double& dy) const;

  struct SorterL2H {
    bool operator() (const std::pair<int,double>& p1, const std::pair<int,double>& p2) {
      return (p1.second<p2.second);
    }
  };

 private:
  ClassDef(Roo2DMomentMorphFunction,1) 

};
 
#endif

