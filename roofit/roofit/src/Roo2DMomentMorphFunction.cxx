/***************************************************************************** 
 * Project: RooFit                                                           * 
 * author: Max Baak (mbaak@cern.ch)                                          *
 *****************************************************************************/ 

// Written by Max Baak (mbaak@cern.ch)
// 2-dimensional morph function between a list of function-numbers as a function of two input parameter (m1 and m2).
// The matrix mrefpoints assigns a function value to each m1,m2 coordinate.
// Morphing can be set to be linear or non-linear, or mixture of the two.

#include "Riostream.h" 

#include "Roo2DMomentMorphFunction.h" 
#include "RooAbsReal.h" 
#include "RooAbsCategory.h" 
#include <math.h> 
#include "TMath.h" 
#include "TTree.h"


/*
  // Example usage:

  TMatrixD foo(9,3);
  
  foo(0,0) =   0;      // coordinate of variable m1
  foo(0,1) =   0;      // coordinate of variable m2
  foo(0,2) =   0**2;   // function value at (m1,m2) = 0,0
  foo(1,0) = 100;
  foo(1,1) =   0;
  foo(1,2) =   1**2;
  foo(2,0) = 200;
  foo(2,1) =   0;
  foo(2,2) =   2**2;
  
  foo(3,0) = 0;
  foo(3,1) = 150;
  foo(3,2) = 4**2;
  foo(4,0) = 100;
  foo(4,1) = 150;
  foo(4,2) = 5**2;
  foo(5,0) = 200;
  foo(5,1) = 150;
  foo(5,2) = 8**2;
  
  foo(6,0) = 0;
  foo(6,1) = 300;
  foo(6,2) = 8**2;
  foo(7,0) = 100;
  foo(7,1) = 300;
  foo(7,2) = 8.5**2;
  foo(8,0) = 200;
  foo(8,1) = 300;
  foo(8,2) = 9**2;
  
  // need to provide at least 4 coordinates to Roo2DMomentMorphFunction for 2d extrapolation
  
  foo.Print();
  
  RooRealVar m1("m1","m1",50,0,200);
  RooRealVar m2("m2","m2",50,0,300);
  Roo2DMomentMorphFunction bar("bar","bar", m1, m2, foo );
  
  bar.Print();
*/


using namespace std;

ClassImp(Roo2DMomentMorphFunction) 


//_____________________________________________________________________________
Roo2DMomentMorphFunction::Roo2DMomentMorphFunction(const char *name, const char *title, 
                        		       RooAbsReal& _m1, RooAbsReal& _m2,
                                               const TMatrixD& mrefpoints, const Setting& setting, const Bool_t& verbose ) :
  RooAbsReal(name,title), 
  m1("m1","m1",this,_m1),
  m2("m2","m2",this,_m2),
  _setting(setting),
  _verbose(verbose),
  _npoints( mrefpoints.GetNrows() ),
  _mref(mrefpoints)
{
  // cross-check that we have enough reference points
  if ( mrefpoints.GetNrows()<4 ) {
    cerr << "Roo2DMomentMorphFunction::constructor(" << GetName() << ") ERROR: less than four reference points provided." << endl ;
    assert(0);
  }
  // cross-check that we have enough reference points
  if ( mrefpoints.GetNcols()!=3 ) {
    cerr << "RooPolyMorph2D::constructor(" << GetName() << ") ERROR: no reference values provided." << endl ;
    assert(0);
  }

  // recast matrix into more useful form
  _frac.ResizeTo( _npoints );

  initialize();
} 


//_____________________________________________________________________________
Roo2DMomentMorphFunction::Roo2DMomentMorphFunction( const char *name, const char *title,
					       RooAbsReal& _m1, RooAbsReal& _m2,
					       const Int_t& nrows, const Double_t* dm1arr, const Double_t* dm2arr, const Double_t* dvalarr, 
					       const Setting& setting, const Bool_t& verbose ) :
  RooAbsReal(name,title),
  m1("m1","m1",this,_m1),
  m2("m2","m2",this,_m2),
  _setting(setting),
  _verbose( verbose ),
  _npoints( nrows )
{
  // cross-check that we have enough reference points
  if ( nrows<4 ) {
    cerr << "Roo2DMomentMorphFunction::constructor(" << GetName() << ") ERROR: less than four reference points provided." << endl ;
    assert(0);
  }

  // recast tree into more useful form
  _mref.ResizeTo( _npoints,3 );
  _frac.ResizeTo( _npoints );

  for (int i=0; i<_npoints; ++i) {
    _mref(i,0) = dm1arr[i] ;
    _mref(i,1) = dm2arr[i]  ;
    _mref(i,2) = dvalarr[i] ;
  }

  initialize();
}


//_____________________________________________________________________________
Roo2DMomentMorphFunction::Roo2DMomentMorphFunction(const Roo2DMomentMorphFunction& other, const char* name) :  
  RooAbsReal(other,name), 
  m1("m1",this,other.m1),
  m2("m2",this,other.m2),
  _setting(other._setting),
  _verbose(other._verbose),
  _npoints(other._npoints),
  _mref(other._mref),
  _frac(other._frac)
{ 
  initialize();
} 


//_____________________________________________________________________________
Roo2DMomentMorphFunction::~Roo2DMomentMorphFunction()
{
}


//_____________________________________________________________________________
void 
Roo2DMomentMorphFunction::initialize() 
{
  Double_t xmin(1e300), xmax(-1e300), ymin(1e300), ymax(-1e300);

  // transformation matrix for non-linear extrapolation, needed in evaluate()
  for (Int_t k=0; k<_npoints; ++k) {
    if (_mref(k,0)<xmin) { xmin=_mref(k,0); _ixmin=k; }
    if (_mref(k,0)>xmax) { xmax=_mref(k,0); _ixmax=k; }
    if (_mref(k,1)<ymin) { ymin=_mref(k,1); _iymin=k; }
    if (_mref(k,1)>ymax) { ymax=_mref(k,1); _iymax=k; }
  }

  // resize
  _MSqr.ResizeTo(4,4);
  _squareVec.ResizeTo(4,2);
}


//_____________________________________________________________________________
Double_t 
Roo2DMomentMorphFunction::evaluate() const 
{ 
  if (_verbose) { cout << "Now in evaluate." << endl; }
  if (_verbose)  { cout << "x = " << m1 << " ; y = " << m2 << endl; }

  calculateFractions(_verbose); // verbose turned off
  
  Double_t ret = 0.0;
  for (Int_t i=0; i<4; ++i) { ret += ( _mref(_squareIdx[i],2) * _frac[_squareIdx[i]] ) ; }

  if (_verbose) { cout << "End of evaluate : " << ret << endl; }

  //return (ret>0 ? ret : 0) ;
  return ret;
} 


//_____________________________________________________________________________
void 
Roo2DMomentMorphFunction::calculateFractions(Bool_t verbose) const
{
  double sumfrac(0.);

  if (_setting == Linear || _setting == LinearPosFractions) {
    // reset all fractions
    for (Int_t i=0; i<_npoints; ++i) { _frac[i]=0.0; }

    // this sets _squareVec and _squareIdx quantities and MSqr
    (void) findSquare(m1,m2); 
  
    std::vector<double> deltavec(4,1.0);
    deltavec[1] = m1-_squareVec(0,0) ;
    deltavec[2] = m2-_squareVec(0,1) ;
    deltavec[3] = deltavec[1]*deltavec[2] ;

    for (Int_t i=0; i<4; ++i) {
      double ffrac=0.;
      for (Int_t j=0; j<4; ++j) { ffrac += _MSqr(j,i) * deltavec[j]; }

      // set fractions 
      _frac[_squareIdx[i]] = ffrac;
      if (ffrac>0) sumfrac += ffrac;

      if (verbose) { 
        cout << _squareIdx[i] << " " << ffrac << " " << _squareVec(i,0) << " " << _squareVec(i,1) << endl; 
      }
    }  
  }

  if (_setting == LinearPosFractions) {
    for (Int_t i=0; i<4; ++i) {
      if (_frac[_squareIdx[i]]<0) { _frac[_squareIdx[i]] = 0; }
      _frac[_squareIdx[i]] *= (1.0/sumfrac) ;
    } 
  }
}


//_____________________________________________________________________________
Bool_t 
Roo2DMomentMorphFunction::findSquare(const double& x, const double& y) const
{
  bool squareFound(false);

  std::vector< std::pair<int,double> > idvec;

  Double_t xspacing = (_mref(_ixmax,0)-_mref(_ixmin,0)) / TMath::Sqrt(_npoints) ;
  Double_t yspacing = (_mref(_iymax,1)-_mref(_iymin,1)) / TMath::Sqrt(_npoints) ; 

  Double_t dx(0), dy(0), delta(0);
  for (Int_t k=0; k<_npoints; ++k) {
    dx = (x-_mref(k,0))/xspacing ;
    dy = (y-_mref(k,1))/yspacing ;
    delta = TMath::Sqrt(dx*dx+dy*dy) ;
    idvec.push_back( std::pair<int,double>(k,delta) );
  }

  // order points closest to (x,y)
  sort(idvec.begin(),idvec.end(),SorterL2H());
  std::vector< std::pair<int,double> >::iterator itr = idvec.begin();
  std::vector<int> cidx;
  for(; itr!=idvec.end(); ++itr) { 
    cidx.push_back(itr->first);
  }

  // 1. point falls outside available ref points: pick three square points.
  //    fall-back option
  _squareVec(0,0) = _mref(cidx[0],0);
  _squareVec(0,1) = _mref(cidx[0],1);
  _squareIdx[0] = cidx[0];
  _squareVec(1,0) = _mref(cidx[1],0);
  _squareVec(1,1) = _mref(cidx[1],1);
  _squareIdx[1] = cidx[1];
  _squareVec(2,0) = _mref(cidx[2],0);
  _squareVec(2,1) = _mref(cidx[2],1);
  _squareIdx[2] = cidx[2];
  _squareVec(3,0) = _mref(cidx[3],0);
  _squareVec(3,1) = _mref(cidx[3],1);
  _squareIdx[3] = cidx[3];

  // 2. try to find square enclosing square
  if ( x>_mref(_ixmin,0) &&
       x<_mref(_ixmax,0) &&
       y>_mref(_iymin,1) &&
       y<_mref(_iymax,1) )
  {
    for (unsigned int i=0; i<cidx.size() && !squareFound; ++i) 
      for (unsigned int j=i+1; j<cidx.size() && !squareFound; ++j)
        for (unsigned int k=j+1; k<cidx.size() && !squareFound; ++k) 
          for (unsigned int l=k+1; l<cidx.size() && !squareFound; ++l) { 
            if ( isAcceptableSquare(_mref(cidx[i],0),_mref(cidx[i],1),_mref(cidx[j],0),_mref(cidx[j],1),_mref(cidx[k],0),_mref(cidx[k],1),_mref(cidx[l],0),_mref(cidx[l],1)) ) {
              if (  pointInSquare(x,y,_mref(cidx[i],0),_mref(cidx[i],1),_mref(cidx[j],0),_mref(cidx[j],1),_mref(cidx[k],0),_mref(cidx[k],1),_mref(cidx[l],0),_mref(cidx[l],1)) ) {
                _squareVec(0,0) = _mref(cidx[i],0);  
                _squareVec(0,1) = _mref(cidx[i],1);  
                _squareIdx[0] = cidx[i];
                _squareVec(1,0) = _mref(cidx[j],0);  
                _squareVec(1,1) = _mref(cidx[j],1);  
                _squareIdx[1] = cidx[j];
                _squareVec(2,0) = _mref(cidx[k],0);  
                _squareVec(2,1) = _mref(cidx[k],1);  
                _squareIdx[2] = cidx[k];
                _squareVec(3,0) = _mref(cidx[l],0);  
                _squareVec(3,1) = _mref(cidx[l],1); 
                _squareIdx[3] = cidx[l];
                squareFound=true;
              }
            }
          }
  }

  // construct transformation matrix for linear extrapolation
  TMatrixD M(4,4);
  for (Int_t k=0; k<4; ++k) {
    dx = _squareVec(k,0) - _squareVec(0,0) ;
    dy = _squareVec(k,1) - _squareVec(0,1) ;
    M(k,0) = 1.0 ;
    M(k,1) = dx ;
    M(k,2) = dy ;
    M(k,3) = dx*dy ;
  }

  _MSqr = M.Invert();

  return squareFound;
}


//_____________________________________________________________________________
Bool_t Roo2DMomentMorphFunction::onSameSide(const double& p1x, const double& p1y, const double& p2x, const double& p2y, const double& ax, const double& ay, const double& bx, const double& by) const
{   
  // p1 and p2 on same side of line b-a ?
  Double_t cp1 = myCrossProduct(bx-ax, by-ay, p1x-ax, p1y-ay);
  Double_t cp2 = myCrossProduct(bx-ax, by-ay, p2x-ax, p2y-ay);
  if (cp1*cp2 >= 0) return true;
  else return false;
}


//_____________________________________________________________________________
Bool_t 
Roo2DMomentMorphFunction::pointInSquare(const double& px, const double& py, const double& ax, const double& ay, const double& bx, const double& by, const double& cx, const double& cy, const double& dx, const double& dy) const
{   
  bool insquare(false);

  int ntri(0);
  if (ntri<2) ntri += static_cast<int>( pointInTriangle(px,py,ax,ay,bx,by,cx,cy) );
  if (ntri<2) ntri += static_cast<int>( pointInTriangle(px,py,ax,ay,bx,by,dx,dy) );
  if (ntri<2) ntri += static_cast<int>( pointInTriangle(px,py,ax,ay,cx,cy,dx,dy) );
  if (ntri<2) ntri += static_cast<int>( pointInTriangle(px,py,bx,by,cx,cy,dx,dy) );

  if (ntri>=2) insquare=true;
  else insquare=false;

  return insquare;
}


//_____________________________________________________________________________
Bool_t 
Roo2DMomentMorphFunction::pointInTriangle(const double& px, const double& py, const double& ax, const double& ay, const double& bx, const double& by, const double& cx, const double& cy) const
{
  if (onSameSide(px,py,ax,ay,bx,by,cx,cy) && onSameSide(px,py,bx,by,ax,ay,cx,cy) && onSameSide(px,py,cx,cy,ax,ay,bx,by)) return true;
  else return false;
}


//_____________________________________________________________________________
Double_t 
Roo2DMomentMorphFunction::myCrossProduct(const double& ax, const double& ay, const double& bx, const double& by) const
{
  return ( ax*by - bx*ay );
}


//_____________________________________________________________________________
Bool_t 
Roo2DMomentMorphFunction::isAcceptableSquare(const double& ax, const double& ay, const double& bx, const double& by, const double& cx, const double& cy, const double& dx, const double& dy) const
{
  // reject kinked shapes
  if ( pointInTriangle(dx,dy,ax,ax,bx,by,cx,cy) ||
       pointInTriangle(cx,cy,ax,ay,bx,by,dx,dy) ||
       pointInTriangle(bx,by,ax,ay,cx,cy,dx,dy) ||
       pointInTriangle(ax,ay,bx,by,cx,cy,dx,dy) ) return false;
  else return true;
}


void
Roo2DMomentMorphFunction::Summary() const
{
  for( Int_t i=0; i<_npoints; i++ ){
    cout << this << " " << i << " " << _mref(i,0) << " " << _mref(i,1) << " " << _mref(i,2) << endl;
  }
}

