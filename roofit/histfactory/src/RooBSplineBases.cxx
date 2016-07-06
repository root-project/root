// @(#)root/roostats:$Id: RooBSplineBases.cxx 873 2014-02-24 22:16:29Z adye $
// Author: Aaron Armbruster
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//_________________________________________________
/*
BEGIN_HTML
<p>
</p>
END_HTML
*/
//

#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include <math.h>
#include "TMath.h"

#include "RooAbsReal.h"
#include "RooRealVar.h"
#include "RooArgList.h"
#include "RooMsgService.h"
#include "TMath.h"

#include "RooStats/HistFactory/RooBSplineBases.h"

using namespace std ;

ClassImp(RooStats::HistFactory::RooBSplineBases)

using namespace RooStats;
using namespace HistFactory;

//_____________________________________________________________________________
RooBSplineBases::RooBSplineBases()
{
  // Default constructor
//   _t_ary=NULL;
//   _bin=NULL;
}


//_____________________________________________________________________________
RooBSplineBases::RooBSplineBases(const char* name, const char* title, int order, vector<double>& tValues,
				 RooAbsReal& t, int nrClose) :
  RooAbsReal(name, title),
  _tValues(tValues),
  _m(tValues.size()+2*order),
  //_t_ary(NULL),
  _t("t_param", "t_param", this, t),
  _n(order),
  _nrClose(nrClose)
  //_nPlusOne(order+1)//,
  //_bin(NULL)
{
  //cout << "in Ctor" << endl;
  //cout << "t = " << t << endl;
  buildTAry();

//   _bin=new double*[_n+1];
//   for (int n=0;n<_n+1;n++)
//   {
//     _bin[n] = new double[_m];
//     for (int i=0;i<_m;i++)
//     {
//       _bin[n][i] = 0;
//     }
//   }

//   _t_ary = new double[_m];
//   for (int i=_n;i<_m-_n;i++) // add the main knots
//   {
//     _t_ary[i] = tValues[i-_n];
//     //cout << "Adding main point:   " << i << " = " << _t_ary[i] << endl;
//   }

//   double firstDelta=_t_ary[_n+1]-_t_ary[_n]; // extrapolate to the lower non-closed knots
//   for (int i=nrClose;i<_n;i++) 
//   {
//     _t_ary[i] = _t_ary[_n]+firstDelta*(i-_n);
//     //cout << "Adding lower open:   " << i << " = " << _t_ary[i] << endl;
//   }

//   for (int i=0;i<nrClose;i++) // add the lower closed knots
//   {
//     _t_ary[i] = _t_ary[nrClose];
//     //cout << "Adding lower closed: " << i << " = " << _t_ary[i] << endl;
//   }


//   double lastDelta=_t_ary[_m-_n-1]-_t_ary[_m-_n-2]; //extrapolate the upper non-closed knots
//   for (int i=_m-_n;i<_m-nrClose;i++) 
//   {
//     _t_ary[i] = _t_ary[_m-_n-1]+lastDelta*(i-(_m-_n-1));
//     //cout << "Adding upper open:   " << i << " = " << _t_ary[i] << endl;
//   }

//   for (int i=_m-nrClose;i<_m;i++) // add the upper closed knots
//   {
//     _t_ary[i] = _t_ary[_m-nrClose-1];
//     //cout << "Adding upper closed: " << i << " = " << _t_ary[i] << endl;
//   }
//   //cout << endl;

//   for (int i=0;i<_m;i++)
//   {
//     if (fabs(_t_ary[i]) < pow(10., -9)) _t_ary[i] = 0;
//   }

  //cout << "out Ctor" << endl;



  _bin.resize(_n+1);
  for (int i=0;i<_n+1;i++)
  {
    _bin[i].resize(_m);
  }

}

//_____________________________________________________________________________
void RooBSplineBases::buildTAry() const
{
  //delete[] _t_ary;
  _t_ary.resize(_m);
  //cout << "In buildTAry. _m=" << _m << ", _n=" << _n << ", _t_ary.size()=" << _t_ary.size() << endl;
  //_t_ary = new double[_m];
  for (int i=_n;i<_m-_n;i++) // add the main knots
  {
    _t_ary[i] = _tValues[i-_n];
    //cout << "Adding main point:   " << i << " = " << _t_ary[i] << endl;
  }

  double firstDelta=_t_ary[_n+1]-_t_ary[_n]; // extrapolate to the lower non-closed knots
//   cout << "Starting loop" << endl;
//   cout << "_nrClose=" << _nrClose << endl;
  for (int i=_nrClose;i<_n;i++) 
  {
    _t_ary[i] = _t_ary[_n]+firstDelta*(i-_n);
    //cout << "Adding lower open:   " << i << " = " << _t_ary[i] << endl;
  }

  for (int i=0;i<_nrClose;i++) // add the lower closed knots
  {
    _t_ary[i] = _t_ary[_nrClose];
    //cout << "Adding lower closed: " << i << " = " << _t_ary[i] << endl;
  }


  double lastDelta=_t_ary[_m-_n-1]-_t_ary[_m-_n-2]; //extrapolate the upper non-closed knots
  for (int i=_m-_n;i<_m-_nrClose;i++) 
  {
    _t_ary[i] = _t_ary[_m-_n-1]+lastDelta*(i-(_m-_n-1));
    //cout << "Adding upper open:   " << i << " = " << _t_ary[i] << endl;
  }

  for (int i=_m-_nrClose;i<_m;i++) // add the upper closed knots
  {
    _t_ary[i] = _t_ary[_m-_nrClose-1];
    //cout << "Adding upper closed: " << i << " = " << _t_ary[i] << endl;
  }
  //cout << endl;

  for (int i=0;i<_m;i++)
  {
    if (fabs(_t_ary[i]) < pow(10., -9)) _t_ary[i] = 0;
  }
}

//_____________________________________________________________________________
RooBSplineBases::RooBSplineBases(const char* name, const char* title) :
  RooAbsReal(name, title)
{
  // Constructor of flat polynomial function
  //_bin=NULL;
  _bin.resize(_n+1);
  for (int i=0;i<_n+1;i++)
  {
    _bin[i].resize(_m);
  }
}

//_____________________________________________________________________________
RooBSplineBases::RooBSplineBases(const RooBSplineBases& other, const char* name) :
  RooAbsReal(other, name), 
  _tValues(other._tValues),
  _m(other._m),
  _t("t_param", this, other._t),
  _n(other._n),
  _nrClose(other._nrClose),
  //_nPlusOne(other._nPlusOne),
  _t_ary(other._t_ary),
  _bin(other._bin)
{
  // Copy constructor

  buildTAry();
  _bin.resize(_n+1);
  for (int i=0;i<_n+1;i++)
  {
    _bin[i].resize(_m);
  }
//   _t_ary = new double[_m];
//   for (int i=0;i<_m;i++)
//   {
//     _t_ary[i] = other._t_ary[i];
//   }
//   if (other._bin)
//   {
//     _bin=new double*[_n+1];
//     for (int n=0;n<_n+1;n++)
//     {
//       _bin[n] = new double[_m];
//       for (int i=0;i<_m;i++)
//       {
// 	if (other._bin[n])
// 	{
// 	  _bin[n][i] = other._bin[n][i];
// 	}
// 	else
// 	{
// 	  _bin[n][i] = 0;
// 	}
//       }
//     }
//   }
}


//_____________________________________________________________________________
RooBSplineBases::~RooBSplineBases() 
{
  // Destructor
  //delete[] _t_ary;
//   _t_ary=NULL;
//   if (_bin)
//   {
//     for (int i=0;i<_n+1;i++) 
//     {
//       delete _bin[i];
//     _bin[i]=NULL;
//     }
//     delete _bin;
//     _bin=NULL;
//   }
}




//_____________________________________________________________________________
Double_t RooBSplineBases::evaluate() const 
{
//   cout << "In eval, _n=" << _n << ", _m=" << _m << ", _nPlusOne=" << _nPlusOne << endl;
//   cout << "_bin=" << _bin << endl;
//   cout << "_t_ary=" << _t_ary << endl;
  if (!_t_ary.size()) buildTAry();

  // Calculate and return value of spline
  //cout << "In RooBSplineBases::evaluate()" << endl;
  double t = _t;
  if (t < _t_ary[_n] || t > _t_ary[_m-_n-1])
  {
    if (t > _t_ary[_m-_n-1]) t = _t_ary[_m-_n-1];
    if (t < _t_ary[_n]) t = _t_ary[_n];
  }




//build the basis splines

//   if (_bin)
//   {
//     //cout << "Deleting bin: " << _bin << endl;
//     for (int i=0;i<_n+1;i++)
//     {
//       //cout << "Deleting bin[" << i << "]: " << _bin[i] << endl;
//       delete[] _bin[i];
//       _bin[i]=NULL;
// //       for (int j=0;j<_m;j++)
// //       {
// // 	_bin[i][j]=0;
// //       }
//     }
//     delete[] _bin;
//     _bin=NULL;
//   }

//   bool remake=(_bin==NULL);
//   if (remake) _bin = new double*[_n+1];

//   if (!_bin)
//   {
//     _bin=new double*[_n+1];
//     for (int n=0;n<_n+1;n++)
//     {
//       _bin[n] = new double[_m];
//     }
//   }
  //_bin = new double*[_n+1];
  for (int n=0;n<_n+1;n++)
  {
    //cout << "_bin[n] = " << _bin[n] << endl;
    //if (remake) _bin[n] = new double[_m];
    //_bin[n] = new double[_m];
    for (int i=0;i<_m;i++)
    {
      //cout << "Resetting to zero" << endl;
      _bin[n][i] = 0;
    }
    for (int i=0;i<_m-n-1;i++)
    {
      if (n == 0)
      {
	if (t >= _t_ary[i] && t < _t_ary[i+1] && i >= _n && i <= _m-_n-1)
	{
	  //cout << "Setting " << i << " to 1" << endl;
	  _bin[n][i] = 1;
	}
      }
      else
      {
	//cout << "Getting term1" << endl;
	double term1 = 0;
	if (_t_ary[i+n] - _t_ary[i] > 0.000000000001) term1 = _bin[n-1][i] / (_t_ary[i+n] - _t_ary[i]);

	//cout << "Getting term2" << endl;
	double term2 = 0;
	if (_t_ary[i+n+1] - _t_ary[i+1] > 0.0000000000001) term2 = _bin[n-1][i+1] / (_t_ary[i+n+1] - _t_ary[i+1]);

	//cout << "Setting bin" << endl;
	_bin[n][i] = (t - _t_ary[i]) * term1 + (_t_ary[i+n+1] - t) * term2;
      }
      if (_bin[n][i] < 0.000000000001) _bin[n][i] = 0;
    }
  }
  //cout << "Out RooBSplineBases::evaluate()" << endl;
  return t;
}

Double_t RooBSplineBases::getBasisVal(int n, int i, bool rebuild) const
{
  if (rebuild) getVal();
//   if (rebuild || !_bin) getVal();
//   if (!_bin) 
//   {
//     getVal();
//   }
  if (i >= _m-_n-1) return 0.;
  //cout << "Getting basis for n=" << n << ", i=" << i << ", rebuild ? " << rebuild << ", order = " << _n << ", name=" << GetName() << endl;  
  return _bin[n][i];
}



