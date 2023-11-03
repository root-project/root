/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//_________________________________________________
//

#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include <math.h>
#include <memory>
#include "TMath.h"

#include "RooAbsReal.h"
#include "RooRealVar.h"
#include "RooArgList.h"
#include "RooMsgService.h"
#include "TMath.h"

#include "RooBSpline.h"

using namespace std ;

ClassImp(RooBSpline)

using namespace RooStats;
using namespace HistFactory;

//_____________________________________________________________________________
RooBSpline::RooBSpline()
{
  // Default constructor
//   _t_ary=NULL;
}


//_____________________________________________________________________________
RooBSpline::RooBSpline(const char* name, const char* title,
		       const RooArgList& controlPoints, RooBSplineBases& bases, const RooArgSet& vars) :
  RooAbsReal(name, title),
  _controlPoints("controlPoints","List of control points",this),
  _m(bases.getTValues().size()+2*bases.getOrder()),
//   _t_ary(NULL),
//   _t("t_param", "t_param", this, *t),
  _n(bases.getOrder()),
  _weights("weights","List of weights",this),
  _bases("bases","Basis polynomials",this,bases),
  _vars("observables","List of observables",this),
  _cacheMgr(this,10)
{
  //cout << "in Ctor" << endl;
  //cout << "t = " << t << endl;

  if (_m-2*_n != controlPoints.getSize())
  {
    cout << "ERROR::Nr t values (" << _m-2*_n << ") != nr control points (" << controlPoints.getSize() << ")" << endl;
  }

  //bool even = fabs(_n/2-_n/2.) < 0.0000000001;
  bool first=1;
  TIterator* pointIter = controlPoints.createIterator() ;
  RooAbsArg* point ;
  RooAbsArg* lastPoint=NULL ;
  while((point = (RooAbsArg*)pointIter->Next())) {
    if (!dynamic_cast<RooAbsReal*>(point)) {
      coutE(InputArguments) << "RooBSpline::ctor(" << GetName() << ") ERROR: control point " << point->GetName() 
			    << " is not of type RooAbsReal" << endl ;
      assert(0) ;
    }
    //RooAbsReal* pointReal = (RooAbsReal*)point;
    //cout << "Adding control point " << point->GetName() << ", has val " << pointReal->getVal() << endl;
    _controlPoints.add(*point) ;
    if (first) 
    {
      for (int i=0;i<(_n)/2;i++)
      {
	_controlPoints.add(*point) ;
      }
    }
    first=false;
    lastPoint=point;
  }
  for (int i=0;i<(_n)/2;i++) _controlPoints.add(*lastPoint);
  delete pointIter ;


  TIterator* varItr = vars.createIterator();
  RooAbsArg* arg;
  while ((arg=(RooAbsArg*)varItr->Next())) {
    //cout << "======== Adding "<<arg->GetName()<<" to list of _vars of "<<name<<"." << endl;
    _vars.add(*arg);
  }
//   cout << "all _vars: " << endl;
//   _vars.Print("V");
  delete varItr;
  //cout << "out Ctor" << endl;
}

//_____________________________________________________________________________
RooBSpline::RooBSpline(const char* name, const char* title) :
  RooAbsReal(name, title),
  _controlPoints("controlPoints","List of coefficients",this),
  _cacheMgr(this,10)
{
  // Constructor of flat polynomial function

}

//_____________________________________________________________________________
RooBSpline::RooBSpline(const RooBSpline& other, const char* name) :
  RooAbsReal(other, name), 
  _controlPoints("controlPoints",this,other._controlPoints),
  _m(other._m),
//   _t_ary(NULL),
//   _t("t_param", this, other._t),
  _n(other._n),
  _weights("weights",this,other._weights),
  _bases("bases",this,other._bases),
  _vars("observables",this,other._vars),
  _cacheMgr(this,10)
{
  // Copy constructor
  
//   _t_ary = new double[_m];
//   for (int i=0;i<_m;i++)
//   {
//     _t_ary[i] = other._t_ary[i];
//   }
}


//_____________________________________________________________________________
RooBSpline::~RooBSpline() 
{
  // Destructor
//   delete _t_ary;
}




//_____________________________________________________________________________
Double_t RooBSpline::evaluate() const 
{
  //cout << "In RooBSpline::evaluate(): " << GetName() << endl;
  // Calculate and return value of spline

  //cout << "computing S" << endl;
  RooBSplineBases* bases = (RooBSplineBases*)&_bases.arg();
  bases->getVal(); // build the basis polynomials
  bool useWeight = _weights.getSize();
  double S = 0;
  //bool even = fabs(_n/2-_n/2.) < 0.0000000001;
  for (int i=0;i<_m-_n-1;i++)
  {
    //if (even && i <_m-_n-2) p=_n-1;
    double basis = bases->getBasisVal(_n,i,false);
    if (basis > 0)
    {
      int p=i;
      //if (even && i > 0) p=i-1;
      RooAbsReal* point = (RooAbsReal*)_controlPoints.at(p);
      //cout << "name=" << GetName() << ", point addy=" << point << endl;
      double weight = 1.0;
      if (useWeight)
      {
	RooAbsReal* weightVar = (RooAbsReal*)_weights.at(p);
	weight = weightVar->getVal();
      }
      S += basis * point->getVal() * weight;
    }
  }

  //cout << "Out RooBSpline::evaluate()" << endl;
  return S;
}



void RooBSpline::setWeights(const RooArgList& weights)
{
  _weights.removeAll();
  bool first=1;
  TIterator* pointIter = weights.createIterator() ;
  RooAbsArg* point ;
  RooAbsArg* lastPoint=NULL ;
  while((point = (RooAbsArg*)pointIter->Next())) {
    if (!dynamic_cast<RooAbsReal*>(point)) {
      coutE(InputArguments) << "RooBSpline::ctor(" << GetName() << ") ERROR: control point " << point->GetName() 
			    << " is not of type RooAbsReal" << endl ;
      assert(0) ;
    }
    _weights.add(*point) ;
    if (first) 
    {
      for (int i=0;i<_n/2;i++)
      {
	_weights.add(*point) ;
      }
    }
    first=false;
    lastPoint=point;
  }
  for (int i=0;i<(_n+1)/2;i++) _weights.add(*lastPoint);
  delete pointIter;
}




//_____________________________________________________________________________
Bool_t RooBSpline::setBinIntegrator(RooArgSet& allVars) 
{
  //cout << "In RooBSpline::setBinIntegrator" << endl;
  if(allVars.getSize()==1){
    RooAbsReal* temp = const_cast<RooBSpline*>(this);
    temp->specialIntegratorConfig(kTRUE)->method1D().setLabel("RooBinIntegrator")  ;
    int nbins = ((RooRealVar*) allVars.first())->numBins();
    temp->specialIntegratorConfig(kTRUE)->getConfigSection("RooBinIntegrator").setRealValue("numBins",nbins);
    return true;
  }else{
    cout << "Currently BinIntegrator only knows how to deal with 1-d "<<endl;
    return false;
  }
  return false;
}


//_____________________________________________________________________________
Int_t RooBSpline::getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, 
					  const RooArgSet* normSet, const char* rangeName) const 
{
//   cout << "In RooBSpline["<<GetName()<<"]::getAnalyticalIntegralWN" << endl;
//   cout << "allVars:" << endl;
//   allVars.Print("V");
//   cout << "analVars:" << endl;
//   analVars.Print("V");
//   cout << "_vars:" << endl;
//   _vars.Print("V");
//   cout << "--- end ---" << endl;
  
  if (_forceNumInt) return 0 ;

  if (_vars.getSize()==0) return 1;
  
  if (matchArgs(allVars, analVars, *_vars.first())) {

      // From RooAddition:
      // check if we already have integrals for this combination of factors

      // we always do things ourselves -- actually, always delegate further down the line ;-)
      analVars.add(allVars);
      if( normSet ) analVars.add(*normSet);
      
      // check if we already have integrals for this combination of factors
      Int_t sterileIndex(-1);
      CacheElem* cache = (CacheElem*) _cacheMgr.getObj(&analVars,&analVars,&sterileIndex,RooNameReg::ptr(rangeName));
      if (cache==0) {
         // we don't, so we make it right here....
         cache = new CacheElem;         
         for (int i=0;i<_m-_n-1;i++) {
            RooAbsReal* point = (RooAbsReal*)_controlPoints.at(i);
            cache->_I.addOwned( *point->createIntegral(analVars,rangeName) );
         }
      }

      Int_t code = _cacheMgr.setObj(&analVars,&analVars,(RooAbsCacheElement*)cache,RooNameReg::ptr(rangeName));
      return 2+code;
  }

  return 0;
}


//_____________________________________________________________________________
Double_t RooBSpline::analyticalIntegralWN(Int_t code, const RooArgSet* /*normSet*/,const char* rangeName) const 
{
  //cout << "In RooBSpline::analyticalIntegralWN" << endl;
  double integral = 0;
  if (code == 1)
  {
    return getVal();
  }
  else if (code >= 2)
  {
//     RooRealVar* obs = (RooRealVar*)_vars.first();
//     int nrBins = obs->getBins();
//     int initValue=obs->getBin();
// 
//     for (int ibin=0;ibin<nrBins;ibin++)
//     {
//       obs->setBin(ibin);
//       integral+=getVal();
//     }
// 
//     obs->setBin(initValue);



// 
//       // From RooAddition:
//       // check if we already have integrals for this combination of factors
// 
//       // we always do things ourselves -- actually, always delegate further down the line ;-)
//       RooArgSet analVars( _vars );
//       analVars.add(*normSet);
// 
//       Int_t sterileIndex(-1);
//       CacheElem* cache = (CacheElem*) _cacheMgr.getObj(&analVars,&analVars,&sterileIndex,RooNameReg::ptr(rangeName));
//       if (cache==0) {
//          // we don't, so we make it right here....
//          cache = new CacheElem;         
//          for (int i=0;i<_m-_n-1;i++) {
//             RooAbsReal* point = (RooAbsReal*)_controlPoints.at(i);
//             cache->_I.addOwned( *point->createIntegral(_vars,*normSet) );
//          }
//       }
// 


     // Calculate integral internally from appropriate integral cache
   
     // note: rangeName implicit encoded in code: see _cacheMgr.setObj in getPartIntList...
     CacheElem *cache = (CacheElem*) _cacheMgr.getObjByIndex(code-2);
     if (cache==0) {
       // cache got sterilized, trigger repopulation of this slot, then try again...
       //cout << "Cache got sterilized" << endl;
       std::unique_ptr<RooArgSet> vars( getParameters(RooArgSet()) );
       RooArgSet dummy;
#if ROOT_VERSION_CODE >= ROOT_VERSION(6,25,0) 
       auto iset(  _cacheMgr.selectFromSet2(*vars,code-2) );
       Int_t code2 = getAnalyticalIntegral(iset,dummy,rangeName);       
#else
       std::unique_ptr<RooArgSet> iset(  _cacheMgr.nameSet2ByIndex(code-2)->select(*vars) );
       Int_t code2 = getAnalyticalIntegral(*iset,dummy,rangeName);
#endif
       assert(code==code2); // must have revived the right (sterilized) slot...
       return analyticalIntegral(code2,rangeName);
     }
     assert(cache!=0);
   



     RooBSplineBases* bases = (RooBSplineBases*)&_bases.arg();
     bases->getVal(); // build the basis polynomials
     bool useWeight = _weights.getSize();
     double S = 0;
     //bool even = fabs(_n/2-_n/2.) < 0.0000000001;
     for (int i=0;i<_m-_n-1;i++)
     {
       //if (even && i <_m-_n-2) p=_n-1;
       double basis = bases->getBasisVal(_n,i,false);
       if (basis > 0)
       {
         int p=i;
         //if (even && i > 0) p=i-1;
         //RooAbsReal* point = (RooAbsReal*)_controlPoints.at(p);
         //cout << "name=" << GetName() << endl;
         double weight = 1.0;
         if (useWeight)
         {
           RooAbsReal* weightVar = (RooAbsReal*)_weights.at(p);
           weight = weightVar->getVal();
         }


         RooAbsReal* intReal = (RooAbsReal*)cache->_I.at(p);   //point->createIntegral(_vars,*normSet);
         S += basis * intReal->getVal() * weight;
         //cout << "adding "<<intReal->getVal()<<" to integral" << endl;
         //delete intReal;
       }
     }
     
     integral = S;
  }
  //cout << "Spline " << GetName() << " has integral " << integral << " obtained with code "<< code << endl;
  return integral;
}

// //_____________________________________________________________________________
// RooBSplinePenalty* RooBSpline::getRealPenalty(int k, RooRealVar* obs,RooRealVar* beta,const char* name) const
// {
//   if (name == "")
//   {
//     stringstream nameStr;
//     nameStr << GetName() << "_penalty";
//     name = nameStr.str().c_str();
//   }

//   RooArgList controlPoints;
//   for (int i=_n/2;i<_controlPoints.getSize()-(_n+1)/2;i++)
//   {
//     //cout << "adding point with val " << ((RooAbsReal*)_controlPoints.at(i))->getVal() << endl;
//     controlPoints.add(*_controlPoints.at(i));
//   }

//   vector<double> tValues;
//   for (int i=_n;i<_m-_n;i++)
//   {
//     tValues.push_back(_t_ary[i]);
//   }

//   RooBSplinePenalty* pen = new RooBSplinePenalty(name, name, _n, tValues, controlPoints, k, obs, beta);

//   int nrWeights = _weights.getSize();
//   if (nrWeights > 0)
//   {
//     RooArgSet weights;
//     int counter = 0;
//     for (int i=_n/2;i<nrWeights-(_n+1)/2;i++)
//     {
//       weights.add(*_weights.at(i));
//       counter++;
//     }
//     //cout << "added " << counter << " weights" << endl;
//     pen->setWeights(weights);
//   }

//   return pen;
// }






//_____________________________________________________________________________
RooArgList RooBSpline::CacheElem::containedArgs(Action)
{
  // Return list of all RooAbsArgs in cache element
  RooArgList ret(_I) ;
  return ret ;
}

RooBSpline::CacheElem::~CacheElem()
{
  // Destructor
}



//_____________________________________________________________________________
#if ROOT_VERSION_CODE >= ROOT_VERSION(6,27,0)

#include <RooFitHS3/RooJSONFactoryWSTool.h>
#include <RooFit/Detail/JSONInterface.h>
#include <RooFitHS3/JSONIO.h>
#include <RooWorkspace.h>

using namespace RooFit::Detail;

namespace {
class RooBSplineStreamer : public RooFit::JSONIO::Exporter {
public:
  virtual const std::string& key() const override {
    static const std::string _key = "BSpline";
    return _key;
  }
  
  virtual bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
  {
    const RooBSpline *spline = static_cast<const RooBSpline *>(func);
    elem["type"] << key();

    auto &bases = elem["bases"];
    bases << spline->getBases()->GetName();
    
    auto &vars = elem["variables"];
    vars.set_seq();
    for(auto w:spline->getVariables()){
      vars.append_child() << w->GetName();
    }
    auto &controlPoints = elem["controlPoints"];
    controlPoints.set_seq();
    for(auto w:spline->getControlPoints()){
      controlPoints.append_child() << w->GetName();
    }    

    return true;
  }
};
} // namespace


namespace {
class RooBSplineFactory : public RooFit::JSONIO::Importer {
public:
   virtual bool importFunction(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
     std::string name = RooJSONFactoryWSTool::name(p);

     RooArgList controlPoints;
     for(const auto& point:p["controlPoints"].children()){
       controlPoints.add(*tool->workspace()->var(point.val().c_str()));
     }
     RooArgList vars;
     for(const auto& var:p["variables"].children()){
       controlPoints.add(*tool->workspace()->var(var.val().c_str()));
     }     
					
     
     RooBSpline* spline = new RooBSpline(name.c_str(),name.c_str(),
					 controlPoints,
					 *(RooBSplineBases*)tool->workspace()->function(p["bases"].val().c_str()),
					 vars);
     tool->workspace()->import(*spline);
     delete spline;
     return true;
   }
};
} // namespace


namespace {
  int register_serializations(){
    RooFit::JSONIO::registerImporter("BSpline", std::make_unique<RooBSplineFactory>());
    RooFit::JSONIO::registerExporter(RooBSpline::Class(), std::make_unique<RooBSplineStreamer>());
    return 1;
  }
  int _dummy = register_serializations();
}

#endif
