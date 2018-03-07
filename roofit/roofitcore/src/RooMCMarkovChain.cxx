/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooMCMarkovChain.h,v 1.00 2017/14/11 11:13:42

 * Author:                                                                  *
 *   OD, Oliver Dahme, University of Zurich, o.dahme@cern.ch      *
 *
 *****************************************************************************/
#include "RooFit.h"
#include <fstream>
#include <algorithm>
#include <iomanip>
#include "TH1.h"
#include "TH2.h"
#include "TMarker.h"
#include "TGraph.h"
#include "TFitter.h"
#include "TMatrixDSym.h"
#include "RooMCMarkovChain.h"
#include "RooArgSet.h"
#include "RooArgList.h"
#include "RooAbsReal.h"
#include "RooAbsRealLValue.h"
#include "RooRealVar.h"
#include "RooFitResult.h"
#include "RooPlot.h"
#include "RooMsgService.h"
#include <TVectorD.h>
#include "TMatrixD.h"
#include <TRandom3.h>
#include "RooDataSet.h"
#include "TDecompChol.h"
#include <iostream>
#include "TStyle.h"
#include "TMultiGraph.h"
#include "TLine.h"

#include <cstdlib>

using namespace std;

ClassImp(RooMCMarkovChain);

/** \class RooMCMarkovChain
  RooMCMarkovChain is used as the RooMinuit class except that it is using a Monte Carlo Markov Chain as a minimizer. A tutorial can be found in the roofit section where a basic comparison with Minuit can be performed.
*/


TVirtualFitter *RooMCMarkovChain::_theFitter = 0 ;


////////////////////////////////////////////////////////////////////////////////
/// cleanup function to reset the fitter
void RooMCMarkovChain::cleanup()
{
  if (_theFitter) {
    delete _theFitter ;
    _theFitter =0 ;
  }
}

////////////////////////////////////////////////////////////////////////////////
/// RooMCMarkovChain constructor: takes the negative log liklihood function as an argument

RooMCMarkovChain::RooMCMarkovChain(RooAbsReal& function)
{
  _func = &function ;
  _verbose = kFALSE ;
  _gaus = kFALSE;
  _interval = kFALSE;

  // Examine parameter list
   RooArgSet* paramSet = function.getParameters(RooArgSet()) ;
   RooArgList paramList(*paramSet) ;
   delete paramSet ;

  _floatParamList = (RooArgList*) paramList.selectByAttrib("Constant",kFALSE) ;
  if (_floatParamList->getSize()>1) {
    _floatParamList->sort() ;
  }
  _floatParamList->setName("floatParamList") ;

  _constParamList = (RooArgList*) paramList.selectByAttrib("Constant",kTRUE) ;
  if (_constParamList->getSize()>1) {
    _constParamList->sort() ;
  }
  _constParamList->setName("constParamList") ;

  // Remove all non-RooRealVar parameters from list
  TIterator* pIter = _floatParamList->createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)pIter->Next())) {
    if (!arg->IsA()->InheritsFrom(RooAbsRealLValue::Class())) {
      _floatParamList->remove(*arg) ;
    }
  }
  _nPar      = _floatParamList->getSize() ;
  delete pIter ;

  updateFloatVec() ;

   // Save snapshot of initial lists
   _initFloatParamList = (RooArgList*) _floatParamList->snapshot(kFALSE) ;
   _initConstParamList = (RooArgList*) _constParamList->snapshot(kFALSE) ;

  // Initialize
   Int_t nPar= _floatParamList->getSize() + _constParamList->getSize() ;
   if (_theFitter) delete _theFitter ;
   _theFitter = new TFitter(nPar*2+1) ;
   _theFitter->SetObjectFit(this) ;
   setPrintLevel(-1) ;
   _theFitter->Clear();

}



////////////////////////////////////////////////////////////////////////////////
/// Destructor: clears all points in the Markov Chain
RooMCMarkovChain::~RooMCMarkovChain()
{
  _floatParamVec.clear() ;
  _pointList.clear();
  _sortPointList.clear();
  _cutoffList.clear();
}

////////////////////////////////////////////////////////////////////////////////
/// minimizes function, cutoff points and calculates errors, "gaus" for symetric ones, and "interval" for asymetric ones.
/// \param[in] npoints = number of steps of the Markov Chain
/// \param[in] cutoff = number of points to be cut off starting from the first
/// \param[in] errorstrategy takes "gaus" or "interval"

Int_t RooMCMarkovChain::mcmc(size_t npoints, size_t cutoff, const char* errorstrategy)
{
  std::cout << "Starting Monte Carlo Markov Chain Fit with "<< npoints <<" points and cutoff after "<< cutoff <<" points" << '\n';
  // RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::Ignore); // turning off errors, because they are unnecessary
  Bool_t verbose = _verbose;
  int pl = 0;
  if (_printLevel > 0) {
    pl = _printLevel;
  }

  if (strcmp(errorstrategy, "gaus") == 0) {
    _gaus = kTRUE;
  } else if (strcmp(errorstrategy, "interval") == 0) {
    _interval = kTRUE;
  } else {
    std::cout << "unknown errorstrategy setting strategy to gaus" << '\n';
    _gaus = kTRUE;
  }

  Double_t seed = _seed;
  if (seed == 0) {
    time_t  timev;
    double systime = std::time(&timev);
    systime /= 1e7;
    seed = systime*13-5;
  }

  TRandom3 *rnd = new TRandom3(seed); //random generator with seed
  unsigned int nparams = _nPar; //number of parameters
  unsigned int nstat = npoints*100;//number of tries
  double maxstep = 0.01; //maximum step size
  double alphastar = _alphaStar; //forced acceptance rate

  Bool_t accepted = kTRUE;
  unsigned int ntested = 0;
  size_t naccepted = 0;

  //creating the negative log-likelihood value
  double nllval;
  TVectorD* last = new TVectorD(nparams); //last state
  TVectorD* curr = new TVectorD(nparams); //current state
  TVectorD* lowLimit = new TVectorD(nparams); //lower Limits
  TVectorD* upLimit = new TVectorD(nparams); //upper Limits
  int indexofbest = -1; //index in the pointlist for the minimum
  _pointList.reserve(npoints); //reserving RAM for pointlist
  _nameList.reserve(nparams+1);

  //Initialize last state
  RooArgList* startpoint = (RooArgList*) _floatParamList->snapshot(kTRUE);
  for (size_t index = 0; index < _nPar; index++) {
    RooRealVar* var = (RooRealVar*) startpoint->at(index);
    (*last)[index] = var->getVal();
    (*lowLimit)[index] = var->getMin();
    (*upLimit)[index] = var->getMax();
    _nameList.push_back(var->GetName());
  }
  _nameList.push_back("nll value");

  double minllh = 1e32; //Initialize container for min value


  //Initialize containers for S Matrix calculation
  std::vector<bool> lastaccepted;
  TMatrixDSym* identity = new TMatrixDSym(nparams);
  identity->UnitMatrix();
  TMatrixDSym* S1 = new TMatrixDSym(nparams);
  S1->Zero();
  (*S1) = (*identity);
  TMatrixDSym* SNminusone = new TMatrixDSym(nparams);
  *SNminusone = *S1;
  TMatrixDSym* SN = new TMatrixDSym(nparams);
  SN->Zero();
  // RooMCMarkovChain* context = (RooMCMarkovChain*) RooMCMarkovChain::_theFitter->GetObjectFit();
  _theFitter->GetObjectFit();
  TVectorD* SW = new TVectorD(nparams);
  TVectorD* WN = new TVectorD(nparams);


  double llh_last;
  double llh_curr;
  Bool_t outofBounds = false;
  size_t progressPercent = 0;

  if (pl > 0) {
    std::cout << "starting minimization" << '\n';
  }
  for (unsigned int i = 0; i < nstat; i++) {
    if (accepted) {
      *curr = *last;//use value of last for current then vary
    }



    for (int j = 0; j < WN->GetNrows() ; j++) {
      (*WN)[j] = rnd->Gaus(0.0, maxstep); //Random inital step size
    }
    *SW =  *SNminusone * *WN; //Step size correction
    *curr += *SW; //vary current point


    for(size_t index= 0; index < nparams; index++) {
      setPdfParamVal(index, (*last)[index],verbose);
    }
    llh_last = _func->getVal();

    outofBounds = false;
    for(size_t index= 0; index < nparams; index++) {
      if (
        ((*curr)[index] < (*lowLimit)[index]) || ((*curr)[index] > (*upLimit)[index])
      ) {
        outofBounds = true;
      }
    }


    // If out of bounds, the negative log-likelihood values gets set very high
    // to provoke big step size away from the bounds
    if (outofBounds) {
      llh_curr = 1e32;
    } else {
      for(size_t index= 0; index < nparams; index++) {
        setPdfParamVal(index, (*curr)[index],verbose);
      }
      llh_curr = _func->getVal(); //get nll for current parameters
    }

  // update minimum value
    if (llh_curr < minllh) {
      minllh = llh_curr;
      indexofbest = naccepted;
    }

  // Computing rejection or acceptance of current point
    double alpha = std::min(1.0, exp(llh_last - llh_curr));
    double r = rnd->Uniform(0,1);
    if (r < alpha) {
      accepted = true; //success
      nllval = llh_curr;
      TVectorD *point = new TVectorD(nparams+1);
      for (size_t index = 0; index < nparams; index++) {
        (*point)[index] = (*curr)[index];
      }
      (*point)[nparams] = nllval;
      _pointList.push_back(point); //adding point the pointlist
      naccepted++;
      *last = *curr;

    } else {
      //reset to last candidate
      accepted = false;
      *curr = *last;
    }

    ntested++;
    //update S matrix
    TMatrixDSym* SNminusoneT = new TMatrixDSym(*SNminusone);
    SNminusoneT->T();
    double etan = std::min(1.0, nparams*pow(double(i), -2.0/3.0));
    TMatrixDSym* WNWNT = new TMatrixDSym(nparams);
    WNWNT->Zero();
    for (int row = 0; row < WNWNT->GetNrows(); row++) {
      for (int col = 0; col < WNWNT->GetNcols(); col++) {
        (*WNWNT)[row][col] = (*WN)[row]*(*WN)[col]/WN->Norm2Sqr();
      }
    }
    TMatrixDSym* SNSNT = new TMatrixDSym(nparams);
    *SNSNT = ((*identity) + (*WNWNT)*etan*(alpha-alphastar));
    *SNSNT = SNSNT->Similarity(*SNminusone);
    TDecompChol* chol = new TDecompChol(nparams);
    *chol = (*SNSNT);
    bool success = chol->Decompose();
    assert(success);
    TMatrixD* SNT = new TMatrixD(nparams,nparams);
    *SNT = chol->GetU();
    SNT->T();
    for (int row = 0; row < SNT->GetNrows(); row++) {
      for (int col = 0; col < SNT->GetNcols(); col++) {
        (*SNminusone)[row][col] = (*SNT)[row][col];
      }
    }
    delete SNminusoneT;
    delete WNWNT;
    delete SNSNT;
    delete chol;
    delete SNT;

    if ( accepted && ((naccepted % (npoints/100)) == 0) ) {
      progressPercent++;
      std::cout << progressPercent << "\% "<<flush;
    }

    if (naccepted == npoints) {
      // Saving minimum point
      _bestParamList =  _pointList[indexofbest];
      TVectorD *point = _pointList[indexofbest];
      for(size_t index= 0; index < nparams; index++) {
        setPdfParamVal(index, (*point)[index],verbose);
      }
      std::cout  << '\n';
      break;
    }




  }

// Filling all points after the cut into cutoffList
  _cutoff = cutoff;
  _cutoffList.reserve(npoints - cutoff);
  for (size_t i = cutoff; i < _pointList.size(); i++) {
    TVectorD* point = _pointList[i];
    _cutoffList.push_back(point);
  }
  // std::cout << "cutoffList filled" << std::endl;


// Calculate and print errors
  if (_gaus) {
    getGausErrors();
  }
  if (_interval) {
    for (size_t i = 0; i < nparams; i++) {
      getPercentile(_nameList[i]);
    }
  }

  // RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::PrintErrors); // turning on Eval errors, got turned off in the beginning

  delete rnd;
  delete last;
  delete curr;
  delete lowLimit;
  delete upLimit;
  delete identity;
  delete SNminusone;
  delete SN;
  delete SW;
  delete WN;


  return 1;
}

////////////////////////////////////////////////////////////////////////////////
///getProfile returns a profile of the nll for a certain parameter, which can be called by name. It does so by creating a TGraph and plots all the nll values of the walk in respect to the parameter. Also, it is possible to include the cutoff points or not.
/// \param[in] name = name of parameter
/// \param[in] cutoff = include or exclude cutoff-points

TGraph* RooMCMarkovChain::getProfile(const char* name, Bool_t cutoff)
{
  if (_pointList.size() == 0) {
    std::cout << "point list empty. Please run mcmc() first" << std::endl;
  }

  unsigned int np =0;
  if (cutoff) {
    np = _cutoffList.size();
  } else {
    np = _pointList.size();
  }


  unsigned int index = getIndex(name);
  TVectorD x(np);
  TVectorD y(np);

  if (cutoff) {
    for (unsigned int i = 0; i < np; i++) {
      TVectorD* point = _cutoffList[i];
      x[i] = (*point)[index];
      y[i] = (*point)[_nPar];
    }

  } else {
    for (unsigned int i = 0; i < np; i++) {
      TVectorD* point = _pointList[i];
      x[i] = (*point)[index];
      y[i] = (*point)[_nPar];
    }
  }

  TGraph* gr = new TGraph(x,y);
  gr->GetXaxis()->SetTitle(name);
  gr->GetYaxis()->SetTitle("nll value");


  return gr;
}

////////////////////////////////////////////////////////////////////////////////
/// getWalkDis returns a TMultigraph pointer of the walk distribution of a parameter, which is called by name. It does so by creating two TGraphs one with the points which had been cutoff and one with the included points. Also, it adds a dotted line where the cutoff has been set.
/// \param[in] name = name of parameter
/// \param[in] cutoff = include or exclude cutoff-points

TMultiGraph* RooMCMarkovChain::getWalkDis(const char* name, Bool_t cutoff)
{
  if (_pointList.size() == 0) {
    std::cout << "point list empty. Please run mcmc() first" << std::endl;
  }

  string graphTitelStr = "Walk Distribution of ";
  graphTitelStr += name;
  const char * graphTitelChar = graphTitelStr.c_str();
  string graphNameStr = "Dis";
  graphNameStr += name;
  const char * graphNameChar = graphNameStr.c_str();

  Int_t index = getIndex(name);
  TMultiGraph* graph = new TMultiGraph(graphNameChar,graphTitelChar);
  size_t np = 0;

  if (cutoff == kFALSE) {
    np = _cutoff;
    TVectorD x1(np);
    TVectorD y1(np);

    for (unsigned int i = 0; i < np; i++) {
      TVectorD* point = _pointList[i];
      x1[i] = i;
      y1[i] = (*point)[index];
    }
    TGraph* gr1 = new TGraph(x1,y1);
    gr1->SetLineColor(2);
    graph->Add(gr1);

    Double_t minVal = getMin(name);
    Double_t maxVal = getMax(name);
    Double_t x[2] = {Double_t(_cutoff),Double_t(_cutoff)};
    Double_t y[2] = {minVal,maxVal};
    TGraph* cutline = new TGraph(2,x,y);
    cutline->SetLineWidth(5);
    cutline->SetLineStyle(2);
    graph->Add(cutline);
  }

  np = _cutoffList.size();
  TVectorD x2(np);
  TVectorD y2(np);

  for (unsigned int i = 0; i < np; i++) {
    TVectorD* point = _cutoffList[i];
    x2[i] = _cutoff+i;
    y2[i] = (*point)[index];
  }
  TGraph* gr2 = new TGraph(x2,y2);
  if (cutoff == kFALSE) {
    gr2->SetLineColor(4);
  }
  graph->Add(gr2);

  graph->Draw("ap");
  graph->GetXaxis()->SetTitle("number of steps");
  graph->GetYaxis()->SetTitle(name);



  return graph;
}

////////////////////////////////////////////////////////////////////////////////
///getWalkDisHis returns a TH1F pointer with a histogram of the walk for a certain parameter, called by name. The number of bins for the histogram can be set by nbinsx. Cutoff points can be included or not. It does so by just adding all the points of the walk to a histogram. The main purpose is to look at the distribution of the points. This function is also used to calculate the symmetric errors of the parameter.
/// \param[in] name = name of parameter
/// \param[in] nbinsx = numer of bins in the histogram
/// \param[in] cutoff = include or exclude cutoff-points

TH1F* RooMCMarkovChain::getWalkDisHis(const char* name,  Int_t nbinsx, Bool_t cutoff)
{
  if (_pointList.size() == 0) {
    std::cout << "point list empty. Please run mcmc() first" << std::endl;
  }

  Double_t xlow = getMin(name);
  Double_t xup = getMax(name);

  string histTitelStr = "Histogram of ";
  histTitelStr += name;
  const char * histTitelChar = histTitelStr.c_str();
  string histNameStr = "hist";
  histNameStr += name;
  const char * histNameChar = histNameStr.c_str();

  Int_t index = getIndex(name);

  TH1F *hist = new TH1F(histNameChar, histTitelChar, nbinsx, xlow, xup);
  hist->GetXaxis()->SetTitle(name);

  if (cutoff) {
    unsigned int np = _cutoffList.size();

    for (unsigned int i = 0; i < np; i++) {
      TVectorD* point = _cutoffList[i];
      hist->Fill((*point)[index]);
    }
    return hist;
  } else {
    unsigned int np = _pointList.size();

    for (unsigned int i = 0; i < np; i++) {
      TVectorD* point = _pointList[i];
      hist->Fill((*point)[index]);
    }
    return hist;
  }
}

////////////////////////////////////////////////////////////////////////////////
///changeCutoff just changes the number of points, which should not be included into the error calculation. The number of cutoff points can be changed without re performing the walk.
/// \param[in] newCutoff = new number of points to be cut off

Int_t RooMCMarkovChain::changeCutoff(Int_t newCutoff)
{
  _cutoff = newCutoff;
  _cutoffList.clear();
  _cutoffList.reserve(_pointList.size() - newCutoff);
  for (size_t i = newCutoff; i < _pointList.size(); i++) {
    TVectorD* point = _pointList[i];
    _cutoffList.push_back(point);
  }
  return 1;
}

////////////////////////////////////////////////////////////////////////////////
///getCornerPlot returns a TH2D pointer with a 2D histogram of two parameters, called by name1 and name2. The number of bins for the name1 parameter are set by nbinsx and for name2 by nbinsy.  It does so just by adding all the points of the two parameters in a TH2D histogram. As always the cutoff can be turned on or off. This plot could for example be used to look for correlations.
/// \param[in] name1 = name of first parameter
/// \param[in] name2 = name of second parameter
/// \param[in] nbinsx = number of bins on the x-axis
/// \param[in] nbinsy = number of bins on the y-axis
/// \param[in] cutoff = include or exclude cutoff-points

TH2D* RooMCMarkovChain::getCornerPlot(const char* name1, const char* name2, Int_t nbinsx, Int_t nbinsy, Bool_t cutoff)
{
  string histNameStr = "cornerhist";
  histNameStr += name1;
  histNameStr += name2;
  const char * histNameChar = histNameStr.c_str();
  string histTitelStr = "Corner Plot of ";
  histTitelStr += name1;
  histTitelStr += " and ";
  histTitelStr += name2;
  const char * histTitelChar = histTitelStr.c_str();
  if (_pointList.size() == 0) {
    std::cout << "point list empty. Please run mcmc() first" << std::endl;
  }
  Int_t index1 = getIndex(name1);
  Int_t index2 = getIndex(name2);
  Double_t xlow = getMin(name1);
  Double_t xup = getMax(name1);
  Double_t ylow = getMin(name2);
  Double_t yup = getMax(name2);

  TH2D *hist = new TH2D(histNameChar,histTitelChar,nbinsx,xlow,xup,nbinsy,ylow,yup);
  hist->GetXaxis()->SetTitle(name1);
  hist->GetYaxis()->SetTitle(name2);

  if (cutoff) {
    unsigned int np = _cutoffList.size();
    Double_t x = 0;
    Double_t y = 0;

    for (unsigned int i = 0; i < np; i++) {
      TVectorD* point = _cutoffList[i];
      x = (*point)[index1];
      y = (*point)[index2];
      hist->Fill(x,y);
    }


    return hist;
  } else {
    unsigned int np = _pointList.size();
    Double_t x = 0;
    Double_t y = 0;

    for (unsigned int i = 0; i < np; i++) {
      TVectorD* point = _pointList[i];
      x = (*point)[index1];
      y = (*point)[index2];
      hist->Fill(x,y);
    }


    return hist;
  }
}
////////////////////////////////////////////////////////////////////////////////
///sortPointList sorts the points according to a value defined by name. It saves them into the _sortPointList.
/// \param[in] name = name of parameter

void RooMCMarkovChain::sortPointList(const char* name)
{
  int index = getIndex(name);
  _sortPointList.clear();
  _sortPointList.reserve(_cutoffList.size());
  for (size_t i = 0; i < _cutoffList.size(); i++) {
    TVectorD* point = _cutoffList[i];
    _sortPointList.push_back(point);
  }
  std::sort(_sortPointList.begin(),_sortPointList.end(), [&index](TVectorD* a, TVectorD* b){
    double var1 = (*a)[index];
    double var2 = (*b)[index];
    if (var1 < var2) {
      return kTRUE;
    }else{
      return kFALSE;
    }
  });
}

////////////////////////////////////////////////////////////////////////////////
///getIndex just returns the index of a parameter given by name.
/// \param[in] name = name of parameter

Int_t RooMCMarkovChain::getIndex(const char* name)
{
  Int_t index = 0;
  for (size_t i = 0; i < _nPar; i++) {
    const char* varname = _nameList[i];
    if (strcmp(name, varname) == 0) {
      index = i;
      break;
    }
  }
  return index;
}

////////////////////////////////////////////////////////////////////////////////
///printError prints symmetric errors of a parameter defined by name at a certain confidence level defined by conf. It does so by scanning the negative log likelihood points computed by the Markov Chain and takes the point left and right of the minimum nearest to the confidence level.
/// \param[in] name = name of parameter
/// \param[in] conf = confidence level must be between [0,1]

Int_t RooMCMarkovChain::printError(const char* name, Double_t conf)
{
  if (conf > 1.0) {
    std::cout << "confidence level must be between 0 and 1, setting to 0.682" << '\n';
    conf = 0.682;
  }
  sortPointList(name);
  Int_t count = int(_sortPointList.size() * conf) ;
  Double_t high = -1e32;
  Double_t low = 1e32;
  Int_t index = getIndex(name);

  for (Int_t i = 0; i < count; i++) {
    TVectorD* point = _sortPointList[i];
    double var = (*point)[index];
    if (var < low) {
      low = var;
    }
    if (var > high) {
      high = var;
    }
  }
  std::cout << "error on "<<name<<" = "<< (high - low)/2 << std::endl;
  return 1;

}

////////////////////////////////////////////////////////////////////////////////
///getPercentile prints the asymmetric errors of a parameter defined by name at a certain confidence level defined by conf. Is does so by scanning the negative log liklihood points computed by the Markov Chain and takes the two points left and right of the minimum nearest to the confidence level
/// \param[in] name = name of parameter
/// \param[in] conf = confidence level must be between [0,1]

Int_t RooMCMarkovChain::getPercentile(const char* name, Double_t conf)
{
  if (conf > 1.0) {
    std::cout << "confidence level must be between 0 and 1, setting to 0.682" << '\n';
    conf = 0.682;
  }
  Double_t per = conf;
  if (conf > 1.0) {
    per = 0.682;
  }
  Double_t left = 0;
  Double_t right = 0;
  Int_t index = getIndex(name);
  sortPointList(name);
  size_t np = _sortPointList.size();
  size_t i = 0;
  while (double(i)/double(np) < (1-per)/2) {
    TVectorD* point = _sortPointList[i];
    left = (*point)[index];
    i++;
  }

  i=np-1;
  size_t n = 0;
  while (double(n)/double(np) < (1-per)/2) {
    TVectorD* point = _sortPointList[i];
    right = (*point)[index];
    i--;
    n++;
  }
  double bestvar = (*_bestParamList)[index];
  std::cout << "ASYMETRIC ERROR at "<<per<<" confidence level for "<< name << std::endl;
  std::cout << "INTERVAL =\t[ "<< left <<" , "<< right <<" ]"<< std::endl;
  std::cout << "BEST     =\t"<< bestvar << std::endl;
  std::cout << "MINUS    =\t"<< bestvar - left << std::endl;
  std::cout << "PLUS     =\t"<< right - bestvar << std::endl;
  std::cout << "" << std::endl;


  return 1;
}
////////////////////////////////////////////////////////////////////////////////
///getGausErrors prints symetric errors of all parameters. It does so py calling getWalkDisHis for every parameter and reading the gaussian error of the histogram and printing it.

Int_t RooMCMarkovChain::getGausErrors()
{
  int nPar = _nPar;
  std::vector<size_t> nOfTabs;
  nOfTabs.reserve(nPar);
  size_t maxnOfTabs = 0;

  for (int i = 0; i < nPar; i++) {
    size_t nOfTabscurr = 0;
    size_t scan = 0;
    while (strlen(_nameList[i]) >= scan) {
      scan+=8;
      nOfTabscurr++;
    }
    nOfTabs.push_back(nOfTabscurr);
    if (maxnOfTabs < nOfTabscurr) {
      maxnOfTabs = nOfTabscurr;
    }
  }
  std::vector<TH1F*> hist1D;
  hist1D.reserve(nPar);
  for (int i = 0; i < nPar;i++) {
    TH1F* hist = getWalkDisHis(_nameList[i],100,kTRUE);
    hist1D.push_back(hist);
  }

  std::vector<TH2D*> hist2D;
  hist2D.reserve(nPar*(nPar-1)/2);
  for (int i = 0; i < nPar; i++) {
    for (int j = i+1; j < nPar; j++) {
      TH2D* hist = getCornerPlot(_nameList[i],_nameList[j],100,100,kTRUE);
      hist2D.push_back(hist);
    }
  }

  for (size_t i = 0; i < nOfTabs.size(); i++) {
    nOfTabs[i] = maxnOfTabs - nOfTabs[i] +1 ;
  }

  cout.precision(5);
  std::cout <<"NO."<<"\t"<<"NAME";
  for (size_t i = 0; i < maxnOfTabs; i++) {std::cout<<"\t";}
  std::cout<<"VALUE"<<"\t\t"<<"ERROR"<< std::endl;

  for (int i = 0; i < nPar; i++) {
    std::cout <<i+1<<std::scientific<<"\t"<<_nameList[i];
    for (size_t j = 0; j < nOfTabs[i]; j++) {std::cout<<"\t";}
    if (hist1D[i]->GetMean() < 0) {
      cout<<" "<< hist1D[i]->GetMean();
    } else {
      cout<< hist1D[i]->GetMean();
    }
    cout<<"\t"<<hist1D[i]->GetRMS()<< std::endl;
    setPdfParamErr(i,hist1D[i]->GetRMS());
  }
  std::cout << "" << std::endl;
  Double_t corr[nPar][nPar];
  int n = 0;
  for (int i = 0; i < nPar; i++) {
    for (int j = i+1; j < nPar; j++) {
      if (i == j) {
        corr[i][j] = 1.0;
      }else{
        corr[i][j] = hist2D[n]->GetCorrelationFactor();
        n++;
      }
    }
  }
  for (int i = 0; i < nPar; i++) {
    for (int j = i; j < nPar; j++) {
      if (i == j) {
        corr[i][j] = 1.0;
      }else{
        corr[j][i] = corr[i][j];
      }
    }
  }
  cout.precision(3);
  std::cout <<std::fixed<< "CORRELATION COEFFICIENTS" << std::endl;
  std::cout << "NO."<<"\t";
  for (int i = 0; i < nPar; i++) {
    std::cout << i+1<< "\t";
  }
  std::cout << "" << std::endl;

  for (int i = 0; i < nPar; i++) {
    std::cout << i+1<<"\t";
    for (int j = 0; j < nPar; j++) {
      std::cout << corr[i][j] <<"\t";
    }
    std::cout << "" << std::endl;
  }
  std::cout << "" << std::endl;

  for (size_t i = 0; i < hist1D.size(); i++) {
    delete hist1D[i];
  }
  for (size_t i = 0; i < hist2D.size(); i++) {
    delete hist2D[i];
  }


  hist1D.clear();
  hist2D.clear();

  return 1;
}

////////////////////////////////////////////////////////////////////////////////
///saveCandidatesAs saves all the points of the Markov Chain in a file defined by name, for example "points.txt" to save them in a text file. One could publish the file alongside a paper, such that somebody can download it to recompute the values and errors of the fit published in the paper.
/// \param[in] name = name of parameter

Int_t RooMCMarkovChain::saveCandidatesAs(const char* name)
{
  ofstream candidates;
  candidates.open(name);
  for (size_t i = 0; i < _nPar; i++) {
    candidates << _nameList[i] << "\t";
  }
  candidates << "\n";

  for (size_t i = 0; i < _pointList.size(); i++) {
    TVectorD* point = _pointList[i];
    for (size_t j = 0; j < _nPar; j++) {
      double var = (*point)[j];
      candidates << var << "\t";
    }
    candidates << "\n";
  }
  candidates.close();
  return 1;
}


////////////////////////////////////////////////////////////////////////////////
///Returns minimal value of a parameter
/// \param[in] name = name of parameter

Double_t RooMCMarkovChain::getMin(const char* name)
{
  size_t index = getIndex(name);
  Double_t minval = 1e32;
  for (size_t i = 0; i < _cutoffList.size(); i++) {
    TVectorD* point = _cutoffList[i];
    double var =  (*point)[index];
    if (var < minval) {
      minval = var;
    }
  }
  return minval;
}

////////////////////////////////////////////////////////////////////////////////
///Returns maximum value of a parameter
/// \param[in] name = name of parameter

Double_t RooMCMarkovChain::getMax(const char* name)
{
  size_t index = getIndex(name);
  Double_t maxval = -1e32;
  for (size_t i = 0; i < _cutoffList.size(); i++) {
    TVectorD* point = _cutoffList[i];
    double var =  (*point)[index];
    if (var > maxval) {
      maxval = var;
    }
  }
  return maxval;
}

////////////////////////////////////////////////////////////////////////////////
///Enable or disable Offesting
/// \param[in] flag = true or false

void RooMCMarkovChain::setOffsetting(Bool_t flag)
{
  _func->enableOffsetting(flag) ;
}

////////////////////////////////////////////////////////////////////////////////
///Set level of output (not implemented)
/// \param[in] newLevel = level of printing

Int_t RooMCMarkovChain::setPrintLevel(Int_t newLevel)
{
  Int_t ret = _printLevel ;
  Double_t arg(newLevel) ;
  _theFitter->ExecuteCommand("SET PRINT",&arg,1);
  _printLevel = newLevel ;
  return ret ;
}

////////////////////////////////////////////////////////////////////////////////
///Set seed for random generator
/// \param[in] seed = new seed

void RooMCMarkovChain::setSeed(Double_t seed)
{
 _seed = seed;
}

////////////////////////////////////////////////////////////////////////////////
///Change forced acceptance rate, not recommended
/// \param[in] newAlpha = new forced acceptance rate

void RooMCMarkovChain::setAlphaStar(Double_t newAlpha) {
 _alphaStar = newAlpha;
}

////////////////////////////////////////////////////////////////////////////////
///Returns number of parameters

size_t RooMCMarkovChain::getNPar() {
  return _nPar ;
}

////////////////////////////////////////////////////////////////////////////////
///Returns names of parameters

std::vector<const char*> RooMCMarkovChain::getNames() {
  return _nameList;
}

////////////////////////////////////////////////////////////////////////////////
/// Modify PDF parameter value by ordinal index

Bool_t RooMCMarkovChain::setPdfParamVal(Int_t index, Double_t value, Bool_t verbose)
{
  Bool_t answer= kFALSE;
  RooRealVar* par = (RooRealVar*)_floatParamVec[index] ;

  if (par->getVal()!=value) {
    if (verbose) cout << par->GetName() << "=" << value << ", " ;
    par->setVal(value) ;
    answer = kTRUE ;
  }
  return answer ;
 }

////////////////////////////////////////////////////////////////////////////////
/// Modify PDF parameter error by ordinal index

void RooMCMarkovChain::setPdfParamErr(Int_t index, Double_t value)
{
  ((RooRealVar*)_floatParamList->at(index))->setError(value) ;
}

////////////////////////////////////////////////////////////////////////////////
///Updates float vector

void RooMCMarkovChain::updateFloatVec()
{
  _floatParamVec.clear() ;
  RooFIter iter = _floatParamList->fwdIterator() ;
  RooAbsArg* arg ;
  _floatParamVec.resize(_floatParamList->getSize()) ;
  Int_t i(0) ;
  while((arg=iter.next())) {
    _floatParamVec[i++] = arg ;
  }
}
