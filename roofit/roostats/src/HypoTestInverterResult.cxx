// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


/**
   HypoTestInverterResult class: holds the array of hypothesis test results and compute a confidence interval.
   Based on the RatioFinder code available in the RooStatsCms package developed by Gregory Schott and Danilo Piparo
   Ported and adapted to RooStats by Gregory Schott
   Some contributions to this class have been written by Matthias Wolf (error estimation)
**/


// include header file of this class 
#include "RooStats/HypoTestInverterResult.h"

#include "RooStats/HypoTestInverterPlot.h"
#include "RooStats/HybridResult.h"
#include "RooStats/SamplingDistribution.h"
#include "RooMsgService.h"
#include "RooGlobalFunc.h"

#include "TF1.h"
#include "TGraphErrors.h"
#include <cmath>
#include "Math/BrentRootFinder.h"
#include "Math/WrappedFunction.h"

#include "TCanvas.h"
#include "RooStats/SamplingDistPlot.h"

ClassImp(RooStats::HypoTestInverterResult)

using namespace RooStats;
using namespace RooFit;


HypoTestInverterResult::HypoTestInverterResult(const char * name ) :
   SimpleInterval(name),
   fUseCLs(false),
   fInterpolateLowerLimit(true),
   fInterpolateUpperLimit(true),
   fFittedLowerLimit(false),
   fFittedUpperLimit(false),
   fInterpolOption(kLinear),
   fLowerLimitError(0),
   fUpperLimitError(0)
{
  // default constructor
}


HypoTestInverterResult::HypoTestInverterResult( const char* name,
						const RooRealVar& scannedVariable,
						double cl ) :
   SimpleInterval(name,scannedVariable,-999,999,cl), 
   fUseCLs(false),
   fInterpolateLowerLimit(true),
   fInterpolateUpperLimit(true),
   fFittedLowerLimit(false),
   fFittedUpperLimit(false),
   fInterpolOption(kLinear),
   fLowerLimitError(0),
   fUpperLimitError(0)
{
   // constructor 
   fYObjects.SetOwner();
   // put a cloned copy of scanned variable to set in the interval
   // to avoid I/O problem of the Result class - 
   // make the set owning the cloned copy (use clone istead of Clone to not copying all links)
   fParameters.removeAll();
   fParameters.takeOwnership();
   fParameters.addOwned(*((RooRealVar *) scannedVariable.clone(scannedVariable.GetName()) ));
}


HypoTestInverterResult::~HypoTestInverterResult()
{
   // destructor
   // no need to delete explictly the objects in the TList since the TList owns the objects
}


bool HypoTestInverterResult::Add( const HypoTestInverterResult& otherResult   )
{
  /// Merge this HypoTestInverterResult with another
  /// HypoTestInverterResult passed as argument

   int nThis = ArraySize();
   int nOther = otherResult.ArraySize();
   if (nOther == 0) return true;
   if (nOther != otherResult.fYObjects.GetSize() ) return false; 
   if (nThis != fYObjects.GetSize() ) return false; 

   // cannot merge in case of inconsistent memebrs
   if (fExpPValues.GetSize() > 0 && fExpPValues.GetSize() != nThis ) return false;
   if (otherResult.fExpPValues.GetSize() > 0 && otherResult.fExpPValues.GetSize() != nOther ) return false;

   oocoutI(this,Eval) << "HypoTestInverterResult::Add - merging result from " << otherResult.GetName()  
                                << " in " << GetName() << std::endl;

   bool addExpPValues = (fExpPValues.GetSize() == 0 && otherResult.fExpPValues.GetSize() > 0);
   bool mergeExpPValues = (fExpPValues.GetSize() > 0 && otherResult.fExpPValues.GetSize() > 0);

   if (nThis == 0) {
      fXValues = otherResult.fXValues;
      for (int i = 0; i < nOther; ++i) 
         fYObjects.Add( otherResult.fYObjects.At(i)->Clone() );
      for (int i = 0; i <  fExpPValues.GetSize() ; ++i)
         fExpPValues.Add( otherResult.fExpPValues.At(i)->Clone() );
   }
   // now to common merge combining point with same value and adding extra ones 
   for (int i = 0; i < nOther; ++i) {
      double otherVal = otherResult.fXValues[i];
      HypoTestResult * otherHTR = (HypoTestResult*) otherResult.fYObjects.At(i);
      if (otherHTR == 0) continue;
      bool sameXFound = false;
      for (int j = 0; j < nOther; ++j) {
         double thisVal = fXValues[j];
         
            // if same value merge the result 
            if ( (std::abs(otherVal) < 1  && TMath::AreEqualAbs(otherVal, thisVal,1.E-12) ) || 
                 (std::abs(otherVal) >= 1 && TMath::AreEqualRel(otherVal, thisVal,1.E-12) ) ) {
               HypoTestResult * thisHTR = (HypoTestResult*) fYObjects.At(j);               
               thisHTR->Append(otherHTR);
               sameXFound = true;
               if (mergeExpPValues) { 
                  ((SamplingDistribution*) fExpPValues.At(j))->Add( (SamplingDistribution*)otherResult.fExpPValues.At(i) );
                  std::cout << "adding expected p -values " << std::endl;
               }
               break;
            }
      }
      if (!sameXFound) { 
         // add the new result 
         fYObjects.Add(otherHTR->Clone() );
         fXValues.push_back( otherVal);
      }
      // add in any case also when same x found
      if (addExpPValues)  
         fExpPValues.Add( otherResult.fExpPValues.At(i)->Clone() );

   }
   
   if (ArraySize() > nThis) 
      oocoutI(this,Eval) << "HypoTestInverterResult::Add  - new number of points is " << fXValues.size()
                         << std::endl;
   else 
      oocoutI(this,Eval) << "HypoTestInverterResult::Add  - new toys/point is " 
                         <<  ((HypoTestResult*) fYObjects.At(0))->GetNullDistribution()->GetSize() 
                         << std::endl;
      
   return true;
}

 
double HypoTestInverterResult::GetXValue( int index ) const
{
 // function to return the value of the parameter of interest for the i^th entry in the results
  if ( index >= ArraySize() || index<0 ) {
    oocoutE(this,InputArguments) << "Problem: You are asking for an impossible array index value\n";
    return -999;
  }

  return fXValues[index];
}

double HypoTestInverterResult::GetYValue( int index ) const
{
// function to return the value of the confidence level for the i^th entry in the results
  if ( index >= ArraySize() || index<0 ) {
    oocoutE(this,InputArguments) << "Problem: You are asking for an impossible array index value\n";
    return -999;
  }

  if (fUseCLs) 
    return ((HypoTestResult*)fYObjects.At(index))->CLs();
  else 
     return ((HypoTestResult*)fYObjects.At(index))->CLsplusb();  // CLs+b
}

double HypoTestInverterResult::GetYError( int index ) const
{
// function to return the estimated error on the value of the confidence level for the i^th entry in the results
  if ( index >= ArraySize() || index<0 ) {
    oocoutE(this,InputArguments) << "Problem: You are asking for an impossible array index value\n";
    return -999;
  }

  if (fUseCLs) 
    return ((HypoTestResult*)fYObjects.At(index))->CLsError();
  else 
    return ((HypoTestResult*)fYObjects.At(index))->CLsplusbError();
}

double HypoTestInverterResult::CLb( int index ) const
{
  // function to return the observed CLb value  for the i-th entry
  if ( index >= ArraySize() || index<0 ) {
    oocoutE(this,InputArguments) << "Problem: You are asking for an impossible array index value\n";
    return -999;
  }
  return ((HypoTestResult*)fYObjects.At(index))->CLb();  
}

double HypoTestInverterResult::CLsplusb( int index ) const
{
  // function to return the observed CLs+b value  for the i-th entry
  if ( index >= ArraySize() || index<0 ) {
    oocoutE(this,InputArguments) << "Problem: You are asking for an impossible array index value\n";
    return -999;
  }
  return ((HypoTestResult*)fYObjects.At(index))->CLsplusb();  
}
double HypoTestInverterResult::CLs( int index ) const
{
  // function to return the observed CLs value  for the i-th entry
  if ( index >= ArraySize() || index<0 ) {
    oocoutE(this,InputArguments) << "Problem: You are asking for an impossible array index value\n";
    return -999;
  }
  return ((HypoTestResult*)fYObjects.At(index))->CLs();  
}

double HypoTestInverterResult::CLbError( int index ) const
{
  // function to return the error on the observed CLb value  for the i-th entry
  if ( index >= ArraySize() || index<0 ) {
    oocoutE(this,InputArguments) << "Problem: You are asking for an impossible array index value\n";
    return -999;
  }
  return ((HypoTestResult*)fYObjects.At(index))->CLbError();  
}

double HypoTestInverterResult::CLsplusbError( int index ) const
{
  // function to return the error on the observed CLs+b value  for the i-th entry
  if ( index >= ArraySize() || index<0 ) {
    oocoutE(this,InputArguments) << "Problem: You are asking for an impossible array index value\n";
    return -999;
  }
  return ((HypoTestResult*)fYObjects.At(index))->CLsplusbError();  
}
double HypoTestInverterResult::CLsError( int index ) const
{
   // function to return the error on the observed CLs value  for the i-th entry
   if ( index >= ArraySize() || index<0 ) {
      oocoutE(this,InputArguments) << "Problem: You are asking for an impossible array index value\n";
      return -999;
   }
   return ((HypoTestResult*)fYObjects.At(index))->CLsError();  
}


HypoTestResult* HypoTestInverterResult::GetResult( int index ) const
{
   // get the HypoTestResult object at the given index point
   if ( index >= ArraySize() || index<0 ) {
      oocoutE(this,InputArguments) << "Problem: You are asking for an impossible array index value\n";
      return 0;
   }
   
   return ((HypoTestResult*) fYObjects.At(index));
}

int HypoTestInverterResult::FindIndex(double xvalue) const
{
   // find the index corresponding at the poi value xvalue
   // If no points is found return -1
   // Note that a tolerance is used of 10^-12 to find the closest point
  const double tol = 1.E-12;
  for (int i=1; i<ArraySize(); i++) {
     double xpoint = fXValues[i];
     if ( (std::abs(xvalue) > 1 && TMath::AreEqualRel( xvalue, xpoint, tol) ) ||
          (std::abs(xvalue) < 1 && TMath::AreEqualAbs( xvalue, xpoint, tol) ) )
        return i; 
  }
  return -1;
}


struct InterpolatedGraph { 
   InterpolatedGraph( const TGraph & g, double target, const char * interpOpt) : 
      fGraph(g), fTarget(target), fInterpOpt(interpOpt) {}

   // return interpolated value for x - target
   double operator() (double x) const { 
      return fGraph.Eval(x, (TSpline*) 0, fInterpOpt) - fTarget;
   }
   const TGraph & fGraph;
   double fTarget;
   TString fInterpOpt;
}; 

double HypoTestInverterResult::GetGraphX(const TGraph & graph, double y0, bool lowSearch, double axmin, double axmax) const  {
   // return the X value of the given graph for the target value y0
   // the graph is evaluated using linea rinterpolation by default. 
   // if option = "S" a TSpline3 is used 

   TString opt = "";
   if (fInterpolOption == kSpline)  opt = "S";

   InterpolatedGraph f(graph,y0,opt);
   ROOT::Math::BrentRootFinder brf;
   ROOT::Math::WrappedFunction<InterpolatedGraph> wf(f);

   // find reasanable xmin and xmax if not given
   const double * y = graph.GetY(); 
   int n = graph.GetN();
   double xmin = axmin; 
   double xmax = axmax;
   if (axmin >= axmax) { 
      xmin = graph.GetX()[0];
      xmax = graph.GetX()[n-1];
      // test if extrapolation is needed (only for case full range is given)
      if (n > 1) { 
         // do lower extrapolation 
         if ( (y[0] < y0 && y[1] < y[0]) || (y[0] > y0 && y[1] > y[0])  ) {         
            const RooRealVar* var = dynamic_cast<RooRealVar*>( fParameters.first() );
            if (var) xmin = var->getMin();
         }
         // do upper extrapolation  
         if ( (y[n-1] > y0 && y[n-2] > y[n-1]) || (y[n-1] < y0 && y[n-2] < y[n-1])  ) {         
            const RooRealVar* var = dynamic_cast<RooRealVar*>( fParameters.first() );
            if (var) xmax = var->getMax();
         }
      }
   }

   brf.SetFunction(wf,xmin,xmax);
   brf.SetNpx(20);
   bool ret = brf.Solve(100, 1.E-6, 1.E-6);
   if (!ret) { 
      ooccoutE(this,Eval) << "HypoTestInverterResult - interpolation failed - return inf" << std::endl;
         return TMath::Infinity();
   }
   double limit =  brf.Root();

   // look in case if a new interseption exists
   // only when boundaries are not given
   if (axmin >= axmax) { 
      int index = TMath::BinarySearch(n, graph.GetX(), limit);
      if (lowSearch && index >= 1 && (y[0] - y0) * ( y[index]- y0) < 0) { 
         //search if  another interseption exists at a lower value
         limit =  GetGraphX(graph, y0, lowSearch, graph.GetX()[0], graph.GetX()[index] );
      }
      else if (!lowSearch && index < n-2 && (y[n-1] - y0) * ( y[index+1]- y0) < 0) { 
         // another interseption exists at an higher value
         limit =  GetGraphX(graph, y0, lowSearch, graph.GetX()[index+1], graph.GetX()[n-1] );
      }
   }
   return limit;
}

double HypoTestInverterResult::FindInterpolatedLimit(double target, bool lowSearch, double xmin, double xmax )
{
   // interpolate to find a limit value
   // Use a linear or a spline interpolation depending on the interpolation option

   ooccoutD(this,Eval) << "HypoTestInverterResult - " 
                       << "Interpolate the upper limit between the 2 results closest to the target confidence level" 
                       << std::endl;


   if (ArraySize()<2) {
      ooccoutW(this,Eval) << "HypoTestInverterResul - not enough points to get the inverted interval \n";
      return (lowSearch) ? xmin : xmax;
   }

   // sort the values in x 
   int n = ArraySize();
   std::vector<unsigned int> index(n );
   TMath::SortItr(fXValues.begin(), fXValues.end(), index.begin(), false); 
   // make a graph with the sorted point
   TGraph graph(n);
   for (int i = 0; i < n; ++i) 
      graph.SetPoint(i, GetXValue(index[i]), GetYValue(index[i] ) );

   
   double limit =  GetGraphX(graph, target, lowSearch, xmin, xmax);


   return limit;
}

int HypoTestInverterResult::FindClosestPointIndex(double target)
{
  // find the object with the smallest error that is < 1 sigma from the target
  double bestValue = fabs(GetYValue(0)-target);
  int bestIndex = 0;
  for (int i=1; i<ArraySize(); i++) {
    if ( fabs(GetYValue(i)-target)<GetYError(i) ) { // less than 1 sigma from target CL
      double value = fabs(GetYValue(i)-target);
      if ( value<bestValue ) {
	bestValue = value;
	bestIndex = i;
      }
    }
  }

  return bestIndex;
}

Double_t HypoTestInverterResult::LowerLimit()
{
  if (fFittedLowerLimit) return fLowerLimit;
  //std::cout << "finding point with cl = " << 1-(1-ConfidenceLevel())/2 << endl;
  if ( fInterpolateLowerLimit ){
     fLowerLimit = FindInterpolatedLimit(1-ConfidenceLevel(),true);
  } else {
     fLowerLimit = GetXValue( FindClosestPointIndex((1-ConfidenceLevel())) );
  }
  return fLowerLimit;
}

Double_t HypoTestInverterResult::UpperLimit()
{
   //std::cout << "finding point with cl = " << (1-ConfidenceLevel())/2 << endl;
  if (fFittedUpperLimit) return fUpperLimit;
  if ( fInterpolateUpperLimit ) {
     double target = 1.-ConfidenceLevel();
     fUpperLimit = FindInterpolatedLimit(target,false);
     // test now if another point exists 
  } else {
     fUpperLimit = GetXValue( FindClosestPointIndex((1-ConfidenceLevel())) );
  }
  return fUpperLimit;
}

Double_t HypoTestInverterResult::CalculateEstimatedError(double target)
{
  // Return an error estimate on the upper limit.  This is the error on
  // either CLs or CLsplusb divided by an estimate of the slope at this
  // point.
   
  if (ArraySize()==0) {
    std::cout << "Empty result \n";
    return 0;
  }

  if (ArraySize()<2) {
    std::cout << "not enough points to get the inverted interval\n";
  }
 
  // The graph contains the points sorted by their x-value
  HypoTestInverterPlot plot("plot", "", this);
  TGraphErrors* graph = plot.MakePlot();
  double* xs = graph->GetX();
  const double minX = xs[0];
  const double maxX = xs[ArraySize()-1];

  TF1 fct("fct", "exp([0] * x + [1] * x**2)", minX, maxX);
  graph->Fit(&fct,"Q");

  int index = FindClosestPointIndex(target);
  double m = fct.Derivative( GetXValue(index) );
  double theError = fabs( GetYError(index) / m);

  delete graph;

  return theError;
}


Double_t HypoTestInverterResult::LowerLimitEstimatedError()
{
  if (fFittedLowerLimit) return fLowerLimitError;

   //std::cout << "The HypoTestInverterResult::LowerLimitEstimatedError() function evaluates only a rought error on the upper limit. Be careful when using this estimation\n";
  if (fInterpolateLowerLimit) std::cout << "The lower limit was an interpolated results... in this case the error is even less reliable (the Y-error bars are currently not used in the interpolation)\n";

  return CalculateEstimatedError(1-ConfidenceLevel());
}


Double_t HypoTestInverterResult::UpperLimitEstimatedError()
{

  if (fFittedUpperLimit) return fUpperLimitError;

   //std::cout << "The HypoTestInverterResult::UpperLimitEstimatedError() function evaluates only a rought error on the upper limit. Be careful when using this estimation\n";
  if (fInterpolateUpperLimit) std::cout << "The upper limit was an interpolated results... in this case the error is even less reliable (the Y-error bars are currently not used in the interpolation)\n";

  return CalculateEstimatedError((1-ConfidenceLevel()));
}

SamplingDistribution *  HypoTestInverterResult::GetBackgroundTestStatDist(int index ) const { 
   // get the background test statistic distribution

   HypoTestResult * firstResult = (HypoTestResult*) fYObjects.At(index);
   if (!firstResult) return 0;
   return firstResult->GetBackGroundIsAlt() ? firstResult->GetAltDistribution() : firstResult->GetNullDistribution();
}

SamplingDistribution *  HypoTestInverterResult::GetSignalAndBackgroundTestStatDist(int index) const { 
   // get the signal and background test statistic distribution
   HypoTestResult * result = (HypoTestResult*) fYObjects.At(index);
   if (!result) return 0;
   return !result->GetBackGroundIsAlt() ? result->GetAltDistribution() : result->GetNullDistribution();       
}

SamplingDistribution *  HypoTestInverterResult::GetExpectedPValueDist(int index) const { 
   // get the expected p-value distribution at the scanned point index

   if (index < 0 || index >=  ArraySize() ) return 0; 

   if (fExpPValues.GetSize() == ArraySize()  ) {
      return (SamplingDistribution*)  fExpPValues.At(index)->Clone();
   }
   
   static bool useFirstB = false;
   // get the S+B distribution 
   int bIndex = (useFirstB) ? 0 : index;
 
   SamplingDistribution * bDistribution = GetBackgroundTestStatDist(bIndex);
   SamplingDistribution * sbDistribution = GetSignalAndBackgroundTestStatDist(index);
   if (!bDistribution || !sbDistribution) return 0;

   HypoTestResult * result = (HypoTestResult*) fYObjects.At(index);

   // create a new HypoTestResult
   HypoTestResult tempResult; 
   tempResult.SetPValueIsRightTail( result->GetPValueIsRightTail() );
   tempResult.SetBackgroundAsAlt( true);
   tempResult.SetNullDistribution( sbDistribution );
   tempResult.SetAltDistribution( bDistribution );

   std::vector<double> values(bDistribution->GetSize()); 
   for (int i = 0; i < bDistribution->GetSize(); ++i) { 
      tempResult.SetTestStatisticData( bDistribution->GetSamplingDistribution()[i] );
      values[i] = (fUseCLs) ? tempResult.CLs() : tempResult.CLsplusb();
   }
   return new SamplingDistribution("expected values","expected values",values);
}



SamplingDistribution *  HypoTestInverterResult::GetLimitDistribution(bool lower ) const { 
   // get the limit distribution (lower/upper depending on the flag)

   //std::cout << "Interpolate the upper limit between the 2 results closest to the target confidence level" << endl;

  if (ArraySize()<2) {
    std::cout << "Error: not enough points to get the inverted interval\n";
    return 0; 
  }

  ooccoutD(this,Eval) << "HypoTestInverterResult - computing  limit distribution...." << std::endl;



  double target = 1-fConfidenceLevel;
  std::vector<SamplingDistribution*> distVec(ArraySize() );
  for (unsigned int i = 0; i < distVec.size(); ++i) {
     distVec[i] =  GetExpectedPValueDist(i);
     // sort the distribution
     // hack (by calling InverseCDF(0) will sort the sampling distribution
     distVec[i]->InverseCDF(0);
  }

  // sort the values in x 
  std::vector<unsigned int> index(ArraySize() );
  TMath::SortItr(fXValues.begin(), fXValues.end(), index.begin(), false); 

  // SamplingDistribution * dist0 = distVec[index[0]];
  // SamplingDistribution * dist1 = distVec[index[1]];

  int n = distVec[0]->GetSize();
  std::vector<double> limits(n);
  // loop on the p values and find the limit for each expcted one 
  for (int j = 0; j < n; ++j ) {

     TGraph g(ArraySize() );
     int npoints = ArraySize();
     for (int k = 0; k < npoints ; ++k) { 
        g.SetPoint(k, GetXValue(index[k]), distVec[index[k]]->GetSamplingDistribution()[j] );
     }

     limits[j] = GetGraphX(g, target, lower);

  }

  for (unsigned int i = 0; i < distVec.size(); ++i) {
     delete distVec[i];
  }

  if (lower) 
     return new SamplingDistribution("Expected lower Limit","Expected lower limits",limits);
  else
     return new SamplingDistribution("Expected upper Limit","Expected upper limits",limits);
  
}


double  HypoTestInverterResult::GetExpectedLowerLimit(double nsig ) const {
   // Get the expected lower limit  
   // nsig is used to specify which expected value of the UpperLimitDistribution
   // For example 
   // nsig = 0 (default value) returns the expected value 
   // nsig = -1 returns the lower band value at -1 sigma 
   // nsig + 1 return the upper value
   return GetExpectedLimit(nsig, true);
} 

double  HypoTestInverterResult::GetExpectedUpperLimit(double nsig ) const {
   // Get the expected upper limit  
   // nsig is used to specify which expected value of the UpperLimitDistribution
   // For example 
   // nsig = 0 (default value) returns the expected value 
   // nsig = -1 returns the lower band value at -1 sigma 
   // nsig + 1 return the upper value
   return GetExpectedLimit(nsig, false);
} 


   
double  HypoTestInverterResult::GetExpectedLimit(double nsig, bool lower ) const {
   // get expected limit (lower/upper) depending on the flag 

   const int nEntries = ArraySize();
   TGraph  g(nEntries);   

   // sort the arrays based on the x values
   std::vector<unsigned int> index(nEntries);
   TMath::SortItr(fXValues.begin(), fXValues.end(), index.begin(), false);

   double p[1];
   double q[1];
   p[0] = ROOT::Math::normal_cdf(nsig,1);
   for (int j=0; j<nEntries; ++j) {
      int i = index[j]; // i is the order index 
      SamplingDistribution * s = GetExpectedPValueDist(i);
      const std::vector<double> & values = s->GetSamplingDistribution();
      double * x = const_cast<double *>(&values[0]); // need to change TMath::Quantiles
      TMath::Quantiles(values.size(), 1, x,q,p,false);
      g.SetPoint(j, fXValues[i], q[0] );
      delete s;
   }

   // interpolate the graph to obtain the limit
   double target = 1-fConfidenceLevel;
   return GetGraphX(g, target, lower);
}

