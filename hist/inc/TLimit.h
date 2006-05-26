// @(#)root/hist:$Name:  $:$Id: TLimit.h,v 1.8 2006/01/15 22:03:51 brun Exp $
// Author: Christophe.Delaere@cern.ch   21/08/2002

#ifndef ROOT_TLimit
#define ROOT_TLimit

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TMath
#include "TMath.h"
#endif

#include "TVectorDfwd.h"

class TConfidenceLevel;
class TRandom;
class TLimitDataSource;
class TArrayD;
class TOrdCollection;
class TH1;

//____________________________________________________________________
//
// TLimit
//
// This class computes 95% Confidence Levels using a given statistic.
// By default, the build-in LogLikelihood is used.
//
// Implemented by C. Delaere from the mclimit code written by Tom Junk.
// reference: HEP-EX/9902006
// See http://cern.ch/thomasj/searchlimits/ecl.html for more details.
//____________________________________________________________________

class TLimit {
 protected:
   static bool Fluctuate(TLimitDataSource * input, TLimitDataSource * output, bool init,TRandom *, bool stat=false);
   inline static Double_t LogLikelihood(Double_t s, Double_t b, Double_t d) { return d * TMath::Log(1 + (s / b)); }

public:
   TLimit() {}
   virtual ~TLimit() {}
   static TConfidenceLevel *ComputeLimit(TLimitDataSource * data,
                                         Int_t nmc =50000,
                                         bool stat = false,
                                         TRandom * generator = 0,
                                         Double_t(*statistic) (Double_t, Double_t,Double_t) = &(TLimit::LogLikelihood));
   static TConfidenceLevel *ComputeLimit(Double_t s, Double_t b, Int_t d,
                                         Int_t nmc =50000,
                                         bool stat = false,
                                         TRandom * generator = 0,
                                         Double_t(*statistic) (Double_t, Double_t,Double_t) = &(TLimit::LogLikelihood));
   static TConfidenceLevel *ComputeLimit(Double_t s, Double_t b, Int_t d,
                                         TVectorD* se, TVectorD* be, TObjArray*,
                                         Int_t nmc =50000,
                                         bool stat = false,
                                         TRandom * generator = 0,
                                         Double_t(*statistic) (Double_t, Double_t,Double_t) = &(TLimit::LogLikelihood));
   static TConfidenceLevel *ComputeLimit(TH1* s, TH1* b, TH1* d,
                                         Int_t nmc =50000,
                                         bool stat = false,
                                         TRandom * generator = 0,
                                         Double_t(*statistic) (Double_t, Double_t,Double_t) = &(TLimit::LogLikelihood));
   static TConfidenceLevel *ComputeLimit(TH1* s, TH1* b, TH1* d,
                                         TVectorD* se, TVectorD* be, TObjArray*,
                                         Int_t nmc =50000,
                                         bool stat = false,
                                         TRandom * generator = 0,
                                         Double_t(*statistic) (Double_t, Double_t,Double_t) = &(TLimit::LogLikelihood));
 private:
   static TArrayD *fgTable;              // a log table... just to speed up calculation
   static TOrdCollection *fgSystNames;   // Collection of systematics names
   ClassDef(TLimit, 2)          // Class to compute 95% CL limits
};

#endif

