// @(#)root/hist:$Name:  $:$Id: TLimit.h,v 1.2 2003/03/21 14:53:49 brun Exp $
// Author: Christophe.Delaere@cern.ch   21/08/2002

#ifndef ROOT_TLimit
#define ROOT_TLimit

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TMath
#include "TMath.h"
#endif

class TConfidenceLevel;
class TRandom;
class TLimitDataSource;
class TArrayD;
class TOrdCollection;

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
 public:
   TLimit() {}
   virtual ~TLimit() {}
   static TConfidenceLevel *ComputeLimit(TLimitDataSource * data,
                                         Int_t nmc =50000, 
                                         bool stat = false,
					 TRandom * generator = NULL,
                                         Double_t(*statistic) (Double_t, Double_t,Double_t) =&(TLimit::LogLikelihood));
 protected:
   static TLimitDataSource *Fluctuate(TLimitDataSource * input, bool init,TRandom *, bool stat=false);
   inline static Double_t LogLikelihood(Double_t s, Double_t b, Double_t d) { return d * TMath::Log(1 + (s / b)); }
 private:
   static TArrayD *fgTable;              // a log table... just to speed up calculation
   static TOrdCollection *fgSystNames;   // Collection of systematics names
   ClassDef(TLimit, 2)          // Class to compute 95% CL limits
};

#endif

