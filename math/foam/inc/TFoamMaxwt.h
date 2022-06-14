// @(#)root/foam:$Id$
// Author: S. Jadach <mailto:Stanislaw.jadach@ifj.edu.pl>, P.Sawicki <mailto:Pawel.Sawicki@ifj.edu.pl>

#ifndef ROOT_TFoamMaxwt
#define ROOT_TFoamMaxwt

#include "TObject.h"

class TH1D;


class TFoamMaxwt : public TObject {
private:
   Double_t  fNent;      ///< No. of MC events
   Int_t     fnBin;      ///< No. of bins on the weight distribution
   Double_t  fwmax;      ///< Maximum analyzed weight
public:
   TH1D   *fWtHst1;      ///< Histogram of the weight wt
   TH1D   *fWtHst2;      ///< Histogram of wt filled with wt

public:
   TFoamMaxwt();                            // NOT IMPLEMENTED (NEVER USED)
   TFoamMaxwt(Double_t, Int_t);             // Principal Constructor
   TFoamMaxwt(TFoamMaxwt &From);            // Copy constructor
   ~TFoamMaxwt() override;                   // Destructor
   void Reset();                            // Reset
   TFoamMaxwt& operator=(const TFoamMaxwt &);    // operator =
   void Fill(Double_t);
   void Make(Double_t, Double_t&);
   void GetMCeff(Double_t, Double_t&, Double_t&);  // get MC efficiency= <w>/wmax

   ClassDefOverride(TFoamMaxwt,1); //Controlling of the MC weight (maximum weight)
};
#endif
