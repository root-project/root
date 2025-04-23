#ifndef T0RESULT_HH
#define T0RESULT_HH

#include "TObject.h"
#include "AthIndex.h"

class T0Result : public TObject {
public:
   T0Result() {};
  virtual ~T0Result() {};

  AthIndex index;

  double bkgt0, sbkgt0;
  double sigt0, ssigt0;
  double meant0, smeant0;
  double sigmat0, ssigmat0;
  double meanstationt0, meanmezzt0;
  double chisqt0;
  int ndft0;
  string minstatt0;
  bool atlimitt0[4];

  double bkgtmax, sbkgtmax;
  double sigtmax, ssigtmax;
  double meantmax, smeantmax;
  double sigmatmax, ssigmatmax;
  double meanstationtmax, meanmezztmax;
  double chisqtmax;
  int ndftmax;
  string minstattmax;
  bool atlimittmax[4];

  int samplesize;

  ClassDef(T0Result,1) // Result of the T0 fit
};

#endif
