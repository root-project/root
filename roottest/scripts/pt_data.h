#include "TObject.h"
#include "TString.h"

class PTVal: public TObject {
public:
   PTVal(): fVal(), fZ(), fMean(), fVar(), fSumVal2() {}
   void Set(double val, const PTVal& prev, unsigned int newnum) {
      *this = prev;
      fVal = val;
      fSumVal2 += val * val;
      fMean = (fMean * (newnum - 1) + val) / newnum;
      double var2 = fSumVal2 / newnum - fMean * fMean;
      if (var2 > 0.) {
         fVar = sqrt(var2);
      } else {
         fVar = 0.;
      }
   }

   double fVal; // Measurement
   double fZ; // Deviation of fVal from fMean, in multiple of fVar
   double fMean; // Average, including fVal
   double fVar; // Variance, including fVal
   double fSumVal2; // Sum of squared measurements, including fVal
   ClassDef(PTVal, 1);
};

class PTData: public TObject {
public:
   PTData(): outlier(), svn(), statEntries(), historyThinningCounter()
   { PSet(); }

   PTData(const PTData& o):
      outlier(o.outlier),
      svn(o.svn),
      statEntries(o.statEntries),
      historyThinningCounter(o.historyThinningCounter),
      date(o.date),
      memleak(o.memleak),
      mempeak(o.mempeak),
      memalloc(o.memalloc),
      cputime(o.cputime)
   { PSet(); }

   PTData& operator=(const PTData& o) {
      outlier = o.outlier;
      svn = o.svn;
      statEntries = o.statEntries;
      historyThinningCounter = o.historyThinningCounter;
      date = o.date;
      memalloc = o.memalloc;
      memleak = o.memleak;
      mempeak = o.mempeak;
      cputime = o.cputime;
      PSet();
      return *this;
   }

   void PSet() {
      pval[0] = &memleak;
      pval[1] = &mempeak;
      pval[2] = &memalloc;
      pval[3] = &cputime;
   }

   int outlier; // memalloc == 1 | memleak == 2 | mempeak == 4 | cputime == 8
   unsigned int svn; // ROOT svn revision
   unsigned int statEntries; // number of measurements in averages etc, incl current
   unsigned int historyThinningCounter; // counter for deletion of old entries
   TString date;
   PTVal memleak;
   PTVal mempeak;
   PTVal memalloc;
   PTVal cputime;

   PTVal* pval[4]; //!

   ClassDef(PTData,1)
}; 
    
