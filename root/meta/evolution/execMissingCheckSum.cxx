#include "TFile.h"
#include "TClass.h"

class MyClass {
public:
   Int_t fValue;
};

#include <stdint.h>
#include <vector>

struct HcalFlagHFDigiTimeParam{
   HcalFlagHFDigiTimeParam() : mId(0), mHFdigiflagFirstSample(0), mHFdigiflagSamplesToAdd(0),mHFdigiflagExpectedPeak(0),mHFdigiflagMinEthreshold(0.0) {}

   uint32_t mId; // detector ID
   uint32_t mHFdigiflagFirstSample;         // first sample used in NTS calculation
   uint32_t mHFdigiflagSamplesToAdd;        // # of sampels to use in NTS calculation
   uint32_t mHFdigiflagExpectedPeak;        // expected peak position; used for calculating TS(peak)
   double    mHFdigiflagMinEthreshold;       // minimum energy for flagged rechit
   std::vector<double> mHFdigiflagCoefficients; // coefficients used to parameterize TS(peak)/NTS threshold:  [0]-exp([1]+[2]*E+....)
};

struct WithLongLong {
   WithLongLong() : mValue (0), mIndex(0) {}

   long long mValue;
   unsigned long long mIndex;
};

void execMissingCheckSum()
{
   TFile *f = new TFile("missingCheckSum.root");
   if (!f->Get("obj")) printf("Error: could not read the object (1)\n");
   ((TObjArray*)TClass::GetClass("MyClass")->GetStreamerInfos())->RemoveAt(1);
   if (!f->Get("obj")) printf("Error: could not read the object (2)\n");
   
   f = new TFile("missingCheckSum2.root");
   f->Get("timeParam");
  ((TObjArray*)TClass::GetClass("HcalFlagHFDigiTimeParam")->GetStreamerInfos())->RemoveAt(1);
   f->Get("timeParam");

   f->Get("withLL");
   ((TObjArray*)TClass::GetClass("WithLongLong")->GetStreamerInfos())->RemoveAt(1);
   f->Get("withLL");

}

