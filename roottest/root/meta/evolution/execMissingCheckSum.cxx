#include "TFile.h"
#include "TClass.h"
#include "TObjArray.h"


class MyClass {
public:
   Int_t fValue;
};

#include <stdint.h>
#include <vector>

struct HcalFlagHFDigiTimeParam {
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

#include "cmsCond01.h"
#include "signedchar.h"

#ifdef __ROOTCLING__
// These two are also on by default since we build the library with ACLiC
// from this file, but still add them here from symetry reasons.
#pragma link C++ class HcalFlagHFDigiTimeParam+;
#pragma link C++ class WithLongLong+;

#pragma link C++ class L1GtCondition+;
#pragma link C++ class TQuality+;
#endif

void DropStreamerInfo(const char *name) {

   TClass *cl = TClass::GetClass(name);
   if (!cl) printf("Error: could not find class %s\n",name);
   else ((TObjArray*)cl->GetStreamerInfos())->RemoveAt(1);
}

void CheckFile(const char *filename, const char *objname, const char *objtype)
{
   TFile *f = new TFile(filename);
   if (!f->Get(objname)) printf("Error: could not read the object (1): %s %s in %s\n",objname,objtype,filename);
   DropStreamerInfo(objtype);
   if (!f->Get(objname)) printf("Error: could not read the object (2): %s %s in %s\n",objname,objtype,filename);

   delete f;
}

void execMissingCheckSum()
{
   CheckFile("missingCheckSum.root","obj","MyClass");
   CheckFile("missingCheckSum2.root","timeParam","HcalFlagHFDigiTimeParam");
   CheckFile("missingCheckSum2.root","withLL","WithLongLong");
   CheckFile("checksumReflexEnum_v5.root","cond","L1GtCondition");
   CheckFile("checksumSignedChar_v5.root","q","TQuality");
}

