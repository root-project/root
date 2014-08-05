/// Simple program to benchmark vdt accuracy and cpu performance.
#include <vector>
#include <iostream>
#include <cmath> //for log2
#include <assert.h>
#include <limits>
#include <iostream>
#include <iomanip>
#include <map>
#include <string>

#include "vdt/vdtMath.h"

#include "TStopwatch.h"
#include "TRandom3.h"
#include "TH1F.h"
#include "TCanvas.h"
#include "TError.h"

const double RANGE=3000.;
const uint32_t SIZE= 16777216;

//------------------------------------------------------------------------------
// Not good for floating point, but just to get the bit difference
template <class T>
uint64_t fp2uint (T /*x*/)
{
   T::not_implemented; // "Static assert" in C++03
   return 0;
}

template <>
uint64_t fp2uint<double> (double x)
{
   return vdt::details::dp2uint64(x);
}

template <>
uint64_t fp2uint<float> (float x)
{
   return vdt::details::sp2uint32(x);
}

//------------------------------------------------------------------------------
/// Returns most significative different bit
template <class T>
inline uint32_t diffbit(const T a,const T b )
{
   uint64_t ia = fp2uint<T>(a);
   uint64_t ib = fp2uint<T>(b);
   uint64_t c = ia>ib? ia-ib : ib-ia;
   return log2(c)+1;
}

//------------------------------------------------------------------------------
// This allows to vectorise on very modern compilers (>=gcc 4.7)
// Note how the templating mechanism allows the compiler to *inline* the
// function. It is much more efficient than a void ptr or std::function!
template <typename T, typename F>
inline void calculateValues(F mathFunc,
                            const std::vector<T>& inputVector,
                            std::vector<T>& outputVector)
{

   const uint32_t size = inputVector.size();

   for (unsigned int i=0;i<size;++i){
      outputVector[i]=mathFunc(inputVector[i]);
   }

}

//------------------------------------------------------------------------------
template <typename T>
void compareOutputs(const std::vector<T>& inputVector1,
                    const std::vector<T>& inputVector2,
                    std::vector<uint32_t>& outputVector)
{
   assert(inputVector1.size()==inputVector2.size() &&
          inputVector1.size()==outputVector.size());

   const uint32_t size = inputVector1.size();

   for (unsigned int i=0;i<size;++i)
      outputVector[i]=diffbit(inputVector1[i],inputVector2[i]);
}

//------------------------------------------------------------------------------

enum rangeType {kReal,
                kExp,
                kExpf,
                kRealPlus,
                km1p1};

template <typename T>
void fillRandom(std::vector<T>& randomV,
                const rangeType theRangeType)
{
   // Yeah, well, maybe it can be done better. But this is not a tutorial about
   // random generation!
   const uint32_t size=randomV.size();
   static TRandom3 rndmGenerator(123);
   T* arr = &(randomV[0]);
   rndmGenerator.RndmArray(size,arr);
   if (kReal == theRangeType )     for (uint32_t i=0;i<size;++i) randomV[i]=(randomV[i]-0.5)*2*RANGE;
   if (kExp == theRangeType )     for (uint32_t i=0;i<size;++i) randomV[i]=(randomV[i]-0.5)*2*705.;
   if (kExpf == theRangeType )     for (uint32_t i=0;i<size;++i) randomV[i]=(randomV[i]-0.5)*2*85.;
   if (kRealPlus == theRangeType ) for (uint32_t i=0;i<size;++i) randomV[i]=randomV[i]*RANGE+0.000001;
   if (km1p1 == theRangeType )     for (uint32_t i=0;i<size;++i) randomV[i]=(randomV[i]-0.5)*2;

}

//------------------------------------------------------------------------------

template<typename T>
void treatBinDiffHisto(TH1F& histo,
                       const std::vector<T>& VDTVals,
                       const std::vector<T>& SystemVals)
{
   const uint32_t size = VDTVals.size();
   std::vector<uint32_t> diff(size);
   compareOutputs(VDTVals,SystemVals,diff);
   uint32_t theDiff=0;
   for (uint32_t i =0;i<size;++i){
      theDiff=diff[i];
      histo.Fill(theDiff);
   }
}

//------------------------------------------------------------------------------
template <typename T, typename F>
inline double measureTiming(F mathFunc,
                            const std::vector<T>& inputVector,
                            std::vector<T>& outputVector)
{
   TStopwatch timer;
   timer.Start();
   calculateValues<T>(mathFunc,inputVector,outputVector);
   timer.Stop();
   return timer.RealTime();
}

//------------------------------------------------------------------------------

   /*
   Reference values on a Intel(R) Core(TM) i7 CPU 950  @ 3.07GHz
   gcc 4.8.2 -Ofast

      Expf  0.110623  0.821863         7.4294        3.35636              6
      Sinf  0.151893   9.83798        64.7692       0.235055              9
      Cosf   0.11277   9.87508        87.5683       0.234271              8
      Tanf  0.167273   9.84792        58.8733       0.519784              9
      Atanf 0.0855272  0.529288        6.18854       0.366963              2
      Logf  0.114023  0.465541        4.08287       0.270999              2
   Isqrtf 0.0328619  0.275043        8.36965        4.35237              7
      Asinf 0.0958891  0.415733        4.33556        0.60167              3
      Acosf  0.099067  0.470179        4.74607       0.480427             10
      Exp  0.204585   0.64904        3.17247       0.137142              2
      Sin  0.327537    1.5099        4.60986       0.253579              2
      Cos  0.299601    1.5038        5.01933       0.253664              2
      Tan  0.276369   2.13009         7.7074       0.351065              5
      Atan  0.355532  0.902413         2.5382       0.326134              2
      Log  0.244172   1.25513        5.14034       0.385112              2
      Isqrt  0.167836  0.283752        1.69065       0.453692              2
      Asin   0.40379  0.869315        2.15289       0.318644              2
      Acos  0.392566  0.864706         2.2027       0.391922             11
   */

   // The reference values: speedup and accuracy. Some contingency is given
struct staticInitHelper{
   std::map<std::string, std::pair<float,uint32_t> > referenceValues;

   staticInitHelper()
   {
      referenceValues["Expf"]  =  std::make_pair(1.f,8);
      referenceValues["Sinf"]  =  std::make_pair(1.f,11);
      referenceValues["Cosf"]  =  std::make_pair(1.f,10);
      referenceValues["Tanf"]  =  std::make_pair(1.f,11);
      referenceValues["Atanf"] =  std::make_pair(1.f,4);
      referenceValues["Logf"]  =  std::make_pair(1.f,4);
      referenceValues["Isqrtf"]=  std::make_pair(1.f,9);
      referenceValues["Asinf"] =  std::make_pair(1.f,5);
      referenceValues["Acosf"] =  std::make_pair(1.f,12);
      referenceValues["Exp"]   =  std::make_pair(1.f,4);
      referenceValues["Sin"]   =  std::make_pair(1.f,4);
      referenceValues["Cos"]   =  std::make_pair(1.f,4);
      referenceValues["Tan"]   =  std::make_pair(1.f,7);
      referenceValues["Atan"]  =  std::make_pair(1.f,4);
      referenceValues["Log"]   =  std::make_pair(1.f,4);
      referenceValues["Isqrt"] =  std::make_pair(.4f,4);  // Fix fluctuation on x86_64-slc5-gcc47
      referenceValues["Asin"]  =  std::make_pair(1.f,4);
      referenceValues["Acos"]  =  std::make_pair(1.f,13);
  }
} gbl;

template <typename T, typename F1, typename F2>
inline void compareFunctions(const std::string& label,
                             F1 vdtFunc,
                             F2 systemFunc,
                             const std::vector<T>& inputVector,
                             std::vector<T>& outputVectorVDT,
                             std::vector<T>& outputVectorSystem,
                             float& speedup,
                             uint32_t& maxdiffBit,
                             TH1F& histo)
{
   double timeVdt = measureTiming<T>(vdtFunc,inputVector,outputVectorVDT);
   double timeSystem = measureTiming<T>(systemFunc,inputVector,outputVectorSystem);
   std::string name(label);
   std::string title(label);
   title+=";Diff. Bit;#";
   histo.Reset();
   histo.SetName(label.c_str());
   histo.SetTitle(title.c_str());
   treatBinDiffHisto(histo,outputVectorVDT,outputVectorSystem);
   double meandiffBit = histo.GetMean();
   maxdiffBit = 0;
   const uint32_t xmax=histo.GetXaxis()->GetXmax();

   for (uint32_t i=1;i<=xmax;i++){
      if ( histo.GetBinContent(i) > 0.f )
         maxdiffBit=i-1;
   }

   speedup = timeSystem/timeVdt ;

   std::cout << std::setw(8)
             << label << std::setw(10)
             << timeVdt << std::setw(10)
             << timeSystem << std::setw(15)
             << speedup << std::setw(15)
             << meandiffBit << std::setw(15)
             << maxdiffBit << std::endl;

   // Draw it
   TCanvas c;
   c.cd();
   histo.SetLineColor(kBlue);
   histo.SetLineWidth(2);
   histo.Draw("hist");
   c.SetLogy();
   name+=".png";
   Int_t oldErrorIgnoreLevel = gErrorIgnoreLevel; // we know we are printing..
   gErrorIgnoreLevel=1001;
   c.Print(name.c_str());
   gErrorIgnoreLevel=oldErrorIgnoreLevel;
}

//
void checkFunction(const std::string& label,float /*speedup*/, uint32_t maxdiffBit)
{
// Remove check on the speed as routinely this program is ran on virtual build nodes
// and several factors may cause fluctuations in the result.
//   if (gbl.referenceValues[label].first > speedup)
//      std::cerr << "Note " << label << " was slower than the system library.\n";
   if (gbl.referenceValues[label].second < maxdiffBit)
      std::cerr << "WARNING " << label << " is too inaccurate!\n";
}

//------------------------------------------------------------------------------
// Some helpers
inline float isqrtf(float x) {return 1.f/sqrt(x);};
inline double isqrt(double x) {return 1./sqrt(x);};

//------------------------------------------------------------------------------
// Artificially chunck the work to help the compiler manage everything.
// If all the content is moved to a single function, the vectorization breaks.
// Same holds for the check of speed and accuracy. Technically it could be part
// of compareFunctions.
void spStep1()
{
   std::vector<float> VDTVals(SIZE);
   std::vector<float> SystemVals(SIZE);
   std::vector<float> realNumbers(SIZE);
   TH1F histo("bitDiffHisto","willbechanged",32,0,32);

   float speedup;
   uint32_t maxdiffBit;

   fillRandom(realNumbers,kExpf);
   compareFunctions<float>("Expf",  vdt::fast_expf,    expf,   realNumbers, VDTVals, SystemVals, speedup, maxdiffBit, histo);
   checkFunction("Expf",speedup, maxdiffBit);

   fillRandom(realNumbers,kReal);
   compareFunctions<float>("Sinf",  vdt::fast_sinf,    sinf,   realNumbers, VDTVals, SystemVals, speedup, maxdiffBit, histo);
   checkFunction("Sinf",speedup, maxdiffBit);
   compareFunctions<float>("Cosf",  vdt::fast_cosf,    cosf,   realNumbers, VDTVals, SystemVals, speedup, maxdiffBit, histo);
   checkFunction("Cosf",speedup, maxdiffBit);
   compareFunctions<float>("Tanf",  vdt::fast_tanf,    tanf,   realNumbers, VDTVals, SystemVals, speedup, maxdiffBit, histo);
   checkFunction("Tanf",speedup, maxdiffBit);
   compareFunctions<float>("Atanf", vdt::fast_atanf,   atanf,  realNumbers, VDTVals, SystemVals, speedup, maxdiffBit, histo);
   checkFunction("Atanf",speedup, maxdiffBit);
}

void spStep2()
{
   std::vector<float> VDTVals(SIZE);
   std::vector<float> SystemVals(SIZE);
   std::vector<float> realNumbers(SIZE);
   TH1F histo("bitDiffHisto","willbechanged",32,0,32);

   float speedup;
   uint32_t maxdiffBit;

   fillRandom(realNumbers,kRealPlus);
   compareFunctions<float>("Logf",   vdt::fast_logf,   logf,   realNumbers, VDTVals, SystemVals, speedup, maxdiffBit, histo);
   checkFunction("Logf",speedup, maxdiffBit);
   compareFunctions<float>("Isqrtf", vdt::fast_isqrtf, isqrtf, realNumbers, VDTVals, SystemVals, speedup, maxdiffBit, histo);
   checkFunction("Isqrtf",speedup, maxdiffBit);
}

void spStep3()
{
   std::vector<float> VDTVals(SIZE);
   std::vector<float> SystemVals(SIZE);
   std::vector<float> realNumbers(SIZE);
   TH1F histo("bitDiffHisto","willbechanged",32,0,32);

   float speedup;
   uint32_t maxdiffBit;

   fillRandom(realNumbers,km1p1);
   compareFunctions<float>("Asinf",  vdt::fast_asinf,  asinf,  realNumbers, VDTVals, SystemVals, speedup, maxdiffBit, histo);
   checkFunction("Asinf",speedup, maxdiffBit);
   compareFunctions<float>("Acosf",  vdt::fast_acosf,  acosf,  realNumbers, VDTVals, SystemVals, speedup, maxdiffBit, histo);
   checkFunction("Acosf",speedup, maxdiffBit);
}

void dpStep1()
{
   std::vector<double> VDTVals(SIZE);
   std::vector<double> SystemVals(SIZE);
   std::vector<double> realNumbers(SIZE);
   TH1F histo("bitDiffHisto","willbechanged",64,0,64);

   float speedup;
   uint32_t maxdiffBit;

   fillRandom(realNumbers,kExp);
   compareFunctions<double>("Exp",  vdt::fast_exp,    exp,  realNumbers, VDTVals, SystemVals, speedup, maxdiffBit, histo);
   checkFunction("Exp",speedup, maxdiffBit);
   fillRandom(realNumbers,kReal);
   compareFunctions<double>("Sin",  vdt::fast_sin,    sin,   realNumbers, VDTVals, SystemVals, speedup, maxdiffBit, histo);
   checkFunction("Sin",speedup, maxdiffBit);
   compareFunctions<double>("Cos",  vdt::fast_cos,    cos,   realNumbers, VDTVals, SystemVals, speedup, maxdiffBit, histo);
   checkFunction("Cos",speedup, maxdiffBit);
   compareFunctions<double>("Tan",  vdt::fast_tan,    tan,   realNumbers, VDTVals, SystemVals, speedup, maxdiffBit, histo);
   checkFunction("Tan",speedup, maxdiffBit);
   compareFunctions<double>("Atan", vdt::fast_atan,   atan,  realNumbers, VDTVals, SystemVals, speedup, maxdiffBit, histo);
   checkFunction("Atan",speedup, maxdiffBit);
}

void dpStep2()
{
   std::vector<double> VDTVals(SIZE);
   std::vector<double> SystemVals(SIZE);
   std::vector<double> realNumbers(SIZE);
   TH1F histo("bitDiffHisto","willbechanged",64,0,64);

   float speedup;
   uint32_t maxdiffBit;

   fillRandom(realNumbers,kRealPlus);
   compareFunctions<double>("Log",   vdt::fast_log,   log,   realNumbers, VDTVals, SystemVals, speedup, maxdiffBit, histo);
   checkFunction("Log",speedup, maxdiffBit);
   compareFunctions<double>("Isqrt", vdt::fast_isqrt, isqrt, realNumbers, VDTVals, SystemVals, speedup, maxdiffBit, histo);
   checkFunction("Isqrt",speedup, maxdiffBit);
}

void dpStep3()
{
   std::vector<double> VDTVals(SIZE);
   std::vector<double> SystemVals(SIZE);
   std::vector<double> realNumbers(SIZE);
   TH1F histo("bitDiffHisto","willbechanged",64,0,64);

   float speedup;
   uint32_t maxdiffBit;

   fillRandom(realNumbers,km1p1);
   compareFunctions<double>("Asin",  vdt::fast_asin,  asin,  realNumbers, VDTVals, SystemVals, speedup, maxdiffBit, histo);
   checkFunction("Asin",speedup, maxdiffBit);
   compareFunctions<double>("Acos",  vdt::fast_acos,  acos,  realNumbers, VDTVals, SystemVals, speedup, maxdiffBit, histo);
   checkFunction("Acos",speedup, maxdiffBit);
}
//------------------------------------------------------------------------------

int main(){

   std::cout << "Test performed on " << SIZE << " random numbers\n"
             << std::setw(8)
             << "Name" << std::setw(10)
             << "VDT (s)" << std::setw(10)
             << "Sys (s)" << std::setw(15)
             << "Speedup" << std::setw(15)
             << "<diff Bit>" << std::setw(15)
             << "max(diff Bit)" << std::endl;

   // Single precision ----
   spStep1();
   spStep2();
   spStep3();

   // Double precision ----
   dpStep1();
   dpStep2();
   dpStep3();

   return 0;

}


















































