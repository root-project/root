// @(#)root/test:$Id$
// Authors: David Gonzalez Maline November 2008
//          Martin Storø Nyfløtt  June 2017

#ifndef ROOT_STRESSHISTOGRAMGLOBAL_H
#define ROOT_STRESSHISTOGRAMGLOBAL_H

#include "TH2.h"
#include "TH3.h"
#include "THn.h"
#include "THnSparse.h"

#include "TProfile.h"
#include "TProfile2D.h"
#include "TProfile3D.h"

#include "TF1.h"
#include "TF2.h"
#include "TF3.h"

#include "Fit/SparseData.h"
#include "HFitInterface.h"

#include "Math/IntegratorOptions.h"

#include "TApplication.h"
#include "TBenchmark.h"
#include "Riostream.h"
#include "TRandom2.h"
#include "TFile.h"
#include "TClass.h"

#include "TROOT.h"

#include "gtest/gtest.h"
#include <sstream>
#include <cmath>

#define minRange 1
#define maxRange 5
#define minRebin 3
#define maxRebin 7
#define nEvents 1000
#define numberOfBins 10
#define defaultErrorLimit 1.E-10
#define refFileOption 1
//#define refFileName "http://root.cern.ch/files/stressHistogram.5.18.00.root"

// In case of deviation, the profiles' content will not work anymore
// try only for testing the statistics
#define centre_deviation 0.3

enum compareOptions { cmpOptNone = 0, cmpOptPrint = 1, cmpOptDebug = 2, cmpOptNoError = 4, cmpOptStats = 8 };

extern int defaultEqualOptions;

enum RefFileEnum { refFileRead = 1, refFileWrite = 2 };

// Methods for histogram comparisions (later implemented)
void FillVariableRange(Double_t v[numberOfBins + 1]);
void FillHistograms(TH1D &h1, TH1D &h2, Double_t c1 = 1.0, Double_t c2 = 1.0);
void FillProfiles(TProfile &p1, TProfile &p2, Double_t c1 = 1.0, Double_t c2 = 1.0);
int Equals(const char *msg, TH1D &h1, TH1D &h2, int options = 0, double ERRORLIMIT = defaultErrorLimit);
int Equals(const char *msg, TH2D &h1, TH2D &h2, int options = 0, double ERRORLIMIT = defaultErrorLimit);
int Equals(const char *msg, TH3D &h1, TH3D &h2, int options = 0, double ERRORLIMIT = defaultErrorLimit);
int Equals(const char *msg, THnBase &h1, THnBase &h2, int options = 0, double ERRORLIMIT = defaultErrorLimit);
int Equals(const char *msg, THnBase &h1, TH1 &h2, int options = 0, double ERRORLIMIT = defaultErrorLimit);

template <typename T, typename U>
int Equals(const char *msg, std::unique_ptr<T> &h1, std::unique_ptr<U> &h2, int options = 0,
           double ERRORLIMIT = defaultErrorLimit)
{
   return Equals(msg, *h1.get(), *h2.get(), options, ERRORLIMIT);
};

template <typename T, typename U>
::testing::AssertionResult HistogramsEquals(T &h1, U &h2, int options = 0, double ERRORLIMIT = defaultErrorLimit)
{
   int differences = Equals("", h1, h2, options, ERRORLIMIT);
   if (differences > 0) {
      return ::testing::AssertionFailure() << "Histograms has " << differences << " differences";
   } else {
      return ::testing::AssertionSuccess();
   }
}

int Equals(Double_t n1, Double_t n2, double ERRORLIMIT = defaultErrorLimit);
int CompareStatistics(TH1 &h1, TH1 &h2, bool debug, double ERRORLIMIT = defaultErrorLimit);
std::ostream &operator<<(std::ostream &out, TH1D &h);

Double_t function1D(Double_t x);

double gaus1d(const double *x, const double *p);
double gaus2d(const double *x, const double *p);
double gaus3d(const double *x, const double *p);

bool operator==(ROOT::Fit::BinData &bd1, ROOT::Fit::BinData &bd2);
bool operator==(ROOT::Fit::SparseData &sd1, ROOT::Fit::SparseData &sd2);

#endif // ROOT_STRESSHISTOGRAMGLOBAL_H
