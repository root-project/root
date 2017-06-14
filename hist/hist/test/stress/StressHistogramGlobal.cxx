// @(#)root/test:$Id$
// Authors: David Gonzalez Maline November 2008
//          Martin Storø Nyfløtt  June 2017

#include <sstream>
#include <cmath>

#include "TH2.h"
#include "TH3.h"
#include "THn.h"
#include "THnSparse.h"

#include "TProfile.h"
#include "TF1.h"
#include "HFitInterface.h"

#include "Riostream.h"
#include "TRandom2.h"
#include "TFile.h"
#include "TClass.h"

#include "gtest/gtest.h"

#include "StressHistogramGlobal.h"

using namespace std;

int defaultEqualOptions = 0;

TRandom2 r;

std::ostream &operator<<(std::ostream &out, TH1D &h)
{
   out << h.GetName() << ": [" << h.GetBinContent(1);
   for (Int_t i = 1; i < h.GetNbinsX(); ++i) out << ", " << h.GetBinContent(i);
   out << "] ";

   return out;
}

void FillVariableRange(Double_t v[numberOfBins + 1])
{
   r.SetSeed(0);
   // Double_t v[numberOfBins+1];
   Double_t minLimit = (maxRange - minRange) / (numberOfBins * 2);
   Double_t maxLimit = (maxRange - minRange) * 4 / (numberOfBins);
   v[0] = 0;
   for (Int_t i = 1; i < numberOfBins + 1; ++i) {
      Double_t limit = r.Uniform(minLimit, maxLimit);
      v[i] = v[i - 1] + limit;
   }

   Double_t k = (maxRange - minRange) / v[numberOfBins];
   for (Int_t i = 0; i < numberOfBins + 1; ++i) {
      v[i] = v[i] * k + minRange;
   }
}

void FillHistograms(TH1D &h1, TH1D &h2, Double_t c1, Double_t c2)
{
   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1.Fill(value, c1);
      h2.Fill(value, c2);
   }
}

void FillProfiles(TProfile &p1, TProfile &p2, Double_t c1, Double_t c2)
{
   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p1.Fill(x, y, c1);
      p2.Fill(x, y, c2);
   }
}

// Methods for histogram comparisions

int Equals(const char *msg, THnBase &h1, THnBase &h2, int options, double ERRORLIMIT)
{
   options = options | defaultEqualOptions;
   bool print = options & cmpOptPrint;
   bool debug = options & cmpOptDebug;
   bool compareError = !(options & cmpOptNoError);

   int differents = 0;

   for (int i = 0; i <= h1.GetAxis(0)->GetNbins() + 1; ++i)
      for (int j = 0; j <= h1.GetAxis(1)->GetNbins() + 1; ++j)
         for (int h = 0; h <= h1.GetAxis(2)->GetNbins() + 1; ++h) {
            Double_t x = h1.GetAxis(0)->GetBinCenter(i);
            Double_t y = h1.GetAxis(1)->GetBinCenter(j);
            Double_t z = h1.GetAxis(2)->GetBinCenter(h);

            Int_t bin[3] = {i, j, h};

            if (debug) {
               std::cout << Equals(x, h2.GetAxis(0)->GetBinCenter(i), ERRORLIMIT) << " "
                         << Equals(y, h2.GetAxis(1)->GetBinCenter(j), ERRORLIMIT) << " "
                         << Equals(z, h2.GetAxis(2)->GetBinCenter(h), ERRORLIMIT) << " "
                         << "[" << x << "," << y << "," << z << "]: " << h1.GetBinContent(bin) << " +/- "
                         << h1.GetBinError(bin) << " | " << h2.GetBinContent(bin) << " +/- " << h2.GetBinError(bin)
                         << " | " << Equals(h1.GetBinContent(bin), h2.GetBinContent(bin), ERRORLIMIT) << " "
                         << Equals(h1.GetBinError(bin), h2.GetBinError(bin), ERRORLIMIT) << " " << differents << " "
                         << (fabs(h1.GetBinContent(bin) - h2.GetBinContent(bin))) << std::endl;
            }
            differents += Equals(x, h2.GetAxis(0)->GetBinCenter(i), ERRORLIMIT);
            differents += Equals(y, h2.GetAxis(1)->GetBinCenter(j), ERRORLIMIT);
            differents += Equals(z, h2.GetAxis(2)->GetBinCenter(h), ERRORLIMIT);
            differents += Equals(h1.GetBinContent(bin), h2.GetBinContent(bin), ERRORLIMIT);
            if (compareError) differents += Equals(h1.GetBinError(bin), h2.GetBinError(bin), ERRORLIMIT);
         }

   // Statistical tests:
   // No statistical tests possible for THnBase so far...
   //    if ( compareStats )
   //       differents += CompareStatistics( h1, h2, debug, ERRORLIMIT);

   if (print || debug) std::cout << msg << ": \t" << (differents ? "FAILED" : "OK") << std::endl;

   return differents;
}

int Equals(const char *msg, THnBase &s, TH1 &h2, int options, double ERRORLIMIT)
{
   options = options | defaultEqualOptions;
   bool print = options & cmpOptPrint;
   bool debug = options & cmpOptDebug;
   bool compareError = !(options & cmpOptNoError);

   int differents = 0;

   const int dim(s.GetNdimensions());
   if (dynamic_cast<TH3 *>(&h2)) {
      if (dim != 3) return 1;
   } else if (dynamic_cast<TH2 *>(&h2)) {
      if (dim != 2) return 1;
   } else if (dim != 1)
      return 1;

   TArray *array = dynamic_cast<TArray *>(&h2);
   if (!array) Fatal("Equals(const char* msg, THnBase* s, TH1* h2, int options, double ERRORLIMIT)", "NO ARRAY!");

   Int_t coord[3];
   for (Long64_t i = 0; i < s.GetNbins(); ++i) {
      Double_t v1 = s.GetBinContent(i, coord);
      Double_t err1 = s.GetBinError(coord);

      int bin = h2.GetBin(coord[0], coord[1], coord[2]);
      Double_t v2 = h2.GetBinContent(bin);
      Double_t err2 = h2.GetBinError(bin);

      differents += Equals(v1, v2, ERRORLIMIT);
      if (compareError) differents += Equals(err1, err2, ERRORLIMIT);
   }

   for (Long64_t i = 0; i < array->GetSize(); ++i) {
      h2.GetBinXYZ(i, coord[0], coord[1], coord[2]);

      Double_t v1 = s.GetBinContent(coord);
      Double_t err1 = s.GetBinError(coord);

      Double_t v2 = h2.GetBinContent(i);
      Double_t err2 = h2.GetBinError(i);

      differents += Equals(v1, v2, ERRORLIMIT);
      if (compareError) differents += Equals(err1, err2, ERRORLIMIT);
   }

   if (print || debug) std::cout << msg << ": \t" << (differents ? "FAILED" : "OK") << std::endl;

   return differents;
}

int Equals(const char *msg, TH3D &h1, TH3D &h2, int options, double ERRORLIMIT)
{
   options = options | defaultEqualOptions;
   bool print = options & cmpOptPrint;
   bool debug = options & cmpOptDebug;
   bool compareError = !(options & cmpOptNoError);
   bool compareStats = options & cmpOptStats;

   int differents = (&h1 == &h2); // Check they are not the same histogram!
   if (debug) {
      std::cout << static_cast<void *>(&h1) << " " << static_cast<void *>(&h2) << " " << (&h1 == &h2) << " " << differents
                << std::endl;
   }

   for (int i = 0; i <= h1.GetNbinsX() + 1; ++i)
      for (int j = 0; j <= h1.GetNbinsY() + 1; ++j)
         for (int h = 0; h <= h1.GetNbinsZ() + 1; ++h) {
            Double_t x = h1.GetXaxis()->GetBinCenter(i);
            Double_t y = h1.GetYaxis()->GetBinCenter(j);
            Double_t z = h1.GetZaxis()->GetBinCenter(h);

            if (debug) {
               std::cout << Equals(x, h2.GetXaxis()->GetBinCenter(i), ERRORLIMIT) << " "
                         << Equals(y, h2.GetYaxis()->GetBinCenter(j), ERRORLIMIT) << " "
                         << Equals(z, h2.GetZaxis()->GetBinCenter(h), ERRORLIMIT) << " "
                         << "[" << x << "," << y << "," << z << "]: " << h1.GetBinContent(i, j, h) << " +/- "
                         << h1.GetBinError(i, j, h) << " | " << h2.GetBinContent(i, j, h) << " +/- "
                         << h2.GetBinError(i, j, h) << " | "
                         << Equals(h1.GetBinContent(i, j, h), h2.GetBinContent(i, j, h), ERRORLIMIT) << " "
                         << Equals(h1.GetBinError(i, j, h), h2.GetBinError(i, j, h), ERRORLIMIT) << " " << differents
                         << " " << (fabs(h1.GetBinContent(i, j, h) - h2.GetBinContent(i, j, h))) << std::endl;
            }
            differents += (bool)Equals(x, h2.GetXaxis()->GetBinCenter(i), ERRORLIMIT);
            differents += (bool)Equals(y, h2.GetYaxis()->GetBinCenter(j), ERRORLIMIT);
            differents += (bool)Equals(z, h2.GetZaxis()->GetBinCenter(h), ERRORLIMIT);
            differents += (bool)Equals(h1.GetBinContent(i, j, h), h2.GetBinContent(i, j, h), ERRORLIMIT);
            if (compareError)
               differents += (bool)Equals(h1.GetBinError(i, j, h), h2.GetBinError(i, j, h), ERRORLIMIT);
         }

   // Statistical tests:
   if (compareStats) differents += CompareStatistics(h1, h2, debug, ERRORLIMIT);

   if (print || debug) std::cout << msg << ": \t" << (differents ? "FAILED" : "OK") << std::endl;

   return differents;
}

::testing::AssertionResult HistogramsEquals(TH2D &h1, TH2D &h2, int options, double ERRORLIMIT)
{
   int differences = Equals("", h1, h2, options, ERRORLIMIT);
   if (differences > 0) {
      return ::testing::AssertionFailure() << "Histograms has " << differences << " differences";
   } else {
      return ::testing::AssertionSuccess();
   }
}

::testing::AssertionResult HistogramsEquals(TH1D &h1, TH1D &h2, int options, double ERRORLIMIT)
{
   int differences = Equals("", h1, h2, options, ERRORLIMIT);
   if (differences > 0) {
      return ::testing::AssertionFailure() << "Histograms has " << differences << " differences";
   } else {
      return ::testing::AssertionSuccess();
   }
}

::testing::AssertionResult HistogramsEquals(TH3D &h1, TH3D &h2, int options, double ERRORLIMIT)
{
   int differences = Equals("", h1, h2, options, ERRORLIMIT);
   if (differences > 0) {
      return ::testing::AssertionFailure() << "Histograms has " << differences << " differences";
   } else {
      return ::testing::AssertionSuccess();
   }
}

::testing::AssertionResult HistogramsEquals(THnBase &h1, THnBase &h2, int options, double ERRORLIMIT)
{
   int differences = Equals("", h1, h2, options, ERRORLIMIT);
   if (differences > 0) {
      return ::testing::AssertionFailure() << "Histograms has " << differences << " differences";
   } else {
      return ::testing::AssertionSuccess();
   }
}

::testing::AssertionResult HistogramsEquals(THnBase &h1, TH1 &h2, int options, double ERRORLIMIT)
{
   int differences = Equals("", h1, h2, options, ERRORLIMIT);
   if (differences > 0) {
      return ::testing::AssertionFailure() << "Histograms has " << differences << " differences";
   } else {
      return ::testing::AssertionSuccess();
   }
}

int Equals(const char *msg, TH2D &h1, TH2D &h2, int options, double ERRORLIMIT)
{
   options = options | defaultEqualOptions;
   bool print = options & cmpOptPrint;
   bool debug = options & cmpOptDebug;
   bool compareError = !(options & cmpOptNoError);
   bool compareStats = options & cmpOptStats;

   int differents = (&h1 == &h2); // Check they are not the same histogram!
   if (debug) {
      std::cout << static_cast<void *>(&h1) << " " << static_cast<void *>(&h2) << " " << (&h1 == &h2) << " " << differents
                << std::endl;
   }

   for (int i = 0; i <= h1.GetNbinsX() + 1; ++i)
      for (int j = 0; j <= h1.GetNbinsY() + 1; ++j) {
         Double_t x = h1.GetXaxis()->GetBinCenter(i);
         Double_t y = h1.GetYaxis()->GetBinCenter(j);

         if (debug) {
            std::cout << Equals(x, h2.GetXaxis()->GetBinCenter(i), ERRORLIMIT) << " "
                      << Equals(y, h2.GetYaxis()->GetBinCenter(j), ERRORLIMIT) << " "
                      << "[" << x << "," << y << "]: " << h1.GetBinContent(i, j) << " +/- " << h1.GetBinError(i, j)
                      << " | " << h2.GetBinContent(i, j) << " +/- " << h2.GetBinError(i, j) << " | "
                      << Equals(h1.GetBinContent(i, j), h2.GetBinContent(i, j), ERRORLIMIT) << " "
                      << Equals(h1.GetBinError(i, j), h2.GetBinError(i, j), ERRORLIMIT) << " " << differents << " "
                      << (fabs(h1.GetBinContent(i, j) - h2.GetBinContent(i, j))) << std::endl;
         }
         differents += (bool)Equals(x, h2.GetXaxis()->GetBinCenter(i), ERRORLIMIT);
         differents += (bool)Equals(y, h2.GetYaxis()->GetBinCenter(j), ERRORLIMIT);
         differents += (bool)Equals(h1.GetBinContent(i, j), h2.GetBinContent(i, j), ERRORLIMIT);
         if (compareError) differents += (bool)Equals(h1.GetBinError(i, j), h2.GetBinError(i, j), ERRORLIMIT);
      }

   // Statistical tests:
   if (compareStats) differents += CompareStatistics(h1, h2, debug, ERRORLIMIT);

   if (print || debug) std::cout << msg << ": \t" << (differents ? "FAILED" : "OK") << std::endl;

   return differents;
}

int Equals(const char *msg, TH1D &h1, TH1D &h2, int options, double ERRORLIMIT)
{
   options = options | defaultEqualOptions;
   bool print = options & cmpOptPrint;
   bool debug = options & cmpOptDebug;
   bool compareError = !(options & cmpOptNoError);
   bool compareStats = options & cmpOptStats;

   int differents = (&h1 == &h2); // Check they are not the same histogram!
   if (debug) {
      std::cout << static_cast<void *>(&h1) << " " << static_cast<void *>(&h2) << " " << (&h1 == &h2) << " " << differents
                << std::endl;
   }

   // check axis

   differents += (bool)Equals(h1.GetXaxis()->GetNbins(), h2.GetXaxis()->GetNbins());
   if (debug) {
      cout << "Nbins  = " << h1.GetXaxis()->GetNbins() << " |  " << h2.GetXaxis()->GetNbins() << " | " << differents
           << std::endl;
   }

   differents += (bool)Equals(h1.GetXaxis()->GetXmin(), h2.GetXaxis()->GetXmin());
   if (debug) {
      cout << "Xmin   = " << h1.GetXaxis()->GetXmin() << " |  " << h2.GetXaxis()->GetXmin() << " | " << differents
           << std::endl;
   }

   differents += (bool)Equals(h1.GetXaxis()->GetXmax(), h2.GetXaxis()->GetXmax());
   if (debug) {
      cout << "Xmax   = " << h1.GetXaxis()->GetXmax() << " |  " << h2.GetXaxis()->GetXmax() << endl;
   }

   for (int i = 0; i <= h1.GetNbinsX() + 1; ++i) {
      Double_t x = h1.GetXaxis()->GetBinCenter(i);

      differents += (bool)Equals(x, h2.GetXaxis()->GetBinCenter(i), ERRORLIMIT);
      differents += (bool)Equals(h1.GetBinContent(i), h2.GetBinContent(i), ERRORLIMIT);

      if (compareError) differents += (bool)Equals(h1.GetBinError(i), h2.GetBinError(i), ERRORLIMIT);

      if (debug) {
         std::cout << Equals(x, h2.GetXaxis()->GetBinCenter(i), ERRORLIMIT) << " [" << x
                   << "]: " << h1.GetBinContent(i) << " +/- " << h1.GetBinError(i) << " | " << h2.GetBinContent(i)
                   << " +/- " << h2.GetBinError(i) << " | "
                   << Equals(h1.GetBinContent(i), h2.GetBinContent(i), ERRORLIMIT) << " "
                   << Equals(h1.GetBinError(i), h2.GetBinError(i), ERRORLIMIT) << " " << differents << std::endl;
      }
   }

   // Statistical tests:
   if (compareStats) differents += CompareStatistics(h1, h2, debug, ERRORLIMIT);

   if (print || debug) std::cout << msg << ": \t" << (differents ? "FAILED" : "OK") << std::endl;

   return differents;
}

int Equals(Double_t n1, Double_t n2, double ERRORLIMIT)
{
   if (n1 != 0)
      return fabs(n1 - n2) > ERRORLIMIT * fabs(n1);
   else
      return fabs(n2) > ERRORLIMIT;
}

int CompareStatistics(TH1 &h1, TH1 &h2, bool debug, double ERRORLIMIT)
{
   int differents = 0;

   int pr = std::cout.precision(12);

   int precLevel = gErrorIgnoreLevel;
   // switch off Info mesaage from chi2 test
   if (!debug) gErrorIgnoreLevel = 1001;

   if (debug) h2.Print();

   std::string option = "WW OF UF";
   const char *opt = option.c_str();

   double chi_12 = h1.Chi2Test(&h2, opt);
   double chi_21 = h2.Chi2Test(&h1, opt);

   differents += (bool)Equals(chi_12, 1, ERRORLIMIT);
   differents += (bool)Equals(chi_21, 1, ERRORLIMIT);
   differents += (bool)Equals(chi_12, chi_21, ERRORLIMIT);
   if (debug) std::cout << "Chi2Test " << chi_12 << " " << chi_21 << " | " << differents << std::endl;

   if (!debug) gErrorIgnoreLevel = precLevel;

   // Mean
   differents += (bool)Equals(h1.GetMean(1), h2.GetMean(1), ERRORLIMIT);
   if (debug)
      std::cout << "Mean: " << h1.GetMean(1) << " " << h2.GetMean(1) << " | " << fabs(h1.GetMean(1) - h2.GetMean(1))
                << " " << differents << std::endl;

   // RMS
   differents += (bool)Equals(h1.GetRMS(1), h2.GetRMS(1), ERRORLIMIT);
   if (debug)
      std::cout << "RMS: " << h1.GetRMS(1) << " " << h2.GetRMS(1) << " | " << fabs(h1.GetRMS(1) - h2.GetRMS(1))
                << " " << differents << std::endl;

   // Number of Entries
   // check if is an unweighted histogram compare entries and  effective entries
   // otherwise only effective entries since entries do not make sense for an unweighted histogram
   // to check if is weighted - check if sum of weights == effective entries
   //   if (h1.GetEntries() == h1.GetEffectiveEntries() ) {
   double stats1[TH1::kNstat];
   h1.GetStats(stats1);
   double stats2[TH1::kNstat];
   h2.GetStats(stats2);
   // check first sum of weights
   differents += (bool)Equals(stats1[0], stats2[0], 100 * ERRORLIMIT);
   if (debug)
      std::cout << "Sum Of Weigths: " << stats1[0] << " " << stats2[0] << " | " << fabs(stats1[0] - stats2[0]) << " "
                << differents << std::endl;

   if (TMath::AreEqualRel(stats1[0], h1.GetEffectiveEntries(), 1.E-12)) {
      // unweighted histograms - check also number of entries
      differents += (bool)Equals(h1.GetEntries(), h2.GetEntries(), 100 * ERRORLIMIT);
      if (debug)
         std::cout << "Entries: " << h1.GetEntries() << " " << h2.GetEntries() << " | "
                   << fabs(h1.GetEntries() - h2.GetEntries()) << " " << differents << std::endl;
   }

   // Number of Effective Entries
   differents += (bool)Equals(h1.GetEffectiveEntries(), h2.GetEffectiveEntries(), 100 * ERRORLIMIT);
   if (debug)
      std::cout << "Eff Entries: " << h1.GetEffectiveEntries() << " " << h2.GetEffectiveEntries() << " | "
                << fabs(h1.GetEffectiveEntries() - h2.GetEffectiveEntries()) << " " << differents << std::endl;

   std::cout.precision(pr);

   return differents;
}

Double_t function1D(Double_t x)
{
   Double_t a = -1.8;

   return a * x;
}

bool normGaussfunc = true;

double gaus1d(const double *x, const double *p)
{
   return p[0] * TMath::Gaus(x[0], p[1], p[2], normGaussfunc);
}

double gaus2d(const double *x, const double *p)
{
   return p[0] * TMath::Gaus(x[0], p[1], p[2], normGaussfunc) * TMath::Gaus(x[1], p[3], p[4], normGaussfunc);
}

double gaus3d(const double *x, const double *p)
{
   return p[0] * TMath::Gaus(x[0], p[1], p[2], normGaussfunc) * TMath::Gaus(x[1], p[3], p[4], normGaussfunc) *
          TMath::Gaus(x[2], p[5], p[6], normGaussfunc);
}

int findBin(ROOT::Fit::SparseData &sd, const std::vector<double> &minRef, const std::vector<double> &maxRef,
            const double valRef, const double errorRef)
{
   const unsigned int ndim = sd.NDim();
   const unsigned int npoints = sd.NPoints();

   for (unsigned int i = 0; i < npoints; ++i) {
      std::vector<double> min(ndim);
      std::vector<double> max(ndim);
      double val;
      double error;
      sd.GetPoint(i, min, max, val, error);

      bool thisIsIt = true;
      thisIsIt &= !Equals(valRef, val, 1E-8);
      thisIsIt &= !Equals(errorRef, error, 1E-15);
      for (unsigned int j = 0; j < ndim && thisIsIt; ++j) {
         thisIsIt &= !Equals(minRef[j], min[j]);
         thisIsIt &= !Equals(maxRef[j], max[j]);
      }
      if (thisIsIt) {
         return i;
      }
   }

   return -1;
}

int findBin(ROOT::Fit::BinData &bd, const double *x)
{
   const unsigned int ndim = bd.NDim();
   const unsigned int npoints = bd.NPoints();

   for (unsigned int i = 0; i < npoints; ++i) {
      double value1 = 0, error1 = 0;
      const double *x1 = bd.GetPoint(i, value1, error1);

      bool thisIsIt = true;
      for (unsigned int j = 0; j < ndim; ++j) {
         thisIsIt &= fabs(x1[j] - x[j]) < 1E-15;
      }
      if (thisIsIt) {
         return i;
      }
   }

   return -1;
}

bool operator==(ROOT::Fit::BinData &bd1, ROOT::Fit::BinData &bd2)
{
   const unsigned int ndim = bd1.NDim();
   const unsigned int npoints = bd1.NPoints();
   const double eps = TMath::Limits<double>::Epsilon();

   for (unsigned int i = 0; i < npoints; ++i) {
      double value1 = 0, error1 = 0;
      const double *x1 = bd1.GetPoint(i, value1, error1);

      int bin = findBin(bd2, x1);

      double value2 = 0, error2 = 0;
      const double *x2 = bd2.GetPoint(bin, value2, error2);

      if (!TMath::AreEqualRel(value1, value2, eps)) return false;
      if (!TMath::AreEqualRel(error1, error2, eps)) return false;

      for (unsigned int j = 0; j < ndim; ++j)
         if (!TMath::AreEqualRel(x1[j], x2[j], eps)) return false;
   }
   return true;
}

bool operator==(ROOT::Fit::SparseData &sd1, ROOT::Fit::SparseData &sd2)
{
   const unsigned int ndim = sd1.NDim();

   const unsigned int npoints1 = sd1.NPoints();
   const unsigned int npoints2 = sd2.NPoints();

   bool equals = (npoints1 == npoints2);

   for (unsigned int i = 0; i < npoints1 && equals; ++i) {
      std::vector<double> min(ndim);
      std::vector<double> max(ndim);
      double val;
      double error;
      sd1.GetPoint(i, min, max, val, error);

      equals &= (findBin(sd2, min, max, val, error) >= 0);
   }

   for (unsigned int i = 0; i < npoints2 && equals; ++i) {
      std::vector<double> min(ndim);
      std::vector<double> max(ndim);
      double val;
      double error;
      sd2.GetPoint(i, min, max, val, error);

      equals &= (findBin(sd1, min, max, val, error) >= 0);
   }

   return equals;
}

int Equals(const char *msg, std::unique_ptr<TH1D> &h1, std::unique_ptr<TH1D> &h2, int options, double ERRORLIMIT)
{
   return Equals(msg, *h1.get(), *h2.get(), options, ERRORLIMIT);
}

int Equals(const char *msg, std::unique_ptr<TH2D> &h1, std::unique_ptr<TH2D> &h2, int options, double ERRORLIMIT)
{
   return Equals(msg, *h1.get(), *h2.get(), options, ERRORLIMIT);
}

int Equals(const char *msg, std::unique_ptr<TH3D> &h1, std::unique_ptr<TH3D> &h2, int options, double ERRORLIMIT)
{
   return Equals(msg, *h1.get(), *h2.get(), options, ERRORLIMIT);
}

int Equals(const char *msg, std::unique_ptr<THnBase> &h1, std::unique_ptr<THnBase> &h2, int options, double ERRORLIMIT)
{
   return Equals(msg, *h1.get(), *h2.get(), options, ERRORLIMIT);
}

int Equals(const char *msg, std::unique_ptr<THnBase> &h1, std::unique_ptr<TH1> &h2, int options, double ERRORLIMIT)
{
   return Equals(msg, *h1.get(), *h2.get(), options, ERRORLIMIT);
}

int Equals(const char *msg, std::unique_ptr<TProfile2D> &h1, std::unique_ptr<TProfile2D> &h2, int options, double ERRORLIMIT)
{
   return Equals(msg, *h1.get(), *h2.get(), options, ERRORLIMIT);
}

int Equals(const char *msg, std::unique_ptr<TProfile> &h1, std::unique_ptr<TH1D> &h2, int options, double ERRORLIMIT)
{
   return Equals(msg, *h1.get(), *h2.get(), options, ERRORLIMIT);
}

int Equals(const char *msg, std::unique_ptr<TProfile> &h1, std::unique_ptr<TProfile> &h2, int options, double ERRORLIMIT)
{
   return Equals(msg, *h1.get(), *h2.get(), options, ERRORLIMIT);
}

int Equals(const char *msg, std::unique_ptr<TProfile2D> &h1, std::unique_ptr<TH2D> &h2, int options, double ERRORLIMIT)
{
   return Equals(msg, *h1.get(), *h2.get(), options, ERRORLIMIT);
}