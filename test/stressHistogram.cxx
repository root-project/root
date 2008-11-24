#include <sstream>
#include <cmath>

#include "TH2.h"
#include "TH3.h"
#include "TH2.h"
#include "THnSparse.h"

#include "TProfile.h"
#include "TProfile2D.h"
#include "TProfile3D.h"

#include "TApplication.h"
#include "Riostream.h"
#include "TMath.h"
#include "TRandom2.h"
#include "TFile.h"

#include "TROOT.h"
#include <algorithm>

const unsigned int __DRAW__ = 0;

const Double_t minRange = 1;
const Double_t maxRange = 5;

const Double_t minRebin = 3;
const Double_t maxRebin = 7;

const int minBinValue = 1;
const int maxBinValue = 10;

const int nEvents = 1000;
const int numberOfBins = 10;

enum compareOptions {
   cmpOptDebug=1,
   cmpOptNoError=2,
   cmpOptStats=4
};

TRandom2 r;

typedef bool ( * pointer2Test) ();

// Methods for histogram comparisions (later implemented)

int equals(const char* msg, TH1D* h1, TH1D* h2, int options = 0, double ERRORLIMIT = 1E-15);
int equals(const char* msg, TH2D* h1, TH2D* h2, int options = 0, double ERRORLIMIT = 1E-15);
int equals(const char* msg, TH3D* h1, TH3D* h2, int options = 0, double ERRORLIMIT = 1E-15);
int equals(Double_t n1, Double_t n2, double ERRORLIMIT = 1E-15);
int compareStatistics( TH1* h1, TH1* h2, bool debug, double ERRORLIMIT = 1E-15);
ostream& operator<<(ostream& out, TH1D* h);
// old stresHistOpts.cxx file

bool testAdd1() 
{
   Double_t c1 = r.Rndm();
   Double_t c2 = r.Rndm();

   TH1D* h1 = new TH1D("t1D1-h1", "h1-Title", numberOfBins, minRange, maxRange);
   TH1D* h2 = new TH1D("t1D1-h2", "h2-Title", numberOfBins, minRange, maxRange);
   TH1D* h3 = new TH1D("t1D1-h3", "h3=c1*h1+c2*h2", numberOfBins, minRange, maxRange);

   h1->Sumw2();h2->Sumw2();h3->Sumw2();

   for ( Int_t e = 0; e < nEvents; ++e ) {
      Double_t value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(value,  1.0);
      h3->Fill(value, c1);
   }

   for ( Int_t e = 0; e < nEvents; ++e ) {
      Double_t value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h2->Fill(value,  1.0);
      h3->Fill(value, c2);
   }

   TH1D* h4 = new TH1D("t1D1-h4", "h4=c1*h1+h2*c2", numberOfBins, minRange, maxRange);
   h4->Add(h1, h2, c1, c2);

   bool ret = equals("Add1D1", h3, h4, cmpOptStats, 1E-13);
   delete h1;
   delete h2;
   delete h3;
   return ret;
}

bool testAdd2() 
{
   Double_t c2 = r.Rndm();

   TH1D* h5 = new TH1D("t1D2-h5", "h5=   h6+c2*h7", numberOfBins, minRange, maxRange);
   TH1D* h6 = new TH1D("t1D2-h6", "h6-Title", numberOfBins, minRange, maxRange);
   TH1D* h7 = new TH1D("t1D2-h7", "h7-Title", numberOfBins, minRange, maxRange);

   h5->Sumw2();h6->Sumw2();h7->Sumw2();

   for ( Int_t e = 0; e < nEvents; ++e ) {
      Double_t value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h6->Fill(value, 1.0);
      h5->Fill(value, 1.0);
   }

   for ( Int_t e = 0; e < nEvents; ++e ) {
      Double_t value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h7->Fill(value,  1.0);
      h5->Fill(value, c2);
   }

   h6->Add(h7, c2);
   
   bool ret = equals("Add1D2", h5, h6, cmpOptStats, 1E-13);
   delete h5;
   delete h7;
   return ret;
}

bool testAdd2D1() 
{
   Double_t c1 = r.Rndm();
   Double_t c2 = r.Rndm();

   TH2D* h1 = new TH2D("t2D1-h1", "h1", 
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange);

   TH2D* h2 = new TH2D("t2D1-h2", "h2", 
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange);

   TH2D* h3 = new TH2D("t2D1-h3", "h3=c1*h1+c2*h2", 
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange);

   h1->Sumw2();h2->Sumw2();h3->Sumw2();

   for ( Int_t e = 0; e < nEvents * nEvents; ++e ) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(x, y,  1.0);
      h3->Fill(x, y, c1);
   }

   for ( Int_t e = 0; e < nEvents * nEvents; ++e ) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h2->Fill(x, y,  1.0);
      h3->Fill(x, y, c2);
   }

   TH2D* h4 = new TH2D("t2D1-h4", "h4=c1*h1+c2*h2", 
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange);
   h4->Add(h1, h2, c1, c2);
   bool ret = equals("Add2D1", h3, h4, cmpOptStats , 1E-10);
   delete h1;
   delete h2;
   delete h3;
   return ret;
}

bool testAdd2D2() 
{
   Double_t c2 = r.Rndm();

   TH2D* h1 = new TH2D("t2D2-h1", "h1", 
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange);

   TH2D* h2 = new TH2D("t2D2-h2", "h2", 
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange);

   TH2D* h3 = new TH2D("t2D2-h3", "h3=h1+c2*h2", 
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange);

   h1->Sumw2();h2->Sumw2();h3->Sumw2();

   for ( Int_t e = 0; e < nEvents * nEvents; ++e ) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(x, y,  1.0);
      h3->Fill(x, y,  1.0);
   }

   for ( Int_t e = 0; e < nEvents * nEvents; ++e ) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h2->Fill(x, y,  1.0);
      h3->Fill(x, y, c2);
   }

   h1->Add(h2, c2);
   bool ret = equals("Add2D2", h3, h1, cmpOptStats, 1E-10);
   delete h2;
   delete h3;
   return ret;
}

bool testAdd3D1() 
{
   Double_t c1 = r.Rndm();
   Double_t c2 = r.Rndm();

   TH3D* h1 = new TH3D("t3D1-h1", "h1", 
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange);

   TH3D* h2 = new TH3D("t3D1-h2", "h2", 
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange);

   TH3D* h3 = new TH3D("t3D1-h3", "h3=c1*h1+c2*h2", 
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange);

   h1->Sumw2();h2->Sumw2();h3->Sumw2();

   for ( Int_t e = 0; e < nEvents * nEvents; ++e ) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(x, y, z,  1.0);
      h3->Fill(x, y, z, c1);
   }

   for ( Int_t e = 0; e < nEvents * nEvents; ++e ) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h2->Fill(x, y, z,  1.0);
      h3->Fill(x, y, z, c2);
   }

   TH3D* h4 = new TH3D("t3D1-h4", "h4=c1*h1+c2*h2", 
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange);
   h4->Add(h1, h2, c1, c2);
   bool ret = equals("Add3D1", h3, h4, cmpOptStats, 1E-10);
   delete h1;
   delete h2;
   delete h3;
   return ret;
}

bool testAdd3D2() 
{
   Double_t c2 = r.Rndm();

   TH3D* h1 = new TH3D("t3D2-h1", "h1", 
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange);

   TH3D* h2 = new TH3D("t3D2-h2", "h2", 
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange);

   TH3D* h3 = new TH3D("t3D2-h3", "h3=h1+c2*h2", 
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange);

   h1->Sumw2();h2->Sumw2();h3->Sumw2();

   for ( Int_t e = 0; e < nEvents * nEvents; ++e ) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(x, y, z,  1.0);
      h3->Fill(x, y, z,  1.0);
   }

   for ( Int_t e = 0; e < nEvents * nEvents; ++e ) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h2->Fill(x, y, z,  1.0);
      h3->Fill(x, y, z, c2);
   }

   h1->Add(h2, c2);
   bool ret = equals("Add3D2", h3, h1, cmpOptStats, 1E-10);
   delete h2;
   delete h3;
   return ret;
}

bool testMul1() 
{
   Double_t c1 = r.Rndm();
   Double_t c2 = r.Rndm();

   TH1D* h1 = new TH1D("m1D1-h1", "h1-Title", numberOfBins, minRange, maxRange);
   TH1D* h2 = new TH1D("m1D1-h2", "h2-Title", numberOfBins, minRange, maxRange);
   TH1D* h3 = new TH1D("m1D1-h3", "h3=c1*h1*c2*h2", numberOfBins, minRange, maxRange);

   h1->Sumw2();h2->Sumw2();h3->Sumw2();

   UInt_t seed = r.GetSeed();
   // For possible problems
   r.SetSeed(seed);
   for ( Int_t e = 0; e < nEvents; ++e ) {
      Double_t value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(value, 1.0);
   }

   for ( Int_t e = 0; e < nEvents; ++e ) {
      Double_t value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h2->Fill(value,  1.0);
      h3->Fill(value,  c1*c2*h1->GetBinContent( h1->GetXaxis()->FindBin(value) ) );
   }

   // h3 has to be filled again so that the erros are properly calculated
   r.SetSeed(seed);
   for ( Int_t e = 0; e < nEvents; ++e ) {
      Double_t value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h3->Fill(value,  c1*c2*h2->GetBinContent( h2->GetXaxis()->FindBin(value) ) );
   }

   // No the bin contents has to be reduced, as it was filled twice!
   for ( Int_t bin = 0; bin <= h3->GetNbinsX() + 1; ++bin ) {
      h3->SetBinContent(bin, h3->GetBinContent(bin) / 2 );
   }

   TH1D* h4 = new TH1D("m1D1-h4", "h4=h1*h2", numberOfBins, minRange, maxRange);
   h4->Multiply(h1, h2, c1, c2);

   bool ret = equals("Multiply1D1", h3, h4, cmpOptStats, 1E-14);
   delete h1;
   delete h2;
   delete h3;
   return ret;
}

bool testMul2() 
{
   TH1D* h1 = new TH1D("m1D2-h1", "h1-Title", numberOfBins, minRange, maxRange);
   TH1D* h2 = new TH1D("m1D2-h2", "h2-Title", numberOfBins, minRange, maxRange);
   TH1D* h3 = new TH1D("m1D2-h3", "h3=h1*h2", numberOfBins, minRange, maxRange);

   h1->Sumw2();h2->Sumw2();h3->Sumw2();

   UInt_t seed = r.GetSeed();
   // For possible problems
   r.SetSeed(seed);
   for ( Int_t e = 0; e < nEvents; ++e ) {
      Double_t value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(value, 1.0);
   }

   for ( Int_t e = 0; e < nEvents; ++e ) {
      Double_t value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h2->Fill(value,  1.0);
      h3->Fill(value,  h1->GetBinContent( h1->GetXaxis()->FindBin(value) ) );
   }

   r.SetSeed(seed);
   for ( Int_t e = 0; e < nEvents; ++e ) {
      Double_t value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h3->Fill(value,  h2->GetBinContent( h2->GetXaxis()->FindBin(value) ) );
   }

   for ( Int_t bin = 0; bin <= h3->GetNbinsX() + 1; ++bin ) {
      h3->SetBinContent(bin, h3->GetBinContent(bin) / 2 );
   }

   h1->Multiply(h2);

   bool ret = equals("Multiply1D2", h3, h1, cmpOptStats, 1E-14);
   delete h2;
   delete h3;
   return ret;
}

bool testMul2D1() 
{
   Double_t c1 = r.Rndm();
   Double_t c2 = r.Rndm();

   TH2D* h1 = new TH2D("m2D1-h1", "h1-Title", 
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange);
   TH2D* h2 = new TH2D("m2D1-h2", "h2-Title", 
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange);
   TH2D* h3 = new TH2D("m2D1-h3", "h3=c1*h1*c2*h2",
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange);

   h1->Sumw2();h2->Sumw2();h3->Sumw2();

   UInt_t seed = r.GetSeed();
   // For possible problems
   r.SetSeed(seed);
   for ( Int_t e = 0; e < nEvents * nEvents; ++e ) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(x, y, 1.0);
   }

   for ( Int_t e = 0; e < nEvents * nEvents; ++e ) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h2->Fill(x, y,  1.0);
      h3->Fill(x, y,  c1*c2*h1->GetBinContent( h1->GetXaxis()->FindBin(x),
                                               h1->GetYaxis()->FindBin(y) ) );
   }

   // h3 has to be filled again so that the erros are properly calculated
   r.SetSeed(seed);
   for ( Int_t e = 0; e < nEvents * nEvents; ++e ) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h3->Fill(x, y,  c1*c2*h2->GetBinContent( h2->GetXaxis()->FindBin(x),
                                               h2->GetYaxis()->FindBin(y) ) );
   }

   // No the bin contents has to be reduced, as it was filled twice!
   for ( Int_t i = 0; i <= h3->GetNbinsX() + 1; ++i ) {
      for ( Int_t j = 0; j <= h3->GetNbinsY() + 1; ++j ) {
         h3->SetBinContent(i, j, h3->GetBinContent(i, j) / 2 );
      }
   }

   TH2D* h4 = new TH2D("m2D1-h4", "h4=h1*h2", 
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange);
   h4->Multiply(h1, h2, c1, c2);

   bool ret = equals("Multiply2D1", h3, h4, cmpOptStats, 1E-12);
   delete h1;
   delete h2;
   delete h3;
   return ret;
}

bool testMul2D2() 
{
   TH2D* h1 = new TH2D("m2D2-h1", "h1-Title", 
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange);
   TH2D* h2 = new TH2D("m2D2-h2", "h2-Title", 
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange);
   TH2D* h3 = new TH2D("m2D2-h3", "h3=h1*h2",
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange);

   h1->Sumw2();h2->Sumw2();h3->Sumw2();

   UInt_t seed = r.GetSeed();
   // For possible problems
   r.SetSeed(seed);
   for ( Int_t e = 0; e < nEvents * nEvents; ++e ) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(x, y, 1.0);
   }

   for ( Int_t e = 0; e < nEvents * nEvents; ++e ) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h2->Fill(x, y,  1.0);
      h3->Fill(x, y,  h1->GetBinContent( h1->GetXaxis()->FindBin(x),
                                         h1->GetYaxis()->FindBin(y) ) );
   }

   // h3 has to be filled again so that the erros are properly calculated
   r.SetSeed(seed);
   for ( Int_t e = 0; e < nEvents * nEvents; ++e ) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h3->Fill(x, y,  h2->GetBinContent( h2->GetXaxis()->FindBin(x),
                                         h2->GetYaxis()->FindBin(y) ) );
   }

   // No the bin contents has to be reduced, as it was filled twice!
   for ( Int_t i = 0; i <= h3->GetNbinsX() + 1; ++i ) {
      for ( Int_t j = 0; j <= h3->GetNbinsY() + 1; ++j ) {
         h3->SetBinContent(i, j, h3->GetBinContent(i, j) / 2 );
      }
   }

   h1->Multiply(h2);

   bool ret = equals("Multiply2D2", h3, h1, cmpOptStats, 1E-12);
   delete h2;
   delete h3;
   return ret;
}

bool testMul3D1() 
{
   Double_t c1 = r.Rndm();
   Double_t c2 = r.Rndm();

   TH3D* h1 = new TH3D("m3D1-h1", "h1-Title", 
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange);
   TH3D* h2 = new TH3D("m3D1-h2", "h2-Title", 
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange);
   TH3D* h3 = new TH3D("m3D1-h3", "h3=c1*h1*c2*h2",
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange);

   h1->Sumw2();h2->Sumw2();h3->Sumw2();

   UInt_t seed = r.GetSeed();
   // For possible problems
   r.SetSeed(seed);
   for ( Int_t e = 0; e < nEvents * nEvents; ++e ) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(x, y, z, 1.0);
   }

   for ( Int_t e = 0; e < nEvents * nEvents; ++e ) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h2->Fill(x, y, z,  1.0);
      h3->Fill(x, y, z,  c1*c2*h1->GetBinContent( h1->GetXaxis()->FindBin(x),
                                                  h1->GetYaxis()->FindBin(y),
                                                  h1->GetZaxis()->FindBin(z) ) );
   }

   // h3 has to be filled again so that the erros are properly calculated
   r.SetSeed(seed);
   for ( Int_t e = 0; e < nEvents * nEvents; ++e ) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h3->Fill(x, y, z,  c1*c2*h2->GetBinContent( h2->GetXaxis()->FindBin(x),
                                                  h2->GetYaxis()->FindBin(y),
                                                  h2->GetZaxis()->FindBin(z) ) );
   }

   // No the bin contents has to be reduced, as it was filled twice!
   for ( Int_t i = 0; i <= h3->GetNbinsX() + 1; ++i ) {
      for ( Int_t j = 0; j <= h3->GetNbinsY() + 1; ++j ) {
         for ( Int_t h = 0; h <= h3->GetNbinsZ() + 1; ++h ) {
            h3->SetBinContent(i, j, h, h3->GetBinContent(i, j, h) / 2 );
         }
      }
   }

   TH3D* h4 = new TH3D("m3D1-h4", "h4=h1*h2", 
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange);
   h4->Multiply(h1, h2, c1, c2);

   bool ret = equals("Multiply3D1", h3, h4, cmpOptStats, 1E-13);
   delete h1;
   delete h2;
   delete h3;
   return ret;
}

bool testMul3D2() 
{
   TH3D* h1 = new TH3D("m3D2-h1", "h1-Title", 
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange);
   TH3D* h2 = new TH3D("m3D2-h2", "h2-Title", 
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange);
   TH3D* h3 = new TH3D("m3D2-h3", "h3=h1*h2",
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange);

   h1->Sumw2();h2->Sumw2();h3->Sumw2();

   UInt_t seed = r.GetSeed();
   // For possible problems
   r.SetSeed(seed);
   for ( Int_t e = 0; e < nEvents * nEvents; ++e ) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(x, y, z, 1.0);
   }

   for ( Int_t e = 0; e < nEvents * nEvents; ++e ) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h2->Fill(x, y, z,  1.0);
      h3->Fill(x, y, z, h1->GetBinContent( h1->GetXaxis()->FindBin(x),
                                           h1->GetYaxis()->FindBin(y),
                                           h1->GetZaxis()->FindBin(z) ) );
   }

   // h3 has to be filled again so that the erros are properly calculated
   r.SetSeed(seed);
   for ( Int_t e = 0; e < nEvents * nEvents; ++e ) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h3->Fill(x, y, z, h2->GetBinContent( h2->GetXaxis()->FindBin(x),
                                           h2->GetYaxis()->FindBin(y),
                                           h2->GetZaxis()->FindBin(z) ) );
   }

   // No the bin contents has to be reduced, as it was filled twice!
   for ( Int_t i = 0; i <= h3->GetNbinsX() + 1; ++i ) {
      for ( Int_t j = 0; j <= h3->GetNbinsY() + 1; ++j ) {
         for ( Int_t h = 0; h <= h3->GetNbinsZ() + 1; ++h ) {
            h3->SetBinContent(i, j, h, h3->GetBinContent(i, j, h) / 2 );
         }
      }
   }

   h1->Multiply(h2);

   bool ret = equals("Multiply3D2", h3, h1, cmpOptStats, 1E-13);
   delete h2;
   delete h3;
   return ret;
}

bool testDivide1() 
{
   Double_t c1 = 1;//r.Rndm();
   Double_t c2 = 1;//r.Rndm();

   TH1D* h1 = new TH1D("d1D1-h1", "h1-Title", numberOfBins, minRange, maxRange);
   TH1D* h2 = new TH1D("d1D1-h2", "h2-Title", numberOfBins, minRange, maxRange);

   h1->Sumw2();h2->Sumw2();

   UInt_t seed = r.GetSeed();
   // For possible problems
   r.SetSeed(seed);
   for ( Int_t e = 0; e < nEvents; ++e ) {
      Double_t value;
      value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(value, 1.0);
      value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h2->Fill(value,  1.0);
   }

   TH1D* h3 = new TH1D("d1D1-h3", "h3=(c1*h1)/(c2*h2)", numberOfBins, minRange, maxRange);
   h3->Divide(h1, h2, c1, c2);
      
   TH1D* h4 = new TH1D("d1D1-h4", "h4=h3*h2)", numberOfBins, minRange, maxRange);
   h4->Multiply(h2, h3, 1, 1.0);
   for ( Int_t bin = 0; bin <= h4->GetNbinsX() + 1; ++bin ) {
      Double_t error = h1->GetBinError(bin) * h1->GetBinError(bin);
      error += 2 * h3->GetBinContent(bin)*h3->GetBinContent(bin)*h3->GetBinError(bin)*h3->GetBinError(bin);
      h4->SetBinError( bin, sqrt(error) );
   }

//    cout << h2 << endl;
//    cout << h3 << endl;

   return equals("Divide1D1", h1, h4, cmpOptStats/* | cmpOptDebug*/, 1E-13);
}

bool stressAssign1D()
{
   TH1D* h1 = new TH1D("=1D-h1", "h1-Title", numberOfBins, minRange, maxRange);

   h1->Sumw2();

   for ( Int_t e = 0; e < nEvents; ++e ) {
      Double_t value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(value, 1.0);
   }

   TH1D* h2 = new TH1D("=1D-h2", "h2-Title", numberOfBins, minRange, maxRange);
   *h2 = *h1;

   bool ret = equals("Assign Oper Hist '='  1D", h1, h2, cmpOptStats);
   delete h1;
   return ret;
}

bool stressAssignProfile1D()
{
   TProfile* p1 = new TProfile("=1D-p1", "p1-Title", numberOfBins, minRange, maxRange);

   //p1->Sumw2();

   for ( Int_t e = 0; e < nEvents; ++e ) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p1->Fill(x, y, 1.0);
   }

   TProfile* p2 = new TProfile("=1D-p2", "p2-Title", numberOfBins, minRange, maxRange);
   *p2 = *p1;

   bool ret = equals("Assign Oper Prof '='  1D", p1, p2, cmpOptStats);
   delete p1;
   return ret;
}

bool stressCopyConstructor1D()
{
   TH1D* h1 = new TH1D("cc1D-h1", "h1-Title", numberOfBins, minRange, maxRange);

   h1->Sumw2();

   for ( Int_t e = 0; e < nEvents; ++e ) {
      Double_t value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(value, 1.0);
   }

   TH1D* h2 = new TH1D(*h1);

   bool ret = equals("Copy Constructor Hist 1D", h1, h2, cmpOptStats);
   delete h1;
   return ret;
}

bool stressCopyConstructorProfile1D()
{
   TProfile* p1 = new TProfile("cc1D-p1", "p1-Title", numberOfBins, minRange, maxRange);

   //p1->Sumw2();

   for ( Int_t e = 0; e < nEvents; ++e ) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p1->Fill(x, y, 1.0);
   }

   TProfile* p2 = new TProfile(*p1);

   bool ret = equals("Copy Constructor Prof 1D", p1, p2, cmpOptStats);
   delete p1;
   return ret;
}

bool stressClone1D()
{
   TH1D* h1 = new TH1D("cl1D-h1", "h1-Title", numberOfBins, minRange, maxRange);

   h1->Sumw2();

   for ( Int_t e = 0; e < nEvents; ++e ) {
      Double_t value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(value, 1.0);
   }

   TH1D* h2 = static_cast<TH1D*> ( h1->Clone() );

   bool ret = equals("Clone Function Hist   1D", h1, h2, cmpOptStats);
   delete h1;
   return ret;
}

bool stressCloneProfile1D()
{
   TProfile* p1 = new TProfile("cl1D-p1", "p1-Title", numberOfBins, minRange, maxRange);

   //p1->Sumw2();

   for ( Int_t e = 0; e < nEvents; ++e ) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p1->Fill(x, y, 1.0);
   }

   TProfile* p2 = static_cast<TProfile*> ( p1->Clone() );

   bool ret = equals("Clone Function Prof   1D", p1, p2, cmpOptStats);
   delete p1;
   return ret;
}

bool stressAssign2D()
{
   TH2D* h1 = new TH2D("=2D-h1", "h1-Title", 
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange);

   h1->Sumw2();

   for ( Int_t e = 0; e < nEvents * nEvents; ++e ) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(x, y, 1.0);
   }
   
   TH2D* h2 = new TH2D("=2D-h2", "h2-Title", 
                       numberOfBins, minRange, maxRange, 
                       numberOfBins, minRange, maxRange);
   *h2 = *h1;

   bool ret = equals("Assign Oper Hist '='  2D", h1, h2, cmpOptStats);
   delete h1;
   return ret;
}

bool stressAssignProfile2D()
{
   TProfile2D* p1 = new TProfile2D("=2D-p1", "p1-Title", 
                                   numberOfBins, minRange, maxRange,
                                   numberOfBins, minRange, maxRange);

   for ( Int_t e = 0; e < nEvents * nEvents; ++e ) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p1->Fill(x, y, z, 1.0);
   }

   TProfile2D* p2 = new TProfile2D("=2D-p2", "p2-Title", 
                                   numberOfBins, minRange, maxRange, 
                                   numberOfBins, minRange, maxRange);
   *p2 = *p1;

   bool ret = equals("Assign Oper Prof '='  2D", p1, p2, cmpOptStats);
   delete p1;
   return ret;
}


bool stressCopyConstructor2D()
{
   TH2D* h1 = new TH2D("cc2D-h1", "h1-Title", 
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange);

   h1->Sumw2();

   for ( Int_t e = 0; e < nEvents * nEvents; ++e ) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(x, y, 1.0);
   }

   TH2D* h2 = new TH2D(*h1);

   bool ret = equals("Copy Constructor Hist 2D", h1, h2, cmpOptStats);
   delete h1;
   return ret;
}

bool stressCopyConstructorProfile2D()
{
   TProfile2D* p1 = new TProfile2D("cc2D-p1", "p1-Title", 
                                   numberOfBins, minRange, maxRange,
                                   numberOfBins, minRange, maxRange);

   for ( Int_t e = 0; e < nEvents * nEvents; ++e ) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p1->Fill(x, y, z, 1.0);
   }

   TProfile2D* p2 = new TProfile2D(*p1);

   bool ret = equals("Copy Constructor Prof 2D", p1, p2, cmpOptStats);
   delete p1;
   return ret;
}

bool stressClone2D()
{
   TH2D* h1 = new TH2D("cl2D-h1", "h1-Title", 
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange);

   h1->Sumw2();

   for ( Int_t e = 0; e < nEvents * nEvents; ++e ) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(x, y, 1.0);
   }

   TH2D* h2 = static_cast<TH2D*> ( h1->Clone() );

   bool ret = equals("Clone Function Hist   2D", h1, h2, cmpOptStats);
   delete h1;
   return ret;
}

bool stressCloneProfile2D()
{
   TProfile2D* p1 = new TProfile2D("cl2D-p1", "p1-Title", 
                                   numberOfBins, minRange, maxRange,
                                   numberOfBins, minRange, maxRange);

   for ( Int_t e = 0; e < nEvents * nEvents; ++e ) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p1->Fill(x, y, z, 1.0);
   }

   TProfile2D* p2 = static_cast<TProfile2D*> ( p1->Clone() );

   bool ret = equals("Clone Function Prof   2D", p1, p2, cmpOptStats);
   delete p1;
   return ret;
}

bool stressAssign3D()
{
   TH3D* h1 = new TH3D("=3D-h1", "h1-Title", 
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange);

   h1->Sumw2();

   for ( Int_t e = 0; e < nEvents * nEvents; ++e ) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(x, y, z, 1.0);
   }

   TH3D* h2 = new TH3D("=3D-h2", "h2-Title", 
                       numberOfBins, minRange, maxRange, 
                       numberOfBins, minRange, maxRange, 
                       numberOfBins, minRange, maxRange);
   *h2 = *h1;

   bool ret = equals("Assign Oper Hist '='  3D", h1, h2, cmpOptStats);
   delete h1;
   return ret;
}

bool stressAssignProfile3D()
{
   TProfile3D* p1 = new TProfile3D("=3D-p1", "p1-Title", 
                                   numberOfBins, minRange, maxRange,
                                   numberOfBins, minRange, maxRange,
                                   numberOfBins, minRange, maxRange);

   for ( Int_t e = 0; e < nEvents * nEvents; ++e ) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t t = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p1->Fill(x, y, z, t, 1.0);
   }

   TProfile3D* p2 = new TProfile3D("=3D-p2", "p2-Title", 
                                   numberOfBins, minRange, maxRange, 
                                   numberOfBins, minRange, maxRange, 
                                   numberOfBins, minRange, maxRange);
   *p2 = *p1;

   bool ret = equals("Assign Oper Prof '='  3D", p1, p2/*, cmpOptDebug | cmpOptStats*/);
   delete p1;
   return ret;
}

bool stressCopyConstructor3D()
{
   TH3D* h1 = new TH3D("cc3D-h1", "h1-Title", 
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange);

   h1->Sumw2();

   for ( Int_t e = 0; e < nEvents * nEvents; ++e ) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(x, y, z, 1.0);
   }

   TH3D* h2 = new TH3D(*h1);

   bool ret = equals("Copy Constructor Hist 3D", h1, h2, cmpOptStats);
   delete h1;
   return ret;
}

bool stressCopyConstructorProfile3D()
{
   TProfile3D* p1 = new TProfile3D("cc3D-p1", "p1-Title", 
                                   numberOfBins, minRange, maxRange,
                                   numberOfBins, minRange, maxRange,
                                   numberOfBins, minRange, maxRange);

   for ( Int_t e = 0; e < nEvents * nEvents; ++e ) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t t = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p1->Fill(x, y, z, t, 1.0);
   }

   TProfile3D* p2 = new TProfile3D(*p1);

   bool ret = equals("Copy Constructor Prof 3D", p1, p2/*, cmpOptStats*/);
   delete p1;
   return ret;
}

bool stressClone3D()
{
   TH3D* h1 = new TH3D("cl3D-h1", "h1-Title", 
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange);

   h1->Sumw2();

   for ( Int_t e = 0; e < nEvents * nEvents; ++e ) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(x, y, z, 1.0);
   }

   TH3D* h2 = static_cast<TH3D*> ( h1->Clone() );

   bool ret = equals("Clone Function Hist   3D", h1, h2, cmpOptStats);
   delete h1;
   return ret;
}

bool stressCloneProfile3D()
{
   TProfile3D* p1 = new TProfile3D("cl3D-p1", "p1-Title", 
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange);

   for ( Int_t e = 0; e < nEvents * nEvents; ++e ) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t t = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p1->Fill(x, y, z, t, 1.0);
   }

   TProfile3D* p2 = static_cast<TProfile3D*> ( p1->Clone() );

   bool ret = equals("Clone Function Prof   3D", p1, p2/*, cmpOptStats*/);
   delete p1;
   return ret;
}

bool testWriteRead1D()
{
   TH1D* h1 = new TH1D("wr1D-h1", "h1-Title", numberOfBins, minRange, maxRange);

   h1->Sumw2();

   for ( Int_t e = 0; e < nEvents; ++e ) {
      Double_t value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(value, 1.0);
   }

   TFile f("tmpHist.root", "RECREATE");
   h1->Write();
   f.Close();

   TFile f2("tmpHist.root");
   TH1D* h2 = static_cast<TH1D*> ( f2.Get("wr1D-h1") );

   bool ret = equals("Read/Write Hist 1D", h1, h2, cmpOptStats);
   delete h1;
   return ret;
}

bool testWriteReadProfile1D()
{
   TProfile* p1 = new TProfile("wr1D-p1", "p1-Title", numberOfBins, minRange, maxRange);

   //p1->Sumw2();

   for ( Int_t e = 0; e < nEvents; ++e ) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p1->Fill(x, y, 1.0);
   }

   TFile f("tmpHist.root", "RECREATE");
   p1->Write();
   f.Close();

   TFile f2("tmpHist.root");
   TProfile* p2 = static_cast<TProfile*> ( f2.Get("wr1D-p1") );

   bool ret = equals("Read/Write Prof 1D", p1, p2, cmpOptStats);
   delete p1;
   return ret;
}

bool testWriteRead2D()
{
   TH2D* h1 = new TH2D("wr2D-h1", "h1-Title", 
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange);

   h1->Sumw2();

   for ( Int_t e = 0; e < nEvents * nEvents; ++e ) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(x, y, 1.0);
   }

   TFile f("tmpHist.root", "RECREATE");
   h1->Write();
   f.Close();

   TFile f2("tmpHist.root");
   TH2D* h2 = static_cast<TH2D*> ( f2.Get("wr2D-h1") );

   bool ret = equals("Read/Write Hist 2D", h1, h2, cmpOptStats);
   delete h1;
   return ret;
}

bool testWriteReadProfile2D()
{
   TProfile2D* p1 = new TProfile2D("wr2D-p1", "p1-Title", 
                                   numberOfBins, minRange, maxRange,
                                   numberOfBins, minRange, maxRange);

   for ( Int_t e = 0; e < nEvents * nEvents; ++e ) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p1->Fill(x, y, z, 1.0);
   }

   TFile f("tmpHist.root", "RECREATE");
   p1->Write();
   f.Close();

   TFile f2("tmpHist.root");
   TProfile2D* p2 = static_cast<TProfile2D*> ( f2.Get("wr2D-p1") );

   bool ret = equals("Read/Write Prof 2D", p1, p2, cmpOptStats);
   delete p1;
   return ret;
}

bool testWriteRead3D()
{
   TH3D* h1 = new TH3D("wr3D-h1", "h1-Title", 
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange);

   h1->Sumw2();

   for ( Int_t e = 0; e < nEvents * nEvents; ++e ) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(x, y, z, 1.0);
   }

   TFile f("tmpHist.root", "RECREATE");
   h1->Write();
   f.Close();

   TFile f2("tmpHist.root");
   TH3D* h2 = static_cast<TH3D*> ( f2.Get("wr3D-h1") );

   bool ret = equals("Read/Write Hist 3D", h1, h2, cmpOptStats);
   delete h1;
   return ret;
}

bool testWriteReadProfile3D()
{
   TProfile3D* p1 = new TProfile3D("wr3D-p1", "p1-Title", 
                                 numberOfBins, minRange, maxRange,
                                 numberOfBins, minRange, maxRange,
                                 numberOfBins, minRange, maxRange);

   for ( Int_t e = 0; e < nEvents * nEvents; ++e ) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t t = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p1->Fill(x, y, z, t, 1.0);
   }

   TFile f("tmpHist.root", "RECREATE");
   p1->Write();
   f.Close();

   TFile f2("tmpHist.root");
   TProfile3D* p2 = static_cast<TProfile3D*> ( f2.Get("wr3D-p1") );

   // In this particular case the statistics are not checked. The
   // Chi2Test is not properly implemented for the TProfile3D
   // class. If the cmpOptStats flag is set, then there will be a
   // crash.
   bool ret = equals("Read/Write Prof 3D", p1, p2);
   delete p1;
   return ret;
}

bool stressHistOpts()
{
   r.SetSeed(0);
   const unsigned int numberOfTests = 36;
   pointer2Test testPointer[numberOfTests] = {  testAdd1,                testAdd2, 
                                                testAdd2D1,              testAdd2D2,
                                                testAdd3D1,              testAdd3D2, 
                                                testMul1,                testMul2,
                                                testMul2D1,              testMul2D2,
                                                testMul3D1,              testMul3D2, 
                                                //testDivide1,
                                                stressAssign1D,          stressAssignProfile1D, 
                                                stressCopyConstructor1D, stressCopyConstructorProfile1D, 
                                                stressClone1D,           stressCloneProfile1D,
                                                stressAssign2D,          stressAssignProfile2D,
                                                stressCopyConstructor2D, stressCopyConstructorProfile2D,
                                                stressClone2D,           stressCloneProfile2D,
                                                stressAssign3D,          stressAssignProfile3D,
                                                stressCopyConstructor3D, stressCopyConstructorProfile3D,
                                                stressClone3D,           stressCloneProfile3D,
                                                testWriteRead1D,         testWriteReadProfile1D,
                                                testWriteRead2D,         testWriteReadProfile2D,
                                                testWriteRead3D,         testWriteReadProfile3D
   };

   // Still to do: testDivide2, testDivide2D1, testDivide2D2 and
   // testDivide3D1, testDivide3D2. 

   // It depends on whether we can solve the problem with the 1D test
   // already done. It seems like there is something wrong with the
   // way the errors are being calculated. We have a formula to
   // calculate it by hand. Nevertheless the Divide method errors and
   // the ones calculated by ourselves differ a bit from those (in the
   // order of 1E-1).
   
   bool status = false;
   for ( unsigned int i = 0; i < numberOfTests; ++i )
      status |= testPointer[i]();

   return status;
}

// end stresHistRebin.cxx file

// old stresHistRebin.cxx file

bool testIntegerRebin()
{
   const int rebin = TMath::Nint( r.Uniform(minRebin, maxRebin) );
   Int_t seed = 4632;//0;
   TH1D* h1 = new TH1D("h1","Original Histogram", TMath::Nint( r.Uniform(1, 5) ) * rebin, minRange, maxRange);
   r.SetSeed(seed);
   for ( Int_t i = 0; i < nEvents; ++i )
      h1->Fill( r.Uniform( minRange * .9 , maxRange * 1.1 ) );

   TH1D* h2 = static_cast<TH1D*>( h1->Rebin(rebin, "testIntegerRebin") );

   TH1D* h3 = new TH1D("testIntegerRebin2", "testIntegerRebin2", 
                       h1->GetNbinsX() / rebin, minRange, maxRange);
   r.SetSeed(seed);
   for ( Int_t i = 0; i < nEvents; ++i )
      h3->Fill( r.Uniform( minRange * .9 , maxRange * 1.1 ) );

   return equals("TestIntegerRebin", h2, h3, cmpOptStats  );
}

bool testIntegerRebinNoName()
{
   const int rebin = TMath::Nint( r.Uniform(minRebin, maxRebin) );
   Int_t seed = 4632;//0;
   TH1D* h1 = new TH1D("h2","Original Histogram", TMath::Nint( r.Uniform(1, 5) ) * rebin, minRange, maxRange);
   r.SetSeed(seed);
   for ( Int_t i = 0; i < nEvents; ++i )
      h1->Fill( r.Uniform( minRange * .9 , maxRange * 1.1 ) );

   TH1D* h2 = dynamic_cast<TH1D*>( h1->Clone() );
   h2->Rebin(rebin);

   TH1D* h3 = new TH1D("testIntegerRebinNoName", "testIntegerRebinNoName", 
                       int(h1->GetNbinsX() / rebin + 0.1), minRange, maxRange);
   r.SetSeed(seed);
   for ( Int_t i = 0; i < nEvents; ++i )
      h3->Fill( r.Uniform( minRange * .9 , maxRange * 1.1 ) );

   return equals("TestIntRebinNoName", h2, h3, cmpOptStats );

   // This method fails because the Chi2Test is different of 1 for
   // both of them. We need to look into both the Rebin method and the
   // Chi2Test method to understand better what is going wrong.
}

bool testArrayRebin()
{
   const int rebin = TMath::Nint( r.Uniform(minRebin, maxRebin) ) + 1;
   Int_t seed = 4632;//0;
   TH1D* h1 = new TH1D("h3","Original Histogram", TMath::Nint( r.Uniform(1, 5) ) * rebin * 2, minRange, maxRange);
   r.SetSeed(seed);
   for ( Int_t i = 0; i < nEvents; ++i )
      h1->Fill( r.Uniform( minRange * .9 , maxRange * 1.1 ) );

   // Create vector 
   Double_t * rebinArray = new Double_t[rebin];
   r.RndmArray(rebin, rebinArray);
   std::sort(rebinArray, rebinArray + rebin);
   for ( Int_t i = 0; i < rebin; ++i ) {
      rebinArray[i] = TMath::Nint( rebinArray[i] * ( h1->GetNbinsX() - 2 ) + 2 );
      rebinArray[i] = h1->GetBinLowEdge( h1->FindBin(rebinArray[i]) );
   }
   

   rebinArray[0] = minRange;
   rebinArray[rebin-1] = maxRange;

   #ifdef __DEBUG__
   for ( Int_t i = 0; i < rebin; ++i ) 
      cout << rebinArray[i] << endl;
   cout << "rebin: " << rebin << endl;
   #endif

   TH1D* h2 = static_cast<TH1D*>( h1->Rebin(rebin - 1, "testArrayRebin", rebinArray) );

   TH1D* h3 = new TH1D("testArrayRebin2", "testArrayRebin2", rebin - 1, rebinArray );
   r.SetSeed(seed);
   for ( Int_t i = 0; i < nEvents; ++i )
      h3->Fill( r.Uniform( minRange * .9 , maxRange * 1.1 ) );

   delete [] rebinArray;
      
   return equals("TestArrayRebin", h2, h3, cmpOptStats);
}

bool test2DRebin()
{
   Int_t xrebin = TMath::Nint( r.Uniform(minRebin, maxRebin) );
   Int_t yrebin = TMath::Nint( r.Uniform(minRebin, maxRebin) );
   TH2D* h2d = new TH2D("h2d","Original Histogram", 
                       xrebin * TMath::Nint( r.Uniform(1, 5) ), minRange, maxRange, 
                       yrebin * TMath::Nint( r.Uniform(1, 5) ), minRange, maxRange);
   
   Int_t seed = 4632;//0;
   r.SetSeed(seed);
   for ( Int_t i = 0; i < nEvents; ++i )
      h2d->Fill( r.Uniform( minRange * .9 , maxRange * 1.1 ), r.Uniform( minRange * .9 , maxRange * 1.1 ) );

   TH2D* h2d2 = (TH2D*) h2d->Rebin2D(xrebin,yrebin, "h2d2");

   TH2D* h3 = new TH2D("test2DRebin", "test2DRebin", 
                       h2d->GetNbinsX() / xrebin, minRange, maxRange,
                       h2d->GetNbinsY() / yrebin, minRange, maxRange );
   r.SetSeed(seed);
   for ( Int_t i = 0; i < nEvents; ++i )
      h3->Fill( r.Uniform( minRange * .9 , maxRange * 1.1 ), r.Uniform( minRange * .9 , maxRange * 1.1 ) );

   return equals("TestIntRebin2D", h2d2, h3, cmpOptStats);
}

bool stressHistRebin()
{
   const unsigned int numberOfTests = 4;
   pointer2Test testPointer[numberOfTests] = { testIntegerRebin, 
                                               testIntegerRebinNoName,
                                               testArrayRebin,
                                               test2DRebin };

   bool status = false;
   for ( unsigned int i = 0; i < numberOfTests; ++i )
      status |= testPointer[i]();

   return status;
}

// end stressHistRebin.cxx file
 

// old stressHistProj file 

// In case of deviation, the profiles' content will not work anymore
// try only for testing the statistics
static const double centre_deviation = 0.0;


class ProjectionTester {
   
private:
   static const unsigned int binsize = 10;
   static const int lower_limit = 0;
   static const int upper_limit = 10;


   TH3D* h3;
   TH2D* h2XY;
   TH2D* h2XZ;
   TH2D* h2YX;
   TH2D* h2YZ;
   TH2D* h2ZX;
   TH2D* h2ZY;
   TH1D* h1X;
   TH1D* h1Y;
   TH1D* h1Z;

   TH1D* h1XStats;
   TH1D* h1YStats;
   TH1D* h1ZStats;
   
   TProfile2D* pe2XY;
   TProfile2D* pe2XZ;
   TProfile2D* pe2YX;
   TProfile2D* pe2YZ;
   TProfile2D* pe2ZX;
   TProfile2D* pe2ZY;
   
   TH2D* h2wXY;
   TH2D* h2wXZ;
   TH2D* h2wYX;
   TH2D* h2wYZ;
   TH2D* h2wZX;
   TH2D* h2wZY;
   
   TProfile* pe1XY;
   TProfile* pe1XZ;
   TProfile* pe1YX;
   TProfile* pe1YZ;
   TProfile* pe1ZX;
   TProfile* pe1ZY;

   TH1D* hw1XZ;
   TH1D* hw1XY;
   TH1D* hw1YX;
   TH1D* hw1YZ;
   TH1D* hw1ZX;
   TH1D* hw1ZY;
   
   THnSparseD* s3;

   bool buildWithWeights;

   TRandom2 r;
   
public:
   ProjectionTester()
   {
      CreateHistograms();
      buildWithWeights = false;
   }
   
   void CreateHistograms()
   {
      h3 = new TH3D("h3","h3", binsize, lower_limit, upper_limit, 
                               binsize, lower_limit, upper_limit, 
                               binsize, lower_limit, upper_limit);

      h2XY = new TH2D("h2XY", "h2XY", binsize, lower_limit, upper_limit, 
                                      binsize, lower_limit, upper_limit);
      h2XZ = new TH2D("h2XZ", "h2XZ", binsize, lower_limit, upper_limit, 
                                      binsize, lower_limit, upper_limit);
      h2YX = new TH2D("h2YX", "h2YX", binsize, lower_limit, upper_limit, 
                                      binsize, lower_limit, upper_limit);
      h2YZ = new TH2D("h2YZ", "h2YZ", binsize, lower_limit, upper_limit, 
                                      binsize, lower_limit, upper_limit);
      h2ZX = new TH2D("h2ZX", "h2ZX", binsize, lower_limit, upper_limit, 
                                      binsize, lower_limit, upper_limit);
      h2ZY = new TH2D("h2ZY", "h2ZY", binsize, lower_limit, upper_limit, 
                                      binsize, lower_limit, upper_limit);

      // The bit is set for all the histograms (It's a statistic variable)
      TH1::StatOverflows(kTRUE);

      h1X = new TH1D("h1X", "h1X", binsize, lower_limit, upper_limit);
      h1Y = new TH1D("h1Y", "h1Y", binsize, lower_limit, upper_limit);
      h1Z = new TH1D("h1Z", "h1Z", binsize, lower_limit, upper_limit);

      h1XStats = new TH1D("h1XStats", "h1XStats", binsize, lower_limit, upper_limit);
      h1YStats = new TH1D("h1YStats", "h1YStats", binsize, lower_limit, upper_limit);
      h1ZStats = new TH1D("h1ZStats", "h1ZStats", binsize, lower_limit, upper_limit);

      pe2XY = new TProfile2D("pe2XY", "pe2XY", binsize, lower_limit, upper_limit, 
                                               binsize, lower_limit, upper_limit);
      pe2XZ = new TProfile2D("pe2XZ", "pe2XZ", binsize, lower_limit, upper_limit, 
                                               binsize, lower_limit, upper_limit);
      pe2YX = new TProfile2D("pe2YX", "pe2YX", binsize, lower_limit, upper_limit, 
                                               binsize, lower_limit, upper_limit);
      pe2YZ = new TProfile2D("pe2YZ", "pe2YZ", binsize, lower_limit, upper_limit, 
                                               binsize, lower_limit, upper_limit);
      pe2ZX = new TProfile2D("pe2ZX", "pe2ZX", binsize, lower_limit, upper_limit, 
                                               binsize, lower_limit, upper_limit);
      pe2ZY = new TProfile2D("pe2ZY", "pe2ZY", binsize, lower_limit, upper_limit, 
                                               binsize, lower_limit, upper_limit);
      
      h2wXY = new TH2D("h2wXY", "h2wXY", binsize, lower_limit, upper_limit, 
                                         binsize, lower_limit, upper_limit);
      h2wXZ = new TH2D("h2wXZ", "h2wXZ", binsize, lower_limit, upper_limit, 
                                         binsize, lower_limit, upper_limit);
      h2wYX = new TH2D("h2wYX", "h2wYX", binsize, lower_limit, upper_limit, 
                                         binsize, lower_limit, upper_limit);
      h2wYZ = new TH2D("h2wYZ", "h2wYZ", binsize, lower_limit, upper_limit, 
                                         binsize, lower_limit, upper_limit);
      h2wZX = new TH2D("h2wZX", "h2wZX", binsize, lower_limit, upper_limit, 
                                         binsize, lower_limit, upper_limit);
      h2wZY = new TH2D("h2wZY", "h2wZY", binsize, lower_limit, upper_limit, 
                                         binsize, lower_limit, upper_limit);

      h2wXY->Sumw2();
      h2wXZ->Sumw2();
      h2wYX->Sumw2();
      h2wYZ->Sumw2();
      h2wZX->Sumw2();
      h2wZY->Sumw2();

      pe1XY = new TProfile("pe1XY", "pe1XY", binsize, lower_limit, upper_limit);
      pe1XZ = new TProfile("pe1XZ", "pe1XZ", binsize, lower_limit, upper_limit);
      pe1YX = new TProfile("pe1YX", "pe1YX", binsize, lower_limit, upper_limit);
      pe1YZ = new TProfile("pe1YZ", "pe1YZ", binsize, lower_limit, upper_limit);
      pe1ZX = new TProfile("pe1ZX", "pe1ZX", binsize, lower_limit, upper_limit);
      pe1ZY = new TProfile("pe1ZY", "pe1ZY", binsize, lower_limit, upper_limit);

      hw1XY = new TH1D("hw1XY", "hw1XY", binsize, lower_limit, upper_limit);
      hw1XZ = new TH1D("hw1XZ", "hw1XZ", binsize, lower_limit, upper_limit);
      hw1YX = new TH1D("hw1YX", "hw1YX", binsize, lower_limit, upper_limit);
      hw1YZ = new TH1D("hw1YZ", "hw1YZ", binsize, lower_limit, upper_limit);
      hw1ZX = new TH1D("hw1ZX", "hw1ZX", binsize, lower_limit, upper_limit);
      hw1ZY = new TH1D("hw1ZY", "hw1ZY", binsize, lower_limit, upper_limit);

      hw1XZ->Sumw2();
      hw1XY->Sumw2();
      hw1YX->Sumw2();
      hw1YZ->Sumw2();
      hw1ZX->Sumw2();
      hw1ZY->Sumw2();

      Int_t bsize[] = {binsize, binsize, binsize};
      Double_t xmin[] = {lower_limit, lower_limit, lower_limit};
      Double_t xmax[] = {upper_limit, upper_limit, upper_limit};
      s3 = new THnSparseD("s3","s3", 3, bsize, xmin, xmax);

   }
   
   void DeleteHistograms()
   {
      delete h3;
      
      delete h2XY;
      delete h2XZ;
      delete h2YX;
      delete h2YZ;
      delete h2ZX;
      delete h2ZY;

      delete h1X;
      delete h1Y;
      delete h1Z;
      
      delete h1XStats;
      delete h1YStats;
      delete h1ZStats;

      delete pe2XY;
      delete pe2XZ;
      delete pe2YX;
      delete pe2YZ;
      delete pe2ZX;
      delete pe2ZY;
      
      delete h2wXY;
      delete h2wXZ;
      delete h2wYX;
      delete h2wYZ;
      delete h2wZX;
      delete h2wZY;

      delete pe1XY;
      delete pe1XZ;
      delete pe1YX;
      delete pe1YZ;
      delete pe1ZY;
      delete pe1ZX;

      delete hw1XY;
      delete hw1XZ;
      delete hw1YX;
      delete hw1YZ;
      delete hw1ZX;
      delete hw1ZY;

      delete s3;
   }
   
   virtual ~ProjectionTester()
   {
      DeleteHistograms();
   }
   
   
   void buildHistograms()
   {
      for ( int ix = 0; ix <= h3->GetXaxis()->GetNbins() + 1; ++ix ) {
         double x = centre_deviation * h3->GetXaxis()->GetBinWidth(ix) + h3->GetXaxis()->GetBinCenter(ix);
         for ( int iy = 0; iy <= h3->GetYaxis()->GetNbins() + 1; ++iy ) {
            double y = centre_deviation * h3->GetYaxis()->GetBinWidth(iy) + h3->GetYaxis()->GetBinCenter(iy);
            for ( int iz = 0; iz <= h3->GetZaxis()->GetNbins() + 1; ++iz ) {
               double z = centre_deviation * h3->GetZaxis()->GetBinWidth(iz) + h3->GetZaxis()->GetBinCenter(iz);
               for ( int i = 0; i < (int) r.Uniform(1,3); ++i )
               {
                  h3->Fill(x,y,z);

                  Double_t points[] = {x,y,z};
                  s3->Fill(points);
                  
                  h2XY->Fill(x,y);
                  h2XZ->Fill(x,z);
                  h2YX->Fill(y,x);
                  h2YZ->Fill(y,z);
                  h2ZX->Fill(z,x);
                  h2ZY->Fill(z,y);
                  
                  h1X->Fill(x);
                  h1Y->Fill(y);
                  h1Z->Fill(z);

                  if ( ix > 0 && ix < h3->GetXaxis()->GetNbins() + 1 &&
                       iy > 0 && iy < h3->GetYaxis()->GetNbins() + 1 &&
                       iz > 0 && iz < h3->GetZaxis()->GetNbins() + 1 )
                  {
                     h1XStats->Fill(x);
                     h1YStats->Fill(y);
                     h1ZStats->Fill(z);
                  }

                  pe2XY->Fill(x,y,z);
                  pe2XZ->Fill(x,z,y);
                  pe2YX->Fill(y,x,z);
                  pe2YZ->Fill(y,z,x);
                  pe2ZX->Fill(z,x,y);
                  pe2ZY->Fill(z,y,x);
                  
                  h2wXY->Fill(x,y,z);
                  h2wXZ->Fill(x,z,y);
                  h2wYX->Fill(y,x,z);
                  h2wYZ->Fill(y,z,x);
                  h2wZX->Fill(z,x,y);
                  h2wZY->Fill(z,y,x);

                  pe1XY->Fill(x,y);
                  pe1XZ->Fill(x,z);
                  pe1YX->Fill(y,x);
                  pe1YZ->Fill(y,z);
                  pe1ZX->Fill(z,x);
                  pe1ZY->Fill(z,y);

                  hw1XY->Fill(x,y);
                  hw1XZ->Fill(x,z);
                  hw1YX->Fill(y,x);
                  hw1YZ->Fill(y,z);
                  hw1ZX->Fill(z,x);
                  hw1ZY->Fill(z,y);
               }
            }
         }
      }

      buildWithWeights = false;
   }

   void buildHistogramsWithWeights()
   {
      for ( int ix = 0; ix <= h3->GetXaxis()->GetNbins() + 1; ++ix ) {
         double x = centre_deviation * h3->GetXaxis()->GetBinWidth(ix) + h3->GetXaxis()->GetBinCenter(ix);
         for ( int iy = 0; iy <= h3->GetYaxis()->GetNbins() + 1; ++iy ) {
            double y = centre_deviation * h3->GetYaxis()->GetBinWidth(iy) + h3->GetYaxis()->GetBinCenter(iy);
            for ( int iz = 0; iz <= h3->GetZaxis()->GetNbins() + 1; ++iz ) {
               double z = centre_deviation * h3->GetZaxis()->GetBinWidth(iz) + h3->GetZaxis()->GetBinCenter(iz);
               Double_t w = (Double_t) r.Uniform(1,3);

               h3->Fill(x,y,z,w);

               Double_t points[] = {x,y,z};
               s3->Fill(points,w);
               
               h2XY->Fill(x,y,w);
               h2XZ->Fill(x,z,w);
               h2YX->Fill(y,x,w);
               h2YZ->Fill(y,z,w);
               h2ZX->Fill(z,x,w);
               h2ZY->Fill(z,y,w);
               
               h1X->Fill(x,w);
               h1Y->Fill(y,w);
               h1Z->Fill(z,w);
                   
               if ( ix > 0 && ix < h3->GetXaxis()->GetNbins() + 1 &&
                    iy > 0 && iy < h3->GetYaxis()->GetNbins() + 1 &&
                    iz > 0 && iz < h3->GetZaxis()->GetNbins() + 1 )
               {
                  h1XStats->Fill(x,w);
                  h1YStats->Fill(y,w);
                  h1ZStats->Fill(z,w);
               }              

               pe2XY->Fill(x,y,z,w);
               pe2XZ->Fill(x,z,y,w);
               pe2YX->Fill(y,x,z,w);
               pe2YZ->Fill(y,z,x,w);
               pe2ZX->Fill(z,x,y,w);
               pe2ZY->Fill(z,y,x,w);
               
               h2wXY->Fill(x,y,z*w);
               h2wXZ->Fill(x,z,y*w);
               h2wYX->Fill(y,x,z*w);
               h2wYZ->Fill(y,z,x*w);
               h2wZX->Fill(z,x,y*w);
               h2wZY->Fill(z,y,x*w);
               
               pe1XY->Fill(x,y,w);
               pe1XZ->Fill(x,z,w);
               pe1YX->Fill(y,x,w);
               pe1YZ->Fill(y,z,w);
               pe1ZX->Fill(z,x,w);
               pe1ZY->Fill(z,y,w);
               
               hw1XY->Fill(x,y*w);
               hw1XZ->Fill(x,z*w);
               hw1YX->Fill(y,x*w);
               hw1YZ->Fill(y,z*w);
               hw1ZX->Fill(z,x*w);
               hw1ZY->Fill(z,y*w);
            }
         }
      }

      buildWithWeights = true;
   }
   
   void buildHistograms(int xmin, int xmax,
                        int ymin, int ymax,
                        int zmin, int zmax)
   {
      for ( int ix = 0; ix <= h3->GetXaxis()->GetNbins() + 1; ++ix ) {
         double x = centre_deviation * h3->GetXaxis()->GetBinCenter(ix);
         for ( int iy = 0; iy <= h3->GetYaxis()->GetNbins() + 1; ++iy ) {
            double y = centre_deviation * h3->GetYaxis()->GetBinCenter(iy);
            for ( int iz = 0; iz <= h3->GetZaxis()->GetNbins() + 1; ++iz ) {
               double z = centre_deviation * h3->GetZaxis()->GetBinCenter(iz);
               for ( int i = 0; i < (int) r.Uniform(1,3); ++i )
               {
                  h3->Fill(x,y,z);

                  Double_t points[] = {x,y,z};
                  s3->Fill(points);
                  
                  if ( h3->GetXaxis()->FindBin(x) >= xmin && h3->GetXaxis()->FindBin(x) <= xmax &&
                       h3->GetYaxis()->FindBin(y) >= ymin && h3->GetYaxis()->FindBin(y) <= ymax &&
                       h3->GetZaxis()->FindBin(z) >= zmin && h3->GetZaxis()->FindBin(z) <= zmax )
                  {
                     cout << "Filling (" << x << "," << y << "," << z << ")!" << endl;
                     
                     h2XY->Fill(x,y);
                     h2XZ->Fill(x,z);
                     h2YX->Fill(y,x);
                     h2YZ->Fill(y,z);
                     h2ZX->Fill(z,x);
                     h2ZY->Fill(z,y);
                     
                     h1X->Fill(x);
                     h1Y->Fill(y);
                     h1Z->Fill(z);
                     
                     pe2XY->Fill(x,y,z);
                     pe2XZ->Fill(x,z,y);
                     pe2YX->Fill(y,x,z);
                     pe2YZ->Fill(y,z,x);
                     pe2ZX->Fill(z,x,y);
                     pe2ZY->Fill(z,y,x);
                     
                     h2wXY->Fill(x,y,z);
                     h2wXZ->Fill(x,z,y);
                     h2wYX->Fill(y,x,z);
                     h2wYZ->Fill(y,z,x);
                     h2wZX->Fill(z,x,y);
                     h2wZY->Fill(z,y,x);

                     pe1XY->Fill(x,y);
                     pe1XZ->Fill(x,z);
                     pe1YX->Fill(y,x);
                     pe1YZ->Fill(y,z);
                     pe1ZX->Fill(z,x);
                     pe1ZY->Fill(z,y);
                     
                     hw1XY->Fill(x,y);
                     hw1XZ->Fill(x,z);
                     hw1YX->Fill(y,x);
                     hw1YZ->Fill(y,z);
                     hw1ZX->Fill(z,x);
                     hw1ZY->Fill(z,y);
                  }
               }
            }
         }
      }
      
      h3->GetXaxis()->SetRange(xmin, xmax);
      h3->GetYaxis()->SetRange(ymin, ymax);
      h3->GetZaxis()->SetRange(zmin, zmax);
      
      h2XY->GetXaxis()->SetRange(xmin, xmax);
      h2XY->GetYaxis()->SetRange(ymin, ymax);
      
      h2XZ->GetXaxis()->SetRange(xmin, xmax);
      h2XZ->GetZaxis()->SetRange(zmin, zmax);
      
      h2YX->GetYaxis()->SetRange(ymin, ymax);
      h2YX->GetXaxis()->SetRange(xmin, xmax);
      
      h2YZ->GetYaxis()->SetRange(ymin, ymax);
      h2YZ->GetZaxis()->SetRange(zmin, zmax);
      
      h2ZX->GetZaxis()->SetRange(zmin, zmax);
      h2ZX->GetXaxis()->SetRange(xmin, xmax);
      
      h2ZY->GetZaxis()->SetRange(zmin, zmax);
      h2ZY->GetYaxis()->SetRange(ymin, ymax);
      
      h1X->GetXaxis()->SetRange(xmin, xmax);
      h1Y->GetXaxis()->SetRange(ymin, ymax);
      h1Z->GetXaxis()->SetRange(zmin, zmax);

      // Neet to set up the rest of the ranges!

      s3->GetAxis(1)->SetRange(xmin, xmax);
      s3->GetAxis(2)->SetRange(ymin, ymax);
      s3->GetAxis(3)->SetRange(zmin, zmax);

      buildWithWeights = false;
   }
   
   int compareHistograms()
   {
      int status = 0;
      int options = 0;
      
      // TH2 derived from TH3
      options = cmpOptStats;
      status += equals("TH3 -> XY", h2XY, (TH2D*) h3->Project3D("yx"), options);
      status += equals("TH3 -> XZ", h2XZ, (TH2D*) h3->Project3D("zx"), options);
      status += equals("TH3 -> YX", h2YX, (TH2D*) h3->Project3D("XY"), options);
      status += equals("TH3 -> YZ", h2YZ, (TH2D*) h3->Project3D("ZY"), options);
      status += equals("TH3 -> ZX", h2ZX, (TH2D*) h3->Project3D("XZ"), options);
      status += equals("TH3 -> ZY", h2ZY, (TH2D*) h3->Project3D("YZ"), options);
      options = 0;
      cout << "----------------------------------------------" << endl;
      
      // TH1 derived from TH3
      options = cmpOptStats;
      status += equals("TH3 -> X", h1X, (TH1D*) h3->Project3D("x"), options);
      status += equals("TH3 -> Y", h1Y, (TH1D*) h3->Project3D("y"), options);
      status += equals("TH3 -> Z", h1Z, (TH1D*) h3->Project3D("z"), options);
      options = 0;
      cout << "----------------------------------------------" << endl;
      
      // TH1 derived from h2XY
      options = cmpOptStats;
      status += equals("TH2XY -> X", h1X, (TH1D*) h2XY->ProjectionX("x"), options);
      status += equals("TH2XY -> Y", h1Y, (TH1D*) h2XY->ProjectionY("y"), options);
      // TH1 derived from h2XZ
      status += equals("TH2XZ -> X", h1X, (TH1D*) h2XZ->ProjectionX("x"), options);
      status += equals("TH2XZ -> Z", h1Z, (TH1D*) h2XZ->ProjectionY("z"), options);
      // TH1 derived from h2YX
      status += equals("TH2YX -> Y", h1Y, (TH1D*) h2YX->ProjectionX("y"), options);
      status += equals("TH2YX -> X", h1X, (TH1D*) h2YX->ProjectionY("x"), options);
      // TH1 derived from h2YZ
      status += equals("TH2YZ -> Y", h1Y, (TH1D*) h2YZ->ProjectionX("y"), options);
      status += equals("TH2YZ -> Z", h1Z, (TH1D*) h2YZ->ProjectionY("z"), options);
      // TH1 derived from h2ZX
      status += equals("TH2ZX -> Z", h1Z, (TH1D*) h2ZX->ProjectionX("z"), options);
      status += equals("TH2ZX -> X", h1X, (TH1D*) h2ZX->ProjectionY("x"), options);
      // TH1 derived from h2ZY
      status += equals("TH2ZY -> Z", h1Z, (TH1D*) h2ZY->ProjectionX("z"), options);
      status += equals("TH2ZY -> Y", h1Y, (TH1D*) h2ZY->ProjectionY("y"), options);
      options = 0;
      cout << "----------------------------------------------" << endl;
      
      // Now the histograms comming from the Profiles!
      options = cmpOptStats;
      status += equals("TH3 -> PBXY", h2XY, (TH2D*) h3->Project3DProfile("yx UF OF")->ProjectionXY("1", "B"), options);
      status += equals("TH3 -> PBXZ", h2XZ, (TH2D*) h3->Project3DProfile("zx UF OF")->ProjectionXY("2", "B"), options);
      status += equals("TH3 -> PBYX", h2YX, (TH2D*) h3->Project3DProfile("xy UF OF")->ProjectionXY("3", "B"), options);
      status += equals("TH3 -> PBYZ", h2YZ, (TH2D*) h3->Project3DProfile("zy UF OF")->ProjectionXY("4", "B"), options);
      status += equals("TH3 -> PBZX", h2ZX, (TH2D*) h3->Project3DProfile("xz UF OF")->ProjectionXY("5", "B"), options);
      status += equals("TH3 -> PBZY", h2ZY, (TH2D*) h3->Project3DProfile("yz UF OF")->ProjectionXY("6", "B"), options);
      options = 0;
      cout << "----------------------------------------------" << endl;
      
      // test directly project3dprofile
      options = cmpOptStats;
      status += equals("TH3 -> PXY", (TH2D*) pe2XY, (TH2D*) h3->Project3DProfile("yx  UF OF"), options);
      status += equals("TH3 -> PXZ", (TH2D*) pe2XZ, (TH2D*) h3->Project3DProfile("zx  UF OF"), options);
      status += equals("TH3 -> PYX", (TH2D*) pe2YX, (TH2D*) h3->Project3DProfile("xy  UF OF"), options);
      status += equals("TH3 -> PYZ", (TH2D*) pe2YZ, (TH2D*) h3->Project3DProfile("zy  UF OF"), options);
      status += equals("TH3 -> PZX", (TH2D*) pe2ZX, (TH2D*) h3->Project3DProfile("xz  UF OF"), options);
      status += equals("TH3 -> PZY", (TH2D*) pe2ZY, (TH2D*) h3->Project3DProfile("yz  UF OF"), options);
      options = 0;
      cout << "----------------------------------------------" << endl;
      
      // test option E of ProjectionXY
      options = 0;
      status += equals("TH3 -> PEXY", (TH2D*) pe2XY, (TH2D*) h3->Project3DProfile("yx  UF OF")->ProjectionXY("1", "E"), options);
      status += equals("TH3 -> PEXZ", (TH2D*) pe2XZ, (TH2D*) h3->Project3DProfile("zx  UF OF")->ProjectionXY("2", "E"), options);
      status += equals("TH3 -> PEYX", (TH2D*) pe2YX, (TH2D*) h3->Project3DProfile("xy  UF OF")->ProjectionXY("3", "E"), options);
      status += equals("TH3 -> PEYZ", (TH2D*) pe2YZ, (TH2D*) h3->Project3DProfile("zy  UF OF")->ProjectionXY("4", "E"), options);
      status += equals("TH3 -> PEZX", (TH2D*) pe2ZX, (TH2D*) h3->Project3DProfile("xz  UF OF")->ProjectionXY("5", "E"), options);
      status += equals("TH3 -> PEZY", (TH2D*) pe2ZY, (TH2D*) h3->Project3DProfile("yz  UF OF")->ProjectionXY("6", "E"), options);
      options = 0;
      cout << "----------------------------------------------" << endl;
      
      // test option W of ProjectionXY
      
      // The error fails when built with weights. It is not properly calculated
      if ( buildWithWeights ) options = cmpOptNoError;
      status += equals("TH3 -> PWXY", (TH2D*) h2wXY, (TH2D*) h3->Project3DProfile("yx  UF OF")->ProjectionXY("1", "W"), options);
      status += equals("TH3 -> PWXZ", (TH2D*) h2wXZ, (TH2D*) h3->Project3DProfile("zx  UF OF")->ProjectionXY("2", "W"), options);
      status += equals("TH3 -> PWYX", (TH2D*) h2wYX, (TH2D*) h3->Project3DProfile("xy  UF OF")->ProjectionXY("3", "W"), options);
      status += equals("TH3 -> PWYZ", (TH2D*) h2wYZ, (TH2D*) h3->Project3DProfile("zy  UF OF")->ProjectionXY("4", "W"), options);
      status += equals("TH3 -> PWZX", (TH2D*) h2wZX, (TH2D*) h3->Project3DProfile("xz  UF OF")->ProjectionXY("5", "W"), options);
      status += equals("TH3 -> PWZY", (TH2D*) h2wZY, (TH2D*) h3->Project3DProfile("yz  UF OF")->ProjectionXY("6", "W"), options);
      options = 0;
      cout << "----------------------------------------------" << endl;
      
      // test 1D histograms
      options = cmpOptStats;
      status += equals("TH2XY -> PBX", h1X, (TH1D*) h2XY->ProfileX("7", 0,h2XY->GetXaxis()->GetNbins()+1)->ProjectionX("1", "B"),options);
      status += equals("TH2XY -> PBX", h1Y, (TH1D*) h2XY->ProfileY("7", 0,h2XY->GetYaxis()->GetNbins()+1)->ProjectionX("1", "B"),options);
      status += equals("TH2XZ -> PBX", h1X, (TH1D*) h2XZ->ProfileX("7", 0,h2XY->GetXaxis()->GetNbins()+1)->ProjectionX("1", "B"),options);
      status += equals("TH2XZ -> PBZ", h1Z, (TH1D*) h2XZ->ProfileY("7", 0,h2XY->GetYaxis()->GetNbins()+1)->ProjectionX("1", "B"),options);
      status += equals("TH2YX -> PBY", h1Y, (TH1D*) h2YX->ProfileX("7", 0,h2XY->GetXaxis()->GetNbins()+1)->ProjectionX("1", "B"),options);
      status += equals("TH2YX -> PBX", h1X, (TH1D*) h2YX->ProfileY("7", 0,h2XY->GetYaxis()->GetNbins()+1)->ProjectionX("1", "B"),options);
      status += equals("TH2YZ -> PBY", h1Y, (TH1D*) h2YZ->ProfileX("7", 0,h2XY->GetXaxis()->GetNbins()+1)->ProjectionX("1", "B"),options);
      status += equals("TH2YZ -> PBZ", h1Z, (TH1D*) h2YZ->ProfileY("7", 0,h2XY->GetYaxis()->GetNbins()+1)->ProjectionX("1", "B"),options);
      status += equals("TH2ZX -> PBZ", h1Z, (TH1D*) h2ZX->ProfileX("7", 0,h2XY->GetXaxis()->GetNbins()+1)->ProjectionX("1", "B"),options);
      status += equals("TH2ZX -> PBX", h1X, (TH1D*) h2ZX->ProfileY("7", 0,h2XY->GetYaxis()->GetNbins()+1)->ProjectionX("1", "B"),options);
      status += equals("TH2ZY -> PBZ", h1Z, (TH1D*) h2ZY->ProfileX("7", 0,h2XY->GetXaxis()->GetNbins()+1)->ProjectionX("1", "B"),options);
      status += equals("TH2ZY -> PBY", h1Y, (TH1D*) h2ZY->ProfileY("7", 0,h2XY->GetYaxis()->GetNbins()+1)->ProjectionX("1", "B"),options);
      options = 0;
      cout << "----------------------------------------------" << endl;

      // 1D testing direct profiles 
      options = cmpOptStats;
      status += equals("TH2XY -> PX", pe1XY, (TH1D*) h2XY->ProfileX("7", 0,h2XY->GetXaxis()->GetNbins()+1), options);
      status += equals("TH2XY -> PX", pe1YX, (TH1D*) h2XY->ProfileY("7", 0,h2XY->GetYaxis()->GetNbins()+1), options);
      status += equals("TH2XZ -> PX", pe1XZ, (TH1D*) h2XZ->ProfileX("7", 0,h2XY->GetXaxis()->GetNbins()+1), options);
      status += equals("TH2XZ -> PZ", pe1ZX, (TH1D*) h2XZ->ProfileY("7", 0,h2XY->GetYaxis()->GetNbins()+1), options);
      status += equals("TH2YX -> PY", pe1YX, (TH1D*) h2YX->ProfileX("7", 0,h2XY->GetXaxis()->GetNbins()+1), options);
      status += equals("TH2YX -> PX", pe1XY, (TH1D*) h2YX->ProfileY("7", 0,h2XY->GetYaxis()->GetNbins()+1), options);
      status += equals("TH2YZ -> PY", pe1YZ, (TH1D*) h2YZ->ProfileX("7", 0,h2XY->GetXaxis()->GetNbins()+1), options);
      status += equals("TH2YZ -> PZ", pe1ZY, (TH1D*) h2YZ->ProfileY("7", 0,h2XY->GetYaxis()->GetNbins()+1), options);
      status += equals("TH2ZX -> PZ", pe1ZX, (TH1D*) h2ZX->ProfileX("7", 0,h2XY->GetXaxis()->GetNbins()+1), options);
      status += equals("TH2ZX -> PX", pe1XZ, (TH1D*) h2ZX->ProfileY("7", 0,h2XY->GetYaxis()->GetNbins()+1), options);
      status += equals("TH2ZY -> PZ", pe1ZY, (TH1D*) h2ZY->ProfileX("7", 0,h2XY->GetXaxis()->GetNbins()+1), options);
      status += equals("TH2ZY -> PY", pe1YZ, (TH1D*) h2ZY->ProfileY("7", 0,h2XY->GetYaxis()->GetNbins()+1), options);
      options = 0;
      cout << "----------------------------------------------" << endl;

      // 1D testing e profiles
      options = 0;
      status += equals("TH2XY -> PEX", pe1XY, 
                       (TH1D*) h2XY->ProfileX("8", 0,h2XY->GetXaxis()->GetNbins()+1)->ProjectionX("1", "E"), options);
      status += equals("TH2XY -> PEX", pe1YX, 
                       (TH1D*) h2XY->ProfileY("8", 0,h2XY->GetYaxis()->GetNbins()+1)->ProjectionX("1", "E"), options);
      status += equals("TH2XZ -> PEX", pe1XZ, 
                       (TH1D*) h2XZ->ProfileX("8", 0,h2XY->GetXaxis()->GetNbins()+1)->ProjectionX("1", "E"), options);
      status += equals("TH2XZ -> PEZ", pe1ZX, 
                       (TH1D*) h2XZ->ProfileY("8", 0,h2XY->GetYaxis()->GetNbins()+1)->ProjectionX("1", "E"), options);
      status += equals("TH2YX -> PEY", pe1YX, 
                       (TH1D*) h2YX->ProfileX("8", 0,h2XY->GetXaxis()->GetNbins()+1)->ProjectionX("1", "E"), options);
      status += equals("TH2YX -> PEX", pe1XY, 
                       (TH1D*) h2YX->ProfileY("8", 0,h2XY->GetYaxis()->GetNbins()+1)->ProjectionX("1", "E"), options);
      status += equals("TH2YZ -> PEY", pe1YZ, 
                       (TH1D*) h2YZ->ProfileX("8", 0,h2XY->GetXaxis()->GetNbins()+1)->ProjectionX("1", "E"), options);
      status += equals("TH2YZ -> PEZ", pe1ZY, 
                       (TH1D*) h2YZ->ProfileY("8", 0,h2XY->GetYaxis()->GetNbins()+1)->ProjectionX("1", "E"), options);
      status += equals("TH2ZX -> PEZ", pe1ZX, 
                       (TH1D*) h2ZX->ProfileX("8", 0,h2XY->GetXaxis()->GetNbins()+1)->ProjectionX("1", "E"), options);
      status += equals("TH2ZX -> PEX", pe1XZ, 
                       (TH1D*) h2ZX->ProfileY("8", 0,h2XY->GetYaxis()->GetNbins()+1)->ProjectionX("1", "E"), options);
      status += equals("TH2ZY -> PEZ", pe1ZY, 
                       (TH1D*) h2ZY->ProfileX("8", 0,h2XY->GetXaxis()->GetNbins()+1)->ProjectionX("1", "E"), options);
      status += equals("TH2ZY -> PEY", pe1YZ, 
                       (TH1D*) h2ZY->ProfileY("8", 0,h2XY->GetYaxis()->GetNbins()+1)->ProjectionX("1", "E"), options);
      options = 0;
      cout << "----------------------------------------------" << endl;

      // 1D testing w profiles
      // The error is not properly propagated when build with weights :S
      if ( buildWithWeights ) options = cmpOptNoError;
      status += equals("TH2XY -> PWX", hw1XY, 
                       (TH1D*) h2XY->ProfileX("7", 0,h2XY->GetXaxis()->GetNbins()+1)->ProjectionX("1", "W"), options);
      status += equals("TH2XY -> PWX", hw1YX, 
                       (TH1D*) h2XY->ProfileY("7", 0,h2XY->GetYaxis()->GetNbins()+1)->ProjectionX("1", "W"), options);
      status += equals("TH2XZ -> PWX", hw1XZ, 
                       (TH1D*) h2XZ->ProfileX("7", 0,h2XY->GetXaxis()->GetNbins()+1)->ProjectionX("1", "W"), options);
      status += equals("TH2XZ -> PWZ", hw1ZX, 
                       (TH1D*) h2XZ->ProfileY("7", 0,h2XY->GetYaxis()->GetNbins()+1)->ProjectionX("1", "W"), options);
      status += equals("TH2YX -> PWY", hw1YX, 
                       (TH1D*) h2YX->ProfileX("7", 0,h2XY->GetXaxis()->GetNbins()+1)->ProjectionX("1", "W"), options);
      status += equals("TH2YX -> PWX", hw1XY, 
                       (TH1D*) h2YX->ProfileY("7", 0,h2XY->GetYaxis()->GetNbins()+1)->ProjectionX("1", "W"), options);
      status += equals("TH2YZ -> PWY", hw1YZ, 
                       (TH1D*) h2YZ->ProfileX("7", 0,h2XY->GetXaxis()->GetNbins()+1)->ProjectionX("1", "W"), options);
      status += equals("TH2YZ -> PWZ", hw1ZY, 
                       (TH1D*) h2YZ->ProfileY("7", 0,h2XY->GetYaxis()->GetNbins()+1)->ProjectionX("1", "W"), options);
      status += equals("TH2ZX -> PWZ", hw1ZX, 
                       (TH1D*) h2ZX->ProfileX("7", 0,h2XY->GetXaxis()->GetNbins()+1)->ProjectionX("1", "W"), options);
      status += equals("TH2ZX -> PWX", hw1XZ, 
                       (TH1D*) h2ZX->ProfileY("7", 0,h2XY->GetYaxis()->GetNbins()+1)->ProjectionX("1", "W"), options);
      status += equals("TH2ZY -> PWZ", hw1ZY, 
                       (TH1D*) h2ZY->ProfileX("7", 0,h2XY->GetXaxis()->GetNbins()+1)->ProjectionX("1", "W"), options);
      status += equals("TH2ZY -> PWY", hw1YZ, 
                       (TH1D*) h2ZY->ProfileY("7", 0,h2XY->GetYaxis()->GetNbins()+1)->ProjectionX("1", "W"), options);
      options = 0;
      cout << "----------------------------------------------" << endl;
      
      // TH2 derived from STH3
      options = cmpOptStats;
      status += equals("STH3 -> XY", h2XY, (TH2D*) s3->Projection(1,0), options);
      status += equals("STH3 -> XZ", h2XZ, (TH2D*) s3->Projection(2,0), options);
      status += equals("STH3 -> YX", h2YX, (TH2D*) s3->Projection(0,1), options);
      status += equals("STH3 -> YZ", h2YZ, (TH2D*) s3->Projection(2,1), options);
      status += equals("STH3 -> ZX", h2ZX, (TH2D*) s3->Projection(0,2), options);
      status += equals("STH3 -> ZY", h2ZY, (TH2D*) s3->Projection(1,2), options);
      options = 0;
      cout << "----------------------------------------------" << endl;

      // TH1 derived from STH3
      options = cmpOptStats;
      status += equals("STH3 -> X", h1X, (TH1D*) s3->Projection(0), options);
      status += equals("STH3 -> Y", h1Y, (TH1D*) s3->Projection(1), options);
      status += equals("STH3 -> Z", h1Z, (TH1D*) s3->Projection(2), options);
      options = 0;
      cout << "----------------------------------------------" << endl;

      return status;
   }
   
};

int stressHistProj(bool testWithoutWeights = true,
                   bool testWithWeights = true)
{
   int status = 0;
   
   if ( testWithoutWeights )
   {
      cout << "**********************************\n"
           << "       Test without weights       \n" 
           << "**********************************\n"
           << endl;
      
      ProjectionTester ht;
      ht.buildHistograms();
      //ht.buildHistograms(2,4,5,6,8,10);
      status += ht.compareHistograms();
   }

   if ( testWithWeights )
   {
      cout << "**********************************\n"
           << "        Test with weights         \n" 
           << "**********************************\n"
           << endl;

      ProjectionTester ht;
      ht.buildHistogramsWithWeights();
      status += ht.compareHistograms();
   }

   return status;
}


// end of stressHistProj file

int main(int argc, char** argv)
{
   r.SetSeed(0);

   TApplication* theApp = 0;

   if ( __DRAW__ )
      theApp = new TApplication("App",&argc,argv);

   bool GlobalStatus = false;
   bool status = false;

   ostringstream output;
   output << "\nTEST RESULTS\n\n";

   cout << "\nstressHistProj\n" << endl;
   status = stressHistProj();
   GlobalStatus |= status;
   output << "stressHistProj Test.............." 
          << (status?"FAILED":"OK") << endl;

   cout << "\nstressHistRebin\n" << endl;
   status = stressHistRebin();
   GlobalStatus |= status;
   output << "stressHistRebin Test............."
          << (status?"FAILED":"OK") << endl;

   cout << "\nstressHistOpts\n" << endl;
   status = stressHistOpts();
   GlobalStatus |= status;
   output << "stressHistOpts Test.............."
          << (status?"FAILED":"OK") << endl;

   cout << output.str() << endl;

   if ( __DRAW__ ) {
      theApp->Run();
      delete theApp;
      theApp = 0;
   }

   return status;
}

ostream& operator<<(ostream& out, TH1D* h)
{
   out << h->GetName() << ": [" << h->GetBinContent(1);
   for ( Int_t i = 1; i < h->GetNbinsX(); ++i )
      out << ", " << h->GetBinContent(i);
   out << "] ";

   return out;
}

// Methods for histogram comparisions

int equals(const char* msg, TH3D* h1, TH3D* h2, int options, double ERRORLIMIT)
{
   bool debug = options & cmpOptDebug;
   bool compareError = ! (options & cmpOptNoError);
   bool compareStats = options & cmpOptStats;
   
   bool differents = ( h1 == h2 ); // Check they are not the same histogram!
   if (debug) {
      cout << static_cast<void*>(h1) << " " << static_cast<void*>(h2) << " "
           << (h1 == h2 ) << " " << differents << endl;
   }
   
   for ( int i = 0; i <= h1->GetNbinsX() + 1; ++i )
      for ( int j = 0; j <= h1->GetNbinsY() + 1; ++j )
         for ( int h = 0; h <= h1->GetNbinsY() + 1; ++h )
      {
         Double_t x = h1->GetXaxis()->GetBinCenter(i);
         Double_t y = h1->GetYaxis()->GetBinCenter(j);
         Double_t z = h1->GetZaxis()->GetBinCenter(h);
         
         if (debug)
         {
            cout << equals(x, h2->GetXaxis()->GetBinCenter(i), ERRORLIMIT) << " "
                 << equals(y, h2->GetYaxis()->GetBinCenter(j), ERRORLIMIT) << " "
                 << equals(z, h2->GetZaxis()->GetBinCenter(h), ERRORLIMIT) << " "
                 << "[" << x << "," << y << "," << z << "]: " 
                 << h1->GetBinContent(i,j,h) << " +/- " << h1->GetBinError(i,j,h) << " | "
                 << h2->GetBinContent(i,j,h) << " +/- " << h2->GetBinError(i,j,h)
                 << " | " << equals(h1->GetBinContent(i,j,h), h2->GetBinContent(i,j,h), ERRORLIMIT)
                 << " "   << equals(h1->GetBinError(i,j,h)  , h2->GetBinError(i,j,h),   ERRORLIMIT)
                 << " "   << differents
                 << " "   << (fabs(h1->GetBinContent(i,j,h) - h2->GetBinContent(i,j,h)))
                 << endl;
         }
         differents |= (bool) equals(x, h2->GetXaxis()->GetBinCenter(i), ERRORLIMIT);
         differents |= (bool) equals(y, h2->GetYaxis()->GetBinCenter(j), ERRORLIMIT);
         differents |= (bool) equals(z, h2->GetZaxis()->GetBinCenter(h), ERRORLIMIT);
         differents |= (bool) equals(h1->GetBinContent(i,j,h), h2->GetBinContent(i,j,h), ERRORLIMIT);
         if ( compareError )
            differents |= (bool) equals(h1->GetBinError(i,j,h)  , h2->GetBinError(i,j,h), ERRORLIMIT);
      }
   
   // Statistical tests:
   if ( compareStats )
     differents |= (bool) compareStatistics( h1, h2, debug, ERRORLIMIT);
   
   cout << msg << ": \t" << (differents?"FAILED":"OK") << endl;
   
   delete h2;
   
   return differents;
}

int equals(const char* msg, TH2D* h1, TH2D* h2, int options, double ERRORLIMIT)
{
   bool debug = options & cmpOptDebug;
   bool compareError = ! (options & cmpOptNoError);
   bool compareStats = options & cmpOptStats;
   
   bool differents = ( h1 == h2 ); // Check they are not the same histogram!
   if (debug) {
      cout << static_cast<void*>(h1) << " " << static_cast<void*>(h2) << " "
           << (h1 == h2 ) << " " << differents << endl;
   }

   for ( int i = 0; i <= h1->GetNbinsX() + 1; ++i )
      for ( int j = 0; j <= h1->GetNbinsY() + 1; ++j )
      {
         Double_t x = h1->GetXaxis()->GetBinCenter(i);
         Double_t y = h1->GetYaxis()->GetBinCenter(j);
         
         if (debug)
         {
            cout << equals(x, h2->GetXaxis()->GetBinCenter(i), ERRORLIMIT) << " "
                 << equals(y, h2->GetYaxis()->GetBinCenter(j), ERRORLIMIT) << " "
                 << "[" << x << "," << y << "]: " 
                 << h1->GetBinContent(i,j) << " +/- " << h1->GetBinError(i,j) << " | "
                 << h2->GetBinContent(i,j) << " +/- " << h2->GetBinError(i,j)
                 << " | " << equals(h1->GetBinContent(i,j), h2->GetBinContent(i,j), ERRORLIMIT)
                 << " "   << equals(h1->GetBinError(i,j)  , h2->GetBinError(i,j),   ERRORLIMIT)
                 << " "   << differents
                 << " "   << (fabs(h1->GetBinContent(i,j) - h2->GetBinContent(i,j)))
                 << endl;
         }
         differents |= (bool) equals(x, h2->GetXaxis()->GetBinCenter(i), ERRORLIMIT);
         differents |= (bool) equals(y, h2->GetYaxis()->GetBinCenter(j), ERRORLIMIT);
         differents |= (bool) equals(h1->GetBinContent(i,j), h2->GetBinContent(i,j), ERRORLIMIT);
         if ( compareError )
            differents |= (bool) equals(h1->GetBinError(i,j)  , h2->GetBinError(i,j), ERRORLIMIT);
      }
   
   // Statistical tests:
   if ( compareStats )
     differents |= (bool) compareStatistics( h1, h2, debug, ERRORLIMIT);
   
   cout << msg << ": \t" << (differents?"FAILED":"OK") << endl;
   
   delete h2;
   
   return differents;
}

int equals(const char* msg, TH1D* h1, TH1D* h2, int options, double ERRORLIMIT)
{
   bool debug = options & cmpOptDebug;
   bool compareError = ! (options & cmpOptNoError);
   bool compareStats = options & cmpOptStats;
   
   bool differents = ( h1 == h2 ); // Check they are not the same histogram!
   if (debug) {
      cout << static_cast<void*>(h1) << " " << static_cast<void*>(h2) << " "
           << (h1 == h2 ) << " " << differents << endl;
   }

   for ( int i = 0; i <= h1->GetNbinsX() + 1; ++i )
   {
      Double_t x = h1->GetXaxis()->GetBinCenter(i);
      if ( debug )
      {
         cout << equals(x, h2->GetXaxis()->GetBinCenter(i), ERRORLIMIT)
              << " [" << x << "]: " 
              << h1->GetBinContent(i) << " +/- " << h1->GetBinError(i) << " | "
              << h2->GetBinContent(i) << " +/- " << h2->GetBinError(i)
              << " | " << equals(h1->GetBinContent(i), h2->GetBinContent(i), ERRORLIMIT)
              << " "   << equals(h1->GetBinError(i),   h2->GetBinError(i),   ERRORLIMIT)
              << " "   << differents
              << endl;
      }
      differents |= (bool) equals(x, h2->GetXaxis()->GetBinCenter(i), ERRORLIMIT);
      differents |= (bool) equals(h1->GetBinContent(i), h2->GetBinContent(i), ERRORLIMIT);
      
      if ( compareError )
         differents |= (bool) equals(h1->GetBinError(i),   h2->GetBinError(i), ERRORLIMIT);
   }
   
   // Statistical tests:
   if ( compareStats )
     differents |= (bool) compareStatistics( h1, h2, debug, ERRORLIMIT);
   
   cout << msg << ": \t" << (differents?"FAILED":"OK") << endl;
   
   delete h2;
   
   return differents;      
}

int equals(Double_t n1, Double_t n2, double ERRORLIMIT)
{
   return fabs( n1 - n2 ) > ERRORLIMIT * fabs(n1);
}

int compareStatistics( TH1* h1, TH1* h2, bool debug, double ERRORLIMIT)
{
   bool differents = 0;

   int precLevel = gErrorIgnoreLevel; 
   // switch off Info mesaage from chi2 test
   if (!debug) gErrorIgnoreLevel = 1001; 
            
   if (debug) h2->Print(); 
   
   std::string option = "WW OF UF";
   const char * opt = option.c_str(); 
   differents |= (h1->Chi2Test(h2, opt) < 1);
   differents |= (h2->Chi2Test(h1,opt) < 1);         
   differents |= (bool) equals(h1->Chi2Test(h2,opt), h2->Chi2Test(h1,opt), ERRORLIMIT);
   if ( debug )
      cout << "Chi2Test " << h1->Chi2Test(h2, opt) << " " << h2->Chi2Test(h1, opt) 
           << " | " << differents
           << endl;

   if (!debug) gErrorIgnoreLevel = precLevel; 

   // Mean
   differents |= (bool) equals(h1->GetMean(1), h2->GetMean(1), ERRORLIMIT);
   if ( debug )
      cout << "Mean: " << h1->GetMean(1) << " " << h2->GetMean(1) 
           << " | " << fabs( h1->GetMean(1) - h2->GetMean(1) ) 
           << " " << differents
           << endl;
   
   // RMS
   differents |= (bool) equals( h1->GetRMS(1), h2->GetRMS(1), ERRORLIMIT);
   if ( debug )
      cout << "RMS: " << h1->GetRMS(1) << " " << h2->GetRMS(1) 
           << " | " << fabs( h1->GetRMS(1) - h2->GetRMS(1) ) 
           << " " << differents
           << endl;      
   
   return differents;
}
