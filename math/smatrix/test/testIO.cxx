//
// Cint macro to test I/O of SMatrix classes and compare with a TMatrix
// A ROOT tree is written and read in both using either a SMatrix  or
/// a TMatrixD.
//
//  To execute the macro type in:
//
// root[0]: .x  smatrixIO.C

#include "Math/SMatrix.h"
#include "TMatrixD.h"
#include "TMatrixDSym.h"
#include "TRandom3.h"
#include "TFile.h"
#include "TTree.h"
#include "TStopwatch.h"
#include "TSystem.h"

#include <iostream>

#ifdef USE_REFLEX
#include "Cintex/Cintex.h"
#include "Reflex/Reflex.h"
#endif

#include "Track.h"

TRandom3 R;
TStopwatch timer;


//#define DEBUG

// if I use double32 or not depends on the dictionary
////#define USE_DOUBLE32
#ifdef USE_DOUBLE32
typedef ROOT::Math::SMatrix<Double32_t,5,5,ROOT::Math::MatRepSym<Double32_t,5> >  SMatrixSym5;
typedef ROOT::Math::SMatrix<Double32_t,5,5 >  SMatrix5;
const std::string sname = "ROOT::Math::SMatrix<Double32_t,5,5,ROOT::Math::MatRepStd<Double32_t,5,5> >";
const std::string sname_sym = "ROOT::Math::SMatrix<Double32_t,5,5,ROOT::Math::MatRepSym<Double32_t,5> >";
double tol = 1.E-6;
#else
typedef ROOT::Math::SMatrix<double,5,5,ROOT::Math::MatRepSym<double,5> >  SMatrixSym5;
typedef ROOT::Math::SMatrix<double,5,5 >  SMatrix5;
const std::string sname = "ROOT::Math::SMatrix<double,5,5,ROOT::Math::MatRepStd<double,5,5> >";
const std::string sname_sym = "ROOT::Math::SMatrix<double,5,5,ROOT::Math::MatRepSym<double,5> >";
double tol = 1.E-16;
#endif
double tol32 = 1.E-6;

#ifdef USE_REFLEX
  std::string sfile1   = "smatrix_rflx.root";
  std::string sfile2   = "smatrix.root";

  std::string symfile1 = "smatrixsym_rflx.root";
  std::string symfile2 = "smatrixsym.root";
#else
  std::string sfile1   = "smatrix.root";
  std::string sfile2   = "smatrix_rflx.root";

  std::string symfile1 = "smatrixsym.root";
  std::string symfile2 = "smatrixsym_rflx.root";
#endif



//using namespace ROOT::Math;


template<class Matrix>
void FillMatrix(Matrix & m) {
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      m(i,j) = R.Rndm() + 1;
    }
  }
}

void FillCArray(double * m) {
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      m[i*5+j] = R.Rndm() + 1;
    }
  }
}

template<class Matrix>
void FillMatrixSym(Matrix & m) {
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      if (j>=i) m(i,j) = R.Rndm() + 1;
      else m(i,j) = m(j,i);
    }
  }
}

template<class R>
double SumSMatrix(ROOT::Math::SMatrix<double,5,5,R>  & m) {
  double sum = 0;
  for (int i = 0; i < 5*5; ++i) {
    sum += m.apply(i);
  }
  return sum;
}

double SumCArray(double *  m) {
  double sum = 0;
  for (int i = 0; i < 5*5; ++i) {
    sum += m[i];
  }
  return sum;
}

template<class TM>
double SumTMatrix(TM & m) {
  double sum = 0;
  const double * d = m.GetMatrixArray();
  for (int i = 0; i < 5*5; ++i) {
    sum += d[i];
  }
  return sum;
}



void initMatrix(int n) {

  //  using namespace ROOT::Math;

  timer.Start();
  SMatrix5 s;
  R.SetSeed(1);   // use same seed
  for (int i = 0; i < n; ++i) {
    FillMatrix(s);
  }
  timer.Stop();
  std::cout << " Time to fill SMatrix     " << timer.RealTime() << "  "  << timer.CpuTime() << std::endl;

  timer.Start();
  SMatrixSym5 ss;
  R.SetSeed(1);   // use same seed
  for (int i = 0; i < n; ++i) {
    FillMatrixSym(ss);
  }
  timer.Stop();
  std::cout << " Time to fill SMatrix Sym " << timer.RealTime() << "  "  << timer.CpuTime() << std::endl;

  timer.Start();
  TMatrixD  t(5,5);
  R.SetSeed(1);   // use same seed
  for (int i = 0; i < n; ++i) {
    FillMatrix(t);
  }
  timer.Stop();
  std::cout << " Time to fill TMatrix     " << timer.RealTime() << "  " << timer.CpuTime() << std::endl;

  timer.Start();
  TMatrixDSym  ts(5);
  R.SetSeed(1);   // use same seed
  for (int i = 0; i < n; ++i) {
    FillMatrixSym(ts);
  }
  timer.Stop();
  std::cout << " Time to fill TMatrix Sym " << timer.RealTime() << "  " << timer.CpuTime() << std::endl;

}



double writeCArray(int n) {

  std::cout << "\n";
  std::cout << "**************************************************\n";
  std::cout << "  Test writing a C Array ........\n";
  std::cout << "**************************************************\n";

  TFile f1("cmatrix.root","RECREATE");


  // create tree
  TTree t1("t1","Tree with C Array");

  double m1[25];
  t1.Branch("C Array branch",m1,"m1[25]/D");

  timer.Start();
  double etot = 0;
  R.SetSeed(1);   // use same seed
  for (int i = 0; i < n; ++i) {
    FillCArray(m1);
    etot += SumCArray(m1);
    t1.Fill();
  }

  f1.Write();
  timer.Stop();

  std::cout << " Time to Write CArray " << timer.RealTime() << "  " << timer.CpuTime() << std::endl;

#ifdef DEBUG
  t1.Print();
  int pr = std::cout.precision(18);
  std::cout << " sum " << n<< "  " << etot << "  " << etot/double(n) << std::endl;
  std::cout.precision(pr);
#endif

  return etot/double(n);
}

double writeSMatrix(int n, const std::string & file) {

  std::cout << "\n";
  std::cout << "**************************************************\n";
  std::cout << "  Test writing SMatrix ........\n";
  std::cout << "**************************************************\n";

  TFile f1(file.c_str(),"RECREATE");

  // create tree
  TTree t1("t1","Tree with SMatrix");

  SMatrix5 * m1 = new  SMatrix5;
  //t1.Branch("SMatrix branch",&m1,16000,0);
  // need to pass type name explicitly to distinguish double/double32
  t1.Branch("SMatrix branch",sname.c_str(),&m1,16000,0);

  timer.Start();
  double etot = 0;
  R.SetSeed(1);   // use same seed
  for (int i = 0; i < n; ++i) {
    FillMatrix(*m1);
    etot += SumSMatrix(*m1);
    t1.Fill();
  }

  f1.Write();
  timer.Stop();

  std::cout << " Time to Write SMatrix " << timer.RealTime() << "  " << timer.CpuTime() << std::endl;
#ifdef DEBUG
  t1.Print();
  int pr = std::cout.precision(18);
  std::cout << " sum " << n<< "  " << etot << "  " << etot/double(n) << std::endl;
  std::cout.precision(pr);
#endif

  return etot/double(n);
}



double writeSMatrixSym(int n, const std::string & file) {

  std::cout << "\n";
  std::cout << "**************************************************\n";
  std::cout << "  Test writing SMatrix Sym.....\n";
  std::cout << "**************************************************\n";

  TFile f1(file.c_str(),"RECREATE");

  // create tree
  TTree t1("t1","Tree with SMatrix");

  SMatrixSym5 * m1 = new  SMatrixSym5;
  //t1.Branch("SMatrixSym branch",&m1,16000,0);
  t1.Branch("SMatrixSym branch",sname_sym.c_str(),&m1,16000,0);

  timer.Start();
  double etot = 0;
  R.SetSeed(1);   // use same seed
  for (int i = 0; i < n; ++i) {
    FillMatrixSym(*m1);
    etot += SumSMatrix(*m1);
    t1.Fill();
  }

  f1.Write();
  timer.Stop();


  std::cout << " Time to Write SMatrix Sym " << timer.RealTime() << "  " << timer.CpuTime() << std::endl;
#ifdef DEBUG
  t1.Print();
  int pr = std::cout.precision(18);
  std::cout << " sum " << n<< "  " << etot << "  " << etot/double(n) << std::endl;
  std::cout.precision(pr);
#endif

  return etot/double(n);
}





double writeTMatrix(int n) {

  // create tree with TMatrix
  std::cout << "\n";
  std::cout << "**************************************************\n";
  std::cout << "  Test writing TMatrix........\n";
  std::cout << "**************************************************\n";


  TFile f2("tmatrix.root","RECREATE");
  TTree t2("t2","Tree with TMatrix");

  TMatrixD * m2 = new TMatrixD(5,5);
  TMatrixD::Class()->IgnoreTObjectStreamer();

  //t2.Branch("TMatrix branch","TMatrixD",&m2,16000,2);
  t2.Branch("TMatrix branch",&m2,16000,2);

  double etot = 0;
  timer.Start();
  R.SetSeed(1);   // use same seed
  for (int i = 0; i < n; ++i) {
    FillMatrix(*m2);
    etot += SumTMatrix(*m2);
    t2.Fill();
  }

  f2.Write();
  timer.Stop();

  std::cout << " Time to Write TMatrix " << timer.RealTime() << "  " << timer.CpuTime() << std::endl;
#ifdef DEBUG
  t2.Print();
  int pr = std::cout.precision(18);
  std::cout << " sum " << n<< "  " << etot << "  " << etot/double(n) << std::endl;
  std::cout.precision(pr);
  std::cout << "\n\n\n";
#endif

  return etot/double(n);

}

double writeTMatrixSym(int n) {

  // create tree with TMatrix
  std::cout << "\n";
  std::cout << "**************************************************\n";
  std::cout << "  Test writing TMatrix.Sym....\n";
  std::cout << "**************************************************\n";


  TFile f2("tmatrixsym.root","RECREATE");
  TTree t2("t2","Tree with TMatrix");

  TMatrixDSym * m2 = new TMatrixDSym(5);
  TMatrixDSym::Class()->IgnoreTObjectStreamer();

  //t2.Branch("TMatrix branch","TMatrixDSym",&m2,16000,0);
  t2.Branch("TMatrixSym branch",&m2,16000,0);

  double etot = 0;
  timer.Start();
  R.SetSeed(1);   // use same seed
  for (int i = 0; i < n; ++i) {
    FillMatrixSym(*m2);
    etot += SumTMatrix(*m2);
    t2.Fill();
  }

  f2.Write();
  timer.Stop();

  std::cout << " Time to Write TMatrix Sym " << timer.RealTime() << "  " << timer.CpuTime() << std::endl;
#ifdef DEBUG
  t2.Print();
  int pr = std::cout.precision(18);
  std::cout << " sum " << n<< "  " << etot << "  " << etot/double(n) << std::endl;
  std::cout.precision(pr);
  std::cout << "\n\n\n";
#endif

  return etot/double(n);

}




double readTMatrix() {


  // read tree with old TMatrix

  std::cout << "\n\n";
  std::cout << "**************************************************\n";
  std::cout << "  Test reading TMatrix........\n";
  std::cout << "**************************************************\n";


  TFile f2("tmatrix.root");
  if (f2.IsZombie() ) {
    std::cerr << "Error opening the ROOT file" << std::endl;
    return -1;
  }
  TTree *t2 = (TTree*)f2.Get("t2");


  TMatrixD * v2 = 0;
  t2->SetBranchAddress("TMatrix branch",&v2);

  timer.Start();
  int n = (int) t2->GetEntries();
  double etot = 0;
  for (int i = 0; i < n; ++i) {
    t2->GetEntry(i);
    etot += SumTMatrix(*v2);
  }

  timer.Stop();
  std::cout << " Time for TMatrix " << timer.RealTime() << "  " << timer.CpuTime() << std::endl;
  double val = etot/double(n);
#ifdef DEBUG
  std::cout << " Tree Entries " << n << std::endl;
  int pr = std::cout.precision(18);
  std::cout << " sum " << n<< "  " << etot << "  " << etot/double(n) << std::endl;
  std::cout.precision(pr);
#endif
  return val;
}


double readTMatrixSym() {


  // read tree with old TMatrix

  std::cout << "\n\n";
  std::cout << "**************************************************\n";
  std::cout << "  Test reading TMatrix.Sym....\n";
  std::cout << "**************************************************\n";


  TFile f2("tmatrixsym.root");
  if (f2.IsZombie() ) {
    std::cerr << "Error opening the ROOT file" << std::endl;
    return -1;
  }


  TTree *t2 = (TTree*)f2.Get("t2");


  TMatrixDSym * v2 = 0;
  t2->SetBranchAddress("TMatrixSym branch",&v2);

  timer.Start();
  int n = (int) t2->GetEntries();
  double etot = 0;
  for (int i = 0; i < n; ++i) {
    t2->GetEntry(i);
    etot += SumTMatrix(*v2);
  }

  timer.Stop();
  std::cout << " Time for TMatrix Sym" << timer.RealTime() << "  " << timer.CpuTime() << std::endl;
  double val = etot/double(n);
#ifdef DEBUG
  std::cout << " Tree Entries " << n << std::endl;
  int pr = std::cout.precision(18);
  std::cout << " sum " << n<< "  " << etot << "  " << etot/double(n) << std::endl;
  std::cout.precision(pr);
#endif

  return val;
}


double readSMatrix(const std::string & file) {


  std::cout << "\n";
  std::cout << "**************************************************\n";
  std::cout << "  Test reading SMatrix........\n";
  std::cout << "**************************************************\n";


  TFile f1(file.c_str());
  if (f1.IsZombie() ) {
    std::cerr << "Error opening the ROOT file" << file << std::endl;
    return -1;
  }

  // create tree
  TTree *t1 = (TTree*)f1.Get("t1");

  SMatrix5 *v1 = 0;
  t1->SetBranchAddress("SMatrix branch",&v1);

  timer.Start();
  int n = (int) t1->GetEntries();
  double etot=0;
  for (int i = 0; i < n; ++i) {
    t1->GetEntry(i);
    etot += SumSMatrix(*v1);
  }


  timer.Stop();
  std::cout << " Time for SMatrix :    " << timer.RealTime() << "  " << timer.CpuTime() << std::endl;

#ifdef DEBUG
  std::cout << " Tree Entries " << n << std::endl;
  int pr = std::cout.precision(18);
  std::cout << " sum " << n<< "  " << etot << "  " << etot/double(n) << std::endl;
  std::cout.precision(pr);
#endif
  std::cout << "\n";


  return etot/double(n);
}


double readSMatrixSym(const std::string & file) {


  std::cout << "\n";
  std::cout << "**************************************************\n";
  std::cout << "  Test reading SMatrix.Sym....\n";
  std::cout << "**************************************************\n";


  TFile f1(file.c_str());
  if (f1.IsZombie() ) {
    std::cerr << "Error opening the ROOT file" << file << std::endl;
    return -1;
  }


  // create tree
  TTree *t1 = (TTree*)f1.Get("t1");

  SMatrixSym5 *v1 = 0;
  t1->SetBranchAddress("SMatrixSym branch",&v1);

  timer.Start();
  int n = (int) t1->GetEntries();
  double etot=0;
  for (int i = 0; i < n; ++i) {
    t1->GetEntry(i);
    etot += SumSMatrix(*v1);
  }


  timer.Stop();
  std::cout << " Time for SMatrix Sym : " << timer.RealTime() << "  " << timer.CpuTime() << std::endl;

#ifdef DEBUG
  std::cout << " Tree Entries " << n << std::endl;
  int pr = std::cout.precision(18);
  std::cout << " sum " << n<< "  " << etot << "  " << etot/double(n) << std::endl;
  std::cout.precision(pr);
#endif
  std::cout << "\n";

  return etot/double(n);
}


double writeTrackD(int n) {

  std::cout << "\n";
  std::cout << "**************************************************\n";
  std::cout << "  Test writing Track class........\n";
  std::cout << "**************************************************\n";

  TFile f1("track.root","RECREATE");

  // create tree
  TTree t1("t1","Tree with Track based on SMatrix");

  TrackD * m1 = new TrackD();

  t1.Branch("Track branch",&m1,16000,0);

  timer.Start();
  double etot = 0;
  R.SetSeed(1);   // use same seed
  for (int i = 0; i < n; ++i) {
    FillMatrix(m1->CovMatrix());
    etot += SumSMatrix(m1->CovMatrix() );
    t1.Fill();
  }

  f1.Write();
  timer.Stop();

  std::cout << " Time to Write TrackD of SMatrix " << timer.RealTime() << "  " << timer.CpuTime() << std::endl;

#ifdef DEBUG
  t1.Print();
  int pr = std::cout.precision(18);
  std::cout << " sum " << n<< "  " << etot << "  " << etot/double(n) << std::endl;
  std::cout.precision(pr);
#endif

  return etot/double(n);
}


double writeTrackD32(int n) {

  std::cout << "\n";
  std::cout << "**************************************************\n";
  std::cout << "  Test writing TrackD32 class........\n";
  std::cout << "**************************************************\n";

  TFile f1("track32.root","RECREATE");

  // create tree
  TTree t1("t1","Tree with Track based on SMatrix");

  TrackD32 * m1 = new TrackD32();
  t1.Branch("Track32 branch",&m1,16000,0);

  timer.Start();
  double etot = 0;
  R.SetSeed(1);   // use same seed
  for (int i = 0; i < n; ++i) {
    FillMatrix(m1->CovMatrix());
    etot += SumSMatrix(m1->CovMatrix() );
    t1.Fill();
  }

  f1.Write();
  timer.Stop();

  std::cout << " Time to Write TrackD32 of SMatrix " << timer.RealTime() << "  " << timer.CpuTime() << std::endl;

#ifdef DEBUG
  t1.Print();
  int pr = std::cout.precision(18);
  std::cout << " sum " << n<< "  " << etot << "  " << etot/double(n) << std::endl;
  std::cout.precision(pr);
#endif

  return etot/double(n);
}


double readTrackD() {

  std::cout << "\n";
  std::cout << "**************************************************\n";
  std::cout << "  Test reading Track class........\n";
  std::cout << "**************************************************\n";

  TFile f1("track.root");
  if (f1.IsZombie() ) {
    std::cerr << "Error opening the ROOT file" << std::endl;
    return -1;
  }

  // create tree
  TTree *t1 = (TTree*)f1.Get("t1");

  TrackD *trk = 0;
  t1->SetBranchAddress("Track branch",&trk);

  timer.Start();
  int n = (int) t1->GetEntries();
  double etot=0;
  for (int i = 0; i < n; ++i) {
    t1->GetEntry(i);
    etot += SumSMatrix(trk->CovMatrix());
  }

  timer.Stop();

  std::cout << " Time to Read TrackD of SMatrix " << timer.RealTime() << "  " << timer.CpuTime() << std::endl;

#ifdef DEBUG
  std::cout << " Tree Entries " << n << std::endl;
  int pr = std::cout.precision(18);
  std::cout << " sum " << n<< "  " << etot << "  " << etot/double(n) << std::endl;
  std::cout.precision(pr);
#endif

  return etot/double(n);
}

double readTrackD32() {

  std::cout << "\n";
  std::cout << "**************************************************\n";
  std::cout << "  Test reading Track32 class........\n";
  std::cout << "**************************************************\n";

  TFile f1("track32.root");
  if (f1.IsZombie() ) {
    std::cerr << "Error opening the ROOT file" << std::endl;
    return -1;
  }

  // create tree
  TTree *t1 = (TTree*)f1.Get("t1");

  TrackD32 *trk = 0;
  t1->SetBranchAddress("Track32 branch",&trk);

  timer.Start();
  int n = (int) t1->GetEntries();
  double etot=0;
  for (int i = 0; i < n; ++i) {
    t1->GetEntry(i);
    etot += SumSMatrix(trk->CovMatrix());
  }

  timer.Stop();

  std::cout << " Time to Read TrackD32 of SMatrix " << timer.RealTime() << "  " << timer.CpuTime() << std::endl;

#ifdef DEBUG
  std::cout << " Tree Entries " << n << std::endl;
  int pr = std::cout.precision(18);
  std::cout << " sum " << n<< "  " << etot << "  " << etot/double(n) << std::endl;
  std::cout.precision(pr);
#endif

  return etot/double(n);
}


//-----------------------------------------------------------------


int testWrite(int nEvents, double & w1, double & w2) {

  int iret = 0;
  double w0 = writeCArray(nEvents);

  w1 = writeTMatrix(nEvents);
  w2 = writeSMatrix(nEvents,sfile1);
  if ( fabs(w1-w2) > tol) {
    std::cout << "\nERROR: Differeces SMatrix-TMatrix found  when writing" << std::endl;
    int pr = std::cout.precision(18);  std::cout << w1 << "   !=    " << w2 << std::endl; std::cout.precision(pr);
    iret = 1;
  }
  if ( fabs(w1-w0) > tol) {
    std::cout << "\nERROR: Differeces TMatrix-C Array found  when writing" << std::endl;
    int pr = std::cout.precision(18);  std::cout << w1 << "   !=    " << w0 << std::endl; std::cout.precision(pr);
    iret = 1;
  }

  std::cout << "\n\n*************************************************************\n";
   if (iret == 0 )
     std::cout << "  Writing Test:\t" << "OK";
   else {
     std::cout << "  Writing Test:\t" << "FAILED";
   }
   std::cout << "\n*************************************************************\n\n";


  return iret;
}

int testRead(double & r1, double & r2, double & r3) {

  int iret = 0;



  r1 = readTMatrix();
  r2 = readSMatrix(sfile1);
  if ( fabs(r1-r2) > tol) {
    std::cout << "\nERROR: Differeces SMatrix-TMatrix found  when reading " << std::endl;
    int pr = std::cout.precision(18);  std::cout << r1 << "   !=    " << r2 << std::endl; std::cout.precision(pr);
    iret = 2;
  }

#ifdef USE_REFLEX
  std::cout << "try to read file written with CINT using Reflex Dictionaries " << std::endl;
#else
  std::cout << "try to read file written with Reflex using CINT Dictionaries " << std::endl;
#endif
  r3 = readSMatrix(sfile2);
  if ( r3 != -1. && fabs(r2-r3) > tol) {
    std::cout << "\nERROR: Differeces Reflex-CINT found  when reading SMatrices" << std::endl;
    int pr = std::cout.precision(18);  std::cout << r2 << "   !=    " << r3 << std::endl; std::cout.precision(pr);
    iret = 3;
  }


  return iret;
}


int testWriteSym(int nEvents, double & w1, double & w2) {

  int iret = 0;



  w1 = writeTMatrixSym(nEvents);
  w2 = writeSMatrixSym(nEvents,symfile1);
  if ( fabs(w1-w2) > tol) {
    std::cout << "\nERROR: Differeces found  when writing" << std::endl;
    int pr = std::cout.precision(18);  std::cout << w1 << "   !=    " << w2 << std::endl; std::cout.precision(pr);
    iret = 11;
  }

  std::cout << "\n\n*************************************************************\n";
  if (iret == 0 )
    std::cout << "  Writing Test:\t" << "OK";
  else {
    std::cout << "  Writing Test:\t" << "FAILED";
  }
  std::cout << "\n*************************************************************\n\n";

  return iret;
}

int testReadSym(double & r1, double & r2, double & r3) {

  int iret = 0;


  r1 = readTMatrixSym();
  r2 = readSMatrixSym(symfile1);
  if ( fabs(r1-r2) > tol) {
    std::cout << "\nERROR: Differeces SMatrixSym-TMAtrixSym found  when reading " << std::endl;
    int pr = std::cout.precision(18);  std::cout << r1 << "   !=    " << r2 << std::endl; std::cout.precision(pr);
    iret = 12;
  }

#ifdef USE_REFLEX
  std::cout << "try to read file written with CINT using Reflex Dictionaries " << std::endl;
#else
  std::cout << "try to read file written with Reflex using CINT Dictionaries " << std::endl;
#endif

  r3 = readSMatrixSym(symfile2);
  if ( r3 != -1. && fabs(r2-r3) > tol) {
    std::cout << "\nERROR: Differeces Reflex-CINT found  when reading SMatricesSym" << std::endl;
    int pr = std::cout.precision(18);  std::cout << r2 << "   !=    " << r3 << std::endl; std::cout.precision(pr);
    iret = 13;
  }


  return iret;
}
int testResult(double w1, double r1, double w2, double r2, double r3) {

  int iret = 0;

  if ( fabs(w1-r1)  > tol) {
    std::cout << "\nERROR: Differeces found  when reading TMatrices" << std::endl;
    int pr = std::cout.precision(18);  std::cout << w1 << "   !=    " << r1 << std::endl; std::cout.precision(pr);
    iret = -1;
  }
  if ( fabs(w2-r2)  > tol) {
    std::cout << "\nERROR: Differeces found  when reading SMatrices" << std::endl;
    int pr = std::cout.precision(18);  std::cout << w2 << "   !=    " << r2 << std::endl; std::cout.precision(pr);
    iret = -2;
  }
  if ( r3 != -1. && fabs(w2-r3)  > tol) {
    std::cout << "\nERROR: Differeces found  when reading SMatrices with different Dictionary" << std::endl;
    int pr = std::cout.precision(18);  std::cout << w2 << "   !=    " << r2 << std::endl; std::cout.precision(pr);
    iret = -3;
  }
  return iret;
}

int testTrack(int nEvents) {

  int iret = 0;

  double wt1 = writeTrackD(nEvents);

#ifdef  USE_REFLEX
  // for the double32 need ROOT Cint
  gSystem->Load("libSmatrix");
#endif

  double wt2 = writeTrackD32(nEvents);

  if ( fabs(wt2-wt1)  > tol) {
    std::cout << "\nERROR: Differeces found  when writing Track" << std::endl;
    int pr = std::cout.precision(18);  std::cout << wt2 << "   !=    " << wt1 << std::endl; std::cout.precision(pr);
    iret = 13;
  }

  double rt1 = readTrackD();
  if ( fabs(rt1-wt1)  > tol) {
    std::cout << "\nERROR: Differeces found  when reading Track" << std::endl;
    int pr = std::cout.precision(18);  std::cout << rt1 << "   !=    " << wt1 << std::endl; std::cout.precision(pr);
    iret = 13;
  }

  double rt2 = readTrackD32();
  if ( fabs(rt2-wt2)  > tol32) {
    std::cout << "\nERROR: Differeces found  when reading Track 32" << std::endl;
    int pr = std::cout.precision(18);  std::cout << rt2 << "   !=    " << wt2 << std::endl; std::cout.precision(pr);
    iret = 13;
  }

  return iret;
}


int testIO() {


  int iret = 0;


#ifdef USE_REFLEX


  gSystem->Load("libReflex");
  gSystem->Load("libCintex");
  ROOT::Cintex::Cintex::SetDebug(1);
  ROOT::Cintex::Cintex::Enable();

  std::cout << "Use Reflex dictionary " << std::endl;

#ifdef USE_REFLEX_SMATRIX
  iret |= gSystem->Load("libSmatrixRflx");
#endif
  iret |= gSystem->Load("libSmatrix");


#else

  iret |= gSystem->Load("libSmatrix");

#endif

  iret |= gSystem->Load("libMatrix");


  int nEvents = 10000;

  initMatrix(nEvents);

  double w1, w2 = 0;
  iret |= testWrite(nEvents,w1,w2);



  double r1, r2, r3  = 0;
  int iret2 = 0;
  iret2 |= testRead(r1,r2,r3);
  iret2 |= testResult(w1,r1,w2,r2,r3);
  std::cout << "\n\n*************************************************************\n";
  if (iret2 == 0 )
    std::cout << "  Reading Test:\t" << "OK";
  else {
    std::cout << "  Reading Test:\t" << "FAILED";
  }
  std::cout << "\n*************************************************************\n\n";

  iret |= iret2;


  std::cout << "\n*****************************************************\n";
  std::cout << "    Test Symmetric matrices";
  std::cout << "\n*****************************************************\n\n";

  iret = testWriteSym(nEvents,w1,w2);
  iret2 = testReadSym(r1,r2,r3);
  iret2 = testResult(w1,r1,w2,r2,r3);

  std::cout << "\n\n*************************************************************\n";
  if (iret2 == 0 )
    std::cout << "  Reading Test:\t" << "OK";
  else {
    std::cout << "  Reading Test:\t" << "FAILED";
  }
  std::cout << "\n*************************************************************\n\n";

  iret |= iret2;


  std::cout << "\n*****************************************************\n";
  std::cout << "    Test Track class";
  std::cout << "\n*****************************************************\n\n";

  // load track dictionary
  iret |= gSystem->Load("libTrackDict");
  if (iret != 0 ) return iret;

  iret |= testTrack(nEvents);
  std::cout << "\n\n*************************************************************\n";
  if (iret2 == 0 )
    std::cout << "  Track  Test:\t" << "OK";
  else {
    std::cout << "  Track  Test:\t" << "FAILED";
  }
  std::cout << "\n*************************************************************\n\n";


  return iret;

}



int main() {
  int iret = testIO();
  std::cout << "\n\n*************************************************************\n";
  if (iret != 0) {
    std::cerr << "\nERROR !!!!! " << iret << std::endl;
    std::cerr << "TESTIO \t FAILED " << std::endl;
  }
  else
    std::cerr << "TESTIO \t OK " << std::endl;
  return iret;
  std::cout << "*************************************************************\n\n";
}
