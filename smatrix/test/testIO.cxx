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

TRandom3 R; 
TStopwatch timer;

// if I use double32 or not depends on the dictionary
//#define USE_DOUBLE32
#ifndef USE_DOUBLE32
typedef ROOT::Math::SMatrix<double,5,5,ROOT::Math::MatRepSym<double,5> >  SMatrixSym5;
typedef ROOT::Math::SMatrix<double,5,5 >  SMatrix5;
double tol = 1.E-16;
#else
typedef ROOT::Math::SMatrix<Double32_t,5,5,ROOT::Math::MatRepSym<Double32_t,5> >  SMatrixSym5;
typedef ROOT::Math::SMatrix<Double32_t,5,5 >  SMatrix5;
double tol = 1.E-6; 
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

  gSystem->Load("libSmatrix");  
  gSystem->Load("libMatrix");  
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


double writeSMatrix(int n) { 

  std::cout << "\n";
  std::cout << "**************************************************\n";
  std::cout << "  Test writing SMatrix ........\n";
  std::cout << "**************************************************\n";

  TFile f1("smatrix.root","RECREATE");

  // create tree
  TTree t1("t1","Tree with SMatrix");

  SMatrix5 * m1 = new  SMatrix5;
  //t1.Branch("SMatrix branch","ROOT::Math::SMatrix<Double32_t,5,5>",&m1);
  t1.Branch("SMatrix branch",&m1);

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

  t1.Print();
  std::cout << " Time to Write SMatrix " << timer.RealTime() << "  " << timer.CpuTime() << std::endl; 
  std::cout << " sum " << n<< "  " << etot << "  " << etot/double(n) << std::endl; 

  return etot/double(n);
}



double writeSMatrixSym(int n) { 

  std::cout << "\n";
  std::cout << "**************************************************\n";
  std::cout << "  Test writing SMatrix Sym.....\n";
  std::cout << "**************************************************\n";

  TFile f1("smatrixsym.root","RECREATE");

  // create tree
  TTree t1("t1","Tree with SMatrix");

  SMatrixSym5 * m1 = new  SMatrixSym5;
  // t1.Branch("SMatrixSym branch","ROOT::Math::SMatrix<Double32_t,5,5,ROOT::Math::MatRepSym<Double32_t,5> >",&m1);
  t1.Branch("SMatrixSym branch",&m1);

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

  t1.Print();
  std::cout << " Time to Write SMatrix " << timer.RealTime() << "  " << timer.CpuTime() << std::endl; 
  std::cout << " sum " << n<< "  " << etot << "  " << etot/double(n) << std::endl; 

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

  t2.Print();
  std::cout << " Time to Write TMatrix " << timer.RealTime() << "  " << timer.CpuTime() << std::endl; 
  std::cout << " sum " << n<< "  " << etot << "  " << etot/double(n) << std::endl; 
  std::cout << "\n\n\n";

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

  t2.Print();
  std::cout << " Time to Write TMatrix Sym " << timer.RealTime() << "  " << timer.CpuTime() << std::endl; 
  std::cout << " sum " << n<< "  " << etot << "  " << etot/double(n) << std::endl; 
  std::cout << "\n\n\n";

  return etot/double(n);

}




double readTMatrix() { 


  // read tree with old TMatrix

  std::cout << "\n\n";
  std::cout << "**************************************************\n";
  std::cout << "  Test reading TMatrix........\n";
  std::cout << "**************************************************\n";


  TFile f2("tmatrix.root");
  TTree *t2 = (TTree*)f2.Get("t2");


  TMatrixD * v2 = 0;
  t2->SetBranchAddress("TMatrix branch",&v2);

  timer.Start();
  int n = (int) t2->GetEntries();
  std::cout << " Tree Entries " << n << std::endl; 
  double etot = 0;
  for (int i = 0; i < n; ++i) { 
    t2->GetEntry(i);
    etot += SumTMatrix(*v2);
  }

  timer.Stop();
  std::cout << " Time for TMatrix " << timer.RealTime() << "  " << timer.CpuTime() << std::endl; 
  double val = etot/double(n);
  std::cout << " sum " << n<< "  " << etot << "  " << val << std::endl; 
  return val;
}


double readTMatrixSym() { 


  // read tree with old TMatrix

  std::cout << "\n\n";
  std::cout << "**************************************************\n";
  std::cout << "  Test reading TMatrix.Sym....\n";
  std::cout << "**************************************************\n";


  TFile f2("tmatrixsym.root");
  TTree *t2 = (TTree*)f2.Get("t2");


  TMatrixDSym * v2 = 0;
  t2->SetBranchAddress("TMatrixSym branch",&v2);

  timer.Start();
  int n = (int) t2->GetEntries();
  std::cout << " Tree Entries " << n << std::endl; 
  double etot = 0;
  for (int i = 0; i < n; ++i) { 
    t2->GetEntry(i);
    etot += SumTMatrix(*v2);
  }

  timer.Stop();
  std::cout << " Time for TMatrix Sym" << timer.RealTime() << "  " << timer.CpuTime() << std::endl; 
  double val = etot/double(n);
  std::cout << " sum " << n<< "  " << etot << "  " << val << std::endl; 
  return val;
}


double readSMatrix() { 


  std::cout << "**************************************************\n";
  std::cout << "  Test reading SMatrix........\n";
  std::cout << "**************************************************\n";


  TFile f1("smatrix.root");

  // create tree
  TTree *t1 = (TTree*)f1.Get("t1");

  SMatrix5 *v1 = 0;
  t1->SetBranchAddress("SMatrix branch",&v1);

  timer.Start();
  int n = (int) t1->GetEntries();
  std::cout << " Tree Entries " << n << std::endl; 
  double etot=0;
  for (int i = 0; i < n; ++i) { 
    t1->GetEntry(i);
    etot += SumSMatrix(*v1);
  }


  timer.Stop();
  std::cout << " Time for SMatrix :    " << timer.RealTime() << "  " << timer.CpuTime() << std::endl; 

  std::cout << " sum " << n<< "  " << etot << "  " << etot/double(n) << std::endl; 

  return etot/double(n);
}


double readSMatrixSym() { 


  std::cout << "**************************************************\n";
  std::cout << "  Test reading SMatrix.Sym....\n";
  std::cout << "**************************************************\n";


  TFile f1("smatrixsym.root");

  // create tree
  TTree *t1 = (TTree*)f1.Get("t1");

  SMatrixSym5 *v1 = 0;
  t1->SetBranchAddress("SMatrixSym branch",&v1);

  timer.Start();
  int n = (int) t1->GetEntries();
  std::cout << " Tree Entries " << n << std::endl; 
  double etot=0;
  for (int i = 0; i < n; ++i) { 
    t1->GetEntry(i);
    etot += SumSMatrix(*v1);
  }


  timer.Stop();
  std::cout << " Time for SMatrix Sym : " << timer.RealTime() << "  " << timer.CpuTime() << std::endl; 

  std::cout << " sum " << n<< "  " << etot << "  " << etot/double(n) << std::endl; 

  return etot/double(n);
}


int testIO() { 

  gSystem->Load("libMatrix");  

  int iret = 0;
  int nEvents = 100000;

  initMatrix(nEvents);

  double w1 = writeTMatrix(nEvents);
  double w2 = writeSMatrix(nEvents);
  if ( fabs(w1-w2) > tol) { 
    std::cout << "ERROR: Differeces found  when writing \n" << std::endl;
    iret = 1;
  }


  double r1 = readTMatrix();
  double r2 = readSMatrix();
  if ( fabs(r1-r2) > tol) { 
    std::cout << "ERROR: Differeces found  when reading \n" << std::endl;
    iret = 2;
  }

  if ( fabs(w1-r1)  > tol) { 
    std::cout << "ERROR: Differeces found  when reading TMatrices\n" << std::endl;
    iret = 3;
  }
  if ( fabs(w2-r2)  > tol) { 
    std::cout << "ERROR: Differeces found  when reading SMatrices\n" << std::endl;
    iret = 4;
  }

  std::cout << "\n*****************************************************\n";
  std::cout << "    Test Symmetric matrices"; 
  std::cout << "\n*****************************************************\n\n";


  w1 = writeTMatrixSym(nEvents);
  w2 = writeSMatrixSym(nEvents);
  if ( fabs(w1-w2) > tol) { 
    std::cout << "ERROR: Differeces found  when writing \n" << std::endl;
    iret = 11;
  }


  r1 = readTMatrixSym();
  r2 = readSMatrixSym();
  if ( fabs(r1-r2) > tol) { 
    std::cout << "ERROR: Differeces found  when reading \n" << std::endl;
    iret = 12;
  }

  if ( fabs(w1-r1)  > tol) { 
    std::cout << "ERROR: Differeces found  when reading TMatrices\n" << std::endl;
    iret = 13;
  }
  if ( fabs(w2-r2)  > tol) { 
    std::cout << "ERROR: Differeces found  when reading SMatrices\n" << std::endl;
    iret = 14;
  }


  return iret;
  

}

int main() { 
  testIO();
}
