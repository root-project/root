// ANoClassDef.C

#include "ANoClassDef.h"
#include "BNoClassDef.h"

#include <iostream>
#include <sstream>
#include <string>

using namespace std;

A::A()
{
  fN = 3;
  fTable = new B[fN];
  fPtrTable = new B*[fN];
  for (int i = 0; i < fN; ++i) {
    fPtrTable[i] = new B;
  }
  for (int i = 0; i < fM; ++i) {
    fFixedTable[i] = new B[fN];
  }
  for (int i = 0; i < fM; ++i) {
    fFixedPtrTable[i] = new B*[fN];
    for (int j = 0; j < fN; ++j) {
      fFixedPtrTable[i][j] = new B;
    }
  }
  Init();
}

A::~A()
{
  delete[] fTable;
  fTable = 0;
  for (int i = 0; i < fN; ++i) {
    delete fPtrTable[i];
    fPtrTable[i] = 0;
  }
  delete[] fPtrTable;
  fPtrTable = 0;
  for (int i = 0; i < fM; ++i) {
    delete[] fFixedTable[i];
    fFixedTable[i] = 0;
  }
  for (int i = 0; i < fM; ++i) {
    for (int j = 0; j < fN; ++j) {
      delete fFixedPtrTable[i][j];
      fFixedPtrTable[i][j] = 0;
    }
    delete[] fFixedPtrTable[i];
    fFixedPtrTable[i] = 0;
  }
}

void A::Init()
{
  double const rmax = RAND_MAX + 1.0;
  fTable[0].SetName("Huey");
  fTable[0].SetX(int(100.0 * rand()/rmax));
  fTable[0].SetY(200.0 * rand()/rmax);
  fTable[1].SetName("Louie");
  fTable[1].SetX(int(300.0 * rand()/rmax));
  fTable[1].SetY(400.0 * rand()/rmax);
  fTable[2].SetName("Dewey");
  fTable[2].SetX(int(500.0 * rand()/rmax));
  fTable[2].SetY(600.0 * rand()/rmax);
  fPtrTable[0]->SetName("Dopey");
  fPtrTable[0]->SetX(int(700.0 * rand()/rmax));
  fPtrTable[0]->SetY(800.0 * rand()/rmax);
  fPtrTable[1]->SetName("Grumpy");
  fPtrTable[1]->SetX(int(900.0 * rand()/rmax));
  fPtrTable[1]->SetY(1000.0 * rand()/rmax);
  fPtrTable[2]->SetName("Sleepy");
  fPtrTable[2]->SetX(int(1100.0 * rand()/rmax));
  fPtrTable[2]->SetY(1200.0 * rand()/rmax);
  std::ostringstream nm;
  int entry = 1;
  for (int i = 0; i < fM; ++i) {
    for (int j = 0; j < fN; ++j, ++entry) {
      nm << "Entry" << entry;
      fFixedTable[i][j].SetName(nm.str().c_str());
      nm.str(std::string());
      nm << "Ptr Entry" << entry;
      fFixedPtrTable[i][j]->SetName(nm.str().c_str());
      nm.str(std::string());
      fFixedTable[i][j].SetX(int(100.0 * rand()/rmax));
      fFixedTable[i][j].SetY(100.0 * rand()/rmax);
      fFixedPtrTable[i][j]->SetX(int(100.0 * rand()/rmax));
      fFixedPtrTable[i][j]->SetY(100.0 * rand()/rmax);
    }
  }
}

A::A(const A& rhs)
{
  fN = rhs.fN;
  fTable = new B[fN];
  for (int i = 0; i < fN; ++i) {
    fTable[i] = rhs.fTable[i];
  }
  fPtrTable = new B*[fN];
  for (int i = 0; i < fN; ++i) {
    fPtrTable[i] = new B(*rhs.fPtrTable[i]);
  }
  for (int i = 0; i < fM; ++i) {
    fFixedTable[i] = new B[fN];
    fFixedPtrTable[i] = new B*[fN];
    for (int j = 0; j < fN; ++j) {
      fFixedTable[i][j] = rhs.fFixedTable[i][j];
      fFixedPtrTable[i][j] = new B(*rhs.fFixedPtrTable[i][j]);
    }
  }
}

A& A::operator=(const A& rhs)
{
  if (&rhs != this) {

    // Delete the old stuff.

    delete[] fTable;
    fTable = 0;
    for (int i = 0; i < fN; ++i) {
      delete fPtrTable[i];
      fPtrTable[i] = 0;
    }
    delete[] fPtrTable;
    fPtrTable = 0;
    for (int i = 0; i < fM; ++i) {
      delete[] fFixedTable[i];
      fFixedTable[i] = 0;
      for (int j = 0; j < fN; ++j) {
        delete fFixedPtrTable[i][j];
        fFixedPtrTable[i][j] = 0;
      }
      delete[] fFixedPtrTable[i];
    }

    // Allocate and copy the new stuff.

    fN = rhs.fN;
    fTable = new B[fN];
    for (int i = 0; i < fN; ++i) {
      fTable[i] = rhs.fTable[i];
    }
    fPtrTable = new B*[fN];
    for (int i = 0; i < fN; ++i) {
      fPtrTable[i] = new B(*rhs.fPtrTable[i]);
    }
    for (int i = 0; i < fM; ++i) {
      fFixedTable[i] = new B[fN];
      fFixedPtrTable[i] = new B*[fN];
      for (int j = 0; j < fN; ++j) {
        fFixedTable[i][j] = rhs.fFixedTable[i][j];
        fFixedPtrTable[i][j] = new B(*rhs.fFixedPtrTable[i][j]);
      }
    }

  }
  return *this;
}

void A::repr() const
{
  cout << "fN: " << fN << endl;
  cout << "fM: " << fM << endl;
  cout << "fTable:" << endl;
  for (int i = 0; i < fN; ++i) {
    cout << i << ": ";
    fTable[i].repr();
  }
  cout << "fPtrTable:" << endl;
  for (int i = 0; i < fN; ++i) {
    cout << i << ": ";
    fPtrTable[i]->repr();
  }
  cout << "fFixedTable:" << endl;
  for (int i = 0; i < fM; ++i) {
    for (int j = 0; j < fN; ++j) {
      cout << i << ", " << j << ": ";
      fFixedTable[i][j].repr();
    }
  }
  cout << "fFixedPtrTable:" << endl;
  for (int i = 0; i < fM; ++i) {
    for (int j = 0; j < fN; ++j) {
      cout << i << ", " << j << ": ";
      fFixedPtrTable[i][j]->repr();
    }
  }
}

