/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/


#include <iostream>
using namespace std;
typedef double Double_t;
typedef int Int_t;

Int_t nrows = 6;
Int_t ncols = 4;
Double_t **table = 0;
Int_t cut = 3;

//______________________________________________________________________
void init() {
// Create table
    table = new Double_t*[nrows];
    for (Int_t i=0; i<nrows; i++) {
        table[i] = 0;
	table[i] = new Double_t[ncols];
    }//for_i

// Fill table
    for (Int_t i=0; i<nrows; i++) {
       for (Int_t k=0; k<ncols; k++) {
          table[i][k] = (k==ncols-1) ? i : (k+1)*1.1;
//         table[i][k] = i;
       }//for_k
    }//for_i
}

void doit() {
// Export table
    for (Int_t i=0; i<nrows; i++) {
       if (cut > 0 && (Int_t)(table[i][ncols-1]) < cut) continue;
       for (Int_t k=0; k<ncols; k++) {
	 //G__dispvalue(stderr,table[i][k]);
          cout << " " << table[i][k];
       }//for_k
       cout << endl;
    }//for_i
}

void clear() {
  for (Int_t i=0; i<nrows; i++) {
    if (table[i]) {delete [] table[i]; table[i] = 0;}
  }//for_i
  if (table) delete [] table;
  
}//TestTable

int main() {
  init();
  doit();
  clear();
  return 0;
}
