#include "TTree.h"

TTree * testTree() {
  int nClus;
  int iClus[10];
  int nCells;
  int iCell[100];
  int iCellClus[100];

  TTree * tt = new TTree("testTree","testTree");
  tt->Branch("nClus",&nClus,"nClus/I");
  tt->Branch("iClus",iClus,"iClus[nClus]/I");
  tt->Branch("nCells",&nCells,"nCells/I");
  tt->Branch("iCell",iCell,"iCell[nCells]/I");
  tt->Branch("iCellClus",iCellClus,"iCellClus[nCells]/I");

  nCells = 0;
  for(nClus=0;nClus<10;nClus++) {
    iClus[nClus] = nClus+1;
    for(int i=0;i<10;i++) {
      iCell[nCells] = 100*nClus+i;
      iCellClus[nCells] = iClus[nClus];
      nCells++;
    }      
  }
  tt->Fill();
  tt->ResetBranchAddresses();
  return tt;
}

void runindexLoad() {

   TTree *tt = testTree();
   tt->SetScanField(0);
   tt->Scan("iCell>900:iCell:iCellClus-1:iCell>900&&iClus[iCellClus-1]==10","","colsize=10");
}
