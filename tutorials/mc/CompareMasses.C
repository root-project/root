// Macro to compare masses in root data base to the values from pdg
// http://pdg.lbl.gov/2009/mcdata/mass_width_2008.mc
//
// the ROOT values are read in by TDatabasePDG from $ROOTSYS/etc/pdg_table.C
//
// Author: Christian.Klein-Boesing@cern.ch
   
#include <fstream>
#include <iostream>
#include "TDatabasePDG.h"
#include "TParticlePDG.h"

using namespace std;

void CompareMasses(){
  
  char *fn = "mass_width_2008.mc.txt";
  
  FILE* file = fopen(fn,"r");

  ifstream in1;
  in1.open(fn);
  if (!in1.good()){
      Printf("Could not open PDG particle file %s",fn);
      return;
  }

  char      c[200];  
  char      cempty;
  Int_t     pdg[4];
  Float_t   mass, err1,err2,err;
  Int_t ndiff = 0;

  
  while (fgets(&c[0],200,file)) {
    if (c[0] != '*' &&  c[0] !='W') {
      // printf("%s",c);      
      sscanf(&c[1],"%8d",&pdg[0]);
      
      // check emptyness
      pdg[1] = 0;
      for(int i = 0;i<8;i++){
	sscanf(&c[9+i],"%c",&cempty);
	if(cempty != ' ')sscanf(&c[9],"%8d",&pdg[1]);
      }

      pdg[2] = 0;
      for(int i = 0;i<8;i++){
	sscanf(&c[17+i],"%c",&cempty);
	if(cempty != ' ')sscanf(&c[17],"%8d",&pdg[2]);
      }


      pdg[3] = 0;
      for(int i = 0;i<8;i++){
	sscanf(&c[25+i],"%c",&cempty);
	if(cempty != ' ')sscanf(&c[25],"%8d",&pdg[3]);
      }

      sscanf(&c[35],"%14f",&mass);
      sscanf(&c[50],"%8f",&err1);
      sscanf(&c[50],"%8f",&err2);
      err = TMath::Max((Double_t)err1,(Double_t)-1.*err2);
      for(int ipdg = 0;ipdg  < 4;ipdg++){
	if(pdg[ipdg]==0)continue;
	TParticlePDG *partRoot = TDatabasePDG::Instance()->GetParticle(pdg[ipdg]);
	if(partRoot){
	  Float_t massRoot = partRoot->Mass();
	  Float_t deltaM = TMath::Abs(massRoot - mass);
	  //      if(deltaM > err){
	  if(deltaM/mass>1E-05){
             ndiff++;
	    Printf("%10s %8d pdg mass %E pdg err %E root Mass %E >> deltaM %E = %3.3f%%",partRoot->GetName(),pdg[ipdg],mass,err,massRoot,deltaM,100.*deltaM/mass);
	  }
	}
      }
    }
  }// while
  if (ndiff == 0) Printf("Crongratulations !! All particles in ROOT and PDG have identical masses");

}
