// example of macro to read data from an ascii file and
// create a root file with a Tree.
// see also a variant in cernbuild.C
// Author: Rene Brun
void staff() {
   
   struct staff_t {
      Int_t           Category;
      UInt_t          Flag;
      Int_t           Age;
      Int_t           Service;
      Int_t           Children;
      Int_t           Grade;
      Int_t           Step;
      Int_t           Hrweek;
      Int_t           Cost;
      Char_t          Division[4];
      Char_t          Nation[3];
    };

   staff_t staff;

   //The input file cern.dat is a copy of the CERN staff data base
   //from 1988
   FILE *fp = fopen("cernstaff.dat","r");

   char line[80];

   TFile *f = new TFile("staff.root","RECREATE");
   TTree *tree = new TTree("T","staff data from ascii file");
   tree->Branch("staff",&staff.Category,"Category/I:Flag:Age:Service:Children:Grade:Step:Hrweek:Cost");
   tree->Branch("Division",staff.Division,"Division/C");
   tree->Branch("Nation",staff.Nation,"Nation/C");
    //note that the branches Division and Nation cannot be on the first branch
   while (fgets(&line,80,fp)) {
      sscanf(&line[0],"%d %d %d %d %d",&staff.Category,&staff.Flag,&staff.Age,&staff.Service,&staff.Children);
      sscanf(&line[32],"%d %d  %d %d %s %s",&staff.Grade,&staff.Step,&staff.Hrweek,&staff.Cost,staff.Division,staff.Nation);
      tree->Fill();
   }
   tree->Print();
   tree->Write();

   fclose(fp);
   delete tree;
   delete f;
}
