// example of macro to read data from an ascii file and
// create a root file with a Tree.
// see also a variant in staff.C
// Author: Rene Brun
void cernbuild() {

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

   //The input file cern.dat is a copy of the CERN staff data base
   //from 1988
   FILE *fp = fopen("cernstaff.dat","r");

   char line[80];

   TFile f("cernstaff.root","RECREATE");
   TTree *tree = new TTree("T","CERN 1988 staff data");
   tree->Branch("Category",&Category,"Category/I");
   tree->Branch("Flag",&Flag,"Flag/i");
   tree->Branch("Age",&Age,"Age/I");
   tree->Branch("Service",&Service,"Service/I");
   tree->Branch("Children",&Children,"Children/I");
   tree->Branch("Grade",&Grade,"Grade/I");
   tree->Branch("Step",&Step,"Step/I");
   tree->Branch("Hrweek",&Hrweek,"Hrweek/I");
   tree->Branch("Cost",&Cost,"Cost/I");
   tree->Branch("Division",Division,"Division/C");
   tree->Branch("Nation",Nation,"Nation/C");
   while (fgets(&line,80,fp)) {
      sscanf(&line[0],"%d %d %d %d %d",&Category,&Flag,&Age,&Service,&Children);
      sscanf(&line[32],"%d %d  %d %d %s %s",&Grade,&Step,&Hrweek,&Cost,Division,Nation);
      tree->Fill();
   }
   tree->Print();
   tree->Write();

   fclose(fp);
}
