{
//   example of macro to read data from an ascii file and
//   create a root file with an histogram and an ntuple.

   gROOT->Reset();

   struct staff_t {
                Int_t cat;
                Int_t division;
                Int_t flag;
                Int_t age;
                Int_t service;
                Int_t children;
                Int_t grade;
                Int_t step;
                Int_t nation;
                Int_t hrweek;
                Int_t cost;
    };

   staff_t staff;

   FILE *fp = fopen("staff.dat","r");

   char line[81];

   TFile *f = new TFile("staff.root","RECREATE");
   TTree *tree = new TTree("tree","staff data from ascii file");
   tree->Branch("staff",&staff.cat,"cat/I:division:flag:age:service:children:grade:step:nation:hrweek:cost");

   while (fgets(&line,80,fp)) {
      sscanf(&line[0] ,"%d%d%d%d", &staff.cat,&staff.division,&staff.flag,&staff.age);
      sscanf(&line[13],"%d%d%d%d", &staff.service,&staff.children,&staff.grade,&staff.step);
      sscanf(&line[24],"%d%d%d",   &staff.nation,&staff.hrweek,&staff.cost);
      tree->Fill();
   }
   tree->Print();

   fclose(fp);
   f->Write();
}
