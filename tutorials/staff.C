{
//   example of macro to read data from an ascii file and
//   create a root file with an histogram and an ntuple.

   gROOT->Reset();

   struct staff_t {
                Float_t cat;
                Float_t division;
                Float_t flag;
                Float_t age;
                Float_t service;
                Float_t children;
                Float_t grade;
                Float_t step;
                Float_t nation;
                Float_t hrweek;
                Float_t cost;
    };

   staff_t staff;

   FILE *fp = fopen("staff.dat","r");

   char line[81];

   TFile *f = new TFile("staff.root","RECREATE");
   TNtuple *ntuple = new TNtuple("ntuple","staff data from ascii file",
        "cat:division:flag:age:service:children:grade:step:nation:hrweek:cost");

   while (fgets(&line,80,fp)) {
      sscanf(&line[0] ,"%f%f%f%f", &staff.cat,&staff.division,&staff.flag,&staff.age);
      sscanf(&line[17],"%f%f%f%f", &staff.service,&staff.children,&staff.grade,&staff.step);
      sscanf(&line[33],"%f%f%f",   &staff.nation,&staff.hrweek,&staff.cost);
      ntuple->Fill(&staff.cat);
   }
   ntuple->Print();

   fclose(fp);
   f->Write();
}
