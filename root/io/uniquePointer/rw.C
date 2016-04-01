#include "classes.h"

void printHistoInfo(TH1F* h, const char* meta) {
   cout << "[" << meta << "]"<< " Histogram "
        << h->GetName() << " information:\n"
        << " - Entries: " << h->GetEntries() << endl
        << " - Mean: " << h->GetMean() << endl
        << " - STD Deviation: " << h->GetRMS() << endl;

}

void w(const char* filename) {
   cout << "Writing " << filename << endl;

   TH1F meansPtr ("meansPtr","meansPtr",64, -4, 4);
   TH1F meansUPtr ("meansUPtr","meansUPtr",64, -4, 4);

   auto f = TFile::Open(filename,"RECREATE");
   auto a = new A("RowWise");

   printHistoInfo(a->GetHPtr(), "Write Row-wise");
   printHistoInfo(a->GetHUPtr(), "Write Row-wise");

   f->WriteObject(a, "theAInstance");

   if (strstr(filename,"root")){

      // Now a Tree
      TTree t("mytree", "mytree");
      auto b = new A("ColumnWise");
      t.Branch("theABranch",&b);
      for (auto i : ROOT::TSeqI(50)) {
         b->Randomize();
         meansPtr.Fill(b->GetHPtr()->GetMean());
         meansUPtr.Fill(b->GetHUPtr()->GetMean());
         t.Fill();
      }
      printHistoInfo(&meansPtr, "Column Row-wise");
      printHistoInfo(&meansUPtr, "Column Row-wise");
      t.Write();
   }


   delete a;
   delete f;
}

void r(const char* filename) {
   cout << "Reading " << filename << endl;

   TH1F meansPtr ("meansPtr","meansPtr",64, -4, 4);
   TH1F meansUPtr ("meansUPtr","meansUPtr",64, -4, 4);

   auto f = TFile::Open(filename);
   auto a = (A*) f->Get("theAInstance");
   printHistoInfo(a->GetHPtr(), "Read Row-wise");
   printHistoInfo(a->GetHUPtr(), "Read Row-wise");

   if (strstr(filename,"root")){
      TTreeReader tr("mytree", f);
      TTreeReaderValue<A> myA(tr, "theABranch");

      while (tr.Next()) {
         auto mean = myA->GetHPtr()->GetMean();
         meansPtr.Fill(mean);
         auto umean = myA->GetHUPtr()->GetMean();
         meansUPtr.Fill(umean);
      }
      printHistoInfo(&meansPtr, "Read Column-wise");
      printHistoInfo(&meansUPtr, "Read Column-wise");
   }

   delete f;


}

int rw(bool write = true){

   auto c = TClass::GetClass("A");
   auto si = c->GetStreamerInfo();
   // Here the output

   auto filenames {"out.root"};//, "out.xml"};
   for (auto filename : filenames){
      if (write) w(filename);
      r(filename);
   }
   return 0;
}
