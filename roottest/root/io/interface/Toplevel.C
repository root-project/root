#include "ToplevelClass.C"
#include "TFile.h"
#include "TH1F.h"
#include <Riostream.h>

Bool_t ReadToplevel() 
{
   TFile *f = new TFile("Toplevel.root","READ");
   MyClass *m;
   TH1F *h;
   Bool_t result = kTRUE;

   // Test successfull read

   f->GetObject("myclass",m);
   
   if (m==0) {
      cerr << "Error: Could not load the MyClass object\n";
      result = kFALSE;
   } else if (m->a != 33) {
      cerr << "Error: Improperly read the MyClass object! Instead of 33, read " << m->a << endl;
      result = kFALSE;
   } else {
      cout << "Good: Did properly load myclass with value " << m->a << endl;
   }

   f->GetObject("histo",h);
   
   if (h==0) {
      cerr << "Error: Could not load the TH1F object\n";
      result = kFALSE;
   } else if (h->GetEntries() != 11) {
      cerr << "Error: Improperly read the TH1F object! Instead of 11 entries, read " << h->GetEntries() << endl;
      result = kFALSE;
   } else {
      cout << "Good: Did properly load TH1F with " << h->GetEntries() << " entries." << endl;
   }

   f->GetObject("myclass",h);
   
   if (h!=0) {
      cerr << "Error: Read the MyClass object into the TH1F pointer!\n";
      result = kFALSE;
   }

   f->GetObject("histo",m);

   if (m!=0) {
      cerr << "Error: Read the TH1F object into the MyClass pointer!\n";
      result = kFALSE;
   }

   delete f;
   return result;
}

Bool_t WriteToplevel() 
{
   MyClass *m = new MyClass(33);
   TH1F *h = new TH1F("histo","histo",100,-10,10);
   for(int i=0; i<11; ++i) h->Fill(i);

   TFile *f = new TFile("Toplevel.root","RECREATE");
   f->WriteObject(m,"myclass");
   f->WriteTObject(h);
   f->Write();
   delete f;
   delete m;
   delete h;
   return kTRUE;
}
