#include "TFile.h"
#include "TProcessID.h"
#include "TRef.h"
#include "Riostream.h"

void Check(TObject* obj) {

   if (obj) cout << "Found the referenced object\n";
   else cout << "Error: Could not find the referenced object\n";
}

const char *filename = "lotsRef.root";
void lotsRef(int what) {
   if (what>2) {
      Int_t size = what;
      TFile *_file0 = TFile::Open(filename,"RECREATE");
      for(int i=0;i<size;++i) {
         TProcessID *id = TProcessID::AddProcessID();
         TProcessID::WriteProcessID(id,_file0);
      }
      _file0->Write(); delete _file0;
   } else if (what==2) {
      TFile *_file0 = TFile::Open(filename,"UPDATE");

      TNamed *n = new TNamed("mine","title"); TRef *r = new TRef(n);
      n->Write();
      r->Write();
      _file0->Write(); delete _file0;
   } else if (what==1) {
      TFile *_file0 = TFile::Open(filename,"UPDATE");

      int i=0;
      while( TProcessID::ReadProcessID(++i,_file0) ) {};

      TNamed *n;_file0->GetObject("mine",n);
      TRef *r;_file0->GetObject("TRef",r);
      if (r==0) { cerr << "Could not find the TRef on file \n"; return; }
      if (n==0) { cerr << "Could not find the TNamed on file \n"; return;}
      Check( r->GetObject() );
      n->Write();
      r->Write();
      _file0->Write();
   } else {
      TFile *_file0 = TFile::Open(filename);
      TRef *r;_file0->GetObject("TRef",r);
      TNamed *n;_file0->GetObject("mine",n);
      Check( r->GetObject() );
   }
}
