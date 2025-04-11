#include <TTree.h>
#include <TFile.h>
#include <TInterpreter.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <iostream>
#include <TChain.h>

#ifdef R__HAS_DATAFRAME
#include <ROOT/RDataFrame.hxx>
#endif
 
struct Int {
   int x;
};

struct FloatInt {
   float f;
   int x;
};

int testSetBranchAddress(TTree *t, Long64_t entry)
{
   t->ResetBranchAddresses();

   int x = -9999;
   float f = -8888;
   FloatInt o;

   // SetBranchAddress before calling GetEntry.
   x = -9999;
   t->SetBranchAddress("i.x", &x);
   t->GetEntry(entry);
   if ( x != -1) {
      std::cout << "SetBranchAddress(\"i.x\") before GetEntry (should print -1): " << x << " at " << (void*)&x << std::endl;
      std::cout << "MakeClass for i.x: " <<t->GetBranch("i.x")->GetMakeClass() <<std::endl;
      return 1;
   }

   t->ResetBranchAddresses();
   t->SetMakeClass(false);
   x = -9999;
   // SetBranchAddress before calling GetEntry;
   t->GetEntry(0);
   t->SetBranchAddress("i.x", &x);
   t->GetEntry(entry);
   if ( x != -1) {
      std::cout << "SetBranchAddress(\"i.x\") after GetEntry (should print -1): " << x << " at " << (void*)&x << std::endl;
      std::cout << "MakeClass for i.x: " <<t->GetBranch("i.x")->GetMakeClass() <<std::endl;
      return 2;
   }

   t->ResetBranchAddresses();
   t->SetMakeClass(false);
   x = -9999;
   t->GetEntry(0);
   t->SetBranchAddress("x", &x);
   t->GetEntry(entry);
   if ( x != 1) {
      std::cout << "SetBranchAddress(\"x\") after GetEntry (should print 1): " << x << " at " << (void*)&x << std::endl;
      std::cout << "MakeClass for x: " <<t->GetBranch("i.x")->GetMakeClass() <<std::endl;
      return 3;
   }

   t->ResetBranchAddresses();
   x = -9999;
   t->SetMakeClass(1);
   t->SetBranchAddress("x", &x);
   t->GetEntry(entry);
   if ( x != 1) {
      std::cout << "SetBranchAddress(\"x\") after GetEntry (should print 1): " << x << " at " << (void*)&x << std::endl;
      std::cout << "MakeClass for x: " <<t->GetBranch("i.x")->GetMakeClass() <<std::endl;
      return 4;
   }


   t->ResetBranchAddresses();
   t->SetMakeClass(false);
   x = -9999;
   t->SetBranchAddress("x.x", &x);
   t->GetEntry(entry);
   if (x != -9999) {
      std::cout << "SetBranchAddress(\"x.x\") after GetEntry (should print 1): " << x << " at " << (void*)&x << std::endl;
      std::cout << "Bracnh for x.x is (should be nullptr): " << t->GetBranch("x.x") <<std::endl;
      return 5;
   }



   // SetBranchAddress before calling GetEntry.
   x = -9999;
   t->SetBranchAddress("m.x", &x);
   t->GetEntry(entry);
   if ( x != -1) {
      std::cout << "SetBranchAddress(\"m.x\") before GetEntry (should print -1): " << x << " at " << (void*)&x << std::endl;
      std::cout << "MakeClass for i.x: " <<t->GetBranch("m.x")->GetMakeClass() <<std::endl;
      return 6;
   }

   t->ResetBranchAddresses();
   t->SetMakeClass(false);
   x = -9999;
   // SetBranchAddress before calling GetEntry;
   t->GetEntry(0);
   t->SetBranchAddress("m.x", &x);
   t->GetEntry(entry);
   if ( x != -1) {
      std::cout << "SetBranchAddress(\"m.x\") after GetEntry (should print -1): " << x << " at " << (void*)&x << std::endl;
      std::cout << "MakeClass for m.x: " <<t->GetBranch("m.x")->GetMakeClass() <<std::endl;
      return 7;
   }



   // SetBranchAddress before calling GetEntry.
   o.x = -9999;
   o.f = -8888;
   t->SetBranchAddress("p", &o);
   t->GetEntry(entry);
   if ( o.x != 3) {
      std::cout << "SetBranchAddress(\"p\") before GetEntry (should print 3 and 2.1): " << o.x << " , " << o.f << " at " << (void*)&o << std::endl;
      return 8;
   }
   if ( abs(o.f - 2.1) > 0.001 ) {
      std::cout << "SetBranchAddress(\"p\") before GetEntry (should print 3 and 2.1): " << o.x << " , " << o.f << " at " << (void*)&o << std::endl;
      return 9;
   }

   t->ResetBranchAddresses();
   t->SetMakeClass(false);
   f = -8888;
   // SetBranchAddress before calling GetEntry;
   t->GetEntry(0);
   t->SetBranchAddress("p", &o);
   t->GetEntry(entry);
   if ( o.x != 3) {
      std::cout << "SetBranchAddress(\"p\") after GetEntry (should print 3 and 2.1): " << o.x << " , " << o.f << " at " << (void*)&o << std::endl;
      return 10;
   }
   if ( abs(o.f - 2.1) > 0.001 ) {
      std::cout << "SetBranchAddress(\"p\") before GetEntry (should print 3 and 2.1): " << o.x << " , " << o.f << " at " << (void*)&o << std::endl;
      return 11;
   }

   return 0;
}

int testTreeReader(TTreeReader &r) {
   TTreeReaderValue<int> rix(r, "i.x");
   TTreeReaderValue<int> rxx(r, "x.x");
   TTreeReaderValue<int> rpx(r, "p.x");
   // This is silly but then again user should not add a trailing dot for a leaflist.
   TTreeReaderValue<int> rqx(r, "q..x");
   std::cout << "TTreeReader::Next:" << std::endl;
   while (r.Next()) {
      if ( *rix != -1 ) {
         std::cout << "TTreeReader failed for i.x it reads: " << *rix << " while we expected -1\n";
         return 101;
      }
      if ( *rxx != 1 ) {
         std::cout << "TTreeReader failed for x/x it reads: " << *rxx << " while we expected 1\n";
         return 102;
      }
      if ( *rpx != 3 ) {
         std::cout << "TTreeReader failed for p.x it reads: " << *rpx << " while we expected 3\n";
         return 103;
      }
      if ( *rqx != 3 ) {
         std::cout << "TTreeReader failed for q.x it reads: " << *rqx << " while we expected 3\n";
         return 104;
      }
   }
   return 0;
}

int execGetBranch()
{
   {
      // Need a physical file so we can also test TChain.
      TFile f("execGetBranch.root", "recreate");
      Int i{-1};
      int x = 1;
      FloatInt s{2.1, 3};
      TTree t("t", "t");
      t.Branch("i", &i);
      t.Branch("x", &x);
      t.Branch("m.", &i);
      t.Branch("p", &s, "f/F:x/I");
      t.Branch("q.", &s,"f/F:x/I");
      t.Fill();
      t.Write();
      f.Close();
   }
 
   TFile file("execGetBranch.root");
   auto t = file.Get<TTree>("t");

   TIter iter(t->GetListOfLeaves());
   while(TLeaf *leaf = (TLeaf*)iter()) {
      std::cout << "Leaf name: " << leaf->GetName() << "\tfull name: " << leaf->GetFullName()
                << "\tbranch name: " << leaf->GetBranch()->GetName() << "  \tbranch full name: " << leaf->GetBranch()->GetFullName()  << std::endl;
   }

   std::cout << "Testing on TTree\n";
   int res = testSetBranchAddress(t, 0);
   if (res != 0)
      return res;

   TChain ch("t");
   ch.AddFile("execGetBranch.root");
   ch.AddFile("execGetBranch.root");

   std::cout << "Testing on TChain\n";
   testSetBranchAddress(&ch, 0);
   if (res != 0)
      return res;
   testSetBranchAddress(&ch, 1);
   if (res != 0)
      return res;

   t->ResetBranchAddresses();
   ch.ResetBranchAddresses();

   TTreeReader r("t", &file);
   res = testTreeReader(r);
   if (res)
      return res;
   TTreeReader rc(&ch);
   res = testTreeReader(rc);
   if (res)
      return res+50;

#ifdef R__HAS_DATAFRAME
   std::cout << "RDataFrame columns:" << std::endl;
   for (const auto &c : ROOT::RDataFrame(*t).GetColumnNames())
      std::cout << c << std::endl;
#else
   // Fill the reference as expected:
   std::cout << "RDataFrame columns:" << std::endl;
   std::cout << "i" << std::endl;
   std::cout << "i.x" << std::endl;
   std::cout << "m." << std::endl;
   std::cout << "m.x" << std::endl;
   std::cout << "p.f" << std::endl;
   std::cout << "p.x" << std::endl;
   std::cout << "q..f" << std::endl;
   std::cout << "q..x" << std::endl;
   std::cout << "x" << std::endl;
#endif
 
   return 0;
}
