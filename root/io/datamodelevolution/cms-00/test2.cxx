//--------------------------------------------------------------------*- C++ -*-
// file:   test2.cxx
// author: Lukasz Janyst <ljanyst@cern.ch>
//------------------------------------------------------------------------------

#include <iostream>
#include <fstream>
#include <utility>
#include <vector>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <../common/Dumper.h>
#include <../common/Generator.h>
#include <DataModelV2.h>
#include <TFile.h>
#include <TTree.h>
#include <TROOT.h>
#include <TSystem.h>


template <typename A>
void do_del( A* obj )
{
   delete obj;
}

int test2(const char *mode = "")
{
   using namespace std;

   Dumper out("", "cms-00", "rv1");

   //---------------------------------------------------------------------------
   // Load the dictionary
   //---------------------------------------------------------------------------
   const char* dictname = "./libDataModelV2_dictcint";

   if( mode && mode[0] == 'r' )
   {
      dictname = "./libDataModelV2_dictrflx";
      gSystem->Load("libCintex");
      gROOT->ProcessLine("ROOT :: Cintex :: Cintex :: Enable();");
      out.fPrefix = "reflex_";
   }
   else {
      gROOT->ProcessLine("#include <vector>");
   }

   if( gSystem->Load(dictname) < 0 )
   {
      cerr << "[!] Unable to load the dictionary: ";
      cerr << dictname << endl;
      return 0;
   }

   //---------------------------------------------------------------------------
   // Open the control files
   //---------------------------------------------------------------------------
   ofstream o03   ( TString::Format("../logs/cms-00/%stest03_rv1.log",out.fPrefix.Data()) );
   ofstream o03ns ( TString::Format("../logs/cms-00/%stest03_rv1NS.log",out.fPrefix.Data()) );
   ofstream o04   ( TString::Format("../logs/cms-00/%stest04_rv1.log",out.fPrefix.Data()) );
   ofstream o04ns ( TString::Format("../logs/cms-00/%stest04_rv1NS.log",out.fPrefix.Data()) );
   ofstream o05   ( TString::Format("../logs/cms-00/%stest05_rv1.log",out.fPrefix.Data()) );
   ofstream o06   ( TString::Format("../logs/cms-00/%stest06_rv1.log",out.fPrefix.Data()) );
   ofstream o06ns ( TString::Format("../logs/cms-00/%stest06_rv1S.log",out.fPrefix.Data()) );
   ofstream o07   ( TString::Format("../logs/cms-00/%stest07_rv1S.log",out.fPrefix.Data()) );
   ofstream o07ns ( TString::Format("../logs/cms-00/%stest07_rv1NS.log",out.fPrefix.Data()) );
   ofstream o08   ( TString::Format("../logs/cms-00/%stest08_rv1S.log",out.fPrefix.Data()) );
   ofstream o08ns ( TString::Format("../logs/cms-00/%stest08_rv1NS.log",out.fPrefix.Data()) );
   ofstream o09   ( TString::Format("../logs/cms-00/%stest09_rv1S.log",out.fPrefix.Data()) );
   ofstream o09ns ( TString::Format("../logs/cms-00/%stest09_rv1NS.log",out.fPrefix.Data()) );
   ofstream o10   ( TString::Format("../logs/cms-00/%stest10_rv1S.log",out.fPrefix.Data()) );
   ofstream o10ns ( TString::Format("../logs/cms-00/%stest10_rv1NS.log",out.fPrefix.Data()) );
   ofstream o11   ( TString::Format("../logs/cms-00/%stest11_rv1S.log",out.fPrefix.Data()) );
   ofstream o11ns ( TString::Format("../logs/cms-00/%stest11_rv1NS.log",out.fPrefix.Data()) );
   ofstream o12   ( TString::Format("../logs/cms-00/%stest12_rv1S.log",out.fPrefix.Data()) );
   ofstream o12ns ( TString::Format("../logs/cms-00/%stest12_rv1NS.log",out.fPrefix.Data()) );
   ofstream o13   ( TString::Format("../logs/cms-00/%stest13_rv1S.log",out.fPrefix.Data()) );
   ofstream o13ns ( TString::Format("../logs/cms-00/%stest13_rv1NS.log",out.fPrefix.Data()) );

   //---------------------------------------------------------------------------
   // Generate the objects
   //---------------------------------------------------------------------------
   cout << "[i] Reading test data model version 1" << endl;
   ClassA                     *objA   = 0;
   ClassA                     *objANS = 0;
   ClassAIns                  *objAI  = 0;
   ClassD                     *objD   = 0;
   ClassD                     *objDNS = 0;
   pair<int, double>          *pr     = 0;
   pair<int, double>          *prNS   = 0;
   vector<pair<int, double> > *vP     = 0;
   vector<pair<int, double> > *vPNS   = 0;
   vector<double>             *vd     = 0;
   vector<ClassA>             *vA     = 0;
   vector<ClassA>             *vANS   = 0;
   vector<ClassA*>            *vAS    = 0;
   vector<ClassA*>            *vASS   = 0;
   vector<ClassB>             *vB     = 0;
   vector<ClassB>             *vBNS   = 0;
   vector<ClassB*>            *vBS    = 0;
   vector<ClassB*>            *vBSS   = 0;
   vector<ClassC>             *vC     = 0;
   vector<ClassC>             *vCNS   = 0;
   vector<ClassC*>            *vCS    = 0;
   vector<ClassC*>            *vCSS   = 0;
   vector<ClassD>             *vD     = 0;
   vector<ClassD>             *vDNS   = 0;
   vector<ClassD*>            *vDS    = 0;
   vector<ClassD*>            *vDSS   = 0;
   LHCb::Track                *ltrack = 0;

   //---------------------------------------------------------------------------
   // Store the objects in a ROOT file
   //---------------------------------------------------------------------------
   TFile *file = new TFile( TString::Format("%stestv1.root",out.fPrefix.Data()), "READ" );

   if( !file->IsOpen() )
   {
      cout << "[i] Unable to open: testv1.root" << endl;
      return 1;
   }

   TTree *tree = (TTree*)file->Get( "TestTree" );
   tree->SetBranchAddress( "ltrack.",       &ltrack );
   tree->SetBranchAddress( "TestAIns.",     &objAI  );
   tree->SetBranchAddress( "TestD.",        &objD   );
   tree->SetBranchAddress( "TestDNS.",      &objDNS );
   tree->SetBranchAddress( "TestA.",        &objA   );
   tree->SetBranchAddress( "TestANS.",      &objANS );
   tree->SetBranchAddress( "TestPair.",     &pr     );
   tree->SetBranchAddress( "TestPairNS.",   &prNS   );
   tree->SetBranchAddress( "TestVectorDbl", &vd     );
   tree->SetBranchAddress( "TestVectorP",   &vP     );
   tree->SetBranchAddress( "TestVectorPNS", &vPNS   );
   tree->SetBranchAddress( "TestVectorA",   &vA     );
   tree->SetBranchAddress( "TestVectorANS", &vANS   );
   tree->SetBranchAddress( "TestVectorAS",  &vAS    );
   tree->SetBranchAddress( "TestVectorASS.",&vASS   );
   tree->SetBranchAddress( "TestVectorB",   &vB     );
   tree->SetBranchAddress( "TestVectorBNS", &vBNS   );
   tree->SetBranchAddress( "TestVectorBS",  &vBS    );
   tree->SetBranchAddress( "TestVectorBSS.",&vBSS   );
   tree->SetBranchAddress( "TestVectorC",   &vC     );
   tree->SetBranchAddress( "TestVectorCNS", &vCNS   );
   tree->SetBranchAddress( "TestVectorCS",  &vCS    );
   tree->SetBranchAddress( "TestVectorCSS.",&vCSS   );
   tree->SetBranchAddress( "TestVectorD",   &vD     );
   tree->SetBranchAddress( "TestVectorDNS", &vDNS   );
   tree->SetBranchAddress( "TestVectorDS",  &vDS    );
   tree->SetBranchAddress( "TestVectorDSS.",&vDSS   );
   
   tree->GetEntry(0);
   file->Close();

   //---------------------------------------------------------------------------
   // Dump what was read
   //---------------------------------------------------------------------------
   printf("Track value: left: %ld right: %ld\n",ltrack->GetLeft(),ltrack->GetRight());
   int var = 0;
   out.dump( objA,   ++var, "S" );
   out.dump( objANS,   var, "NS" );
   out.dump( objAI,  ++var, "" );
   dump( objD,   o03 );
   dump( objDNS, o03ns);
   dump( pr,     o04  );
   dump( prNS,   o04ns );
   dump( vd,     o05  );
   dump( vP,     o06  );
   dump( vPNS,   o06ns  );
   dump( vA,     o07  );
   dump( vANS,   o07ns );
   dump( vAS,    o08 );
   dump( vASS,   o08ns );
   dump( vB,     o09  );
   dump( vBNS,   o09ns );
   dump( vBS,    o10 );
   dump( vBSS,   o10ns );
   dump( vC,     o11  );
   dump( vCNS,   o11ns );
   dump( vCS,    o12 );
   dump( vCSS,   o12ns );
   dump( vD,     o13  );
   dump( vDNS,   o13ns );
   out.dump( vDS,   14, "S" );
   out.dump( vDSS,  14, "NS" );
   
   //---------------------------------------------------------------------------
   // Cleanup
   //---------------------------------------------------------------------------
   //delete objA;
   //delete pr;
   //delete vd;
   //delete vP;
   //for_each( vAS->begin(), vAS->end(), do_del<ClassA> );
   //delete vAS;
   return 0;
}
