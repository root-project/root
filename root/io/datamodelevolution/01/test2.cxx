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
#include <DataModelV1.h>
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
   srandom( time( 0 ) );

   //---------------------------------------------------------------------------
   // Load the dictionary
   //---------------------------------------------------------------------------
   const char* dictname = "./libDataModelV1_dictcint.so";
   const char* prefix = "";
   if( mode && mode[0] == 'r' )
   {
      dictname = "./libDataModelV1_dictrflx.so";
      gROOT->ProcessLine("ROOT :: Cintex :: Cintex :: Enable();");
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
   ofstream o1 ( TString::Format("../logs/01/%stest01_rv1.log",prefix) );
   ofstream o2 ( TString::Format("../logs/01/%stest01_rv1NS.log",prefix) );
   ofstream o3 ( TString::Format("../logs/01/%stest02_rv1.log",prefix) );
   ofstream o4 ( TString::Format("../logs/01/%stest02_rv1NS.log",prefix) );
   ofstream o5 ( TString::Format("../logs/01/%stest03_rv1.log",prefix) );
   ofstream o6 ( TString::Format("../logs/01/%stest04_rv1.log",prefix) );
   ofstream o7 ( TString::Format("../logs/01/%stest04_rv1NS.log",prefix) );
   ofstream o8 ( TString::Format("../logs/01/%stest05_rv1.log",prefix) );
   ofstream o9 ( TString::Format("../logs/01/%stest05_rv1NS.log",prefix) );
   ofstream o10( TString::Format("../logs/01/%stest06_rv1.log",prefix) );
   ofstream o11( TString::Format("../logs/01/%stest06_rv1S.log",prefix) );

   //---------------------------------------------------------------------------
   // Generate the objects
   //---------------------------------------------------------------------------
   cout << "[i] Reading test data model version 1" << endl;
   ClassA                     *objA   = 0;
   ClassA                     *objANS = 0;
   pair<int, double>          *pr     = 0;
   pair<int, double>          *prNS   = 0;
   vector<double>             *vd     = 0;
   vector<ClassA>             *vA     = 0;
   vector<ClassA>             *vANS   = 0;
   vector<pair<int, double> > *vP     = 0;
   vector<pair<int, double> > *vPNS   = 0;
   vector<ClassA*>            *vAS    = 0;
   vector<ClassA*>            *vASS   = 0;

   //---------------------------------------------------------------------------
   // Store the objects in a ROOT file
   //---------------------------------------------------------------------------
   TFile *file = new TFile( TString::Format("%stestv1.root",prefix), "READ" );

   if( !file->IsOpen() )
   {
      cout << "[i] Unable to open: testv1.root" << endl;
      return 1;
   }

   TTree *tree = (TTree*)file->Get( "TestTree" );
   tree->SetBranchAddress( "TestA",         &objA   );
   tree->SetBranchAddress( "TestANS",       &objANS );
   tree->SetBranchAddress( "TestPair",      &pr     );
   tree->SetBranchAddress( "TestPairNS",    &prNS   );
   tree->SetBranchAddress( "TestVectorD",   &vd     );
   tree->SetBranchAddress( "TestVectorA",   &vA     );
   tree->SetBranchAddress( "TestVectorANS", &vANS   );
   tree->SetBranchAddress( "TestVectorP",   &vP     );
   tree->SetBranchAddress( "TestVectorPNS", &vPNS   );
   tree->SetBranchAddress( "TestVectorAS",  &vAS    );
   tree->SetBranchAddress( "TestVectorASS", &vASS   );

   tree->GetEntry(0);
   file->Close();

   //---------------------------------------------------------------------------
   // Dump what was read
   //---------------------------------------------------------------------------
   dump( objA,   o1  );
   dump( objANS, o2  );
   dump( pr,     o3  );
   dump( prNS,   o4  );
   dump( vd,     o5  );
   dump( vA,     o6  );
   dump( vANS,   o7  );
   dump( vP,     o8  );
   dump( vPNS,   o9  );
   dump( vAS,    o10 );
   dump( vASS,   o11 );

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
