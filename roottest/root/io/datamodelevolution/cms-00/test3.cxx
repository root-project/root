//--------------------------------------------------------------------*- C++ -*-
// file:   test3.cxx
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

int test3(const char *mode = "")
{
   using namespace std;

   Dumper out("", "01", "rv1");

   //---------------------------------------------------------------------------
   // Load the dictionary
   //---------------------------------------------------------------------------
   const char* dictname = "./libDataModelV2_dictcint";

   if( mode && mode[0] == 'r' )
   {
      dictname = "./libDataModelV2_dictrflx";
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

   // TClass::GetClass( "vector<ClassA>" )->GetStreamerInfo( 10 );

   //---------------------------------------------------------------------------
   // Generate the objects
   //---------------------------------------------------------------------------
   cout << "[i] Reading test data model version 2" << endl;
   ClassA2                    *objA   = 0;
   ClassA2                    *objANS = 0;
   pair<int, float>           *pr     = 0;
   pair<int, float>           *prNS   = 0;
   vector<double>             *vd     = 0;
   vector<ClassA2>            *vA     = 0;
   vector<ClassA2>            *vANS   = 0;
   vector<pair<int, float> >  *vP     = 0;
   vector<pair<int, float> >  *vPNS   = 0;
   vector<ClassA2*>           *vAS    = 0;
   vector<ClassA2*>           *vASS   = 0;
   ClassAIns2                 *objAI  = 0;
   ClassD                     *objD   = 0;
   ClassD                     *objDNS = 0;

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
   tree->SetBranchAddress( "TestAIns",      &objAI  );
   tree->SetBranchAddress( "TestD",         &objD   );
   tree->SetBranchAddress( "TestDNS",       &objDNS );
   //tree->SetBranchAddress( "TestA",         &objA   );
//   tree->SetBranchAddress( "TestANS",       &objANS );
//   tree->SetBranchAddress( "TestPair",      &pr     );
//   tree->SetBranchAddress( "TestPairNS",    &prNS   );
//   tree->SetBranchAddress( "TestVectorD",   &vd     );
//   tree->SetBranchAddress( "TestVectorA",   &vA     );
//   tree->SetBranchAddress( "TestVectorANS", &vANS   );
//   tree->SetBranchAddress( "TestVectorP",   &vP     );
//   tree->SetBranchAddress( "TestVectorPNS", &vPNS   );
//   tree->SetBranchAddress( "TestVectorAS",  &vAS    );
//   tree->SetBranchAddress( "TestVectorASS", &vASS   );

   tree->GetEntry(0);
   //tree->Print("debugAddress");
   tree->Print("debugInfo");
   file->Close();

   //---------------------------------------------------------------------------
   // Dump what was read
   //---------------------------------------------------------------------------
   unsigned int var = 1;
   //dump( objA,   o1  );
   out.dump( objAI,  ++var, "" );
   out.dump( objD,   ++var, "S" );
   out.dump( objDNS, var,   "NS");
//   dump( objANS, o2  );
//   dump( pr,     o3  );
//   dump( prNS,   o4  );
//   dump( vd,     o5  );
//   dump( vA,     o6  );
//   dump( vANS,   o7  );
//   dump( vP,     o8  );
//   dump( vPNS,   o9  );
//   dump( vAS,    o10 );
//   dump( vASS,   o11 );

   //---------------------------------------------------------------------------
   // Cleanup
   //---------------------------------------------------------------------------
//   delete objA;
//   delete pr;
//   delete vd;
//   delete vP;
   if (vAS) for_each( vAS->begin(), vAS->end(), do_del<ClassA2> );
   if (vASS) for_each( vASS->begin(), vASS->end(), do_del<ClassA2> );
   delete vAS;
   delete vASS;
   return 0;
}
