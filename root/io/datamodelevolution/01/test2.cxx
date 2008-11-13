//--------------------------------------------------------------------*- C++ -*-
// file:   test2.cxx
// author: Lukasz Janyst <ljanyst@cern.ch>
//------------------------------------------------------------------------------

#include <iostream>
#include <fstream>
#include <utility>
#include <vector>
#include <algorithm>
#include <dlfcn.h>
#include <ctime>
#include <cstdlib>
#include <Dumper.h>
#include <Generator.h>
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

int main( int argc, char** argv )
{
   using namespace std;
   srandom( time( 0 ) );

   //---------------------------------------------------------------------------
   // Load the dictionary
   //---------------------------------------------------------------------------
   const char* dictname = "./libDataModelV1_dictCINT.so";
   if( argc == 2 && argv[1][0] == 'r' )
   {
      dictname = "./libDataModelV1_dictREFLEX.so";
      gROOT->ProcessLine("ROOT :: Cintex :: Cintex :: Enable();");
      // gROOT->ProcessLine("ROOT :: Cintex :: Cintex :: SetDebug( 1 );");
   }
   else
      gROOT->ProcessLine("#include <vector>");

   if( gSystem->Load(dictname) < 0 )
   {
      cerr << "[!] Unable to load the dictionary: ";
      cerr << dictname << endl;
      return 0;
   }

   //---------------------------------------------------------------------------
   // Open the control files
   //---------------------------------------------------------------------------
   ofstream o1 ( "../logs/01/test01_rv1.log" );
   ofstream o2 ( "../logs/01/test01_rv1NS.log" );
   ofstream o3 ( "../logs/01/test02_rv1.log" );
   ofstream o4 ( "../logs/01/test02_rv1NS.log" );
   ofstream o5 ( "../logs/01/test03_rv1.log" );
   ofstream o6 ( "../logs/01/test04_rv1.log" );
   ofstream o7 ( "../logs/01/test04_rv1NS.log" );
   ofstream o8 ( "../logs/01/test05_rv1.log" );
   ofstream o9 ( "../logs/01/test05_rv1NS.log" );
   ofstream o10( "../logs/01/test06_rv1.log" );
   ofstream o11( "../logs/01/test06_rv1S.log" );

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
   TFile *file = new TFile( "testv1.root", "READ" );

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
