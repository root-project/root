//--------------------------------------------------------------------*- C++ -*-
// file:   test1.cxx
// author: Lukasz Janyst <ljanyst@cern.ch>
//------------------------------------------------------------------------------

#include <iostream>
#include <fstream>
#include <utility>
#include <vector>
#include <algorithm>
#include <ctime>
#include <../common/Dumper.h>
#include <../common/Generator.h>
#include <cstdlib>
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

int test1(const char *mode = "") {
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
      prefix = "rflx_";
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
   ofstream o1( TString::Format("../logs/01/%stest01_wv1.log",prefix) );
   ofstream o2( TString::Format("../logs/01/%stest02_wv1.log",prefix) );
   ofstream o3( TString::Format("../logs/01/%stest03_wv1.log",prefix) );
   ofstream o4( TString::Format("../logs/01/%stest04_wv1.log",prefix) );
   ofstream o5( TString::Format("../logs/01/%stest05_wv1.log",prefix) );
   ofstream o6( TString::Format("../logs/01/%stest06_wv1.log",prefix) );

   //---------------------------------------------------------------------------
   // Generate the objects
   //---------------------------------------------------------------------------
   cout << "[i] Generating test data model version 1" << endl;
   ClassA *objA =  new ClassA();
   generate( objA );
   dump( objA, o1 );

   pair<int, double> *pr = 0;
   generate( pr );
   dump( pr, o2 );

   vector<double> *vd = 0;
   generate( vd );
   dump( vd, o3 );

   vector<ClassA> *vA = 0;
   generate( vA );
   dump( vA, o4 );
 
   vector<pair<int, double> > *vP = 0;
   generate( vP );
   dump( vP, o5 );

   vector<ClassA*> *vAS = 0;
   generate( vAS );
   dump( vAS, o6 );

   //---------------------------------------------------------------------------
   // Store the objects in a ROOT file
   //---------------------------------------------------------------------------
   TFile *file = new TFile( TString::Format("%stestv1.root",prefix), "RECREATE" );
   TTree *tree = new TTree( "TestTree", "" );

   tree->Branch( "TestA",          &objA );
   tree->Branch( "TestANS",        &objA, 32000, 0 );
   tree->Branch( "TestPair",       &pr );
   tree->Branch( "TestPairNS",     &pr, 32000, 0 );
   tree->Branch( "TestVectorD",    &vd );
   tree->Branch( "TestVectorA",    &vA );
   tree->Branch( "TestVectorANS",  &vA, 32000, 0 );
   tree->Branch( "TestVectorP",    &vP );
   tree->Branch( "TestVectorPNS",  &vP, 32000, 0 );
   tree->Branch( "TestVectorAS",   &vAS );
   tree->Branch( "TestVectorASS",  &vAS, 32000, 200 );

   tree->Fill();
   file->Write();
   file->Close();

   //---------------------------------------------------------------------------
   // Cleanup
   //---------------------------------------------------------------------------
   delete objA;
   delete pr;
   delete vd;
   delete vP;
   for_each( vAS->begin(), vAS->end(), do_del<ClassA> );
   delete vAS;
   return 0;
}
