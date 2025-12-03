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
#include <cstdlib>
#include <TFile.h>
#include <TTree.h>
#include <TROOT.h>
#include <TSystem.h>

#include "DataModelV1.h"

#include "../common/Dumper.h"
#include "../common/Generator.h"

template <typename A>
void do_del( A* obj )
{
   delete obj;
}

int test1(const char *mode = "") {
   using namespace std;

   //---------------------------------------------------------------------------
   // Load the dictionary
   //---------------------------------------------------------------------------
   const char* dictname = "libDataModelV1_dictcint";
   const char* prefix = "";
   if( mode && mode[0] == 'r' )
   {
      dictname = "libDataModelV1_dictrflx";
      prefix = "reflex_";
   }

   if( gSystem->Load(dictname) < 0 )
   {
      cerr << "[!] Unable to load the dictionary: " << dictname << endl;
      return 0;
   }

   //---------------------------------------------------------------------------
   // Open the control files
   //---------------------------------------------------------------------------
   //ofstream o1( TString::Format("%stest01_wv1.log",prefix) );
   //ofstream o2( TString::Format("%stest02_wv1.log",prefix) );
   //ofstream o3( TString::Format("%stest03_wv1.log",prefix) );
   //ofstream o4( TString::Format("%stest04_wv1.log",prefix) );
   //ofstream o5( TString::Format("%stest05_wv1.log",prefix) );
   //ofstream o6( TString::Format("%stest06_wv1.log",prefix) );
   //ofstream o7( TString::Format("%stest07_wv1.log",prefix) );
   //ofstream o8( TString::Format("%stest08_wv1.log",prefix) );
   //ofstream o9( TString::Format("%stest09_wv1.log",prefix) );
   //ofstream o10( TString::Format("%stest10_wv1.log",prefix) );
   //ofstream o11( TString::Format("%stest11_wv1.log",prefix) );
   //ofstream o12( TString::Format("%stest12_wv1.log",prefix) );
   //ofstream o13( TString::Format("%stest13_wv1.log",prefix) );
   //ofstream o14( TString::Format("%stest14_wv1.log",prefix) );
   // ofstream o15( TString::Format("../logs/cms-00/%stest11_wv1.log",prefix) );

   //---------------------------------------------------------------------------
   // Generate the objects
   //---------------------------------------------------------------------------
   cout << "[i] Generating test data model version 1" << endl;
   ClassA *objA =  new ClassA();
   generate( objA );
   test_dump( objA, prefix, 1, "wv1" );

   ClassAIns *objAI = 0;
   generate( objAI );
   test_dump( objAI, prefix, 2, "wv1" );

   ClassD *objD = 0;
   generate( objD );
   test_dump( objD, prefix, 3, "wv1" );

   pair<int, double> *pr = 0;
   generate( pr );
   test_dump( pr, prefix, 4, "wv1" );

   vector<double> *vd = 0;
   generate( vd );
   test_dump( vd, prefix, 5, "wv1" );

   vector<pair<int, double> > *vP = 0;
   generate( vP );
   test_dump( vP, prefix, 6, "wv1" );

   vector<ClassA> *vA = 0;
   generate( vA );
   test_dump( vA, prefix, 7, "wv1" );

   vector<ClassA*> *vAS = 0;
   generate( vAS );
   test_dump( vAS, prefix, 8, "wv1" );

   vector<ClassB> *vB = 0;
   generate( vB );
   test_dump( vB, prefix, 9, "wv1" );

   vector<ClassB*> *vBS = 0;
   generate( vBS );
   test_dump( vBS, prefix, 10, "wv1" );

   vector<ClassC> *vC = 0;
   generate( vC );
   test_dump( vC, prefix, 11, "wv1" );

   vector<ClassC*> *vCS = 0;
   generate( vCS );
   test_dump( vCS, prefix, 12, "wv1" );

   vector<ClassD> *vD = 0;
   generate( vD );
   test_dump( vD, prefix, 13, "wv1" );

   vector<ClassD*> *vDS = 0;
   generate( vDS );
   test_dump( vDS, prefix, 14, "wv1" );

   LHCb::Track ltrack;
   ltrack.SetRef(2,3);

   //---------------------------------------------------------------------------
   // Store the objects in a ROOT file
   //---------------------------------------------------------------------------
   TFile *file = new TFile( TString::Format("%stestv1.root",prefix), "RECREATE" );
   TTree *tree = new TTree( "TestTree", "" );

   tree->Branch( "ltrack.", &ltrack, 32000, 99);
   tree->Branch( "TestAIns.",       &objAI, 32000, 99 );
   tree->Branch( "TestD.",          &objD, 32000, 99 );
   tree->Branch( "TestDNS.",        &objD, 32000,  0 );
   tree->Branch( "TestA.",          &objA );
   tree->Branch( "TestANS.",        &objA, 32000, 0 );
   tree->Branch( "TestPair.",       &pr );
   tree->Branch( "TestPairNS.",     &pr, 32000, 0 );
   tree->Branch( "TestVectorDbl.",  &vd );
   tree->Branch( "TestVectorP.",    &vP );
   tree->Branch( "TestVectorPNS.",  &vP, 32000, 0 );
   tree->Branch( "TestVectorA.",    &vA );
   tree->Branch( "TestVectorANS.",  &vA, 32000, 0 );
   tree->Branch( "TestVectorAS.",   &vAS );
   tree->Branch( "TestVectorASS.",  &vAS, 32000, 200 );
   tree->Branch( "TestVectorA.",    &vA );
   tree->Branch( "TestVectorANS.",  &vA, 32000, 0 );
   tree->Branch( "TestVectorAS.",   &vAS );
   tree->Branch( "TestVectorASS.",  &vAS, 32000, 200 );
   tree->Branch( "TestVectorB.",    &vB );
   tree->Branch( "TestVectorBNS.",  &vB, 32000, 0 );
   tree->Branch( "TestVectorBS.",   &vBS );
   tree->Branch( "TestVectorBSS.",  &vBS, 32000, 200 );
   tree->Branch( "TestVectorC.",    &vC );
   tree->Branch( "TestVectorCNS.",  &vC, 32000, 0 );
   tree->Branch( "TestVectorCS.",   &vCS );
   tree->Branch( "TestVectorCSS.",  &vCS, 32000, 200 );
   tree->Branch( "TestVectorD.",    &vD );
   tree->Branch( "TestVectorDNS.",  &vD, 32000, 0 );
   tree->Branch( "TestVectorDS.",   &vDS );
   tree->Branch( "TestVectorDSS.",  &vDS, 32000, 200 );

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
