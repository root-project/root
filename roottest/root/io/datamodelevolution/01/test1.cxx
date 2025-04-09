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
   ofstream o7( TString::Format("../logs/01/%stest07_wv1.log",prefix) );
   ofstream o8( TString::Format("../logs/01/%stest08_wv1.log",prefix) );
   ofstream o9( TString::Format("../logs/01/%stest09_wv1.log",prefix) );
   ofstream o10( TString::Format("../logs/01/%stest10_wv1.log",prefix) );
   ofstream o11( TString::Format("../logs/01/%stest11_wv1.log",prefix) );
   ofstream o12( TString::Format("../logs/01/%stest12_wv1.log",prefix) );
   ofstream o13( TString::Format("../logs/01/%stest13_wv1.log",prefix) );
   ofstream o14( TString::Format("../logs/01/%stest14_wv1.log",prefix) );
   //ofstream o15( TString::Format("../logs/01/%stest11_wv1.log",prefix) );

   //---------------------------------------------------------------------------
   // Generate the objects
   //---------------------------------------------------------------------------
   cout << "[i] Generating test data model version 1" << endl;
   ClassA *objA =  new ClassA();
   generate( objA );
   dump( objA, o1 );

   ClassAIns *objAI = 0;
   generate( objAI );
   dump( objAI, o2 );
   
   ClassD *objD = 0;
   generate( objD );
   dump( objD, o3 );
   
   pair<int, double> *pr = 0;
   generate( pr );
   dump( pr, o4 );

   vector<double> *vd = 0;
   generate( vd );
   dump( vd, o5 );

   vector<pair<int, double> > *vP = 0;
   generate( vP );
   dump( vP, o6 );

   vector<ClassA> *vA = 0;
   generate( vA );
   dump( vA, o7 );
 
   vector<ClassA*> *vAS = 0;
   generate( vAS );
   dump( vAS, o8 );
   
   vector<ClassB> *vB = 0;
   generate( vB );
   dump( vB, o9 );

   vector<ClassB*> *vBS = 0;
   generate( vBS );
   dump( vBS, o10 );
   
   vector<ClassC> *vC = 0;
   generate( vC );
   dump( vC, o11 );

   vector<ClassC*> *vCS = 0;
   generate( vCS );
   dump( vCS, o12 );
   
   vector<ClassD> *vD = 0;
   generate( vD );
   dump( vD, o13 );
   
   vector<ClassD*> *vDS = 0;
   generate( vDS );
   dump( vDS, o14 );
   
   //---------------------------------------------------------------------------
   // Store the objects in a ROOT file
   //---------------------------------------------------------------------------
   TFile *file = new TFile( TString::Format("%stestv1.root",prefix), "RECREATE" );
   TTree *tree = new TTree( "TestTree", "" );

   tree->Branch( "TestAIns",       &objAI, 32000, 99 );
   tree->Branch( "TestD",          &objD, 32000, 99 );
   tree->Branch( "TestDNS",        &objD, 32000,  0 );
   tree->Branch( "TestA",          &objA );
   tree->Branch( "TestANS",        &objA, 32000, 0 );
   tree->Branch( "TestPair",       &pr );
   tree->Branch( "TestPairNS",     &pr, 32000, 0 );
   tree->Branch( "TestVectorDbl",  &vd );
   tree->Branch( "TestVectorP",    &vP );
   tree->Branch( "TestVectorPNS",  &vP, 32000, 0 );
   tree->Branch( "TestVectorA",    &vA );
   tree->Branch( "TestVectorANS",  &vA, 32000, 0 );
   tree->Branch( "TestVectorAS",   &vAS );
   tree->Branch( "TestVectorASS",  &vAS, 32000, 200 );
   tree->Branch( "TestVectorA",    &vA );
   tree->Branch( "TestVectorANS",  &vA, 32000, 0 );
   tree->Branch( "TestVectorAS",   &vAS );
   tree->Branch( "TestVectorASS",  &vAS, 32000, 200 );
   tree->Branch( "TestVectorB",    &vB );
   tree->Branch( "TestVectorBNS",  &vB, 32000, 0 );
   tree->Branch( "TestVectorBS",   &vBS );
   tree->Branch( "TestVectorBSS",  &vBS, 32000, 200 );
   tree->Branch( "TestVectorC",    &vC );
   tree->Branch( "TestVectorCNS",  &vC, 32000, 0 );
   tree->Branch( "TestVectorCS",   &vCS );
   tree->Branch( "TestVectorCSS",  &vCS, 32000, 200 );
   tree->Branch( "TestVectorD",    &vD );
   tree->Branch( "TestVectorDNS",  &vD, 32000, 0 );
   tree->Branch( "TestVectorDS",   &vDS );
   tree->Branch( "TestVectorDSS",  &vDS, 32000, 200 );
   
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
