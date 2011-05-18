// @(#)root/table:$Id$
// Author: Valery Fine(fine@bnl.gov)   30/06/2001

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include "TGenericTable.h"

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TGenericTable                                                       //
//                                                                      //
//  This is the class to represent the array of C-struct                //
//  defined at run-time                                                 //
//                                                                      //
//   Example: see $ROOTSYS/tutorials/tree/staff.C                       //
//   -------                                                            //
// !{
// !//   example of macro to read data from an ascii file and
// !//   create a root file with an histogram and an ntuple.
// !//   A'la the famous ROOT/PAW staff data example
// !//   ( see PAW - Long write up, CERN, page33. )
// !
// !   gROOT->Reset();
// !   gSystem->Load("libRootKernel");
// !
// !   struct staff_t {
// !                Int_t cat;
// !                Int_t division;
// !                Int_t flag;
// !                Int_t age;
// !                Int_t service;
// !                Int_t children;
// !                Int_t grade;
// !                Int_t step;
// !                Int_t nation;
// !                Int_t hrweek;
// !                Int_t cost;
// !    };
// !
// !   staff_t staff;
// !
// !   // open ASCII data file
// !   FILE *fp = fopen("staff.dat","r");
// !
// !   char line[81];
// !
// !   // Create the generic table for 1000 rows (it may grow then)
// !   TGenericTable *allStaff = new TGenericTable("staff_t","Staff-data",1000);
// !
// !   // Fill the memory resident table
// !   while (fgets(&line,80,fp)) {
// !      sscanf(&line[0] ,"%d%d%d%d", &staff.cat,&staff.division,&staff.flag,&staff.age);
// !      sscanf(&line[13],"%d%d%d%d", &staff.service,&staff.children,&staff.grade,&staff.step);
// !      sscanf(&line[24],"%d%d%d",   &staff.nation,&staff.hrweek,&staff.cost);
// !      allStaff->AddAt(&staff);
// !   }
// !   fclose(fp);
// !   // Delete unused space;
// !   allStaff->Purge();
// !
// !   allStaff->Print(0,10);
// !
// !//  Create ROOT file
// !   TFile *f = new TFile("aptuple.root","RECREATE");
// !          allStaff->Write();
// !   f->Write();
// !
// !   // We should close  TFile otherwise all histograms we create below
// !   // may be written to the file too occasionaly
// !   f->Close();
// !
// !//  Create ROOT Browser
// !   new TBrowser("staff",allStaff);
// !
// !//  Create couple of the histograms
// !   TCanvas *canva = new TCanvas("Staff","CERN Population",600,600);
// !   canva->Divide(1,2);
// !
// !
// !// one can use 2 meta variable:
// !//  n$ - the total number of the rows in the table
// !//  i$ - stands for the current row index i = [0 -> (n$-1)]
// !
// !   gStyle->SetHistFillColor(10);
// !   gStyle->SetHistFillStyle(3013);
// !   canva->cd(1);
// !   allStaff->Draw("age");
// !   canva->Update();
// !   canva->cd(2);
// !   allStaff->Draw("cost");
// !   canva->Update();
// !}
//
//////////////////////////////////////////////////////////////////////////

   ClassImp(TGenericTable)
   TableClassStreamerImp(TGenericTable)

// Create TGenericTable by TTableDescriptor pointer
//______________________________________________________________________________
TGenericTable::TGenericTable(const TTableDescriptor &dsc, const char *name) : TTable(name,dsc.Sizeof()),fColDescriptors(0)
{
  ///////////////////////////////////////////////////////////
  //
  //   Create TGenericTable by TTableDescriptor pointer:
  //
  //   dsc   - Pointer to the table descriptor
  //   name  - The name of this object
  //
  ///////////////////////////////////////////////////////////

   // Create a private copy of the descriptor provided;
   SetDescriptorPointer(new TTableDescriptor(dsc));
   SetGenericType();
}
//______________________________________________________________________________
TGenericTable::TGenericTable(const TTableDescriptor &dsc, Int_t n) : TTable("TGenericTable",n,dsc.Sizeof()),fColDescriptors(0)
{
  ///////////////////////////////////////////////////////////
  //
  //   Create TGenericTable by TTableDescriptor pointer:
  //
  //   dsc   - Pointer to the table descriptor
  //   name  - "TGenericTable"
  //   n     - The initial number of allocated rows
  //
  ///////////////////////////////////////////////////////////

  // Create a provate copy of the descriptor provided;
   SetDescriptorPointer(new TTableDescriptor(dsc));
   SetGenericType();
}

//______________________________________________________________________________
TGenericTable::TGenericTable(const TTableDescriptor &dsc,const char *name,Int_t n) : TTable(name,n,dsc.Sizeof()),fColDescriptors(0)
{
  ///////////////////////////////////////////////////////////
  //
  //   Create TGenericTable by TTableDescriptor pointer:
  //
  //   dsc   - Pointer to the table descriptor
  //   name  - The name of this object
  //   n     - The initial number of allocated rows
  //
  ///////////////////////////////////////////////////////////

  // Create a provate copy of the descriptor provided;
   SetDescriptorPointer(new TTableDescriptor(dsc));
   SetGenericType();
}

// Create TGenericTable by C structure name provided
//______________________________________________________________________________
TGenericTable::TGenericTable(const char *structName, const char *name) : TTable(name,-1),fColDescriptors(0)
{
  ///////////////////////////////////////////////////////////
  //
  //  Create TGenericTable by C structure name provided:
  //
  //   dsc   - Pointer to the table descriptor
  //   name  - The name of this object
  //   n     - The initial number of allocated rows
  //
  ///////////////////////////////////////////////////////////
   TTableDescriptor *dsc = TTableDescriptor::MakeDescriptor(structName);
   if (dsc) {
      SetDescriptorPointer(dsc);
      fSize = dsc->Sizeof();
   }
   if ( !dsc || !fSize) Warning("TGenericTable","Wrong table format");
   SetGenericType();
}
//______________________________________________________________________________
TGenericTable::TGenericTable(const char *structName, Int_t n) : TTable("TGenericTable",-1),fColDescriptors(0)
{
  ///////////////////////////////////////////////////////////
  //
  //  Create TGenericTable by C structure name provided:
  //
  //   dsc   - Pointer to the table descriptor
  //   name  - The name of this object
  //   n     - The initial number of allocated rows
  //
  ///////////////////////////////////////////////////////////

   TTableDescriptor *dsc = TTableDescriptor::MakeDescriptor(structName);
   if (dsc) {
      SetDescriptorPointer(dsc);
      fSize = dsc->Sizeof();
   }
   if ( !dsc || !fSize) Warning("TGenericTable","Wrong table format");
   if (n > 0) Set(n);
   SetGenericType();
}
//______________________________________________________________________________
TGenericTable::TGenericTable(const char *structName, const char *name,Int_t n) : TTable(name,-1),fColDescriptors(0)
{
  ///////////////////////////////////////////////////////////
  //
  //   Create TGenericTable by C structure name provided:
  //
  //   dsc   - Pointer to the table descriptor
  //   name  - The name of this object
  //   n     - The initial number of allocated rows
  //
  ///////////////////////////////////////////////////////////

   TTableDescriptor *dsc = TTableDescriptor::MakeDescriptor(structName);
   if (dsc) {
      SetDescriptorPointer(dsc);
      fSize = dsc->Sizeof();
   }
   if ( !dsc || !fSize) Warning("TGenericTable","Wrong table format dsc=0x%lx, size=%ld",(Long_t)dsc,fSize);
   if (n > 0) Set(n);
   SetGenericType();
}

//______________________________________________________________________________
TGenericTable::~TGenericTable()
{
   //destructor
   delete fColDescriptors;
}
