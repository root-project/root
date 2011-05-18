// @(#)root/tree:$Id$
// Author: Rene Brun   12/08/2001

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TNtupleD                                                             //
//                                                                      //
// A simple tree restricted to a list of double variables only.         //
//                                                                      //
// Each variable goes to a separate branch.                             //
//                                                                      //
//  A Ntuple is created via                                             //
//     TNtupleD(name,title,varlist,bufsize)                             //
//  It is filled via:                                                   //
//     TNtupleD::Fill(*x)  or                                           //
//     TNtupleD::Fill(v1,v2,v3.....)                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TNtupleD.h"
#include "TTree.h"
#include "TBranch.h"
#include "TLeaf.h"
#include "TBrowser.h"
#include "Riostream.h"
#include "TClass.h"

ClassImp(TNtupleD)

//______________________________________________________________________________
TNtupleD::TNtupleD(): TTree()
{
//*-*-*-*-*-*Default constructor for Ntuple*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*        ==============================

   fNvar = 0;
   fArgs = 0;
}

//______________________________________________________________________________
TNtupleD::TNtupleD(const char *name, const char *title, const char *varlist, Int_t bufsize)
       :TTree(name,title)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Create an Ntuple*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ================
//       The parameter varlist describes the list of the ntuple variables
//       separated by a colon:
//         example:  "x:y:z:energy"
//       For each variable in the list a separate branch is created.
//
//      NOTE:
//       -Use TTree to create branches with variables of different data types.
//       -Use TTree when the number of branches is large (> 100). 
//*-*

   Int_t i;
   fNvar = 0;
   fArgs = 0;

//   Count number of variables (separated by :)
   Int_t nch = strlen(varlist);
   if (nch == 0) return;
   char *vars = new char[nch+1];
   strlcpy(vars,varlist,nch+1);
   Int_t *pvars = new Int_t[nch+1];
   fNvar = 1;
   pvars[0] = 0;
   for (i=1;i<nch;i++) {
      if (vars[i] == ':') {
         pvars[fNvar] = i+1;
         vars[i] = 0;
         fNvar++;
      }
   }
   fArgs = new Double_t[fNvar];

//  Create one branch for each variable
   char descriptor[100];
   for (i=0;i<fNvar;i++) {
      Int_t pv = pvars[i];
      snprintf(descriptor,100,"%s/D",&vars[pv]);      
      TTree::Branch(&vars[pv],&fArgs[i],descriptor,bufsize);
   }

   delete [] vars;
   delete [] pvars;
}

//______________________________________________________________________________
TNtupleD::~TNtupleD()
{
//*-*-*-*-*-*Default destructor for an Ntuple*-*-*-*-*-*-*-*-*-*-*-*
//*-*        ================================

   delete [] fArgs;
   fArgs = 0;
}

//______________________________________________________________________________
void TNtupleD::ResetBranchAddress(TBranch *branch)
{
   // Reset the branch addresses to the internal fArgs array. Use this
   // method when the addresses were changed via calls to SetBranchAddress().

   if (branch) {
      UInt_t index = fBranches.IndexOf(branch);
      if (index>0) {
         branch->SetAddress(&fArgs[index]);
      }
   }
}

//______________________________________________________________________________
void TNtupleD::ResetBranchAddresses()
{
   // Reset the branch addresses to the internal fArgs array. Use this
   // method when the addresses were changed via calls to SetBranchAddress().

   for (Int_t i = 0; i < fNvar; i++) {
      TBranch *branch = (TBranch*)fBranches.UncheckedAt(i);
      if (branch) branch->SetAddress(&fArgs[i]);
   }
}

//______________________________________________________________________________
void TNtupleD::Browse(TBrowser *b)
{
   // Browse content.

   fLeaves.Browse( b );
}


//______________________________________________________________________________
Int_t TNtupleD::Fill()
{
//*-*-*-*-*-*-*-*-*Fill a Ntuple with current values in fArgs*-*-*-*-*-*-*
//*-*              ==========================================
// Note that this function is protected.
// Currently called only by TChain::Merge

   return TTree::Fill();
}

//______________________________________________________________________________
Int_t TNtupleD::Fill(const Double_t *x)
{
//*-*-*-*-*-*-*-*-*Fill a Ntuple with an array of floats*-*-*-*-*-*-*-*-*-*
//*-*              =====================================

//*-*- Store array x into buffer
   for (Int_t i=0;i<fNvar;i++)  {
      fArgs[i] = x[i];
   }

   return TTree::Fill();
}


//______________________________________________________________________________
Int_t TNtupleD::Fill(Double_t x0,Double_t x1,Double_t x2,Double_t x3,Double_t x4
              ,Double_t x5,Double_t x6,Double_t x7,Double_t x8,Double_t x9
              ,Double_t x10,Double_t x11,Double_t x12,Double_t x13,Double_t x14)
{
//*-*-*-*-*-*-*-*-*Fill a Ntuple: Each Ntuple item is an argument*-*-*-*-*-*-*
//*-*              ==============================================

   if (fNvar >  0) fArgs[0]  = x0;
   if (fNvar >  1) fArgs[1]  = x1;
   if (fNvar >  2) fArgs[2]  = x2;
   if (fNvar >  3) fArgs[3]  = x3;
   if (fNvar >  4) fArgs[4]  = x4;
   if (fNvar >  5) fArgs[5]  = x5;
   if (fNvar >  6) fArgs[6]  = x6;
   if (fNvar >  7) fArgs[7]  = x7;
   if (fNvar >  8) fArgs[8]  = x8;
   if (fNvar >  9) fArgs[9]  = x9;
   if (fNvar > 10) fArgs[10] = x10;
   if (fNvar > 11) fArgs[11] = x11;
   if (fNvar > 12) fArgs[12] = x12;
   if (fNvar > 13) fArgs[13] = x13;
   if (fNvar > 14) fArgs[14] = x14;

   return TTree::Fill();
}

//_______________________________________________________________________
Long64_t TNtupleD::ReadStream(istream &inputStream, const char * /*branchDescriptor*/, char delimiter)
{
// read from filename as many columns as variables in the ntuple
// the function returns the number of rows found in the file
// The second argument "branchDescriptor" is currently not used.
// Lines in the input file starting with "#" are ignored.
         
   Long64_t nlines = 0;
   char newline = GetNewlineValue(inputStream);
   while (1) {
      if ( inputStream.peek() != '#' ) {
         for (Int_t i=0;i<fNvar;i++) {
            inputStream >> fArgs[i];
            if (inputStream.peek() == delimiter) {
               inputStream.get(); // skip delimiter.
            }
         }
         if (!inputStream.good()) break;
         TTree::Fill();
         ++nlines;
      }
      inputStream.ignore(8192,newline);   
   }
   return nlines;
}

//_______________________________________________________________________
void TNtupleD::Streamer(TBuffer &b)
{
//*-*-*-*-*-*-*-*-*Stream a class object*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*              =========================================
   if (b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = b.ReadVersion(&R__s, &R__c);
      b.ReadClassBuffer(TNtupleD::Class(), this, R__v, R__s, R__c);
      if (fNvar <= 0) return;
      fArgs = new Double_t[fNvar];
      for (Int_t i=0;i<fNvar;i++) {
         TBranch *branch = (TBranch*)fBranches.UncheckedAt(i);
         if (branch) branch->SetAddress(&fArgs[i]);
      }      
   } else {
      b.WriteClassBuffer(TNtupleD::Class(),this);
   }
}
