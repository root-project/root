// @(#)root/tree:$Id$
// Author: Rene Brun   06/04/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TNtuple                                                              //
//                                                                      //
// A simple tree restricted to a list of float variables only.          //
//                                                                      //
// Each variable goes to a separate branch.                             //
//                                                                      //
//  A Ntuple is created via                                             //
//     TNtuple(name,title,varlist,bufsize)                              //
//  It is filled via:                                                   //
//     TNtuple::Fill(*x)  or                                            //
//     TNtuple::Fill(v1,v2,v3.....)                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TNtuple.h"
#include "TTree.h"
#include "TBranch.h"
#include "TLeaf.h"
#include "TBrowser.h"
#include "Riostream.h"
#include "TClass.h"

#include <string>

ClassImp(TNtuple)

//______________________________________________________________________________
TNtuple::TNtuple(): TTree()
{
//*-*-*-*-*-*Default constructor for Ntuple*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*        ==============================

   fNvar = 0;
   fArgs = 0;
}

//______________________________________________________________________________
TNtuple::TNtuple(const char *name, const char *title, const char *varlist, Int_t bufsize)
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
   fArgs = new Float_t[fNvar];

//  Create one branch for each variable
   for (i=0;i<fNvar;i++) {
      Int_t pv = pvars[i];
      TTree::Branch(&vars[pv],&fArgs[i],&vars[pv],bufsize);
   }

   delete [] vars;
   delete [] pvars;
}

//______________________________________________________________________________
TNtuple::~TNtuple()
{
//*-*-*-*-*-*Default destructor for an Ntuple*-*-*-*-*-*-*-*-*-*-*-*
//*-*        ================================

   delete [] fArgs;
   fArgs = 0;
}

//______________________________________________________________________________
void TNtuple::ResetBranchAddress(TBranch *branch)
{
   // Reset the branch addresses to the internal fArgs array. Use this
   // method when the addresses were changed via calls to SetBranchAddress().

   if (branch) {
      Int_t index = fBranches.IndexOf(branch);
      if (index>=0) {
         branch->SetAddress(&fArgs[index]);
      }
   }
}

//______________________________________________________________________________
void TNtuple::ResetBranchAddresses()
{
   // Reset the branch addresses to the internal fArgs array. Use this
   // method when the addresses were changed via calls to SetBranchAddress().

   for (Int_t i = 0; i < fNvar; i++) {
      TBranch *branch = (TBranch*)fBranches.UncheckedAt(i);
      if (branch) branch->SetAddress(&fArgs[i]);
   }
}

//______________________________________________________________________________
void TNtuple::Browse(TBrowser *b)
{
   // Browse content of the ntuple

   fLeaves.Browse( b );
}


//______________________________________________________________________________
Int_t TNtuple::Fill()
{
//*-*-*-*-*-*-*-*-*Fill a Ntuple with current values in fArgs*-*-*-*-*-*-*
//*-*              ==========================================
// Note that this function is protected.
// Currently called only by TChain::Merge

   return TTree::Fill();
}

//______________________________________________________________________________
Int_t TNtuple::Fill(const Float_t *x)
{
   // Fill a Ntuple with an array of floats


   // Store array x into buffer
   for (Int_t i=0;i<fNvar;i++)  {
      fArgs[i] = x[i];
   }

   return TTree::Fill();
}


//______________________________________________________________________________
Int_t TNtuple::Fill(Float_t x0,Float_t x1,Float_t x2,Float_t x3,Float_t x4
              ,Float_t x5,Float_t x6,Float_t x7,Float_t x8,Float_t x9
              ,Float_t x10,Float_t x11,Float_t x12,Float_t x13,Float_t x14)
{
   // Fill a Ntuple: Each Ntuple item is an argument

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
Long64_t TNtuple::ReadFile(const char *filename, const char * /*branchDescriptor*/)
{
   // Read from filename as many columns as variables in the ntuple
   // the function returns the number of rows found in the file
   // The second argument "branchDescriptor" is currently not used.
   // Lines in the input file starting with "#" are ignored.

   Long64_t nlines = 0;
   ifstream in;
   in.open(filename);
   while (1) {
      if ( in.peek() != '#' ) {
         for (Int_t i=0;i<fNvar;i++) in >> fArgs[i];
         if (!in.good()) break;
         TTree::Fill();
         nlines++;
      }
      in.ignore(8192,'\n');
   }
   in.close();
   return nlines;
}


//_______________________________________________________________________
void TNtuple::Streamer(TBuffer &b)
{
//*-*-*-*-*-*-*-*-*Stream a class object*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*              =========================================
   if (b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = b.ReadVersion(&R__s, &R__c);
      if (R__v > 1) {
         b.ReadClassBuffer(TNtuple::Class(), this, R__v, R__s, R__c);
      } else {
         //====process old versions before automatic schema evolution
         TTree::Streamer(b);
         b >> fNvar;
         b.CheckByteCount(R__s, R__c, TNtuple::IsA());
         //====end of old versions
      }
      if (fNvar <= 0) return;
      fArgs = new Float_t[fNvar];
      for (Int_t i=0;i<fNvar;i++) {
         TBranch *branch = (TBranch*)fBranches.UncheckedAt(i);
         if (branch) branch->SetAddress(&fArgs[i]);
      }
   } else {
      b.WriteClassBuffer(TNtuple::Class(),this);
   }
}
