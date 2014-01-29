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
#include <cctype>

ClassImp(TNtuple)

//Some aux. functions to read tuple from a text file. No reason to make them memeber-functions.

namespace {

//file:
//    lines
//
//lines:
//    line
//    line lines
//
//line:
//    comment
//    tuple
//    empty-line


//comment:
// '#' non-newline-character-sequence newline-character
//
//non-newline-character-sequence:
// any symbol except '\r' or '\n'
//
//newline-character:
// '\r' | '\n'
void SkipComment(std::istream &input);

//empty-line:
//    newline-character
//    ws-sequence newline-character
//    ws-sequence comment
void SkipEmptyLines(std::istream &input);

//ws-sequence:
//    c such that isspace(c) and c is not a newline-character.
void SkipWSCharacters(std::istream &input);

//Next character is either newline-character, eof or we have some problems reading
//the next symbol.
bool NextCharacterIsEOL(std::istream &input);

}


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
TTree* TNtuple::CloneTree(Long64_t nentries /* = -1 */, Option_t* option /* = "" */)
{
   // Create a clone of this tree and copy nentries.
   //
   // By default copy all entries.
   // Note that only active branches are copied.
   // The compression level of the cloned tree is set to the destination file's
   // compression level.
   //
   // See TTree::CloneTree for more details.

   TNtuple *newtuple = dynamic_cast<TNtuple*> (TTree::CloneTree(nentries,option) );
   if (newtuple) {
      // To deal with the cases of some of the branches where dropped.
      newtuple->fNvar = newtuple->fBranches.GetEntries();
   }
   return newtuple;
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
Long64_t TNtuple::ReadStream(std::istream &inputStream, const char * /*branchDescriptor*/, char delimiter)
{
   // Read from filename as many columns as variables in the ntuple
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

namespace {

//Aux. functions to read tuples from text files.

//______________________________________________________________________________
void SkipComment(std::istream &input)
{
   //Skips everything from '#' to (including) '\r' or '\n'.

   //
   if (0) {//Stupid hack to suppress warnings until I've re-implemented ReadStream.
      SkipEmptyLines(input);
      SkipWSCharacters(input);
      NextCharacterIsEOL(input);
   }
   //
   
   while (input.good()) {
      const char next = input.peek();
      if (input.good()) {
         input.get();
         if (next == '\r' || next == '\n')
            break;
      }
   }
}

//______________________________________________________________________________
void SkipEmptyLines(std::istream &input)
{
   //Skips empty lines (newline-characters), ws-lines (consisting only of whitespace characters + newline-characters).
   
   if (0) {//Stupid hack to suppress warnings until I've re-implemented ReadStream.
      SkipComment(input);
   }
   
   while (input.good()) {
      const char c = input.peek();
      if (!input.good())
         break;

      if (c == '#')
         SkipComment(input);
      else if (!std::isspace(c))//'\r' and '\n' are also 'isspaces'.
         break;
      else
         input.get();
   }
}

//______________________________________________________________________________
void SkipWSCharacters(std::istream &input)
{
   //Skip whitespace characters, but not newline-characters we support ('\r' or '\n').
   while (input.good()) {
      const char next = input.peek();
      if (input.good()) {
         if (std::isspace(next) && next != '\n' && next != '\r')
            input.get();
         else
            break;
      }
   }
}

//______________________________________________________________________________
bool NextCharacterIsEOL(std::istream &input)
{
   //Valid here means it's a digit.
   if (!input.good())
      return true;
   
   const char next = input.peek();
   if (!input.good())
      return true;
  
   return next == '\r' || next == '\n';
}

}
