// @(#)root/tmva $Id: Rule.cxx,v 1.9 2006/11/23 17:43:39 rdm Exp $
// Author: Andreas Hoecker, Joerg Stelzer, Fredrik Tegenfeldt, Helge Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Rule                                                                  *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      A class describung a 'rule'                                               * 
 *      Each internal node of a tree defines a rule from all the parental nodes.  *
 *      A rule consists of atleast 2 nodes.                                       *
 *      Input: a decision tree (in the constructor)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Fredrik Tegenfeldt <Fredrik.Tegenfeldt@cern.ch> - Iowa State U., USA      *
 *      Helge Voss         <Helge.Voss@cern.ch>         - MPI-KP Heidelberg, Ger. *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      Iowa State U.                                                             *
 *      MPI-K Heidelberg, Germany                                                 * 
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//________________________________________________________________________________
//
// Implementation of a rule
//
// A rule is simply a branch or a part of a branch in a tree.
// It fullfills the following:
// * First node is the root node of the originating tree
// * Consists of a minimum of 2 nodes
// * A rule returns for a given event:
//    0 : if the event fails at any node
//    1 : otherwise
// * If the rule contains <2 nodes, it returns 0 SHOULD NOT HAPPEN!
//
// The coefficient is found by either brute force or some sort of
// intelligent fitting. See the RuleEnsemble class for more info.
//________________________________________________________________________________

#include "Riostream.h"

#include "TMVA/Event.h"
#include "TMVA/RuleCut.h"
#include "TMVA/Rule.h"
#include "TMVA/RuleFit.h"
#include "TMVA/RuleEnsemble.h"
#include "TMVA/MethodRuleFit.h"

//_______________________________________________________________________
TMVA::Rule::Rule( RuleEnsemble *re,
                  const std::vector< const Node * >& nodes )
   : fNorm          ( 1.0 )
   , fSupport       ( 0.0 )
   , fSigma         ( 0.0 )
   , fCoefficient   ( 0.0 )
   , fImportance    ( 0.0 )
   , fImportanceRef ( 1.0 )
   , fRuleEnsemble  ( re )
   , fLogger( "RuleFit" )
{
   // the main constructor for a Rule

   //
   // input:
   //   nodes  - a vector of TMVA::Node; from these all possible rules will be created
   //
   //
   
   fCut     = new RuleCut( nodes );
   fSSB     = fCut->GetPurity();
   fSSBNeve = fCut->GetCutNeve();
}

//_______________________________________________________________________
Bool_t TMVA::Rule::ContainsVariable(UInt_t iv) const
{
   // check if variable in node
   Bool_t found    = kFALSE;
   Bool_t doneLoop = kFALSE;
   UInt_t ncuts    = fCut->GetNcuts();
   UInt_t i        = 0;
   //
   while (!doneLoop) {
      found = (fCut->GetSelector(i) == iv);
      i++;
      doneLoop = (found || (i==ncuts));
   }
   return found;
}

//_______________________________________________________________________
Bool_t TMVA::Rule::Equal( const Rule& other, Bool_t useCutValue, Double_t mindist ) const
{
   //
   // Compare two rules.
   // useCutValue: true -> calculate a distance between the two rules based on the cut values
   //                      if the rule cuts are not equal, the distance is < 0 (-1.0)
   //                      return true if d<mindist
   //              false-> ignore mindist, return true if rules are equal, ignoring cut values
   // mindist:     min distance allowed between rules; if < 0 => set useCutValue=false;
   //
   Bool_t rval=kFALSE;
   if (mindist<0) useCutValue=kFALSE;
   Double_t d = RuleDist( other, useCutValue );
   // cut value used - return true if 0<=d<mindist
   if (useCutValue) rval = ( (!(d<0)) && (d<mindist) );
   else rval = (!(d<0));
   // cut value not used, return true if <> -1
   return rval;
}

//_______________________________________________________________________
Double_t TMVA::Rule::RuleDist( const Rule& other, Bool_t useCutValue ) const
{
   // Returns:
   // -1.0 : rules are NOT equal, i.e, variables and/or cut directions are wrong
   //   >=0: rules are equal apart from the cutvalue, returns d = sqrt(sum(c1-c2)^2)
   // If not useCutValue, the distance is exactly zero if they are equal
   //
   if (fCut->GetNcuts()!=other.GetRuleCut()->GetNcuts()) return -1.0; // check number of cuts
   //
   const UInt_t ncuts  = fCut->GetNcuts();
   //
   Int_t    sel;         // cut variable
   Double_t rms;         // rms of cut variable
   Double_t smin;        // distance between the lower range
   Double_t smax;        // distance between the upper range
   Double_t vminA,vmaxA; // min,max range of cut A (cut from this Rule)
   Double_t vminB,vmaxB; // idem from other Rule
   //
   // compare nodes
   // A 'distance' is assigned if the two rules has exactly the same set of cuts but with
   // different cut values.
   // The distance is given in number of sigmas
   //
   UInt_t   in     = 0;    // cut index
   Double_t sumdc2 = 0;    // sum of 'distances'
   Bool_t   equal  = true; // flag if cut are equal
   //
   const RuleCut *otherCut = other.GetRuleCut();
   while ((equal) && (in<ncuts)) {
      // check equality in cut topology
      equal = ( (fCut->GetSelector(in) == (otherCut->GetSelector(in))) &&
                (fCut->GetCutDoMin(in) == (otherCut->GetCutDoMin(in))) &&
                (fCut->GetCutDoMax(in) == (otherCut->GetCutDoMax(in))) );
      // if equal topology, check cut values
      if (equal) {
         if (useCutValue) {
            sel   = fCut->GetSelector(in);
            vminA = fCut->GetCutMin(in);
            vmaxA = fCut->GetCutMax(in);
            vminB = other.GetRuleCut()->GetCutMin(in);
            vmaxB = other.GetRuleCut()->GetCutMax(in);
            // messy - but ok...
            rms = fRuleEnsemble->GetRuleFit()->GetMethodRuleFit()->GetRMS(sel);
            smin=0;
            smax=0;
            if (fCut->GetCutDoMin(in))
               smin = ( rms>0 ? (vminA-vminB)/rms : 0 );
            if (fCut->GetCutDoMax(in))
               smax = ( rms>0 ? (vmaxA-vmaxB)/rms : 0 );
            sumdc2 += smin*smin + smax*smax;
            //            sumw   += 1.0/(rms*rms); // TODO: probably not needed
         }
      }
      in++;
   }
   if (!useCutValue) sumdc2 = (equal ? 0.0:-1.0); // ignore cut values
   else              sumdc2 = (equal ? sqrt(sumdc2) : -1.0);

   return sumdc2;
}

//_______________________________________________________________________
Bool_t TMVA::Rule::operator==( const TMVA::Rule& other ) const
{
   // comparison operator ==

   return this->Equal( other, kTRUE, 1e-3 );
}

//_______________________________________________________________________
Bool_t TMVA::Rule::operator<( const TMVA::Rule& other ) const
{
   // comparison operator <
   return (fImportance < other.GetImportance());
}

//_______________________________________________________________________
ostream& TMVA::operator<< ( ostream& os, const TMVA::Rule& rule )
{
   // ostream operator
   rule.Print( os );
   return os;
}

//_______________________________________________________________________
const TString & TMVA::Rule::GetVarName( Int_t i ) const
{
   // returns the name of a rule

   return fRuleEnsemble->GetMethodRuleFit()->GetInputExp(i);
}

//_______________________________________________________________________
void TMVA::Rule::Copy( const Rule& other )
{
   // copy function
   if(this != &other) {
      SetRuleEnsemble( other.GetRuleEnsemble() );
      fCut = new RuleCut( *(other.GetRuleCut()) );
      fSSB     = other.GetSSB();
      fSSBNeve = other.GetSSBNeve();
      SetCoefficient(other.GetCoefficient());
      SetSupport( other.GetSupport() );
      SetSigma( other.GetSigma() );
      SetNorm( other.GetNorm() );
      CalcImportance();
      SetImportanceRef( other.GetImportanceRef() );
   }
}

//_______________________________________________________________________
void TMVA::Rule::Print( ostream& os ) const
{
   // print function
   const Int_t ncuts = fCut->GetNcuts();
   if (ncuts<1) os << "     *** WARNING - <EMPTY RULE> ***" << endl; // TODO: Fix this, use fLogger
   //
   Int_t sel;
   Double_t valmin, valmax;
   //
   os << "    Importance  = " << Form("%1.4f", fImportance/fImportanceRef) << endl;
   os << "    Coefficient = " << Form("%1.4f", fCoefficient) << endl;
   os << "    Support     = " << Form("%1.4f", fSupport)  << endl;
   os << "    S/(S+B)     = " << Form("%1.4f", fSSB)  << endl;  

   for ( Int_t i=0; i<ncuts; i++) {
      os << "    ";
      sel    = fCut->GetSelector(i);
      valmin = fCut->GetCutMin(i);
      valmax = fCut->GetCutMax(i);
      //
      os << Form("* Cut %2d",i+1) << " : " << std::flush;
      if (fCut->GetCutDoMin(i)) os << Form("%10.3g",valmin) << " < " << std::flush;
      else                      os << "             " << std::flush;
      os << GetVarName(sel) << std::flush;
      if (fCut->GetCutDoMax(i)) os << " < " << Form("%10.3g",valmax) << std::flush;
      else                      os << "             " << std::flush;
      os << std::endl;
   }
}

//_______________________________________________________________________
void TMVA::Rule::PrintLogger(const char *title) const
{
   // print function
   const Int_t ncuts = fCut->GetNcuts();
   if (ncuts<1) fLogger << kWARNING << "BUG TRAP: EMPTY RULE!!!" << Endl;
   //
   Int_t sel;
   Double_t valmin, valmax;
   //
   if (title) fLogger << kINFO << title;
   fLogger << kINFO
           << "Importance  = " << Form("%1.4f", fImportance/fImportanceRef) << Endl;
   //           << "Coefficient = " << Form("%1.4f", fCoefficient) << Endl;


   for ( Int_t i=0; i<ncuts; i++) {
      
      fLogger << kINFO << "            ";
      sel    = fCut->GetSelector(i);
      valmin = fCut->GetCutMin(i);
      valmax = fCut->GetCutMax(i);
      //
      fLogger << kINFO << Form("Cut %2d",i+1) << " : ";
      if (fCut->GetCutDoMin(i)) fLogger << kINFO << Form("%10.3g",valmin) << " < ";
      else                      fLogger << kINFO << "             ";
      fLogger << kINFO << GetVarName(sel);
      if (fCut->GetCutDoMax(i)) fLogger << kINFO << " < " << Form("%10.3g",valmax);
      else                      fLogger << kINFO << "             ";
      fLogger << Endl;
   }
}

//_______________________________________________________________________
void TMVA::Rule::PrintRaw( ostream& os ) const
{
   // extensive print function used to print info for the weight file
   const UInt_t ncuts = fCut->GetNcuts();
   os << "Parameters: "
      << fImportance << " "
      << fImportanceRef << " "
      << fCoefficient << " "
      << fSupport << " "
      << fSigma << " "
      << fNorm << " "
      << fSSB << " "
      << fSSBNeve << " "
      << endl;                                                  \
   os << "N(cuts): " << ncuts << endl; // mark end of nodes
   for ( UInt_t i=0; i<ncuts; i++) {
      os << "Cut " << i << " : " << flush;
      os <<        fCut->GetSelector(i)
         << " " << fCut->GetCutMin(i)
         << " " << fCut->GetCutMax(i)
         << " " << (fCut->GetCutDoMin(i) ? "T":"F")
         << " " << (fCut->GetCutDoMax(i) ? "T":"F")
         << endl;
   }
}

//_______________________________________________________________________
void TMVA::Rule::ReadRaw( istream& istr )
{
   // read function (format is the same as written by PrintRaw)

   TString dummy;
   UInt_t ncuts;
   istr >> dummy
        >> fImportance
        >> fImportanceRef
        >> fCoefficient
        >> fSupport
        >> fSigma
        >> fNorm
        >> fSSB
        >> fSSBNeve;
   istr >> dummy >> ncuts;
   Double_t cutmin,cutmax;
   UInt_t   sel,idum;
   Char_t   bA, bB;
   //
   fCut = new TMVA::RuleCut();
   fCut->SetNcuts( ncuts );
   for ( UInt_t i=0; i<ncuts; i++) {
      istr >> dummy >> idum; // get 'Node' and index
      istr >> dummy;         // get ':'
      istr >> sel >> cutmin >> cutmax >> bA >> bB;
      fCut->SetSelector(i,sel);
      fCut->SetCutMin(i,cutmin);
      fCut->SetCutMax(i,cutmax);
      fCut->SetCutDoMin(i,(bA=='T' ? kTRUE:kFALSE));
      fCut->SetCutDoMax(i,(bB=='T' ? kTRUE:kFALSE));
   }
}

