// @(#)root/proofplayer:$Id$
// Author: G.Ganis July 2011

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


/** \class TProofMonSender
\ingroup proofkernel

Provides the interface for PROOF monitoring to different writers.
Allows to decouple the information sent from the backend.

*/

#include "TProofDebug.h"
#include "TProofMonSender.h"

////////////////////////////////////////////////////////////////////////////////
/// Parse send options from string 'sendopts'.
/// Format is:
///              "[-,+]S[n]|[-,+]D[m]|[-,+]F[j]"
/// where:
///   1. The big letter refers to the 'table' following
///
///           S      table with summary log
///           D      table with dataset info
///           F      table files info
///
///   2. The '-,+' in front disables/enables the related table; if
///      absent '+' is assumed
///
///   3. The number after the letter is the version of the related
///      table
///
/// Returns -1 if nothing is enabled; 0 otherwise

Int_t TProofMonSender::SetSendOptions(const char *sendopts)
{

   // Must have something to parse
   if (sendopts && strlen(sendopts) > 0) {

      PDB(kMonitoring,1) Info("SetSendOptions", "sendopts: '%s'", sendopts);

      Bool_t doit = kTRUE;
      Char_t t = 0;
      Int_t v = -1;
      TString oos(sendopts), oo;
      Ssiz_t from = 0;
      while (oos.Tokenize(oo, from, ":")) {
         PDB(kMonitoring,2) Info("SetSendOptions", "oo: '%s'", oo.Data());
         // Parse info
         doit = kTRUE;
         if (oo.BeginsWith("+")) oo.Remove(0,1);
         if (oo.BeginsWith("-")) { doit = kFALSE; oo.Remove(0,1); }
         PDB(kMonitoring,2) Info("SetSendOptions", "oo: '%s' doit:%d", oo.Data(), doit);
         t = oo[0];
         oo.Remove(0,1);
         PDB(kMonitoring,2) Info("SetSendOptions", "oo: '%s' doit:%d t:'%c'", oo.Data(), doit, t);
         v = -1;
         if (!oo.IsNull() && oo.IsDigit()) v = oo.Atoi();
         PDB(kMonitoring,2) Info("SetSendOptions", "oo: '%s' doit:%d t:'%c' v:%d", oo.Data(), doit, t, v);
         // Fill relevant variables
         TProofMonSender::EConfigBits cbit = kSendSummary;
         if (t == 'D') cbit = kSendDataSetInfo;
         if (t == 'F') cbit = kSendFileInfo;
         if (doit)
            SetBit(cbit);
         else
            ResetBit(cbit);
         if (v > -1) {
            if (t == 'S') fSummaryVrs = v;
            if (t == 'D') fDataSetInfoVrs = v;
            if (t == 'F') fFileInfoVrs = v;
         }
      }
   }

   // Something must be enabled
   if (!(TestBit(kSendSummary) || TestBit(kSendDataSetInfo) || TestBit(kSendFileInfo))) {
      Warning("SetSendOptions", "all tables are disabled!");
      return -1;
   }

   // Notify
   TString snot = TString::Format("%s: sending:", GetTitle());
   if (TestBit(kSendSummary)) snot += TString::Format(" 'summary' (v:%d)", fSummaryVrs);
   if (TestBit(kSendDataSetInfo)) snot += TString::Format(" 'dataset info' (v:%d)", fDataSetInfoVrs);
   if (TestBit(kSendFileInfo)) snot += TString::Format(" 'file info' (v:%d)", fFileInfoVrs);
   Info("SetSendOptions", "%s", snot.Data());

   // Done
   return 0;
}

