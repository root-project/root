// @(#)root/proof:$Id$
// Author: G. Ganis   31/08/06

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofLog                                                            //
//                                                                      //
// Implementation of the PROOF session log handler                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TFile.h"
#include "TMacro.h"
#include "TProofLog.h"
#include "TProofMgr.h"
#include "TObjString.h"
#include "TUrl.h"

ClassImp(TProofLog)

//________________________________________________________________________
TProofLog::TProofLog(const char *stag, const char *url, TProofMgr *mgr)
          : TNamed(stag, url)
{
   // Constructor.

   SetLogToBox();
   fFILE = 0;
   fElem = new TList;
   fElem->SetOwner();
   fMgr = mgr;
   // Set a fake starting time
   fStartTime.Set((UInt_t)0);
   // Extract real starting time
   TString st(stag);
   Int_t idx = st.Index('-');
   if (idx != kNPOS) {
      st.Remove(0, idx+1);
      idx = st.Index('-');
      if (idx != kNPOS) {
         st.Remove(idx);
         if (st.IsDigit()) {
            fStartTime.Set(st.Atoi());
         }
      }
   }
}

//________________________________________________________________________
TProofLog::~TProofLog()
{
   // Destructor.

   SafeDelete(fElem);
}

//________________________________________________________________________
TProofLogElem *TProofLog::Add(const char *ord, const char *url)
{
   // Add new entry to the list of elements.

   TProofLogElem *ple = new TProofLogElem(ord, url, this);
   fElem->Add(ple);
   // Done
   return ple;
}

//________________________________________________________________________
Int_t TProofLog::Retrieve(const char *ord, TProofLog::ERetrieveOpt opt,
                          const char *fname, const char *pattern)
{
   // Retrieve the content of the log file associated with worker 'ord'.
   // If 'ord' is "*" (default), all the workers are retrieved. If 'all'
   // is true, the whole files are retrieved; else a max of
   // fgMaxTransferSize (about 1000 lines) per file is read, starting from
   // the end (i.e. the last ~1000 lines).
   // The received buffer is added to the file fname, if the latter is defined.
   // If opt == TProofLog::kGrep only the lines containing 'pattern' are
   // retrieved (remote grep functionality); to filter out a pattern 'pat' use
   // pattern = "-v pat".
   // Return 0 on success, -1 in case of any error.

   // Validate inputs
   if (opt == TProofLog::kGrep && (!pattern || strlen(pattern) <= 0)) {
      Error("Retrieve", "option 'Grep' requires a pattern");
      return -1;
   }

   Int_t nel = (ord[0] == '*') ? fElem->GetSize() : 1;
   // Iterate over the elements
   TIter nxe(fElem);
   TProofLogElem *ple = 0;
   Int_t nd = 0, nb = 0;
   TString msg;
   while ((ple = (TProofLogElem *) nxe())) {
      if (ord[0] == '*' || !strcmp(ord, ple->GetName())) {
         if (ple->Retrieve(opt, pattern) != 0) {
            nb++;
         } else {
            nd++;
         }
         Float_t frac = ((Float_t)nd + (Float_t)nb) * 100. / (Float_t)nel;
         msg.Form("Retrieving logs: %d ok, %d not ok (%.0f%% processed)\r", nd, nb, frac);
         Prt(msg.Data(), kFALSE);
      }
   }
   Prt("\n");

   // Save to file, if required
   if (fname)
      Save(ord, fname);

   // Done
   return 0;
}

//________________________________________________________________________
void TProofLog::Display(const char *ord, Int_t from, Int_t to)
{
   // Display the content associated with worker 'ord' from line 'from'
   // to line 'to' inclusive. A negative value
   // for 'from' indicates lines counted from the end (tail action); 'to'
   // is ignored in such a case.
   // If 'ord' is "*" (default), all the workers are displayed.

   TString msg;
   if (ord[0] == '*') {
      Int_t nel = (fElem) ? fElem->GetSize() : 0;
      // Write global header
      msg.Form("\n// --------- Displaying PROOF Session logs --------\n"
               "// Server: %s \n// Session: %s \n// # of elements: %d \n"
               "// ------------------------------------------------\n\n",
               GetTitle(), GetName(), nel);
      Prt(msg.Data());
   }
   // Iterate over the elements
   TIter nxe(fElem);
   TProofLogElem *ple = 0;
   while ((ple = (TProofLogElem *) nxe())) {
      if (ord[0] == '*' || !strcmp(ord, ple->GetName()))
         ple->Display(from, to);
   }
   if (ord[0] == '*')
      // Write global tail
      Prt("// --------- End of PROOF Session logs ---------\n");
}

//________________________________________________________________________
void TProofLog::Print(Option_t *opt) const
{
   // Print head info about the content

   Int_t nel = (fElem) ? fElem->GetSize() : 0;
   // Write global header
   fprintf(stderr, "// --------- PROOF Session logs object --------\n");
   fprintf(stderr, "// Server: %s \n", GetTitle());
   fprintf(stderr, "// Session: %s \n", GetName());
   fprintf(stderr, "// # of elements: %d \n", nel);
   fprintf(stderr, "// --------------------------------------------\n");

   // Iterate over the elements
   TIter nxe(fElem);
   TProofLogElem *ple = 0;
   while ((ple = (TProofLogElem *) nxe()))
      ple->Print(opt);

   // Write global tail
   fprintf(stderr, "// --------------------------------------------\n");
}

//________________________________________________________________________
void TProofLog::Prt(const char *what, Bool_t newline)
{
   // Special printing procedure

   if (what) {
      if (LogToBox()) {
         // Send to log box:
         EmitVA("Prt(const char*)", 2, what, kFALSE);
      } else {
         FILE *where = (fFILE) ? (FILE *)fFILE : stderr;
         fputs(what, where);
         if (newline) fputc('\n', where);
      }
   }
}

//________________________________________________________________________
Int_t TProofLog::Save(const char *ord, const char *fname, Option_t *opt)
{
   // Save the content associated with worker 'ord' to finel 'fname'.
   // If 'ord' is "*" (default), the log from all the workers is saved.
   // If 'opt' is "a" the file is open in append mode; otherwise the file
   // is truncated.

   // Make sure we got a file name
   if (!fname) {
      Warning("Save", "filename undefined - do nothing");
      return -1;
   }

   // Open file to write header
   // Check, if the option is to append
   TString option = opt;
   option.ToLower();
   FILE *fout=0;
   if (option.Contains("a")){
      fout = fopen(fname, "a");
   } else {
      fout = fopen(fname, "w");
   }
   if (!fout) {
      Warning("Save", "file could not be opened - do nothing");
      return -1;
   }
   fFILE = (void *) fout;

   TString msg;
   if (ord[0] == '*') {
      Int_t nel = (fElem) ? fElem->GetSize() : 0;
      // Write global header
      msg.Form("\n// --------- Displaying PROOF Session logs --------\n"
               "// Server: %s \n// Session: %s \n// # of elements: %d \n"
               "// ------------------------------------------------\n\n",
               GetTitle(), GetName(), nel);
      Prt(msg.Data());
   }

   // Iterate over the elements
   TIter nxe(fElem);
   TProofLogElem *ple = 0;
   while ((ple = (TProofLogElem *) nxe())) {
      if (ord[0] == '*' || !strcmp(ord, ple->GetName()))
         ple->Display(0);
   }

   if (ord[0] == '*') {
      // Write global tail
      Prt("// --------- End of PROOF Session logs ---------\n");
   }

   // Close file
   fclose(fout);
   fFILE = 0;

   // Done
   return 0;
}

//________________________________________________________________________
Int_t TProofLog::Grep(const char *txt, Int_t from)
{
   // Search lines containing 'txt', starting from line 'from'.
   // Print the lines where this happens.

   if (!txt || strlen(txt) <= 0) {
      Warning("Grep", "text to be searched for is undefined - do nothing");
      return -1;
   }

   Int_t nel = (fElem) ? fElem->GetSize() : 0;
   // Write global header
   TString msg;
   msg.Form("\n// --------- Search in PROOF Session logs --------\n"
            "// Server: %s \n// Session: %s \n// # of elements: %d \n"
            "// Text searched for: \"%s\"", GetTitle(), GetName(), nel, txt);
   Prt(msg.Data());
   if (from > 1) {
      msg.Form("// starting from line %d \n", from);
   } else {
      msg = "\n";
   }
   Prt(msg.Data());
   Prt("// ------------------------------------------------\n");

   // Iterate over the elements
   TIter nxe(fElem);
   TProofLogElem *ple = 0;
   while ((ple = (TProofLogElem *) nxe())) {
      TString res;
      Int_t nf = ple->Grep(txt, res, from);
      if (nf > 0) {
         msg.Form("// Ord: %s - line(s): %s\n", ple->GetName(), res.Data());
         Prt(msg.Data());
      }
   }

   Prt("// ------------------------------------------------\n");

   // Done
   return 0;
}

//________________________________________________________________________
void TProofLog::SetMaxTransferSize(Long64_t maxsz)
{
   // Set max transfer size.

   TProofLogElem::SetMaxTransferSize(maxsz);
}

//
// TProofLogElem
//

Long64_t TProofLogElem::fgMaxTransferSize = 100000; // about 1000 lines

//________________________________________________________________________
TProofLogElem::TProofLogElem(const char *ord, const char *url,
                             TProofLog *logger)
              : TNamed(ord, url)
{
   // Constructor.

   fLogger = logger;
   fMacro = new TMacro;
   fSize = -1;
   fFrom = -1;
   fTo = -1;

   //Note the role here, don't redo at each call of Display()
   if (strstr(GetTitle(), "worker-")) {
      fRole = "worker";
   } else {
      if (strchr(GetName(), '.')) {
         fRole = "submaster";
      } else {
         fRole = "master";
      }
   }
}

//________________________________________________________________________
TProofLogElem::~TProofLogElem()
{
   // Destructor.

   SafeDelete(fMacro);
}

//________________________________________________________________________
Long64_t TProofLogElem::GetMaxTransferSize()
{
   // Get max transfer size.

   return fgMaxTransferSize;
}

//________________________________________________________________________
void TProofLogElem::SetMaxTransferSize(Long64_t maxsz)
{
   // Set max transfer size.

   fgMaxTransferSize = maxsz;
}

//________________________________________________________________________
Int_t TProofLogElem::Retrieve(TProofLog::ERetrieveOpt opt, const char *pattern)
{
   // Retrieve the content of the associated file. The approximate number
   // of lines to be retrieved is given by 'lines', with the convention that
   // 0 means 'all', a positive number means the first 'lines' and a negative
   // number means the last '-lines'. Default is -1000.
   // If opt == TProofLog::kGrep only the lines containing 'pattern' are
   // retrieved (remote grep functionality); to filter out a pattern 'pat' use
   // pattern = "-v pat".
   // Return 0 on success, -1 in case of any error.

   // Make sure we have a reference manager
   if (!fLogger->fMgr || !fLogger->fMgr->IsValid()) {
      Warning("Retrieve", "No reference manager: corruption?");
      return -1;
   }

   // Print some info on the file
   if (gDebug >= 2) {
      Info("Retrieve", "Retrieving from ordinal %s file %s with pattern %s",
         GetName(), GetTitle(), (pattern ? pattern : "(no pattern)"));
   }

   // Determine offsets
   if (opt == TProofLog::kAll) {
      // Re-read everything
      fFrom = 0;
      fTo = -1;
      if (gDebug >= 1)
         Info("Retrieve", "Retrieving the whole file");
   } else if (opt == TProofLog::kLeading) {
      // Read leading part
      fFrom = 0;
      fTo = fgMaxTransferSize;
      if (gDebug >= 1)
         Info("Retrieve", "Retrieving the leading %lld lines of file", fTo);
   } else if (opt == TProofLog::kGrep) {
      // Retrieve lines containing 'pattern', which must be defined
      if (!pattern || strlen(pattern) <= 0) {
         Error("Retrieve", "option 'Grep' requires a pattern");
         return -1;
      }
      if (gDebug >= 1)
         Info("Retrieve", "Retrieving only lines filtered with %s", pattern);
   } else {
      // Read trailing part
      fFrom = -fgMaxTransferSize;
      fTo = -1;
      if (gDebug >= 1)
         Info("Retrieve", "Retrieving the last %lld lines of file", -fFrom);
   }

   // Reset the macro
   SafeDelete(fMacro);
   fMacro = new TMacro;

   // Size to be read
   Long64_t len = (fTo > fFrom) ? fTo - fFrom : -1;

   // Readout the buffer
   TObjString *os = 0;
   if (fLogger->fMgr) {
      TString fileName = GetTitle();
      if (fileName.Contains("__igprof.pp__")) {
         // File is an IgProf log. Override all patterns and preprocess it
         if (gDebug >= 1)
            Info("Retrieve", "Retrieving analyzed IgProf performance profile");
         TString analyzeAndFilter = \
           "|( T=`mktemp` && cat > \"$T\" ; igprof-analyse -d -g \"$T\" ; rm -f \"$T\" )";
         if (pattern && (*pattern == '|'))
            analyzeAndFilter.Append(pattern);
         os = fLogger->fMgr->ReadBuffer(fileName.Data(), analyzeAndFilter.Data());
      }
      else if (opt == TProofLog::kGrep)
         os = fLogger->fMgr->ReadBuffer(fileName.Data(), pattern);
      else
         os = fLogger->fMgr->ReadBuffer(fileName.Data(), fFrom, len);
   }
   if (os) {
      // Loop over lines
      TString ln;
      Ssiz_t from = 0;
      while (os->String().Tokenize(ln, from, "\n"))
         fMacro->AddLine(ln.Data());

      // Cleanup
      delete os;
   }

   // Done
   return 0;
}

//_____________________________________________________________________________
void TProofLogElem::Display(Int_t from, Int_t to)
{
   // Display the current content starting from line 'from' to line 'to'
   // inclusive.
   // A negative value for 'from' indicates lines counted from the end
   // (tail action); 'to' is ignored in such a case.
   // TProofLog::Prt is called to display: the location (screen, file, box)
   // is defined there.
   // Return 0 on success, -1 in case of any error.

   Int_t nls = (fMacro->GetListOfLines()) ?
                fMacro->GetListOfLines()->GetSize() : 0;

   // Starting line
   Int_t i = 0;
   Int_t ie = (to > -1 && to < nls) ? to : nls;
   if (from > 1) {
      if (from <= nls)
         i = from - 1;
   } else if (from < 0) {
      // Tail action
      if (-from <= nls)
         i = nls + from;
      ie = nls;
   }
   // Write header
   TString msg;
   Prt("// --------- Start of element log -----------------\n");
   msg.Form("// Ordinal: %s (role: %s)\n", GetName(), fRole.Data());
   Prt(msg.Data());
   // Separate out the submaster path, if any
   TString path(GetTitle());
   Int_t ic = path.Index(",");
   if (ic != kNPOS) {
      TString subm(path);
      path.Remove(0, ic+1);
      subm.Remove(ic);
      msg.Form("// Submaster: %s \n", subm.Data());
      Prt(msg.Data());
   }
   msg.Form("// Path: %s \n// # of retrieved lines: %d ", path.Data(), nls);
   Prt(msg.Data());
   if (i > 0 || ie < nls) {
      msg.Form("(displaying lines: %d -> %d)\n", i+1, ie);
   } else {
      msg = "\n";
   }
   Prt(msg.Data());
   Prt("// ------------------------------------------------\n");
   // Write lines
   msg = "";
   if (fMacro->GetListOfLines()) {
      TIter nxl(fMacro->GetListOfLines());
      TObjString *os = 0;
      Int_t kk = 0;
      while ((os = (TObjString *) nxl())) {
         kk++;
         if (kk > i) {
            if (msg.Length() < 100000) {
               if (msg.Length() > 0) msg += "\n";
               msg += os->GetName();
            } else {
               Prt(msg.Data());
               msg = "";
            }
         }
         if (kk > ie) break;
      }
   }
   if (msg.Length() > 0) Prt(msg.Data());
   // Write tail
   Prt("// --------- End of element log -------------------\n\n");
}

//________________________________________________________________________
void TProofLogElem::Print(Option_t *) const
{
   // Print a line with the relevant info.

   Int_t nls = (fMacro->GetListOfLines()) ?
                fMacro->GetListOfLines()->GetSize() : 0;
   const char *role = (strstr(GetTitle(), "worker-")) ? "worker" : "master";

   fprintf(stderr, "Ord: %s Host: Role: %s lines: %d\n", GetName(), role, nls);
}

//________________________________________________________________________
void TProofLogElem::Prt(const char *what)
{
   // Special printing procedure.

   if (fLogger)
      fLogger->Prt(what);
}

//________________________________________________________________________
Int_t TProofLogElem::Grep(const char *txt, TString &res, Int_t from)
{
   // Search lines containing 'txt', starting from line 'from'. Return
   // their blanck-separated list into 'res'.
   // Return the number of lines found, or -1 in case of error.

   Int_t nls = (fMacro->GetListOfLines()) ?
                fMacro->GetListOfLines()->GetSize() : 0;

   Int_t nf = 0;
   Int_t i = (from > 0) ? (from - 1) : 0;
   for( ; i < nls; i++) {
      TObjString *os = (TObjString *) fMacro->GetListOfLines()->At(i);
      if (os) {
         if (strstr(os->GetName(), txt)) {
            if (res.Length() > 0)
               res += " ";
            res += (i + 1);
            nf++;
         }
      }
   }

   // Done
   return nf;
}
