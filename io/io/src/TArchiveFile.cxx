// @(#)root/io:$Id$
// Author: Fons Rademakers   30/6/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**
\file TArchiveFile.cxx
\class TArchiveFile
\ingroup IO

Class describing an archive file containing multiple sub-files, like a ZIP
or TAR archive.
*/

#include "TArchiveFile.h"
#include "TPluginManager.h"
#include "TROOT.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TError.h"
#include "TUrl.h"
#include <stdlib.h>


ClassImp(TArchiveFile);

////////////////////////////////////////////////////////////////////////////////
/// Specify the archive name and member name.
///
/// \param[in] archive Name of the archive file
/// \param[in] member Name of the ROOT file or integer number
/// \param[in] file Address of the TFile instance from where the call takes place
///
/// The member can be a decimal
/// number which allows to access the n-th sub-file. This method is
/// normally only called via TFile.

TArchiveFile::TArchiveFile(const char *archive, const char *member, TFile *file)
{
   if (!file)
      Error("TArchiveFile", "must specify a valid TFile");

   fFile        = file;
   fArchiveName = archive;
   fMemberName  = member;
   fMemberIndex = -1;
   if (fMemberName.IsDigit())
      fMemberIndex = atoi(fMemberName);
   fMembers     = new TObjArray;
   fMembers->SetOwner();
   fCurMember   = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Dtor.

TArchiveFile::~TArchiveFile()
{
   delete fMembers;
}

////////////////////////////////////////////////////////////////////////////////
/// Return position in archive of current member.

Long64_t TArchiveFile::GetMemberFilePosition() const
{
   return fCurMember ? fCurMember->GetFilePosition() : 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns number of members in archive.

Int_t TArchiveFile::GetNumberOfMembers() const
{
   return fMembers->GetEntriesFast();
}

////////////////////////////////////////////////////////////////////////////////
/// Explicitely make the specified member the current member.
/// Returns -1 in case of error, 0 otherwise.

Int_t TArchiveFile::SetMember(const char *member)
{
   fMemberName  = member;
   fMemberIndex = -1;

   return SetCurrentMember();
}

////////////////////////////////////////////////////////////////////////////////
/// Explicitely make the member with the specified index the current member.
/// Returns -1 in case of error, 0 otherwise.

Int_t TArchiveFile::SetMember(Int_t idx)
{
   fMemberName  = "";
   fMemberIndex = idx;

   return SetCurrentMember();
}

////////////////////////////////////////////////////////////////////////////////
/// Return proper archive file handler depending on passed url.
///
/// The handler is loaded via the plugin manager and is triggered by
/// the extension of the archive file. In case no handler is found 0
/// is returned. The file argument is used to access the archive.
/// The archive should be specified as url with the member name as the
/// anchor, e.g. "root://pcsalo.cern.ch/alice/event_1.zip#tpc.root",
/// where tpc.root is the file in the archive to be opened.
/// Alternatively the sub-file can be specified via its index number,
/// e.g. "root://pcsalo.cern.ch/alice/event_1.zip#3".
/// This function is normally only called via TFile::Open().

TArchiveFile *TArchiveFile::Open(const char *url, TFile *file)
{
   if (!file) {
      ::Error("TArchiveFile::Open", "must specify a valid TFile to access %s",
              url);
      return 0;
   }

   TString archive, member, type;

   if (!ParseUrl(url, archive, member, type))
      return 0;

   TArchiveFile *f = 0;
   TPluginHandler *h;
   if ((h = gROOT->GetPluginManager()->FindHandler("TArchiveFile", type))) {
      if (h->LoadPlugin() == -1)
         return 0;
      f = (TArchiveFile*) h->ExecPlugin(3, archive.Data(), member.Data(), file);
   }

   return f;
}

////////////////////////////////////////////////////////////////////////////////
/// Try to determine if url contains an anchor specifying an archive member.
/// Returns kFALSE in case of an error.

Bool_t TArchiveFile::ParseUrl(const char *url, TString &archive, TString &member,
                              TString &type)
{
   TUrl u(url, kTRUE);

   archive = "";
   member  = "";
   type    = "";

   // get the options and see, if the archive was specified by an option
   // FIXME: hard coded for "zip" archive format
   TString urloptions = u.GetOptions();
   TObjArray *objOptions = urloptions.Tokenize("&");
   for (Int_t n = 0; n < objOptions->GetEntries(); n++) {

      TString loption = ((TObjString*)objOptions->At(n))->GetName();
      TObjArray *objTags = loption.Tokenize("=");
      if (objTags->GetEntries() == 2) {

         TString key   = ((TObjString*)objTags->At(0))->GetName();
         TString value = ((TObjString*)objTags->At(1))->GetName();

         if (!key.CompareTo("zip", TString::kIgnoreCase)) {
            archive = u.GetFile();
            member = value;
            type = "dummy.zip";
         }
      }
      delete objTags;
   }
   delete objOptions;

   if (member != "") {
      // member set by an option
      return kTRUE;
   }

   if (!strlen(u.GetAnchor())) {
      archive = u.GetFile();
      type    = archive;
      return kTRUE;
   }

   archive = u.GetFile();
   member  = u.GetAnchor();
   type    = archive;

   if (archive == "" || member == "") {
      archive = "";
      member  = "";
      type    = "";
      return kFALSE;
   }
   return kTRUE;
}


ClassImp(TArchiveMember);

////////////////////////////////////////////////////////////////////////////////
/// Default ctor.

TArchiveMember::TArchiveMember()
{
   fName         = "";
   fComment      = "";
   fPosition     = 0;
   fFilePosition = 0;
   fCsize        = 0;
   fDsize        = 0;
   fDirectory    = kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Create an archive member file.

TArchiveMember::TArchiveMember(const char *name)
{
   fName         = name;
   fComment      = "";
   fPosition     = 0;
   fFilePosition = 0;
   fCsize        = 0;
   fDsize        = 0;
   fDirectory    = kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Copy ctor.

TArchiveMember::TArchiveMember(const TArchiveMember &member)
   : TObject(member)
{
   fName         = member.fName;
   fComment      = member.fComment;
   fModTime      = member.fModTime;
   fPosition     = member.fPosition;
   fFilePosition = member.fFilePosition;
   fCsize        = member.fCsize;
   fDsize        = member.fDsize;
   fDirectory    = member.fDirectory;
}

////////////////////////////////////////////////////////////////////////////////
/// Assignment operator.

TArchiveMember &TArchiveMember::operator=(const TArchiveMember &rhs)
{
   if (this != &rhs) {
      TObject::operator=(rhs);
      fName         = rhs.fName;
      fComment      = rhs.fComment;
      fModTime      = rhs.fModTime;
      fPosition     = rhs.fPosition;
      fFilePosition = rhs.fFilePosition;
      fCsize        = rhs.fCsize;
      fDsize        = rhs.fDsize;
      fDirectory    = rhs.fDirectory;
   }
   return *this;
}
