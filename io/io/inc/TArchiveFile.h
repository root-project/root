// @(#)root/io:$Id$
// Author: Fons Rademakers   30/6/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TArchiveFile
#define ROOT_TArchiveFile

#include "TObject.h"
#include "TString.h"
#include "TDatime.h"

class TFile;
class TArchiveMember;
class TObjArray;


class TArchiveFile : public TObject {

private:
   TArchiveFile(const TArchiveFile&);            ///< Not implemented because TArchiveFile can not be copied.
   TArchiveFile& operator=(const TArchiveFile&); ///< Not implemented because TArchiveFile can not be copied.

protected:
   TString         fArchiveName;  ///< Archive file name
   TString         fMemberName;   ///< Sub-file name
   Int_t           fMemberIndex;  ///< Index of sub-file in archive
   TFile          *fFile;         ///< File stream used to access the archive
   TObjArray      *fMembers;      ///< Members in this archive
   TArchiveMember *fCurMember;    ///< Current archive member

   static Bool_t ParseUrl(const char *url, TString &archive, TString &member, TString &type);

public:
   TArchiveFile() : fArchiveName(""), fMemberName(""), fMemberIndex(-1), fFile(0), fMembers(0), fCurMember(0) { }
   TArchiveFile(const char *archive, const char *member, TFile *file);
   virtual ~TArchiveFile();

   virtual Int_t   OpenArchive() = 0;
   virtual Int_t   SetCurrentMember() = 0;
   virtual Int_t   SetMember(const char *member);
   virtual Int_t   SetMember(Int_t idx);

   Long64_t        GetMemberFilePosition() const;
   TArchiveMember *GetMember() const { return fCurMember; }
   TObjArray      *GetMembers() const { return fMembers; }
   Int_t           GetNumberOfMembers() const;

   const char     *GetArchiveName() const { return fArchiveName; }
   const char     *GetMemberName() const { return fMemberName; }
   Int_t           GetMemberIndex() const { return fMemberIndex; }

   static TArchiveFile *Open(const char *url, TFile *file);

   ClassDef(TArchiveFile,1)  //An archive file containing multiple sub-files (like a ZIP archive)
};


class TArchiveMember : public TObject {

friend class TArchiveFile;

protected:
   TString    fName;          ///< Name of member
   TString    fComment;       ///< Comment field
   TDatime    fModTime;       ///< Modification time
   Long64_t   fPosition;      ///< Byte position in archive
   Long64_t   fFilePosition;  ///< Byte position in archive where member data starts
   Long64_t   fCsize;         ///< Compressed size
   Long64_t   fDsize;         ///< Decompressed size
   Bool_t     fDirectory;     ///< Flag indicating this is a directory

public:
   TArchiveMember();
   TArchiveMember(const char *name);
   TArchiveMember(const TArchiveMember &member);
   TArchiveMember &operator=(const TArchiveMember &rhs);
   virtual ~TArchiveMember() { }

   const char *GetName() const { return fName; }
   const char *GetComment() const { return fComment; }
   TDatime     GetModTime() const { return fModTime; }
   Long64_t    GetPosition() const { return fPosition; }
   Long64_t    GetFilePosition() const { return fFilePosition; }
   Long64_t    GetCompressedSize() const { return fCsize; }
   Long64_t    GetDecompressedSize() const { return fDsize; }
   Bool_t      IsDirectory() const { return fDirectory; }

   ClassDef(TArchiveMember,1)  //An archive member file
};

#endif
