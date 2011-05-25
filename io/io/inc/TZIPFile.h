// @(#)root/io:$Id$
// Author: Fons Rademakers   30/6/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TZIPFile
#define ROOT_TZIPFile


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TZIPFile                                                             //
//                                                                      //
// This class describes a ZIP archive file containing multiple          //
// sub-files. Typically the sub-files are ROOT files. Notice that       //
// the ROOT files should not be compressed when being added to the      //
// ZIP file, since ROOT files are normally already compressed.          //
// Such a ZIP file should be created like:                              //
//                                                                      //
//    zip -n root multi file1.root file2.root                           //
//                                                                      //
// which creates a ZIP file multi.zip.                                  //
//                                                                      //
// For more on the ZIP file structure see TZIPFile.cxx.                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TArchiveFile
#include "TArchiveFile.h"
#endif

class TZIPMember;


class TZIPFile : public TArchiveFile {

protected:
   Long64_t    fDirPos;     // Central directory position
   Long64_t    fDirSize;    // Central directory size
   Long64_t    fDirOffset;  // Central directory offset (from the beginning of the archive)
   TString     fComment;    // Archive comment

   Long64_t   FindEndHeader();
   Int_t      ReadEndHeader(Long64_t pos);
   Long64_t   ReadZip64EndLocator(Long64_t pos);
   Int_t      ReadZip64EndRecord(Long64_t pos);
   Int_t      ReadDirectory();
   Int_t      ReadMemberHeader(TZIPMember *member);
   UInt_t     Get(const void *buffer, Int_t bytes);
   ULong64_t  Get64(const void *buffer, Int_t bytes);
   Int_t      DecodeZip64ExtendedExtraField(TZIPMember *m, Bool_t global = kTRUE);

   // ZIP archive constants
   enum EZIPConstants {
      // - Archive version required (and made)
      kARCHIVE_VERSION     = 45,

      // - Magic header constants
      kDIR_HEADER_MAGIC       = 0x02014b50,
      kENTRY_HEADER_MAGIC     = 0x04034b50,
      kEND_HEADER_MAGIC       = 0x06054b50,
      kZIP64_EDR_HEADER_MAGIC = 0x06064b50,
      kZIP64_EDL_HEADER_MAGIC = 0x07064b50,
      kZIP64_EXTENDED_MAGIC   = 0x0001,     // Zip64 Extended Information Extra Field
      kZIP_MAGIC_LEN          = 4,          // Length of magic's
      kMAX_VAR_LEN            = 0xffff,     // Max variable-width field length
      kMAX_SIZE               = 0xffffffff, // Max size of things

      // - Offsets into the central directory headers
      kDIR_MAGIC_OFF      = 0,
      kDIR_VMADE_OFF      = 4,    kDIR_VMADE_LEN          = 2,
      kDIR_VREQD_OFF      = 6,    kDIR_VREQD_LEN          = 2,
      kDIR_FLAG_OFF       = 8,    kDIR_FLAG_LEN           = 2,
      kDIR_METHOD_OFF     = 10,   kDIR_METHOD_LEN         = 2,
      kDIR_DATE_OFF       = 12,   kDIR_DATE_LEN           = 4,
      kDIR_CRC32_OFF      = 16,   kDIR_CRC32_LEN          = 4,
      kDIR_CSIZE_OFF      = 20,   kDIR_CSIZE_LEN          = 4,
      kDIR_USIZE_OFF      = 24,   kDIR_USIZE_LEN          = 4,
      kDIR_NAMELEN_OFF    = 28,   kDIR_NAMELEN_LEN        = 2,
      kDIR_EXTRALEN_OFF   = 30,   kDIR_EXTRALEN_LEN       = 2,
      kDIR_COMMENTLEN_OFF = 32,   kDIR_COMMENTLEN_LEN     = 2,
      kDIR_DISK_START_OFF = 34,   kDIR_DISK_START_LEN     = 2,
      kDIR_INT_ATTR_OFF   = 36,   kDIR_INT_ATTR_LEN       = 2,
      kDIR_EXT_ATTR_OFF   = 38,   kDIR_EXT_ATTR_LEN       = 4,
      kDIR_ENTRY_POS_OFF  = 42,   kDIR_ENTRY_POS_LEN      = 4,
      kDIR_HEADER_SIZE    = 46,
      
      // - Offsets into the Zip64 end of central directory record (EDR)
      kZIP64_EDR_MAGIC_OFF      = 0,
      kZIP64_EDR_SIZE_OFF       = 4,  kZIP64_EDR_SIZE_LEN       = 8,
      kZIP64_EDR_VERS_MADE_OFF  = 12, kZIP64_EDR_VERS_MADE_LEN  = 2,
      kZIP64_EDR_VERS_EXT_OFF   = 14, kZIP64_EDR_VERS_EXT_LEN   = 2,
      kZIP64_EDR_DISK_OFF       = 16, kZIP64_EDR_DISK_LEN       = 4,
      kZIP64_EDR_DIR_DISK_OFF   = 20, kZIP64_EDR_DIR_DISK_LEN   = 4,
      kZIP64_EDR_DISK_HDRS_OFF  = 24, kZIP64_EDR_DISK_HDRS_LEN  = 8,
      kZIP64_EDR_TOTAL_HDRS_OFF = 32, kZIP64_EDR_TOTAL_HDRS_LEN = 8,
      kZIP64_EDR_DIR_SIZE_OFF   = 40, kZIP64_EDR_DIR_SIZE_LEN   = 8,
      kZIP64_EDR_DIR_OFFSET_OFF = 48, kZIP64_EDR_DIR_OFFSET_LEN = 8,
      kZIP64_EDR_HEADER_SIZE    = 56,
      
      // - Offsets into the Zip64 end of central directory locator (EDL)
      kZIP64_EDL_MAGIC_OFF      = 0,
      kZIP64_EDL_DISK_OFF       = 4,  kZIP64_EDL_DISK_LEN       = 4,
      kZIP64_EDL_REC_OFFSET_OFF = 8,  kZIP64_EDL_REC_OFFSET_LEN = 8,
      kZIP64_EDL_TOTAL_DISK_OFF = 16, kZIP64_EDL_TOTAL_DISK_LEN = 4,
      kZIP64_EDL_HEADER_SIZE    = 20,

      // - Offsets into the end-of-archive header
      kEND_MAGIC_OFF      = 0,
      kEND_DISK_OFF       = 4,    kEND_DISK_LEN           = 2,
      kEND_DIR_DISK_OFF   = 6,    kEND_DIR_DISK_LEN       = 2,
      kEND_DISK_HDRS_OFF  = 8,    kEND_DISK_HDRS_LEN      = 2,
      kEND_TOTAL_HDRS_OFF = 10,   kEND_TOTAL_HDRS_LEN     = 2,
      kEND_DIR_SIZE_OFF   = 12,   kEND_DIR_SIZE_LEN       = 4,
      kEND_DIR_OFFSET_OFF = 16,   kEND_DIR_OFFSET_LEN     = 4,
      kEND_COMMENTLEN_OFF = 20,   kEND_COMMENTLEN_LEN     = 2,
      kEND_HEADER_SIZE    = 22,

      // - Offsets into the local entry headers
      kENTRY_MAGIC_OFF    = 0,
      kENTRY_VREQD_OFF    = 4,    kENTRY_VREQD_LEN        = 2,
      kENTRY_FLAG_OFF     = 6,    kENTRY_FLAG_LEN         = 2,
      kENTRY_METHOD_OFF   = 8,    kENTRY_METHOD_LEN       = 2,
      kENTRY_DATE_OFF     = 10,   kENTRY_DATE_LEN         = 4,
      kENTRY_CRC32_OFF    = 14,   kENTRY_CRC32_LEN        = 4,
      kENTRY_CSIZE_OFF    = 18,   kENTRY_CSIZE_LEN        = 4,
      kENTRY_USIZE_OFF    = 22,   kENTRY_USIZE_LEN        = 4,
      kENTRY_NAMELEN_OFF  = 26,   kENTRY_NAMELEN_LEN      = 2,
      kENTRY_EXTRALEN_OFF = 28,   kENTRY_EXTRALEN_LEN     = 2,
      kENTRY_HEADER_SIZE  = 30,
      
      // - Offsets into the Zip64 Extended Information Extra Field
      kZIP64_EXTENDED_MAGIC_OFF      = 0,  kZIP64_EXTENDED_MAGIC_LEN      = 2,
      kZIP64_EXTENDED_SIZE_OFF       = 2,  kZIP64_EXTENDED_SIZE_LEN       = 2,
      kZIP64_EXTENDED_USIZE_OFF      = 4,  kZIP64_EXTENDED_USIZE_LEN      = 8,
      kZIP64_EXTENTED_CSIZE_OFF      = 12, kZIP64_EXTENDED_CSIZE_LEN      = 8,
      kZIP64_EXTENDED_HDR_OFFSET_OFF = 20, kZIP64_EXTENDED_HDR_OFFSET_LEN = 8,
      kZIP64_EXTENDED_DISK_OFF       = 28, kZIP64_EXTENDED_DISK_LEN       = 4,
      kZIP64_EXTENDED_SIZE           = 32,

      // - Compression method and strategy
      kSTORED              = 0,            // Stored as is
      kDEFLATED            = 8             // Stored using deflate
   };

   TZIPFile(const TZIPFile&); // Not implemented
   TZIPFile& operator=(const TZIPFile&); // Not implemented

public:
   TZIPFile();
   TZIPFile(const char *archive, const char *member, TFile *file);
   virtual ~TZIPFile() { }

   virtual Int_t OpenArchive();
   virtual Int_t SetCurrentMember();

   void          Print(Option_t *option = "") const;

   ClassDef(TZIPFile,1)  //A ZIP archive file
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TZIPMember                                                           //
//                                                                      //
// A ZIP archive consists of files compressed with the popular ZLIB     //
// compression algorithm; this class records the information about a    //
// single archive member.                                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TZIPMember : public TArchiveMember {

friend class TZIPFile;

private:
   void          *fLocal;     // Extra file header data
   UInt_t         fLocalLen;  // Length of extra file header data
   void          *fGlobal;    // Extra directory data
   UInt_t         fGlobalLen; // Length of extra directory data
   UInt_t         fCRC32;     // CRC-32 for all decompressed data
   UInt_t         fAttrInt;   // Internal file attributes
   UInt_t         fAttrExt;   // External file attributes
   UInt_t         fMethod;    // Compression type
   UInt_t         fLevel;     // Compression level

public:
   TZIPMember();
   TZIPMember(const char *name);
   TZIPMember(const TZIPMember &member);
   TZIPMember &operator=(const TZIPMember &rhs);
   virtual ~TZIPMember();

   void     *GetLocal() const { return fLocal; }
   UInt_t    GetLocalLen() const { return fLocalLen; }
   void     *GetGlobal() const { return fGlobal; }
   UInt_t    GetGlobalLen() const { return fGlobalLen; }
   UInt_t    GetCRC32() const { return fCRC32; }
   UInt_t    GetAttrInt() const { return fAttrInt; }
   UInt_t    GetAttrExt() const { return fAttrExt; }
   UInt_t    GetMethod() const { return fMethod; }
   UInt_t    GetLevel() const { return fLevel; }

   void      Print(Option_t *option = "") const;

   ClassDef(TZIPMember,2)  //A ZIP archive member file
};

#endif
