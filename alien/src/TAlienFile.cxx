// @(#)root/alien:$Name:  $:$Id: TAlienFile.cxx,v 1.1 2003/11/13 15:15:11 rdm Exp $
// Author: Andreas Peters 11/09/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAlienFile                                                           //
//                                                                      //
// A TAlienFile is like a normal TFile except that it reads and writes  //
// its data via an AliEn service.                                       //
// Filenames are standard URL format with protocol "alien".             //
// The following are valid TAlienFile URL's:                            //
//                                                                      //
//    alien:///alice/cern.ch/user/p/peters/test.root                    //
//    alien://alien.cern.ch/alice/cern.ch/user/p/peters/test.root       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TAlienFile.h"
#include "TGrid.h"
#include "TAlien.h"
#include "TROOT.h"

#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#ifndef R__WIN32
#include <unistd.h>
#if defined(R__SUN) || defined(R__SGI) || defined(R__HPUX) || \
    defined(R__AIX) || defined(R__LINUX) || defined(R__SOLARIS) || \
    defined(R__ALPHA) || defined(R__HIUX) || defined(R__FBSD) || \
    defined(R__MACOSX) || defined(R__HURD)
#define HAS_DIRENT
#endif
#endif

#ifdef HAS_DIRENT
#include <dirent.h>
#else
struct dirent {
   ino_t d_ino;
   off_t d_reclen;
   unsigned short d_namlen;
   char d_name[232];
};
#endif


ClassImp(TAlienFile)
ClassImp(TAlienSystem)

//______________________________________________________________________________
TAlienFile::TAlienFile(const char *url, Option_t * option,
                       const char *ftitle, Int_t compress)
   : TFile(url, "NET", ftitle, compress), fUrl(url)
{
   // Create an Alien File Object. An AliEn File is the same as a TFile
   // except that it is being accessed via an Alien service. The url
   // argument must be of the form: alien:/[machine]/path/file.root
   // Using the option access, another access protocol (PFN) can be
   // specified for an LFN e.g.:
   //     "alien:///alice/test.root?access=rfio:///castor/alice/test.root"
   // forces to read the LFN "alien:///alice/test.root" with the physical
   // filename "rfio:///castor/alice/test.root" and the TRFIOFile class.
   //
   // If the file specified in the URL does not exist, is not accessable
   // or can not be created the kZombie bit will be set in the TAlienFile
   // object. Use IsZombie() to see if the file is accessable.
   // For a description of the option and other arguments see the TFile ctor.
   // The preferred interface to this constructor is via TFile::Open().

   TString stmp;
   Bool_t create;
   Bool_t recreate;
   Bool_t update;
   Bool_t read;
   Bool_t isExisting = kFALSE;
   Bool_t isFile = kFALSE;
   char *options;
   char *optionstart;
   TUrl purl(url);

   fOffset = 0;
   fOption = option;
   fOption.ToUpper();
   fSubFile = 0;

   if (fOption == "NEW")
      fOption = "CREATE";

   create = (fOption == "CREATE") ? kTRUE : kFALSE;
   recreate = (fOption == "RECREATE") ? kTRUE : kFALSE;
   update = (fOption == "UPDATE") ? kTRUE : kFALSE;
   read = (fOption == "READ") ? kTRUE : kFALSE;

   if (!create && !recreate && !update && !read) {
      read = kTRUE;
      fOption = "READ";
   }

   // check for a sub protocol in the options like:
   // - alien:///alice/test.root?access=castor:///castor/alice/test.root
   {
      options = StrDup(purl.GetOptions());
      optionstart = options - 1;

      do {
         optionstart++;
         if (!(strncmp(optionstart, "access=", 7))) {
            // filter out the path upto the next options seperated by a comma
            char accesspath[4096];
            char *accessend = strchr(optionstart + 7, ',');
            if (accessend) {
               memcpy(accesspath, optionstart + 7,
                      accessend - (optionstart + 7));
               accesspath[accessend - (optionstart + 7)] = 0;
            } else {
               sprintf(accesspath, "%s", optionstart + 7);
            }

            // create a new URL
            TUrl newUrl(accesspath);
            TUrl convUrl("");
            TString convName = "";

            convUrl = newUrl;

            // convert the AliEn URL naming scheme to the ROOT URL naming scheme

            // castor->rfio
            if ((!(strncmp(newUrl.GetProtocol(), "castor", 6))) || (!(strncmp(newUrl.GetProtocol(), "rfio", 4)))) {
               // convert castor://<machine><path> to rfio:<path>
               convName += "rfio:";
               const char* newUrlp = newUrl.GetFile();
               // do this inconsistant rfio:<filename> stuff ...
               if (((*newUrlp) == '/') && ((*(newUrlp+1)) == '/') && ((*(newUrlp+2) == '/')))
                  newUrlp+=2;
               if (((*newUrlp) == '/') && ((*(newUrlp+1)) == '/') && ((*(newUrlp+2) != '/')))
                  newUrlp+=1;

               convName += newUrlp;
               convUrl = TUrl(convName);
            }
            // convert the AliEn URL
            // file->file
            if (!(strncmp(newUrl.GetProtocol(), "file", 4))) {
               // convert file://<machine><path> to file://<path>
               // need a small trick
               TString fakeString = "alien:";
               fakeString += newUrl.GetFile();
               TUrl fakeUrl(fakeString);
               convName += "file:";
               convName += fakeUrl.GetFile();
               convUrl = TUrl(convName);
            }

            // this works only, if the API runs on the same machine ...
            // use the AliEn file copy mechanism, if we want to read the file
            if (read) {
               printf("Opening File with URL %s\n", convUrl.GetUrl());
               fSubFile = TFile::Open(convUrl.GetUrl(), option, ftitle,
                                      compress);
               if (!fSubFile) {
                  Error("TAlienFile",
                        "Cannot open %s using access url %s!",
                        purl.GetFile(), convUrl.GetUrl());
                  goto zombie;
               } else {
                  return;
               }
            }
         }
      } while ((optionstart = strstr(options, ",")));
      delete [] options;
   }

   // first get an active Grid connection
   if (!gGrid) {
      // no TAlien existing ....
      Error("TAlienFile", "No active GRID connection found");
      goto zombie;
   } else {
      if ((strcmp(gGrid->GetGrid(), "alien"))) {
         Error("TAlienFile", "You don't have an active <alien> grid!\n");
         goto zombie;
      }
   }

   char *fname;
   if ((fname = gSystem->ExpandPathName(fUrl.GetFile()))) {
      if (!strstr(fname, ":/")) {
         stmp = fname;
      } else
         stmp = fname;
      delete[]fname;
      fname = (char *) stmp.Data();
   } else {
      Error("TAlienFile", "error expanding path %s", fUrl.GetFile());
      goto zombie;
   }

   TGrid::gridstat_t statbuf;
   if (gGrid->GridStat(fname, &statbuf) == 0) {
      isExisting = kTRUE;
      if (statbuf.st_mode & S_IFREG) {
         isFile = kTRUE;
      }
   }

   if (recreate) {
      if (isExisting) {
         if (isFile) {
            if ((gGrid->Rm(fname) < 0)) {
               Error("TAlienFile", "error deleting file %s for recreation",
                     fname);
               goto zombie;
            }
         } else {
            Error("TAlienFIle",
                  "error recreating file %s. It is not a plain file",
                  fname);
            goto zombie;
         }
      }

      recreate = kFALSE;
      create = kTRUE;
      fOption = "CREATE";
   }

   if (create && isFile) {
      Error("TAlienFile", "file %s already exists", fname);
      goto zombie;
   }
   if (update) {
      if (!isExisting) {
         update = kFALSE;
         create = kTRUE;
      }
      // for the moment no update of Files ....
      if (update && 1) {
         Error("TAlienFile", "no write permission, could not open file %s",
               fname);
         goto zombie;
      }
   }

   if (read) {
      if ((!isExisting)) {
         Error("TAlienFile", "file %s does not exist", fname);
         goto zombie;
      }

      if ((!isFile)) {
         Error("TAlienFile", "%s is not a file", fname);
         goto zombie;
      }
   }


   if (create || update) {
#ifndef WIN32
      fD = SysOpen(url, O_WRONLY , 0644);
#else
      fD = SysOpen(url, O_WRONLY, S_IREAD | S_IWRITE);
#endif
      if (fD == -1) {
         SysError("TAlienFile", "file %s can not be opened", fname);
         goto zombie;
      }
      fWritable = kTRUE;
   } else {
#ifndef WIN32
      fD = SysOpen(url, O_RDONLY, 0644);
#else
      fD = SysOpen(url, O_RDONLY | O_BINARY, S_IREAD | S_IWRITE);
#endif
      if (fD <= 0) {
         SysError("TAlienFile", "file %s can not be opened for reading",
                  fname);
         goto zombie;
      }
      fWritable = kFALSE;
   }

   Init(create);

   return;

 zombie:
   // error in file opening occured, make this object a zombie
   MakeZombie();
   gDirectory = gROOT;
   return;
}

//______________________________________________________________________________
TAlienFile::~TAlienFile()
{
   // TAlienFile file dtor. Close and flush directory structure.

   if (fSubFile)
      fSubFile->Close();
   else
      Close();
}

//______________________________________________________________________________
Int_t TAlienFile::SysOpen(const char *pathname, Int_t flags, UInt_t mode)
{
   // Interface to system open. All arguments like in "man 2 open".

   if (!gGrid)
      return -1;

   Int_t ret =
       gGrid->GridOpen((char *) pathname, (Int_t) flags, (UInt_t) mode);
   if (ret <= 0)
      gSystem->SetErrorStr("TAlienFile::open failed!");

   return ret;
}

//______________________________________________________________________________
Int_t TAlienFile::SysClose(Int_t fd)
{
   // Interface to system close. All arguments like in "man 2 close".

   if (!gGrid)
      return -1;

   Int_t ret = gGrid->GridClose(fd);
   if (ret < 0)
      gSystem->SetErrorStr("TAlienFile::close failed!");
   return ret;
}

//______________________________________________________________________________
Int_t TAlienFile::SysRead(Int_t fd, void *buf, Int_t len)
{
   // Interface to system read. All arguments like in "man 2 read".

   if (!gGrid)
      return -1;

   Int_t ret = gGrid->GridRead(fd, (char *) buf, len, fOffset);

   if (ret > 0)
      fOffset += ret;

   if (ret < 0)
      gSystem->SetErrorStr("TAlienFile::read failed!");
   return ret;
}

//______________________________________________________________________________
Int_t TAlienFile::SysWrite(Int_t fd, const void *buf, Int_t len)
{
   // Interface to system write. All arguments like in "man 2 write".

   if (!gGrid)
      return -1;

   Int_t ret = gGrid->GridWrite(fd, (char *) buf, len, fOffset);

   if (ret > 0)
      fOffset += ret;

   if (ret < 0)
      gSystem->SetErrorStr("TAlienFile::write failed!");
   return ret;
}

//______________________________________________________________________________
Long64_t TAlienFile::SysSeek(Int_t fd, Long64_t offset, Int_t whence)
{
   // Interface to system lseek. All arguments like in "man 2 lseek"
   // except that the offset and return value are Long_t to be able to
   // handle 64 bit file systems.

   // in the AliEn API the seek is done with the read/write call!
   switch (whence) {
   case SEEK_SET:
      if (offset == fOffset)
         return offset;
      else
         fOffset = offset;
      break;
   case SEEK_CUR:
      fOffset += offset;
      break;
   case SEEK_END:
      fOffset = fEND + offset;
      break;
   }

   return fOffset;
}

//______________________________________________________________________________
Int_t TAlienFile::SysStat(Int_t fd, Long_t * id, Long_t * size,
                          Long_t * flags, Long_t * modtime)
{
   // Interface to TSystem:GetPathInfo(). Generally implemented via
   // stat() or fstat().

   if (!gGrid)
      return -1;

   TGrid::gridstat_t statbuf;

   if (gGrid->GridFstat(fd, &statbuf) >= 0) {
      if (id)
#if defined(R__KCC) && defined(R__LINUX)
         *id = (statbuf.st_dev.__val[0] << 24) + statbuf.st_ino;
#else
         *id = (statbuf.st_dev << 24) + statbuf.st_ino;
#endif
      if (size)
         *size = statbuf.st_size;
      if (modtime)
         *modtime = statbuf.st_mtime;
      if (flags) {
         *flags = 0;
         if (statbuf.
             st_mode & ((S_IEXEC) | (S_IEXEC >> 3) | (S_IEXEC >> 6)))
            *flags |= 1;
         if ((statbuf.st_mode & S_IFMT) == S_IFDIR)
            *flags |= 2;
         if ((statbuf.st_mode & S_IFMT) != S_IFREG &&
             (statbuf.st_mode & S_IFMT) != S_IFDIR)
            *flags |= 4;
      }

      return 0;
   }

   gSystem->SetErrorStr("TAlienFile::fstat failed!");
   return 1;
}

//______________________________________________________________________________
Int_t TAlienFile::SysSync(Int_t fd)
{
   // Interface to system fsync. All arguments like in POSIX fsync().

   if (!gGrid)
      return -1;

   if (TestBit(kDevNull))
      return 0;
   return (gGrid->GridFsync(fd));
}


//______________________________________________________________________________
Bool_t TAlienFile::ReadBuffer(char *buf, Int_t len)
{
   // Read specified byte range from alien.
   // Returns kTRUE in case of error.

   if (fSubFile)
      return fSubFile->ReadBuffer(buf, len);
   if (IsOpen()) {
      ssize_t siz;
      while ((siz = SysRead(fD, buf, len)) < 0 && GetErrno() == EINTR)
         ResetErrno();
      if (siz < 0) {
         SysError("ReadBuffer", "error reading from file %s", GetName());
         return kTRUE;
      }
      if (siz != len) {
         Error("ReadBuffer",
               "error reading all requested bytes from file %s, got %d of %d",
               GetName(), siz, len);
         return kTRUE;
      }
      fBytesRead += siz;
      fgBytesRead += siz;

      return kFALSE;
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TAlienFile::WriteBuffer(const char *buf, Int_t len)
{
   // Write specified byte range to local file
   // Returns kTRUE in case of error.

   if (fSubFile)
      return fSubFile->WriteBuffer(buf, len);

   if (IsOpen() && fWritable) {
      ssize_t siz;
      gSystem->IgnoreInterrupt();
      while ((siz = SysWrite(fD, buf, len)) < 0 && GetErrno() == EINTR)
         ResetErrno();
      gSystem->IgnoreInterrupt(kFALSE);
      if (siz < 0) {
         // Write the system error only once for this file
         SetBit(kWriteError);
         SetWritable(kFALSE);
         SysError("WriteBuffer", "error writing to file %s(%d)", GetName(),siz);
         return kTRUE;
      }
      if (siz != len) {
         Error("WriteBuffer",
               "error writing all requested bytes to file %s, wrote %d of %d",
               GetName(), siz, len);
         return kTRUE;
      }
      fBytesWrite += siz;
      fgBytesWrite += siz;

      return kFALSE;
   }
   return kTRUE;
}

//______________________________________________________________________________
Int_t TAlienFile::GetErrno() const
{
   // Method returning rfio_errno. For RFIO files must use this
   // function since we need to check rfio_errno then serrno and finally errno.
   if (fSubFile)
      return fSubFile->GetErrno();

   return TSystem::GetErrno();
}

//______________________________________________________________________________
void TAlienFile::ResetErrno() const
{
   // Method resetting the rfio_errno, serrno and errno.
   if (fSubFile)
      return fSubFile->ResetErrno();

   TSystem::ResetErrno();
}


//______________________________________________________________________________
TAlienSystem::TAlienSystem():TSystem("-alien", "TAlien Helper System")
{
   // Create helper class that allows directory access via alien.

   // name must start with '-' to bypass the TSystem singleton check
   SetName("alien");

   fDirp = 0;
}

//______________________________________________________________________________
Int_t TAlienSystem::MakeDirectory(const char *dir)
{
   // Make a directory via rfiod.

   TUrl url(dir);

   Int_t ret = gGrid->Mkdir(url.GetFile());
   if (ret < 0)
      gSystem->SetErrorStr("TAlienSystem::mkdir failed!");
   return ret;
}

//______________________________________________________________________________
void *TAlienSystem::OpenDirectory(const char *dir)
{
   // Open a directory via alien. Returns an opaque pointer to a dir
   // structure. Returns 0 in case of error.

   if (!gGrid)
      return 0;

   if (fDirp) {
      Error("OpenDirectory",
            "invalid directory pointer (should never happen)");
      fDirp = 0;
   }

   TUrl url(dir);

   TGrid::gridstat_t finfo;

   if (gGrid->GridStat(url.GetFile(), &finfo) < 0)
      return 0;

   if ((finfo.st_mode & S_IFMT) != S_IFDIR)
      return 0;

   fDirp = (void *) gGrid->GridOpendir(url.GetFile());

   if (!fDirp)
      gSystem->SetErrorStr("TAlienSystem::Opendir failed!");

   return fDirp;
}

//______________________________________________________________________________
void TAlienSystem::FreeDirectory(void *dirp)
{
   // Free directory via alien.

   if (!gGrid)
      return;

   if (dirp != fDirp) {
      Error("FreeDirectory",
            "invalid directory pointer (should never happen)");
      return;
   }

   if (dirp)
      gGrid->GridClosedir((Grid_FileHandle_t) dirp);

   fDirp = 0;
}

//______________________________________________________________________________
const char *TAlienSystem::GetDirEntry(void *dirp)
{
   // Get directory entry via alien. Returns 0 in case no more entries.

   if (!gGrid)
      return 0;

   if (dirp != fDirp) {
      Error("GetDirEntry",
            "invalid directory pointer (should never happen)");
      return 0;
   }

   Grid_Result_t *dp;

   if (dirp) {
      dp = (Grid_Result_t *) gGrid->GridReaddir((Grid_FileHandle_t) dirp);
      if (!dp)
         return 0;
      return dp->name.c_str();
   }
   return 0;
}

//______________________________________________________________________________
Int_t TAlienSystem::GetPathInfo(const char *path, Long_t * id,
                                Long_t * size, Long_t * flags,
                                Long_t * modtime)
{
   // Get info about a file: id, size, flags, modification time.
   // Id      is 0 for RFIO file
   // Size    is the file size
   // Flags   is file type: 0 is regular file, bit 0 set executable,
   //                       bit 1 set directory, bit 2 set special file
   //                       (socket, fifo, pipe, etc.)
   // Modtime is modification time.
   // The function returns 0 in case of success and 1 if the file could
   // not be stat'ed.

   if (!gGrid)
      return -1;

   TUrl url(path);

   TGrid::gridstat_t statbuf;

   if (path && (gGrid->GridStat(url.GetFile(), &statbuf) >= 0)) {
      if (id)
         *id = 0;
      if (size)
         *size = statbuf.st_size;
      if (modtime)
         *modtime = statbuf.st_mtime;
      if (flags) {
         *flags = 0;
         if (statbuf.
             st_mode & ((S_IEXEC) | (S_IEXEC >> 3) | (S_IEXEC >> 6)))
            *flags |= 1;
         if ((statbuf.st_mode & S_IFMT) == S_IFDIR)
            *flags |= 2;
         if ((statbuf.st_mode & S_IFMT) != S_IFREG &&
             (statbuf.st_mode & S_IFMT) != S_IFDIR)
            *flags |= 4;
      }
      return 0;
   }
   return 1;
}

//______________________________________________________________________________
Bool_t TAlienSystem::AccessPathName(const char *path, EAccessMode mode)
{
   // Returns FALSE if one can access a file using the specified access mode.
   // Mode is the same as for the Unix access(2) function.
   // Attention, bizarre convention of return value!!

   if (!gGrid)
      return kTRUE;

   TGrid::gridstat_t statbuf;
   if ((gGrid->GridStat(path, &statbuf)) == 0) {
      return kFALSE;
   }

   return kTRUE;
}
