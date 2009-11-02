// $Id$

const char *XrdSutPFileCVSID = "$Id$";
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>

#include "XrdSut/XrdSutAux.hh"
#include "XrdSut/XrdSutPFEntry.hh"
#include "XrdSut/XrdSutPFile.hh"
#include "XrdSut/XrdSutTrace.hh"

//_________________________________________________________________
XrdSutPFEntInd::XrdSutPFEntInd(const char *n, kXR_int32 no,
                               kXR_int32 eo, kXR_int32 es)
{
   // Constructor

   name = 0;
   if (n) {
      name = new char[strlen(n)+1];
      if (name)
         strcpy(name,n);
   }
   nxtofs = no;
   entofs = eo;
   entsiz = es;
} 

//_________________________________________________________________
XrdSutPFEntInd::XrdSutPFEntInd(const XrdSutPFEntInd &ei)
{
   //Copy constructor

   name = 0;
   if (ei.name) {
      name = new char[strlen(ei.name)+1];
      if (name)
         strcpy(name,ei.name);
   }
   nxtofs = ei.nxtofs;
   entofs = ei.entofs;
   entsiz = ei.entsiz;
}

//_________________________________________________________________
void XrdSutPFEntInd::SetName(const char *n)
{
   // Name setter

   if (name) {
      delete[] name;
      name = 0;
   }
   if (n) {
      name = new char[strlen(n)+1];
      if (name)
         strcpy(name,n);
   }
}

//______________________________________________________________________________
XrdSutPFEntInd& XrdSutPFEntInd::operator=(const XrdSutPFEntInd ei)
{
   // Assign index entry ei to local index entry.

   name = 0;
   if (ei.name) {
      name = new char[strlen(ei.name)+1];
      if (name)
         strcpy(name,ei.name);
   }
   nxtofs = ei.nxtofs;
   entofs = ei.entofs;
   entsiz = ei.entsiz;

   return *this;
}

//_________________________________________________________________
XrdSutPFHeader::XrdSutPFHeader(const char *id, kXR_int32 v, kXR_int32 ct,
                           kXR_int32 it, kXR_int32 ent, kXR_int32 ofs)
{
   // Constructor

   memset(fileID,0,kFileIDSize);
   if (id) {
      kXR_int32 lid = strlen(id); 
      if (lid  > kFileIDSize)
         lid = kFileIDSize; 
      memcpy(fileID,id,lid);
   }
   version = v;
   ctime = ct;
   itime = it;
   entries = ent;
   indofs = ofs;
   jnksiz = 0;     // At start everything is reachable
} 

//_________________________________________________________________
XrdSutPFHeader::XrdSutPFHeader(const XrdSutPFHeader &fh)
{
   // Copy constructor

   memcpy(fileID,fh.fileID,kFileIDSize); 
   version = fh.version;
   ctime = fh.ctime;
   itime = fh.itime;
   entries = fh.entries;
   indofs = fh.indofs;
   jnksiz = fh.jnksiz;
}

//_________________________________________________________________
void XrdSutPFHeader::Print() const
{
   // Header printout

   struct tm tst;

   // String form for time of last change
   char sctime[256] = {0};
   time_t ttmp = ctime;
   localtime_r(&ttmp,&tst);
   asctime_r(&tst,sctime);
   sctime[strlen(sctime)-1] = 0;

   // String form for time of last index change
   char sitime[256] = {0};
   ttmp = itime;
   localtime_r(&ttmp,&tst);
   asctime_r(&tst,sitime);
   sitime[strlen(sitime)-1] = 0;

   fprintf(stdout,
       "//------------------------------------"
                "------------------------------//\n"
       "// \n"
       "//  File Header dump \n"
       "// \n"
       "//  File ID:          %s \n"
       "//  version:          %d \n"
       "//  last changed on:  %s (%d sec) \n"
       "//  index changed on: %s (%d sec) \n"
       "//  entries:          %d  \n"
       "//  unreachable:      %d  \n"
       "//  first ofs:        %d  \n"
       "// \n"
       "//------------------------------------"
                "------------------------------//\n",
       fileID,version,sctime,ctime,sitime,itime,entries,jnksiz,indofs); 
}

//________________________________________________________________
XrdSutPFile::XrdSutPFile(const char *n, kXR_int32 openmode,
                         kXR_int32 createmode, bool hashtab)
{
   // Constructor

   name = 0; 
   if (n) {
      name = new char[strlen(n)+1];
      if (name)
         strcpy(name,n);
   }
   valid = 0;
   fFd = -1;
   fHTutime = -1;
   fHashTable = 0;

   valid = Init(n, openmode, createmode, hashtab);
}

//________________________________________________________________
XrdSutPFile::XrdSutPFile(const XrdSutPFile &f)
{
   // Copy constructor

   name = 0; 
   if (f.name) { 
      name = new char[strlen(f.name)+1];
      if (name)
         strcpy(name,f.name);
   }
   fFd = f.fFd ;
}

//________________________________________________________________
XrdSutPFile::~XrdSutPFile()
{
   // Destructor

   if (name)
      delete[] name;
   name = 0;
   if (fHashTable)
      delete fHashTable;
   fHashTable = 0;

   Close(); 
}

//________________________________________________________________
bool XrdSutPFile::Init(const char *n, kXR_int32 openmode,
                       kXR_int32 createmode, bool hashtab)
{
   // (re)initialize PFile

   // Make sure it is closed
   Close();

   // Reset members
   if (name)
      delete[] name;
   name = 0; 
   if (n) {
      name = new char[strlen(n)+1];
      if (name)
         strcpy(name,n);
   }
   valid = 0;
   fFd = -1;
   fHTutime = -1;
   if (fHashTable)
      delete fHashTable;
   fHashTable = 0;

   // If name is missing nothing can be done
   if (!name)
      return 0;

   // open modes
   bool create    = (openmode & kPFEcreate);
   bool leaveopen = (openmode & kPFEopen);

   // If file does not exists, create it with default header
   struct stat st;
   if (stat(name, &st) == -1) {
      if (errno == ENOENT) {
         if (create) {
            if (Open(1,0,0,createmode) > 0) {
               kXR_int32 ct = (kXR_int32)time(0);
               XrdSutPFHeader hdr(kDefFileID,kXrdIFVersion,ct,ct,0,0);
               WriteHeader(hdr);
               valid = 1;
               if (!leaveopen)
                  Close();
            }
         } else {
            Err(kPFErrNoFile,"Init",name);
         }
      }
   } else {
      // Fill the the hash table
      if (Open(1) > 0) {
         if (hashtab)
            UpdateHashTable();
         valid = 1;
         if (!leaveopen)
            Close();
      }
   }
   // We are done
   return valid;
}

//_________________________________________________________________
kXR_int32 XrdSutPFile::Open(kXR_int32 opt, bool *wasopen,
                            const char *nam, kXR_int32 createmode)
{
   // Open the stream, so defining fFd .
   // Valid options:
   //        0      read only
   //        1      read/write append
   //        2      read/write truncate
   // For options 1 and 2 the file is created, if not existing,
   // and permission set to createmode (default: 0600).
   // If the file name ends with 'XXXXXX' and it does not exist,
   // it is created as temporary using mkstemp.
   // The file is also exclusively locked.
   // If nam is defined it is used as file name
   // If the file is already open and wasopen is allocated, then *wasopen
   // is set to true
   // The file descriptor of the open file is returned
   XrdOucString copt(opt);

   // Reset was open flag
   if (wasopen) *wasopen = 0;

   // File name must be defined
   char *fnam = (char *)nam;
   if (!fnam)
      fnam = name;
   if (!fnam)
      return Err(kPFErrBadInputs,"Open");

   // If already open, do nothing
   if (!nam && fFd > -1) {
      if (opt > 0) {
         // Make sure that the write flag is set
         long omode = 0;
         if (fcntl(fFd, F_GETFL, &omode) != -1) {
            if (!(omode | O_WRONLY))
               return Err(kPFErrFileAlreadyOpen,"Open");
         }
      }
      if (wasopen) *wasopen = 1;
      return fFd;
   }

   // Ok, we have a file name ... check if it exists already
   bool newfile = 0;
   struct stat st;
   if (stat(fnam, &st) == -1) {
      if (errno != ENOENT) {
         return Err(kPFErrNoFile,"Open",fnam);
      } else {
         if (opt == 0)
            return Err(kPFErrStat,"Open",fnam);
         newfile = 1;
      }
   }

   // Now open it
   if (!nam)
      fFd = -1;
   kXR_int32 fd = -1;
   //
   // If we have to create a new file and the file name ends with
   // 'XXXXXX', make it temporary with mkstemp
   char *pn = strstr(fnam,"XXXXXX");
   if (pn && (pn == (fnam + strlen(fnam) - 6))) {
      if (opt > 0 && newfile) {
         fd = mkstemp(fnam);
         if (fd <= -1)
            return Err(kPFErrFileOpen,"Open",fnam);
      }
   }
   //
   // If normal file act according to requests
   if (fd <= -1) {
      kXR_int32 mode = 0;
      switch (opt) {
         case 2:
            //
            // Forcing truncation 
            mode |= O_TRUNC ;
         case 1:
            //
            // Read / Write
            mode |= O_RDWR ;
            if (newfile)
               mode |= O_CREAT ;
            break;
         case 0:
            //
            // Read only
            mode = O_RDONLY ;
            break;
         default:
            //
            // Unknown option
            return Err(kPFErrBadOp,"Open",copt.c_str());
      }

      // Open file (createmode is only used if O_CREAT is set)
      fd = open(fnam, mode, createmode);
      if (fd <= -1)
         return Err(kPFErrFileOpen,"Open",fnam);
   }

   //
   // Shared or exclusive lock of the whole file
   int lockmode = (opt > 0) ? (F_WRLCK | F_RDLCK) : F_RDLCK;
   int lck = kMaxLockTries;
   int rc = 0;
   while (lck && rc == -1) {
#ifdef __macos__
      struct flock flck = {0, 0, 0, lockmode, SEEK_SET};
#else
      struct flock flck = {lockmode, SEEK_SET, 0, 0};
#endif
      if ((rc = fcntl(fd, F_SETLK, &flck)) == 0)
         break;
      struct timespec lftp, rqtp = {1, 0};
      while (nanosleep(&rqtp, &lftp) < 0 && errno == EINTR) {
         rqtp.tv_sec  = lftp.tv_sec;
         rqtp.tv_nsec = lftp.tv_nsec;
      }
   }
   if (rc == -1) {
      if (errno == EACCES || errno == EAGAIN) {
         // File locked by other process
         int pid = -1;
#ifdef __macos__
         struct flock flck = {0, 0, 0, lockmode, SEEK_SET};
#else
         struct flock flck = {lockmode, SEEK_SET, 0, 0};
#endif
         if (fcntl(fd,F_GETLK,&flck) != -1)
            pid = flck.l_pid;
         close(fd);
         return Err(kPFErrFileLocked,"Open",fnam,(const char *)&pid);
      } else {
         // Error
         return Err(kPFErrLocking,"Open",fnam,(const char *)&fd);
      }
   }

   // Ok, we got the file open and locked
   if (!nam)
      fFd = fd;
   return fd;
}

//_________________________________________________________________
kXR_int32 XrdSutPFile::Close(kXR_int32 fd)
{
   // Close the open stream or descriptor fd, if > -1 .
   // The file is unlocked before.

   // If not open, do nothing
   if (fd < 0)
      fd = fFd;
   if (fd < 0)
   return 0;

   //
   // Unlock the file
#ifdef __macos__
   struct flock flck = {0, 0, 0, F_UNLCK, SEEK_SET};
#else
   struct flock flck = {F_UNLCK, SEEK_SET, 0, 0};
#endif
   if (fcntl(fd, F_SETLK, &flck) == -1) {
      close(fd);
      return Err(kPFErrUnlocking,"Close",(const char *)&fd);
   }

   //
   // Close it
   close(fd);

   // Reset file descriptor
   if (fd == fFd)
      fFd = -1;

   return 0;
}

//_________________________________________________________________
kXR_int32 XrdSutPFile::UpdateHeader(XrdSutPFHeader hd)
{
   // Write/Update header to beginning of file

   //
   // Open the file
   if (Open(1) < 0)
      return -1;

   // Write
   kXR_int32 nw = WriteHeader(hd);

   // Close the file
   Close();
 
   return nw;
}

//_________________________________________________________________
kXR_int32 XrdSutPFile::RetrieveHeader(XrdSutPFHeader &hd)
{
   // Retrieve number of entries in the file

   //
   // Open the file
   bool wasopen = 0;
   if (Open(1, &wasopen) < 0)
      return -1;

   // Read header
   kXR_int32 rc = ReadHeader(hd);

   // Close the file
   if (!wasopen) Close();
 
   return rc;
}

//_________________________________________________________________
kXR_int32 XrdSutPFile::WriteHeader(XrdSutPFHeader hd)
{
   // Write/Update header to beginning of opne stream

   //
   // Build output buffer
   // Get total lenght needed
   kXR_int32 ltot = hd.Length();
   //
   // Allocate the buffer
   char *bout = new char[ltot];
   if (!bout)
      return Err(kPFErrOutOfMemory,"WriteHeader");
   //
   // Fill the buffer
   kXR_int32 lp = 0;
   // File ID
   memcpy(bout+lp,hd.fileID,kFileIDSize);
   lp += kFileIDSize;
   // version
   memcpy(bout+lp,&hd.version,sizeof(kXR_int32));
   lp += sizeof(kXR_int32);
   // change time
   memcpy(bout+lp,&hd.ctime,sizeof(kXR_int32));
   lp += sizeof(kXR_int32);
   // index change time
   memcpy(bout+lp,&hd.itime,sizeof(kXR_int32));
   lp += sizeof(kXR_int32);
   // entries
   memcpy(bout+lp,&hd.entries,sizeof(kXR_int32));
   lp += sizeof(kXR_int32);
   // offset of the first index entry
   memcpy(bout+lp,&hd.indofs,sizeof(kXR_int32));
   lp += sizeof(kXR_int32);
   // number of unused bytes
   memcpy(bout+lp,&hd.jnksiz,sizeof(kXR_int32));
   lp += sizeof(kXR_int32);
   // Check length
   if (lp != ltot) {
      if (bout) delete[] bout;
      return Err(kPFErrLenMismatch,"WriteHeader",
                 (const char *)&lp, (const char *)&ltot);
   }
   //
   // Ready to write: check we got the file
   if (fFd < 0)
      return Err(kPFErrFileNotOpen,"WriteHeader");
   //
   // Set the offset
   if (lseek(fFd, 0, SEEK_SET) == -1) {
      return Err(kPFErrSeek,"WriteHeader","SEEK_SET",(const char *)&fFd);
   }

   kXR_int32 nw = 0;
   // Now write the buffer to the stream
   while ((nw = write(fFd, bout, ltot)) < 0 && errno == EINTR)
      errno = 0;
 
   return nw;
}

//______________________________________________________________________
kXR_int32 XrdSutPFile::WriteEntry(XrdSutPFEntry ent)
{
   // Write entry to file
   // Look first if an entry with the same name exists: in such
   // case try to overwrite the allocated file region; if the space
   // is not enough, set the existing entry inactive and write
   // the new entry at the end of the file, updating all the
   // pointers.
   // File must be opened in read/write mode (O_RDWR).

   // Make sure that the entry is named (otherwise we can't do nothing)
   if (!ent.name)
      return Err(kPFErrBadInputs,"WriteEntry");

   //
   // Ready to write: open the file
   bool wasopen = 0;
   if (Open(1, &wasopen) < 0)
      return -1;

   kXR_int32 ofs = 0;
   kXR_int32 nw = 0;
   kXR_int32 indofs = 0;
   // Read the header
   XrdSutPFHeader header;
   if (ReadHeader(header) < 0) {
      if (!wasopen) Close();
      return -1;
   }
   if ((ofs = lseek(fFd, 0, SEEK_CUR)) == -1) {
      if (!wasopen) Close();
      return Err(kPFErrSeek,"WriteEntry","SEEK_CUR",(const char *)&fFd);
   }

   XrdSutPFEntInd ind;
   // If first entry, write it, update the info and return
   if (header.entries == 0) {
      if ((nw = WriteEnt(ofs, ent)) < 0) {
         if (!wasopen) Close();
         return -1;
      }
      ind.SetName(ent.name);
      ind.nxtofs = 0;
      ind.entofs = ofs;
      ind.entsiz = nw;
      indofs = ofs + nw;
      if (WriteInd(indofs, ind) < 0) {
         if (!wasopen) Close();
         return -1;
      }
      // Update header
      header.entries = 1;
      header.indofs  = indofs;
      header.ctime   = time(0);
      header.itime   = header.ctime;
      if (WriteHeader(header) < 0) {
         if (!wasopen) Close();
         return -1;
      }
      if (!wasopen) Close();
      return nw;
   }

   // First Localize existing entry, if any
   kXR_int32 nr = 1;
   bool found = 0;
   indofs = header.indofs;
   kXR_int32 lastindofs = indofs;
   while (!found && nr > 0 && indofs > 0) {
      nr = ReadInd(indofs, ind);
      if (nr) {
         if (ind.entofs > 0 && !strcmp(ent.name,ind.name)) {
            found = 1;
            break;
         }
         lastindofs = indofs;
         indofs = ind.nxtofs;
      }
   }

   //
   // If an entry already exists and there is enough space to
   // store the update, write the update at the already allocated
   // space; if not, add it at the end.
   if (found) {
      // Update 
      kXR_int32 ct = 0;
      if (ind.entsiz >= ent.Length()) {
         // The offset is set inside ...
         if ((nw = WriteEnt(ind.entofs, ent)) < 0) {
            if (!wasopen) Close();
            return -1;
         }
      } else {
         // Add it at the end
         kXR_int32 entofs = 0;
         if ((entofs = lseek(fFd, 0, SEEK_END)) == -1) {
            if (!wasopen) Close();
            return Err(kPFErrSeek,"WriteEntry",
                       "SEEK_END",(const char *)&fFd);
         }
         if ((nw = WriteEnt(entofs, ent)) < 0) {
            if (!wasopen) Close();
            return -1;
         }
         // Set existing entry inactive
         kXR_int32 wrtofs = ind.entofs;
         if (lseek(fFd, wrtofs, SEEK_SET) == -1) {
            if (!wasopen) Close();
            return Err(kPFErrSeek,"WriteEntry",
                       "SEEK_SET",(const char *)&fFd);
         }
         short status = kPFE_inactive;
         while (write(fFd, &status, sizeof(short)) < 0 &&
                errno == EINTR) errno = 0;
         // Reset entry area
         if (Reset(wrtofs + sizeof(short), ind.entsiz - sizeof(short)) < 0) {
            if (!wasopen) Close();
            return -1;
         }
         // Count as unused bytes
         header.jnksiz += ind.entsiz;
         if (lseek(fFd, kOfsJnkSiz, SEEK_SET) == -1) {
            if (!wasopen) Close();
            return Err(kPFErrSeek,"WriteEntry",
                       "SEEK_SET",(const char *)&fFd);
         }
         while (write(fFd, &header.jnksiz, sizeof(kXR_int32)) < 0 &&
            errno == EINTR) errno = 0;
         // Update the entry index and new size
         wrtofs = indofs + 2*sizeof(kXR_int32);
         if (lseek(fFd, wrtofs, SEEK_SET) == -1) {
            if (!wasopen) Close();
            return Err(kPFErrSeek,"WriteEntry",
                       "SEEK_SET",(const char *)&fFd);
         }
         while (write(fFd, &entofs, sizeof(kXR_int32)) < 0 &&
                errno == EINTR) errno = 0;
         while (write(fFd, &nw, sizeof(kXR_int32)) < 0 &&
                errno == EINTR) errno = 0;
         // Update time of change of index
         ct   = (kXR_int32)time(0); 
         header.itime = ct; 
         if (lseek(fFd, kOfsItime, SEEK_SET) == -1) {
            if (!wasopen) Close();
            return Err(kPFErrSeek,"WriteEntry",
                       "SEEK_SET",(const char *)&fFd);
         }
         while (write(fFd, &header.itime, sizeof(kXR_int32)) < 0 &&
            errno == EINTR) errno = 0;
      }
      // Update time of change in header
      header.ctime = (ct > 0) ? ct : time(0); 
      if (lseek(fFd, kOfsCtime, SEEK_SET) == -1) {
         if (!wasopen) Close();
         return Err(kPFErrSeek,"WriteEntry",
                    "SEEK_SET",(const char *)&fFd);
      }
      while (write(fFd, &header.ctime, sizeof(kXR_int32)) < 0 &&
         errno == EINTR) errno = 0;
      if (!wasopen) Close();
      return nw;
   }

   //
   // If new name, add the entry at the end
   if ((ofs = lseek(fFd, 0, SEEK_END)) == -1) {
      if (!wasopen) Close();
      return Err(kPFErrSeek,"WriteEntry",
                          "SEEK_END",(const char *)&fFd);
   }
   if ((nw = WriteEnt(ofs, ent)) < 0) {
      if (!wasopen) Close();
      return -1;
   }
   //
   // Create new index entry
   XrdSutPFEntInd newind(ent.name, 0, ofs, nw);
   if (WriteInd(ofs+nw, newind) < 0) {
      if (!wasopen) Close();
      return -1;
   }
   //
   // Update previous index entry 
   ind.nxtofs = ofs + nw;
   kXR_int32 wrtofs = lastindofs + sizeof(kXR_int32);
   if (lseek(fFd, wrtofs, SEEK_SET) == -1) {
      if (!wasopen) Close();
      return Err(kPFErrSeek,"WriteEntry",
                          "SEEK_SET",(const char *)&fFd);
   }
   while (write(fFd, &ind.nxtofs, sizeof(kXR_int32)) < 0 &&
      errno == EINTR) errno = 0;

   // Update header
   header.entries += 1;
   header.ctime   = time(0);
   header.itime   = header.ctime;
   if (WriteHeader(header) < 0) {
      if (!wasopen) Close();
      return -1;
   }

   // Close the file
   if (!wasopen) Close();

   return nw;
}

//________________________________________________________________
kXR_int32 XrdSutPFile::UpdateCount(const char *tag, int *cnt,
                                   int step, bool reset)
{
   // Update counter for entry with 'tag', if any.
   // If reset is true, counter is firts reset.
   // The counter is updated by 'step'.
   // Default: no reset, increase by 1.
   // If cnt is defined, fill it with the updated counter.
   // Returns 0 or -1 in case of error

   // Make sure that we got a tag (otherwise we can't do nothing)
   if (!tag)
      return Err(kPFErrBadInputs,"UpdateCount");

   // Make sure we got an open stream
   if (Open(1) < 0)
      return -1;

   // Read the header
   XrdSutPFHeader header;
   if (ReadHeader(header) < 0) {
      Close();
      return -1;
   }

   // Check if the HashTable needs to be updated
   if (fHashTable && header.itime > fHTutime) {
      // Update the table
      if (UpdateHashTable() < 0) {
         Close();
         return -1;
      }
   }
   //
   // Get index entry associated with tag, if any
   XrdSutPFEntInd ind;
   bool found = 0;
   if (fHashTable) {
      kXR_int32 *refofs = fHashTable->Find(tag);
      if (*refofs > 0) {
         // Read it out
         if (ReadInd(*refofs, ind) < 0) {
            Close();
            return -1;
         }
         found = 1;
      }
   } else {
      // Get offset of the first index entry
      kXR_int32 indofs = header.indofs;
      while (indofs > 0) {
         // Read it out
         if (ReadInd(indofs, ind) < 0) {
            Close();
            return -1;
         }
         // Check compatibility
         if (strlen(ind.name) == strlen(tag)) {
            if (!strncmp(ind.name,tag,strlen(tag))) {
               found = 1;
               break;
            }
         }
         // Next index entry
         indofs = ind.nxtofs;
      }
   }
   //
   // Read the entry, if found
   XrdSutPFEntry ent;
   bool changed = 0;
   if (found) {

      // Read entry if active
      if (ind.entofs) {
         if (ReadEnt(ind.entofs, ent) < 0) {
            Close();
            return -1;
         }
         //
         // Reset counter if required
         if (reset && ent.cnt != 0) {
            changed = 1;
            ent.cnt = 0;
         }
         //
         // Update counter
         if (step != 0) {
            changed = 1;
            ent.cnt += step;
         }
         //
         // Update entry in file, if anything changed
         if (changed) {
            ent.mtime = (kXR_int32)time(0);
            if (WriteEnt(ind.entofs, ent) < 0) {
               Close();
               return -1;
            }
         }
         //
         // Fill output
         if (cnt)
            *cnt = ent.cnt;
      }
   }

   // Close the file
   Close();

   return 0;
}

//________________________________________________________________
kXR_int32 XrdSutPFile::ReadEntry(const char *tag,
                                 XrdSutPFEntry &ent, int opt)
{
   // Read entry with tag from file
   // If it does not exist, if opt == 1 search also for wild-card
   // matching entries; if more than 1 return the one that matches
   // the best, base on the number of characters matching.
   // If more wild-card entries have the same level of matching,
   // the first found is returned.
   ent.Reset();

   // Make sure that we got a tag (otherwise we can't do nothing)
   if (!tag)
      return Err(kPFErrBadInputs,"ReadEntry");

   // Make sure we got an open stream
   bool wasopen = 0;
   if (Open(1 &wasopen) < 0)
      return -1;

   // Read the header
   XrdSutPFHeader header;
   if (ReadHeader(header) < 0) {
      if (!wasopen) Close();
      return -1;
   }

   // Check if the HashTable needs to be updated
   if (fHashTable && header.itime > fHTutime) {
      // Update the table
      if (UpdateHashTable() < 0) {
         if (!wasopen) Close();
         return -1;
      }
   }
   //
   // Get index entry associated with tag, if any
   XrdSutPFEntInd ind;
   bool found = 0;
   if (fHashTable) {
      kXR_int32 *reftmp = fHashTable->Find(tag);
      kXR_int32 refofs = reftmp ? *reftmp : -1;
      if (refofs > 0) {
         // Read it out
         if (ReadInd(refofs, ind) < 0) {
            if (!wasopen) Close();
            return -1;
         }
         found = 1;
      }
   } else {
      // Get offset of the first index entry
      kXR_int32 indofs = header.indofs;
      while (indofs > 0) {
         // Read it out
         if (ReadInd(indofs, ind) < 0) {
            if (!wasopen) Close();
            return -1;
         }
         // Check compatibility
         if (strlen(ind.name) == strlen(tag)) {
            if (!strncmp(ind.name,tag,strlen(tag))) {
               found = 1;
               break;
            }
         }
         // Next index entry
         indofs = ind.nxtofs;
      }
   }
   //
   // If not found and requested, try also wild-cards
   if (!found && opt == 1) {
      //
      // If > 1 we will keep the best matching, i.e. the one
      // matching most of the chars in tag
      kXR_int32 refofs = -1;
      kXR_int32 nmmax = 0;
      kXR_int32 iofs = header.indofs;
      XrdOucString stag(tag);
      while (iofs) {
         //
         // Read it out
         if (ReadInd(iofs, ind) < 0) {
            if (!wasopen) Close();
            return -1;
         }
         //      
         // Check compatibility, if active
         if (ind.entofs > 0) {
            int match = stag.matches(ind.name);
            if (match > nmmax && ind.entofs > 0) {
               nmmax = match;
               refofs = iofs;
            }
         }
         //
         // Next index entry
         iofs = ind.nxtofs;
      }
      //
      // Read it out
      if (refofs > 0) {
         if (ReadInd(refofs, ind) < 0) {
            if (!wasopen) Close();
            return -1;
         }
         found = 1;
      }
   }

   // Read the entry, if found
   kXR_int32 nr = 0;
   if (found) {

      // Read entry if active
      if (ind.entofs) {
         if ((nr = ReadEnt(ind.entofs, ent)) < 0) {
            if (!wasopen) Close();
            return -1;
         }
         // Fill the name
         ent.SetName(ind.name);
      }
   }

   // Close the file
   if (!wasopen) Close();

   return nr;
}

//________________________________________________________________
kXR_int32 XrdSutPFile::ReadEntry(kXR_int32 ofs, XrdSutPFEntry &ent)
{
   // Read entry at ofs from file

   // Make sure that ofs makes sense
   if (ofs <= 0)
      return Err(kPFErrBadInputs,"ReadEntry");

   // Make sure we got an open stream
   bool wasopen = 0;
   if (Open(1, &wasopen) < 0)
      return -1;

   kXR_int32 nr = 0;

   // Read index entry out
   XrdSutPFEntInd ind;
   if (ReadInd(ofs, ind) < 0) {
      if (!wasopen) Close();
      return -1;
   }

   // Read entry
   if ((nr = ReadEnt(ind.entofs, ent)) < 0) {
      if (!wasopen) Close();
      return -1;
   }

   // Fill the name
   ent.SetName(ind.name);

   // Close the file
   if (!wasopen) Close();

   return nr;
}

//________________________________________________________________
kXR_int32 XrdSutPFile::RemoveEntry(const char *tag)
{
   // Remove entry with tag from file
   // The entry is set inactive, so that it is hidden and it will
   // be physically removed at next Trim 

   // Make sure that we got a tag (otherwise we can't do nothing)
   if (!tag || !strlen(tag))
      return Err(kPFErrBadInputs,"RemoveEntry");

   // Make sure we got an open stream
   if (Open(1) < 0)
      return -1;

   // Read the header
   XrdSutPFHeader header;
   if (ReadHeader(header) < 0) {
      Close();
      return -1;
   }

   // Check if the HashTable needs to be updated
   if (fHashTable && header.itime > fHTutime) {
      // Update the table
      if (UpdateHashTable() < 0) {
         Close();
         return -1;
      }
   }

   // Get offset of the index entry associated with tag, if any
   XrdSutPFEntInd ind;
   bool found = 0;
   kXR_int32 indofs = -1;
   if (fHashTable) {
      kXR_int32 *indtmp = fHashTable->Find(tag);
      indofs = indtmp ? *indtmp : indofs;
      if (indofs > 0) {
         // Read it out
         if (ReadInd(indofs, ind) < 0) {
            Close();
            return -1;
         }
         found = 1;
      }
   } else {
      // Get offset of the first index entry
      indofs = header.indofs;
      while (indofs > 0) {
         // Read it out
         if (ReadInd(indofs, ind) < 0) {
            Close();
            return -1;
         }
         // Check compatibility
         if (strlen(ind.name) == strlen(tag)) {
            if (!strncmp(ind.name,tag,strlen(tag))) {
               found = 1;
               break;
            }
         }
         // Next index entry
         indofs = ind.nxtofs;
      }
   }
   //
   // Get entry now, if index found
   if (found) {
      // Reset entry area
      short status = kPFE_inactive;
      if (lseek(fFd, ind.entofs, SEEK_SET) == -1) {
         Close();
         return Err(kPFErrSeek,"RemoveEntry",
                               "SEEK_SET",(const char *)&fFd);
      }
      while (write(fFd, &status, sizeof(short)) < 0 &&
             errno == EINTR) errno = 0;
      // Reset entry area
      if (Reset(ind.entofs + sizeof(short), ind.entsiz - sizeof(short)) < 0) {
         Close();
         return -1;
      }
      // Set entofs to null
      ind.entofs = 0;
      if (WriteInd(indofs, ind) < 0) {
         Close();
         return -1;
      }
      // Count as unused bytes
      header.jnksiz += ind.entsiz;
      // Decrease number of entries
      header.entries--;
      // Update times
      header.ctime = (kXR_int32)time(0);
      header.itime = header.ctime;
      // Update header
      if (WriteHeader(header) < 0) {
         Close();
         return -1;
      }

      // Ok: close the file and return
      Close();
      return 0;
   }

   // Close the file
   Close();
   // entry non-existing
   return -1;
}

//________________________________________________________________
kXR_int32 XrdSutPFile::RemoveEntry(kXR_int32 ofs)
{
   // Remove entry at entry index offset ofs from file
   // The entry is set inactive, so that it is hidden and it will
   // be physically removed at next Trim 

   // Make sure that we got a tag (otherwise we can't do nothing)
   if (ofs <= 0)
      return Err(kPFErrBadInputs,"RemoveEntry");

   // Make sure we got an open stream
   if (Open(1) < 0)
      return -1;

   // Read the header
   XrdSutPFHeader header;
   if (ReadHeader(header) < 0) {
      Close();
      return -1;
   }

   // Check if the HashTable needs to be updated
   if (header.itime > fHTutime) {
      // Update the table
      if (UpdateHashTable() < 0) {
         Close();
         return -1;
      }
   }
   //
   // Read it out
   XrdSutPFEntInd ind;
   if (ReadInd(ofs, ind) < 0) {
      Close();
      return -1;
   }
   //
   // Reset entry area
   short status = kPFE_inactive;
   if (lseek(fFd, ind.entofs, SEEK_SET) == -1) {
      Close();
      return Err(kPFErrSeek,"RemoveEntry",
                          "SEEK_SET",(const char *)&fFd);
   }
   while (write(fFd, &status, sizeof(short)) < 0 &&
          errno == EINTR) errno = 0;
   // Reset entry area
   if (Reset(ind.entofs + sizeof(short), ind.entsiz - sizeof(short)) < 0) {
      Close();
      return -1;
   }
   // Set entofs to null
   ind.entofs = 0;
   if (WriteInd(ofs, ind) < 0) {
      Close();
      return -1;
   }
   // Count as unused bytes
   header.jnksiz += ind.entsiz;
   // Decrease number of entries
   header.entries--;
   // Update times
   header.ctime = (kXR_int32)time(0);
   header.itime = header.ctime;
   // Update header
   if (WriteHeader(header) < 0) {
      Close();
      return -1;
   }
   //
   // Ok: close the file and return
   Close();
   return 0;
}

//_________________________________________________________________
kXR_int32 XrdSutPFile::Reset(kXR_int32 ofs, kXR_int32 siz)
{
   // Reset size bytes starting at ofs in the open stream 

   //
   // Set the offset
   if (lseek(fFd, ofs, SEEK_SET) == -1)
      return Err(kPFErrSeek,"Reset",
                          "SEEK_SET",(const char *)&fFd);

   kXR_int32 nrs = 0;
   // Now write the buffer to the stream
   while (nrs < siz) {
      char c = 0;
      while (write(fFd, &c, 1) < 0 && errno == EINTR)
         errno = 0;
      nrs++;
   }

   return nrs;
}


//__________________________________________________________________
kXR_int32 XrdSutPFile::WriteInd(kXR_int32 ofs, XrdSutPFEntInd ind)
{
   // Write entry index to open stream fFd

   // Make sure we got an open stream
   if (fFd < 0)
      return Err(kPFErrFileNotOpen,"WriteInd");
   //
   // Set the offset
   if (lseek(fFd, ofs, SEEK_SET) == -1)
      return Err(kPFErrSeek,"WriteInd",
                          "SEEK_SET",(const char *)&fFd);
   //
   // Build output buffer
   //
   // Get total lenght needed
   kXR_int32 ltot = ind.Length();
   //
   // Allocate the buffer
   char *bout = new char[ltot];
   if (!bout)
      return Err(kPFErrOutOfMemory,"WriteInd");
   //
   // Fill the buffer
   kXR_int32 lp = 0;
   // Name length
   kXR_int32 lnam = strlen(ind.name);
   memcpy(bout+lp,&lnam,sizeof(kXR_int32));
   lp += sizeof(kXR_int32);
   // Offset of next index entry
   memcpy(bout+lp,&ind.nxtofs,sizeof(kXR_int32));
   lp += sizeof(kXR_int32);
   // Offset of entry
   memcpy(bout+lp,&ind.entofs,sizeof(kXR_int32));
   lp += sizeof(kXR_int32);
   // Size allocated for entry
   memcpy(bout+lp,&ind.entsiz,sizeof(kXR_int32));
   lp += sizeof(kXR_int32);
   // name
   memcpy(bout+lp,ind.name,lnam);
   lp += lnam;
   // Check length
   if (lp != ltot) {
      if (bout) delete[] bout;
      return Err(kPFErrLenMismatch,"WriteInd",
                 (const char *)&lp, (const char *)&ltot);
   }

   kXR_int32 nw = 0;
   // Now write the buffer to the stream
   while ((nw = write(fFd, bout, ltot)) < 0 && errno == EINTR)
      errno = 0;
 
   return nw;
}

//__________________________________________________________________
kXR_int32 XrdSutPFile::WriteEnt(kXR_int32 ofs, XrdSutPFEntry ent)
{
   // Write ent to stream out

   // Make sure we got an open stream
   if (fFd < 0)
      return Err(kPFErrFileNotOpen,"WriteEnt");
   //
   // Set the offset
   if (lseek(fFd, ofs, SEEK_SET) == -1)
      return Err(kPFErrSeek,"WriteEnt",
                          "SEEK_SET",(const char *)&fFd);
   //
   // Build output buffer
   //
   // Get total lenght needed
   kXR_int32 ltot = ent.Length();
   //
   // Allocate the buffer
   char *bout = new char[ltot];
   if (!bout)
      return Err(kPFErrOutOfMemory,"WriteEnt");
   //
   // Fill the buffer
   kXR_int32 lp = 0;
   // status
   memcpy(bout+lp,&ent.status,sizeof(short));
   lp += sizeof(short);
   // count
   memcpy(bout+lp,&ent.cnt,sizeof(short));
   lp += sizeof(short);
   // time of modification / creation
   memcpy(bout+lp,&ent.mtime,sizeof(kXR_int32));
   lp += sizeof(kXR_int32);
   // length of first buffer
   memcpy(bout+lp,&ent.buf1.len,sizeof(kXR_int32));
   lp += sizeof(kXR_int32);
   // length of second buffer
   memcpy(bout+lp,&ent.buf2.len,sizeof(kXR_int32));
   lp += sizeof(kXR_int32);
   // length of third buffer
   memcpy(bout+lp,&ent.buf3.len,sizeof(kXR_int32));
   lp += sizeof(kXR_int32);
   // length of fourth buffer
   memcpy(bout+lp,&ent.buf4.len,sizeof(kXR_int32));
   lp += sizeof(kXR_int32);
   if (ent.buf1.len > 0) {
      // first buffer
      memcpy(bout+lp,ent.buf1.buf,ent.buf1.len);
      lp += ent.buf1.len;
   }
   if (ent.buf2.len > 0) {
      // second buffer
      memcpy(bout+lp,ent.buf2.buf,ent.buf2.len);
      lp += ent.buf2.len;
   }
   if (ent.buf3.len > 0) {
      // third buffer
      memcpy(bout+lp,ent.buf3.buf,ent.buf3.len);
      lp += ent.buf3.len;
   }
   if (ent.buf4.len > 0) {
      // third buffer
      memcpy(bout+lp,ent.buf4.buf,ent.buf4.len);
      lp += ent.buf4.len;
   }
   // Check length
   if (lp != ltot) {
      if (bout) delete[] bout;
      return Err(kPFErrLenMismatch,"WriteEnt",
                 (const char *)&lp, (const char *)&ltot);
   }

   kXR_int32 nw = 0;
   // Now write the buffer to the stream
   while ((nw = write(fFd, bout, ltot)) < 0 && errno == EINTR)
      errno = 0;
 
   return nw;
}

//__________________________________________________________________
kXR_int32 XrdSutPFile::ReadHeader(XrdSutPFHeader &hd)
{
   // Read header from beginning of stream 

   //
   // Make sure that we got an open file description
   if (fFd < 0)
      return Err(kPFErrFileNotOpen,"ReadHeader");
   //
   // Set the offset
   if (lseek(fFd, 0, SEEK_SET) == -1)
      return Err(kPFErrSeek,"ReadHeader",
                          "SEEK_SET",(const char *)&fFd);

   kXR_int32 nr = 0, nrdt = 0;
   //
   // Now read the information step by step:
   // the file ID ...
   if ((nr = read(fFd,hd.fileID,kFileIDSize)) != kFileIDSize)
      return Err(kPFErrRead,"ReadHeader",(const char *)&fFd);
   hd.fileID[kFileIDSize-1] = 0;
   nrdt += nr;
   // the version ...
   if ((nr = read(fFd,&hd.version,sizeof(kXR_int32))) != sizeof(kXR_int32))
      return Err(kPFErrRead,"ReadHeader",(const char *)&fFd);
   nrdt += nr;
   // the time of last change ...
   if ((nr = read(fFd,&hd.ctime,sizeof(kXR_int32))) != sizeof(kXR_int32))
      return Err(kPFErrRead,"ReadHeader",(const char *)&fFd);
   nrdt += nr;
   // the time of last index change ...
   if ((nr = read(fFd,&hd.itime,sizeof(kXR_int32))) != sizeof(kXR_int32))
      return Err(kPFErrRead,"ReadHeader",(const char *)&fFd);
   nrdt += nr;
   // the number of entries ...
   if ((nr = read(fFd,&hd.entries,sizeof(kXR_int32))) != sizeof(kXR_int32))
      return Err(kPFErrRead,"ReadHeader",(const char *)&fFd);
   nrdt += nr;
   // the offset of first index entry ...
   if ((nr = read(fFd,&hd.indofs,sizeof(kXR_int32))) != sizeof(kXR_int32))
      return Err(kPFErrRead,"ReadHeader",(const char *)&fFd);
   nrdt += nr;
   // the number of unused bytes ...
   if ((nr = read(fFd,&hd.jnksiz,sizeof(kXR_int32))) != sizeof(kXR_int32))
      return Err(kPFErrRead,"ReadHeader",(const char *)&fFd);
   nrdt += nr;

   return nrdt;
}

//_____________________________________________________________________
kXR_int32 XrdSutPFile::ReadInd(kXR_int32 ofs, XrdSutPFEntInd &ind)
{
   // Read entry index from offset ofs of open stream fFd

   //
   // Make sure that we got an open file description
   if (fFd < 0)
      return Err(kPFErrFileNotOpen,"ReadInd");
   //
   // Set the offset
   if (lseek(fFd, ofs, SEEK_SET) == -1)
      return Err(kPFErrSeek,"ReadInd",
                          "SEEK_SET",(const char *)&fFd);

   kXR_int32 nr = 0, nrdt = 0;
   //
   // Now read the information step by step:
   // the length of the name ...
   kXR_int32 lnam = 0;
   if ((nr = read(fFd,&lnam,sizeof(kXR_int32))) != sizeof(kXR_int32))
      return Err(kPFErrRead,"ReadInd",(const char *)&fFd);
   nrdt += nr;
   // the offset of next entry index ...
   if ((nr = read(fFd,&ind.nxtofs,sizeof(kXR_int32))) != sizeof(kXR_int32))
      return Err(kPFErrRead,"ReadInd",(const char *)&fFd);
   nrdt += nr;
   // the offset of the entry ...
   if ((nr = read(fFd,&ind.entofs,sizeof(kXR_int32))) != sizeof(kXR_int32))
      return Err(kPFErrRead,"ReadInd",(const char *)&fFd);
   nrdt += nr;
   // the size allocated for the entry ...
   if ((nr = read(fFd,&ind.entsiz,sizeof(kXR_int32))) != sizeof(kXR_int32))
      return Err(kPFErrRead,"ReadInd",(const char *)&fFd);
   nrdt += nr;
   // the name ... cleanup first
   if (ind.name) {
      delete[] ind.name;
      ind.name = 0;
   }
   if (lnam) {
      ind.name = new char[lnam+1];
      if (ind.name) {
         if ((nr = read(fFd,ind.name,lnam)) != lnam)
            return Err(kPFErrRead,"ReadInd",(const char *)&fFd);
         ind.name[lnam] = 0; // null-terminated
         nrdt += nr;
      } else 
         return Err(kPFErrOutOfMemory,"ReadInd");
   }

   return nrdt;
}

//____________________________________________________________________
kXR_int32 XrdSutPFile::ReadEnt(kXR_int32 ofs, XrdSutPFEntry &ent)
{
   // Read ent from current position at stream 

   //
   // Make sure that we got an open file description
   if (fFd < 0)
      return Err(kPFErrFileNotOpen,"ReadEnt");
   //
   // Set the offset
   if (lseek(fFd, ofs, SEEK_SET) == -1)
      return Err(kPFErrSeek,"ReadEnt",
                          "SEEK_SET",(const char *)&fFd);

   kXR_int32 nr = 0, nrdt = 0;
   //
   // Now read the information step by step:
   // the status ...
   if ((nr = read(fFd,&ent.status,sizeof(short))) != sizeof(short))
      return Err(kPFErrRead,"ReadEnt",(const char *)&fFd);
   nrdt += nr;
   // the count var ...
   if ((nr = read(fFd,&ent.cnt,sizeof(short))) != sizeof(short))
      return Err(kPFErrRead,"ReadEnt",(const char *)&fFd);
   nrdt += nr;
   // the the time of modification / creation ...
   if ((nr = read(fFd,&ent.mtime,sizeof(kXR_int32))) != sizeof(kXR_int32))
      return Err(kPFErrRead,"ReadEnt",(const char *)&fFd);
   nrdt += nr;
   // the length of the first buffer ...
   if ((nr = read(fFd,&ent.buf1.len,sizeof(kXR_int32))) != sizeof(kXR_int32))
      return Err(kPFErrRead,"ReadEnt",(const char *)&fFd);
   nrdt += nr;
   // the length of the second buffer ...
   if ((nr = read(fFd,&ent.buf2.len,sizeof(kXR_int32))) != sizeof(kXR_int32))
      return Err(kPFErrRead,"ReadEnt",(const char *)&fFd);
   nrdt += nr;
   // the length of the third buffer ...
   if ((nr = read(fFd,&ent.buf3.len,sizeof(kXR_int32))) != sizeof(kXR_int32))
      return Err(kPFErrRead,"ReadEnt",(const char *)&fFd);
   nrdt += nr;
   // the length of the fourth buffer ...
   if ((nr = read(fFd,&ent.buf4.len,sizeof(kXR_int32))) != sizeof(kXR_int32))
      return Err(kPFErrRead,"ReadEnt",(const char *)&fFd);
   nrdt += nr;
   // Allocate space for the first buffer and read it (if any) ...
   if (ent.buf1.len) {
      ent.buf1.buf = new char[ent.buf1.len];
      if (!ent.buf1.buf)
         return Err(kPFErrOutOfMemory,"ReadEnt");
      if ((nr = read(fFd,ent.buf1.buf,ent.buf1.len)) != ent.buf1.len)
         return Err(kPFErrRead,"ReadEnt",(const char *)&fFd);
      nrdt += nr;
   }
   // Allocate space for the second buffer and read it (if any) ...
   if (ent.buf2.len) {
      ent.buf2.buf = new char[ent.buf2.len];
      if (!ent.buf2.buf)
         return Err(kPFErrOutOfMemory,"ReadEnt");
      if ((nr = read(fFd,ent.buf2.buf,ent.buf2.len)) != ent.buf2.len)
         return Err(kPFErrRead,"ReadEnt",(const char *)&fFd);
      nrdt += nr;
   }
   // Allocate space for the third buffer and read it (if any) ...
   if (ent.buf3.len) {
      ent.buf3.buf = new char[ent.buf3.len];
      if (!ent.buf3.buf)
         return Err(kPFErrOutOfMemory,"ReadEnt");
      if ((nr = read(fFd,ent.buf3.buf,ent.buf3.len)) != ent.buf3.len)
         return Err(kPFErrRead,"ReadEnt",(const char *)&fFd);
      nrdt += nr;
   }
   // Allocate space for the fourth buffer and read it (if any) ...
   if (ent.buf4.len) {
      ent.buf4.buf = new char[ent.buf4.len];
      if (!ent.buf4.buf)
         return Err(kPFErrOutOfMemory,"ReadEnt");
      if ((nr = read(fFd,ent.buf4.buf,ent.buf4.len)) != ent.buf4.len)
         return Err(kPFErrRead,"ReadEnt",(const char *)&fFd);
      nrdt += nr;
   }

   return nrdt;
}

//________________________________________________________________
kXR_int32 XrdSutPFile::Browse(void *oout)
{
   // Display the content of the file

   // Make sure we got an open stream
   if (Open(1) < 0)
      return -1;

   // Read header
   XrdSutPFHeader hdr;
   if (ReadHeader(hdr) < 0) {
      Close();
      return -1;
   }

   // Time strings
   struct tm tst;
   char sctime[256] = {0};
   time_t ttmp = hdr.ctime;
   localtime_r(&ttmp,&tst);
   asctime_r(&tst,sctime);
   sctime[strlen(sctime)-1] = 0;
   char sitime[256] = {0};
   ttmp = hdr.itime;
   localtime_r(&ttmp,&tst);
   asctime_r(&tst,sitime);
   sitime[strlen(sitime)-1] = 0;

   // Default is stdout
   FILE *out = oout ? (FILE *)oout : stdout;

   fprintf(out,"//-----------------------------------------------------"
                                              "--------------------//\n");
   fprintf(out,"//\n");
   fprintf(out,"//  File:         %s\n",name);
   fprintf(out,"//  ID:           %s\n",hdr.fileID);
   fprintf(out,"//  Version:      %d\n",hdr.version);
   fprintf(out,"//  Last change : %s (%d sec)\n",sctime,hdr.ctime);
   fprintf(out,"//  Index change: %s (%d sec)\n",sitime,hdr.itime);
   fprintf(out,"//\n");
   fprintf(out,"//  Number of Entries: %d\n",hdr.entries);
   fprintf(out,"//  Bytes unreachable: %d\n",hdr.jnksiz);
   fprintf(out,"//\n");

   if (hdr.entries > 0) {

      // Special entries first, if any
      kXR_int32 ns = SearchSpecialEntries();
      if (ns > 0) {
         // Allocate space for offsets
         kXR_int32 *sofs = new kXR_int32[ns];
         if (sofs) {
            // Get offsets
            ns = SearchSpecialEntries(sofs,ns);
            fprintf(out,"//  Special entries (%d):\n",ns);
            int i = 0;
            for (; i<ns; i++) {
               
               // Read entry index at ofs
               XrdSutPFEntInd ind;
               if (ReadInd(sofs[i], ind) < 0) {
                  Close();
                  return -1;
               }

               if (ind.entofs) {
                  // Read entry
                  XrdSutPFEntry ent;
                  if (ReadEnt(ind.entofs, ent) < 0) {
                     Close();
                     return -1;
                  }
                  char smt[20] = {0};
                  XrdSutTimeString(ent.mtime,smt);
                  char buf[2048] = {0};
                  memset(buf,0,2048);
                  sprintf(buf,"// #%d mod:%s",i+1,smt);
                  sprintf(buf,"%s name:%s",buf,ind.name);
                  fprintf(out,"%s\n",buf);
                  sprintf(buf,"//    buf");
                  if (ent.cnt == 1) {
                     if (ent.buf1.len && ent.buf1.buf)
                        sprintf(buf,"%s: %.*s",buf,ent.buf1.len,ent.buf1.buf);
                     if (ent.buf2.len && ent.buf2.buf)
                        sprintf(buf,"%s: %.*s",buf,ent.buf2.len,ent.buf2.buf);
                     if (ent.buf3.len && ent.buf3.buf)
                        sprintf(buf,"%s: %.*s",buf,ent.buf3.len,ent.buf3.buf);
                     if (ent.buf4.len && ent.buf4.buf)
                        sprintf(buf,"%s: %.*s",buf,ent.buf4.len,ent.buf4.buf);
                  } else {
                     sprintf(buf,"%s:%d:%d:%d:%d",buf,
                                 ent.buf1.len,ent.buf2.len,ent.buf3.len,
                                 ent.buf4.len);
                     sprintf(buf,"%s (protected)",buf);
                  }
                  fprintf(out,"%s\n",buf);
               }
            }
            fprintf(out,"//\n");
            delete[] sofs;
         }
      }

      if (hdr.entries > ns)
         fprintf(out,"//  Normal entries (%d):\n",hdr.entries-ns);

      kXR_int32 nn = 0;
      kXR_int32 nxtofs = hdr.indofs;
      while (nxtofs) {

         // Read entry index at ofs
         XrdSutPFEntInd ind;
         if (ReadInd(nxtofs, ind) < 0) {
            Close();
            return -3;
         }

         if (ind.entofs) {
            // Read entry
            XrdSutPFEntry ent;
            if (ReadEnt(ind.entofs, ent) < 0) {
               Close();
               return -4;
            }
            if (ent.status != kPFE_special) {
               char smt[20] = {0};
               XrdSutTimeString(ent.mtime,smt);
               
               nn++;
               fprintf(out,
                   "// #:%d  st:%d cn:%d  buf:%d,%d,%d,%d mod:%s name:%s\n",
                   nn,ent.status,ent.cnt,ent.buf1.len,ent.buf2.len,ent.buf3.len,
                   ent.buf4.len,smt,ind.name);
            }
         }

         // Read next
         nxtofs = ind.nxtofs;
      }
      fprintf(out,"//\n");
   }
   fprintf(out,"//-----------------------------------------------------"
                                              "--------------------//\n");

   // Close the file
   Close();

   return 0;
}

//________________________________________________________________
kXR_int32 XrdSutPFile::Trim(const char *fbak)
{
   // Trim away unreachable entries from the file
   // Previous content is save in a file name fbak, the default
   // being 'name'.bak
   EPNAME("PFile::Trim");

   // Retrieve header, first, to check if there is anything to trim
   XrdSutPFHeader header;
   if (RetrieveHeader(header) < 0)
      return -1;
   if (header.jnksiz <= 0) {
      DEBUG("nothing to trim - return ");
      return -1;
   }

   // Get name of backup file
   char *nbak = (char *)fbak;
   if (!nbak) {
      // Use default
      nbak = new char[strlen(name)+5];
      if (!nbak)
         return Err(kPFErrOutOfMemory,"Trim");
      sprintf(nbak,"%s.bak",name);
      DEBUG("backup file: "<<nbak);
   }

   // Move file
   if (rename(name,nbak) == -1)
      return Err(kPFErrFileRename,"Trim",name,nbak);

   // Create new file
   int fdnew = Open(1);
   if (fdnew < 0)
      return -1;

   // Open backup file
   int fdbck = Open(1,0,nbak);
   if (fdbck < 0) {
      Close();
      return -1;
   }

   // Read the header from backup file
   fFd = fdbck;
   if (ReadHeader(header) < 0) {
      Close(fdnew); Close(fdbck);
      return -1;
   }

   // Copy it to new file
   fFd = fdnew;
   if (WriteHeader(header) < 0) {
      Close(fdnew); Close(fdbck);
      return -1;
   }
   kXR_int32 wrofs = lseek(fdnew, 0, SEEK_CUR);
   if (wrofs == -1) {
      Close(fdnew); Close(fdbck);
      return Err(kPFErrSeek,"Trim",
                          "SEEK_CUR",(const char *)&fdnew);
   }

   // Read active entries now and save them to new file
   bool firstind = 1;
   XrdSutPFEntInd ind, indlast;
   XrdSutPFEntry ent;

   kXR_int32 nxtofs = header.indofs;
   kXR_int32 lastofs = nxtofs;

   while (nxtofs) {

      // Read index entry
      fFd = fdbck;
      if (ReadInd(nxtofs,ind) < 0) {
         Close(fdnew); Close(fdbck);
         return -1;
      }

      // Get Next index entry before updating index entry
      nxtofs = ind.nxtofs;
      
      // Read entry, if active
      if (ind.entofs > 0) {
         fFd = fdbck;
         if (ReadEnt(ind.entofs,ent) < 0) {
            Close(fdnew); Close(fdbck);
            return -1;
         }
         // Update index entry
         ind.entofs = wrofs;

         // Write active entry
         fFd = fdnew;
         if (WriteEnt(wrofs,ent) < 0) {
            Close(fdnew); Close(fdbck);
            return -1;
         }

         // Update write offset
         if ((wrofs = lseek(fdnew, 0, SEEK_CUR)) == -1) {
            Close(fdnew); Close(fdbck);
            return Err(kPFErrSeek,"Trim",
                       "SEEK_CUR",(const char *)&fdnew);
         }

         if (firstind) {
            // Update header
            header.indofs = wrofs;
            firstind = 0;
         } else {
            // Update previous index entry
            indlast.nxtofs = wrofs;
            fFd = fdnew;
            if (WriteInd(lastofs,indlast) < 0) {
               Close(fdnew); Close(fdbck);
               return -1;
            }
         }

         // Save this index for later updates
         indlast = ind;
         lastofs = wrofs;

         // Last index entry, for now
         ind.nxtofs = 0;

         // Write active index entry
         fFd = fdnew;
         if (WriteInd(wrofs,ind) < 0) {
            Close(fdnew); Close(fdbck);
            return -1;
         }

         // Update write offset
         if ((wrofs = lseek(fdnew, 0, SEEK_CUR)) == -1) {
            Close(fdnew); Close(fdbck);
            return Err(kPFErrSeek,"Trim",
                       "SEEK_CUR",(const char *)&fdnew);
         }
      }
   }

   // Close backup file
   Close(fdbck);
   fFd = fdnew;

   // Update header
   header.ctime = (kXR_int32)time(0);
   header.itime = header.ctime;
   header.jnksiz = 0;

   // Copy it to new file
   if (WriteHeader(header) < 0) {
      Close();;
      return -1;
   }

   // Close the file
   Close();

   return 0;
}

//________________________________________________________________
kXR_int32 XrdSutPFile::UpdateHashTable(bool force)
{
   // Update hash table reflecting the index of the file
   // If force is .true. the table is recreated even if no recent 
   // change in the index has occured.
   // Returns the number of entries in the table.

   // The file must be open
   if (fFd < 0)
      return Err(kPFErrFileNotOpen,"UpdateHashTable");

   // Read the header
   XrdSutPFHeader header;
   if (ReadHeader(header) < 0)
      return -1;

   // If no recent changes and no force option, return
   if (!force && header.itime < fHTutime)
      return 0;

   // Clean up the table or create it
   if (fHashTable)
      fHashTable->Purge();
   else
      fHashTable = new XrdOucHash<kXR_int32>;
   // Make sure we have it
   if (!fHashTable)
      return Err(kPFErrOutOfMemory,"UpdateHashTable");

   // Read entries
   kXR_int32 ne = 0;
   if (header.entries > 0) {
      XrdSutPFEntInd ind;
      kXR_int32 nxtofs = header.indofs;
      while (nxtofs > 0) {
         if (ReadInd(nxtofs, ind) < 0)
            return -1;
         ne++;
         // Fill the table 
         kXR_int32 *key = new kXR_int32(nxtofs);
         fHashTable->Add(ind.name,key);
         // Go to next
         nxtofs = ind.nxtofs;
      }
   }

   // Update the time stamp
   fHTutime = (kXR_int32)time(0);

   return ne;
}

//________________________________________________________________
kXR_int32 XrdSutPFile::RemoveEntries(const char *tag, char opt)
{
   // Remove entries whose tag is compatible with 'tag', according
   // to compatibility option 'opt'.
   // For opt = 0 tags starting with 'tag'
   // for opt = 1 tags containing the wild card '*' are matched.
   // Return number of entries removed
   EPNAME("PFile::RemoveEntries");

   //
   // Get number of entries related
   int nm = SearchEntries(tag,opt);
   if (nm) {
      DEBUG("found "<<nm<<" entries for tag '"<<tag<<"'");
      //
      // Book vector for offsets
      int *ofs = new int[nm];
      //
      // Get number of entries related
      SearchEntries(tag,0,ofs,nm);
      //
      // Read entries now
      int i = 0;
      for (; i < nm ; i++) {
         if (RemoveEntry(ofs[i]) == 0) {
            DEBUG("entry for tag '"<<tag<<"' removed from file");
         } else {
            DEBUG("entry for tag '"<<tag<<"' not found in file");
         }
      }
   } else {
      DEBUG("no entry for tag '"<<tag<<"' found in file: "<<Name());
   }
   // We are done
   return nm;
}

//________________________________________________________________
kXR_int32 XrdSutPFile::SearchEntries(const char *tag, char opt,
                                     kXR_int32 *ofs, kXR_int32 nofs)
{
   // Get offsets of the first nofs entries whose tag is compatible
   // with 'tag', according to compatibility option 'opt'.
   // For opt = 0 tags starting with 'tag' are searched for;
   // For opt = 1 tags containing the wild card '*' matching tag
   // are searched for.
   // For opt = 2 tags matching tag are searched for; tag may contain
   // the wild card '*'.
   // The caller is responsible for memory pointed by 'ofs'.
   // Return number of entries found (<= nofs).
   // If ofs = 0, return total number of entries matching the
   // condition.

   // Make sure that we got a tag
   if (!tag)
      return Err(kPFErrBadInputs,"SearchEntries");

   // Make sure we got an open stream
   bool wasopen = 0;
   if (Open(1,&wasopen) < 0)
      return -1;

   // Read the header
   XrdSutPFHeader header;
   if (ReadHeader(header) < 0) {
      if (!wasopen) Close();
      return -1;
   }

   // Get offset of the first index entry
   kXR_int32 indofs = header.indofs;

   // Scan entries
   kXR_int32 no = 0;
   XrdOucString smatch;
   if (opt == 1)
      smatch.assign(tag, 0);
   while (indofs) {

      // Read it out
      XrdSutPFEntInd ind;
      if (ReadInd(indofs, ind) < 0) {
         if (!wasopen) Close();
         return -1;
      }

      // Check compatibility
      int match = 0;
      if (opt == 0) {
         if (!strncmp(ind.name,tag,strlen(tag)))
            match = 1;
      } else if (opt == 1) {
         match = smatch.matches(ind.name);
      } else if (opt == 2) {
         smatch.assign(ind.name, 0);
         match = smatch.matches(tag);
      }

      if (match > 0 && ind.entofs > 0) {
         no++;
         if (ofs) {
            ofs[no-1] = indofs;
            if (no == nofs) {
               // We are done
               break;
            }
         }
      }

      // Next index entry
      indofs = ind.nxtofs;
   }

   // Close the file
   if (!wasopen) Close();

   return no;
}

//________________________________________________________________
kXR_int32 XrdSutPFile::SearchSpecialEntries(kXR_int32 *ofs,
                                            kXR_int32 nofs)
{
   // Get offsets of the first nofs entries with status
   // kPFE_special.
   // The caller is responsible for memory pointed by 'ofs'.
   // Return number of entries found (<= nofs).
   // If ofs = 0, return total number of special entries.

   // Make sure we got an open stream
   bool wasopen = 0;
   if (Open(1,&wasopen) < 0)
      return -1;

   // Read the header
   XrdSutPFHeader header;
   if (ReadHeader(header) < 0) {
      if (!wasopen) Close();
      return -1;
   }

   // Get offset of the first index entry
   kXR_int32 indofs = header.indofs;

   // Scan entries
   kXR_int32 no = 0;
   while (indofs) {

      // Read index
      XrdSutPFEntInd ind;
      if (ReadInd(indofs, ind) < 0) {
         if (!wasopen) Close();
         return -1;
      }

      // If active ...
      if (ind.entofs > 0) {

         // Read entry out
         XrdSutPFEntry ent;
         if (ReadEnt(ind.entofs, ent) < 0) {
            if (!wasopen) Close();
            return -1;
         }
         // If special ...
         if (ent.status == kPFE_special) {
            // Record the offset ...
            no++;
            if (ofs) {
               ofs[no-1] = indofs;
               if (no == nofs) {
                  // We are done
                  break;
               }
            }
         }
      }

      // Next index entry
      indofs = ind.nxtofs;
   }

   // Close the file
   if (!wasopen) Close();

   return no;
}

//________________________________________________________________
kXR_int32 XrdSutPFile::Err(kXR_int32 code, const char *loc,
                           const char *em1, const char *em2)
{
   // Save code and, if requested, format and print an error
   // message
   EPNAME("PFile::Err");

   char buf[XrdSutMAXBUF];
   int fd = 0, lp = 0, lt = 0;

   // Save code for later use
   fError = code;

   // Build string following the error code
   char *errbuf = strerror(errno);
   switch (code) {
      case kPFErrBadInputs:
         snprintf(buf,XrdSutMAXBUF,
                  "XrdSutPFile::%s: bad input arguments",loc);
         break;
      case kPFErrFileAlreadyOpen:
         snprintf(buf,XrdSutMAXBUF,
                  "XrdSutPFile::%s: file already open"
                  " in incompatible mode",loc);
         break;
      case kPFErrNoFile:
         snprintf(buf,XrdSutMAXBUF,
                  "XrdSutPFile::%s: file %s does not exists",
                  loc,em1);
         break;
      case kPFErrFileRename:
         snprintf(buf,XrdSutMAXBUF,
                  "XrdSutPFile::%s: error renaming file %s to %s"
                  " (%s)",loc,em1,em2,errbuf);
         break;
      case kPFErrStat:
         snprintf(buf,XrdSutMAXBUF,
                  "XrdSutPFile::%s: cannot file %s (%s)",
                  loc,em1,errbuf);
         break;
      case kPFErrFileOpen:
         snprintf(buf,XrdSutMAXBUF,
                  "XrdSutPFile::%s: cannot open file %s (%s)",
                  loc,em1,errbuf);
         break;
      case kPFErrFileNotOpen:
         snprintf(buf,XrdSutMAXBUF,
                  "XrdSutPFile::%s: file is not open", loc);
         break;
      case kPFErrLocking:
         fd = *((int *)em1);
         snprintf(buf,XrdSutMAXBUF,
                  "XrdSutPFile::%s: cannot lock file descriptor %d (%s)",
                  loc,fd,errbuf);
         break;
      case kPFErrUnlocking:
         fd = *((int *)em1);
         snprintf(buf,XrdSutMAXBUF,
                  "XrdSutPFile::%s: cannot unlock file descriptor %d (%s)",
                  loc,fd,errbuf);
         break;
      case kPFErrFileLocked:
         fd = *((int *)em2);
         snprintf(buf,XrdSutMAXBUF,
                  "XrdSutPFile::%s: file %s is locked by process %d",
                  loc,em1,fd);
         break;
      case kPFErrSeek:
         fd = *((int *)em2);
         snprintf(buf,XrdSutMAXBUF,
                  "XrdSutPFile::%s: lseek %s error on descriptor %d (%s)",
                  loc,em1,fd,errbuf);
         break;
      case kPFErrRead:
         fd = *((int *)em1);
         snprintf(buf,XrdSutMAXBUF,
                  "XrdSutPFile::%s: read error on descriptor %d (%s)",
                  loc,fd,errbuf);
         break;
      case kPFErrOutOfMemory:
         snprintf(buf,XrdSutMAXBUF,
                  "XrdSutPFile::%s: out of memory (%s)",
                  loc,errbuf);
         break;
      case kPFErrLenMismatch:
         lp = *((int *)em1);
         lt = *((int *)em2);
         snprintf(buf,XrdSutMAXBUF,
                  "XrdSutPFile::%s: length mismatch: %d (expected: %d)",
                  loc,lp,lt);
         break;
      case kPFErrBadOp:
         snprintf(buf,XrdSutMAXBUF,
                  "XrdSutPFile::%s: bad option: %s", loc,em1);
         break;
      default:
         DEBUG("unknown error code: "<<code);
   }

   // Print error string if requested
   DEBUG(buf);

   // Save error string
   fErrStr = buf;

   return -1;
}
