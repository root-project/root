/******************************************************************************/
/*                                                                            */
/*                     X r d A c c A u t h F i l e . c c                      */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

const char *XrdAccAuthFileCVSID = "$Id$";

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>

#include "XrdAcc/XrdAccAuthFile.hh"
  
/******************************************************************************/
/*                   X r d A c c A u t h D B _ O b j e c t                    */
/******************************************************************************/
  
XrdAccAuthDB *XrdAccAuthDBObject(XrdSysError *erp)
{
      static XrdAccAuthFile mydatabase(erp);

      return (XrdAccAuthDB *)&mydatabase;
}
  
/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdAccAuthFile::XrdAccAuthFile(XrdSysError *erp)
{

// Set starting values
//
   authfn = 0;
   flags = Noflags;
   modtime = 0;
   Eroute = erp;

// Setup for an error in the first record
//
   strcpy(path_buff, "start of file");
}

/******************************************************************************/
/*                            D e s t r u c t o r                             */
/******************************************************************************/

XrdAccAuthFile::~XrdAccAuthFile()
{

// If the file is open, close it
//
   if (flags &isOpen) Close();

// Free the authfn string
//
   if (authfn) free(authfn);
}

/******************************************************************************/
/*                               C h a n g e d                                */
/******************************************************************************/

int XrdAccAuthFile::Changed(const char *dbfn)
{
    struct stat statbuff;

// If no file here, indicate nothing changed
//
   if (!authfn || !*authfn) return 0;

// If file paths differ, indicate that something has changed
//
   if (dbfn && strcmp(dbfn, authfn)) return 1;

// Get the modification timestamp for this file
//
   if (stat(authfn, &statbuff))
      {Eroute->Emsg("AuthFile", errno, "find", authfn);
       return 0;
      }

// Indicate whether or not the file has changed
//
   return (modtime < statbuff.st_mtime);
}
  
/******************************************************************************/
/*                                 C l o s e                                  */
/******************************************************************************/
  
int XrdAccAuthFile::Close()
{
// Return is the file is not open
//
   if (!(flags & isOpen)) return 1;

// Close the stream
//
   DBfile.Close();

// Unlock the protecting mutex
//
   DBcontext.UnLock();

// Indicate file is no longer open
//
   flags = (DBflags)(flags & ~isOpen);

// Return indicator of whether we had any errors
//
   if (flags & dbError) return 0;
   return 1;
}

/******************************************************************************/
/*                                 g e t P P                                  */
/******************************************************************************/
  
int XrdAccAuthFile::getPP(char **path, char **priv)
{
   char *pp, *bp;

// If a record has not been read, return end of record (i.e., 0)
//
   if (!(flags & inRec)) return 0;

// read the next word from the record (if none, simulate end of record)
//
   if (!(pp = DBfile.GetWord()))
      {flags = (DBflags)(flags & ~inRec);
       return 0;
      }

// Copy the value since the stream buffer might get overlaid.
//
   bp = Copy(path_buff, pp, sizeof(path_buff)-1);
   *path = path_buff;

// Check if this is really a path or a template
//
   if (*path_buff != '/') {*priv = (char *)0; return 1;}

// Verify that the path ends correctly (normally we would force a slash to
// appear at the end but that prevents caps on files. So, we commented the
// code out until we decide that maybe we really need to do this, sigh.
//
// bp--;
// if (*bp != '/') {bp++; *bp = '/'; bp++; *bp = '\0';}

// Get the next word which should be the privilege string
//
   if (!(pp = DBfile.GetWord()))
      {flags = (DBflags)(flags & ~inRec);
       Eroute->Emsg("AuthFile", "Privileges missing after", path_buff);
       flags = (DBflags)(flags | dbError);
       *priv = (char *)0;
       return 0;
      }

// All done here
//
   *priv = pp;
   return 1;
}

/******************************************************************************/
/*                                g e t R e c                                 */
/******************************************************************************/
  
char XrdAccAuthFile::getRec(char **recname)
{
   char *pp;
   int idok;

// Do this until we get a vlaid record
//
   while(1)
        {
         // If we arer still in the middle of a record, flush it
         //
         if (flags & inRec) while(DBfile.GetWord()) {}
            else flags = (DBflags)(flags | inRec);

        // Get the next word, the record type
        //
        if (!(pp = DBfile.GetWord()))
           {*recname = (char *)0; return '\0';}

        // Verify the id-type
        //
        idok = 0;
        if (strlen(pp) == 1)
           switch(*pp)
                 {case 'g':
                  case 'h':
                  case 's':
                  case 'n':
                  case 't':
                  case 'u': idok = 1;
                            break;
                   default: break;
                 }

        // Check if the record type was valid
        //
        if (!idok) {Eroute->Emsg("AuthFile", "Invalid id type -", pp);
                    flags = (DBflags)(flags | dbError);
                    continue;
                   }
        rectype = *pp;

        // Get the record name. It must exist
        //
        if (!(pp = DBfile.GetWord()))
           {Eroute->Emsg("AuthFile","Record name is missing after",path_buff);
            flags = (DBflags)(flags | dbError);
            continue;
           }

        // Copy the record name
        //
        Copy(recname_buff, pp, sizeof(recname_buff));
        *recname = recname_buff;
        return rectype;
       }
   return '\0'; // Keep the compiler happy :-)
}

/******************************************************************************/
/*                                  O p e n                                   */
/******************************************************************************/

int XrdAccAuthFile::Open(XrdSysError &eroute, const char *path)
{
   struct stat statbuff;
   int authFD;

// Enter the DB context (serialize use of this database)
//
   DBcontext.Lock();
   Eroute = &eroute;

// Use whichever path is the more recent
//
   if (path)
      {if (authfn) free(authfn); authfn = strdup(path);}
   if( !authfn || !*authfn) return Bail(0, "Authorization file not specified.");

// Get the modification timestamp for this file
//
   if (stat(authfn, &statbuff)) return Bail(errno, "find", authfn);

// Try to open the authorization file.
//
   if ( (authFD = open(authfn, O_RDONLY, 0)) < 0)
      return Bail(errno,"open authorization file",authfn);

// Copy in all the relevant information
//
   modtime = statbuff.st_mtime;
   flags = isOpen;
   DBfile.SetEroute(Eroute);
   DBfile.Tabs(0);

// Attach the file to the stream
//
   if (DBfile.Attach(authFD))
      return Bail(DBfile.LastError(), "initialize stream for", authfn);
   return 1;
}
  
/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                                  B a i l                                   */
/******************************************************************************/
  
int XrdAccAuthFile::Bail(int retc, const char *txt1, const char *txt2)
{
// This routine is typically used by open and the DBcontext lock must be held
//
   flags = (DBflags)(flags & ~isOpen);
   DBcontext.UnLock();
   if (retc) Eroute->Emsg("AuthFile", retc, txt1, txt2);
      else   Eroute->Emsg("AuthFile", txt1, txt2);
   return 0;
}

/******************************************************************************/
/*                                  C o p y                                   */
/******************************************************************************/

// This routine is used instead of strncpy because, frankly, it's a lot smarter
  
char *XrdAccAuthFile::Copy(char *dp, char *sp, int dplen)
{
   // Copy one less that the size of the buffer so that we have room for null
   //
   while(--dplen && *sp) {*dp = *sp; dp++; sp++;}

// Insert a null character and return a pointer to it.
//
   *dp = '\0';
   return dp;
}
