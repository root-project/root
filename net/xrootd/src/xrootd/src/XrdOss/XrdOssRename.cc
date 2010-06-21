/******************************************************************************/
/*                                                                            */
/*                       X r d O s s R e n a m e . c c                        */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

const char *XrdOssRenameCVSID = "$Id$";

#include <unistd.h>
#include <errno.h>
#include <strings.h>
#include <limits.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/param.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "XrdSys/XrdSysHeaders.hh"
#include "XrdOss/XrdOssApi.hh"
#include "XrdOss/XrdOssCache.hh"
#include "XrdOss/XrdOssError.hh"
#include "XrdOss/XrdOssLock.hh"
#include "XrdOss/XrdOssPath.hh"
#include "XrdOss/XrdOssTrace.hh"
#include "XrdOuc/XrdOucExport.hh"
#include "XrdOuc/XrdOucUtils.hh"

/******************************************************************************/
/*           G l o b a l   E r r o r   R o u t i n g   O b j e c t            */
/******************************************************************************/

extern XrdSysError OssEroute;

extern XrdOucTrace OssTrace;
  
/******************************************************************************/
/*                                R e n a m e                                 */
/******************************************************************************/

/*
  Function: Renames a file with name 'old_name' to 'new_name'.

  Input:    old_name  - Is the fully qualified name of the file to be renamed.
            new_name  - Is the fully qualified name that the file is to have.

  Output:   Returns XrdOssOK upon success and -errno upon failure.
*/
int XrdOssSys::Rename(const char *oldname, const char *newname)
{
    EPNAME("Rename")
    static const mode_t pMode = S_IRWXU | S_IRWXG;
    unsigned long long remotefs_Old, remotefs_New, remotefs, haslf;
    unsigned long long old_popts, new_popts;
    int i, retc2, retc = XrdOssOK;
    XrdOssLock old_file, new_file;
    struct stat statbuff;
    char  *slashPlus, sPChar;
    char  local_path_Old[MAXPATHLEN+8], *lpo;
    char  local_path_New[MAXPATHLEN+8], *lpn;
    char remote_path_Old[MAXPATHLEN+1];
    char remote_path_New[MAXPATHLEN+1];

// Determine whether we can actually rename a file on this server.
//
   remotefs_Old = Check_RO(Rename, old_popts, oldname, "renaming ");
   remotefs_New = Check_RO(Rename, new_popts, newname, "renaming to ");

// Make sure we are renaming within compatible file systems
//
   if (remotefs_Old ^ remotefs_New
   || ((old_popts & XRDEXP_MIG) ^ (new_popts & XRDEXP_MIG)))
      {char buff[MAXPATHLEN+128];
       snprintf(buff, sizeof(buff), "rename %s to ", oldname);
       return OssEroute.Emsg("Rename",-XRDOSS_E8011,buff,(char *)newname);
      }
   remotefs = remotefs_Old | remotefs_New;
   haslf    = (XRDEXP_MAKELF & (old_popts | new_popts));

// Construct the filename that we will be dealing with.
//
   if ( (retc = GenLocalPath( oldname, local_path_Old))
     || (retc = GenLocalPath( newname, local_path_New)) ) return retc;
   if (remotefs
     && (((retc = GenRemotePath(oldname, remote_path_Old))
     ||   (retc = GenRemotePath(newname, remote_path_New)))) ) return retc;

// Lock the target directory if this is a remote backed filesystem
//
   if (remotefs &&
       (retc2 = new_file.Serialize(local_path_New, XrdOssDIR|XrdOssEXC)) < 0)
      return retc2;

// Make sure that the target file does not exist
//
   retc2 = lstat(local_path_New, &statbuff);
   if (remotefs) new_file.UnSerialize(0);
   if (!retc2) return -EEXIST;

// We need to create the directory path if it does not exist.
//
   if (!(slashPlus = rindex(local_path_New, '/'))) return -EINVAL;
   slashPlus++; sPChar = *slashPlus; *slashPlus = '\0';
   retc2 = XrdOucUtils::makePath(local_path_New, pMode);
   *slashPlus = sPChar;
   if (retc2) return retc2;

// Serialize access to the source directory.
//
     if (remotefs &&
         (retc = old_file.Serialize(local_path_Old, XrdOssDIR|XrdOssEXC)) < 0)
       return retc;

// Check if this path is really a symbolic link elsewhere
//
    if (lstat(local_path_Old, &statbuff)) retc = -errno;
       else if ((statbuff.st_mode & S_IFMT) == S_IFLNK)
               retc = RenameLink(local_path_Old, local_path_New);
               else if (rename(local_path_Old, local_path_New)) retc = -errno;
    DEBUG("lcl rc=" <<retc <<" op=" <<local_path_Old <<" np=" <<local_path_New);

// For migratable space, rename all suffix variations of the base file
//
   if (haslf)
      {if ((!retc || retc == -ENOENT))
          {i = strlen(local_path_Old); lpo = &local_path_Old[i];
           i = strlen(local_path_New); lpn = &local_path_New[i];
           for (i = 0;  i < XrdOssPath::sfxMigL; i++)
               {strcpy(lpo,XrdOssPath::Sfx[i]); strcpy(lpn,XrdOssPath::Sfx[i]);
                if (rename(local_path_Old,local_path_New) && ENOENT != errno)
                   DEBUG("sfx retc=" <<errno <<" op=" <<local_path_Old);
               }
          }
      }

// Now rename the data file in the remote system if the local rename "worked".
// Do not do this if we really should not use the MSS (but unserialize!).
//
   if (remotefs)
      {if (remotefs && (!retc || retc == -ENOENT) && RSSCmd)
          {if ( (retc2 = MSS_Rename(remote_path_Old, remote_path_New))
              != -ENOENT) retc = retc2;
           DEBUG("rmt rc=" <<retc2 <<" op=" <<remote_path_Old <<" np=" <<remote_path_New);
          }

      // All done.
      //
         old_file.UnSerialize(0);
      }

// All done.
//
   return retc;
}
 
/******************************************************************************/
/*                       p r i v a t e   m e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                            R e n a m e L i n k                             */
/******************************************************************************/

int XrdOssSys::RenameLink(char *old_path, char *new_path)
{
   struct stat statbuff;
   char oldlnk[MAXPATHLEN+32], newlnk[MAXPATHLEN+32];
   int lnklen, n, rc = 0;

// Read the contents of the link
//
   if ((lnklen = readlink(old_path,oldlnk,sizeof(oldlnk)-1)) < 0) return -errno;
   oldlnk[lnklen] = '\0';

// Check if this is new or old style cache. Check if this is an offline rename
// and if so, add the space to the usage to account for stage-ins
//
   if (oldlnk[lnklen-1] == XrdOssPath::xChar)
      {if ((rc=RenameLink2(lnklen,oldlnk,old_path,newlnk,new_path))) return rc;
       if (Solitary && UDir)
          {n = strlen(old_path);
           if (n < 6 || strcmp(old_path+n-5, ".anew")
           ||  stat(new_path, &statbuff) || !statbuff.st_size) return 0;
           XrdOssPath::Trim2Base(oldlnk+lnklen-1);
           XrdOssCache::Adjust(oldlnk, statbuff.st_size);
          }
       return 0;
      }

// Convert old name to the new name
//
   if ((rc = XrdOssPath::Convert(newlnk, sizeof(newlnk), oldlnk, new_path)))
      {OssEroute.Emsg("RenameLink", rc, "convert", oldlnk);
       return rc;
      }

// Make sure that the target name does not exist
//
   if (!lstat(newlnk, &statbuff))
      {OssEroute.Emsg("RenameLink",-EEXIST,"check new target", newlnk);
       return -EEXIST;
      }

// Insert a new link in the target cache
//
   if (symlink(newlnk, new_path))
      {rc = errno;
       OssEroute.Emsg("RenameLink", rc, "symlink to", newlnk);
       return -rc;
      }

// Rename the actual target file
//
   if (rename(oldlnk, newlnk))
      {rc = errno;
       OssEroute.Emsg("RenameLink", rc, "rename", oldlnk);
       unlink(new_path);
       return -rc;
      }

// Now, unlink the source path
//
   if (unlink(old_path))
      OssEroute.Emsg("RenameLink", rc, "unlink", old_path);

// All done
//
   return 0;
}

/******************************************************************************/
/*                           R e n a m e L i n k 2                            */
/******************************************************************************/
  
int XrdOssSys::RenameLink2(int Llen, char *oLnk, char *old_path,
                                     char *nLnk, char *new_path)
{
   int rc;

// Setup to create new pfn file for this file
//
   strcpy(nLnk, oLnk);
   strcpy(nLnk+Llen, ".pfn");
   unlink(nLnk);

// Create the new pfn symlink
//
   if (symlink(new_path, nLnk))
      {rc = errno;
       OssEroute.Emsg("RenameLink", rc, "create symlink", nLnk);
       return -rc;
      }

// Create the new lfn symlink
//
   if (symlink(oLnk, new_path))
      {rc = errno;
       OssEroute.Emsg("RenameLink", rc, "symlink to", oLnk);
       unlink(nLnk);
       return -rc;
      }

// Now, unlink the old lfn path
//
   if (unlink(old_path))
      OssEroute.Emsg("RenameLink", errno, "unlink", old_path);

// All done (well, as well as it needs to be at this point)
//
   return 0;
}
