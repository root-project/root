#ifndef __ENVOFS__H__
#define __ENVOFS__H__
/******************************************************************************/
/*                                                                            */
/*                       X r d E n v O f s . h h                              */
/*                                                                            */
/******************************************************************************/
#include "XrdVersion.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysLogger.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdSec/XrdSecInterface.hh"
#include "XrdSfs/XrdSfsAio.hh"
#include "XrdSfs/XrdSfsNative.hh"
#include "XrdOuc/XrdOucEnv.hh"
#include "XrdOfs/XrdOfs.hh"

/******************************************************************************/
/*                 X r d E n v O f s D i r e c t o r y                  */
/******************************************************************************/
  
class XrdTokenAuthzOfsDirectory : public XrdOfsDirectory
{
public:

        int         open(const char              *dirName,
			   const XrdSecClientName  *client = 0) {return XrdOfsDirectory::open(dirName,client);}

	const char *nextEntry(){return XrdOfsDirectory::nextEntry();}

        int         close(){return XrdOfsDirectory::close();}
	  ;
const   char       *FName() {return XrdOfsDirectory::FName();}
                    XrdTokenAuthzOfsDirectory(char* user=0) : XrdOfsDirectory(user) {}
private:
};

/******************************************************************************/
/*                      X r d E n v O f s F i l e                       */
/******************************************************************************/
  

class XrdTokenAuthzOfsFile : public XrdOfsFile
{
public:

        int            open(const char                *fileName,
                                  XrdSfsFileOpenMode   openMode,
                                  mode_t               createMode,
                            const XrdSecClientName    *client = 0,
	                    const char                *opaque = 0) ;
        int            close(){return XrdOfsFile::close();}

        const char    *FName() {return XrdOfsFile::FName();}

        int            getMmap(void **Addr, off_t &Size){return XrdOfsFile::getMmap(Addr,Size);}

        int            read(XrdSfsFileOffset   fileOffset,
 	                    XrdSfsXferSize     preread_sz) {return XrdOfsFile::read(fileOffset,preread_sz);}

        XrdSfsXferSize read(XrdSfsFileOffset   fileOffset,
                            char              *buffer,
			    XrdSfsXferSize     buffer_size) {return XrdOfsFile::read(fileOffset,buffer,buffer_size);}

        int            read(XrdSfsAio *aioparm) {return XrdOfsFile::read(aioparm);}

        XrdSfsXferSize write(XrdSfsFileOffset   fileOffset,
                             const char        *buffer,
			     XrdSfsXferSize     buffer_size){return XrdOfsFile::write(fileOffset,buffer,buffer_size);}

        int            write(XrdSfsAio *aioparm){return XrdOfsFile::write(aioparm);}

        int            sync(){return XrdOfsFile::sync();}

        int            sync(XrdSfsAio *aiop){return XrdOfsFile::sync(aiop);}

        int            stat(struct stat *buf){return XrdOfsFile::stat(buf);}

        int            truncate(XrdSfsFileOffset   fileOffset){return XrdOfsFile::truncate(fileOffset);}

        int            getCXinfo(char cxtype[4], int &cxrsz) {return XrdOfsFile::getCXinfo(cxtype,cxrsz);}
                       XrdTokenAuthzOfsFile(char* user=0) : XrdOfsFile(user) {}
private:
};

/******************************************************************************/
/*                          X r d S f s N a t i v e                           */
/******************************************************************************/
  
class XrdTokenAuthzOfs : public XrdOfs
{
friend class XrdTokenAuthzOfsDirectory;
friend class XrdTokenAuthzOfsFile;
public:

// Object Allocation Functions
//
        XrdSfsDirectory *newDir(char *user=0)
                        {return (XrdSfsDirectory *)new XrdTokenAuthzOfsDirectory(user);}

        XrdSfsFile      *newFile(char *user=0)
                        {return      (XrdSfsFile *)new XrdTokenAuthzOfsFile(user);}

// Other Functions
//
        int            chmod(const char             *Name,
                                   XrdSfsMode        Mode,
                                   XrdOucErrInfo    &out_error,
                       	     const XrdSecClientName *client = 0){return XrdOfs::chmod(Name,Mode,out_error,client);}

        int            exists(const char                *fileName,
                                    XrdSfsFileExistence &exists_flag,
                                    XrdOucErrInfo       &out_error, 
	                      const XrdSecClientName    *client = 0){return XrdOfs::exists(fileName,exists_flag,out_error,client);}

        int            fsctl(const int               cmd,
                             const char             *args,
                                   XrdOucErrInfo    &out_error,
			     const XrdSecClientName *client = 0) {return XrdOfs::fsctl(cmd,args,out_error,client);}

        int            getStats(char *buff, int blen) {return XrdOfs::getStats(buff,blen);}

 const   char          *getVersion(){return XrdOfs::getVersion();}

        int            mkdir(const char             *dirName,
                                   XrdSfsMode        Mode,
                                   XrdOucErrInfo    &out_error,
                             const XrdSecClientName *client = 0){return XrdOfs::mkdir(dirName,Mode,out_error,client);}

        int            prepare(      XrdSfsPrep       &pargs,
                                     XrdOucErrInfo    &out_error,
                               const XrdSecClientName *client = 0) {return XrdOfs::prepare(pargs,out_error,client);}

        int            rem(const char             *path,
		                 XrdOucErrInfo    &out_error,
		           const XrdSecEntity     *client,
		           const char             *info = 0);


        int            remdir(const char             *dirName,
                                    XrdOucErrInfo    &out_error,
                              const XrdSecClientName *client = 0){return XrdOfs::remdir(dirName,out_error,client);};

        int            rename(const char             *oldFileName,
                              const char             *newFileName,
                                    XrdOucErrInfo    &out_error,
                              const XrdSecClientName *client = 0){return XrdOfs::rename(oldFileName,newFileName,out_error,client);};

        int            stat(const char             *Name,
                                  struct stat      *buf,
                                  XrdOucErrInfo    &out_error,
			    const XrdSecEntity     *client,
                            const char             *opaque = 0);

        int            stat(const char             *Name,
                                  mode_t           &mode,
                                  XrdOucErrInfo    &out_error,
			    const XrdSecEntity     *client,
                            const char             *opaque = 0);

};
#endif
