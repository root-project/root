#include "XrdTokenAuthzOfs.hh"
#include "XrdOfs/XrdOfsTrace.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdOuc/XrdOucTrace.hh"
#include "XrdOss/XrdOssApi.hh"
#include "XrdOuc/XrdOucString.hh"
#define WITHTHREADS
#include "TTokenAuthz.h"
#include <sys/time.h>
#include <stdarg.h>

// if you want to allow the alien shell to do normal work define ALIEN_BACKDOOR
#define ALIEN_BACKDOOR

extern XrdTokenAuthzOfs XrdOfsFS;

extern XrdSysError      OfsEroute;
extern XrdOssSys        XrdOssSS;
extern XrdOucTrace      OfsTrace;
extern void         *XrdOfsIdleScan(void *);

extern "C"
{
XrdSfsFileSystem *XrdSfsGetFileSystem(XrdSfsFileSystem *native_fs, 
                                      XrdSysLogger     *lp,
                                      const char       *configfn)
{
   pthread_t tid;
   int retc;
// Do the herald thing
//
   OfsEroute.SetPrefix("XrdTokenAuthzOfs_");
   OfsEroute.logger(lp);
   OfsEroute.Emsg("Init", "(c) 2005 CERN/Alice, XrdTokenAuthzOfs Version "
                          XrdVSTRING);

// Initialize the subsystems
//
   XrdOfsFS.ConfigFN = (configfn && *configfn ? strdup(configfn) : 0);
   if ( XrdOfsFS.Configure(OfsEroute) ) return 0;
   XrdOfsFS.Config_Display(OfsEroute);

   OfsEroute.Emsg("XrdOfsinit","initializing the XrdTokenAuthzOfs object");

// Initialize the target storage system
//
   if (XrdOssSS.Init(lp, configfn)) return 0;

// Start a thread to periodically scan for idle file handles
//
   if ((retc = XrdSysThread::Run(&tid, XrdOfsIdleScan, (void *)0)))
      OfsEroute.Emsg("XrdOfsinit", retc, "create idle scan thread");

// All done, we can return the callout vector to these routines.
//

   return (XrdSfsFileSystem*) &XrdOfsFS;
}
}
/////////////////////////////////////////////////////////////////////////////////////////
// the authz open function
/////////////////////////////////////////////////////////////////////////////////////////
int
XrdTokenAuthzOfsFile::open(const char                *fileName,
		XrdSfsFileOpenMode   openMode,
		mode_t               createMode,
		const XrdSecClientName    *client,
		const char                *opaque) 
{ 
  static const char *epname = "open";
  TAuthzXMLreader* authz=0;
  bool write_once = false;
 

  ZTRACE(open,"lfn    = " << fileName);
  
  std::map<std::string,std::string> env;
  TTokenAuthz::Tokenize(opaque,env,"&");

  // set the vo string from the opaque information
  std::string vo="*";
  if ( (env["vo"].length()) > 0) {
    vo = env["vo"];
  }
  
  // if we have the vo defined in the credentials, use this one
  if (client) {
    if ((client->vorg) && (strlen(client->vorg)))
      vo = client->vorg;
  }

  // set the certificate, if we have one
  const char* certsubject=0;
  if (client) {
    if ((client->name) && (strlen(client->name))) {
      certsubject = client->name;
    }
  }
 
  TTokenAuthz* tkauthz = 0;
  if (GTRACE(ALL)) {
    tkauthz = TTokenAuthz::GetTokenAuthz("xrootd",true); // with debug output
  } else {
    tkauthz = TTokenAuthz::GetTokenAuthz("xrootd",true);// no debug output
  }

  // set the opening mode
  std::string authzopenmode;
  if (!(openMode & (SFS_O_WRONLY + SFS_O_RDWR + SFS_O_CREAT + SFS_O_TRUNC))) {
    authzopenmode="read";
  } else {
    if (openMode & SFS_O_CREAT) {
      authzopenmode="write-once";
    } else {
      if (openMode & SFS_O_RDWR) {
	authzopenmode="read-write";
      }
    }
  }


  // if no authorization info is provided, we do the namespace export+authz checks
#ifdef ALIEN_BACKDOOR
  if ( ((env["authz"].length()) == 0 ) || (env["authz"] == "alien") ) {
#else
  if ( ((env["authz"].length()) == 0 ) ) {
#endif
    // check if the directory asked is exported 
    if (tkauthz->PathIsExported(fileName,vo.c_str(),certsubject)) {
      if (!tkauthz->PathHasAuthz(fileName,authzopenmode.c_str(),vo.c_str(),certsubject)) {
	// the pass through 
	return XrdOfsFile::open(fileName,openMode,createMode,client,opaque);        
      } else {
	// path needs authorization
	XrdOfsFS.Emsg(epname, error, EACCES, "give access for lfn - path has to be authorized", fileName);
	return  XrdTokenAuthzOfs::Emsg(epname, error, EACCES, "open", fileName);
      } 
    } else {
      // path is not exported for this VO
      XrdOfsFS.Emsg(epname, error, EACCES, "give access for lfn - path not exported", fileName);
      return XrdTokenAuthzOfs::Emsg(epname, error, EACCES, "open", fileName);
    }
  }

  int garesult=0;
  float t1,t2;
  garesult = tkauthz->GetAuthz(fileName,opaque,&authz,GTRACE(ALL),&t1,&t2);

  ZTRACE(ALL,"Time for Authz decoding: " << t1 << " ms" << " / " << t2 << "ms");

  if (garesult != TTokenAuthz::kAuthzOK) {
    // if the authorization decoding failed for any reason
    XrdOfsFS.Emsg(epname, error, tkauthz->PosixError(garesult), tkauthz->ErrorMsg(garesult) , fileName);
    if (authz) delete authz;
    return XrdTokenAuthzOfs::Emsg(epname, error, tkauthz->PosixError(garesult), "open", fileName);
  }

  // check the access permissions
  if (!(openMode & (SFS_O_WRONLY + SFS_O_RDWR + SFS_O_CREAT + SFS_O_TRUNC))) {
    // check that we have the READ access
    if (strcmp(authz->GetKey((char*)fileName,"access"), "read")) {
      // we have no read access
      XrdOfsFS.Emsg(epname, error, EACCES, "have read access for lfn", fileName);
      if (authz) delete authz;  
      return XrdTokenAuthzOfs::Emsg(epname, error, EACCES, "open", fileName);
    }
  } else {
    if (openMode & SFS_O_CREAT) {
      // check that we have the WRITE access
      if (strcmp(authz->GetKey((char*)fileName,"access"), "write-once")) {
	// we have no write-once access
	XrdOfsFS.Emsg(epname, error, EACCES, "have write access for lfn", fileName);
	if (authz) delete authz;  
	return XrdTokenAuthzOfs::Emsg(epname, error, EACCES, "open", fileName);
      }
      // force the creation of that directory
      createMode |= SFS_O_MKPTH;
      // force the write-once check
      write_once = true;

    } else {
      if (openMode & SFS_O_RDWR) {
	// check that we have the READ-WRITE access
	if (strcmp(authz->GetKey((char*)fileName,"access"), "read-write")) {
	  // we have no read-write access
	  XrdOfsFS.Emsg(epname, error, EACCES, "have read-write access for lfn", fileName);
	  if (authz) delete authz;  
	  return XrdTokenAuthzOfs::Emsg(epname, error, EACCES, "open", fileName);
	} else {
	  XrdOfsFS.Emsg(epname, error, EACCES, "have access for lfn", fileName);
	  if (authz) delete authz;  
	  return XrdTokenAuthzOfs::Emsg(epname, error, EACCES, "open", fileName);
	}
      }      
    }
  }
 

  // get the turl
  const char* newfilename = TTokenAuthz::GetPath(authz->GetKey((char*)fileName,"turl"));
  std::string copyfilename = newfilename;

  
  // check if the asked filename is exported 
  if (!tkauthz->PathIsExported(newfilename,vo.c_str())) {
    // path is not exported for this VO
    XrdOfsFS.Emsg(epname, error, EACCES, "give access for turl - path not exported", newfilename);
    if (authz) delete authz;
    return XrdTokenAuthzOfs::Emsg(epname, error, EACCES, "open", fileName);    
  }

  // do certifcate check, if it is required
  if (tkauthz->CertNeedsMatch(newfilename,vo.c_str())) {
    if (certsubject != authz->GetKey((char*)fileName,"certsubject")) {
      XrdOfsFS.Emsg(epname, error, EACCES, "give access for turl - certificate subject does not match", newfilename);
      return XrdTokenAuthzOfs::Emsg(epname, error, EACCES, "open", fileName);
    }
  }


  // if we are in write mode create the full directory path
  if ((openMode & (SFS_O_CREAT)) && (write_once)) {
    XrdSfsFileExistence exists_flag = (XrdSfsFileExistence) 0 ;
    XrdOfsFS.exists(copyfilename.c_str(),exists_flag, error, client);
    if (exists_flag != XrdSfsFileExistNo) {
      XrdOfsFS.Emsg(epname, error, EEXIST, "to open file for write - file exists for lfn", fileName);
      if (authz) delete authz;
      return XrdTokenAuthzOfs::Emsg(epname, error, EEXIST, "open", fileName);
    }
  }


  if (authz) delete authz;  
  std::string newopaque=""; 
  return XrdOfsFile::open(copyfilename.c_str(),openMode,createMode,client,newopaque.c_str());  
}

/////////////////////////////////////////////////////////////////////////////////////////
// the authz stat function
/////////////////////////////////////////////////////////////////////////////////////////
int
XrdTokenAuthzOfs::stat(const char             *Name,
		      struct stat             *buf,
		       XrdOucErrInfo          &out_error,
		       const XrdSecEntity     *client,
		       const char             *opaque) {
  static const char *epname = "stat";
  TAuthzXMLreader* authz=0;
  const char *tident = out_error.getErrUser();

  ZTRACE(stat,"lfn    = " << Name);

  std::map<std::string,std::string> env;
  TTokenAuthz::Tokenize(opaque,env,"&");
  // set the vo string from the opaque information
  std::string vo="*";
  if ( (env["vo"].length()) > 0) {
    vo = env["vo"];
  }
  TTokenAuthz* tkauthz = 0;
  if (GTRACE(ALL)) {
    tkauthz = TTokenAuthz::GetTokenAuthz("xrootd",true); // with debug output
  } else {
    tkauthz = TTokenAuthz::GetTokenAuthz("xrootd",false);// no debug output
  }
  // if no authorization info is provided, we do the namespace export+authz checks
#ifdef ALIEN_BACKDOOR
  if ( ((env["authz"].length()) == 0 ) || (env["authz"] == "alien") ) {
#else
  if ( ((env["authz"].length()) == 0 ) ) {
#endif
    // check if the directory asked is exported 
    if (tkauthz->PathIsExported(Name,vo.c_str())) {
      if (!tkauthz->PathHasAuthz(Name,"read",vo.c_str())) {
	// the pass through 
	return XrdOfs::stat(Name,buf,out_error,client);
      } else {
	// path needs authorization
	XrdOfsFS.Emsg(epname, out_error, EACCES, "give access for lfn - path has to be authorized", Name);
	return  XrdTokenAuthzOfs::Emsg(epname, out_error, EACCES, "stat", Name);
      } 
    } else {
      // path is not exported for this VO
      XrdOfsFS.Emsg(epname, out_error, EACCES, "give access for lfn - path not exported", Name);
      return XrdTokenAuthzOfs::Emsg(epname, out_error, EACCES, "stat", Name);
    }
  }

  // get the authorization data
  int garesult=0;
  float t1,t2;
  garesult = tkauthz->GetAuthz(Name,opaque,&authz,GTRACE(ALL),&t1,&t2);
  ZTRACE(ALL,"Time for Authz decoding: " << t1 << " ms" << " / " << t2 << "ms");

  if (garesult != TTokenAuthz::kAuthzOK) {
    // if the authorization decoding failed for any reason
    XrdOfsFS.Emsg(epname, out_error, tkauthz->PosixError(garesult), tkauthz->ErrorMsg(garesult) , Name);
    if (authz) delete authz;
    return XrdTokenAuthzOfs::Emsg(epname, out_error, tkauthz->PosixError(garesult), "stat", Name);
  }

  // get the turl
  const char* newfilename = TTokenAuthz::GetPath(authz->GetKey(Name,"turl"));
  std::string copyfilename = newfilename;

  // check if the asked filename is exported 
  if (!tkauthz->PathIsExported(newfilename,vo.c_str())) {
    // path is not exported for this VO
    XrdOfsFS.Emsg(epname, out_error, EACCES, "give access for turl - path not exported", newfilename);
    if (authz) delete authz;
    return XrdTokenAuthzOfs::Emsg(epname, out_error, EACCES, "open", Name);    
  }

  if (authz) delete authz;
  return XrdOfs::stat(copyfilename.c_str(),buf,out_error,client);
}

/////////////////////////////////////////////////////////////////////////////////////////
// the authz stat function
/////////////////////////////////////////////////////////////////////////////////////////
int
XrdTokenAuthzOfs::stat(const char             *Name,
		       mode_t                 &mode,
		       XrdOucErrInfo          &out_error,
		       const XrdSecEntity     *client,
		       const char             *opaque) {
  static const char *epname = "stat";

  TAuthzXMLreader* authz=0;
  const char *tident = out_error.getErrUser();

  ZTRACE(stat,"lfn    = " << Name);
  std::map<std::string,std::string> env;
  TTokenAuthz::Tokenize(opaque,env,"&");
  // set the vo string from the opaque information
  std::string vo="*";
  if ( (env["vo"].length()) > 0) {
    vo = env["vo"];
  }
  TTokenAuthz* tkauthz = 0;
  if (GTRACE(ALL)) {
    tkauthz = TTokenAuthz::GetTokenAuthz("xrootd",true); // with debug output
  } else {
    tkauthz = TTokenAuthz::GetTokenAuthz("xrootd",false);// no debug output
  }
  // if no authorization info is provided, we do the namespace export+authz checks
#ifdef ALIEN_BACKDOOR
  if ( ((env["authz"].length()) == 0 ) || (env["authz"] == "alien") ) {
#else
  if ( ((env["authz"].length()) == 0 ) ) {
#endif
    // check if the directory asked is exported 
    if (tkauthz->PathIsExported(Name,vo.c_str())) {
      if (!tkauthz->PathHasAuthz(Name,"read",vo.c_str())) {
	// the pass through 
	return XrdOfs::stat(Name,mode,out_error,client);
      } else {
	// path needs authorization
	XrdOfsFS.Emsg(epname, out_error, EACCES, "give access for lfn - path has to be authorized", Name);
	return  XrdTokenAuthzOfs::Emsg(epname, out_error, EACCES, "stat", Name);
      } 
    } else {
      // path is not exported for this VO
      XrdOfsFS.Emsg(epname, out_error, EACCES, "give access for lfn - path not exported", Name);
      return XrdTokenAuthzOfs::Emsg(epname, out_error, EACCES, "stat", Name);
    }
  }

  // get the authorization data
  int garesult=0;
  float t1,t2;
  garesult = tkauthz->GetAuthz(Name,opaque,&authz,GTRACE(ALL),&t1,&t2);

  ZTRACE(ALL,"Time for Authz decoding: " << t1 << " ms" << " / " << t2 << "ms");

  if (garesult != TTokenAuthz::kAuthzOK) {
    // if the authorization decoding failed for any reason
    XrdOfsFS.Emsg(epname, out_error, tkauthz->PosixError(garesult), tkauthz->ErrorMsg(garesult) , Name);
    if (authz) delete authz;
    return XrdTokenAuthzOfs::Emsg(epname, out_error, tkauthz->PosixError(garesult), "stat", Name);
  }

  // get the turl
  const char* newfilename = TTokenAuthz::GetPath(authz->GetKey(Name,"turl"));
  std::string copyfilename = newfilename;

  // check if the asked filename is exported 
  if (!tkauthz->PathIsExported(newfilename,vo.c_str())) {
    // path is not exported for this VO
    XrdOfsFS.Emsg(epname, out_error, EACCES, "give access for turl - path not exported", newfilename);
    if (authz) delete authz;
    return XrdTokenAuthzOfs::Emsg(epname, out_error, EACCES, "open", Name);    
  }

  if (authz) delete authz;
  return XrdOfs::stat(copyfilename.c_str(),mode,out_error,client);
}

/////////////////////////////////////////////////////////////////////////////////////////
// the authz rem function
/////////////////////////////////////////////////////////////////////////////////////////
int
XrdTokenAuthzOfs::rem(const char             *path,
	              XrdOucErrInfo          &out_error,
		      const XrdSecEntity     *client,
		      const char             *opaque) {

  static const char *epname = "rem";
  std::string url(path);
  std::string filename("");
  const char *tident = out_error.getErrUser();
  TAuthzXMLreader* authz=0;

  filename = path;

  ZTRACE(remove,"lfn    = " << filename);

  std::map<std::string,std::string> env;
  TTokenAuthz::Tokenize(opaque,env,"&");

  // set the vo string from the opaque information
  std::string vo="*";
  if ( (env["vo"].length()) > 0) {
    vo = env["vo"];
  }

  TTokenAuthz* tkauthz = 0;
  if (GTRACE(ALL)) {
    tkauthz = TTokenAuthz::GetTokenAuthz("xrootd",true); // with debug output
  } else {
    tkauthz = TTokenAuthz::GetTokenAuthz("xrootd",false);// no debug output
  }

  // if no authorization info is provided, we do the namespace export+authz checks
#ifdef ALIEN_BACKDOOR
  if ( ((env["authz"].length()) == 0 ) || (env["authz"] == "alien") ) {
#else
  if ( ((env["authz"].length()) == 0 ) ) {
#endif
    // check if the directory asked is exported 
    if (tkauthz->PathIsExported(filename.c_str(),vo.c_str())) {
      if (!tkauthz->PathHasAuthz(filename.c_str(),"delete",vo.c_str())) {
	// the pass through 
	return XrdOfs::rem(filename.c_str(),out_error,client);
      } else {
	// path needs authorization
	XrdOfsFS.Emsg(epname, out_error, EACCES, "give access for lfn - path has to be authorized", filename.c_str());
	return  XrdTokenAuthzOfs::Emsg(epname, out_error, EACCES, "rem", filename.c_str());
      } 
    } else {
      // path is not exported for this VO
      XrdOfsFS.Emsg(epname, out_error, EACCES, "give access for lfn - path not exported", filename.c_str());
      return XrdTokenAuthzOfs::Emsg(epname, out_error, EACCES, "rem", filename.c_str());
    }
  }

  // get the authorization data
  int garesult=0;
  float t1,t2;
  garesult = tkauthz->GetAuthz(filename.c_str(),opaque,&authz,GTRACE(ALL),&t1,&t2);

  ZTRACE(ALL,"Time for Authz decoding: " << t1 << " ms" << " / " << t2 << "ms");

  if (garesult != TTokenAuthz::kAuthzOK) {
    // if the authorization decoding failed for any reason
    XrdOfsFS.Emsg(epname, out_error, tkauthz->PosixError(garesult), tkauthz->ErrorMsg(garesult) , filename.c_str());
    if (authz) delete authz;
    return XrdTokenAuthzOfs::Emsg(epname, out_error, tkauthz->PosixError(garesult), "stat", filename.c_str());
  }

  // get the turl
  const char* newfilename = TTokenAuthz::GetPath(authz->GetKey(filename.c_str(),"turl"));
  std::string deletefilename = newfilename;

  if (strcmp(authz->GetKey(filename.c_str(),"access"), "delete")) {
    if (authz) delete authz;  
    return -1;
  }
  if (authz) delete authz;  

  // check if the asked filename is exported 
  if (!tkauthz->PathIsExported(newfilename,vo.c_str())) {
    // path is not exported for this VO
    XrdOfsFS.Emsg(epname, out_error, EACCES, "give access for turl - path not exported", newfilename);
    if (authz) delete authz;
    return XrdTokenAuthzOfs::Emsg(epname, out_error, EACCES, "open", filename.c_str());    
  }  

  return XrdOfs::rem(deletefilename.c_str(),out_error,client);
}

