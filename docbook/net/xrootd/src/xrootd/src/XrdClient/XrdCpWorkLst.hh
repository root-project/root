//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdCpWorkLst                                                         //
//                                                                      //
// Author: Fabrizio Furano (INFN Padova, 2004)                          //
//                                                                      //
// A class implementing a list of cp to do for XrdCp                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//   $Id$

#include <sys/types.h>
#include "XrdClient/XrdClientAdmin.hh"
#include "XrdClient/XrdClient.hh"

class XrdSysDir;
void PrintLastServerError(XrdClient *cli);
bool PedanticOpen4Write(XrdClient *cli, kXR_unt16 mode, kXR_unt16 options);

class XrdCpWorkLst {

   vecString fWorkList;
   int fWorkIt;

   XrdClientAdmin *xrda_src, *xrda_dst;

   XrdOucString fSrc, fDest;
   bool fDestIsDir, fSrcIsDir;

 public:
   
   XrdCpWorkLst();
   ~XrdCpWorkLst();

   // Sets the source path for the file copy
   int SetSrc(XrdClient **srccli, XrdOucString url,
	      XrdOucString urlopaquedata, bool do_recurse);

   // Sets the destination of the file copy
   int SetDest(XrdClient **xrddest, const char *url,
	       const char *urlopaquedata,
	       kXR_unt16 xrdopenflags);

   inline void GetDest(XrdOucString &dest, bool& isdir) {
      dest = fDest;
      isdir = fDestIsDir;
   }

   inline void GetSrc(XrdOucString &src, bool& isdir) {
      src = fSrc;
      isdir = fSrcIsDir;
   }


   // Actually builds the worklist
   int BuildWorkList_xrd(XrdOucString url, XrdOucString opaquedata);
   int BuildWorkList_loc(XrdSysDir *dir, XrdOucString pat);

   bool GetCpJob(XrdOucString &src, XrdOucString &dest);
   
};
