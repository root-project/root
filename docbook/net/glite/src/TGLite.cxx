// @(#) root/glite:$Id$
// Author: Anar Manafov <A.Manafov@gsi.de> 2006-03-20

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/************************************************************************/
/*! \file TGLite.cxx
  Implementation of the class which
  defines interface to gLite GRID services. */ /*

         version number:    $LastChangedRevision: 1688 $
         created by:        Anar Manafov
                            2006-03-20
         last changed by:   $LastChangedBy: manafov $ $LastChangedDate: 2008-01-23 13:42:32 +0100 (Wed, 23 Jan 2008) $

         Copyright (c) 2006-2008 GSI GridTeam. All rights reserved.
*************************************************************************/

//*-- Last Update : $LastChangedDate: 2008-01-23 13:42:32 +0100 (Wed, 23 Jan 2008) $ by $LastChangedBy: manafov $
//*-- Author : Anar Manafov (A.Manafov@gsi.de) 2006-03-20
//*-- Copyright: Copyright (c) 2006-2008 GSI GridTeam. All rights reserved.

// glite-api-wrapper
#include <glite-api-wrapper/gLiteAPIWrapper.h>
// ROOT
#include "TMap.h"
#include "TObjString.h"
// ROOT RGLite
#include "TGLiteResult.h"
#include "TGLiteJob.h"
#include "TGLite.h"

////////////////////////////////////////////////////////////////////////////////
/* BEGIN_HTML
The TGLite class defines an interface to <A href="http://glite.web.cern.ch/glite/" name="gLite GRID services">gLite GRID services</A>. This class is a part of RGlite plug-in developed at <A href="http://www.gsi.de" name="GSI, Darmstadt">GSI, Darmstadt</A>.<br>
The RGLite plug-in uses <A href="https://subversion.gsi.de/trac/dgrid/wiki">glite-api-wrapper library (GAW)</A> to perform Grid operations and provides the following features:
<ul>
<li>Workload Management System operations:
<ul>
    <li>job submission - normal, DAG and parametric jobs (gLite WMProxy API),</li>
    <li>smart look-up algorithm for WMP-Endpoints,</li>
    <li>job status querying (gLite LB API),</li>
    <li>job output retrieving (Globus GridFTP).</li>
</ul>
</li>
<li>File Catalog operations (gLite/LCG LFC API):
<ul>
    <li>smart session manager,</li>
    <li>set/query the current working catalog directory,</li>
    <li>list files, directories and their stats,</li>
    <li>add/remove files in a catalog namespace,</li>
    <li>add/remove directories,</li>
    <li>add/remove replicas from a given file.</li>
</ul>
</li>
<li>An executive logging.</li>
<li>Support of an external xml configuration file with according XML schema.</li>
</ul>

<h3>Content</h3>
<ol style="list-style-type: upper-roman;">
    <li><a href="#requirements">Requirements</a></li>
    <li><a href="#conf">Configuration</a></li>
    <li><a href="#usage">Usage</a>
        <ol>
            <li><a href="#usage:job_opt">Job operations</a></li>
            <li><a href="#usage:file_catalog_opt">File Catalog operations</a></li>
        </ol>
    </li>
</ol>
<br>

<h3><a name="requirements">Requirements</a></h3>
<ol>
    <li><A href="http://glite.web.cern.ch/glite/packages/userInterface.asp">gLite UI 3.1</A></li>
    <li><A href="https://subversion.gsi.de/trac/dgrid/wiki">glite-api-wrapper library</A></li>
    <li>A Grid proxy (in order to perform gLite operations RGLite requires a Grid proxy, users therefore should create the Grid proxy before using the plug-in. One can create the Grid proxy with help of <em>voms-proxy-init</em> command, for example)</li>
</ol>

<h3><a name="conf">Configuration</a></h3>
Since RGLite plug-in is based on <A href="https://subversion.gsi.de/trac/dgrid/wiki">glite-api-wrapper library</A> one should use GAW configuration file for tuning RGLite options up. Please refer to <A href="https://subversion.gsi.de/trac/dgrid/wiki">a GAW Trac</A> for more information.

<h3><a name="usage">Usage</a></h3>
Be advised that the call of <em>TGrid::Connect("glite")</em> should be the very first one. It initializes RGLite plug-in and assigns a global variable <em>gGrid</em> with a pointer to the RGLite plug-in object. The plug-in on his side will initialize a GAW singleton and the <em>TGrid::Connect("glite")</em> call therefore can be performed only once. If you successfully "connected" to gLite, you can call other methods of TGrid interface. Please see the following examples.
<h4><a name="usage:job_opt">Job operations</a></h4>
<table width="100%" border="0">
  <tbody bgcolor="#ffdca8">
    <tr>
        <td>
    <font color="Green">// loading RGLite plug-in</font><br>
    TGrid::Connect("glite");<br>
    <font color="Green">// submitting Grid job</font><br>
    TGridJob *job = gGrid->Submit("JDLs/simple.jdl");<br>
    <font color="Green">// getting status object</font><br>
    TGridJobStatus *status = job->GetJobStatus();<br>
    <font color="Green">// getting status of the job.</font><br>
    TGridJobStatus::EGridJobStatus st( status->GetStatus() );<br>
    <font color="Green">// when the st is TGridJobStatus::kDONE you can retrieve job&#039;s output</font><br>
    job->GetOutputSandbox("/tmp");<br>
    </td>
    </tr>
  </tbody>
</table>
<h4><a name="usage:file_catalog_opt">File Catalog operations</a></h4>
<h5>Cd, Pwd, Ls</h5>
<table width="100%">
  <tbody bgcolor="#ffdca8">
    <tr>
        <td>
    <font color="Green">// loading RGLite plug-in</font><br>
    TGrid::Connect("glite");<br>
    <font color="Green">// current Catalog directory</font><br>
         cout &lt;&lt; &quot;Working Directory is &quot; &lt;&lt; gGrid-&gt;Pwd() &lt;&lt; endl;
        <br>
    <font color="Green">// listing the current directory</font><br>
    TGridResult* result = gGrid->Ls();<br>
    result->Print("all");<br>
    <font color="Green">// changing the current directory to "dech"</font><br>
    gGrid->Cd("dech");<br>
    <font color="Green">// listing only file names</font><br>
    TGridResult * res = gGrid->Ls();<br>
    Int_t i = 0;<br>
         while ( res-&gt;GetFileName( i ) )
        <br>
         &nbsp;&nbsp; cout &lt;&lt; &quot;File: &quot; &lt;&lt; res-&gt;GetFileName( i++ ) &lt;&lt; endl;
        <br>
        </td>
    </tr>
  </tbody>
</table>

<h5>Mkdir, Rmdir</h5>

<table width="100%">
  <tbody bgcolor="#ffdca8">
    <tr>
        <td>
    <font color="Green">// loading RGLite plug-in</font><br>
    TGrid::Connect("glite");<br>
    <font color="Green">// changing the current directory to "/grid/dech"</font><br>
    gGrid->Cd("/grid/dech");<br>
    <font color="Green">// using Mkdir to create a new directory</font><br>
    Bool_t b = gGrid->Mkdir("root_test2");<br>
    <font color="Green">// listing the current directory</font><br>
    TGridResult* result = gGrid->Ls();<br>
    <font color="Green">// full file information</font><br>
    result->Print("all");<br>
    <font color="Green">// removing the directory </font><br>
    b = gGrid->Rmdir("root_test2");<br>
        </td>
    </tr>
  </tbody>
</table>
END_HTML */
////////////////////////////////////////////////////////////////////////////////

ClassImp(TGLite)

using namespace std;
using namespace glite_api_wrapper;
using namespace LFCHelper;
using namespace MiscCommon;

template<class _T>
void add_map_element(TMap *_map, const string &_key, const _T &_value)
{
   ostringstream ss;
   ss << _value;
   TObjString * key(new TObjString(_key.c_str()));
   TObjString *value(new TObjString(ss.str().c_str()));
   _map->Add(dynamic_cast<TObject*>(key), dynamic_cast<TObject*>(value));
}

struct SAddRepInfo: public binary_function<SLFCRepInfo_t, TMap*, bool> {
   bool operator()(first_argument_type _rep, second_argument_type m_Map) const {
      ostringstream strSFN;
      strSFN << "sfn" << _rep.id;
      stringstream strHost;
      strHost << "host" << _rep.id;
      add_map_element(m_Map, strSFN.str(), _rep.sfn);
      add_map_element(m_Map, strHost.str(), _rep.host);
      return true;
   }
};

struct SAddMapElementFunc: public binary_function<SLFCFileInfo_t, TGLiteResult*, bool> {
   bool operator()(first_argument_type _lfc_info, second_argument_type _Result) const {
      TMap * map = new TMap();

      add_map_element(map, "fileid", _lfc_info.m_nFileID);
      add_map_element(map, "name", _lfc_info.m_sName);
      add_map_element(map, "size", _lfc_info.m_nSize);
      add_map_element(map, "guid", _lfc_info. m_sGUID);
      add_map_element(map, "rep_count", _lfc_info.m_LFCRepInfoVector.size());

      // Add replication info
      for_each(_lfc_info.m_LFCRepInfoVector.begin(),
               _lfc_info.m_LFCRepInfoVector.end(),
               bind2nd(SAddRepInfo(), map));

      _Result->Add(map);
      return true;
   }
};

//______________________________________________________________________________
TGLite::TGLite(const char */*_gridurl*/, const char* /*uid*/, const char* /*passwd*/, const char* /*options*/)
{
   // Initializing the RGLite plug-in and making a connection to gLite UI.
   // INPUT:
   //      _gridurl    [in] - must be a "glite" string.
   // NOTE:
   //      The other parameters are unsupported.

   if (!CGLiteAPIWrapper::Instance().Init()) {
      gGrid = this;
      fPort = 0; // Will be used in TGLite::IsConnected
      Info("TGLite", "gLite API Wrapper engine has been successfully initialized.");
   } else {
      // failed to connect to gLite
      fPort = -1;
   }
}


//______________________________________________________________________________
TGLite::~TGLite()
{
   // Destructor
}


//______________________________________________________________________________
Bool_t TGLite::IsConnected() const
{
   // Use this method to find out whether the RGLite plug-in is connected to gLite UI or not.
   // RETURN:
   //      kTRUE if connected and kFALSE otherwise.

   return (-1 == fPort ? kFALSE : kTRUE);
}


//______________________________________________________________________________
void TGLite::Shell()
{
   // Not implemented for RGLite

   MayNotUse("Shell");
}


//______________________________________________________________________________
void TGLite::Stdout()
{
   // Not implemented for RGLite

   MayNotUse("Stdout");
}


//______________________________________________________________________________
void TGLite::Stderr()
{
   // Not implemented for RGLite

   MayNotUse("Stderr");
}


//______________________________________________________________________________
TGridResult* TGLite::Command(const char* /*command*/, Bool_t /*interactive*/, UInt_t /*stream*/)
{
   // Not implemented for RGLite

   MayNotUse("Command");
   return NULL;
}


//______________________________________________________________________________
TGridResult* TGLite::Query(const char *_path, const char *_pattern /*= NULL*/, const char* /*conditions*/, const char* /*options*/)
{
   // A File Catalog method. Querying a File Catalog.
   // INPUT:
   //      _path       [in] - a File Catalog directory which query will be executed on.
   //      _pattern    [in] - a POSIX regular expression pattern.
   //                          If a NULL value provided the default pattern will be used,
   //                          which is ".*" - match any.
   // NOTE:
   //      The third and the forth parameters are unsupported.
   // RETURN:
   //      A TGridResult object, which holds the result of the query.

   if (!_path)
      return NULL; // TODO: msg me!

   // Call for a Catalog manager
   CCatalogManager * pCatalog(&CGLiteAPIWrapper::Instance().GetCatalogManager());
   if (!pCatalog)
      return NULL; // TODO: Log me!

   LFCFileInfoVector_t container;
   gaw_lfc_ls ls;
   ls.m_dir = _path;
   if (_pattern)
      ls.m_pattern = _pattern;
   try {
      pCatalog->Run(ls, &container);
   } catch (const exception &e) {
      Error("Query", "Exception: %s", e.what());
      return NULL;
   }

   // Creating ROOT containers to store the resultset
   TGLiteResult *result = new TGLiteResult();
   for_each(container.begin(), container.end(), bind2nd(SAddMapElementFunc(), result));

   return result;
}


//______________________________________________________________________________
TGridResult* TGLite::LocateSites()
{
   // Not implemented for RGLite

   MayNotUse("LocalSites");
   return NULL;
}

//______________________________________________________________________________
//--- Catalog Interface
TGridResult* TGLite::Ls(const char *_ldn, Option_t* /*options*/, Bool_t /*verbose*/)
{
   // A File Catalog method. Listing content of the current working directory.
   // INPUT:
   //      _ldn    [in] - a logical name of the directory to list.
   // NOTE:
   //      The other parameters are unsupported.
   // RETURN:
   //      A TGridResult object, which holds the result of the listing.
   //      The method returns NULL in case of if an error occurred.

   if (!_ldn)
      return NULL; // TODO: report error

   // Call for a Catalog manager
   CCatalogManager * pCatalog(&CGLiteAPIWrapper::Instance().GetCatalogManager());
   if (!pCatalog)
      return NULL; // TODO: Log me!

   LFCFileInfoVector_t container;
   gaw_lfc_ls ls;
   ls.m_dir = _ldn;
   try {
      pCatalog->Run(ls, &container);
   } catch (const exception &e) {
      Error("Ls", "Exception: %s", e.what());
      return NULL;
   }

   // Creating a ROOT container to store the resultset
   TGLiteResult *result = new TGLiteResult();
   for_each(container.begin(), container.end(), bind2nd(SAddMapElementFunc(), result));

   return result;
}


//______________________________________________________________________________
const char* TGLite::Pwd(Bool_t /*verbose*/)
{
   // A File Catalog method. Retrieving a name of the current working directory.
   // NOTE:
   //      The parameter is unsupported.
   // RETURN:
   //      a logical name of the new current working directory.


   // Call for a Catalog manager
   CCatalogManager *pCatalog(&CGLiteAPIWrapper::Instance().GetCatalogManager());
   if (!pCatalog)
      return NULL; // TODO: Log me!

   gaw_lfc_pwd pwd;
   try {
      pCatalog->Run(pwd, &fFileCatalog_WrkDir);
   } catch (const exception &e) {
      Error("Pwd", "Exception: %s", e.what());
      return NULL;
   }

   return fFileCatalog_WrkDir.c_str();
}


//______________________________________________________________________________
Bool_t TGLite::Cd(const char *_ldn, Bool_t /*verbose*/)
{
   // A File Catalog method. Changing the current working directory.
   // INPUT:
   //      _ldn    [in] - a logical name of the destination directory
   // NOTE:
   //      The other parameter is unsupported.
   // RETURN:
   //      kTRUE if succeeded and kFALSE otherwise.


   if (!_ldn)
      return kFALSE;

   // Call for a Catalog manager
   CCatalogManager * pCatalog(&CGLiteAPIWrapper::Instance().GetCatalogManager());
   if (!pCatalog)
      return kFALSE; // TODO: Log me!

   gaw_lfc_cwd cwd;
   cwd.m_dir = _ldn;
   try {
      pCatalog->Run(cwd);
   } catch (const exception &e) {
      Error("Cd", "Exception: %s", e.what());
      return kFALSE;
   }

   return kTRUE;
}


//______________________________________________________________________________
Int_t TGLite::Mkdir(const char *_ldn, Option_t* /*options*/, Bool_t /*verbose*/)
{
   // A File Catalog method. Create a new directory on the name server.
   // INPUT:
   //      _ldn    [in] - a logical name of the directory to create.
   // NOTE:
   //      The other parameters are unsupported.
   // RETURN:
   //      kTRUE if succeeded and kFALSE otherwise.

   if (!_ldn)
      return kFALSE;

   // Call for a Catalog manager
   // TODO: implement *options* in order to use mkdir with "mode" and "guid"
   CCatalogManager *pCatalog(&CGLiteAPIWrapper::Instance().GetCatalogManager());
   if (!pCatalog)
      return kFALSE; // TODO: Log me!

   gaw_lfc_mkdir mkdir;
   mkdir.m_dir = _ldn;
   try {
      pCatalog->Run(mkdir);
   } catch (const exception &e) {
      Error("Mkdir", "Exception: %s", e.what());
      return kFALSE;
   }

   return kTRUE;
}


//______________________________________________________________________________
Bool_t TGLite::Rmdir(const char *_ldn, Option_t* /*options*/, Bool_t /*verbose*/)
{
   // A File Catalog method, it removes a directory from the name server if it is an empty one.
   // INPUT:
   //      _ldn    [in] - a logical name of the directory to remove.
   // NOTE:
   //      The other parameters are unsupported.
   // RETURN:
   //      kTRUE if succeeded and kFALSE otherwise.

   if (!_ldn)
      return kFALSE;

   // Call for a Catalog manager
   CCatalogManager * pCatalog(&CGLiteAPIWrapper::Instance().GetCatalogManager());
   if (!pCatalog)
      return kFALSE; // TODO: Log me!

   gaw_lfc_rmdir rmdir;
   rmdir.m_dir = _ldn;
   try {
      pCatalog->Run(rmdir);
   } catch (const exception &e) {
      Error("Rmdir", "Exception: %s", e.what());
      return kFALSE;
   }

   return kTRUE;
}


//______________________________________________________________________________
Bool_t TGLite::Register(const char *_lfn, const char *_turl , Long_t /*size*/, const char *_se, const char *_guid, Bool_t /*verbose*/)
{
   // A File Catalog method, it creates a new LFC file in the name server and registering a replication.
   // INPUT:
   //      _lfn    [in] - a logical file name of the file to create.
   //      _turl   [in] - Storage File Name - is either the Site URL or
   //                      the Physical File Name for the replica.
   //      _se     [in] - either the Storage Element fully qualified hostname or the disk server.
   //      _guid   [in] - a GUID for the new file.
   // NOTE:
   //      The other parameters are unsupported.
   // RETURN:
   //      kTRUE if succeeded and kFALSE otherwise.

   // Call for a Catalog manager
   CCatalogManager *pCatalog(&CGLiteAPIWrapper::Instance().GetCatalogManager());
   if (!pCatalog)
      return kFALSE; // TODO: Log me!

   gaw_lfc_register reg;
   reg.m_file_name = _lfn;
   reg.m_guid = _guid;
   reg.m_SE_server = _se;
   reg.m_sfn = _turl;
   try {
      pCatalog->Run(reg);
   } catch (const exception &e) {
      Error("Register", "Exception: %s", e.what());
      return kFALSE;
   }

   return kTRUE;
}


//______________________________________________________________________________
Bool_t TGLite::Rm(const char *_lfn, Option_t* /*option*/, Bool_t /*verbose*/)
{
   // A File Catalog method, it removes an LFC file entry from the name server.
   // The methods deletes all replicas from the file.
   // INPUT:
   //      _lfn    [in] - a logical name of the file to remove.
   // NOTE:
   //      The other parameters are unsupported.
   // RETURN:
   //      kTRUE if succeeded and kFALSE otherwise.

   if (!_lfn)
      return kFALSE;

   // Call for a Catalog manager
   // TODO: Implement "-f" option, which will force to remove all replicas from the file,
   // otherwise file will be deleted only if no replicas exist
   // Currently "-f" is by default. Now Rm deletes all file's replicas and the file itself from catalog namespace.
   CCatalogManager * pCatalog(&CGLiteAPIWrapper::Instance().GetCatalogManager());
   if (!pCatalog)
      return kFALSE; // TODO: Log me!

   gaw_lfc_rm rm;
   rm.m_file_name = _lfn;
   try {
      pCatalog->Run(rm);
   } catch (const exception &e) {
      Error("Rm", "Exception: %s", e.what());
      return kFALSE;
   }

   return kTRUE;
}


//______________________________________________________________________________
//--- Job Submission Interface
TGridJob* TGLite::Submit(const char *_jdl)
{
   // A Grid Job operations method, it processes a job submission.
   // INPUT:
   //      _jdl    [in] - a name of the job description file (JDL). The JDL file path can contain environment variables and a "~" (home) symbol.
   // RETURN:
   //      a TGridJob object, which represents the newly submitted job.
   //      The method returns NULL in case if an error occurred.

   if (!_jdl)
      return NULL; // TODO: msg me!

   try {
      // Call for a job submission
      CGLiteAPIWrapper::Instance().GetJobManager().DelegationCredential();

      string strVer;
      CGLiteAPIWrapper::Instance().GetJobManager().GetVersion(&strVer);
      Info("Submit", "GAW Job Manager uses WMProxy version: %s", strVer.c_str());

      string sJobID;
      CGLiteAPIWrapper::Instance().GetJobManager().JobSubmit(_jdl, &sJobID);
      Info("Submit", "Job successfully submitted. Job ID \"%s\"", sJobID.c_str());
      return dynamic_cast<TGridJob*>(new TGLiteJob(sJobID.c_str()));
   } catch (const exception &e) {
      Error("Submit", "Exception: %s", e.what());
      return NULL;
   }
}


//______________________________________________________________________________
TGridJDL* TGLite::GetJDLGenerator()
{
   // Not implemented for RGLite.

   MayNotUse("GetJDLGenerator");
   return 0;
}


//______________________________________________________________________________
Bool_t  TGLite::Kill(TGridJob *_gridjob)
{
   // A Grid Job operations method, it cancels a given gLite job.
   // INPUT:
   //      _gridjob    [in] - a TGridJob object.
   // RETURN:
   //      kTRUE if succeeded and kFALSE otherwise.

   if (!_gridjob)
      return kFALSE;

   return _gridjob->Cancel();
}


//______________________________________________________________________________
Bool_t  TGLite::KillById(TString _id)
{
   // A Grid Job operations method, it cancels a gLite job by the given id.
   // INPUT:
   //      _id    [in] - a gLite job ID.
   // RETURN:
   //      kTRUE if succeeded and kFALSE otherwise.

   TGLiteJob gridjob(_id);
   return gridjob.Cancel();
}
