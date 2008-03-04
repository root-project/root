/********************************************************************************/
/*                     X T N e t A d m i n _ c i n t f . h h                    */
/*                                    2004                                      */
/*     Produced by Alvise Dorigo & Fabrizio Furano for INFN padova              */
/*                 A C wrapper for XTNetAdmin functionalities                   */
/********************************************************************************/
//
//   $Id$
//
// Author: Alvise Dorigo, Fabrizio Furano


#ifdef SWIG
%module XrdClientAdmin
%include typemaps.i                       // Load the typemaps librayr

 // This tells SWIG to treat an char * argument with name res as
 // an output value.  

%typemap(argout) char *OUTPUT {
   $result = sv_newmortal();
   sv_setnv($result, arg2);
   argvi++;                     /* Increment return count -- important! */
}

// We don't care what the input value is. Ignore, but set to a temporary variable

%typemap(in,numinputs=0) char *OUTPUT(char junk) {
   $1 = &junk;
}

%apply char *OUTPUT { char *ans };

// For the stat function to return an array containing the
// various fields of the answer
%apply long *OUTPUT {long *id};   // Make "result" an output parameter
%apply long long *OUTPUT {long long *size};   // Make "result" an output parameter
%apply long *OUTPUT {long *flags};   // Make "result" an output parameter
%apply long *OUTPUT {long *modtime};   // Make "result" an output parameter

%{
#include "XrdClient/XrdClientAdmin_c.hh"
   %}

#endif

extern "C" {
   // Some prototypes to wrap ctor and dtor
   // In this version we support only one instance to be handled
   // by this wrapper. Supporting more than one instance should be no
   // problem.
   bool XrdInitialize(const char *url, const char *EnvValues);
   bool XrdTerminate();

   // The other functions, slightly modified from the originals
   char *XrdSysStatX(const char *paths_list);

   char *XrdExistFiles(const char *filepaths);
   char *XrdExistDirs(const char *filepaths);
   char *XrdIsFileOnline(const char *filepaths);

   bool XrdMv(const char *fileSrc, const char *fileDest);
   bool XrdMkdir(const char *dir, int user, int group, int other);
   bool XrdChmod(const char *file, int user, int group, int other);
   bool XrdRm(const char *file);
   bool XrdRmdir(const char *path);
   bool XrdPrepare(const char *filepaths, unsigned char opts, unsigned char prty);
   char *XrdDirList(const char *dir);
   char *XrdGetChecksum(const char *path);
   char *XrdGetCurrentHost();

   bool XrdStat(const char *fname, long *id, long long *size, long *flags, long *modtime);
}
