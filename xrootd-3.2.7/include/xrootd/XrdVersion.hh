// this file is automatically updated by the genversion.sh script
// if you touch anything make sure that it still works

#ifndef __XRD_VERSION_H__
#define __XRD_VERSION_H__

#define XrdVERSION  "v3.2.7"

// Numeric representation of the version tag
// The format for the released code is: xyyzz, where: x is the major version,
// y is the minor version and zz is the bugfix revision number
// For the non-released code the value is 1000000
#define XrdVNUMBER  30207

#if XrdDEBUG
#define XrdVSTRING XrdVERSION "_dbg"
#else
#define XrdVSTRING XrdVERSION
#endif

#define XrdDEFAULTPORT 1094;

#endif
