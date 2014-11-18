#ifndef __XRDCKSCALC_HH__
#define __XRDCKSCALC_HH__
/******************************************************************************/
/*                                                                            */
/*                         X r d C k s C a l c . h h                          */
/*                                                                            */
/* (c) 2011 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

// This class defines the interface to a checksum computation. When this class
// is used to define a plugin computation, the initial XrdCksCalc computation object
// is created by the XrdCksCalcInit() function defined at the end of this file.
  
class XrdCksCalc
{
public:

// Calc()     Calculates a one-time checksum. The obvious default implementation
//            is provided and assumes that Init() may be called more than once.
//
virtual char *Calc(const char *Buff, int BLen)
                  {Init(); Update(Buff, BLen); return Final();}

// Current()  returns the current binary checksum value (defaults to final).
//            The final checksum result is not affected.
//
virtual char *Current() {return Final();}

// Final()    Returns the actual checksum in binary format.
//
virtual char *Final() = 0;

// Init()     Initializes data structures (must be called by constructor). This
//            is always called to reuse the object for a new checksum.
//
virtual void  Init() = 0;

// New()      Must provide a new instance of the underlying object.
//
virtual
XrdCksCalc   *New() = 0;

// Recycle()  Is called when the object is no longer needed. A default is given.
//
virtual void  Recycle() {delete this;}

// Type()     returns the character name of the checksum object and the number
//            bytes (i.e. size) required for the checksum value.
//
virtual const char *Type(int &csSize) = 0;

// Update()   computes a running checksum and may be called repeatedly for
//            data segments; with Final() returning the full checksum.
//
virtual void  Update(const char *Buff, int BLen) = 0;

              XrdCksCalc() {}
virtual      ~XrdCksCalc() {}
};

/******************************************************************************/
/*               C h e c k s u m   O b j e c t   C r e a t o r                */
/******************************************************************************/
  
/* When building a shared library plugin, the following "C" entry point must
   exist in the library:

   extern "C"
   {XrdCksCalc *XrdCksCalcInit(XrdSysError *eDest,  // The error msg object
                               const char  *csName, // Name of checksum
                               const char  *cFN,    // Config file name
                               const char  *Parms); // Parms on lib directive
   }

   This entry is called to get an instance of the checksum object which must
   match the passed checksum name. If the object cannot be created; return 0.
*/
#endif
