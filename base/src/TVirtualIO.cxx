// @(#)root/base:$Name:  $:$Id: TVirtualIO.cxx,v 1.2 2007/01/25 14:31:27 brun Exp $
// Author: Rene Brun   24/01/2007
/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      
// TVirtualIO                                                       
//                                                                     
// TVirtualIO is an interface class for File I/O operations that cannot
// be performed via TDirectory.
// The class is used to decouple the base classes from the I/O packages.
// The concrete I/O sub-system is dynamically linked by the PluginManager.
// The default implementation TFileIO can be changed in system.rootrc.
//                                     
//////////////////////////////////////////////////////////////////////////

#include "TVirtualIO.h"
#include "TROOT.h"
#include "TPluginManager.h"
#include "TEnv.h"


TVirtualIO *TVirtualIO::fgIO      = 0;
TString     TVirtualIO::fgDefault = "FileIO";

ClassImp(TVirtualIO)

//______________________________________________________________________________
TVirtualIO::TVirtualIO()
{
   // Default constructor.
}

//______________________________________________________________________________
TVirtualIO::TVirtualIO(const TVirtualIO& io) : TObject(io) 
{ 
   //copy constructor
}

//______________________________________________________________________________
TVirtualIO& TVirtualIO::operator=(const TVirtualIO& io)
{
   //assignment operator
   if(this!=&io) {
      TObject::operator=(io);
   } 
   return *this;
}

//______________________________________________________________________________
TVirtualIO::~TVirtualIO()
{
   // destructor

   //delete fgIO;  do not delete these statics
   //fgIO = 0;
}

//______________________________________________________________________________
TVirtualIO *TVirtualIO::GetIO()
{
   // Static function returning a pointer to the current I/O class.
   // If the class does not exist, the default TFileIO is created.

   if (!fgIO) {
      TPluginHandler *h;
      if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualIO",fgDefault))) {
         if (h->LoadPlugin() == -1)
            return 0;
         fgIO = (TVirtualIO*) h->ExecPlugin(0);
      }
   }

   return fgIO;
}

//______________________________________________________________________________
const char *TVirtualIO::GetDefaultIO()
{
   // static: return the name of the default I/O class

   return fgDefault.Data();
}

//______________________________________________________________________________
void TVirtualIO::SetDefaultIO(const char *name)
{
   // static: set name of default I/O system

   if (fgDefault == name) return;
   delete fgIO;
   fgIO = 0;
   fgDefault = name;
}
