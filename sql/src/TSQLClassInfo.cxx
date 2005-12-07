// @(#)root/sql:$Name:  $:$Id: TSQLClassInfo.cxx,v 1.2 2005/11/22 20:42:36 pcanal Exp $
// Author: Sergey Linev  20/11/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//________________________________________________________________________
// 
// TSQLClassInfo class containes info about tables specific to one class and
// version. It provides names of table for that class. For each version of 
// class not more than two tables can exists. Normal table has typically
// name like TH1_ver4 and additional table has name like TH1_streamer_ver4
// List of this objects are kept by TSQLFile class
//
//________________________________________________________________________

#include "TSQLClassInfo.h"

#include "TObjArray.h"

ClassImp(TSQLClassInfo)

//______________________________________________________________________________
TSQLClassInfo::TSQLClassInfo() :
   TObject(),
   fClassName(),
   fClassVersion(0),
   fClassTable(),
   fRawTable(),
   fColumns(0),
   fRawtableExist(kFALSE)
{
// default constructor
}
  
//______________________________________________________________________________
TSQLClassInfo::TSQLClassInfo(const char* classname, Int_t version) : 
   TObject(),
   fClassName(classname),
   fClassVersion(version),
   fClassTable(),
   fRawTable(),
   fColumns(0),
   fRawtableExist(kFALSE)
{
// normal constructor of TSQLClassInfo class
// Sets names of tables, which are used for that version of class    
   fClassTable.Form("%s_ver%d", classname, version);
   fRawTable.Form("%s_streamer_ver%d", classname, version);
}
   
//______________________________________________________________________________
TSQLClassInfo::~TSQLClassInfo()
{
// destructor

   if (fColumns!=0) {
      fColumns->Delete();  
      delete fColumns; 
   }
   
}

//______________________________________________________________________________
void TSQLClassInfo::SetColumns(TObjArray* columns)
{
// assigns new list of columns
    
   if (fColumns!=0) {
      fColumns->Delete();  
      delete fColumns; 
   }
   fColumns = columns;
}

//______________________________________________________________________________
void TSQLClassInfo::SetTableStatus(TObjArray* columns, Bool_t israwtable)
{
// set current status of class tables
    
   SetColumns(columns); 
   fRawtableExist = israwtable;
}
