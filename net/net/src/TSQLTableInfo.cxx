// @(#)root/net:$Id$
// Author: Sergey Linev   31/05/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////////////
//
// TSQLTableInfo
//
// Contains information about table and table columns.
// For MySQL additional information like engine type,
// creation and last update time is provided
//
////////////////////////////////////////////////////////////////////////////////



#include "TSQLTableInfo.h"

#include "TSQLColumnInfo.h"
#include "TList.h"
#include "TROOT.h"
#include "Riostream.h"

ClassImp(TSQLTableInfo);

////////////////////////////////////////////////////////////////////////////////
/// default constructor

TSQLTableInfo::TSQLTableInfo() :
   TNamed(),
   fColumns(0),
   fEngine(),
   fCreateTime(),
   fUpdateTime()
{
}

////////////////////////////////////////////////////////////////////////////////
/// normal constructor

TSQLTableInfo::TSQLTableInfo(const char* tablename,
                             TList* columns,
                             const char* comment,
                             const char* engine,
                             const char* create_time,
                             const char* update_time) :
   TNamed(tablename, comment),
   fColumns(columns),
   fEngine(engine),
   fCreateTime(create_time),
   fUpdateTime(update_time)
{
}

////////////////////////////////////////////////////////////////////////////////
/// destructor

TSQLTableInfo::~TSQLTableInfo()
{
   if (fColumns!=0) {
      fColumns->Delete();
      delete fColumns;
      fColumns = 0;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Prints table and table columns info

void TSQLTableInfo::Print(Option_t*) const
{
   TROOT::IndentLevel();
   std::cout << "Table:" << GetName();

   if ((GetTitle()!=0) && (strlen(GetTitle())!=0))
      std::cout << " comm:'" << GetTitle() << "'";

   if (fEngine.Length()>0)
      std::cout << " engine:" << fEngine;

   if (fCreateTime.Length()>0)
      std::cout << " create:" << fCreateTime;

   if (fUpdateTime.Length()>0)
      std::cout << " update:" << fUpdateTime;

   std::cout << std::endl;

   TROOT::IncreaseDirLevel();
   if (fColumns!=0)
      fColumns->Print("*");
   TROOT::DecreaseDirLevel();
}

////////////////////////////////////////////////////////////////////////////////
/// Return column info object of given name

TSQLColumnInfo* TSQLTableInfo::FindColumn(const char* columnname)
{
   if ((columnname==0) || (fColumns==0)) return 0;

   return dynamic_cast<TSQLColumnInfo*> (fColumns->FindObject(columnname));

}
