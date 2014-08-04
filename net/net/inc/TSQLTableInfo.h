// @(#)root/net:$Id$
// Author: Sergey Linev   31/05/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSQLTableInfo
#define ROOT_TSQLTableInfo

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

class TList;
class TSQLColumnInfo;

class TSQLTableInfo : public TNamed {

protected:
   TList*    fColumns;    //! list of TSQLColumnInfo objects, describing each table column
   TString   fEngine;     //! SQL tables engine name
   TString   fCreateTime; //! table creation time
   TString   fUpdateTime; //! table update time

public:
   TSQLTableInfo();
   TSQLTableInfo(const char* tablename,
                 TList* columns,
                 const char* comment = "SQL table",
                 const char* engine = 0,
                 const char* create_time = 0,
                 const char* update_time = 0);
   virtual ~TSQLTableInfo();

   virtual void Print(Option_t* option = "") const;

   TList* GetColumns() const { return fColumns; }

   TSQLColumnInfo* FindColumn(const char* columnname);

   const char* GetEngine()     const { return fEngine.Data(); }
   const char* GetCreateTime() const { return fCreateTime.Data(); }
   const char* GetUpdateTime() const { return fUpdateTime.Data(); }

   ClassDef(TSQLTableInfo, 0) // Summury information about SQL table
};

#endif
