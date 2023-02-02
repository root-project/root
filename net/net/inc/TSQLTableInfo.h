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

#include "TNamed.h"

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
                 const char* engine = nullptr,
                 const char* create_time = nullptr,
                 const char* update_time = nullptr);
   virtual ~TSQLTableInfo();

   void Print(Option_t* option = "") const override;

   TList* GetColumns() const { return fColumns; }

   TSQLColumnInfo* FindColumn(const char* columnname);

   const char* GetEngine()     const { return fEngine.Data(); }
   const char* GetCreateTime() const { return fCreateTime.Data(); }
   const char* GetUpdateTime() const { return fUpdateTime.Data(); }

   ClassDefOverride(TSQLTableInfo, 0) // Summary information about SQL table
};

#endif
