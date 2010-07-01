// @(#)root/proofplayer:$Id$
// Author: J.F. Grosse-Oetringhaus, G.Ganis

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSQLMonitoringWriter                                                 //
//                                                                      //
// SQL implementation of TVirtualMonitoringWriter.                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TList.h"
#include "TParameter.h"
#include "TEnv.h"
#include "TSQLMonitoring.h"
#include "TSQLServer.h"
#include "TSQLResult.h"

//______________________________________________________________________________
TSQLMonitoringWriter::TSQLMonitoringWriter(const char *serv, const char *user,
                                           const char *pass, const char *table)
   : TVirtualMonitoringWriter("SQL", 0.0), fTable(table)
{
   // Constructor.

   // Open connection to SQL server
   fDB = TSQLServer::Connect(serv, user, pass);
   if (!fDB || fDB->IsZombie()) {
      SafeDelete(fDB);
      // Invalid object
      MakeZombie();
   }
}

//______________________________________________________________________________
TSQLMonitoringWriter::~TSQLMonitoringWriter()
{
   // Destructor

   SafeDelete(fDB);
}

//______________________________________________________________________________
Bool_t TSQLMonitoringWriter::SendParameters(TList *values, const char *)
{
   // Register query log using the information in the list which is in the form
   // TParameter(<par>,<value>) or TNamed(<name>,<string>).
   // The first element in the list is a TNamed object called TABLE with the
   // table name in the title field. Of course the specified table must already
   // have been created in the DB.

   if (!fDB) {
      // Invalid instance
      return kFALSE;
   }

   // the list must contain something
   if (!values || values->GetSize() <= 1)
      return kFALSE;

   TIter nxi(values);

   // now prepare the strings
   TString sql = Form("INSERT INTO %s", fTable.Data());

   // the column and values strings
   TObject *o = 0;
   char c = '(';
   TString cols, vals;
   while ((o = nxi())) {
      if (!strncmp(o->ClassName(), "TNamed", 6)) {
         cols += Form("%c'%s'", c, ((TNamed *)o)->GetName());
         vals += Form("%c'%s'", c, ((TNamed *)o)->GetTitle());
      } else if (!strcmp(o->ClassName(), "TParameter<Long64_t>")) {
         cols += Form("%c'%s'", c, ((TParameter<Long64_t> *)o)->GetName());
         vals += Form("%c%lld", c, ((TParameter<Long64_t> *)o)->GetVal());
      } else if (!strcmp(o->ClassName(), "TParameter<double>")) {
         cols += Form("%c'%s'", c, ((TParameter<double> *)o)->GetName());
         vals += Form("%c%f", c, ((TParameter<double> *)o)->GetVal());
      } else if (!strcmp(o->ClassName(), "TParameter<float>")) {
         cols += Form("%c'%s'", c, ((TParameter<float> *)o)->GetName());
         vals += Form("%c%f", c, ((TParameter<float> *)o)->GetVal());
      } else if (!strcmp(o->ClassName(), "TParameter<int>")) {
         cols += Form("%c'%s'", c, ((TParameter<int> *)o)->GetName());
         vals += Form("%c%d", c, ((TParameter<int> *)o)->GetVal());
      } else if (!strcmp(o->ClassName(), "TParameter<long>")) {
         cols += Form("%c'%s'", c, ((TParameter<long> *)o)->GetName());
         vals += Form("%c%ld", c, ((TParameter<long> *)o)->GetVal());
      }
      c = ',';
   }
   cols += ")";
   vals += ")";

   // Put everything together
   sql += Form(" %s VALUES %s", cols.Data(), vals.Data());

   // Post query
   TSQLResult *res = fDB->Query(sql);
   if (!res) {
      Error("SendParameters", "insert into %s failed", fTable.Data());
      printf("%s\n", sql.Data());
      return kFALSE;
   }
   delete res;

   // Done successfully
   return kTRUE;
}
