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
#include "TObjString.h"
#include "TSQLMonitoring.h"
#include "TSQLServer.h"
#include "TSQLResult.h"

//______________________________________________________________________________
TSQLMonitoringWriter::TSQLMonitoringWriter(const char *serv, const char *user,
                                           const char *pass, const char *table)
  : TVirtualMonitoringWriter("SQL", 0.0), fTable(table), fVerbose(kFALSE)
{
   // Constructor.

   // Open connection to SQL server
   fDB = TSQLServer::Connect(serv, user, pass);
   if (!fDB || fDB->IsZombie()) {
      SafeDelete(fDB);
      // Invalid object
      MakeZombie();
   }
   // Set the max bulk insertion size
   fMaxBulkSize = 16 * 1024 * 1024;
   TString smx = gEnv->GetValue("SQLMonitoringWriter.MaxBulkSize", "16M");
   if (!smx.IsDigit()) {
      if (smx.EndsWith("K", TString::kIgnoreCase)) {
         smx.Remove(smx.Length()-1);
         if (smx.IsDigit()) fMaxBulkSize = smx.Atoi() * 1024;
      } else if (smx.EndsWith("M", TString::kIgnoreCase)) {
         smx.Remove(smx.Length()-1);
         if (smx.IsDigit()) fMaxBulkSize = smx.Atoi() * 1024 * 1024;
      } else if (smx.EndsWith("G", TString::kIgnoreCase)) {
         smx.Remove(smx.Length()-1);
         if (smx.IsDigit()) fMaxBulkSize = smx.Atoi() * 1024 * 1024 * 1024;
      }
   } else {
      fMaxBulkSize = smx.Atoi();
   }
}

//______________________________________________________________________________
TSQLMonitoringWriter::~TSQLMonitoringWriter()
{
   // Destructor

   SafeDelete(fDB);
}

//______________________________________________________________________________
Bool_t TSQLMonitoringWriter::SendParameters(TList *values, const char *opt)
{
   // Register query log using the information in the list which is in the form
   // TParameter(<par>,<value>) or TNamed(<name>,<string>). For bulk sending,
   // the first entry in the list is an TObjString defining the variable names
   // in the format
   //                    VARname1,VARname2,...
   // while the other entries are TObjStrings with the multiplets to be sent
   //                    VARvalue1,VARvalue2,...
   //
   // The string 'opt' allows the following additional control:
   //      table=[<db>.]<table>  allows to insert to a different table from the
   //                            one defined at construction (change is not
   //                            persistent); if <db> is not specified, the same
   //                            db defined at cinstruction is used.
   //      bulk                  Do a bulk insert
   // More options can be given concurrently, comma-separated .
   // The specified table must already have been created in the DB.

   if (!fDB) {
      // Invalid instance
      return kFALSE;
   }

   // The list must contain something
   if (!values || (values && values->GetSize() < 1))
      return kFALSE;

   // Parse options
   TString table(fTable), op, ops(opt);
   Ssiz_t from = 0;
   Bool_t bulk = kFALSE;
   while (ops.Tokenize(op, from, ",")) {
      if (op == "bulk") {
         bulk = kTRUE;
      } else if (op.BeginsWith("table=")) {
         op.ReplaceAll("table=", "");
         if (!op.IsNull()) {
            Ssiz_t idot = table.Index('.');
            if (idot != kNPOS && op.Index('.') == kNPOS) {
               table.Remove(idot+1);
               table += op;
            } else {
               table = op;
            }
         }
      }
   }

   TIter nxi(values);
   TObject *o = 0;

   // now prepare the strings
   TString sql = TString::Format("INSERT INTO %s", table.Data());

   TSQLResult *res = 0;
   if (!bulk) {

      // the column and values strings
      char c = '(';
      TString cols, vals;
      while ((o = nxi())) {
         if (!strncmp(o->ClassName(), "TNamed", 6)) {
            cols += TString::Format("%c%s", c, ((TNamed *)o)->GetName());
            vals += TString::Format("%c'%s'", c, ((TNamed *)o)->GetTitle());
         } else if (!strcmp(o->ClassName(), "TParameter<Long64_t>")) {
            cols += TString::Format("%c%s", c, ((TParameter<Long64_t> *)o)->GetName());
            vals += TString::Format("%c%lld", c, ((TParameter<Long64_t> *)o)->GetVal());
         } else if (!strcmp(o->ClassName(), "TParameter<double>")) {
            cols += TString::Format("%c%s", c, ((TParameter<double> *)o)->GetName());
            vals += TString::Format("%c%f", c, ((TParameter<double> *)o)->GetVal());
         } else if (!strcmp(o->ClassName(), "TParameter<float>")) {
            cols += TString::Format("%c%s", c, ((TParameter<float> *)o)->GetName());
            vals += TString::Format("%c%f", c, ((TParameter<float> *)o)->GetVal());
         } else if (!strcmp(o->ClassName(), "TParameter<int>")) {
            cols += TString::Format("%c%s", c, ((TParameter<int> *)o)->GetName());
            vals += TString::Format("%c%d", c, ((TParameter<int> *)o)->GetVal());
         } else if (!strcmp(o->ClassName(), "TParameter<long>")) {
            cols += TString::Format("%c%s", c, ((TParameter<long> *)o)->GetName());
            vals += TString::Format("%c%ld", c, ((TParameter<long> *)o)->GetVal());
         }
         c = ',';
      }
      cols += ")";
      vals += ")";

      // Put everything together
      sql += TString::Format(" %s VALUES %s", cols.Data(), vals.Data());

      // Post query
      if (fVerbose) Info("SendParameters", "sending: '%s'", sql.Data());
      if (!(res = fDB->Query(sql))) {
         Error("SendParameters", "insert into %s failed", table.Data());
         if (sql.Length() > 1024) {
            TString head(sql(0,508)), tail(sql(sql.Length()-512,512));
            Printf("%s...%s", head.Data(), tail.Data());
         } else {
            Printf("%s", sql.Data());
         }
         return kFALSE;
      }
      delete res;

   } else {
      // Prepare for bulk submission
      o = nxi();
      TObjString *os = dynamic_cast<TObjString *>(o);
      if (!os) {
         Error("SendParameters", "bulk insert: first entry in list is not 'TObjString' but '%s'", o->ClassName() );
         return kFALSE;
      }
      // Continue preparing the string
      sql += TString::Format(" (%s) VALUES ", os->GetName());
      TString head = sql;
      if (fVerbose) Info("SendParameters", "sending: '%s' (bulk of %d nplets)", head.Data(), values->GetSize() - 1);
      char c = ' ';
      while ((o = nxi())) {
         if ((os = dynamic_cast<TObjString *>(o))) {
           sql += TString::Format("%c(%s)", c, os->GetName());
           c = ',';
         } else {
            Warning("SendParameters", "bulk insert: ignoring not 'TObjString' entry ('%s')", o->ClassName() );
         }
         // Check size (we cannot exceed fMaxBulkSize ('max_allowed_packet' in [mysqld] conf section)
         if (sql.Length() > 0.9 * fMaxBulkSize) {
            if (!(res = fDB->Query(sql))) {
               Error("SendParameters", "bulk insert into %s failed", table.Data());
               if (sql.Length() > 1024) {
                  TString hd(sql(0,508)), tl(sql(sql.Length()-512,512));
                  Printf("%s...%s", hd.Data(), tl.Data());
               } else {
                  Printf("%s", sql.Data());
               }
               return kFALSE;
            }
            delete res;
            sql = head;
            c = ' ';
         }
      }
      // Check if there is still something to send
      if (sql.Length() > head.Length()) {
         if (!(res = fDB->Query(sql))) {
            Error("SendParameters", "bulk insert into %s failed", table.Data());
            if (sql.Length() > 1024) {
               TString hd(sql(0,508)), tl(sql(sql.Length()-512,512));
               Printf("%s...%s", hd.Data(), tl.Data());
            } else {
               Printf("%s", sql.Data());
            }
            return kFALSE;
         }
         delete res;
      }
   }

   // Done successfully
   return kTRUE;
}
