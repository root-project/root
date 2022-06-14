/// \file
/// \ingroup tutorial_sql
/// Query example to MySQL test database.
/// Example of query by using the test database made in MySQL, you need the
/// database test installed in localhost, with user nobody without password.
///
/// \macro_code
///
/// \author Sergey Linev, Juan Fernando Jaramillo Botero

#include <TSQLServer.h>
#include <TSQLResult.h>
#include <TSQLRow.h>


void sqlselect()
{
   TSQLServer *db = TSQLServer::Connect("mysql://localhost/test","nobody", "");

   printf("Server info: %s\n", db->ServerInfo());

   TSQLRow *row;
   TSQLResult *res;

   // list databases available on server
   printf("\nList all databases on server %s\n", db->GetHost());
   res = db->GetDataBases();
   while ((row = res->Next())) {
      printf("%s\n", row->GetField(0));
      delete row;
   }
   delete res;

   // list tables in database "test" (the permission tables)
   printf("\nList all tables in database \"test\" on server %s\n",
          db->GetHost());
   res = db->GetTables("test");
   while ((row = res->Next())) {
      printf("%s\n", row->GetField(0));
      delete row;
   }
   delete res;

   // list columns in table "runcatalog" in database "mysql"
   printf("\nList all columns in table \"runcatalog\" in database \"test\" on server %s\n",
          db->GetHost());
   res = db->GetColumns("test", "runcatalog");
   while ((row = res->Next())) {
      printf("%s\n", row->GetField(0));
      delete row;
   }
   delete res;

   // start timer
   TStopwatch timer;
   timer.Start();

   // query database and print results
   const char *sql = "select dataset,rawfilepath from test.runcatalog "
                     "WHERE tag&(1<<2) AND (run=490001 OR run=300122)";
   // const char *sql = "select count(*) from test.runcatalog "
   //                   "WHERE tag&(1<<2)";

   res = db->Query(sql);

   int nrows = res->GetRowCount();
   printf("\nGot %d rows in result\n", nrows);

   int nfields = res->GetFieldCount();
   for (int i = 0; i < nfields; i++)
      printf("%40s", res->GetFieldName(i));
   printf("\n");
   for (int i = 0; i < nfields*40; i++)
      printf("=");
   printf("\n");

   for (int i = 0; i < nrows; i++) {
      row = res->Next();
      for (int j = 0; j < nfields; j++) {
         printf("%40s", row->GetField(j));
      }
      printf("\n");
      delete row;
   }

   delete res;
   delete db;

   // stop timer and print results
   timer.Stop();
   Double_t rtime = timer.RealTime();
   Double_t ctime = timer.CpuTime();

   printf("\nRealTime=%f seconds, CpuTime=%f seconds\n", rtime, ctime);
}
