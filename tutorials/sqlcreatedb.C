void sqlcreatedb()
{
   // Create a runcatalog table in a MySQL test database.
   
   // read in runcatalog table definition
   FILE *fp = fopen("runcatalog.sql", "r");
   const char sql[4096];
   fread(sql, 1, 4096, fp);
   fclose(fp);
   
   // open connection to MySQL server on localhost
   TSQLServer *db = TSQLServer::Connect("mysql://localhost/test", "nobody", "");
   
   TSQLResult *res;

   // create new table (delete old one first if exists)
   res = db->Query("DROP TABLE runcatalog");
   delete res;
   
   res = db->Query(sql);
   delete res;
   
   delete db;
}
