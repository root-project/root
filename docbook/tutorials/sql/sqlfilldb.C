void sqlfilldb(int nfiles = 1000)
{
   // Fill run catalog with nfiles entries
   
   const char *ins = "INSERT INTO runcatalog VALUES ('%s', %d,"
      " %d, %d, %d, %10.2f, '%s', '%s', '1997-01-15 20:16:28',"
      " '1999-01-15 20:16:28', '%s', '%s')";
   
   char sql[4096];
   char dataset[32];
   char rawfile[128];
   int  tag, evt = 0;
   
   // open connection to MySQL server on localhost
   TSQLServer *db = TSQLServer::Connect("mysql://localhost/test", "nobody", "");
   TSQLResult *res;
   
   // first clean table of old entries
   res = db->Query("DELETE FROM runcatalog");
   delete res;

   // start timer
   TStopwatch timer;
   timer.Start();
   
   // fill run catalog
   for (int i = 0; i < nfiles; i++) {
      sprintf(dataset, "testrun_%d", i);
      sprintf(rawfile, "/v1/data/lead/test/run_%d.root", i);
      tag = int(gRandom->Rndm()*10.);
      sprintf(sql, ins, dataset, i, evt, evt+10000, tag, 25.5, "test", "lead",
              rawfile, "test run dummy data");
      evt += 10000;
      res = db->Query(sql);
      delete res;
      //printf("%s\n", sql);
   }
   
   delete db;

   // stop timer and print results
   timer.Stop();
   Double_t rtime = timer.RealTime();
   Double_t ctime = timer.CpuTime();

   printf("\n%d files in run catalog\n", nfiles);
   printf("RealTime=%f seconds, CpuTime=%f seconds\n", rtime, ctime);
}
