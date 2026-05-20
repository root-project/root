{
   const char *file = "tref_test_pid.root";
   const char *file_r = "tref_test_pid_r.root";
   const char *file_rc = "tref_test_pid_rc.root";
   const char *file_rcr = "tref_test_pid_rcr.root";
   const char *file_rcrr = "tref_test_pid_rcrr.root";

   std::string s = STAGE;

   if(s == "first") {
      create(file);
   } else if (s == "create-r") {
      addref(file, file_r, "r1", "N");
   } else if (s == "test-r") {
      readauto(file_r, "r1");
   } else if (s == "create-rc") {
      clone(file_r, file_rc);
   } else if (s == "test-rc") {
      readauto(file_rc, "r1");
      readall(file_rc);
   } else if (s == "create-rcr") {
      addref(file_rc, file_rcr, "r2", "r1");
   } else if (s == "test-rcr") {
      readauto(file_rcr, "r1");
      readall(file_rcr);
   } else if (s == "test-rcr2") {
      readauto(file_rcr, "r2");
      readall(file_rcr);
   } else if (s == "create-rcrr") {
      addref(file_rcr, file_rcrr, "r3", "N");
   } else if (s == "test-rcrr") {
      readauto(file_rcrr, "r1");
      readall(file_rcrr);
   } else if (s == "test-rcrr2") {
      readauto(file_rcrr, "r2");
      readall(file_rcrr);
   } else if (s == "test-rcrr3") {
      readauto(file_rcrr, "r3");
      readall(file_rcrr);
   } else if (s == "test-rcrr123") {
      readauto(file_rcrr, "r1:r2:r3");
   }



}
