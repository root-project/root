int customRootrcPath()
{

   if (!gSystem->Getenv("ROOTENV_USER_PATH")) {
      cerr << "Error: env variable 'ROOTENV_USER_PATH' cannot be found." << endl;
      return 1;
   }

   if (1 != gEnv->GetValue("customVal.customVal", (int)-1)) {
      cerr << "Error: variable customVal is not 1" << endl;
      return 2;
   }
   return 0;
}
