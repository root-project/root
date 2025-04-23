void execscript()
{
   const int   argc       = 2;
   const char *argv[argc] = {"foo", "bar"};
   std::string execscript = "execscript.py";
   TPython::ExecScript(execscript.c_str(), argc, argv);
}
