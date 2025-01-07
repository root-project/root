int hadd_args_verify(const char *fname)
{
   std::unique_ptr<TFile> file { TFile::Open(fname, "READ") };
   auto *tree = file->Get<TTree>("t");
   if (!tree)
     return 1;

   auto *branch = tree->GetBranch("x");
   return !branch;
}
