{
//fprintf(stdout,"STL I/O test\n");
enum EWhich { 
   kVector = 1,
   kDeque = 2,
   kList = 4,
   kSet = 8,
   kMultiSet = 16,
   kMap = 32,
   kMultiMap = 64,
   kEnd = 128
};

int which = kEnd - 1;
int exec = 0;
int pass = 0;

TString dirname = gROOT->GetVersion();
dirname.ReplaceAll(".","-");
dirname.ReplaceAll("/","-");
dirname.Append(".libs");
gSystem->SetBuildDir(dirname);

}
