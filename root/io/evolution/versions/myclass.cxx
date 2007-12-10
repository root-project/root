int compile(const char *what)
{
   //static const TString make( gSystem->GetMakeSharedLib() );
   //TString work( make );
   //work.ReplaceAll("$Opt","$Opt -Dregular");
   //gSystem->SetMakeSharedLib( work );

   static const TString incs( gSystem->GetIncludePath() );
   TString work( incs );
   TString mytype( what );
 
   work.Append( Form(" -DCASE_%s -DVERSION=%s ",what,what) ); 
   gSystem->SetIncludePath( work );
   TString lib( Form("lib%s",what) );
   int r = !gSystem->CompileMacro("myclass.h","k",lib);
   gSystem->Unlink(Form("%s.rootmap",lib.Data()));
}

int wcomp(const char *what)
{
   int r = compile(what);
   if (!r) r = write_what(what);
}