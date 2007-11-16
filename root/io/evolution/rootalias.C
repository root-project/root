int compile(const char *what)
{
   //static const TString make( gSystem->GetMakeSharedLib() );
   //TString work( make );
   //work.ReplaceAll("$Opt","$Opt -Dregular");
   //gSystem->SetMakeSharedLib( work );

   static const TString incs( gSystem->GetIncludePath() );
   TString work( incs );
   TString mytype( what );
   mytype.ReplaceAll("longlong","long long");
   if (mytype[0] == 'u') {
      mytype = Form("unsigned %s",mytype.Data()+1);
   }
   work.Append( Form(" -DCASE_%s \"-DMYTYPE=%s\" -DWHAT=\\\"%s\\\" ",what,mytype.Data(),what) ); 
   gSystem->SetIncludePath( work );
   TString lib( Form("lib%s",what) );
   return !gSystem->CompileMacro("float16.cxx","k",lib);
}

void run(const char *what) {
   compile(what);
   write();
}