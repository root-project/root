/// \file
/// \ingroup tutorial_legacy
/// Defines aliases:
///   - `ls(path)`
///   - `edit(filename)`
///   - `dir(path)`
///   - `pwd()`
///   - `cd(path)`
///
/// \macro_code
///
/// \author Rene Brun

//______________________________________________________________________________
void edit(char *file)
{
   char s[64], *e;
   if (!strcmp(gSystem->GetName(), "WinNT")) {
      if ((e = getenv("EDITOR")))
         sprintf(s, "start %s %s", e, file);
      else
         sprintf(s, "start notepad %s", file);
   } else {
      if ((e = getenv("EDITOR")))
         sprintf(s, "%s %s", e, file);
      else
         sprintf(s, "xterm -e vi %s &", file);
   }
   gSystem->Exec(s);
}

//______________________________________________________________________________
void ls(char *path=0)
{
   char s[256];
   strcpy(s, (!strcmp(gSystem->GetName(), "WinNT")) ? "dir /w " : "ls ");
   if (path) strcat(s,path);
   gSystem->Exec(s);
}

//______________________________________________________________________________
void dir(char *path=0)
{
   char s[256];
   strcpy(s,(!strcmp(gSystem->GetName(), "WinNT")) ? "dir " : "ls -l ");
   if (path) strcat(s,path);
   gSystem->Exec(s);
}

//______________________________________________________________________________
const char *pwd()
{
    return gSystem->WorkingDirectory();
}

//______________________________________________________________________________
const char *cd(char *path=0)
{
 if (path)
   gSystem->ChangeDirectory(path);
 return pwd();
}

TCanvas *bench = 0;
//______________________________________________________________________________
void bexec2(char *macro)
{
   printf("in bexec dir=%s\n",pwd());
   if (gROOT->IsBatch()) printf("Processing benchmark: %s\n",macro);
   TPaveText *summary = (TPaveText*)bench->GetPrimitive("TPave");
   TText *tmacro = summary->GetLineWith(macro);
   if (tmacro) tmacro->SetTextColor(4);
   bench->Modified(); bench->Update();

   gROOT->Macro(macro);

   TPaveText *summary2 = (TPaveText*)bench->GetPrimitive("TPave");
   TText *tmacro2 = summary2->GetLineWith(macro);
   if (tmacro2) tmacro2->SetTextColor(2);
   bench->Modified(); bench->Update();
}
