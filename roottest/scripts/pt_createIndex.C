#include "TSystem.h"
#include "TList.h"

class TDirectoryIter
{
   void *fDirectory;
public:
   TDirectoryIter(const char *dirname) : fDirectory(gSystem->OpenDirectory(dirname))
   {
      // Usual constructor.
   }
   
   ~TDirectoryIter()
   {
      // Destructor.

      gSystem->FreeDirectory(fDirectory);
      fDirectory = 0;
   }
   
   const char *Next()
   {
      // Return the name of the 
      if (fDirectory) {
         return gSystem->GetDirEntry(fDirectory);
      }
      return 0;
   }
};

/*
 <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
 <html xmlns="http://www.w3.org/1999/xhtml" >
 <head>
 <title>Performance summary for cint/const</title>
 </head>
 <body>
 <h1>Performance summary for cint/const</h1>
 <img src="/icons/back.gif" alt="[DIR]">   <a href="..">Parent Directory</a>
 <br/><img src="/icons/folder.gif" alt="[DIR]"> <a href="array/">array/</a>
 <br/><a href="pt_cintconst705754717.gif"><img src="pt_cintconst705754717.gif" width="200" height="200"/></a>
 <br/><a href="pt_cintconst705754717.root">pt_cintconst705754717.root</a>
 <br/><a href="pt_cintconstrunConst219183205.gif"><img src="pt_cintconstrunConst219183205.gif" width="200" height="200"/></a>
 <br/><a href="pt_cintconstrunConst219183205.root">pt_cintconstrunConst219183205.root</a>
 <br/><a href="pt_cintconstrunConst3440392973.gif"><img src="pt_cintconstrunConst3440392973.gif" width="200" height="200"/></a>
 <br/><a href="pt_cintconstrunConst3440392973.root"pt_cintconstrunConst>3440392973.root</a>
 </body>
 </html>
*/

const char *gPreamble = "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\" \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd\">\n"
    "<html xmlns=\"http://www.w3.org/1999/xhtml\" >\n" 
    "<head>\n"
    "  <title>Performance summary for %s</title>\n"
    "</head>\n"
    "<body>\n" 
    "<h1>Performance summary for %s</h1>\n";

const char *gTitles = "<tr><th align=\"right\"><img src=\"/icons/blank.gif\" alt=\"[ICO]\"></th><th>Name</th><th>Description</th></tr>\n";
const char *gLine = "<tr><th colspan=\"2\"><hr></th></tr>\n";
const char *gParentDir = "<tr><td colspan=\"2\"><table><tr><td><img src=\"/icons/back.gif\" alt=\"[DIR]\"></td><td><a href=\"..\">Parent Directory</a></td></tr></table> </td></tr>\n";
const char *gDirFmt =  "<tr><td colspan=\"2\"><table><tr><td><img src=\"/icons/folder.gif\" alt=\"[DIR]\"></td><td><a href=\"%s\">%s/</a></td></tr></table> </td></tr>\n";

const char *gFiles = "<td><a href=\"%s.gif\"><img src=\"%s.gif\" width=\"200\" height=\"200\"/></a>\n"
    "<br/><a href=\"%s.root\">%s.root</a></td>\n";

void scanDirectory(const char *dirname) 
{
   TDirectoryIter iter(dirname);
   const char *filename = 0;
   TString ent;
   TString file;
   TString html;
   html.Form(gPreamble,dirname,dirname);
   
   TList dirList;
   TList fileList;
 
   while( (filename=iter.Next()) )
   {
      if (filename[0]!='.') {
         ent.Form("%s/%s", dirname, filename);
         FileStat_t st;
         gSystem->GetPathInfo(ent.Data(), st);
         if (R_ISDIR(st.fMode)) {
            //fprintf(stderr,"Seeing directory %s\n",ent.Data());
            scanDirectory(ent.Data());
            dirList.Add(new TObjString(filename));
         } else {
            size_t len = strlen(filename);
            if (len > 8 && strncmp(filename,"pt_",3)==0 && strncmp(filename+len-5,".root",5)==0) {
               //fprintf(stderr,"Seeing file %s\n",ent.Data());
               file = filename;
               file[len-5]='\0';
               fileList.Add(new TObjString(file));
            }
         }
      }
   }
   dirList.Sort();
   fileList.Sort();
   TIter next(&dirList);
   TObjString *obj;
   html += "<table width=\"500\">\n";
   html += gLine;
   html += gParentDir;
   while ( (obj = (TObjString*)next()) ) {
      html += TString::Format(gDirFmt,obj->GetName(),obj->GetName());
   }
   html += gLine;
   
   if (!fileList.IsEmpty()) {

      next = &fileList;
      while ( (obj = (TObjString*)next()) ) {
         html += "<tr>";
         html += TString::Format(gFiles,obj->GetName(),obj->GetName(),obj->GetName(),obj->GetName());
         obj = (TObjString*)next();
         if (obj) {
            html += TString::Format(gFiles,obj->GetName(),obj->GetName(),obj->GetName(),obj->GetName());
         } else {
            html += "<td></td></tr>";
            break;
         }
      }
      html += gLine;
   }
   html += "</table>\n";
   dirList.Delete();
   fileList.Delete();
   html += "</body>\n";
   html += "</html>\n";
   ent.Form("%s/pt_index.html",dirname);
   FILE *output = fopen(ent.Data(),"w");
   fprintf(output,"%s",html.Data());
   fclose(output);
}


void pt_createIndex() {
   scanDirectory(".");
}
