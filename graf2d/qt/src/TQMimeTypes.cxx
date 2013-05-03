// @(#)root/qt:$Id: 98e9b85f582807ea6469888efd07a1ab980ac70a $
// Author: Valeri Fine   21/01/2003
/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * Copyright (C) 2003 by Valeri Fine.                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TQMimeTypes and TQMime                                               //
//                                                                      //
// This class handles mime types, used by browsers to map file types    //
// to applications and icons. TQMime is internally used by TQMimeType.  //
// This class does allow the Qt-base gui to get the same Mime types     //
// as ROOT mime defines                                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TQMimeTypes.h"
#include "TOrdCollection.h"
#include "TSystem.h"
#include "TSystemFile.h"
#include "TDatime.h"
#include "TRegexp.h"

#include "TQtRConfig.h"

#include <QIcon>
#include <QFileIconProvider>
#include <QPixmap>
#include <qfileinfo.h>

ClassImp(TQMimeTypes)
QFileIconProvider  *TQMimeTypes::fgDefaultProvider = 0; // Default provider of the system icons;
//______________________________________________________________________________
QIcon
TQMimeTypes::IconProvider(const QFileInfo &info)
{
    if (!fgDefaultProvider) {
      fgDefaultProvider = new QFileIconProvider;
    }
    return fgDefaultProvider->icon(info);
}

//______________________________________________________________________________
TQMimeTypes::TQMimeTypes(const char *iconPath, const char *filename)
{
   // Create a mime type cache. Read the mime types file "filename" and
   // built a list of mime types.

   char     line[1024];
   char     mime[1024];
   char     pattern[256];
   char     icon[256];
   char     sicon[256];
   char     action[256];
   char    *s;

   fIconPath = iconPath;
   fFilename = filename;
   fChanged  = kFALSE;
   fList     = new TOrdCollection(50);

   FILE *mfp;
   mfp = fopen(filename, "r");
   if (!mfp) {
      Warning("TQMimeTypes", "error opening mime type file %s", filename);
      return;
   }

   int cnt = 0;
   while (fgets(line, 1024, mfp)) {
      s = line;
      s[strlen(line)-1] = 0;       // strip off trailing \n
      while (*s == ' ') s++;       // strip leading blanks
      if (*s == '#') continue;     // skip comments
      if (!s[0]) continue;    // skip empty lines

      if (*s == '[') {
         strlcpy(mime, line,1024);
         cnt = 0;
         continue;
      }
      if (!strncmp(s, "pattern", 7)) {
         if (!(s = strchr(line, '='))) {
            Error("TQMimeTypes", "malformed pattern line, = missing");
            pattern[0] = 0;
         } else {
            s++;
            s = Strip(s);
            strlcpy(pattern, s,256);
            delete [] s;
         }
         cnt++;
      } else if (!strncmp(s, "icon", 4)) {
         if (!(s = strchr(line, '='))) {
            Error("TQMimeTypes", "malformed icon line, = missing");
            icon[0] = 0;
         } else {
            s++;
            s = Strip(s);
            char *s2;
            if ((s2 = strchr(s, ' '))) {
               *s2 = 0;
               strlcpy(icon, s,256);
               s2++;
               s2 = Strip(s2);
               strlcpy(sicon, s2,256);
               delete [] s2;
            } else {
               strlcpy(icon, s,256);
               strlcpy(sicon, s,256);
            }
            delete [] s;
         }
         cnt++;
      } else if (!strncmp(s, "action", 6)) {
         if (!(s = strchr(line, '='))) {
            Error("TQMimeTypes", "malformed action line, = missing");
            action[0] = 0;
         } else {
            s++;
            s = Strip(s);
            strlcpy(action, s,256);
            delete [] s;
         }
         cnt++;
      }

      if (cnt == 3) {
         if (strchr(pattern, ' ')) {
            char *tmppattern = strtok(pattern, " ");
            while (tmppattern && (*tmppattern != ' ')) {
               AddType(mime, tmppattern, icon, sicon, action);
               tmppattern = strtok(0, " ");
            }
         } else {
            AddType(mime, pattern, icon, sicon, action);
         }
      }
   }

   fclose(mfp);

   fChanged = kFALSE;
}

//______________________________________________________________________________
TQMimeTypes::~TQMimeTypes()
{
   // Delete mime type pool.

   if (fChanged) SaveMimes();
   fList->Delete();
   delete fList;
}

//______________________________________________________________________________
TQMime *TQMimeTypes::Find(const char *filename) const
{
   // Given a filename find the matching mime type object.

   if (!filename) return 0;

   TString fn = filename;

   TQMime  *mime;
   TIter    next(fList);

   while ((mime = (TQMime *) next()))
      if (fn.Index(*(mime->fReg)) != kNPOS) return mime;

   return 0;
}

//______________________________________________________________________________
const QIcon *TQMimeTypes::GetIcon(const char *filename) const
{
   // Return icon belonging to mime type of filename.
   TQMime *mime= Find(filename);
   if (mime)  return mime->fIcon;
   return 0;
}
//______________________________________________________________________________
const QIcon *TQMimeTypes::GetIcon(const TSystemFile *filename)
{
   // Return icon belonging to mime type of TSystemFile extension
   const char *name = filename->GetName();
   const QIcon *set = GetIcon(name);
   if (!set) set = AddType(filename);
   return set;
}
//______________________________________________________________________________
const QIcon *TQMimeTypes::AddType(const TSystemFile *filename)
{
   //

   QFileInfo info(filename->GetName());
   const QIcon    icon = IconProvider(info);
   if (icon.isNull()) return 0;

   // Add an artificial mime type to the list of mime types from the default system
   TQMime *mime = new TQMime;
   mime->fType    = "system/file";
   mime->fPattern = "*.";
   mime->fPattern += info.suffix().toAscii().data();
   mime->fIcon  = 0;
   mime->fIcon  = new QIcon(icon) ;
#ifdef R__QTWIN32
   mime->fAction  = "!%s";
#else
   mime->fAction  = "";
#endif

   mime->fReg = new TRegexp(mime->fPattern, kTRUE);

   fList->Add(mime);

   fChanged = kTRUE;
   return mime->fIcon;
}
//______________________________________________________________________________
Bool_t TQMimeTypes::GetAction(const char *filename, char *action) const
{
   // Return in action the mime action string belonging to filename.

   TQMime *mime;

   action[0] = 0;
   if ((mime = Find(filename))) {
      strcpy(action, mime->fAction.Data());
      return (strlen(action) > 0);
   }
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TQMimeTypes::GetType(const char *filename, char *type) const
{
   // Return in type the mime type belonging to filename.

   TQMime *mime;

   memset(type, 0, strlen(type));
   if ((mime = Find(filename))) {
      strcpy(type, mime->fType.Data());
      return (strlen(type) > 0);
   }
   return kFALSE;
}

//______________________________________________________________________________
void TQMimeTypes::Print(Option_t *) const
{
   // Print list of mime types.

   TQMime *m;
   TIter next(fList);

   while ((m = (TQMime *) next())) {
      printf("Type:    %s\n", m->fType.Data());
      printf("Pattern: %s\n", m->fPattern.Data());
      printf("Icon:    %p\n", m->fIcon);
      printf("Action:  %s\n", m->fAction.Data());
      printf("------------\n\n");
   }
}

//______________________________________________________________________________
void TQMimeTypes::SaveMimes()
{
   // Save mime types in user's mime type file.

   char filename[1024];
   snprintf(filename,1024, "%s/.root.mimes",  gSystem->HomeDirectory());

   FILE *fp = fopen(filename, "w");

   if (!fp) {
      Error("SaveMimes", "can not open %s to store mime types", filename);
      return;
   }

   TDatime dt;
   fprintf(fp, "# %s written on %s\n\n", filename, dt.AsString());

   TQMime *m;
   TIter next(fList);

   while ((m = (TQMime *) next())) {
      fprintf(fp, "%s\n",            m->fType.Data());
      fprintf(fp, "pattern = %s\n",  m->fPattern.Data());
      fprintf(fp, "icon = %p\n",    m->fIcon);
      fprintf(fp, "action = %s\n\n", m->fAction.Data());
   }

   fclose(fp);

   fChanged = kFALSE;
}

//______________________________________________________________________________
void TQMimeTypes::AddType(const char *type, const char *pattern, const char *icon,
                          const char * /*sicon*/, const char *action)
{
   // Add a mime type to the list of mime types.

   TQMime *mime = new TQMime;

   mime->fType    = type;
   mime->fPattern = pattern;
   mime->fIcon    = 0;
   char *picnam = gSystem->Which(fIconPath.Data(),icon, kReadPermission);
   if (picnam) mime->fIcon  = new QIcon( QPixmap(picnam) ) ;
   delete [] picnam;
   mime->fAction  = action;

   mime->fReg = new TRegexp(pattern, kTRUE);

   fList->Add(mime);

   fChanged = kTRUE;
}

//______________________________________________________________________________
TQMime::~TQMime()
{
   // Delete mime object.
   delete fIcon; fIcon = 0;
   delete fReg;
}
