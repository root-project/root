// @(#)root/gui:$Name:  $:$Id: TGMimeTypes.cxx,v 1.1.1.1 2000/05/16 17:00:42 rdm Exp $
// Author: Fons Rademakers   18/01/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
/**************************************************************************

    This source is based on Xclass95, a Win95-looking GUI toolkit.
    Copyright (C) 1996, 1997 David Barth, Ricky Ralston, Hector Peraza.

    Xclass95 is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

**************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGMimeTypes and TGMime                                               //
//                                                                      //
// This class handles mime types, used by browsers to map file types    //
// to applications and icons. TGMime is internally used by TGMimeType.  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGMimeTypes.h"
#include "TOrdCollection.h"
#include "TSystem.h"
#include "TDatime.h"
#include "TRegexp.h"


ClassImp(TGMimeTypes)

//______________________________________________________________________________
TGMimeTypes::TGMimeTypes(TGClient *client, const char *filename)
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

   fClient   = client;
   fFilename = filename;
   fChanged  = kFALSE;
   fList     = new TOrdCollection(50);

   FILE *mfp;
   mfp = fopen(filename, "r");
   if (!mfp) {
      Warning("TGMimeTypes", "error opening mime type file %s", filename);
      return;
   }

   int cnt = 0;
   while (fgets(line, 1024, mfp)) {
      s = line;
      s[strlen(line)-1] = 0;       // strip off trailing \n
      while (*s == ' ') s++;       // strip leading blanks
      if (*s == '#') continue;     // skip comments
      if (!strlen(s)) continue;    // skip empty lines

      if (*s == '[') {
         strcpy(mime, line);
         cnt = 0;
         continue;
      }
      if (!strncmp(s, "pattern", 7)) {
         if (!(s = strchr(line, '='))) {
            Error("TGMimeTypes", "malformed pattern line, = missing");
            pattern[0] = 0;
         } else {
            s++;
            s = Strip(s);
            strcpy(pattern, s);
            delete [] s;
         }
         cnt++;
      } else if (!strncmp(s, "icon", 4)) {
         if (!(s = strchr(line, '='))) {
            Error("TGMimeTypes", "malformed icon line, = missing");
            icon[0] = 0;
         } else {
            s++;
            s = Strip(s);
            char *s2;
            if ((s2 = strchr(s, ' '))) {
               *s2 = 0;
               strcpy(icon, s);
               s2++;
               s2 = Strip(s2);
               strcpy(sicon, s2);
               delete [] s2;
            } else {
               strcpy(icon, s);
               strcpy(sicon, s);
            }
            delete [] s;
         }
         cnt++;
      } else if (!strncmp(s, "action", 6)) {
         if (!(s = strchr(line, '='))) {
            Error("TGMimeTypes", "malformed action line, = missing");
            action[0] = 0;
         } else {
            s++;
            s = Strip(s);
            strcpy(action, s);
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
TGMimeTypes::~TGMimeTypes()
{
   // Delete mime type pool.

   if (fChanged) SaveMimes();
   fList->Delete();
   delete fList;
}

//______________________________________________________________________________
TGMime *TGMimeTypes::Find(const char *filename)
{
   // Given a filename find the matching mime type object.

   if (!filename) return 0;

   TString fn = filename;

   TGMime  *mime;
   TIter    next(fList);

   while ((mime = (TGMime *) next()))
      if (fn.Index(*(mime->fReg)) != kNPOS) return mime;

   return 0;
}

//______________________________________________________________________________
const TGPicture *TGMimeTypes::GetIcon(const char *filename, Bool_t small_icon)
{
   // Return icon belonging to mime type of filename.

   TGMime *mime;
   const TGPicture  *mypic;

   if ((mime = Find(filename))) {
      if (small_icon)
         mypic = fClient->GetPicture(mime->fSIcon.Data(), 16, 16);
      else
         mypic = fClient->GetPicture(mime->fIcon.Data(), 32, 32);
      return mypic;
   }
   return 0;
}

//______________________________________________________________________________
Bool_t TGMimeTypes::GetAction(const char *filename, char *action)
{
   // Return in action the mime action string belonging to filename.

   TGMime *mime;

   action[0] = 0;
   if ((mime = Find(filename))) {
      strcpy(action, mime->fAction.Data());
      return (strlen(action) > 0);
   }
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGMimeTypes::GetType(const char *filename, char *type)
{
   // Return in type the mime type belonging to filename.

   TGMime *mime;

   memset(type, 0, strlen(type));
   if ((mime = Find(filename))) {
      strcpy(type, mime->fType.Data());
      return (strlen(type) > 0);
   }
   return kFALSE;
}

//______________________________________________________________________________
void TGMimeTypes::Print(Option_t *) const
{
   // Print list of mime types.

   TGMime *m;
   TIter next(fList);

   while ((m = (TGMime *) next())) {
      printf("Type:    %s\n", m->fType.Data());
      printf("Pattern: %s\n", m->fPattern.Data());
      if (m->fIcon != m->fSIcon)
         printf("Icon:    %s %s\n", m->fIcon.Data(), m->fSIcon.Data());
      else
         printf("Icon:    %s\n",    m->fIcon.Data());
      printf("Action:  %s\n", m->fAction.Data());
      printf("------------\n\n");
   }
}

//______________________________________________________________________________
void TGMimeTypes::SaveMimes()
{
   // Save mime types in user's mime type file.

   char filename[1024];
#ifdef WIN32
   sprintf(filename, "%s\\.root.mimes", gSystem->Getenv("HOME"));
#else
   sprintf(filename, "%s/.root.mimes", gSystem->Getenv("HOME"));
#endif

   FILE *fp = fopen(filename, "w");

   if (!fp) {
      Error("SaveMimes", "can not open %s to store mime types", filename);
      return;
   }

   TDatime dt;
   fprintf(fp, "# %s written on %s\n\n", filename, dt.AsString());

   TGMime *m;
   TIter next(fList);

   while ((m = (TGMime *) next())) {
      fprintf(fp, "%s\n",            m->fType.Data());
      fprintf(fp, "pattern = %s\n",  m->fPattern.Data());
      if (m->fIcon != m->fSIcon)
         fprintf(fp, "icon = %s %s\n", m->fIcon.Data(), m->fSIcon.Data());
      else
         fprintf(fp, "icon = %s\n",    m->fIcon.Data());
      fprintf(fp, "action = %s\n\n", m->fAction.Data());
   }

   fclose(fp);

   fChanged = kFALSE;
}

//______________________________________________________________________________
void TGMimeTypes::AddType(const char *type, const char *pattern, const char *icon,
                          const char *sicon, const char *action)
{
   // Add a mime type to the list of mime types.

  TGMime *mime = new TGMime;

  mime->fType    = type;
  mime->fPattern = pattern;
  mime->fIcon    = icon;
  mime->fSIcon   = sicon;
  mime->fAction  = action;

  mime->fReg = new TRegexp(pattern, kTRUE);

  fList->Add(mime);

  fChanged = kTRUE;
}

//______________________________________________________________________________
TGMime::~TGMime()
{
   // Delete mime object.

   delete fReg;
}
