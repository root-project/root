// @(#)root/gui:$Name:  $:$Id: TQMimeTypes.h,v 1.2 2003/11/18 18:41:55 fine Exp $
// Author: Valeri Fine   21/01/2003
/****************************************************************************
** $Id: TQMimeTypes.h,v 1.2 2003/11/18 18:41:55 fine Exp $
**
** Copyright (C) 2002 by Valeri Fine. Brookhaven National Laboratory.
**                                    All rights reserved.
**
** This file may be distributed under the terms of the Q Public License
** as defined by Trolltech AS of Norway and appearing in the file
** LICENSE.QPL included in the packaging of this file.
**
*****************************************************************************/

#ifndef ROOT_TQMimeTypes
#define ROOT_TQMimeTypes


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TQMimeTypes and TQMime                                               //
//                                                                      //
// This class handles mime types, used by browsers to map file types    //
// to applications and icons. TQMime is internally used by TQMimeTypes. //
//                                                                      //
// This classes are based on TGMimeTypes and TGMime class from          //
// ROOT "gui"  package                                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TObject.h"
#include "TString.h"
#include <qfiledialog.h> 

class TOrdCollection;
class TRegexp;

class QIconSet;
class TSystemFile;

class TQMime : public TObject {

friend class TQMimeTypes;

private:
   TString   fType;      // mime type
   TString   fPattern;   // filename pattern
   TString   fAction;    // associated action
   QIconSet  *fIcon;     // associated icon set
   TRegexp   *fReg;      // pattern regular expression

public:
  ~TQMime();
};


class TQMimeTypes : public TObject {

protected:
   TString          fIconPath;   // the path to the icon directory
   TString          fFilename;   // file name of mime type file
   Bool_t           fChanged;    // true if file has changed
   TOrdCollection  *fList;       // list of mime types
   QFileIconProvider fDefaultProvider; // Default provider of the system icons;

   TQMime    *Find(const char *filename) const;
   const QIconSet *AddType(const TSystemFile *filename);

public:
   TQMimeTypes(const char *iconPath, const char *file);
   virtual ~TQMimeTypes();
   static TQMimeTypes *Instantiate(const char *iconPath, const char *file);
   void   SaveMimes();
   Bool_t HasChanged() const { return fChanged; }
   void   AddType(const char *type, const char *pat, const char *icon, const char *sicon, const char *action);
   void   Print(Option_t *option="") const;
   Bool_t GetAction(const char *filename, char *action) const;
   Bool_t GetType(const char *filename, char *type) const;
   const  QIconSet *GetIcon(const char *filename) const;
   const  QIconSet *GetIcon(const TSystemFile *filename);


   // ClassDef(TQMimeTypes,0)  // Pool of mime type objects
};

#endif
