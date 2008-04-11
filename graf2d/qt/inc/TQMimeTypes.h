// @(#)root/qt:$Id$
// Author: Valeri Fine   21/01/2003

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * Copyright (C) 2002 by Valeri Fine.                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

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
#ifndef __CINT__
#  include "qglobal.h"
#endif

class QFileIconProvider;
class TOrdCollection;
class TRegexp;
class QFileInfo;
class QPixmap;

#if (QT_VERSION > 0x39999)
   class QIcon;
#else /* QT_VERSION */
   class QIconSet;
#endif /* QT_VERSION */
class TSystemFile;

class TQMime : public TObject {

friend class TQMimeTypes;

private:
   TString   fType;      // mime type
   TString   fPattern;   // filename pattern
   TString   fAction;    // associated action
#if (QT_VERSION > 0x39999)
   QIcon  *fIcon;     // associated icon set
#else /* QT_VERSION */
   QIconSet  *fIcon;     // associated icon set
#endif /* QT_VERSION */
   TRegexp   *fReg;      // pattern regular expression

public:
  ~TQMime();
};


class TQMimeTypes : public TObject {
private:
   void operator=(const TQMimeTypes&);
   TQMimeTypes(const TQMimeTypes&);

protected:
   TString          fIconPath;   // the path to the icon directory
   TString          fFilename;   // file name of mime type file
   Bool_t           fChanged;    // true if file has changed
   TOrdCollection  *fList;       // list of mime types

   static QFileIconProvider  *fgDefaultProvider; // Default provider of the system icons;

   TQMime    *Find(const char *filename) const;
#if (QT_VERSION > 0x39999)
   const QIcon *AddType(const TSystemFile *filename);
   static QIcon  IconProvider(const QFileInfo &);
#else /* QT_VERSION */
   const QIconSet *AddType(const TSystemFile *filename);
   static const QPixmap  &IconProvider(const QFileInfo &);
#endif /* QT_VERSION */
public:
   TQMimeTypes(const char *iconPath, const char *file);
   virtual ~TQMimeTypes();
   void   SaveMimes();
   Bool_t HasChanged() const { return fChanged; }
   void   AddType(const char *type, const char *pat, const char *icon, const char *sicon, const char *action);
   void   Print(Option_t *option="") const;
   Bool_t GetAction(const char *filename, char *action) const;
   Bool_t GetType(const char *filename, char *type) const;
#if (QT_VERSION > 0x39999)
   const  QIcon *GetIcon(const char *filename) const;
   const  QIcon *GetIcon(const TSystemFile *filename);
#else /* QT_VERSION */
   const  QIconSet *GetIcon(const char *filename) const;
   const  QIconSet *GetIcon(const TSystemFile *filename);
#endif /* QT_VERSION */


#ifndef Q_MOC_RUN
//MOC_SKIP_BEGIN
   ClassDef(TQMimeTypes,0)  // Pool of mime type objects
//MOC_SKIP_END
#endif
};

#endif
