/// \file ROOT/TWebMenuItem.h
/// \ingroup Base ROOT7
/// \author Sergey Linev
/// \date 2017-06-29

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TWebMenuItem
#define ROOT_TWebMenuItem

#include "TString.h"
#include "TClass.h"

#include <string>
#include <vector>

class TClass;

/** \class TWebMenuItem
  Class contains info for producing menu item on the JS side.
  */

class TWebMenuItem {
protected:
   std::string fName;       //  name of the menu item
   std::string fTitle;      //  title of menu item
   std::string fExec;       // execute when item is activated
   std::string fClassName;  // class name
public:

   TWebMenuItem() : fName(), fTitle(), fExec(), fClassName() {}
   TWebMenuItem(const std::string &name, const std::string &title) : fName(name), fTitle(title), fExec(), fClassName() {}
   TWebMenuItem(const TWebMenuItem &rhs) : fName(rhs.fName), fTitle(rhs.fTitle), fExec(rhs.fExec), fClassName(rhs.fClassName) {}
   virtual ~TWebMenuItem() {}

   /** Set execution string with all required arguments,
    * which will be executed when menu item is selected  */
   void SetExec(const std::string &exec) { fExec = exec; }

   /** Set class name, to which method belons to  */
   void SetClassName(const std::string &clname) { fClassName = clname; }

   /** Returns menu item name */
   const std::string &GetName() const { return fName; }

   /** Returns execution string for the menu item */
   const std::string &GetExec() const { return fExec; }
};

class TWebCheckedMenuItem : public TWebMenuItem {
protected:
   bool fChecked; ///< -1 not exists, 0 - off, 1 - on
public:
   /** Default constructor */
   TWebCheckedMenuItem() : TWebMenuItem(), fChecked(false) {}

   /** Create checked menu item  */
   TWebCheckedMenuItem(const std::string &name, const std::string &title, bool checked = false)
      : TWebMenuItem(name, title), fChecked(checked)
   {
   }

   /** virtual destructor need for vtable, used when vector of TMenuItem* is stored */
   virtual ~TWebCheckedMenuItem() {}

   /** Set checked state for the item, default is none */
   void SetChecked(bool on = true) { fChecked = on; }

   bool IsChecked() const { return fChecked; }
};

class TWebMenuArgument {
protected:
   std::string fName;     ///<  name of call argument
   std::string fTitle;    ///<  title of call argument
   std::string fTypeName; ///<  typename
   std::string fDefault;  ///<  default value
public:
   /** Default constructor */
   TWebMenuArgument() : fName(), fTitle(), fTypeName(), fDefault() {}

   TWebMenuArgument(const std::string &name, const std::string &title, const std::string &typname,
                 const std::string &dflt = "")
      : fName(name), fTitle(title), fTypeName(typname), fDefault(dflt)
   {
   }

   void SetDefault(const std::string &dflt) { fDefault = dflt; }
};

class TWebArgsMenuItem : public TWebMenuItem {
protected:
   std::vector<TWebMenuArgument> fArgs;

public:
   /** Default constructor */
   TWebArgsMenuItem() : TWebMenuItem(), fArgs() {}

   TWebArgsMenuItem(const std::string &name, const std::string &title) : TWebMenuItem(name, title), fArgs() {}

   /** virtual destructor need for vtable, used when vector of TMenuItem* is stored */
   virtual ~TWebArgsMenuItem() {}

   void AddArg(const TWebMenuArgument &arg) { fArgs.push_back(arg); }
};

///////////////////////////////////////////////////////////////////////

class TWebMenuItems {
protected:
   std::vector<TWebMenuItem *> fItems; ///< list of items in the menu
public:
   /** Default constructor */
   TWebMenuItems() : fItems() {}

   ~TWebMenuItems() { Cleanup(); }

   void Add(TWebMenuItem *item) { fItems.push_back(item); }

   void AddMenuItem(const std::string &name, const std::string &title, const std::string &exec, TClass *cl = 0)
   {
      TWebMenuItem *item = new TWebMenuItem(name, title);
      item->SetExec(exec);
      if (cl) item->SetClassName(cl->GetName());
      Add(item);
   }

   void AddChkMenuItem(const std::string &name, const std::string &title, bool checked, const std::string &toggle, TClass *cl = 0)
   {
      TWebCheckedMenuItem *item = new TWebCheckedMenuItem(name, title, checked);
      item->SetExec(toggle);
      if (cl) item->SetClassName(cl->GetName());
      Add(item);
   }

   unsigned Size() const { return fItems.size(); }

   void Cleanup();

   void PopulateObjectMenu(void *obj, TClass *cl);

   TString ProduceJSON();
};



#endif
