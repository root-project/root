/// \file ROOT/TMenuItem.h
/// \ingroup Base ROOT7
/// \author Sergey Linev
/// \date 2017-06-29
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_TMenuItem
#define ROOT7_TMenuItem

#include <string>
#include <vector>

class TClass;

namespace ROOT {
namespace Experimental {
namespace Detail {

/** \class TMenuItem
  Class contains info for producing menu item on the JS side.
  */

class TMenuItem {
protected:
   std::string fName;  ///<  name of the menu item
   std::string fTitle; ///<  title of menu item
   std::string fExec;  ///< execute when item is activated
public:
   /** Default constructor */
   TMenuItem() = default;

   /** Create menu item with the name and title
    *  name used to display item in the object context menu,
    *  title shown as hint info for that item  */
   TMenuItem(const std::string &name, const std::string &title) : fName(name), fTitle(title), fExec() {}

   /** virtual destructor need for vtable, used when vector of TMenuItem* is stored */
   virtual ~TMenuItem() {}

   /** Set execution string with all required arguments,
    * which will be executed when menu item is selected  */
   void SetExec(const std::string &exec) { fExec = exec; }

   /** Returns menu item name */
   const std::string &GetName() const { return fName; }

   /** Returns execution string for the menu item */
   const std::string &GetExec() const { return fExec; }
};

class TCheckedMenuItem : public TMenuItem {
protected:
   bool fChecked = false; ///< -1 not exists, 0 - off, 1 - on
public:
   /** Default constructor */
   TCheckedMenuItem() = default;

   /** Create checked menu item  */
   TCheckedMenuItem(const std::string &name, const std::string &title, bool checked = false)
      : TMenuItem(name, title), fChecked(checked)
   {
   }

   /** virtual destructor need for vtable, used when vector of TMenuItem* is stored */
   virtual ~TCheckedMenuItem() {}

   /** Set checked state for the item, default is none */
   void SetChecked(bool on = true) { fChecked = on; }

   bool IsChecked() const { return fChecked; }
};

class TMenuArgument {
protected:
   std::string fName;     ///<  name of call argument
   std::string fTitle;    ///<  title of call argument
   std::string fTypeName; ///<  typename
   std::string fDefault;  ///<  default value
public:
   /** Default constructor */
   TMenuArgument() = default;

   TMenuArgument(const std::string &name, const std::string &title, const std::string &typname,
                 const std::string &dflt = "")
      : fName(name), fTitle(title), fTypeName(typname), fDefault(dflt)
   {
   }

   void SetDefault(const std::string &dflt) { fDefault = dflt; }
};

class TArgsMenuItem : public TMenuItem {
protected:
   std::vector<TMenuArgument> fArgs;

public:
   /** Default constructor */
   TArgsMenuItem() = default;

   TArgsMenuItem(const std::string &name, const std::string &title) : TMenuItem(name, title) {}

   /** virtual destructor need for vtable, used when vector of TMenuItem* is stored */
   virtual ~TArgsMenuItem() {}

   void AddArg(const TMenuArgument &arg) { fArgs.push_back(arg); }
};

} // namespace Detail

///////////////////////////////////////////////////////////////////////

class TMenuItems {
protected:
   std::vector<Detail::TMenuItem *> fItems; ///< list of items in the menu
public:
   /** Default constructor */
   TMenuItems() = default;

   ~TMenuItems() { Cleanup(); }

   void Add(Detail::TMenuItem *item) { fItems.push_back(item); }

   void AddMenuItem(const std::string &name, const std::string &title, const std::string &exec)
   {
      Detail::TMenuItem *item = new Detail::TMenuItem(name, title);
      item->SetExec(exec);
      Add(item);
   }

   void AddChkMenuItem(const std::string &name, const std::string &title, bool checked, const std::string &toggle)
   {
      Detail::TCheckedMenuItem *item = new Detail::TCheckedMenuItem(name, title, checked);
      item->SetExec(toggle);
      Add(item);
   }

   unsigned Size() const { return fItems.size(); }

   void Cleanup();

   void PopulateObjectMenu(void *obj, TClass *cl);

   std::string ProduceJSON();
};

} // namespace Experimental
} // namespace ROOT

#endif
