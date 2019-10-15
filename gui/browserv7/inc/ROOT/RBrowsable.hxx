/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RBrowsable
#define ROOT7_RBrowsable

#include <memory>
#include <string>
#include <map>


class TClass;

namespace ROOT {
namespace Experimental {


/** \class RBrowsableInfo
\ingroup GpadROOT7
\brief Basic information about RBrowsable
\author Sergey Linev <S.Linev@gsi.de>
\date 2019-10-14
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RBrowsableInfo {
public:
   virtual ~RBrowsableInfo() = default;

   /** Class information, must be provided in derived classes */
   virtual const TClass *GetGlass() const = 0;

   /** Name of RBrowsable, must be provided in derived classes */
   virtual std::string GetName() const = 0;

   /** Title of RBrowsable (optional) */
   virtual std::string GetTitle() const { return ""; }

   /** Number of childs in RBrowsable (optional) */
   virtual int GetNumChilds() const { return 0; }
};

/** \class RBrowsableLevelIter
\ingroup GpadROOT7
\brief Iterator over single level hierarchy like TList
\author Sergey Linev <S.Linev@gsi.de>
\date 2019-10-14
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/


class RBrowsableLevelIter {
public:
   virtual ~RBrowsableLevelIter() = default;

   /** Shift to next element */
   virtual bool Next() { return false; }

   /** Reset iterator to the first element, returns false if not supported */
   virtual bool Reset() { return false; }

   /** Returns information for current element */
   virtual std::unique_ptr<RBrowsableInfo> GetInfo() { return nullptr; }
};

/** \class RBrowsable
\ingroup GpadROOT7
\brief Way to browse (hopefully) everything in ROOT
\author Sergey Linev <S.Linev@gsi.de>
\date 2019-10-14
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/


class RBrowsable {
public:
   RBrowsable() = default;
   virtual ~RBrowsable() = default;


};


/** \class RBrowsableProvider
\ingroup GpadROOT7
\brief Provider of different browsing methods for supported classes
\author Sergey Linev <S.Linev@gsi.de>
\date 2019-10-14
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/


class RBrowsableProvider {

   using Map_t = std::map<const TClass*, std::shared_ptr<RBrowsableProvider>>;

   static Map_t &GetMap();

public:
   virtual ~RBrowsableProvider() = default;

   /** Returns supported class */
   virtual const TClass *GetSupportedClass() const = 0;

   /** Returns true if derived classes supported as well */
   virtual bool SupportDerivedClasses() const { return false; }


   static void Register(std::shared_ptr<RBrowsableProvider> provider);
   static std::shared_ptr<RBrowsableProvider> GetProvider(const TClass *cl, bool check_base = true);

};


} // namespace Experimental
} // namespace ROOT

#endif
