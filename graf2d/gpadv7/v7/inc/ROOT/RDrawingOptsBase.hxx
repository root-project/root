/// \file ROOT/RDrawingOptionsBase.hxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2018-02-12
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RDrawingOptsBase
#define ROOT7_RDrawingOptsBase

#include <ROOT/RDrawingAttr.hxx>

#include <string>
#include <vector>

namespace ROOT {
namespace Experimental {

class RDrawingOptsBase {
public:
   using Name_t = RDrawingAttrBase::Name;
   using Path_t = RDrawingAttrBase::Path;

private:
   /// The RDrawingAttrHolder of the attribute values.
   std::unique_ptr<RDrawingAttrHolder> fHolder;

protected:
   RDrawingOptsBase() = default;

   /// Initialize the options with a (possibly empty) set of style classes.
   RDrawingOptsBase(const std::vector<std::string> &styleClasses);

   RDrawingOptsBase(const RDrawingOptsBase &other);

   RDrawingOptsBase(RDrawingOptsBase &&other) = default;

   RDrawingOptsBase &operator=(const RDrawingOptsBase &other);

   RDrawingOptsBase &operator=(RDrawingOptsBase &&other) = default;

   /// Get the root class name of these options in the style file,
   /// e.g. "hist" in "hist.line.width".
   virtual Name_t GetName() const = 0;

public:
   virtual ~RDrawingOptsBase() = default;

   /// Get the attribute style classes of these options.
   const std::vector<std::string> &GetStyleClasses() const;

   /// Get the attribute style classes of these options.
   void SetStyleClasses(const std::vector<std::string> &styles);

   /// Get the holder of the attributes.
   RDrawingAttrHolder &GetHolder();

   /// Construct an attribute from the RDrawingAttrHolder's data given the
   /// attribute's Name.
   template <class ATTR>
   ATTR GetAttribute(const Name_t &name) const;

   /// Initialize an attribute from the styles and custom settings, insert it into the
   /// holder and return a reference to it.
   template <class ATTR>
   ATTR &Get(const Name_t &name)
   {
      if (RDrawingAttrBase* exists = GetHolder().AtIf(name))
         return *static_cast<ATTR*>(exists);
      return GetHolder().Insert<ATTR>(name);
   }

   /// Collect all attribute members' values into keyval, using their name as the first string
   /// and their stringified value as the second. This pair is only inserted into keyval if
   /// the attribute's value is different than value provided by the style.
   std::vector<std::pair<std::string, std::string>> GetModifiedAttributeStrings();
};

} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_RDrawingOptsBase
