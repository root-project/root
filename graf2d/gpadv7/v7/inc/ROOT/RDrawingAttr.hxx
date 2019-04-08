/// \file ROOT/RDrawingAttr.hxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2017-09-26
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RDrawingAttr
#define ROOT7_RDrawingAttr

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace ROOT {
namespace Experimental {

class RDrawingAttrHolderBase;

/** \class ROOT::Experimental::RDrawingAttrBase
 A collection of graphics attributes, for instance everything describing a line:
 color, width, opacity and style.
 It has a name, so it can be found in the style.
 */
class RDrawingAttrBase {
public:
   /// Name parts: for "hist1d.box.line.width", name parts are "hist1d", "box",
   /// border", and "width".
   using NamePart_t = std::string;

   /// Combination of names, e.g. "hist", "box", "line", "width".
   using Name_t = std::vector<NamePart_t>;

   /// Names of available attribute values if any, e.g. "color", "width".
   /// These must be string literals.
   std::unique_ptr<std::vector<NamePart_t>> fValueNames;

protected:
   /// The final element of the attribute name, as used in style files.
   /// I.e. for "hist1D.hist.box.line", this will be "line".
   NamePart_t fNamePart;  ///<!

   /// The container of the attribute values.
   RDrawingAttrHolderBase *fHolder{nullptr};

   /// The attribute that this attribute belongs to, if any.
   const RDrawingAttrBase *fParent{nullptr};

   /// Contained attributes, if any.
   std::unique_ptr<std::vector<const RDrawingAttrBase*>> fChildren;

protected:
   /// Register a child `subAttr` with the parent `*this`.
   void Register(const RDrawingAttrBase &subAttr);

   /// Build the name for the attribute value at index `valueIndex`.
   Name_t BuildNameForVal(std::size_t valueIndex) const;

   /// Insert or update the attribute value identified by the valueIndex (in fValueNames)
   /// to the value `strVal`.
   void Set(std::size_t valueIndex, const std::string &strVal);

   /// Get the attribute value as string, for a given index (in fValueNames).
   /// `second` is `true` if it comes from our `RDrawingAttrHolderBase` (i.e. was
   /// explicitly set through `Set()`) and `false` if it was determined from the
   /// styles, i.e. through `RDrawingAttrHolderBase::GetAttrFromStyle()`.
   std::pair<std::string, bool> Get(std::size_t valueIndex) const;

   /// Get the available value names.
   const std::vector<NamePart_t> *GetValueNames() const { return fValueNames.get(); }

public:
   /// Construct a default, unnamed, unconnected attribute.
   RDrawingAttrBase() = default;

   /// Construct a named attribute that does not have a parent; e.g.
   /// because it's the top-most attribute in a drawing option object.
   RDrawingAttrBase(const char* namePart): fNamePart(namePart) {}

   /// Construct a named attribute that has a parent, e.g.
   /// because it's some line attribute of the histogram attributes.
   /// Registers `*this` with the parent.
   RDrawingAttrBase(const char* namePart, RDrawingAttrHolderBase *holder, RDrawingAttrBase *parent);

   /// Construct a named attribute that has a parent, e.g.
   /// because it's some line attribute of the histogram attributes.
   /// Registers `*this` with the parent.
   /// Also provide the names of available values.
   RDrawingAttrBase(const char* namePart, RDrawingAttrHolderBase *holder, RDrawingAttrBase *parent,
      const std::vector<NamePart_t> &valueNames);

   /// Get the (partial, i.e. without parent context) name of this attribute.
   std::string GetNamePart() const { return fNamePart; }

   /// Collect the attribute names that lead to this attribute, starting
   /// with the topmost attribute, i.e. the parent that does not have a parent
   /// itself, down to the name of *this (the last entry in the vector).
   void GetName(Name_t &name) const;

   /// Convert a Name_t to "hist.box.line.color", for diagnostic purposes.
   static std::string NameToDottedDiagName(const Name_t &name);

   /// Assemble all attribute names below *this.
   void CollectChildNames(std::vector<Name_t> &names) const;

   /// Actual attribute holder.
   RDrawingAttrHolderBase* GetHolder() const { return fHolder; }
};


/** \class ROOT::Experimental::RDrawingAttrHolderBase
 A container of attributes for which values have been provided;
 top-most attribute edge. Provides an interface to the RStyle world.
 */
class RDrawingAttrHolderBase {
public:
   using Name_t = RDrawingAttrBase::Name_t;
private:
   struct StringVecHash {
      std::size_t operator()(const Name_t &vec) const;
   };

   /// Map attribute names to their values.
   std::unordered_map<Name_t, std::string, StringVecHash> fAttrNameVals;

public:
   virtual ~RDrawingAttrHolderBase();
   /// Get an attribute value as string, given its name path.
   std::string &At(const Name_t &attrName) { return fAttrNameVals[attrName]; }

   /// Get an attribute value as pointer to string, given its name path, or
   /// `nullptr` if the attribute does not exist.
   const std::string *AtIf(const Name_t &attrName) const;

   virtual std::string GetAttrFromStyle(const Name_t &attrName) = 0;
};

} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_RDrawingAttr
