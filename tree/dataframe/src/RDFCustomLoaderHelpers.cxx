#include <ROOT/RDF/CustomLoaderHelpers.hxx>
#include <ROOT/RDF/Utils.hxx> // TypeID2TypeName

#include <stdexcept>
#include <typeinfo>

namespace {

using ROOT::Internal::RDF::CustomLoaderHelper;

template <typename T>
void AppendAs(ROOT::RVecF &dest, void *value)
{
   dest.push_back(static_cast<float>(*static_cast<T *>(value)));
}

/// Pick the handler matching the type of a column
CustomLoaderHelper::ColHandler_t ResolveColHandler(const std::type_info &colType)
{
   if (colType == typeid(float))
      return &AppendAs<float>;
   if (colType == typeid(double))
      return &AppendAs<double>;
   if (colType == typeid(bool))
      return &AppendAs<bool>;
   if (colType == typeid(char))
      return &AppendAs<char>;
   if (colType == typeid(signed char))
      return &AppendAs<signed char>;
   if (colType == typeid(unsigned char))
      return &AppendAs<unsigned char>;
   if (colType == typeid(short))
      return &AppendAs<short>;
   if (colType == typeid(unsigned short))
      return &AppendAs<unsigned short>;
   if (colType == typeid(int))
      return &AppendAs<int>;
   if (colType == typeid(unsigned int))
      return &AppendAs<unsigned int>;
   if (colType == typeid(long))
      return &AppendAs<long>;
   if (colType == typeid(unsigned long))
      return &AppendAs<unsigned long>;
   if (colType == typeid(long long))
      return &AppendAs<long long>;
   if (colType == typeid(unsigned long long))
      return &AppendAs<unsigned long long>;

   throw std::invalid_argument("CustomLoaderHelper: column type '" + ROOT::Internal::RDF::TypeID2TypeName(colType) +
                               "' cannot be converted to float.");
}

} // namespace

ROOT::Internal::RDF::CustomLoaderHelper::CustomLoaderHelper(const std::shared_ptr<ROOT::RVecF> &location,
                                                            const unsigned int nSlots,
                                                            const std::vector<const std::type_info *> &colTypeIDs)
   : fLocation(location), fNSlots(nSlots), fColTypeIDs(colTypeIDs)
{
   fColHandlers.reserve(fColTypeIDs.size());
   for (const auto *colType : fColTypeIDs)
      fColHandlers.push_back(ResolveColHandler(*colType));
}

void ROOT::Internal::RDF::CustomLoaderHelper::Exec(unsigned int /*slot*/, const std::vector<void *> &values)
{
   // The readers deliver the values in column order, the same order the handlers were resolved in
   auto nValues{values.size()};
   for (decltype(nValues) i{}; i < nValues; i++)
      fColHandlers[i](*fLocation, values[i]);
}
