#ifndef DataVector_h
#define DataVector_h

namespace DataModel_detail {


   /**
    * @brief Marker for the base of the inheritance hierarchy.
    *
    * Normally, @c DataVectorBase<T>::Base is the struct from which
    * @c DataVector<T> should derive.
    * If, however, @c T has not had a base declared to @c DataVector,
    * then @c DataVectorBase<T>::Base will be @c NoBase.
    * (And similarly for @c DataList.)
    */
   struct NoBase {};
   
   
} // namespace DataModel_detail


template <typename T> struct DataVectorBase {
   typedef  DataModel_detail::NoBase Base;
   int fGeneric;
};

template <typename T, typename BASE = typename DataVectorBase<T>::Base>
struct DataVector : public BASE
{
   DataVector() : fValue(nullptr) {}

   T *fValue;
};

template <typename T>
struct SDataVector : DataVectorBase<T>
{
   SDataVector() : fValue(nullptr) {}

   T *fValue;
};

template <typename T> struct DataLink
{
   DataLink() : fLink(nullptr) {}

   T *fLink;
};

template <typename T> struct ElementLink
{
   ElementLink() : fLink(nullptr) {}

   T *fLink;
};

template <typename T> struct ElementLinkVector
{
   ElementLinkVector() : fLink(nullptr) {}

   T *fLink;
};

namespace SG {
   struct IAuxTypeVectorFactory {
   public:
      virtual void* Generate() = 0;
   };

   template <typename T> struct AuxTypeVectorFactory
   {
      virtual void* Generate() { return new T; }
   };

   struct AuxElement {};
}

namespace xAOD {
   struct IParticle {};
}

#endif
