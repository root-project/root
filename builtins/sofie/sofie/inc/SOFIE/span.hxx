#ifndef SOFIE_SPAN
#define SOFIE_SPAN

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

namespace SOFIE {

namespace SpanDetail {

/// True if a pointer to the elements of Container can be converted to T *.
template <typename T, typename Container, typename = void>
struct IsCompatibleContainer : std::false_type {
};

template <typename T, typename Container>
struct IsCompatibleContainer<T, Container, std::void_t<decltype(std::data(std::declval<Container &>()))>>
   : std::is_convertible<std::remove_pointer_t<decltype(std::data(std::declval<Container &>()))> (*)[], T (*)[]> {
};

} // namespace SpanDetail

/// Minimal non-owning view over a contiguous sequence of objects.
///
/// This is the subset of std::span with dynamic extent that is used by the
/// library, which makes it independent of the C++ standard (std::span needs
/// C++20). It lives in the library namespace, so it never interferes with
/// std::span or with any other span implementation.
template <typename T>
class span {
public:
   using element_type = T;
   using value_type = std::remove_cv_t<T>;
   using size_type = std::size_t;
   using pointer = T *;
   using reference = T &;
   using iterator = T *;

   constexpr span() noexcept = default;
   constexpr span(pointer ptr, size_type count) noexcept : fData(ptr), fSize(count) {}

   /// Construct from a contiguous container (e.g. std::vector or std::array).
   template <typename Container, typename = std::enable_if_t<SpanDetail::IsCompatibleContainer<T, Container>::value>>
   constexpr span(Container &c) noexcept : fData(std::data(c)), fSize(std::size(c))
   {
   }

   /// Construct from a const container or a temporary, only if the elements are const.
   template <typename Container,
             typename = std::enable_if_t<std::is_const<T>::value &&
                                         SpanDetail::IsCompatibleContainer<T, const Container>::value>>
   constexpr span(const Container &c) noexcept : fData(std::data(c)), fSize(std::size(c))
   {
   }

   constexpr iterator begin() const noexcept { return fData; }
   constexpr iterator end() const noexcept { return fData + fSize; }
   constexpr reference front() const { return *fData; }
   constexpr reference back() const { return *(fData + fSize - 1); }
   constexpr reference operator[](size_type idx) const { return fData[idx]; }
   constexpr pointer data() const noexcept { return fData; }
   constexpr size_type size() const noexcept { return fSize; }
   constexpr bool empty() const noexcept { return fSize == 0; }

private:
   pointer fData = nullptr;
   size_type fSize = 0;
};

} // namespace SOFIE

#endif // SOFIE_SPAN
