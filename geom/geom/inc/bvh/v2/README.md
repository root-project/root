These headers, providing routines to construct and navigate
bounding volume hierarchies, have been copied from https://github.com/madmann91/bvh
commit 66e445b92f68801a6dd8ef94.


Minor changes have been subsequently been applied to achieve compilation with C++17:

- inclusion of alternative span, when std::span is not found
- replacement of C++20 defaulted comparison operators with actual implementation
- old-style struct construction for objects of type "Reinsertion"
- use of std::inner_product instead of std::transform_reduce (gcc 8.5 had problems)

This is needed since ROOT should compile with C++17.
