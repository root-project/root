#ifndef REMOVE_AT_HPP
#define REMOVE_AT_HPP

// check if c++11 is supported
#if __cplusplus >= 201103L || (defined(_MSC_VER) && _MSC_VER >= 1900)
#define REMOVE_AT_CXX11_IS_SUPPORTED
#elif !defined(__cplusplus) && !defined(_MSC_VER)
typedef char couldnt_parse_cxx_standard[-1];
#endif

#include <algorithm>
#include <iterator>

/*!
 * Remove elements in the range [first; last) with indices from the sorted
 * unique range [ii_first, ii_last)
 */
template <class ForwardIt, class SortUniqIndsFwdIt>
inline ForwardIt remove_at(
    ForwardIt first,
    ForwardIt last,
    SortUniqIndsFwdIt ii_first,
    SortUniqIndsFwdIt ii_last)
{
    if(ii_first == ii_last) // no indices-to-remove are given
        return last;
    typedef typename std::iterator_traits<ForwardIt>::difference_type diff_t;
    typedef typename std::iterator_traits<SortUniqIndsFwdIt>::value_type ind_t;
    ForwardIt destination = first + static_cast<diff_t>(*ii_first);
    while(ii_first != ii_last)
    {
        // advance to an index after a chunk of elements-to-keep
        for(ind_t cur = *ii_first++; ii_first != ii_last; ++ii_first)
        {
            const ind_t nxt = *ii_first;
            if(nxt - cur > 1)
                break;
            cur = nxt;
        }
        // move the chunk of elements-to-keep to new destination
        const ForwardIt source_first =
            first + static_cast<diff_t>(*(ii_first - 1)) + 1;
        const ForwardIt source_last =
            ii_first != ii_last ? first + static_cast<diff_t>(*ii_first) : last;
#ifdef REMOVE_AT_CXX11_IS_SUPPORTED
        std::move(source_first, source_last, destination);
#else
        std::copy(source_first, source_last, destination); // c++98 version
#endif
        destination += source_last - source_first;
    }
    return destination;
}

#endif // REMOVE_AT_HPP
