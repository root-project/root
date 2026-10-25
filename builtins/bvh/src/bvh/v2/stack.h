#ifndef BVH_V2_STACK_H
#define BVH_V2_STACK_H

#include <vector>
#include <cassert>

namespace bvh::v2 {

/// Fixed-size stack that can be used for a BVH traversal.
template <typename T, unsigned Capacity>
struct SmallStack {
    static constexpr unsigned capacity = Capacity;

    T elems[capacity];
    unsigned size = 0;

    bool is_empty() const { return size == 0; }
    bool is_full() const { return size >= capacity; }

    void push(const T& t) {
        assert(!is_full());
        elems[size++] = t;
    }

    T pop() {
        assert(!is_empty());
        return elems[--size];
    }
};

/// Growing stack that can be used for BVH traversal. Its performance may be lower than a small,
/// fixed-size stack, depending on the architecture.
template <typename T>
struct GrowingStack {
    std::vector<T> elems;

    bool is_empty() const { return elems.empty(); }
    void push(const T& t) { elems.push_back(t); }

    T pop() {
        assert(!is_empty());
        auto top = std::move(elems.back());
        elems.pop_back();
        return top;
    }
};

} // namespace bvh::v2

#endif
