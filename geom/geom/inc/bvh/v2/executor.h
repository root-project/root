#ifndef BVH_V2_EXECUTOR_H
#define BVH_V2_EXECUTOR_H

#include "bvh/v2/thread_pool.h"

#include <cstddef>
#include <algorithm>
#include <vector>

namespace bvh::v2 {

/// Helper object that provides iteration and reduction over one-dimensional ranges.
template <typename Derived>
struct Executor {
    template <typename Loop>
    inline void for_each(size_t begin, size_t end, const Loop& loop) {
        return static_cast<Derived*>(this)->for_each(begin, end, loop);
    }

    template <typename T, typename Reduce, typename Join>
    inline T reduce(size_t begin, size_t end, const T& init, const Reduce& reduce, const Join& join) {
        return static_cast<Derived*>(this)->reduce(begin, end, init, reduce, join);
    }
};

/// Executor that executes serially.
struct SequentialExecutor : Executor<SequentialExecutor> {
    template <typename Loop>
    void for_each(size_t begin, size_t end, const Loop& loop) {
        loop(begin, end);
    }

    template <typename T, typename Reduce, typename Join>
    T reduce(size_t begin, size_t end, const T& init, const Reduce& reduce, const Join&) {
        T result(init);
        reduce(result, begin, end);
        return result;
    }
};

/// Executor that executes in parallel using the given thread pool.
struct ParallelExecutor : Executor<ParallelExecutor> {
    ThreadPool& thread_pool;
    size_t parallel_threshold;

    ParallelExecutor(ThreadPool& thread_pool, size_t parallel_threshold = 1024)
        : thread_pool(thread_pool), parallel_threshold(parallel_threshold)
    {}

    template <typename Loop>
    void for_each(size_t begin, size_t end, const Loop& loop) {
        if (end - begin < parallel_threshold)
            return loop(begin, end);

        auto chunk_size = std::max(size_t{1}, (end - begin) / thread_pool.get_thread_count());
        for (size_t i = begin; i < end; i += chunk_size) {
            size_t next = std::min(end, i + chunk_size);
            thread_pool.push([=] (size_t) { loop(i, next); });
        }
        thread_pool.wait();
    }

    template <typename T, typename Reduce, typename Join>
    T reduce(size_t begin, size_t end, const T& init, const Reduce& reduce, const Join& join) {
        if (end - begin < parallel_threshold) {
            T result(init);
            reduce(result, begin, end);
            return result;
        }

        auto chunk_size = std::max(size_t{1}, (end - begin) / thread_pool.get_thread_count());
        std::vector<T> per_thread_result(thread_pool.get_thread_count(), init);
        for (size_t i = begin; i < end; i += chunk_size) {
            size_t next = std::min(end, i + chunk_size);
            thread_pool.push([&, i, next] (size_t thread_id) {
                auto& result = per_thread_result[thread_id];
                reduce(result, i, next);
            });
        }
        thread_pool.wait();
        for (size_t i = 1; i < thread_pool.get_thread_count(); ++i)
            join(per_thread_result[0], std::move(per_thread_result[i]));
        return per_thread_result[0];
    }
};

} // namespace bvh::v2

#endif
