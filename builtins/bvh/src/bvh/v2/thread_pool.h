#ifndef BVH_V2_THREAD_POOL_H
#define BVH_V2_THREAD_POOL_H

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

namespace bvh::v2 {

class ThreadPool {
public:
    /// Groups multiple tasks under a single unit of work. All tasks in the same group can be
    /// be waited for completion using ``ThreadPool::wait()``.
    class TaskGroup {
    public:
        TaskGroup() = default;

    private:
        std::atomic<std::size_t> remaining_{0};

        TaskGroup(const TaskGroup&) = delete;
        TaskGroup& operator=(const TaskGroup&) = delete;
        TaskGroup(TaskGroup&&) = delete;
        TaskGroup& operator=(TaskGroup&&) = delete;

        friend ThreadPool;

        inline void increment();
        inline void decrement();
        inline size_t remaining() const;
    };

    /// Creates a thread pool with the given number of threads (a value of 0 tries to autodetect
    /// the number of threads and uses that as a thread count).
    ThreadPool(size_t thread_count = 0) { start(thread_count); }

    ~ThreadPool() {
        wait_all();
        stop();
        join();
    }

    /// Pushes a task on the thread pool, associated with a given `group`.
    template<typename F>
    inline void push(TaskGroup& group, F&& fun);

    /// Wait for completion of all tasks scoped under given `group`.
    inline void wait(TaskGroup& group);

    /// Wait for completion of all tasks.
    inline void wait_all();

    size_t get_thread_count() const { return threads_.size(); }

private:
    struct Task {
        TaskGroup* group = nullptr;
        std::function<void(size_t)> fn;
    };

    static inline void worker(ThreadPool*, size_t);

    inline void start(size_t);
    inline void stop();
    inline void join();

    int busy_count_ = 0;
    bool should_stop_ = false;
    std::mutex mutex_;
    std::vector<std::thread> threads_;
    std::condition_variable avail_;
    std::condition_variable group_done_;
    std::queue<Task> tasks_;
};

template<typename F>
void ThreadPool::push(TaskGroup& group, F&& fun) {
    {
        std::unique_lock<std::mutex> lock(mutex_);
        group.increment();
        tasks_.emplace(Task{&group, std::forward<F>(fun)});
    }
    avail_.notify_one();
}

inline void ThreadPool::wait(TaskGroup& group) {
    std::unique_lock<std::mutex> lock(mutex_);
    group_done_.wait(lock, [&group] { return group.remaining() == 0; });
}

inline void ThreadPool::wait_all() {
    std::unique_lock<std::mutex> lock(mutex_);
    group_done_.wait(lock, [this] { return busy_count_ == 0 && tasks_.empty(); });
}

void ThreadPool::worker(ThreadPool* pool, size_t thread_id) {
    while (true) {
        Task task;
        {
            std::unique_lock<std::mutex> lock(pool->mutex_);
            pool->avail_.wait(lock, [pool] { return pool->should_stop_ || !pool->tasks_.empty(); });
            if (pool->should_stop_ && pool->tasks_.empty())
                break;

            task = std::move(pool->tasks_.front());
            pool->tasks_.pop();
            pool->busy_count_++;
        }

        task.fn(thread_id);
        task.group->decrement();

        {
            std::unique_lock<std::mutex> lock(pool->mutex_);
            pool->busy_count_--;
        }

        if (task.group->remaining() == 0)
            pool->group_done_.notify_all();
    }
}

void ThreadPool::start(size_t thread_count) {
    if (thread_count == 0)
        thread_count = std::max(1u, std::thread::hardware_concurrency());
    for (size_t i = 0; i < thread_count; ++i)
        threads_.emplace_back(worker, this, i);
}

void ThreadPool::stop() {
    {
        std::unique_lock<std::mutex> lock(mutex_);
        should_stop_ = true;
    }
    avail_.notify_all();
}

void ThreadPool::join() {
    for (auto& thread : threads_)
        thread.join();
}

inline void ThreadPool::TaskGroup::increment() {
    remaining_.fetch_add(1, std::memory_order_release);
}

inline void ThreadPool::TaskGroup::decrement() {
    remaining_.fetch_sub(1, std::memory_order_release);
}

inline size_t ThreadPool::TaskGroup::remaining() const {
    return remaining_.load(std::memory_order_acquire);
}

} // namespace bvh::v2

#endif
