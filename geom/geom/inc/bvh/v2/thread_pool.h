#ifndef BVH_V2_THREAD_POOL_H
#define BVH_V2_THREAD_POOL_H

#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <queue>
#include <functional>

namespace bvh::v2 {

class ThreadPool {
public:
    using Task = std::function<void(size_t)>;

    /// Creates a thread pool with the given number of threads (a value of 0 tries to autodetect
    /// the number of threads and uses that as a thread count).
    ThreadPool(size_t thread_count = 0) { start(thread_count); }

    ~ThreadPool() {
        wait();
        stop();
        join();
    }

    inline void push(Task&& fun);
    inline void wait();

    size_t get_thread_count() const { return threads_.size(); }

private:
    static inline void worker(ThreadPool*, size_t);

    inline void start(size_t);
    inline void stop();
    inline void join();

    int busy_count_ = 0;
    bool should_stop_ = false;
    std::mutex mutex_;
    std::vector<std::thread> threads_;
    std::condition_variable avail_;
    std::condition_variable done_;
    std::queue<Task> tasks_;
};

void ThreadPool::push(Task&& task) {
    {
        std::unique_lock<std::mutex> lock(mutex_);
        tasks_.emplace(std::move(task));
    }
    avail_.notify_one();
}

void ThreadPool::wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [this] { return busy_count_ == 0 && tasks_.empty(); });
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
        task(thread_id);
        {
            std::unique_lock<std::mutex> lock(pool->mutex_);
            pool->busy_count_--;
        }
        pool->done_.notify_one();
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

} // namespace bvh::v2

#endif
