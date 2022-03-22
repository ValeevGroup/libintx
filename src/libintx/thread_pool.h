#ifndef LIBINTX_THREAD_POOL_H
#define LIBINTX_THREAD_POOL_H

#include <thread>
#include <mutex>
#include <utility>
#include <tuple>
#include <queue>
#include <atomic>
#include <future>

namespace libintx {

  struct thread_pool {

    thread_pool(size_t threads = std::thread::hardware_concurrency()) {
      if (!threads) threads = std::thread::hardware_concurrency();
      create_threads(threads);
    }

    ~thread_pool() {
      wait_nothrow();
      active_ = false;
      destroy_threads();
    }

    void wait_nothrow() {
      while (remaining_) {
         sleep_or_yield();
       }
    }

    void wait() {
      while (remaining_) {
         if (this->exception_.ptr) break;
         sleep_or_yield();
      }
      std::unique_lock exception_lock(exception_.mutex);
      if (exception_.ptr) {
         auto exception = exception_.ptr;
         exception_.ptr = nullptr;
         std::rethrow_exception(exception);
       }
    }

    template<typename F, typename ... Args>
    auto push_task(const F &f, Args&& ... args) {
      using R = std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>;
      auto promise = std::make_shared< std::promise<R> >();
      auto future = promise->get_future();
      auto task = [this,f,args...,promise]() mutable {
        try {
          if constexpr (std::is_same_v<R,void>) {
            f(args...);
            promise->set_value();
          }
          else {
            promise->set_value(f(args...));
          }
        }
        catch (...) {
          auto exception = std::current_exception();
          try {
            promise->set_exception(exception);
          }
          catch (...) {
          }
          if (!exception_.ptr) {
            std::unique_lock lock(exception_.mutex);
            if (!exception_.ptr) exception_.ptr = exception;
          }
        }
      };
      push(std::move(task));
      return future;
    }

  private:

    void create_threads(size_t threads) {
      threads_.clear();
      for (size_t i = 0; i < threads; ++i) {
        threads_.emplace_back(&thread_pool::task_thread, this);
      }
    }

    void destroy_threads() {
      for (auto &thread : threads_) {
        thread.join();
      }
      threads_.clear();
    }

    void sleep_or_yield() {
      std::this_thread::yield();
    }

    void task_thread() {
      while (active_) {
        auto task = pop();
        if (task) {
          task();
          --remaining_;
        }
        else {
          sleep_or_yield();
        }
      }
    }

    void push(std::function<void()> f) {
      std::unique_lock lock(mutex_);
      tasks_.push(f);
      ++remaining_;
    }

    std::function<void()> pop() {
      std::unique_lock lock(mutex_);
      if (tasks_.empty()) return nullptr;
      auto f = tasks_.front();
      tasks_.pop();
      return f;
    }

  private:
    std::vector<std::thread> threads_;
    std::queue< std::function<void()> > tasks_;
    std::mutex mutex_;
    std::atomic<size_t> remaining_ = 0;
    std::atomic<bool> active_ = true;
    struct {
      std::mutex mutex;
      std::exception_ptr ptr;
    } exception_;

  };

}

#endif /* LIBINTX_THREAD_POOL_H */
