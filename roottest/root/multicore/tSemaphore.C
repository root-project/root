#include "TSemaphore.h"
#include "TSystem.h"

#include <atomic>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>

bool notify = 0;
std::atomic_int	 cnt(0);
std::condition_variable cond;
std::mutex		 mtx, pmtx;
TSemaphore sem(0);

void consumer()
{
	std::thread::id this_id = std::this_thread::get_id();
    if (notify) { std::unique_lock<std::mutex> l(pmtx);
                  std::cout << "consumer: started, id:" << this_id << std::endl; }

	if (sem.Wait() == 0) cnt++;
    if (notify) { std::unique_lock<std::mutex> l(pmtx);
    	          std::cout << "consumer: signaled, id:" << this_id << std::endl; }
}

void producer()
{
	std::thread::id this_id = std::this_thread::get_id();
    if (notify) { std::unique_lock<std::mutex> l(pmtx);
    	          std::cout << "producer: started, id:" << this_id << std::endl; }

    std::unique_lock<std::mutex> lk(mtx);
    cond.wait(lk);
    sem.Post();
    if (notify) { std::unique_lock<std::mutex> l(pmtx);
    	          std::cout << "producer: posted, id:" << this_id << std::endl; }
}

void tSemaphore(bool deb = 0, int nth = 5)
{

    notify = deb;

    std::vector<std::thread> consumers, producers;

	for (unsigned int i = 0; i < nth; i++) {
        consumers.emplace_back(consumer);
	}

	for (unsigned int i = 0; i < nth; i++) {
        producers.emplace_back(producer);
	}

    gSystem->Sleep(1000);
	cond.notify_all();
    gSystem->Sleep(1000);

    for (auto && cons : consumers) cons.join();
    for (auto && prods : producers) prods.detach();

    std::cout << "main: "<< cnt << " consumer threads signaled" << std::endl;
}