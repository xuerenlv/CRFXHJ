#ifndef PARALLEL_H
#define PARALLEL_H

#include <iostream>
#include <thread>
#include <mutex>
#include <functional>
#include <algorithm>
#include <vector>
#include <deque>
#include <list>
namespace Parallel {

struct Parallel {
protected:
	//how many threads to divide ourselves among
	int numThreads;

	//deque of tasks
	std::deque<std::function<void()>> tasks;

	//vector of worker threads
	std::vector<std::thread> workers;

	//one per thread ... ?  semaphore instead?
	std::vector<std::mutex> doneMutexes;

	//flag whether each thread needs to unlock their doneMutex if it finds an empty queue.
	//set by singleton before populating tasks, cleared by each thread once an empty queue is found.
	//access is protected by taskMutex.
	std::vector<char> needToUnlockDone;

	//lock/unlock surrounding access of the 'tasks' deque
	// and surrounding access of 'needToUnlock'
	std::mutex tasksMutex;

	//singleton acquires when filling task queue & blocks on.
	//last thread releases when emptying the task queue.
	std::mutex doneMutex;

	bool done;
public:
	Parallel(int numThreads_ = 4)
	: numThreads(numThreads_)
	, workers(numThreads)
	, doneMutexes(numThreads)
	, needToUnlockDone(numThreads)
	, done(false)
	{
		for (int i = 0; i < numThreads; ++i) {
			doneMutexes[i].lock();
			needToUnlockDone[i] = false;
		}
		for (int i = 0; i < numThreads; ++i) {
			workers[i] = std::thread([&,i](){
				while (true) {

					//upon 'queue' signal ...
					std::function<void()> next;
					bool gotEmpty = false;
					{
						std::lock_guard<std::mutex> taskLock(tasksMutex);
						if (done) return;
						if (!tasks.empty()) {
							next = tasks.front();
							tasks.pop_front();
						}
						//if we got the last one then say so
						if (tasks.empty() && needToUnlockDone[i]) {
							needToUnlockDone[i] = false;
							gotEmpty = true;
						}
					}
					if (next) {
						next();
					}
					if (gotEmpty) {
						doneMutexes[i].unlock();
					}
				}
			});
		}
	}

	~Parallel() {
		done = true;	//protect this if you want
		for (std::thread &worker : workers) {
			worker.join();
		}
	}

	template<typename Iterator, typename Callback>
	void foreach(Iterator begin, Iterator end, Callback callback) {
		//spawn
		{
			//prep
			std::lock_guard<std::mutex> taskLock(tasksMutex);		//acquire task mutex
			for (char &flag : needToUnlockDone) { flag = true; }	//clear all thread done flags

			//add threads
			int totalRange = end - begin;
			for (int i = 0; i < numThreads; ++i) {
				int beginIndex = i * totalRange / numThreads;
				int endIndex = (i + 1) * totalRange / numThreads;

				//std::cout << beginIndex << " " << endIndex << " " << endIndex-beginIndex << " " << totalRange << std::endl;
				tasks.push_back([=]() {
					std::for_each(begin + beginIndex, begin + endIndex, callback);
				});
			}
		}

		//join
		for (int i = 0; i < (int)doneMutexes.size(); ++i) {
			doneMutexes[i].lock();
		}
	};

	// a shy step away from std::accumulate
	// in that the values in the iterator are mapped first (via callback)
	// before they are accumulated.
	// I could make a new structure for buffering my intermediate values, but I don't really want to.
	template<
		typename Iterator,
		typename Result,
		typename Callback = std::function<Result (typename std::iterator_traits<Iterator>::value_type &)>,
		typename Combiner = std::function<Result (Result, Result)>
	>
	Result reduce(
		Iterator begin,
		Iterator end,
		Callback callback,
		Result initialValue = Result(),
		Combiner combiner = std::plus<Result>())
	{
		std::vector<Result> results(numThreads);

		//spawn
		{
			std::lock_guard<std::mutex> taskLock(tasksMutex);		//acquire task mutex
			for (char &flag : needToUnlockDone) { flag = true; }	//clear all thread done flags

			int totalRange = end - begin;
			for (int i = 0; i < numThreads; ++i) {
				int beginIndex = i * totalRange / numThreads;
				int endIndex = (i + 1) * totalRange / numThreads;
				tasks.push_back([&,beginIndex,endIndex,i]() {
					results[i] = initialValue;
					std::for_each(begin + beginIndex, begin + endIndex, [&](
						typename std::iterator_traits<Iterator>::value_type &value)
					{
						results[i] = combiner(results[i], callback(value));
					});
				});
			}
		}

		//join
		for (int i = 0; i < (int)doneMutexes.size(); ++i) {
			doneMutexes[i].lock();
		}

		//combine result
		for (int i = 0; i < numThreads; ++i) {
			initialValue = combiner(initialValue, results[i]);
		}

		return initialValue;
	}
};

};

#endif // PARALLEL_H
