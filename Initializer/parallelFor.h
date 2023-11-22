#pragma once
#include <iostream>
#include <ppl.h>
#include <Windows.h>
#include <thread>

namespace alge
{
	namespace parallel
	{
		struct loopParameters
		{
			size_t
				start,
				end,
				step;
			void* extraParameters;
		};

		struct threadsInfoStruct
		{
			uint32_t nThreads = std::thread::hardware_concurrency();
			HANDLE* threads = new HANDLE[nThreads];
			loopParameters* parameters = new loopParameters[nThreads];
			~threadsInfoStruct() { delete[] threads, parameters; };
		};

		threadsInfoStruct threadsInfo;

		// start, end, step
		typedef void (*loopFunctionType)(loopParameters*);

		void parallelFor(loopFunctionType loopFunction, size_t start, size_t end, size_t step, void* extraParameters = nullptr)
		{
			size_t space = (end - start);
			size_t actualSteps = space / threadsInfo.nThreads;
			int64_t remainder = space % threadsInfo.nThreads;
			size_t lastEnd = 0;

			size_t i = 0;
			for (; i < threadsInfo.nThreads && !(actualSteps == 0 && remainder == 0); i++)
			{
				threadsInfo.parameters[i].start = lastEnd;

				lastEnd += actualSteps;

				if (remainder > 0) { lastEnd++; remainder--; }

				threadsInfo.parameters[i].end = lastEnd;
				threadsInfo.parameters[i].step = step;
				threadsInfo.parameters[i].extraParameters = extraParameters;

				threadsInfo.threads[i] = CreateThread(NULL, 0, reinterpret_cast<LPTHREAD_START_ROUTINE>(loopFunction), &threadsInfo.parameters[i], 0, NULL);
			}
			WaitForMultipleObjects(threadsInfo.nThreads, threadsInfo.threads, true, INFINITE);

			for (size_t j = 0; j < i; j++)
			{
				CloseHandle(threadsInfo.threads[j]);
			}

		}
	}
}
