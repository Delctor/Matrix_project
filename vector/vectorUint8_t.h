#pragma once
#include <initializer.h>
#include <vectorDouble.h>
#include <vectorFloat.h>
#include <vectorUint64_t.h>
#include <vectorInt.h>

namespace alge
{
	template <>
	class vector<uint8_t>
	{
	public:
		/*I am using uint8_t to emulate bool that is why this class has no arithmetic operation methods*/

		inline vector();

		inline vector(size_t size);

		inline vector(uint8_t* data, size_t size);

		inline vector(std::initializer_list<uint8_t> list);

		inline ~vector();

		// Friend classes

		template <typename T, bool tranposed, bool contiguous>
		friend class matrix;

		template <typename T>
		friend class vector;

		// Friend functions

		// These functions are like numpy.where

		friend inline vector<double> where(vector<uint8_t>&, vector<double>&, vector<double>&);

		friend inline vector<float> where(vector<uint8_t>&, vector<float>&, vector<float>&);

		friend inline vector<uint64_t> where(vector<uint8_t>&, vector<uint64_t>&, vector<uint64_t>&);

		friend inline vector<int> where(vector<uint8_t>&, vector<int>&, vector<int>&);

		// Double

		friend inline vector<double> where(vector<uint8_t>&, double, double);

		friend inline vector<double> where(vector<uint8_t>&, vector<double>&, double);

		friend inline vector<double> where(vector<uint8_t>&, double, vector<double>&);

		// Float

		friend inline vector<float> where(vector<uint8_t>&, float, float);

		friend inline vector<float> where(vector<uint8_t>&, vector<float>&, float);

		friend inline vector<float> where(vector<uint8_t>&, float, vector<float>&);

		// Int

		friend inline vector<int> where(vector<uint8_t>&, int, int);

		friend inline vector<int> where(vector<uint8_t>&, vector<int>&, int);

		friend inline vector<int> where(vector<uint8_t>&, int, vector<int>&);

		// uint64_t

		friend inline vector<uint64_t> where(vector<uint8_t>&, uint64_t, uint64_t);

		friend inline vector<uint64_t> where(vector<uint8_t>&, vector<uint64_t>&, uint64_t);

		friend inline vector<uint64_t> where(vector<uint8_t>&, uint64_t, vector<uint64_t>&);

		friend inline vector<uint64_t> where(vector<uint8_t>&);

		// Binary search

		template<typename T>
		friend inline size_t upper_bound(vector<T>&, size_t, size_t, T);

		template<typename T>
		friend inline size_t lower_bound(vector<T>&, size_t, size_t, T);

		// Cout

		friend std::ostream& operator<<(std::ostream&, const vector<uint8_t>&);

		template<typename T>
		friend inline vector<T> concatenate(vector<T>&, vector<T>&);

		// Block

		inline vector<uint8_t> block(size_t, size_t);

		// Copy

		inline vector<uint8_t> copy();

		// Set Constant

		inline void set_const(uint8_t);

		// =

		inline vector<uint8_t>& operator=(vector<uint8_t>&);

		// Transfer

		inline void transfer(vector<uint8_t>&);

		inline uint8_t* data();

		inline size_t size();

		inline uint8_t& operator[](size_t);

		inline const uint8_t& operator[](size_t) const;

		inline vector<uint8_t> operator[](vector<uint64_t>&);

		inline size_t capacity();

		template<bool reduceCapacity = true>
		inline void clear();

		inline void reserve(size_t);

		inline void append(uint8_t);

		inline void append(std::initializer_list<uint8_t>);

		inline void append(vector<uint8_t>&);

		inline void insert(uint8_t, size_t);

		inline void erase(size_t);

		// &&

		inline vector<uint8_t> operator&&(vector<uint8_t>&);

		inline vector<uint8_t> operator&&(uint8_t);

		// ||

		inline vector<uint8_t> operator||(vector<uint8_t>&);

		inline vector<uint8_t> operator||(uint8_t);

		// !

		inline vector<uint8_t> operator!();

		inline void self_not();

		// Count

		inline uint64_t count();

		// sort

		inline void sort();

		// Argsort

		inline vector<uint64_t> argsort();

		// Cast

		template<typename T>
		inline vector<T> cast();

	private:
		uint8_t* _data;
		uint8_t* dataToDelete;
		size_t _size;
		size_t finalPos, finalPos256;
		size_t _capacity;
	};
}

