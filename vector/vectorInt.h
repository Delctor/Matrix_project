#pragma once
#include <initializer.h>
#include <vectorDouble.h>
#include <vectorFloat.h>
#include <vectorUint64_t.h>
#include <vectorUint8_t.h>

namespace alge
{
	template <>
	class vector<int>
	{
	public:
		inline vector();

		inline vector(size_t);

		inline vector(int*, size_t);

		inline vector(std::initializer_list<int>);

		inline ~vector();

		// Friend classes

		template <typename T, bool tranposed, bool contiguous>
		friend class matrix;

		template <typename T>
		friend class vector;

		// Friend functions

		friend inline vector<int> where(vector<uint8_t>&, vector<int>&, vector<int>&);

		friend inline vector<int> where(vector<uint8_t>&, int, int);

		friend inline vector<int> where(vector<uint8_t>&, vector<int>&, int);

		friend inline vector<int> where(vector<uint8_t>&, int, vector<int>&);

		friend std::ostream& operator<<(std::ostream&, const vector<int>&);

		template<typename T>
		friend inline vector<T> concatenate(vector<T>&, vector<T>&);

		friend inline vector<int> operator+(int, vector<int>&);

		friend inline vector<int> operator-(int, vector<int>&);

		friend inline vector<int> operator/(int, vector<int>&);

		friend inline vector<int> operator*(int, vector<int>&);

		friend inline vector<uint8_t> operator==(int, vector<int>&);

		friend inline vector<uint8_t> operator!=(int, vector<int>&);

		friend inline vector<uint8_t> operator>(int, vector<int>&);

		friend inline vector<uint8_t> operator>=(int, vector<int>&);

		friend inline vector<uint8_t> operator<(int, vector<int>&);

		friend inline vector<uint8_t> operator<=(int, vector<int>&);

		template<typename T>
		friend inline size_t upper_bound(vector<T>&, size_t, size_t, T);

		template<typename T>
		friend inline size_t lower_bound(vector<T>&, size_t, size_t, T);

		// -----

		// Block

		inline vector<int> block(size_t, size_t);

		// Copy

		inline vector<int> copy();

		// =

		inline vector<int>& operator=(vector<int>&);

		// Transfer

		inline void transfer(vector<int>&);

		inline int* data();

		inline size_t size();

		inline int& operator[](size_t);

		inline const int& operator[](size_t) const;

		inline vector<int> operator[](vector<uint64_t>&);

		inline size_t capacity();

		template<bool reduceCapacity = true>
		inline void clear();

		inline void reserve(size_t);

		inline void append(int);

		inline void append(std::initializer_list<int>);

		inline void append(vector<int>&);

		inline void insert(int, size_t);

		inline void erase(size_t);

		template<bool binarySearch = false>
		inline size_t find(int);

		// neg

		inline vector<int> operator-();

		inline void self_neg();

		// Set Constant

		inline void set_const(int);

		// +

		inline vector<int> operator+(vector<int>&);

		inline vector<int> operator+(int);

		inline void operator+=(vector<int>&);

		inline void operator+=(int);

		// -

		inline vector<int> operator-(vector<int>&);

		inline vector<int> operator-(int);

		inline void operator-=(vector<int>&);

		inline void operator-=(int);

		// *

		inline vector<int> operator*(vector<int>&);

		inline vector<int> operator*(int);

		inline void operator*=(vector<int>&);

		inline void operator*=(int);

		// /

		inline vector<int> operator/(vector<int>&);

		inline vector<int> operator/(int);

		inline void operator/=(vector<int>&);

		inline void operator/=(int);

		// ==

		inline vector<uint8_t> operator==(vector<int>&);

		inline vector<uint8_t> operator==(int);

		// !=

		inline vector<uint8_t> operator!=(vector<int>&);

		inline vector<uint8_t> operator!=(int);

		// >

		inline vector<uint8_t> operator>(vector<int>&);

		inline vector<uint8_t> operator>(int);

		// < 

		inline vector<uint8_t> operator<(vector<int>&);

		inline vector<uint8_t> operator<(int);

		// >=

		inline vector<uint8_t> operator>=(vector<int>&);

		inline vector<uint8_t> operator>=(int);

		// <=

		inline vector<uint8_t> operator<=(vector<int>&);

		inline vector<uint8_t> operator<=(int);

		// Pow

		inline vector<int> pow(int);

		inline vector<int> pow(vector<int>&);

		inline void self_pow(int);

		inline void self_pow(vector<int>&);

		// Abs

		inline vector<int> abs();

		inline void self_abs();

		// Sort

		inline void sort();

		// Argsort

		inline vector<uint64_t> argsort();

		// Cast

		template<typename T>
		inline vector<T> cast();

	private:
		int* _data;
		int* dataToDelete;
		size_t _size;
		size_t finalPos;
		size_t _capacity;
	};
}