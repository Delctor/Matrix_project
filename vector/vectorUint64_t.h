#pragma once
#include <initializer.h>
#include <vectorDouble.h>
#include <vectorFloat.h>
#include <vectorInt.h>
#include <vectorUint8_t.h>

namespace alge
{
	template <>
	class vector<uint64_t>
	{
	public:
		inline vector();

		inline vector(size_t);

		inline vector(uint64_t*, size_t);

		inline vector(std::initializer_list<uint64_t>);

		inline ~vector();

		// Friend classes

		template <typename T, bool tranposed, bool contiguous>
		friend class matrix;

		template <typename T>
		friend class vector;

		// Friend functions

		friend inline vector<uint64_t> where(vector<uint8_t>&, vector<uint64_t>&, vector<uint64_t>&);

		friend inline vector<uint64_t> where(vector<uint8_t>&, uint64_t, uint64_t);

		friend inline vector<uint64_t> where(vector<uint8_t>&, vector<uint64_t>&, uint64_t);

		friend inline vector<uint64_t> where(vector<uint8_t>&, uint64_t, vector<uint64_t>&);

		friend inline vector<uint64_t> where(vector<uint8_t>&);

		friend std::ostream& operator<<(std::ostream& os, const vector<uint64_t>& vector);

		friend inline vector<uint64_t> operator+(uint64_t, vector<uint64_t>&);

		friend inline vector<uint64_t> operator-(uint64_t, vector<uint64_t>&);

		friend inline vector<uint64_t> operator/(uint64_t, vector<uint64_t>&);

		friend inline vector<uint64_t> operator*(uint64_t, vector<uint64_t>&);

		friend inline vector<uint8_t> operator==(uint64_t, vector<uint64_t>&);

		friend inline vector<uint8_t> operator!=(uint64_t, vector<uint64_t>&);

		friend inline vector<uint8_t> operator>(uint64_t, vector<uint64_t>&);

		friend inline vector<uint8_t> operator>=(uint64_t, vector<uint64_t>&);

		friend inline vector<uint8_t> operator<(uint64_t, vector<uint64_t>&);

		friend inline vector<uint8_t> operator<=(uint64_t, vector<uint64_t>&);

		template<typename T>
		friend inline size_t upper_bound(vector<T>&, size_t, size_t, T);

		template<typename T>
		friend inline size_t lower_bound(vector<T>&, size_t, size_t, T);

		template<typename T>
		friend inline vector<T> concatenate(vector<T>&, vector<T>&);

		// Block

		inline vector<uint64_t> block(size_t, size_t);

		// Copy

		inline vector<uint64_t> copy();

		// = 

		inline vector<uint64_t>& operator=(vector<uint64_t>&);

		// Transfer

		inline void transfer(vector<uint64_t>&);

		// Set Constant

		inline void set_const(uint64_t);

		inline uint64_t* data();

		inline size_t size();

		inline uint64_t& operator[](size_t);

		inline const uint64_t& operator[](size_t) const;

		inline size_t capacity();

		template<bool reduceCapacity = true>
		inline void clear();

		inline void reserve(size_t);

		inline void append(uint64_t);

		inline void append(std::initializer_list<uint64_t>);

		inline void append(vector<uint64_t>&);

		inline void insert(uint64_t, size_t);

		inline void erase(size_t);

		template<bool binarySearch = false>
		inline size_t find(uint64_t);

		// +

		inline vector<uint64_t> operator+(vector<uint64_t>&);

		inline vector<uint64_t> operator+(uint64_t);

		inline void operator+=(vector<uint64_t>&);

		inline void operator+=(uint64_t);

		// -

		inline vector<uint64_t> operator-(vector<uint64_t>&);

		inline vector<uint64_t> operator-(uint64_t);

		inline void operator-=(vector<uint64_t>&);

		inline void operator-=(uint64_t);

		// *

		inline vector<uint64_t> operator*(vector<uint64_t>&);

		inline vector<uint64_t> operator*(uint64_t);

		inline void operator*=(vector<uint64_t>&);

		inline void operator*=(uint64_t);

		// /

		inline vector<uint64_t> operator/(vector<uint64_t>&);

		inline vector<uint64_t> operator/(uint64_t);

		inline void operator/=(vector<uint64_t>&);

		inline void operator/=(uint64_t);

		// ==

		inline vector<uint8_t> operator==(vector<uint64_t>&);

		inline vector<uint8_t> operator==(uint64_t);

		// !=

		inline vector<uint8_t> operator!=(vector<uint64_t>&);

		inline vector<uint8_t> operator!=(uint64_t);

		// >

		inline vector<uint8_t> operator>(vector<uint64_t>&);

		inline vector<uint8_t> operator>(uint64_t);

		// < 

		inline vector<uint8_t> operator<(vector<uint64_t>&);

		inline vector<uint8_t> operator<(uint64_t);

		// >=

		inline vector<uint8_t> operator>=(vector<uint64_t>&);

		inline vector<uint8_t> operator>=(uint64_t);

		// <=

		inline vector<uint8_t> operator<=(vector<uint64_t>&);

		inline vector<uint8_t> operator<=(uint64_t);

		// Functions

		// Pow

		inline vector<uint64_t> pow(uint64_t);

		inline vector<uint64_t> pow(vector<uint64_t>&);

		inline void self_pow(uint64_t);

		inline void self_pow(vector<uint64_t>&);

		// <<

		inline vector<uint64_t> operator<<(int);

		inline vector<uint64_t> operator<<(vector<uint64_t>&);

		inline void operator<<=(int);

		inline void operator<<=(vector<uint64_t>&);

		// >>

		inline vector<uint64_t> operator>>(int);

		inline vector<uint64_t> operator>>(vector<uint64_t>&);

		inline void operator>>=(int);

		inline void operator>>=(vector<uint64_t>&);

		// Sort

		inline void sort();

		// Argsort

		inline vector<uint64_t> argsort();

		// Cast

		template<typename T>
		inline vector<T> cast();

	private:
		uint64_t* _data;
		uint64_t* dataToDelete;
		size_t _size;
		size_t finalPos;
		size_t _capacity;
	};
}

