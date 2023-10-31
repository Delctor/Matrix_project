#pragma once
#include <initializer.h>
#include <vectorDouble.h>
#include <vectorUint64_t.h>
#include <vectorInt.h>
#include <vectorUint8_t.h>

namespace alge
{
	template <>
	class vector<float>
	{
	public:
		inline vector();

		inline vector(size_t);

		inline vector(float*, size_t);

		inline vector(std::initializer_list<float>);

		inline ~vector();

		template <typename T, bool tranposed, bool contiguous>
		friend class matrix;

		template <typename T>
		friend class vector;

		friend std::ostream& operator<<(std::ostream&, const vector<float>&);

		friend inline vector<float> where(vector<uint8_t>&, vector<float>&, vector<float>&);

		friend inline vector<float> where(vector<uint8_t>&, float, float);

		friend inline vector<float> where(vector<uint8_t>&, vector<float>&, float);

		friend inline vector<float> where(vector<uint8_t>&, float, vector<float>&);

		friend inline float dot(vector<float>&, vector<float>&);

		friend inline vector<float> operator+(float, vector<float>&);

		friend inline vector<float> operator-(float, vector<float>&);

		friend inline vector<float> operator/(float, vector<float>&);

		friend inline vector<float> operator*(float, vector<float>&);

		friend inline vector<uint8_t> operator==(float, vector<float>&);

		friend inline vector<uint8_t> operator!=(float, vector<float>&);

		friend inline vector<uint8_t> operator>(float, vector<float>&);

		friend inline vector<uint8_t> operator>=(float, vector<float>&);

		friend inline vector<uint8_t> operator<(float, vector<float>&);

		friend inline vector<uint8_t> operator<=(float, vector<float>&);

		template<typename T>
		friend inline size_t upper_bound(vector<T>&, size_t, size_t, T);

		template<typename T>
		friend inline size_t lower_bound(vector<T>&, size_t, size_t, T);

		template<typename T>
		friend inline vector<T> concatenate(vector<T>&, vector<T>&);

		// Block

		inline vector<float> block(size_t, size_t);

		// Copy

		inline vector<float> copy();

		// =

		inline vector<float>& operator=(vector<float>&);

		// Transfer

		inline void transfer(vector<float>&);

		inline float& operator[](size_t);

		inline const float& operator[](size_t) const;

		inline vector<float> operator[](vector<uint64_t>&);

		inline float* data();

		inline size_t capacity();

		template<bool reduceCapacity = true>
		inline void clear();

		inline void reserve(size_t);

		inline void append(float);

		inline void append(std::initializer_list<float>);

		inline void append(vector<float>&);

		inline void insert(float, size_t);

		inline void erase(size_t);

		template<bool binarySearch = false>
		inline size_t find(float);

		// neg

		inline vector<float> operator-();

		inline void self_neg();
		
		// Set Constant

		inline void set_const(float);

		// Rand

		inline void rand();

		// +

		inline vector<float> operator+(vector<float>&);

		inline vector<float> operator+(float);

		inline void operator+=(vector<float>&);
		
		inline void operator+=(float);

		// -

		inline vector<float> operator-(vector<float>&);

		inline vector<float> operator-(float);

		inline void operator-=(vector<float>&);

		inline void operator-=(float);

		// *

		inline vector<float> operator*(vector<float>&);

		inline vector<float> operator*(float);

		inline void operator*=(vector<float>&);

		inline void operator*=(float);

		// /

		inline vector<float> operator/(vector<float>&);

		inline vector<float> operator/(float);
		
		inline void operator/=(vector<float>&);

		inline void operator/=(float);

		// ==

		inline vector<uint8_t> operator==(vector<float>&);

		inline vector<uint8_t> operator==(float);

		// !=

		inline vector<uint8_t> operator!=(vector<float>&);

		inline vector<uint8_t> operator!=(float);

		// >

		inline vector<uint8_t> operator>(vector<float>&);

		inline vector<uint8_t> operator>(float);

		// >=

		inline vector<uint8_t> operator>=(vector<float>&);

		inline vector<uint8_t> operator>=(float);

		// <

		inline vector<uint8_t> operator<(vector<float>&);

		inline vector<uint8_t> operator<(float);

		// <=

		inline vector<uint8_t> operator<=(vector<float>&);

		inline vector<uint8_t> operator<=(float);

		// Functions

		// Pow

		inline vector<float> pow(float);

		inline vector<float> pow(vector<float>&);

		inline void self_pow(float);

		inline void self_pow(vector<float>&);

		// Root

		inline vector<float> root(float);

		inline vector<float> root(vector<float>&);

		inline void self_root(float);

		inline void self_root(vector<float>&);

		// Log

		inline vector<float> log();

		inline void self_log();

		// Log2

		inline vector<float> log2();

		inline void self_log2();

		// Log10

		inline vector<float> log10();

		inline void self_log10();

		// Exp

		inline vector<float> exp();

		inline void self_exp();

		// Exp2

		inline vector<float> exp2();
		
		inline void self_exp2();

		// Tan

		inline vector<float> tan();

		inline void self_tan();

		// Cos

		inline vector<float> cos();

		inline void self_cos();

		// Acos

		inline vector<float> acos();

		inline void self_acos();

		// Atan

		inline vector<float> atan();

		inline void self_atan();

		// Abs

		inline vector<float> abs();

		inline void self_abs();

		// Round

		inline vector<float> round();

		inline void self_round();

		// Floor

		inline vector<float> floor();

		inline void self_floor();

		// Ceil

		inline vector<float> ceil();

		inline void self_ceil();

		// Max

		inline float max();

		inline uint64_t argmax();

		// Min

		inline float min();
		
		inline uint64_t argmin();

		// Sum
		
		inline float sum();

		// Mean

		inline float mean();

		// Std

		inline float std(float, float*);

		// Activation Functions

		// Tanh

		inline vector<float> tanh();

		inline void self_tanh();

		// Cosh

		inline vector<float> cosh();

		inline void self_cosh();

		// ReLU

		inline vector<float> relu();

		inline void self_relu();

		// LReLU

		inline vector<float> lrelu();

		inline void self_lrelu();

		// Sigmoid

		inline vector<float> sigmoid();

		inline void self_sigmoid();

		// Softplus

		inline vector<float> softplus();

		inline void self_softplus();

		// Softmax

		inline vector<float> softmax();

		// Sort

		inline void sort();

		// Argsort

		inline vector<uint64_t> argsort();

		// Cast

		template <typename T>
		inline vector<T> cast();

	private:
		float* _data;
		float* dataToDelete;
		size_t _size;
		size_t finalPos;
		size_t _capacity;
	};
}
