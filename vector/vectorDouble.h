#pragma once
#include <initializer.h>
#include <vectorFloat.h>
#include <vectorUint64_t.h>
#include <vectorInt.h>
#include <vectorUint8_t.h>

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

namespace alge
{
	template <>
	class vector<double>
	{
	public:
		inline vector();

		inline vector(size_t size);

		inline vector(double* data, size_t size);

		inline vector(std::initializer_list<double> list);

		inline ~vector();

		// Friend class

		template <typename T, bool tranposed, bool contiguous>
		friend class matrix;

		template <typename T>
		friend class vector;

		// Friend function

		friend inline vector<double> where(vector<uint8_t>&, vector<double>&, vector<double>&);

		friend inline vector<double> where(vector<uint8_t>&, vector<double>&, double);

		friend inline vector<double> where(vector<uint8_t>&, double, vector<double>&);

		friend inline vector<double> where(vector<uint8_t>&, double, double);

		friend inline double dot(vector<double>&, vector<double>&);

		friend inline vector<double> operator+(double, vector<double>&);

		friend inline vector<double> operator-(double, vector<double>&);

		friend inline vector<double> operator/(double, vector<double>&);

		friend inline vector<double> operator*(double, vector<double>&);

		friend inline vector<uint8_t> operator==(double, vector<double>&);

		friend inline vector<uint8_t> operator!=(double, vector<double>&);

		friend inline vector<uint8_t> operator>(double, vector<double>&);

		friend inline vector<uint8_t> operator>=(double, vector<double>&);

		friend inline vector<uint8_t> operator<(double, vector<double>&);

		friend inline vector<uint8_t> operator<=(double, vector<double>&);

		template<bool useSteps, bool thisContiguous>
		friend inline matrix<double> randomGenerator(matrix<double, false, thisContiguous>&, size_t);

		template<bool matrix1Contiguous, bool matrix2Contiguous>
		friend inline vector<double> kernelDensity(matrix<double, false, matrix1Contiguous>&, matrix<double, false, matrix2Contiguous>&, double);

		template<typename T>
		friend inline size_t upper_bound(vector<T>&, size_t, size_t, T);

		template<typename T>
		friend inline size_t lower_bound(vector<T>&, size_t, size_t, T);

		friend std::ostream& operator<<(std::ostream&, const vector<double>&);

		template<typename T>
		friend inline vector<T> concatenate(vector<T>&, vector<T>&);

		template<typename T>
		friend inline vector<T> concatenate(vector<T>**, size_t, size_t);

		template<typename T>
		friend inline vector<T> concatenate(vector<T>**, size_t);

		template<typename T>
		friend inline void vectorize(vector<T>&, T(T));

		template<typename T, typename T2>
		friend inline void vectorize(vector<T>&, T2(T2));

		// Block

		inline vector<double> block(size_t initial, size_t final);

		// Copy

		inline vector<double> copy();

		// =

		inline vector<double>& operator=(vector<double>& other);

		// Transfer

		inline void transfer(vector<double>& other);

		inline double& operator[](size_t index);

		inline const double& operator[](size_t index) const;

		inline vector<double> operator[](vector<uint64_t>& indices);

		inline double* data();

		inline size_t size();

		inline size_t capacity();

		template<bool reduceCapacity = true>
		inline void clear();

		inline void reserve(size_t newCapacity);

		inline void append(double num);

		inline void append(std::initializer_list<double> list);

		inline void append(vector<double>& other);

		inline void insert(double num, size_t index);

		inline void erase(size_t index);

		template<bool binarySearch = false>
		inline size_t find(double num);

		// Rand

		inline void rand();

		// Set Constant

		inline void set_const(double num);

		// neg

		inline vector<double> operator-();

		inline void self_neg();

		// +

		template<bool parallel = false>
		inline vector<double> operator+(vector<double>& other);

		template<bool parallel = false>
		inline vector<double> operator+(double num);

		template<bool parallel = false>
		inline void operator+=(vector<double>& other);

		template<bool parallel = false>
		inline void operator+=(double num);

		// -

		template<bool parallel = false>
		inline vector<double> operator-(vector<double>& other);

		template<bool parallel = false>
		inline vector<double> operator-(double num);

		template<bool parallel = false>
		inline void operator-=(vector<double>& other);

		template<bool parallel = false>
		inline void operator-=(double num);

		// *

		template<bool parallel = false>
		inline vector<double> operator*(vector<double>& other);

		template<bool parallel = false>
		inline vector<double> operator*(double num);

		template<bool parallel = false>
		inline void operator*=(vector<double>& other);

		template<bool parallel = false>
		inline void operator*=(double num);

		// /

		template<bool parallel = false>
		inline vector<double> operator/(vector<double>& other);

		template<bool parallel = false>
		inline vector<double> operator/(double num);

		template<bool parallel = false>
		inline void operator/=(vector<double>& other);

		template<bool parallel = false>
		inline void operator/=(double num);

		// ==

		inline vector<uint8_t> operator==(vector<double>& other);

		inline vector<uint8_t> operator==(double num);

		// !=

		inline vector<uint8_t> operator!=(vector<double>& other);

		inline vector<uint8_t> operator!=(double num);

		// >

		inline vector<uint8_t> operator>(vector<double>& other);

		inline vector<uint8_t> operator>(double num);

		// >=


		inline vector<uint8_t> operator>=(vector<double>& other);

		inline vector<uint8_t> operator>=(double num);

		// <


		inline vector<uint8_t> operator<(vector<double>& other);

		inline vector<uint8_t> operator<(double num);

		// <=


		inline vector<uint8_t> operator<=(vector<double>& other);

		inline vector<uint8_t> operator<=(double num);

		// Functions

		// Pow

		inline vector<double> pow(double exponent);

		inline vector<double> pow(vector<double>& other);

		inline void self_pow(double exponent);

		inline void self_pow(vector<double>& other);

		// Root

		inline vector<double> root(double index);

		inline vector<double> root(vector<double>& other);

		inline void self_root(double index);

		inline void self_root(vector<double>& other);

		// Log

		inline vector<double> log();

		inline void self_log();

		// Log2

		inline vector<double> log2();

		inline void self_log2();

		// Log10

		inline vector<double> log10();

		inline void self_log10();

		// Exp

		inline vector<double> exp();

		inline void self_exp();

		// Exp2

		inline vector<double> exp2();

		inline void self_exp2();

		// Tan

		inline vector<double> tan();

		inline void self_tan();

		// Cos

		inline vector<double> cos();

		inline void self_cos();

		// Acos

		inline vector<double> acos();

		inline void self_acos();

		// Atan

		inline vector<double> atan();

		inline void self_atan();

		// Abs

		inline vector<double> abs();

		inline void self_abs();

		// Round

		inline vector<double> round();

		inline void self_round();

		// Floor

		inline vector<double> floor();

		inline void self_floor();

		// Ceil

		inline vector<double> ceil();
		
		inline void self_ceil();

		// Max

		inline double max();

		inline uint64_t argmax();

		// Min

		inline double min();

		inline uint64_t argmin();

		// Sum

		inline double sum();

		// Mean

		inline double mean();

		// Std

		inline double std(double ddof = 0.0, double* mean = nullptr);

		// Activation Functions

		// Tanh

		inline vector<double> tanh();

		inline void self_tanh();

		// Cosh

		inline vector<double> cosh();

		inline void self_cosh();

		// ReLU

		inline vector<double> relu();

		inline void self_relu();

		// LReLU

		inline vector<double> lrelu();

		inline void self_lrelu();

		// Sigmoid

		inline vector<double> sigmoid();

		inline void self_sigmoid();

		// Softplus

		inline vector<double> softplus();

		inline void self_softplus();

		// Softmax

		inline vector<double> softmax();

		// Sort

		inline void sort();

		// Argsort

		inline vector<uint64_t> argsort();

		// Cast

		template <typename T>
		inline vector<T> cast();

	private:
		double* _data;
		double* dataToDelete;
		size_t _size;
		size_t finalPos;
		size_t _capacity;
	};
}

