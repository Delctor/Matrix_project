#pragma once
#include <initializer.h>
#include <matrixUint8_t.h>
#include <matrixFloat.h>

namespace alge
{
	template <bool thisTransposed, bool thisContiguous>
	class matrix<double, thisTransposed, thisContiguous>
	{
	public:

		inline matrix();

		inline matrix(size_t, size_t);

		inline matrix(double*, size_t, size_t, size_t, size_t);

		inline matrix(std::initializer_list<std::initializer_list<double>>);

		inline ~matrix();

		// Friend classes

		template <typename T, bool tranposed, bool contiguous>
		friend class matrix;

		template <typename T>
		friend class vector;

		// Friend functions

		template<bool returnTransposed, bool matrix1Transposed, bool matrix1Contiguous,
			bool matrix2Transposed, bool matrix2Contiguous>
		friend inline matrix<double> dot(matrix<double, matrix1Transposed, matrix1Contiguous>&, matrix<double, matrix2Transposed, matrix2Contiguous>&);

		template<bool otherTransposed, bool otherContiguous>
		friend std::ostream& operator<<(std::ostream& os, const matrix<double, otherTransposed, otherContiguous>& matrix);

		template<bool returnTransposed, typename T, bool matrix1Transposed, bool matrix1Contiguous,
			bool matrix2Transposed, bool matrix2Contiguous>
		friend inline matrix<T> concatenate_rowwise(matrix<T, matrix1Transposed, matrix1Contiguous>&, matrix<T, matrix2Transposed, matrix2Contiguous>&);

		template<bool returnTransposed, typename T, bool matrix1Transposed, bool matrix1Contiguous,
			bool matrix2Transposed, bool matrix2Contiguous>
		friend inline matrix<T> concatenate_colwise(matrix<T, matrix1Transposed, matrix1Contiguous>&, matrix<T, matrix2Transposed, matrix2Contiguous>&);

		template<typename T, bool returnTransposed>
		friend inline matrix<T> concatenate_rowwise(void**, size_t, size_t, size_t);

		template<typename T, bool returnTransposed>
		friend inline matrix<T> concatenate_colwise(void**, size_t, size_t, size_t);

		template<bool returnTransposed, bool thisTransposed, bool thisContiguous>
		friend inline matrix<double> operator+(double, matrix<double, thisTransposed, thisContiguous>&);

		template<bool returnTransposed, bool thisTransposed, bool thisContiguous>
		friend inline matrix<double> operator-(double, matrix<double, thisTransposed, thisContiguous>&);

		template<bool returnTransposed, bool thisTransposed, bool thisContiguous>
		friend inline matrix<double> operator*(double, matrix<double, thisTransposed, thisContiguous>&);

		template<bool returnTransposed, bool thisTransposed, bool thisContiguous>
		friend inline matrix<double> operator/(double, matrix<double, thisTransposed, thisContiguous>&);

		template<bool returnTransposed, bool thisTransposed, bool thisContiguous>
		friend inline matrix<uint8_t> operator==(double, matrix<double, thisTransposed, thisContiguous>&);

		template<bool returnTransposed, bool thisTransposed, bool thisContiguous>
		friend inline matrix<uint8_t> operator!=(double, matrix<double, thisTransposed, thisContiguous>&);

		template<bool returnTransposed, bool thisTransposed, bool thisContiguous>
		friend inline matrix<uint8_t> operator>(double, matrix<double, thisTransposed, thisContiguous>&);

		template<bool returnTransposed, bool thisTransposed, bool thisContiguous>
		friend inline matrix<uint8_t> operator>=(double, matrix<double, thisTransposed, thisContiguous>&);

		template<bool returnTransposed, bool thisTransposed, bool thisContiguous>
		friend inline matrix<uint8_t> operator<(double, matrix<double, thisTransposed, thisContiguous>&);

		template<bool returnTransposed, bool thisTransposed, bool thisContiguous>
		friend inline matrix<uint8_t> operator<=(double, matrix<double, thisTransposed, thisContiguous>&);

		template<bool returnTransposed, bool matrx1Transposed, bool matrix1Contiguous>
		friend inline matrix<double> where(matrix<uint8_t, matrx1Transposed, matrix1Contiguous>&, double, double);

		template<bool returnTransposed, bool matrx1Transposed, bool matrix1Contiguous
			, bool matrx2Transposed, bool matrix2Contiguous, bool matrx3Transposed, bool matrix3Contiguous>
		friend inline matrix<double> where(matrix<uint8_t, matrx1Transposed, matrix1Contiguous>&, matrix<double, matrx2Transposed, matrix2Contiguous>&, matrix<double, matrx3Transposed, matrix3Contiguous>&);

		template<bool returnTransposed, bool matrx1Transposed, bool matrix1Contiguous
			, bool matrx2Transposed, bool matrix2Contiguous>
		friend inline matrix<double> where(matrix<uint8_t, matrx1Transposed, matrix1Contiguous>&, matrix<double, matrx2Transposed, matrix2Contiguous>&, double);

		template<bool returnTransposed, bool matrx1Transposed, bool matrix1Contiguous
			, bool matrx2Transposed, bool matrix2Contiguous>
		friend inline matrix<double> where(matrix<uint8_t, matrx1Transposed, matrix1Contiguous>&, double, matrix<double, matrx2Transposed, matrix2Contiguous>&);

		template<bool returnTransposed, bool thisTransposed, bool thisContiguous>
		friend inline matrix<double> clip(matrix<double, thisTransposed, thisContiguous>&, vector<double>&, double, double);

		template<bool useSteps, bool thisContiguous>
		friend inline matrix<double> randomGenerator(matrix<double, false, thisContiguous>&, vector<double>&, size_t);

		template<bool matrix1Contiguous, bool matrix2Contiguous>
		friend inline vector<double> kernelDensity(matrix<double, false, matrix1Contiguous>&, matrix<double, false, matrix2Contiguous>&, double bandwidth);

		//----------------

		inline size_t rows();

		inline size_t cols();

		inline double* data();

		inline matrix<double, thisTransposed, thisContiguous && !thisTransposed> row(size_t);

		inline matrix<double, thisTransposed, thisContiguous && thisTransposed> col(size_t);

		inline matrix<double, !thisTransposed, thisContiguous> tranpose();

		template<bool blockContiguous = false>
		inline matrix<double, thisTransposed, thisContiguous && blockContiguous> block(size_t, size_t, size_t, size_t);

		inline double& operator()(size_t, size_t);

		inline const double& operator()(size_t, size_t) const;

		inline size_t capacity();

		template<bool reduceCapacity = true>
		inline void clear();

		inline void reserve(size_t);

		inline void append(std::initializer_list<std::initializer_list<double>>);

		template<bool otherTransposed, bool otherContiguous>
		inline void append(matrix<double, otherTransposed, otherContiguous>&);

		inline void erase(size_t);

		inline size_t find(vector<double>&);

		template<bool otherTransposed, bool otherContiguous>
		inline vector<uint64_t> find(matrix<double, otherTransposed, otherContiguous>&);

		inline void insert(std::initializer_list<double>, size_t);

		inline void insert(vector<double>&, size_t);

		template<bool otherTransposed, bool otherContiguous>
		inline void insert(matrix<double, otherTransposed, otherContiguous>&, size_t);

		template<bool otherContiguous>
		inline vector<uint8_t> in(matrix<double, false, otherContiguous>& other);

		// Copy

		template<bool returnTransposed = false>
		inline matrix<double> copy();

		// =

		template<bool otherTransposed, bool otherContiguous>
		inline matrix<double, thisTransposed, thisContiguous>& operator=(matrix<double, otherTransposed, otherContiguous>&);

		// Transfer

		template<bool otherContiguous>
		inline void transfer(matrix<double, thisTransposed, otherContiguous>&);

		// neg

		template<bool returnTransposed = false>
		inline matrix<double> operator-();

		inline void self_neg();

		// Set constant

		inline void set_const(double);

		// Rand

		inline void rand();

		// Identity

		inline void identity();

		// +

		template<bool returnTransposed = false, bool otherTransposed, bool otherContiguous>
		inline matrix<double> operator+(const matrix<double, otherTransposed, otherContiguous>&);

		template<bool otherTransposed, bool otherContiguous>
		inline void operator+=(const matrix<double, otherTransposed, otherContiguous>&);

		template<bool returnTransposed = false>
		inline matrix<double> operator+(double);

		inline void operator+=(double);

		template<bool returnTransposed = false>
		inline matrix<double> operator+(const vector<double>&);

		inline void operator+=(const vector<double>&);

		// -

		template<bool returnTransposed = false, bool otherTransposed, bool otherContiguous>
		inline matrix<double> operator-(const matrix<double, otherTransposed, otherContiguous>&);

		template<bool otherTransposed, bool otherContiguous>
		inline void operator-=(const matrix<double, otherTransposed, otherContiguous>&);

		template<bool returnTransposed = false>
		inline matrix<double> operator-(double);

		inline void operator-=(double);

		template<bool returnTransposed = false>
		inline matrix<double> operator-(const vector<double>&);

		inline void operator-=(const vector<double>&);

		// *

		template<bool returnTransposed = false, bool otherTransposed, bool otherContiguous>
		inline matrix<double> operator*(const matrix<double, otherTransposed, otherContiguous>&);

		template<bool otherTransposed, bool otherContiguous>
		inline void operator*=(const matrix<double, otherTransposed, otherContiguous>&);

		template<bool returnTransposed = false>
		inline matrix<double> operator*(double);

		inline void operator*=(double);

		template<bool returnTransposed = false>
		inline matrix<double> operator*(const vector<double>&);

		inline void operator*=(const vector<double>&);

		// /

		template<bool returnTransposed = false, bool otherTransposed, bool otherContiguous>
		inline matrix<double> operator/(const matrix<double, otherTransposed, otherContiguous>&);

		template<bool otherTransposed, bool otherContiguous>
		inline void operator/=(const matrix<double, otherTransposed, otherContiguous>&);

		template<bool returnTransposed = false>
		inline matrix<double> operator/(double);

		inline void operator/=(double);

		template<bool returnTransposed = false>
		inline matrix<double> operator/(const vector<double>&);

		inline void operator/=(const vector<double>&);

		// ==

		template<bool returnTransposed = false, bool otherTransposed, bool otherContiguous>
		inline matrix<uint8_t> operator==(const matrix<double, otherTransposed, otherContiguous>&);

		template<bool returnTransposed = false>
		inline matrix<uint8_t> operator==(double);

		// !=

		template<bool returnTransposed = false, bool otherTransposed, bool otherContiguous>
		inline matrix<uint8_t> operator!=(const matrix<double, otherTransposed, otherContiguous>&);

		template<bool returnTransposed = false>
		inline matrix<uint8_t> operator!=(double);

		// >

		template<bool returnTransposed = false, bool otherTransposed, bool otherContiguous>
		inline matrix<uint8_t> operator>(const matrix<double, otherTransposed, otherContiguous>&);

		template<bool returnTransposed = false>
		inline matrix<uint8_t> operator>(double);

		// <

		template<bool returnTransposed = false, bool otherTransposed, bool otherContiguous>
		inline matrix<uint8_t> operator<(const matrix<double, otherTransposed, otherContiguous>&);

		template<bool returnTransposed = false>
		inline matrix<uint8_t> operator<(double);

		// >=

		template<bool returnTransposed = false, bool otherTransposed, bool otherContiguous>
		inline matrix<uint8_t> operator>=(const matrix<double, otherTransposed, otherContiguous>&);

		template<bool returnTransposed = false>
		inline matrix<uint8_t> operator>=(double);

		// <=

		template<bool returnTransposed = false, bool otherTransposed, bool otherContiguous>
		inline matrix<uint8_t> operator<=(const matrix<double, otherTransposed, otherContiguous>&);

		template<bool returnTransposed = false>
		inline matrix<uint8_t> operator<=(double);

		// Functions

		template<bool returnTransposed = false>
		inline matrix<double> exp();

		inline void self_exp();

		template<bool returnTransposed = false>
		inline matrix<double> exp2();

		inline void self_exp2();

		template<bool returnTransposed = false>
		inline matrix<double> log();

		inline void self_log();

		template<bool returnTransposed = false>
		inline matrix<double> log2();

		inline void self_log2();

		template<bool returnTransposed = false>
		inline matrix<double> log10();

		inline void self_log10();

#define _mm256_abs_pd(a) _mm256_andnot_pd(mask, (a))

		template<bool returnTransposed = false>
		inline matrix<double> abs();

		inline void self_abs();

		template<bool returnTransposed = false>
		inline matrix<double> cos();

		inline void self_cos();

		template<bool returnTransposed = false>
		inline matrix<double> tan();

		inline void self_tan();

		template<bool returnTransposed = false>
		inline matrix<double> acos();

		inline void self_acos();

		template<bool returnTransposed = false>
		inline matrix<double> round();

		inline void self_round();

		template<bool returnTransposed = false>
		inline matrix<double> floor();

		inline void self_floor();

		template<bool returnTransposed = false>
		inline matrix<double> ceil();

		inline void self_ceil();

		// pow

		template<bool returnTransposed = false>
		inline matrix<double> pow(double);

		template<bool returnTransposed = false, bool otherTransposed, bool otherContiguous>
		inline matrix<double> pow(const matrix<double, otherTransposed, otherContiguous>&);

		inline void self_pow(double);

		template<bool otherTransposed, bool otherContiguous>
		inline void self_pow(const matrix<double, otherTransposed, otherContiguous>&);

		// root

		template<bool returnTransposed = false>
		inline matrix<double> root(double);

		template<bool returnTransposed = false, bool otherTransposed, bool otherContiguous>
		inline matrix<double> root(const matrix<double, otherTransposed, otherContiguous>&);

		inline void self_root(double);
		
		template<bool otherTransposed, bool otherContiguous>
		inline void self_root(const matrix<double, otherTransposed, otherContiguous>&);

		// Mean 

		inline vector<double> mean_rowwise();

		inline vector<double> mean_colwise();

		inline double mean_all();

		template<char axis = 'a'>
		inline std::conditional<axis == 'a', double, vector<double>> mean();

		// Sum

		inline vector<double> sum_rowwise();

		inline vector<double> sum_colwise();

		inline double sum_all();

		template<char axis = 'a'>
		inline std::conditional<axis == 'a', double, vector<double>> sum();

		// Std

		inline vector<double> std_rowwise(double ddof = 0.0);

		inline vector<double> std_colwise(double ddof = 0.0);

		inline double std_all(double ddof = 0.0, double* mean = nullptr);

		template<char axis = 'a'>
		inline std::conditional<axis == 'a', double, vector<double>> std(double ddof = 0.0, double* mean = nullptr);

		// Min

		inline vector<double> min_rowwise();

		inline vector<double> min_colwise();

		inline double min_all();

		template<char axis = 'a'>
		inline std::conditional<axis == 'a', double, vector<double>> min();

		inline void argmin_all(size_t*, size_t*);

		inline vector<uint64_t> argmin_rowwise();

		inline vector<uint64_t> argmin_colwise();

		template<char axis = 'a'>
		inline std::conditional<axis == 'a', double, vector<double>> argmin(size_t* row = nullptr, size_t* col = nullptr);

		// Max

		inline vector<double> max_rowwise();

		inline vector<double> max_colwise();

		inline double max_all();

		template<char axis = 'a'>
		inline std::conditional<axis == 'a', double, vector<double>> max();

		inline void argmax_all(size_t*, size_t*);

		inline vector<uint64_t> argmax_rowwise();

		inline vector<uint64_t> argmax_colwise();

		template<char axis = 'a'>
		inline std::conditional<axis == 'a', double, vector<double>> argmax(size_t* row = nullptr, size_t* col = nullptr);

		// Activation functions

		// ReLU

		template<bool returnTransposed = false>
		inline matrix<double> relu();

		inline void self_relu();

		// LReLU

		template<bool returnTransposed = false>
		inline matrix<double> lrelu();

		inline void self_lrelu();

		// Sigmoid

		template<bool returnTransposed = false>
		inline matrix<double> sigmoid();

		inline void self_sigmoid();

		// Softplus

		template<bool returnTransposed = false>
		inline matrix<double> softplus();

		inline void self_softplus();

		// Tanh

		template<bool returnTransposed = false>
		inline matrix<double> tanh();

		inline void self_tanh();

		// Cast

		template <typename T>
		inline matrix<T> cast();

	private:
		double* _data;
		double* dataToDelete;
		size_t _rows, _cols, _size;
		size_t actualRows, actualCols;
		size_t finalPosRows, finalPosCols, finalPosSize;
		size_t _capacityRows;
		bool transposed;
	};
}
