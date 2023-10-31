#pragma once
#include <initializer.h>
#include <matrixDouble.h>
#include <matrixUint8_t.h>

namespace alge
{
	template <bool thisTransposed, bool thisContiguous>
	class matrix<float, thisTransposed, thisContiguous>
	{
	public:

		inline matrix();

		inline matrix(size_t rows, size_t cols);

		inline matrix(float* data, size_t rows, size_t cols, size_t actualRows, size_t actualCols);

		inline matrix(std::initializer_list<std::initializer_list<float>> list);

		inline ~matrix();

		// Friend classes

		template <typename T, bool tranposed, bool contiguous>
		friend class matrix;

		template <typename T>
		friend class vector;

		// Friend functions

		template<bool returnTransposed, bool matrix1Transposed, bool matrix1Contiguous,
			bool matrix2Transposed, bool matrix2Contiguous>
		friend inline matrix<float> dot(matrix<float, matrix1Transposed, matrix1Contiguous>&, matrix<float, matrix2Transposed, matrix2Contiguous>&);

		template<bool otherTransposed, bool otherContiguous>
		friend std::ostream& operator<<(std::ostream& os, const matrix<float, otherTransposed, otherContiguous>& matrix);

		template<bool returnTransposed, typename T, bool matrix1Transposed, bool matrix1Contiguous,
			bool matrix2Transposed, bool matrix2Contiguous>
		friend inline matrix<T> concatenate_rowwise(matrix<T, matrix1Transposed, matrix1Contiguous>&, matrix<T, matrix2Transposed, matrix2Contiguous>&);

		template<bool returnTransposed, typename T, bool matrix1Transposed, bool matrix1Contiguous,
			bool matrix2Transposed, bool matrix2Contiguous>
		friend inline matrix<T> concatenate_colwise(matrix<T, matrix1Transposed, matrix1Contiguous>&, matrix<T, matrix2Transposed, matrix2Contiguous>&);

		template<bool returnTransposed, bool thisTransposed, bool thisContiguous>
		friend inline matrix<float> operator+(float, matrix<float, thisTransposed, thisContiguous>&);

		template<bool returnTransposed, bool thisTransposed, bool thisContiguous>
		friend inline matrix<float> operator-(float, matrix<float, thisTransposed, thisContiguous>&);

		template<bool returnTransposed, bool thisTransposed, bool thisContiguous>
		friend inline matrix<float> operator*(float, matrix<float, thisTransposed, thisContiguous>&);

		template<bool returnTransposed, bool thisTransposed, bool thisContiguous>
		friend inline matrix<float> operator/(float, matrix<float, thisTransposed, thisContiguous>&);

		template<bool returnTransposed, bool thisTransposed, bool thisContiguous>
		friend inline matrix<uint8_t> operator==(float, matrix<float, thisTransposed, thisContiguous>&);

		template<bool returnTransposed, bool thisTransposed, bool thisContiguous>
		friend inline matrix<uint8_t> operator!=(float, matrix<float, thisTransposed, thisContiguous>&);

		template<bool returnTransposed, bool thisTransposed, bool thisContiguous>
		friend inline matrix<uint8_t> operator>(float, matrix<float, thisTransposed, thisContiguous>&);

		template<bool returnTransposed, bool thisTransposed, bool thisContiguous>
		friend inline matrix<uint8_t> operator>=(float, matrix<float, thisTransposed, thisContiguous>&);

		template<bool returnTransposed, bool thisTransposed, bool thisContiguous>
		friend inline matrix<uint8_t> operator<(float, matrix<float, thisTransposed, thisContiguous>&);

		template<bool returnTransposed, bool thisTransposed, bool thisContiguous>
		friend inline matrix<uint8_t> operator<=(float, matrix<float, thisTransposed, thisContiguous>&);

		template<bool returnTransposed, bool thisTransposed, bool thisContiguous>
		friend inline matrix<float> clip(matrix<float, thisTransposed, thisContiguous>&, float, float);

		template<bool returnTransposed, bool matrx1Transposed, bool matrix1Contiguous
			, bool matrx2Transposed, bool matrix2Contiguous, bool matrx3Transposed, bool matrix3Contiguous>
		friend inline matrix<float> where(matrix<uint8_t, matrx1Transposed, matrix1Contiguous>&, matrix<float, matrx2Transposed, matrix2Contiguous>&, matrix<float, matrx3Transposed, matrix3Contiguous>&);

		template<bool returnTransposed, bool matrx1Transposed, bool matrix1Contiguous
			, bool matrx2Transposed, bool matrix2Contiguous>
		friend inline matrix<float> where(matrix<uint8_t, matrx1Transposed, matrix1Contiguous>&, matrix<float, matrx2Transposed, matrix2Contiguous>&, float);

		template<bool returnTransposed, bool matrx1Transposed, bool matrix1Contiguous
			, bool matrx2Transposed, bool matrix2Contiguous>
		friend inline matrix<float> where(matrix<uint8_t, matrx1Transposed, matrix1Contiguous>&, float, matrix<float, matrx2Transposed, matrix2Contiguous>&);

		template<bool returnTransposed, bool matrx1Transposed, bool matrix1Contiguous>
		friend inline matrix<float> where(matrix<uint8_t, matrx1Transposed, matrix1Contiguous>&, float, float);

		template<bool useSteps, bool thisContiguous>
		friend inline matrix<float> randomGenerator(alge::matrix<float, false, thisContiguous>&, size_t);

		//----------------

		inline size_t rows();

		inline size_t cols();

		inline float* data();

		inline size_t capacity();

		template<bool reduceCapacity = true>
		inline void clear();

		inline void reserve(size_t newCapacity);

		inline void append(std::initializer_list<std::initializer_list<float>> list);

		template<bool otherTransposed, bool otherContiguous>
		inline void append(matrix<float, otherTransposed, otherContiguous>& other);

		inline void erase(size_t index);

		inline size_t find(vector<float>& other);

		template<bool otherTransposed, bool otherContiguous>
		inline vector<uint64_t> find(matrix<float, otherTransposed, otherContiguous>& other);

		inline matrix<float, thisTransposed, thisContiguous> row(size_t row);

		inline matrix<float, thisTransposed, thisContiguous> col(size_t col);

		inline matrix<float, !thisTransposed, thisContiguous> tranpose();

		template<bool blockContiguous = false>
		inline matrix<float, thisTransposed, thisContiguous&& blockContiguous> block(size_t initial_row, size_t initial_col, size_t final_row, size_t final_col);

		inline float& operator()(size_t row, size_t col);

		inline const float& operator()(size_t row, size_t col) const;

		// Indentity

		void identity();

		// Copy

		template<bool returnTransposed = false>
		inline matrix<float> copy();

		// =

		template<bool otherTransposed, bool otherContiguous>
		inline matrix<float, thisTransposed, thisContiguous>& operator=(matrix<float, otherTransposed, otherContiguous>& other);

		// Transfer

		template<bool otherContiguous>
		inline void transfer(matrix<float, thisTransposed, otherContiguous>& other);

		// neg

		template<bool returnTransposed = false>
		inline matrix<float> operator-();

		inline void self_not();

		// Set constant

		inline void set_const(float num);

		// Rand

		inline void rand();

		// +

		template<bool returnTransposed = false, bool otherTransposed, bool otherContiguous>
		inline matrix<float> operator+(const matrix<float, otherTransposed, otherContiguous>& other);

		template<bool otherTransposed, bool otherContiguous>
		inline void operator+=(const matrix<float, otherTransposed, otherContiguous>& other);

		template<bool returnTransposed = false>
		inline matrix<float> operator+(float num);

		inline void operator+=(float num);

		// -

		template<bool returnTransposed = false, bool otherTransposed, bool otherContiguous>
		inline matrix<float> operator-(const matrix<float, otherTransposed, otherContiguous>& other);

		template<bool otherTransposed, bool otherContiguous>
		inline void operator-=(const matrix<float, otherTransposed, otherContiguous>& other);

		template<bool returnTransposed = false>
		inline matrix<float> operator-(float num);

		inline void operator-=(float num);

		// *

		template<bool returnTransposed = false, bool otherTransposed, bool otherContiguous>
		inline matrix<float> operator*(const matrix<float, otherTransposed, otherContiguous>& other);

		template<bool otherTransposed, bool otherContiguous>
		inline void operator*=(const matrix<float, otherTransposed, otherContiguous>& other);

		template<bool returnTransposed = false>
		inline matrix<float> operator*(float num);

		inline void operator*=(float num);

		// /

		template<bool returnTransposed = false, bool otherTransposed, bool otherContiguous>
		inline matrix<float> operator/(const matrix<float, otherTransposed, otherContiguous>& other);

		template<bool otherTransposed, bool otherContiguous>
		inline void operator/=(const matrix<float, otherTransposed, otherContiguous>& other);

		template<bool returnTransposed = false>
		inline matrix<float> operator/(float num);

		inline void operator/=(float num);

		// ==

		template<bool returnTransposed = false, bool otherTransposed, bool otherContiguous>
		inline matrix<uint8_t> operator==(const matrix<float, otherTransposed, otherContiguous>& other);

		template<bool returnTransposed = false>
		inline matrix<uint8_t> operator==(float num);

		// !=

		template<bool returnTransposed = false, bool otherTransposed, bool otherContiguous>
		inline matrix<uint8_t> operator!=(const matrix<float, otherTransposed, otherContiguous>& other);

		template<bool returnTransposed = false>
		inline matrix<uint8_t> operator!=(float num);

		// >

		template<bool returnTransposed = false, bool otherTransposed, bool otherContiguous>
		inline matrix<uint8_t> operator>(const matrix<float, otherTransposed, otherContiguous>& other);

		template<bool returnTransposed = false>
		inline matrix<uint8_t> operator>(float num);

		// <

		template<bool returnTransposed = false, bool otherTransposed, bool otherContiguous>
		inline matrix<uint8_t> operator<(const matrix<float, otherTransposed, otherContiguous>& other);

		template<bool returnTransposed = false>
		inline matrix<uint8_t> operator<(float num);

		// >=

		template<bool returnTransposed = false, bool otherTransposed, bool otherContiguous>
		inline matrix<uint8_t> operator>=(const matrix<float, otherTransposed, otherContiguous>& other);

		template<bool returnTransposed = false>
		inline matrix<uint8_t> operator>=(float num);

		// <=

		template<bool returnTransposed = false, bool otherTransposed, bool otherContiguous>
		inline matrix<uint8_t> operator<=(const matrix<float, otherTransposed, otherContiguous>& other);

		template<bool returnTransposed = false>
		inline matrix<uint8_t> operator<=(float num);

		// Functions

		template<bool returnTransposed = false>
		inline matrix<float> exp();

		inline void self_exp();

		template<bool returnTransposed = false>
		inline matrix<float> exp2();

		inline void self_exp2();

		template<bool returnTransposed = false>
		inline matrix<float> log();

		inline void self_log();

		template<bool returnTransposed = false>
		inline matrix<float> log2();

		inline void self_log2();

		template<bool returnTransposed = false>
		inline matrix<float> log10();

		inline void self_log10();

#define _mm256_abs_ps(a) _mm256_andnot_ps(mask, (a))

		template<bool returnTransposed = false>
		inline matrix<float> abs();

		inline void self_abs();

		template<bool returnTransposed = false>
		inline matrix<float> cos();

		inline void self_cos();

		template<bool returnTransposed = false>
		inline matrix<float> tan();

		inline void self_tan();

		template<bool returnTransposed = false>
		inline matrix<float> acos();

		inline void self_acos();

		template<bool returnTransposed = false>
		inline matrix<float> round();

		inline void self_round();

		template<bool returnTransposed = false>
		inline matrix<float> floor();

		inline void self_floor();

		// pow

		template<bool returnTransposed = false>
		inline matrix<float> pow(float num);

		template<bool returnTransposed = false, bool otherTransposed, bool otherContiguous>
		inline matrix<float> pow(const matrix<float, otherTransposed, otherContiguous>& other);

		inline void self_pow(float num);

		template<bool otherTransposed, bool otherContiguous>
		inline void self_pow(const matrix<float, otherTransposed, otherContiguous>& other);

		// root

		template<bool returnTransposed = false>
		inline matrix<float> root(float num);

		template<bool returnTransposed = false, bool otherTransposed, bool otherContiguous>
		inline matrix<float> root(const matrix<float, otherTransposed, otherContiguous>& other);

		inline void self_root(float num);

		template<bool otherTransposed, bool otherContiguous>
		inline void self_root(const matrix<float, otherTransposed, otherContiguous>& other);

		// Mean 

		inline vector<float> mean_rowwise();

		inline vector<float> mean_colwise();

		inline float mean_all();

		// Sum

		inline vector<float> sum_rowwise();

		inline vector<float> sum_colwise();

		inline float sum_all();

		// Std

		inline vector<float> std_rowwise(float ddof = 0.0f);

		inline vector<float> std_colwise(float ddof = 0.0f);

		inline float std_all(float ddof = 0.0f, float* mean = nullptr);

		// Min

		inline vector<float> min_rowwise();

		inline vector<float> min_colwise();

		inline float min_all();

		inline void argmin_all(size_t* row, size_t* col);

		inline vector<uint64_t> argmin_rowwise();

		inline vector<uint64_t> argmin_colwise();

		// Max

		inline vector<float> max_rowwise();

		inline vector<float> max_colwise();

		inline float max_all();

		inline void argmax_all(size_t* row, size_t* col);

		inline vector<uint64_t> argmax_rowwise();

		inline vector<uint64_t> argmax_colwise();

		// Activation functions

		// ReLU

		template<bool returnTransposed = false>
		inline matrix<float> relu();

		inline void self_relu();

		// LReLU

		template<bool returnTransposed = false>
		inline matrix<float> lrelu();

		inline void self_lrelu();

		// Sigmoid

		template<bool returnTransposed = false>
		inline matrix<float> sigmoid();

		inline void self_sigmoid();

		// Softplus

		template<bool returnTransposed = false>
		inline matrix<float> softplus();

		inline void self_softplus();

		// Tanh

		template<bool returnTransposed = false>
		inline matrix<float> tanh();

		inline void self_tanh();

		// Cast
		template<typename T>
		inline matrix<T> cast();

	private:
		float* _data;
		float* dataToDelete;
		size_t _rows, _cols, _size;
		size_t actualRows, actualCols;
		size_t finalPosRows, finalPosCols, finalPosSize;
		size_t _capacityRows;
	};

}

