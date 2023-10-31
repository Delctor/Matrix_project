#pragma once
#include <initializer.h>
#include <matrixDouble.h>
#include <matrixFloat.h>

namespace alge
{
	template <bool thisTransposed, bool thisContiguous>
	class matrix<uint8_t, thisTransposed, thisContiguous>
	{
	public:
		inline matrix();

		inline matrix(size_t, size_t);

		inline matrix(uint8_t*, size_t, size_t, size_t, size_t);

		inline matrix(std::initializer_list<std::initializer_list<double>>);

		inline ~matrix();

		// Friend classes

		template <typename T, bool tranposed, bool contiguous>
		friend class matrix;

		template <typename T>
		friend class vector;

		// Friend functions

		template<bool otherTransposed, bool otherContiguous>
		friend std::ostream& operator<<(std::ostream&, const matrix<uint8_t, otherTransposed, otherContiguous>&);

		template<bool returnTransposed, typename T, bool matrix1Transposed, bool matrix1Contiguous,
			bool matrix2Transposed, bool matrix2Contiguous>
		friend inline matrix<T> concatenate_rowwise(matrix<T, matrix1Transposed, matrix1Contiguous>&, matrix<T, matrix2Transposed, matrix2Contiguous>&);

		template<bool returnTransposed, typename T, bool matrix1Transposed, bool matrix1Contiguous,
			bool matrix2Transposed, bool matrix2Contiguous>
		friend inline matrix<T> concatenate_colwise(matrix<T, matrix1Transposed, matrix1Contiguous>&, matrix<T, matrix2Transposed, matrix2Contiguous>&);

		template<bool returnTransposed, bool matrx1Transposed, bool matrix1Contiguous
			, bool matrx2Transposed, bool matrix2Contiguous, bool matrx3Transposed, bool matrix3Contiguous>
		friend inline matrix<double> where(matrix<uint8_t, matrx1Transposed, matrix1Contiguous>&, matrix<double, matrx2Transposed, matrix2Contiguous>&, matrix<double, matrx3Transposed, matrix3Contiguous>&);

		template<bool returnTransposed, bool matrx1Transposed, bool matrix1Contiguous
			, bool matrx2Transposed, bool matrix2Contiguous>
		friend inline matrix<double> where(matrix<uint8_t, matrx1Transposed, matrix1Contiguous>&, matrix<double, matrx2Transposed, matrix2Contiguous>&, double);

		template<bool returnTransposed, bool matrx1Transposed, bool matrix1Contiguous
			, bool matrx2Transposed, bool matrix2Contiguous>
		friend inline matrix<double> where(matrix<uint8_t, matrx1Transposed, matrix1Contiguous>&, double, matrix<double, matrx2Transposed, matrix2Contiguous>&);

		template<bool returnTransposed, bool matrx1Transposed, bool matrix1Contiguous>
		friend inline matrix<double> where(matrix<uint8_t, matrx1Transposed, matrix1Contiguous>&, double, double);

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

		inline size_t rows();

		inline size_t cols();

		inline uint8_t* data();

		inline matrix<uint8_t, thisTransposed, thisContiguous> row(size_t);

		inline matrix<uint8_t, thisTransposed, thisContiguous> col(size_t);

		inline matrix<uint8_t, !thisTransposed, thisContiguous> tranpose();

		template<bool blockContiguous = false>
		inline matrix<uint8_t, thisTransposed, thisContiguous&& blockContiguous> block(size_t, size_t, size_t, size_t);

		inline uint8_t& operator()(size_t, size_t);

		inline const uint8_t& operator()(size_t, size_t) const;

		inline size_t capacity();

		template<bool reduceCapacity = true>
		inline void clear();

		inline void reserve(size_t);

		inline void append(std::initializer_list<std::initializer_list<uint8_t>>);

		template<bool otherTransposed, bool otherContiguous>
		inline void append(matrix<uint8_t, otherTransposed, otherContiguous>&);

		inline void erase(size_t);

		inline size_t find(vector<uint8_t>&);

		template<bool otherTransposed, bool otherContiguous>
		inline vector<uint64_t> find(matrix<uint8_t, otherTransposed, otherContiguous>&);

		// Copy

		template<bool returnTransposed = false>
		inline matrix<uint8_t> copy();

		// = 

		template<bool otherTransposed, bool otherContiguous>
		inline matrix<uint8_t, thisTransposed, thisContiguous>& operator=(matrix<uint8_t, otherTransposed, otherContiguous>&);

		// Transfer

		template<bool otherContiguous>
		inline void transfer(matrix<uint8_t, thisTransposed, otherContiguous>&);

		// Set constant

		inline void set_const(uint8_t);

		template<bool returnTransposed = false, bool otherTransposed, bool otherContiguous>
		inline matrix<uint8_t> operator&&(matrix<uint8_t, otherTransposed, otherContiguous>&);

		template<bool returnTransposed = false, bool otherTransposed, bool otherContiguous>
		inline matrix<uint8_t> operator||(matrix<uint8_t, otherTransposed, otherContiguous>&);

		template<bool returnTransposed = false>
		inline matrix<uint8_t> operator!();

		inline void self_not();

		inline size_t count_all();

		inline vector<uint64_t> count_colwise();

		inline vector<uint64_t> count_rowwise();

		template<typename T>
		inline matrix<T> cast();

	private:

		uint8_t* _data;
		uint8_t* dataToDelete;
		size_t _rows, _cols, _size;
		size_t actualRows, actualCols;
		size_t finalPosSize, finalPosRows, finalPosCols;
		size_t finalPosSize256, finalPosRows256, finalPosCols256;
	};
}