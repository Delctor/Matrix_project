#pragma once
#include <vectorDouble.h>
#include <vectorFloat.h>
#include <vectorUint8_t.h>
#include <vectorUint64_t.h>
#include <vectorInt.h>
#include <matrixDouble.h>
#include <matrixFloat.h>
#include <matrixUint8_t.h>

namespace alge
{
	// +

	inline vector<double> operator+(double num, vector<double>& vector1)
	{
		return vector1 + num;
	}

	inline vector<float> operator+(float num, vector<float>& vector1)
	{
		return vector1 + num;
	}

	inline vector<uint64_t> operator+(uint64_t num, vector<uint64_t>& vector1)
	{
		return vector1 + num;
	}

	inline vector<int> operator+(int num, vector<int>& vector1)
	{
		return vector1 + num;
	}

	template<bool returnTransposed = false, bool thisTransposed, bool thisContiguous>
	inline matrix<double> operator+(double num, matrix<double, thisTransposed, thisContiguous>& matrix1)
	{
		return matrix1.operator+<returnTransposed>(num);
	}

	template<bool returnTransposed = false, bool thisTransposed, bool thisContiguous>
	inline matrix<float> operator+(float num, matrix<float, thisTransposed, thisContiguous>& matrix1)
	{
		return matrix1.operator+<returnTransposed>(num);
	}

	// -

	inline vector<double> operator-(double num, vector<double>& vector1)
	{
		size_t size = vector1._size;

		size_t finalPos = vector1.finalPos;

		double* data1 = vector1._data;

		vector<double> result(size);

		double* dataResult = result._data;

		__m256d b = _mm256_set1_pd(num);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_load_pd(&data1[i]);

			_mm256_store_pd(&dataResult[i], _mm256_sub_pd(b, a));
		}

		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = num - data1[i];
		}
		return result;
	}

	inline vector<float> operator-(float num, vector<float>& vector1)
	{
		size_t size = vector1._size;

		size_t finalPos = vector1.finalPos;

		float* data1 = vector1._data;

		vector<float> result(size);

		float* dataResult = result._data;

		__m256 b = _mm256_set1_ps(num);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256 a = _mm256_load_ps(&data1[i]);

			_mm256_store_ps(&dataResult[i], _mm256_sub_ps(b, a));
		}

		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = num - data1[i];
		}
		return result;
	}

	inline vector<uint64_t> operator-(uint64_t num, vector<uint64_t>& vector1)
	{
		size_t size = vector1._size;

		size_t finalPos = vector1.finalPos;

		uint64_t* data1 = vector1._data;

		vector<uint64_t> result(size);

		uint64_t* dataResult = result._data;

		__m256i b = _mm256_set1_epi64x(num);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256i a = _mm256_loadu_epi64(&data1[i]);

			_mm256_storeu_epi64(&dataResult[i], _mm256_sub_epi64(b, a));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = num - data1[i];
		}
		return result;
	}

	inline vector<int> operator-(int num, vector<int>& vector1)
	{
		size_t size = vector1._size;

		size_t finalPos = vector1.finalPos;

		int* data1 = vector1._data;

		vector<int> result(size);

		int* dataResult = result._data;

		__m256i b = _mm256_set1_epi32(num);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256i a = _mm256_loadu_epi32(&data1[i]);

			_mm256_storeu_epi32(&dataResult[i], _mm256_div_epi32(b, a));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = num / data1[i];
		}
		return result;
	}

	template<bool returnTransposed = false, bool thisTransposed, bool thisContiguous>
	inline matrix<double> operator-(double num, matrix<double, thisTransposed, thisContiguous>& matrix1)
	{
		size_t rows = matrix1._rows;
		size_t cols = matrix1._cols;

		double* data1 = matrix1._data;

		__m256d b = _mm256_set1_pd(num);

		size_t size = matrix1._size;

		size_t finalPosSize = matrix1.finalPosSize;

		size_t finalPosRows = matrix1.finalPosRows;
		size_t finalPosCols = matrix1.finalPosCols;

		size_t matrix1ActualRows = matrix1.actualRows;
		size_t matrix1ActualCols = matrix1.actualCols;

		if constexpr (returnTransposed)
		{
			matrix<double> result(cols, rows);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&dataResult[i], _mm256_sub_pd(b, a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = num - data1[i];
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

							_mm256_store_pd(&dataResult[j * rows + i], _mm256_sub_pd(b, a));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = num - data1[j * matrix1ActualRows + i];
						}
					}
				}
			}
			else
			{
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						__m256d sub = _mm256_sub_pd(b, a);

						__m128d val1 = _mm256_extractf128_pd(sub, 1);
						__m128d val2 = _mm256_castpd256_pd128(sub);

						_mm_store_sd(&dataResult[j * rows + i], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

						_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						dataResult[j * rows + i] = num - data1[i * matrix1ActualCols + j];
					}
				}
			}
			return result;
		}
		else
		{
			matrix<double> result(rows, cols);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				for (size_t j = 0; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						__m256d sub = _mm256_sub_pd(b, a);

						__m128d val1 = _mm256_extractf128_pd(sub, 1);
						__m128d val2 = _mm256_castpd256_pd128(sub);

						_mm_store_sd(&dataResult[i * cols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

						_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						dataResult[i * cols + j] = num - data1[j * matrix1ActualRows + i];
					}
				}
			}
			else
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&dataResult[i], _mm256_sub_pd(b, a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = num - data1[i];
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

							_mm256_store_pd(&dataResult[i * cols + j], _mm256_sub_pd(b, a));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = num - data1[i * matrix1ActualCols + j];
						}
					}
				}
			}
			return result;
		}
	}

	template <bool returnTransposed, bool thisTransposed, bool thisContiguous>
	inline matrix<float> operator-(float num, matrix<double, thisTransposed, thisContiguous>& matrix1)
	{
		size_t rows = matrix1._rows;
		size_t cols = matrix1._cols;

		size_t size = matrix1._size;

		float* data1 = matrix1._data;

		__m256 b = _mm256_set1_ps(num);

		size_t finalPosRows = matrix1.finalPosRows;
		size_t finalPosCols = matrix1.finalPosCols;
		size_t finalPosSize = matrix1.finalPosSize;

		size_t matrix1ActualRows = matrix1.actualRows;
		size_t matrix1ActualCols = matrix1.actualCols;

		if constexpr (returnTransposed)
		{
			matrix<float> result(cols, rows);

			float* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 8)
					{
						__m256 a = _mm256_load_ps(&data1[i]);

						_mm256_store_ps(&dataResult[i], _mm256_sub_ps(b, a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = num - data1[i];
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 8)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256 a = _mm256_load_ps(&data1[j * matrix1ActualRows + i]);

							_mm256_store_ps(&dataResult[j * rows + i], _mm256_sub_ps(b, a));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = num - data1[j * matrix1ActualRows + i];
						}
					}
				}
			}
			else
			{
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 8)
					{
						__m256 a = _mm256_load_ps(&data1[i * matrix1ActualCols + j]);

						__m256 sub = _mm256_sub_ps(b, a);

						__m128 high = _mm256_extractf128_ps(sub, 1);
						__m128 low = _mm256_castps256_ps128(sub);

						// 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0

						// 1.0, 2.0, 3.0, 4.0
						_mm_store_ss(&dataResult[j * rows + i], low);

						low = _mm_shuffle_ps(low, low, 0b11100001);
						// 2.0, 1.0, 3.0, 4.0
						_mm_store_ss(&dataResult[(j + 1) * rows + i], low);

						low = _mm_shuffle_ps(low, low, 0b11000110);
						// 3.0, 1.0, 2.0, 4.0
						_mm_store_ss(&dataResult[(j + 2) * rows + i], low);

						low = _mm_shuffle_ps(low, low, 0b00100111);
						// 4.0, 1.0, 2.0, 3.0
						_mm_store_ss(&dataResult[(j + 3) * rows + i], low);

						// --

						// 5.0, 6.0, 7.0, 8.0
						_mm_store_ss(&dataResult[(j + 4) * rows + i], high);

						high = _mm_shuffle_ps(high, high, 0b11100001);
						// 6.0, 5.0, 7.0, 8.0
						_mm_store_ss(&dataResult[(j + 5) * rows + i], high);

						high = _mm_shuffle_ps(high, high, 0b11000110);
						// 7.0, 5.0, 6.0, 8.0
						_mm_store_ss(&dataResult[(j + 6) * rows + i], high);

						high = _mm_shuffle_ps(high, high, 0b00100111);
						// 8.0, 5.0, 6.0, 7.0
						_mm_store_ss(&dataResult[(j + 7) * rows + i], high);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						dataResult[j * rows + i] = num - data1[i * matrix1ActualCols + j];
					}
				}
			}
			return result;
		}
		else
		{
			matrix<float> result(rows, cols);

			float* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				for (size_t j = 0; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 8)
					{
						__m256 a = _mm256_load_ps(&data1[j * matrix1ActualRows + i]);

						__m256 sub = _mm256_sub_ps(b, a);

						__m128 high = _mm256_extractf128_ps(sub, 1);
						__m128 low = _mm256_castps256_ps128(sub);

						// 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0

						// 1.0, 2.0, 3.0, 4.0
						_mm_store_ss(&dataResult[i * cols + j], low);

						low = _mm_shuffle_ps(low, low, 0b11100001);
						// 2.0, 1.0, 3.0, 4.0
						_mm_store_ss(&dataResult[(i + 1) * cols + j], low);

						low = _mm_shuffle_ps(low, low, 0b11000110);
						// 3.0, 1.0, 2.0, 4.0
						_mm_store_ss(&dataResult[(i + 2) * cols + j], low);

						low = _mm_shuffle_ps(low, low, 0b00100111);
						// 4.0, 1.0, 2.0, 3.0
						_mm_store_ss(&dataResult[(i + 3) * cols + j], low);

						// --

						// 5.0, 6.0, 7.0, 8.0
						_mm_store_ss(&dataResult[(i + 4) * cols + j], high);

						high = _mm_shuffle_ps(high, high, 0b11100001);
						// 6.0, 5.0, 7.0, 8.0
						_mm_store_ss(&dataResult[(i + 5) * cols + j], high);

						high = _mm_shuffle_ps(high, high, 0b11000110);
						// 7.0, 5.0, 6.0, 8.0
						_mm_store_ss(&dataResult[(i + 6) * cols + j], high);

						high = _mm_shuffle_ps(high, high, 0b00100111);
						// 8.0, 5.0, 6.0, 7.0
						_mm_store_ss(&dataResult[(i + 7) * cols + j], high);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						dataResult[i * cols + j] = num - data1[j * matrix1ActualRows + i];
					}
				}
			}
			else
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 8)
					{
						__m256 a = _mm256_load_ps(&data1[i]);

						_mm256_store_ps(&dataResult[i], _mm256_sub_ps(b, a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = num - data1[i];
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 8)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256 a = _mm256_load_ps(&data1[i * matrix1ActualCols + j]);

							_mm256_store_ps(&dataResult[i * cols + j], _mm256_sub_ps(b, a));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = num - data1[i * matrix1ActualCols + j];
						}
					}
				}
			}
			return result;
		}
	}

	// *

	inline vector<double> operator*(double num, vector<double>& vector1)
	{
		return vector1 * num;
	}

	inline vector<float> operator*(float num, vector<float>& vector1)
	{
		return vector1 * num;
	}

	inline vector<uint64_t> operator*(uint64_t num, vector<uint64_t>& vector1)
	{
		return vector1 * num;
	}

	inline vector<int> operator*(int num, vector<int>& vector1)
	{
		return vector1 * num;
	}

	template<bool returnTransposed = false, bool thisTransposed, bool thisContiguous>
	inline matrix<double> operator*(double num, matrix<double, thisTransposed, thisContiguous>& matrix1)
	{
		return matrix1 * num;
	}

	template<bool returnTransposed = false, bool thisTransposed, bool thisContiguous>
	inline matrix<float> operator*(float num, matrix<float, thisTransposed, thisContiguous>& matrix1)
	{
		return matrix1 * num;
	}

	// /

	inline vector<double> operator/(double num, vector<double>& vector1)
	{
		size_t size = vector1._size;

		size_t finalPos = vector1.finalPos;

		double* data1 = vector1._data;

		vector<double> result(size);

		double* dataResult = result._data;

		__m256d b = _mm256_set1_pd(num);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_load_pd(&data1[i]);

			_mm256_store_pd(&dataResult[i], _mm256_div_pd(b, a));
		}

		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = num / data1[i];
		}
		return result;
	}

	inline vector<float> operator/(float num, vector<float>& vector1)
	{
		size_t size = vector1._size;

		size_t finalPos = vector1.finalPos;

		float* data1 = vector1._data;

		vector<float> result(size);

		float* dataResult = result._data;

		__m256 b = _mm256_set1_ps(num);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256 a = _mm256_load_ps(&data1[i]);

			_mm256_store_ps(&dataResult[i], _mm256_div_ps(b, a));
		}

		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = num / data1[i];
		}
		return result;
	}

	inline vector<uint64_t> operator/(uint64_t num, vector<uint64_t>& vector1)
	{
		size_t size = vector1._size;

		size_t finalPos = vector1.finalPos;

		uint64_t* data1 = vector1._data;

		vector<uint64_t> result(size);

		uint64_t* dataResult = result._data;

		__m256i b = _mm256_set1_epi64x(num);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256i a = _mm256_loadu_epi64(&data1[i]);

			_mm256_storeu_epi64(&dataResult[i], _mm256_div_epi64(b, a));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = num / data1[i];
		}
		return result;
	}

	inline vector<int> operator/(int num, vector<int>& vector1)
	{
		size_t size = vector1._size;

		size_t finalPos = vector1.finalPos;

		int* data1 = vector1._data;

		vector<int> result(size);

		int* dataResult = result._data;

		__m256i b = _mm256_set1_epi32(num);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256i a = _mm256_loadu_epi32(&data1[i]);

			_mm256_storeu_epi32(&dataResult[i], _mm256_div_epi32(b, a));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = num / data1[i];
		}
		return result;
	}

	template<bool returnTransposed = false, bool thisTransposed, bool thisContiguous>
	inline matrix<double> operator/(double num, matrix<double, thisTransposed, thisContiguous>& matrix1)
	{
		size_t rows = matrix1._rows;
		size_t cols = matrix1._cols;

		size_t size = matrix1._size;

		double* data1 = matrix1._data;

		__m256d b = _mm256_set1_pd(num);

		size_t finalPosRows = matrix1.finalPosRows;
		size_t finalPosCols = matrix1.finalPosCols;
		size_t finalPosSize = matrix1.finalPosSize;

		size_t matrix1ActualRows = matrix1.actualRows;
		size_t matrix1ActualCols = matrix1.actualCols;

		if constexpr (returnTransposed)
		{
			matrix<double> result(cols, rows);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 8)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&dataResult[i], _mm256_div_pd(b, a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = num / data1[i];
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 8)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

							_mm256_store_pd(&dataResult[j * rows + i], _mm256_div_pd(b, a));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = num / data1[j * matrix1ActualRows + i];
						}
					}
				}
			}
			else
			{
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 8)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						__m256d sub = _mm256_div_pd(b, a);

						__m128d val1 = _mm256_extractf128_pd(sub, 1);
						__m128d val2 = _mm256_castpd256_pd128(sub);

						_mm_store_sd(&dataResult[j * rows + i], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

						_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						dataResult[j * rows + i] = num / data1[i * matrix1ActualCols + j];
					}
				}
			}
			return result;
		}
		else
		{
			matrix<double> result(rows, cols);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				for (size_t j = 0; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 8)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						__m256d sub = _mm256_div_pd(b, a);

						__m128d val1 = _mm256_extractf128_pd(sub, 1);
						__m128d val2 = _mm256_castpd256_pd128(sub);

						_mm_store_sd(&dataResult[i * cols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

						_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						dataResult[i * cols + j] = num / data1[j * matrix1ActualRows + i];
					}
				}
			}
			else
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 8)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&dataResult[i], _mm256_div_pd(b, a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = num / data1[i];
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 8)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

							_mm256_store_pd(&dataResult[i * cols + j], _mm256_div_pd(b, a));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = num / data1[i * matrix1ActualCols + j];
						}
					}
				}
			}
			return result;
		}
	}

	template <bool returnTransposed, bool thisTransposed, bool thisContiguous>
	inline matrix<float> operator/(float num, matrix<double, thisTransposed, thisContiguous>& matrix1)
	{
		size_t rows = matrix1._rows;
		size_t cols = matrix1._cols;

		size_t size = matrix1._size;

		float* data1 = matrix1._data;

		__m256 b = _mm256_set1_ps(num);

		size_t finalPosRows = matrix1.finalPosRows;
		size_t finalPosCols = matrix1.finalPosCols;
		size_t finalPosSize = matrix1.finalPosSize;

		size_t matrix1ActualRows = matrix1.actualRows;
		size_t matrix1ActualCols = matrix1.actualCols;

		if constexpr (returnTransposed)
		{
			matrix<float> result(cols, rows);

			float* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 8)
					{
						__m256 a = _mm256_load_ps(&data1[i]);

						_mm256_store_ps(&dataResult[i], _mm256_div_ps(b, a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = num / data1[i];
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 8)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256 a = _mm256_load_ps(&data1[j * matrix1ActualRows + i]);

							_mm256_store_ps(&dataResult[j * rows + i], _mm256_div_ps(b, a));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = num / data1[j * matrix1ActualRows + i];
						}
					}
				}
			}
			else
			{
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 8)
					{
						__m256 a = _mm256_load_ps(&data1[i * matrix1ActualCols + j]);

						__m256 div = _mm256_div_ps(b, a);

						__m128 high = _mm256_extractf128_ps(div, 1);
						__m128 low = _mm256_castps256_ps128(div);

						// 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0

						// 1.0, 2.0, 3.0, 4.0
						_mm_store_ss(&dataResult[j * rows + i], low);

						low = _mm_shuffle_ps(low, low, 0b11100001);
						// 2.0, 1.0, 3.0, 4.0
						_mm_store_ss(&dataResult[(j + 1) * rows + i], low);

						low = _mm_shuffle_ps(low, low, 0b11000110);
						// 3.0, 1.0, 2.0, 4.0
						_mm_store_ss(&dataResult[(j + 2) * rows + i], low);

						low = _mm_shuffle_ps(low, low, 0b00100111);
						// 4.0, 1.0, 2.0, 3.0
						_mm_store_ss(&dataResult[(j + 3) * rows + i], low);

						// --

						// 5.0, 6.0, 7.0, 8.0
						_mm_store_ss(&dataResult[(j + 4) * rows + i], high);

						high = _mm_shuffle_ps(high, high, 0b11100001);
						// 6.0, 5.0, 7.0, 8.0
						_mm_store_ss(&dataResult[(j + 5) * rows + i], high);

						high = _mm_shuffle_ps(high, high, 0b11000110);
						// 7.0, 5.0, 6.0, 8.0
						_mm_store_ss(&dataResult[(j + 6) * rows + i], high);

						high = _mm_shuffle_ps(high, high, 0b00100111);
						// 8.0, 5.0, 6.0, 7.0
						_mm_store_ss(&dataResult[(j + 7) * rows + i], high);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						dataResult[j * rows + i] = num / data1[i * matrix1ActualCols + j];
					}
				}
			}
			return result;
		}
		else
		{
			matrix<float> result(rows, cols);

			float* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				for (size_t j = 0; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 8)
					{
						__m256 a = _mm256_load_ps(&data1[j * matrix1ActualRows + i]);

						__m256 div = _mm256_div_ps(b, a);

						__m128 high = _mm256_extractf128_ps(div, 1);
						__m128 low = _mm256_castps256_ps128(div);

						// 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0

						// 1.0, 2.0, 3.0, 4.0
						_mm_store_ss(&dataResult[i * cols + j], low);

						low = _mm_shuffle_ps(low, low, 0b11100001);
						// 2.0, 1.0, 3.0, 4.0
						_mm_store_ss(&dataResult[(i + 1) * cols + j], low);

						low = _mm_shuffle_ps(low, low, 0b11000110);
						// 3.0, 1.0, 2.0, 4.0
						_mm_store_ss(&dataResult[(i + 2) * cols + j], low);

						low = _mm_shuffle_ps(low, low, 0b00100111);
						// 4.0, 1.0, 2.0, 3.0
						_mm_store_ss(&dataResult[(i + 3) * cols + j], low);

						// --

						// 5.0, 6.0, 7.0, 8.0
						_mm_store_ss(&dataResult[(i + 4) * cols + j], high);

						high = _mm_shuffle_ps(high, high, 0b11100001);
						// 6.0, 5.0, 7.0, 8.0
						_mm_store_ss(&dataResult[(i + 5) * cols + j], high);

						high = _mm_shuffle_ps(high, high, 0b11000110);
						// 7.0, 5.0, 6.0, 8.0
						_mm_store_ss(&dataResult[(i + 6) * cols + j], high);

						high = _mm_shuffle_ps(high, high, 0b00100111);
						// 8.0, 5.0, 6.0, 7.0
						_mm_store_ss(&dataResult[(i + 7) * cols + j], high);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						dataResult[i * cols + j] = num / data1[j * matrix1ActualRows + i];
					}
				}
			}
			else
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 8)
					{
						__m256 a = _mm256_load_ps(&data1[i]);

						_mm256_store_ps(&dataResult[i], _mm256_div_ps(b, a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = num / data1[i];
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 8)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256 a = _mm256_load_ps(&data1[i * matrix1ActualCols + j]);

							_mm256_store_ps(&dataResult[i * cols + j], _mm256_div_ps(b, a));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = num / data1[i * matrix1ActualCols + j];
						}
					}
				}
			}
			return result;
		}
	}

	// ==

	inline vector<uint8_t> operator==(double num, vector<double>& vector1)
	{
		return vector1 == num;
	}

	inline vector<uint8_t> operator==(float num, vector<float>& vector1)
	{
		return vector1 == num;
	}

	inline vector<uint8_t> operator==(uint64_t num, vector<uint64_t>& vector1)
	{
		return vector1 == num;
	}

	inline vector<uint8_t> operator==(int num, vector<int>& vector1)
	{
		return vector1 == num;
	}

	template<bool returnTransposed = false, bool thisTransposed, bool thisContiguous>
	inline matrix<uint8_t> operator==(double num, matrix<double, thisTransposed, thisContiguous>& matrix1)
	{
		return matrix1 == num;
	}

	template<bool returnTransposed = false, bool thisTransposed, bool thisContiguous>
	inline matrix<uint8_t> operator==(float num, matrix<float, thisTransposed, thisContiguous>& matrix1)
	{
		return matrix1 == num;
	}

	// != 

	inline vector<uint8_t> operator!=(double num, vector<double>& vector1)
	{
		return vector1 != num;
	}

	inline vector<uint8_t> operator!=(float num, vector<float>& vector1)
	{
		return vector1 != num;
	}

	inline vector<uint8_t> operator!=(uint64_t num, vector<uint64_t>& vector1)
	{
		return vector1 != num;
	}

	inline vector<uint8_t> operator!=(int num, vector<int>& vector1)
	{
		return vector1 != num;
	}

	template<bool returnTransposed = false, bool thisTransposed, bool thisContiguous>
	inline matrix<uint8_t> operator!=(double num, matrix<double, thisTransposed, thisContiguous>& matrix1)
	{
		return matrix1 != num;
	}

	template<bool returnTransposed = false, bool thisTransposed, bool thisContiguous>
	inline matrix<uint8_t> operator!=(float num, matrix<float, thisTransposed, thisContiguous>& matrix1)
	{
		return matrix1 != num;
	}

	// >

	inline vector<uint8_t> operator>(double num, vector<double>& vector1)
	{
		return vector1 < num;
	}

	inline vector<uint8_t> operator>(float num, vector<float>& vector1)
	{
		return vector1 < num;
	}

	inline vector<uint8_t> operator>(uint64_t num, vector<uint64_t>& vector1)
	{
		return vector1 < num;
	}

	inline vector<uint8_t> operator>(int num, vector<int>& vector1)
	{
		return vector1 < num;
	}

	template<bool returnTransposed = false, bool thisTransposed, bool thisContiguous>
	inline matrix<uint8_t> operator>(double num, matrix<double, thisTransposed, thisContiguous>& matrix1)
	{
		return matrix1 < num;
	}

	template<bool returnTransposed = false, bool thisTransposed, bool thisContiguous>
	inline matrix<uint8_t> operator>(float num, matrix<float, thisTransposed, thisContiguous>& matrix1)
	{
		return matrix1 < num;
	}

	// >=

	inline vector<uint8_t> operator>=(double num, vector<double>& vector1)
	{
		return vector1 <= num;
	}

	inline vector<uint8_t> operator>=(float num, vector<float>& vector1)
	{
		return vector1 <= num;
	}

	inline vector<uint8_t> operator>=(uint64_t num, vector<uint64_t>& vector1)
	{
		return vector1 <= num;
	}

	inline vector<uint8_t> operator>=(int num, vector<int>& vector1)
	{
		return vector1 <= num;
	}

	template<bool returnTransposed = false, bool thisTransposed, bool thisContiguous>
	inline matrix<uint8_t> operator>=(double num, matrix<double, thisTransposed, thisContiguous>& matrix1)
	{
		return matrix1 <= num;
	}

	template<bool returnTransposed = false, bool thisTransposed, bool thisContiguous>
	inline matrix<uint8_t> operator>=(float num, matrix<float, thisTransposed, thisContiguous>& matrix1)
	{
		return matrix1 <= num;
	}

	// <

	inline vector<uint8_t> operator<(double num, vector<double>& vector1)
	{
		return vector1 > num;
	}

	inline vector<uint8_t> operator<(float num, vector<float>& vector1)
	{
		return vector1 > num;
	}

	inline vector<uint8_t> operator<(uint64_t num, vector<uint64_t>& vector1)
	{
		return vector1 > num;
	}

	inline vector<uint8_t> operator<(int num, vector<int>& vector1)
	{
		return vector1 > num;
	}

	template<bool returnTransposed = false, bool thisTransposed, bool thisContiguous>
	inline matrix<uint8_t> operator<(double num, matrix<double, thisTransposed, thisContiguous>& matrix1)
	{
		return matrix1 > num;
	}

	template<bool returnTransposed = false, bool thisTransposed, bool thisContiguous>
	inline matrix<uint8_t> operator<(float num, matrix<float, thisTransposed, thisContiguous>& matrix1)
	{
		return matrix1 > num;
	}

	// <= 

	inline vector<uint8_t> operator<=(double num, vector<double>& vector1)
	{
		return vector1 >= num;
	}

	inline vector<uint8_t> operator<=(float num, vector<float>& vector1)
	{
		return vector1 >= num;
	}

	inline vector<uint8_t> operator<=(uint64_t num, vector<uint64_t>& vector1)
	{
		return vector1 >= num;
	}

	inline vector<uint8_t> operator<=(int num, vector<int>& vector1)
	{
		return vector1 >= num;
	}

	template<bool returnTransposed = false, bool thisTransposed, bool thisContiguous>
	inline matrix<uint8_t> operator<=(double num, matrix<double, thisTransposed, thisContiguous>& matrix1)
	{
		return matrix1 >= num;
	}

	template<bool returnTransposed = false, bool thisTransposed, bool thisContiguous>
	inline matrix<uint8_t> operator<=(float num, matrix<float, thisTransposed, thisContiguous>& matrix1)
	{
		return matrix1 >= num;
	}

}
