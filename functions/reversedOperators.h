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

		matrix<double> result(rows, cols);

		double* dataResult = result._data;

		__m256d b = _mm256_set1_pd(num);

		if constexpr (thisTransposed)
		{
			if constexpr (returnTransposed)
			{
				if constexpr (thisContiguous)
				{
					size_t size = matrix1._size;

					size_t finalPosSize = matrix1.finalPosSize;

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
					size_t finalPosRows = matrix1.finalPosRows;
					size_t finalPosCols = matrix1.finalPosCols;

					size_t matrix1ActualRows = matrix1.actualRows;

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
						for (size_t j = 0; j < finalPosCols; j += 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
								data1[(j + 1) * matrix1ActualRows + i],
								data1[(j + 2) * matrix1ActualRows + i],
								data1[(j + 3) * matrix1ActualRows + i]);

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
							dataResult[j * rows + i] = num - data1[j * matrix1ActualRows + i];
						}
					}
				}
			}
			else
			{
				size_t finalPosCols = matrix1.finalPosCols;
				size_t finalPosRows = matrix1.finalPosRows;

				size_t matrix1ActualRows = matrix1.actualRows;

				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
							data1[(j + 1) * matrix1ActualRows + i],
							data1[(j + 2) * matrix1ActualRows + i],
							data1[(j + 3) * matrix1ActualRows + i]);

						_mm256_store_pd(&dataResult[i * cols + j], _mm256_sub_pd(b, a));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
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
		}
		else
		{
			if constexpr (returnTransposed)
			{
				size_t matrix1ActualCols = matrix1.actualCols;

				size_t finalPosRows = matrix1.finalPosRows;
				size_t finalPosCols = matrix1.finalPosCols;

				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
							data1[(i + 1) * matrix1ActualCols + j],
							data1[(i + 2) * matrix1ActualCols + j],
							data1[(i + 3) * matrix1ActualCols + j]);

						_mm256_store_pd(&dataResult[j * rows + i], _mm256_sub_pd(b, a));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
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
			else
			{
				if constexpr (thisContiguous)
				{
					size_t finalPosSize = matrix1.finalPosSize;
					size_t size = matrix1._size;

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
					size_t matrix1ActualCols = matrix1.actualCols;

					size_t finalPosCols = matrix1.finalPosCols;
					size_t finalPosRows = matrix1.finalPosRows;

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
						for (size_t i = 0; i < finalPosRows; i += 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
								data1[(i + 1) * matrix1ActualCols + j],
								data1[(i + 2) * matrix1ActualCols + j],
								data1[(i + 3) * matrix1ActualCols + j]);

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
							dataResult[i * cols + j] = num - data1[i * matrix1ActualCols + j];
						}
					}
				}
			}
		}
		return result;
	}

	template<bool returnTransposed = false, bool thisTransposed, bool thisContiguous>
	inline matrix<float> operator-(float num, matrix<float, thisTransposed, thisContiguous>& matrix1)
	{
		size_t rows = matrix1._rows;
		size_t cols = matrix1._cols;

		float* data1 = matrix1._data;

		matrix<float> result(rows, cols);

		float* dataResult = result._data;

		__m256 b = _mm256_set1_ps(num);

		if constexpr (thisTransposed)
		{
			if constexpr (returnTransposed)
			{
				if constexpr (thisContiguous)
				{
					size_t size = matrix1._size;

					size_t finalPosSize = matrix1.finalPosSize;

					for (size_t i = 0; i < finalPosSize; i += 8)
					{
						__m256 a = _mm256_load_ps(&data1[i]);

						_mm256_store_ps(&dataResult[i], _mm256_sub_ps(b, a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = data1[i] - num;
					}
				}
				else
				{
					size_t finalPosRows = matrix1.finalPosRows;
					size_t finalPosCols = matrix1.finalPosCols;

					size_t matrix1ActualRows = matrix1.actualRows;

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
							dataResult[j * rows + i] = data1[j * matrix1ActualRows + i] - num;
						}
					}
				}
			}
			else
			{
				size_t finalPosCols = matrix1.finalPosCols;
				size_t finalPosRows = matrix1.finalPosRows;

				size_t matrix1ActualRows = matrix1.actualRows;

				for (size_t j = 0; j < finalPosCols; j += 8)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256 a = _mm256_setr_ps(data1[j * matrix1ActualRows + i],
							data1[(j + 1) * matrix1ActualRows + i],
							data1[(j + 2) * matrix1ActualRows + i],
							data1[(j + 3) * matrix1ActualRows + i],
							data1[(j + 4) * matrix1ActualRows + i],
							data1[(j + 5) * matrix1ActualRows + i],
							data1[(j + 6) * matrix1ActualRows + i],
							data1[(j + 7) * matrix1ActualRows + i]);

						_mm256_store_ps(&dataResult[i * cols + j], _mm256_sub_ps(b, a));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < rows; i++)
					{
						dataResult[i * cols + j] = data1[j * matrix1ActualRows + i] - num;
					}
				}
			}
		}
		else
		{
			if constexpr (returnTransposed)
			{
				size_t matrix1ActualCols = matrix1.actualCols;

				size_t finalPosRows = matrix1.finalPosRows;
				size_t finalPosCols = matrix1.finalPosCols;

				for (size_t i = 0; i < finalPosRows; i += 8)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256 a = _mm256_setr_ps(data1[i * matrix1ActualCols + j],
							data1[(i + 1) * matrix1ActualCols + j],
							data1[(i + 2) * matrix1ActualCols + j],
							data1[(i + 3) * matrix1ActualCols + j],
							data1[(i + 4) * matrix1ActualCols + j],
							data1[(i + 5) * matrix1ActualCols + j],
							data1[(i + 6) * matrix1ActualCols + j],
							data1[(i + 7) * matrix1ActualCols + j]);

						_mm256_store_ps(&dataResult[j * rows + i], _mm256_sub_ps(b, a));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						dataResult[j * rows + i] = data1[i * matrix1ActualCols + j] - num;
					}
				}
			}
			else
			{
				if constexpr (thisContiguous)
				{
					size_t finalPosSize = matrix1.finalPosSize;
					size_t size = matrix1._size;

					for (size_t i = 0; i < finalPosSize; i += 8)
					{
						__m256 a = _mm256_load_ps(&data1[i]);

						_mm256_store_ps(&dataResult[i], _mm256_sub_ps(b, a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = data1[i] - num;
					}
				}
				else
				{
					size_t matrix1ActualCols = matrix1.actualCols;

					size_t finalPosCols = matrix1.finalPosCols;
					size_t finalPosRows = matrix1.finalPosRows;

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
							dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] - num;
						}
					}
				}
			}
		}
		return result;
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

		double* data1 = matrix1._data;

		matrix<double> result(rows, cols);

		double* dataResult = result._data;

		__m256d b = _mm256_set1_pd(num);

		if constexpr (thisTransposed)
		{
			if constexpr (returnTransposed)
			{
				if constexpr (thisContiguous)
				{
					size_t size = matrix1._size;

					size_t finalPosSize = matrix1.finalPosSize;

					for (size_t i = 0; i < finalPosSize; i += 4)
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
					size_t finalPosRows = matrix1.finalPosRows;
					size_t finalPosCols = matrix1.finalPosCols;

					size_t matrix1ActualRows = matrix1.actualRows;

					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

							_mm256_store_pd(&dataResult[j * rows + i], _mm256_div_pd(b, a));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < finalPosCols; j += 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
								data1[(j + 1) * matrix1ActualRows + i],
								data1[(j + 2) * matrix1ActualRows + i],
								data1[(j + 3) * matrix1ActualRows + i]);

							__m256d div = _mm256_div_pd(b, a);

							__m128d val1 = _mm256_extractf128_pd(div, 1);
							__m128d val2 = _mm256_castpd256_pd128(div);

							_mm_store_sd(&dataResult[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

							_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
						}
						for (size_t j = finalPosCols; j < cols; j++)
						{
							dataResult[j * rows + i] = num / data1[j * matrix1ActualRows + i];
						}
					}
				}
			}
			else
			{
				size_t finalPosCols = matrix1.finalPosCols;
				size_t finalPosRows = matrix1.finalPosRows;

				size_t matrix1ActualRows = matrix1.actualRows;

				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
							data1[(j + 1) * matrix1ActualRows + i],
							data1[(j + 2) * matrix1ActualRows + i],
							data1[(j + 3) * matrix1ActualRows + i]);

						_mm256_store_pd(&dataResult[i * cols + j], _mm256_div_pd(b, a));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						__m256d div = _mm256_div_pd(b, a);

						__m128d val1 = _mm256_extractf128_pd(div, 1);
						__m128d val2 = _mm256_castpd256_pd128(div);

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
		}
		else
		{
			if constexpr (returnTransposed)
			{
				size_t matrix1ActualCols = matrix1.actualCols;

				size_t finalPosRows = matrix1.finalPosRows;
				size_t finalPosCols = matrix1.finalPosCols;

				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
							data1[(i + 1) * matrix1ActualCols + j],
							data1[(i + 2) * matrix1ActualCols + j],
							data1[(i + 3) * matrix1ActualCols + j]);

						_mm256_store_pd(&dataResult[j * rows + i], _mm256_div_pd(b, a));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						__m256d div = _mm256_div_pd(b, a);

						__m128d val1 = _mm256_extractf128_pd(div, 1);
						__m128d val2 = _mm256_castpd256_pd128(div);

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
			else
			{
				if constexpr (thisContiguous)
				{
					size_t finalPosSize = matrix1.finalPosSize;
					size_t size = matrix1._size;

					for (size_t i = 0; i < finalPosSize; i += 4)
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
					size_t matrix1ActualCols = matrix1.actualCols;

					size_t finalPosCols = matrix1.finalPosCols;
					size_t finalPosRows = matrix1.finalPosRows;

					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

							_mm256_store_pd(&dataResult[i * cols + j], _mm256_div_pd(b, a));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < finalPosRows; i += 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
								data1[(i + 1) * matrix1ActualCols + j],
								data1[(i + 2) * matrix1ActualCols + j],
								data1[(i + 3) * matrix1ActualCols + j]);

							__m256d div = _mm256_div_pd(b, a);

							__m128d val1 = _mm256_extractf128_pd(div, 1);
							__m128d val2 = _mm256_castpd256_pd128(div);

							_mm_store_sd(&dataResult[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

							_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
						}
						for (size_t i = finalPosRows; i < rows; i++)
						{
							dataResult[i * cols + j] = num / data1[i * matrix1ActualCols + j];
						}
					}
				}
			}
		}
		return result;
	}

	template<bool returnTransposed = false, bool thisTransposed, bool thisContiguous>
	inline matrix<float> operator/(float num, matrix<float, thisTransposed, thisContiguous>& matrix1)
	{
		size_t rows = matrix1._rows;
		size_t cols = matrix1._cols;

		float* data1 = matrix1._data;

		matrix<float> result(rows, cols);

		float* dataResult = result._data;

		__m256 b = _mm256_set1_ps(num);

		if constexpr (thisTransposed)
		{
			if constexpr (returnTransposed)
			{
				if constexpr (thisContiguous)
				{
					size_t size = matrix1._size;

					size_t finalPosSize = matrix1.finalPosSize;

					for (size_t i = 0; i < finalPosSize; i += 8)
					{
						__m256 a = _mm256_load_ps(&data1[i]);

						_mm256_store_ps(&dataResult[i], _mm256_div_ps(b, a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = data1[i] / num;
					}
				}
				else
				{
					size_t finalPosRows = matrix1.finalPosRows;
					size_t finalPosCols = matrix1.finalPosCols;

					size_t matrix1ActualRows = matrix1.actualRows;

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
							dataResult[j * rows + i] = data1[j * matrix1ActualRows + i] / num;
						}
					}
				}
			}
			else
			{
				size_t finalPosCols = matrix1.finalPosCols;
				size_t finalPosRows = matrix1.finalPosRows;

				size_t matrix1ActualRows = matrix1.actualRows;

				for (size_t j = 0; j < finalPosCols; j += 8)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256 a = _mm256_setr_ps(data1[j * matrix1ActualRows + i],
							data1[(j + 1) * matrix1ActualRows + i],
							data1[(j + 2) * matrix1ActualRows + i],
							data1[(j + 3) * matrix1ActualRows + i],
							data1[(j + 4) * matrix1ActualRows + i],
							data1[(j + 5) * matrix1ActualRows + i],
							data1[(j + 6) * matrix1ActualRows + i],
							data1[(j + 7) * matrix1ActualRows + i]);

						_mm256_store_ps(&dataResult[i * cols + j], _mm256_div_ps(b, a));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < rows; i++)
					{
						dataResult[i * cols + j] = data1[j * matrix1ActualRows + i] / num;
					}
				}
			}
		}
		else
		{
			if constexpr (returnTransposed)
			{
				size_t matrix1ActualCols = matrix1.actualCols;

				size_t finalPosRows = matrix1.finalPosRows;
				size_t finalPosCols = matrix1.finalPosCols;

				for (size_t i = 0; i < finalPosRows; i += 8)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256 a = _mm256_setr_ps(data1[i * matrix1ActualCols + j],
							data1[(i + 1) * matrix1ActualCols + j],
							data1[(i + 2) * matrix1ActualCols + j],
							data1[(i + 3) * matrix1ActualCols + j],
							data1[(i + 4) * matrix1ActualCols + j],
							data1[(i + 5) * matrix1ActualCols + j],
							data1[(i + 6) * matrix1ActualCols + j],
							data1[(i + 7) * matrix1ActualCols + j]);

						_mm256_store_ps(&dataResult[j * rows + i], _mm256_div_ps(b, a));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						dataResult[j * rows + i] = data1[i * matrix1ActualCols + j] / num;
					}
				}
			}
			else
			{
				if constexpr (thisContiguous)
				{
					size_t finalPosSize = matrix1.finalPosSize;
					size_t size = matrix1._size;

					for (size_t i = 0; i < finalPosSize; i += 8)
					{
						__m256 a = _mm256_load_ps(&data1[i]);

						_mm256_store_ps(&dataResult[i], _mm256_div_ps(b, a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = data1[i] / num;
					}
				}
				else
				{
					size_t matrix1ActualCols = matrix1.actualCols;

					size_t finalPosCols = matrix1.finalPosCols;
					size_t finalPosRows = matrix1.finalPosRows;

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
							dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] / num;
						}
					}
				}
			}
		}
		return result;
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