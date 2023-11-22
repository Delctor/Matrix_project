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
	inline double dot(vector<double>& vector1, vector<double>& vector2)
	{
#ifdef _DEBUG
		if (vector1._size != vector2._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = vector1._size;

		size_t finalPos = vector1.finalPos;

		double* data1 = vector1._data;
		double* data2 = vector2._data;

		vector<double> result(size);

		double* dataResult = result._data;

		double dotProduct = 0;

		__m256d _dotProduct = _mm256_setzero_pd();

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_load_pd(&data1[i]);
			__m256d b = _mm256_load_pd(&data2[i]);

			_dotProduct = _mm256_fmadd_pd(a, b, _dotProduct);
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dotProduct += data1[i] * data2[i];
		}

		__m128d vlow = _mm256_castpd256_pd128(_dotProduct);
		__m128d vhigh = _mm256_extractf128_pd(_dotProduct, 1);
		vlow = _mm_add_pd(vlow, vhigh);

		__m128d high64 = _mm_unpackhi_pd(vlow, vlow);
		dotProduct += _mm_cvtsd_f64(_mm_add_sd(vlow, high64));

		return dotProduct;
	}

	template<bool returnTransposed = false, bool matrix1Transposed, bool matrix1Contiguous,
		bool matrix2Transposed, bool matrix2Contiguous>
	inline matrix<double> dot(matrix<double, matrix1Transposed, matrix1Contiguous>& matrix1, matrix<double, matrix2Transposed, matrix2Contiguous>& matrix2)
	{
#ifdef _DEBUG
		if (matrix1._cols != matrix2._rows) throw std::invalid_argument("Wrong dimensions");
#else
#endif

		size_t matrix1Rows = matrix1._rows;
		size_t matrix1Cols = matrix1._cols;

		size_t matrix2Rows = matrix2._rows;
		size_t matrix2Cols = matrix2._cols;

		double* data1 = matrix1._data;
		double* data2 = matrix1._data;

		size_t matrix1ActualRows = matrix1.actualRows;
		size_t matrix1ActualCols = matrix1.actualCols;
		size_t matrix2ActualRows = matrix2.actualRows;
		size_t matrix2ActualCols = matrix2.actualCols;

		size_t matrix1FinalPosRows = matrix1.finalPosRows;
		size_t matrix1FinalPosCols = matrix1.finalPosCols;
		size_t matrix2FinalPosRows = matrix2.finalPosRows;
		size_t matrix2FinalPosCols = matrix2.finalPosCols;

		if constexpr (returnTransposed)
		{
			matrix<double> result(matrix2Cols, matrix1Rows);

			double* dataResult = result._data;

			if constexpr (matrix1Transposed)
			{
				if constexpr (matrix2Transposed)
				{
					for (size_t i = 0; i < matrix1FinalPosRows; i += 4)
					{
						for (size_t j = 0; j < matrix2Cols; j++)
						{
							__m256d _sum = _mm256_setzero_pd();

							for (size_t k = 0; k < matrix1Cols; k++)
							{
								_sum = _mm256_fmadd_pd(_mm256_load_pd(&data1[k * matrix1ActualRows + i]),
									_mm256_broadcast_sd(&data2[j * matrix2ActualRows + k]), _sum);
							}
							_mm256_store_pd(&dataResult[j * matrix1Rows + i], _sum);
						}
					}
					for (size_t i = matrix1FinalPosRows; i < matrix1Rows; i++)
					{
						for (size_t j = 0; j < matrix2Cols; j++)
						{
							double sum = 0.0;
							for (size_t k = 0; k < matrix1Cols; k++)
							{
								sum += data1[k * matrix1ActualRows + i] * data2[j * matrix2ActualRows + k];
							}

							dataResult[j * matrix1Rows + i] = sum;
						}
					}
				}
				else
				{
					for (size_t i = 0; i < matrix1FinalPosRows; i += 4)
					{
						for (size_t j = 0; j < matrix2Cols; j++)
						{
							__m256d _sum = _mm256_setzero_pd();

							for (size_t k = 0; k < matrix1Cols; k++)
							{
								_sum = _mm256_fmadd_pd(_mm256_load_pd(&data1[k * matrix1ActualRows + i]),
									_mm256_broadcast_sd(&data2[k * matrix2ActualCols + j]), _sum);
							}
							_mm256_store_pd(&dataResult[j * matrix1Rows + i], _sum);
						}
					}
					for (size_t i = matrix1FinalPosRows; i < matrix1Rows; i++)
					{
						for (size_t j = 0; j < matrix2FinalPosCols; j += 4)
						{
							__m256d _sum = _mm256_setzero_pd();
							for (size_t k = 0; k < matrix1Cols; k++)
							{
								__m256d a = _mm256_broadcast_sd(&data1[k * matrix1ActualRows + i]);
								__m256d b = _mm256_load_pd(&data2[k * matrix2ActualCols + j]);

								_sum = _mm256_fmadd_pd(a, b, _sum);
							}

							__m128d val1 = _mm256_extractf128_pd(_sum, 1);
							__m128d val2 = _mm256_castpd256_pd128(_sum);

							_mm_store_sd(&dataResult[j * matrix1Rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(j + 1) * matrix1Rows + i], val2);

							_mm_store_sd(&dataResult[(j + 2) * matrix1Rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(j + 3) * matrix1Rows + i], val1);
						}
						for (size_t j = matrix2FinalPosCols; j < matrix2Cols; j++)
						{
							double sum = 0.0;
							
							for (size_t k = 0; k < matrix1Cols; k++)
							{
								sum += data1[k * matrix1ActualRows + i] * data2[k * matrix2ActualCols + j];
							}
							dataResult[j * matrix1Rows + i] = sum;
						}
					}
				}
			}
			else
			{
				if constexpr (matrix2Transposed)
				{
					for (size_t i = 0; i < matrix1Rows; i++)
					{
						for (size_t j = 0; j < matrix2Cols; j++)
						{
							__m256d _sum = _mm256_setzero_pd();

							for (size_t k = 0; k < matrix1FinalPosCols; k += 4)
							{
								_sum = _mm256_fmadd_pd(_mm256_load_pd(&data1[i * matrix1ActualCols + k]),
									_mm256_load_pd(&data2[j * matrix2ActualRows + k]), _sum);
							}
							__m128d vlow = _mm256_castpd256_pd128(_sum);
							__m128d vhigh = _mm256_extractf128_pd(_sum, 1);
							vlow = _mm_add_pd(vlow, vhigh);

							__m128d high64 = _mm_unpackhi_pd(vlow, vlow);
							__m128d _sum128 = _mm_add_sd(vlow, high64);

							for (size_t k = matrix1FinalPosCols; k < matrix1Cols; k++)
							{
								_sum128 = _mm_fmadd_sd(_mm_load_sd(&data1[i * matrix1ActualCols + k]), 
									_mm_load_sd(&data2[j * matrix2ActualRows + k]), _sum128);
							}
							_mm_store_sd(&dataResult[j * matrix1Rows + i], _sum128);
						}
					}
				}
				else
				{
					for (size_t i = 0; i < matrix1Rows; i++)
					{
						for (size_t j = 0; j < matrix2FinalPosCols; j += 4)
						{
							__m256d _sum = _mm256_setzero_pd();

							for (size_t k = 0; k < matrix1Cols; k++)
							{
								_sum = _mm256_fmadd_pd(_mm256_broadcast_sd(&data1[i * matrix1ActualCols + k]),
									_mm256_load_pd(&data2[k * matrix2ActualCols + j]), _sum);
							}

							__m128d val1 = _mm256_extractf128_pd(_sum, 1);
							__m128d val2 = _mm256_castpd256_pd128(_sum);

							_mm_store_sd(&dataResult[j * matrix1Rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(j + 1) * matrix1Rows + i], val2);

							_mm_store_sd(&dataResult[(j + 2) * matrix1Rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(j + 3) * matrix1Rows + i], val1);
						}
						for (size_t j = matrix2FinalPosCols; j < matrix2Cols; j++)
						{
							double sum = 0.0;
							for (size_t k = 0; k < matrix1Cols; k++)
							{
								sum += data1[i * matrix1ActualCols + k] * data2[k * matrix2ActualCols + j];
							}
							dataResult[j * matrix1Rows + i] = sum;
						}
					}
				}
			}
			return result;
		}
		else
		{
			matrix<double> result(matrix1Rows, matrix2Cols);

			double* dataResult = result._data;

			if constexpr (matrix1Transposed)
			{
				if constexpr (matrix2Transposed)
				{
					for (size_t j = 0; j < matrix2Cols; j++)
					{
						for (size_t i = 0; i < matrix1FinalPosRows; i += 4)
						{
							__m256d _sum = _mm256_setzero_pd();

							for (size_t k = 0; k < matrix1Cols; k++)
							{
								_sum = _mm256_fmadd_pd(_mm256_load_pd(&data1[k * matrix1ActualRows + i]),
									_mm256_broadcast_sd(&data2[j * matrix2ActualRows + k]),
									_sum);
							}
							__m128d val1 = _mm256_extractf128_pd(_sum, 1);
							__m128d val2 = _mm256_castpd256_pd128(_sum);

							_mm_store_sd(&dataResult[i * matrix2Cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(i + 1) * matrix2Cols + j], val2);

							_mm_store_sd(&dataResult[(i + 2) * matrix2Cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(i + 3) * matrix2Cols + j], val1);
						}
						for (size_t i = matrix1FinalPosRows; i < matrix1Rows; i++)
						{
							double sum = 0.0;

							for (size_t k = 0; k < matrix1Cols; k++)
							{
								sum += data1[k * matrix1ActualRows + i] * data2[j * matrix2ActualRows + k];
							}
							dataResult[i * matrix2Cols + j] = sum;
						}
					}
				}
				else
				{
					for (size_t j = 0; j < matrix2FinalPosCols; j += 4)
					{
						for (size_t i = 0; i < matrix1Rows; i++)
						{
							__m256d _sum = _mm256_setzero_pd();

							for (size_t k = 0; k < matrix1Cols; k++)
							{
								_sum = _mm256_fmadd_pd(_mm256_broadcast_sd(&data1[k * matrix1ActualRows + i]),
									_mm256_load_pd(&data2[k * matrix2ActualCols + j]),
									_sum);
							}
							_mm256_store_pd(&dataResult[i * matrix2Cols + j], _sum);
						}
					}
					for (size_t j = matrix2FinalPosCols; j < matrix2Cols; j++)
					{
						for (size_t i = 0; i < matrix1FinalPosRows; i += 4)
						{
							__m256d _sum = _mm256_setzero_pd();

							for (size_t k = 0; k < matrix1Cols; k++)
							{
								_sum = _mm256_fmadd_pd(_mm256_load_pd(&data1[k * matrix1ActualRows + i]),
									_mm256_broadcast_sd(&data2[k * matrix2ActualCols + j]),
									_sum);
							}

							__m128d val1 = _mm256_extractf128_pd(_sum, 1);
							__m128d val2 = _mm256_castpd256_pd128(_sum);

							_mm_store_sd(&dataResult[i * matrix2Cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(i + 1) * matrix2Cols + j], val2);

							_mm_store_sd(&dataResult[(i + 2) * matrix2Cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(i + 3) * matrix2Cols + j], val1);
						}
						for (size_t i = matrix1FinalPosRows; i < matrix1Rows; i++)
						{
							double sum = 0.0;

							for (size_t k = 0; k < matrix1Cols; k++)
							{
								sum += data1[k * matrix1ActualRows + i] * data2[k * matrix2ActualCols + j];
							}
							dataResult[i * matrix2Cols + j] = sum;
						}
					}	  
				}
			}
			else
			{
				if constexpr (matrix2Transposed)
				{
					for (size_t j = 0; j < matrix2Cols; j++)
					{
						for (size_t i = 0; i < matrix1Rows; i++)
						{
							__m256d _sum = _mm256_setzero_pd();

							for (size_t k = 0; k < matrix1FinalPosCols; k += 4)
							{
								_sum = _mm256_fmadd_pd(_mm256_load_pd(&data1[i * matrix1ActualCols + k]),
									_mm256_load_pd(&data2[j * matrix2ActualRows + k]), _sum);
							}
							__m128d vlow = _mm256_castpd256_pd128(_sum);
							__m128d vhigh = _mm256_extractf128_pd(_sum, 1);
							vlow = _mm_add_pd(vlow, vhigh);

							__m128d high64 = _mm_unpackhi_pd(vlow, vlow);
							__m128d _sum128 = _mm_add_sd(vlow, high64);

							for (size_t k = matrix1FinalPosCols; k < matrix1Cols; k++)
							{
								_sum128 = _mm_fmadd_sd(_mm_load_sd(&data1[i * matrix1ActualCols + k]), 
									_mm_load_sd(&data2[j * matrix2ActualRows + k]), _sum128);
							}
							_mm_store_sd(&dataResult[i * matrix1Cols + j], _sum128);
						}
					}
				}
				else
				{
					for (size_t j = 0; j < matrix2FinalPosCols; j += 4)
					{
						for (size_t i = 0; i < matrix1Rows; i++)
						{
							__m256d _sum = _mm256_setzero_pd();

							for (size_t k = 0; k < matrix1Cols; k++)
							{
								_sum = _mm256_fmadd_pd(_mm256_broadcast_sd(&data1[i * matrix1ActualCols + k]),
									_mm256_load_pd(&data2[k * matrix2ActualCols + j]), _sum);
							}
							_mm256_store_pd(&dataResult[i * matrix2Cols + j], _sum);
						}
					}
					for (size_t j = matrix2FinalPosCols; j < matrix2Cols; j++)
					{
						for (size_t i = 0; i < matrix1Rows; i++)
						{
							double sum = 0.0;
							for (size_t k = 0; k < matrix1Cols; k++)
							{
								sum += data1[i * matrix1ActualCols + k] * data2[k * matrix2ActualCols + j];
							}
							dataResult[i * matrix2Cols + j] = sum;
						}
					}
				}
			}
			return result;
		}
	}

	inline float dot(vector<float>& vector1, vector<float>& vector2)
	{
#ifdef _DEBUG
		if (vector1._size != vector2._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = vector1._size;

		size_t finalPos = vector1.finalPos;

		float* data1 = vector1._data;
		float* data2 = vector2._data;

		vector<float> result(size);

		float* dataResult = result._data;

		__m256 _dotProduct = _mm256_setzero_ps();

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256 a = _mm256_load_ps(&data1[i]);
			__m256 b = _mm256_load_ps(&data2[i]);

			_dotProduct = _mm256_fmadd_ps(a, b, _dotProduct);
		}

		__m256 _sum1 = _mm256_hadd_ps(_dotProduct, _dotProduct);
		__m256 _sum2 = _mm256_hadd_ps(_sum1, _sum1);

		__m128 lo128 = _mm256_castps256_ps128(_sum2);
		__m128 hi128 = _mm256_extractf128_ps(_sum2, 1);
		__m128 result128 = _mm_add_ps(lo128, hi128);
		float dotProduct = _mm_cvtss_f32(result128);

		for (size_t i = finalPos; i < size; i++)
		{
			dotProduct += data1[i] * data2[i];
		}

		return dotProduct;
	}

	template<bool returnTransposed = false, bool matrix1Transposed, bool matrix1Contiguous,
		bool matrix2Transposed, bool matrix2Contiguous>
	inline matrix<float> dot(matrix<float, matrix1Transposed, matrix1Contiguous>& matrix1, matrix<float, matrix2Transposed, matrix2Contiguous>& matrix2)
	{
#ifdef _DEBUG
		if (matrix1._cols != matrix2._rows) throw std::invalid_argument("Wrong dimensions");
#else
#endif

		size_t matrix1Rows = matrix1._rows;
		size_t matrix1Cols = matrix1._cols;

		size_t matrix2Rows = matrix2._rows;
		size_t matrix2Cols = matrix2._cols;

		float* data1 = matrix1._data;
		float* data2 = matrix1._data;

		size_t matrix1ActualRows = matrix1.actualRows;
		size_t matrix1ActualCols = matrix1.actualCols;
		size_t matrix2ActualRows = matrix2.actualRows;
		size_t matrix2ActualCols = matrix2.actualCols;

		size_t matrix1FinalPosRows = matrix1.finalPosRows;
		size_t matrix1FinalPosCols = matrix1.finalPosCols;
		size_t matrix2FinalPosRows = matrix2.finalPosRows;
		size_t matrix2FinalPosCols = matrix2.finalPosCols;

		if constexpr (returnTransposed)
		{
			matrix<float> result(matrix2Cols, matrix1Rows);

			float* dataResult = result._data;

			if constexpr (matrix1Transposed)
			{
				if constexpr (matrix2Transposed)
				{
					for (size_t i = 0; i < matrix1FinalPosRows; i += 8)
					{
						for (size_t j = 0; j < matrix2Cols; j++)
						{
							__m256 _sum = _mm256_setzero_ps();

							for (size_t k = 0; k < matrix1Cols; k++)
							{
								_sum = _mm256_fmadd_ps(_mm256_load_ps(&data1[k * matrix1ActualRows + i]),
									_mm256_broadcast_ss(&data2[j * matrix2ActualRows + k]), _sum);
							}
							_mm256_store_ps(&dataResult[j * matrix1Rows + i], _sum);
						}
					}
					for (size_t i = matrix1FinalPosRows; i < matrix1Rows; i++)
					{
						for (size_t j = 0; j < matrix2Cols; j++)
						{
							float sum = 0.0;
							for (size_t k = 0; k < matrix1Cols; k++)
							{
								sum += data1[k * matrix1ActualRows + i] * data2[j * matrix2ActualRows + k];
							}

							dataResult[j * matrix1Rows + i] = sum;
						}
					}
				}
				else
				{
					for (size_t i = 0; i < matrix1FinalPosRows; i += 8)
					{
						for (size_t j = 0; j < matrix2Cols; j++)
						{
							__m256 _sum = _mm256_setzero_ps();

							for (size_t k = 0; k < matrix1Cols; k++)
							{
								_sum = _mm256_fmadd_ps(_mm256_load_ps(&data1[k * matrix1ActualRows + i]),
									_mm256_broadcast_ss(&data2[k * matrix2ActualCols + j]), _sum);
							}
							_mm256_store_ps(&dataResult[j * matrix1Rows + i], _sum);
						}
					}
					for (size_t i = matrix1FinalPosRows; i < matrix1Rows; i++)
					{
						for (size_t j = 0; j < matrix2FinalPosCols; j += 8)
						{
							__m256 _sum = _mm256_setzero_ps();
							for (size_t k = 0; k < matrix1Cols; k++)
							{
								__m256 a = _mm256_broadcast_ss(&data1[k * matrix1ActualRows + i]);
								__m256 b = _mm256_load_ps(&data2[k * matrix2ActualCols + j]);

								_sum = _mm256_fmadd_ps(a, b, _sum);
							}

							__m128 high = _mm256_extractf128_ps(_sum, 1);
							__m128 low = _mm256_castps256_ps128(_sum);

							// 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0

							// 1.0, 2.0, 3.0, 4.0
							_mm_store_ss(&dataResult[j * matrix1Rows + i], low);

							low = _mm_shuffle_ps(low, low, 0b11100001);
							// 2.0, 1.0, 3.0, 4.0
							_mm_store_ss(&dataResult[(j + 1) * matrix1Rows + i], low);

							low = _mm_shuffle_ps(low, low, 0b11000110);
							// 3.0, 1.0, 2.0, 4.0
							_mm_store_ss(&dataResult[(j + 2) * matrix1Rows + i], low);

							low = _mm_shuffle_ps(low, low, 0b00100111);
							// 4.0, 1.0, 2.0, 3.0
							_mm_store_ss(&dataResult[(j + 3) * matrix1Rows + i], low);

							// --

							// 5.0, 6.0, 7.0, 8.0
							_mm_store_ss(&dataResult[(j + 4) * matrix1Rows + i], high);

							high = _mm_shuffle_ps(high, high, 0b11100001);
							// 6.0, 5.0, 7.0, 8.0
							_mm_store_ss(&dataResult[(j + 5) * matrix1Rows + i], high);

							high = _mm_shuffle_ps(high, high, 0b11000110);
							// 7.0, 5.0, 6.0, 8.0
							_mm_store_ss(&dataResult[(j + 6) * matrix1Rows + i], high);

							high = _mm_shuffle_ps(high, high, 0b00100111);
							// 8.0, 5.0, 6.0, 7.0
							_mm_store_ss(&dataResult[(j + 7) * matrix1Rows + i], high);
						}
						for (size_t j = matrix2FinalPosCols; j < matrix2Cols; j++)
						{
							float sum = 0.0;

							for (size_t k = 0; k < matrix1Cols; k++)
							{
								sum += data1[k * matrix1ActualRows + i] * data2[k * matrix2ActualCols + j];
							}
							dataResult[j * matrix1Rows + i] = sum;
						}
					}
				}
			}
			else
			{
				if constexpr (matrix2Transposed)
				{
					for (size_t i = 0; i < matrix1Rows; i++)
					{
						for (size_t j = 0; j < matrix2Cols; j++)
						{
							__m256 _sum = _mm256_setzero_ps();

							for (size_t k = 0; k < matrix1FinalPosCols; k += 8)
							{
								_sum = _mm256_fmadd_ps(_mm256_load_ps(&data1[i * matrix1ActualCols + k]),
									_mm256_load_ps(&data2[j * matrix2ActualRows + k]), _sum);
							}
							__m256 _sum1 = _mm256_hadd_ps(_sum, _sum);
							__m256 _sum2 = _mm256_hadd_ps(_sum1, _sum1);

							__m128 lo128 = _mm256_castps256_ps128(_sum2, 0);
							__m128 hi128 = _mm256_extractf128_ps(_sum2, 1);
							__m128 _sum128 = _mm_add_ps(lo128, hi128);

							for (size_t k = matrix1FinalPosCols; k < matrix1Cols; k++)
							{
								_sum128 = _mm_fmadd_ss(_mm_load_ss(&data1[i * matrix1ActualCols + k]),
									_mm_load_ss(&data2[j * matrix2ActualRows + k]), _sum128);
							}
							_mm_store_ss(&dataResult[j * matrix1Rows + i], _sum128);
						}
					}
				}
				else
				{
					for (size_t i = 0; i < matrix1Rows; i++)
					{
						for (size_t j = 0; j < matrix2FinalPosCols; j += 8)
						{
							__m256 _sum = _mm256_setzero_ps();

							for (size_t k = 0; k < matrix1Cols; k++)
							{
								_sum = _mm256_fmadd_ps(_mm256_broadcast_ss(&data1[i * matrix1ActualCols + k]),
									_mm256_load_ps(&data2[k * matrix2ActualCols + j]), _sum);
							}

							__m128 high = _mm256_extractf128_ps(_sum, 1);
							__m128 low = _mm256_castps256_ps128(_sum);

							// 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0

							// 1.0, 2.0, 3.0, 4.0
							_mm_store_ss(&dataResult[j * matrix1Rows + i], low);

							low = _mm_shuffle_ps(low, low, 0b11100001);
							// 2.0, 1.0, 3.0, 4.0
							_mm_store_ss(&dataResult[(j + 1) * matrix1Rows + i], low);

							low = _mm_shuffle_ps(low, low, 0b11000110);
							// 3.0, 1.0, 2.0, 4.0
							_mm_store_ss(&dataResult[(j + 2) * matrix1Rows + i], low);

							low = _mm_shuffle_ps(low, low, 0b00100111);
							// 4.0, 1.0, 2.0, 3.0
							_mm_store_ss(&dataResult[(j + 3) * matrix1Rows + i], low);

							// --

							// 5.0, 6.0, 7.0, 8.0
							_mm_store_ss(&dataResult[(j + 4) * matrix1Rows + i], high);

							high = _mm_shuffle_ps(high, high, 0b11100001);
							// 6.0, 5.0, 7.0, 8.0
							_mm_store_ss(&dataResult[(j + 5) * matrix1Rows + i], high);

							high = _mm_shuffle_ps(high, high, 0b11000110);
							// 7.0, 5.0, 6.0, 8.0
							_mm_store_ss(&dataResult[(j + 6) * matrix1Rows + i], high);

							high = _mm_shuffle_ps(high, high, 0b00100111);
							// 8.0, 5.0, 6.0, 7.0
							_mm_store_ss(&dataResult[(j + 7) * matrix1Rows + i], high);
						}
						for (size_t j = matrix2FinalPosCols; j < matrix2Cols; j++)
						{
							float sum = 0.0;
							for (size_t k = 0; k < matrix1Cols; k++)
							{
								sum += data1[i * matrix1ActualCols + k] * data2[k * matrix2ActualCols + j];
							}
							dataResult[j * matrix1Rows + i] = sum;
						}
					}
				}
			}
			return result;
		}
		else
		{
			matrix<float> result(matrix1Rows, matrix2Cols);

			float* dataResult = result._data;

			if constexpr (matrix1Transposed)
			{
				if constexpr (matrix2Transposed)
				{
					for (size_t j = 0; j < matrix2Cols; j++)
					{
						for (size_t i = 0; i < matrix1FinalPosRows; i += 8)
						{
							__m256 _sum = _mm256_setzero_ps();

							for (size_t k = 0; k < matrix1Cols; k++)
							{
								_sum = _mm256_fmadd_ps(_mm256_load_ps(&data1[k * matrix1ActualRows + i]),
									_mm256_broadcast_ss(&data2[j * matrix2ActualRows + k]),
									_sum);
							}
							__m128 high = _mm256_extractf128_ps(_sum, 1);
							__m128 low = _mm256_castps256_ps128(_sum);

							// 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0

							// 1.0, 2.0, 3.0, 4.0
							_mm_store_ss(&dataResult[i * matrix2Cols + j], low);

							low = _mm_shuffle_ps(low, low, 0b11100001);
							// 2.0, 1.0, 3.0, 4.0
							_mm_store_ss(&dataResult[(i + 1) * matrix2Cols + j], low);

							low = _mm_shuffle_ps(low, low, 0b11000110);
							// 3.0, 1.0, 2.0, 4.0
							_mm_store_ss(&dataResult[(i + 2) * matrix2Cols + j], low);

							low = _mm_shuffle_ps(low, low, 0b00100111);
							// 4.0, 1.0, 2.0, 3.0
							_mm_store_ss(&dataResult[(i + 3) * matrix2Cols + j], low);

							// --

							// 5.0, 6.0, 7.0, 8.0
							_mm_store_ss(&dataResult[(i + 4) * matrix2Cols + j], high);

							high = _mm_shuffle_ps(high, high, 0b11100001);
							// 6.0, 5.0, 7.0, 8.0
							_mm_store_ss(&dataResult[(i + 5) * matrix2Cols + j], high);

							high = _mm_shuffle_ps(high, high, 0b11000110);
							// 7.0, 5.0, 6.0, 8.0
							_mm_store_ss(&dataResult[(i + 6) * matrix2Cols + j], high);

							high = _mm_shuffle_ps(high, high, 0b00100111);
							// 8.0, 5.0, 6.0, 7.0
							_mm_store_ss(&dataResult[(i + 7) * matrix2Cols + j], high);
						}
						for (size_t i = matrix1FinalPosRows; i < matrix1Rows; i++)
						{
							float sum = 0.0;

							for (size_t k = 0; k < matrix1Cols; k++)
							{
								sum += data1[k * matrix1ActualRows + i] * data2[j * matrix2ActualRows + k];
							}
							dataResult[i * matrix2Cols + j] = sum;
						}
					}
				}
				else
				{
					for (size_t j = 0; j < matrix2FinalPosCols; j += 8)
					{
						for (size_t i = 0; i < matrix1Rows; i++)
						{
							__m256 _sum = _mm256_setzero_ps();

							for (size_t k = 0; k < matrix1Cols; k++)
							{
								_sum = _mm256_fmadd_ps(_mm256_broadcast_ss(&data1[k * matrix1ActualRows + i]),
									_mm256_load_ps(&data2[k * matrix2ActualCols + j]),
									_sum);
							}
							_mm256_store_ps(&dataResult[i * matrix2Cols + j], _sum);
						}
					}
					for (size_t j = matrix2FinalPosCols; j < matrix2Cols; j++)
					{
						for (size_t i = 0; i < matrix1FinalPosRows; i += 8)
						{
							__m256 _sum = _mm256_setzero_ps();

							for (size_t k = 0; k < matrix1Cols; k++)
							{
								_sum = _mm256_fmadd_ps(_mm256_load_ps(&data1[k * matrix1ActualRows + i]),
									_mm256_broadcast_ss(&data2[k * matrix2ActualCols + j]),
									_sum);
							}

							__m128 high = _mm256_extractf128_ps(_sum, 1);
							__m128 low = _mm256_castps256_ps128(_sum);

							// 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0

							// 1.0, 2.0, 3.0, 4.0
							_mm_store_ss(&dataResult[i * matrix2Cols + j], low);

							low = _mm_shuffle_ps(low, low, 0b11100001);
							// 2.0, 1.0, 3.0, 4.0
							_mm_store_ss(&dataResult[(i + 1) * matrix2Cols + j], low);

							low = _mm_shuffle_ps(low, low, 0b11000110);
							// 3.0, 1.0, 2.0, 4.0
							_mm_store_ss(&dataResult[(i + 2) * matrix2Cols + j], low);

							low = _mm_shuffle_ps(low, low, 0b00100111);
							// 4.0, 1.0, 2.0, 3.0
							_mm_store_ss(&dataResult[(i + 3) * matrix2Cols + j], low);

							// --

							// 5.0, 6.0, 7.0, 8.0
							_mm_store_ss(&dataResult[(i + 4) * matrix2Cols + j], high);

							high = _mm_shuffle_ps(high, high, 0b11100001);
							// 6.0, 5.0, 7.0, 8.0
							_mm_store_ss(&dataResult[(i + 5) * matrix2Cols + j], high);

							high = _mm_shuffle_ps(high, high, 0b11000110);
							// 7.0, 5.0, 6.0, 8.0
							_mm_store_ss(&dataResult[(i + 6) * matrix2Cols + j], high);

							high = _mm_shuffle_ps(high, high, 0b00100111);
							// 8.0, 5.0, 6.0, 7.0
							_mm_store_ss(&dataResult[(i + 7) * matrix2Cols + j], high);
						}
						for (size_t i = matrix1FinalPosRows; i < matrix1Rows; i++)
						{
							float sum = 0.0;

							for (size_t k = 0; k < matrix1Cols; k++)
							{
								sum += data1[k * matrix1ActualRows + i] * data2[k * matrix2ActualCols + j];
							}
							dataResult[i * matrix2Cols + j] = sum;
						}
					}
				}
			}
			else
			{
				if constexpr (matrix2Transposed)
				{
					for (size_t j = 0; j < matrix2Cols; j++)
					{
						for (size_t i = 0; i < matrix1Rows; i++)
						{
							__m256 _sum = _mm256_setzero_ps();

							for (size_t k = 0; k < matrix1FinalPosCols; k += 8)
							{
								_sum = _mm256_fmadd_ps(_mm256_load_ps(&data1[i * matrix1ActualCols + k]),
									_mm256_load_ps(&data2[j * matrix2ActualRows + k]), _sum);
							}
							__m256 _sum1 = _mm256_hadd_ps(_sum, _sum);
							__m256 _sum2 = _mm256_hadd_ps(_sum1, _sum1);


							__m128 lo128 = _mm256_castps256_ps128(_sum2, 0);
							__m128 hi128 = _mm256_extractf128_ps(_sum2, 1);
							__m128 _sum128 = _mm_add_ps(lo128, hi128);

							for (size_t k = matrix1FinalPosCols; k < matrix1Cols; k++)
							{
								_sum128 = _mm_fmadd_ss(_mm_load_ss(&data1[i * matrix1ActualCols + k]),
									_mm_load_ss(&data2[j * matrix2ActualRows + k]), _sum128);
							}
							_mm_store_ss(&dataResult[i * matrix1Cols + j], _sum128);
						}
					}
				}
				else
				{
					for (size_t j = 0; j < matrix2FinalPosCols; j += 8)
					{
						for (size_t i = 0; i < matrix1Rows; i++)
						{
							__m256 _sum = _mm256_setzero_ps();

							for (size_t k = 0; k < matrix1Cols; k++)
							{
								_sum = _mm256_fmadd_ps(_mm256_broadcast_ss(&data1[i * matrix1ActualCols + k]),
									_mm256_load_ps(&data2[k * matrix2ActualCols + j]), _sum);
							}
							_mm256_store_ps(&dataResult[i * matrix2Cols + j], _sum);
						}
					}
					for (size_t j = matrix2FinalPosCols; j < matrix2Cols; j++)
					{
						for (size_t i = 0; i < matrix1Rows; i++)
						{
							float sum = 0.0;
							for (size_t k = 0; k < matrix1Cols; k++)
							{
								sum += data1[i * matrix1ActualCols + k] * data2[k * matrix2ActualCols + j];
							}
							dataResult[i * matrix2Cols + j] = sum;
						}
					}
				}
			}
			return result;
		}
	}

}
