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

		matrix<double> result(matrix1Rows, matrix2Cols);

		double* dataResult = result._data;

		if constexpr (matrix1Transposed)
		{
			if constexpr (matrix2Transposed)
			{
				size_t matrix1ActualRows = matrix1.actualRows;
				size_t matrix2ActualRows = matrix2.actualRows;

				if constexpr (returnTransposed)
				{
					size_t matrix1FinalPosRows = matrix1.finalPosRows;
					size_t matrix1FinalPosCols = matrix1.finalPosCols;

					for (size_t i = 0; i < matrix1FinalPosRows; i += 4)
					{
						for (size_t j = 0; j < matrix2Cols; j++)
						{
							__m256d _sum = _mm256_setzero_pd();

							for (size_t k = 0; k < matrix1Cols; k++)
							{
								_sum = _mm256_fmadd_pd(_mm256_load_pd(&data1[k * matrix1ActualRows + i]),
									_mm256_set1_pd(data2[j * matrix2ActualRows + k]), _sum);
							}
							_mm256_store_pd(&dataResult[j * matrix1Rows + i], _sum);
						}
					}
					for (size_t i = matrix1FinalPosRows; i < matrix1Rows; i++)
					{
						for (size_t j = 0; j < matrix2Cols; j++)
						{
							__m256d _sum = _mm256_setzero_pd();

							for (size_t k = 0; k < matrix1FinalPosCols; k += 4)
							{
								_sum = _mm256_fmadd_pd(_mm256_setr_pd(data1[k * matrix1ActualRows + i],
									data1[(k + 1) * matrix1ActualRows + i],
									data1[(k + 2) * matrix1ActualRows + i],
									data1[(k + 3) * matrix1ActualRows + i]),
									_mm256_load_pd(&data2[j * matrix2ActualRows + k]),
									_sum);
							}

							__m128d vlow = _mm256_castpd256_pd128(_sum);
							__m128d vhigh = _mm256_extractf128_pd(_sum, 1);
							vlow = _mm_add_pd(vlow, vhigh);

							__m128d high64 = _mm_unpackhi_pd(vlow, vlow);
							double sum = _mm_cvtsd_f64(_mm_add_sd(vlow, high64));

							for (int k = matrix1FinalPosCols; k < matrix1Cols; k++)
							{
								sum += data1[k * matrix1ActualRows + i] * data2[j * matrix2ActualRows + k];
							}

							dataResult[j * matrix1Rows + i] = sum;
						}
					}
				}
				else
				{
					size_t matrix2FinalPosCols = matrix2.finalPosCols;
					size_t matrix1FinalPosCols = matrix1.finalPosCols;

					for (size_t j = 0; j < matrix2FinalPosCols; j += 4)
					{
						for (size_t i = 0; i < matrix1Rows; i++)
						{
							__m256d _sum = _mm256_setzero_pd();

							for (size_t k = 0; k < matrix1Cols; k++)
							{
								_sum = _mm256_fmadd_pd(_mm256_set1_pd(data1[k * matrix1ActualRows + i]),
									_mm256_setr_pd(data2[j * matrix2ActualRows + k],
										data2[(j + 1) * matrix2ActualRows + k],
										data2[(j + 2) * matrix2ActualRows + k],
										data2[(j + 3) * matrix2ActualRows + k]),
									_sum);
							}
							_mm256_store_pd(&dataResult[i * matrix2Cols + j], _sum);
						}
					}
					for (size_t j = matrix2FinalPosCols; j < matrix2Cols; j++)
					{
						for (size_t i = 0; i < matrix1Rows; i++)
						{
							__m256d _sum = _mm256_setzero_pd();

							for (size_t k = 0; k < matrix1FinalPosCols; k += 4)
							{
								_sum = _mm256_fmadd_pd(_mm256_setr_pd(data1[k * matrix1ActualRows + i],
									data1[(k + 1) * matrix1ActualRows + i],
									data1[(k + 2) * matrix1ActualRows + i],
									data1[(k + 3) * matrix1ActualRows + i]),
									_mm256_load_pd(&data2[j * matrix2ActualRows + k]),
									_sum);
							}
							__m128d vlow = _mm256_castpd256_pd128(_sum);
							__m128d vhigh = _mm256_extractf128_pd(_sum, 1);
							vlow = _mm_add_pd(vlow, vhigh);

							__m128d high64 = _mm_unpackhi_pd(vlow, vlow);
							double sum = _mm_cvtsd_f64(_mm_add_sd(vlow, high64));

							for (size_t k = matrix1FinalPosCols; k < matrix1Cols; k++)
							{
								sum += data1[k * matrix1ActualRows + i] * data2[j * matrix2ActualRows + k];
							}
							dataResult[i * matrix2Cols + j] = sum;
						}
					}
				}
			}
			else
			{
				size_t matrix1ActualRows = matrix1.actualRows;
				size_t matrix2ActualCols = matrix2.actualCols;

				if constexpr (returnTransposed)
				{
					size_t matrix1FinalPosRows = matrix1.finalPosRows;
					size_t matrix1FinalPosCols = matrix1.finalPosCols;

					for (size_t i = 0; i < matrix1FinalPosRows; i += 4)
					{
						for (size_t j = 0; j < matrix2Cols; j++)
						{
							__m256d _sum = _mm256_setzero_pd();

							for (size_t k = 0; k < matrix1Cols; k++)
							{
								_sum = _mm256_fmadd_pd(_mm256_load_pd(&data1[k * matrix1ActualRows + i]),
									_mm256_set1_pd(data2[k * matrix2ActualCols + j]), _sum);
							}
							_mm256_store_pd(&dataResult[j * matrix1Rows + i], _sum);
						}
					}
					for (size_t i = matrix1FinalPosRows; i < matrix1Rows; i++)
					{
						for (size_t j = 0; j < matrix2Cols; j++)
						{
							__m256d _sum = _mm256_setzero_pd();

							for (size_t k = 0; k < matrix1FinalPosCols; k += 4)
							{
								_sum = _mm256_fmadd_pd(_mm256_setr_pd(data1[k * matrix1ActualRows + i],
									data1[(k + 1) * matrix1ActualRows + i],
									data1[(k + 2) * matrix1ActualRows + i],
									data1[(k + 3) * matrix1ActualRows + i]),
									_mm256_setr_pd(data2[k * matrix2ActualCols + j],
										data2[(k + 1) * matrix2ActualCols + j],
										data2[(k + 2) * matrix2ActualCols + j],
										data2[(k + 3) * matrix2ActualCols + j]),
									_sum);
							}

							__m128d vlow = _mm256_castpd256_pd128(_sum);
							__m128d vhigh = _mm256_extractf128_pd(_sum, 1);
							vlow = _mm_add_pd(vlow, vhigh);

							__m128d high64 = _mm_unpackhi_pd(vlow, vlow);
							double sum = _mm_cvtsd_f64(_mm_add_sd(vlow, high64));

							for (int k = matrix1FinalPosCols; k < matrix1Cols; k++)
							{
								sum += data1[k * matrix1ActualRows + i] * data2[k * matrix2ActualCols + j];
							}

							dataResult[j * matrix1Rows + i] = sum;
						}
					}
				}
				else
				{
					size_t matrix2FinalPosCols = matrix2.finalPosCols;
					size_t matrix1FinalPosCols = matrix1.finalPosCols;

					for (size_t j = 0; j < matrix2FinalPosCols; j += 4)
					{
						for (size_t i = 0; i < matrix1Rows; i++)
						{
							__m256d _sum = _mm256_setzero_pd();

							for (size_t k = 0; k < matrix1Cols; k++)
							{
								_sum = _mm256_fmadd_pd(_mm256_set1_pd(data1[k * matrix1ActualRows + i]),
									_mm256_load_pd(&data2[k * matrix2ActualCols + j]),
									_sum);
							}
							_mm256_store_pd(&dataResult[i * matrix2Cols + j], _sum);
						}
					}
					for (size_t j = matrix2FinalPosCols; j < matrix2Cols; j++)
					{
						for (size_t i = 0; i < matrix1Rows; i++)
						{
							__m256d _sum = _mm256_setzero_pd();
							for (size_t k = 0; k < matrix1FinalPosCols; k += 4)
							{
								_sum = _mm256_fmadd_pd(_mm256_setr_pd(data1[k * matrix1ActualRows + i],
									data1[(k + 1) * matrix1ActualRows + i],
									data1[(k + 2) * matrix1ActualRows + i],
									data1[(k + 3) * matrix1ActualRows + i]),
									_mm256_setr_pd(data2[k * matrix2ActualCols + j],
										data2[(k + 1) * matrix2ActualCols + j],
										data2[(k + 2) * matrix2ActualCols + j],
										data2[(k + 3) * matrix2ActualCols + j]),
									_sum);
							}
							__m128d vlow = _mm256_castpd256_pd128(_sum);
							__m128d vhigh = _mm256_extractf128_pd(_sum, 1);
							vlow = _mm_add_pd(vlow, vhigh);

							__m128d high64 = _mm_unpackhi_pd(vlow, vlow);
							double sum = _mm_cvtsd_f64(_mm_add_sd(vlow, high64));

							for (size_t k = matrix1FinalPosCols; k < matrix1Cols; k++)
							{
								sum += data1[k * matrix1ActualRows + i] * data2[k * matrix2ActualCols + j];
							}

							dataResult[i * matrix2Cols + j] = sum;
						}
					}
				}
			}
		}
		else
		{
			if constexpr (matrix2Transposed)
			{
				size_t matrix1ActualCols = matrix1.actualCols;
				size_t matrix2ActualRows = matrix2.actualRows;

				if constexpr (returnTransposed)
				{
					size_t matrix1FinalPosRows = matrix1.finalPosRows;
					size_t matrix1FinalPosCols = matrix1.finalPosCols;

					for (size_t i = 0; i < matrix1FinalPosRows; i += 4)
					{
						for (size_t j = 0; j < matrix2Cols; j++)
						{
							__m256d _sum = _mm256_setzero_pd();

							for (size_t k = 0; k < matrix1Cols; k++)
							{
								_sum = _mm256_fmadd_pd(_mm256_setr_pd(data1[i * matrix1ActualCols + k],
									data1[(i + 1) * matrix1ActualCols + k],
									data1[(i + 2) * matrix1ActualCols + k],
									data1[(i + 3) * matrix1ActualCols + k]),
									_mm256_set1_pd(data2[j * matrix2ActualRows + k]), _sum);
							}
							_mm256_store_pd(&dataResult[j * matrix1Rows + i], _sum);
						}
					}
					for (size_t i = matrix1FinalPosRows; i < matrix1Rows; i++)
					{
						for (size_t j = 0; j < matrix2Cols; j++)
						{
							__m256d _sum = _mm256_setzero_pd();

							for (size_t k = 0; k < matrix1ActualCols; k += 4)
							{
								_sum = _mm256_fmadd_pd(_mm256_load_pd(&data1[i * matrix1ActualCols + k]),
									_mm256_load_pd(&data2[j * matrix2ActualRows + k]), _sum);
							}
							__m128d vlow = _mm256_castpd256_pd128(_sum);
							__m128d vhigh = _mm256_extractf128_pd(_sum, 1);
							vlow = _mm_add_pd(vlow, vhigh);

							__m128d high64 = _mm_unpackhi_pd(vlow, vlow);
							double sum = _mm_cvtsd_f64(_mm_add_sd(vlow, high64));

							for (size_t k = matrix1ActualCols; k < matrix1Cols; k++)
							{
								sum += data1[i * matrix1ActualCols + k] * data2[j * matrix2ActualRows + k];
							}
							dataResult[j * matrix1Rows + i] = sum;
						}
					}
				}
				else
				{
					size_t matrix2FinalPosCols = matrix2.finalPosCols;
					size_t matrix1FinalPosCols = matrix1.finalPosCols;

					for (size_t j = 0; j < matrix2FinalPosCols; j += 4)
					{
						for (size_t i = 0; i < matrix1Rows; i++)
						{
							__m256d _sum = _mm256_setzero_pd();

							for (size_t k = 0; k < matrix1Cols; k++)
							{
								_sum = _mm256_fmadd_pd(_mm256_set1_pd(data1[i * matrix1ActualCols + k]),
									_mm256_setr_pd(data2[j * matrix2ActualRows + k],
										data2[(j + 1) * matrix2ActualRows + k],
										data2[(j + 2) * matrix2ActualRows + k],
										data2[(j + 3) * matrix2ActualRows + k]), _sum);
							}
							_mm256_store_pd(&dataResult[i * matrix1Cols + j], _sum);
						}
					}
					for (size_t j = matrix2FinalPosCols; j < matrix2Cols; j++)
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
							double sum = _mm_cvtsd_f64(_mm_add_sd(vlow, high64));

							for (size_t k = matrix1FinalPosCols; k < matrix1Cols; k++)
							{
								sum += data1[i * matrix1ActualCols + k] * data2[j * matrix2ActualRows + k];
							}
							dataResult[i * matrix1Cols + j] = sum;
						}
					}
				}
			}
			else
			{
				size_t matrix1ActualCols = matrix1.actualCols;
				size_t matrix2ActualCols = matrix2.actualCols;

				if constexpr (returnTransposed)
				{
					size_t matrix1FinalPosRows = matrix1.finalPosRows;
					size_t matrix1FinalPosCols = matrix1.finalPosCols;

					for (size_t i = 0; i < matrix1FinalPosRows; i += 4)
					{
						for (size_t j = 0; j < matrix2Cols; j++)
						{
							__m256d _sum = _mm256_setzero_pd();

							for (size_t k = 0; k < matrix1Cols; k++)
							{
								_sum = _mm256_fmadd_pd(_mm256_setr_pd(data1[i * matrix1ActualCols + k],
									data1[(i + 1) * matrix1ActualCols + k],
									data1[(i + 2) * matrix1ActualCols + k],
									data1[(i + 3) * matrix1ActualCols + k]),
									_mm256_set1_pd(data2[k * matrix2ActualCols + j]), _sum);
							}
							_mm256_store_pd(&dataResult[j * matrix1Rows + i], _sum);
						}
					}
					for (size_t i = matrix1FinalPosRows; i < matrix1Rows; i++)
					{
						for (size_t j = 0; j < matrix2Cols; j++)
						{
							__m256d _sum = _mm256_setzero_pd();

							for (size_t k = 0; k < matrix1FinalPosCols; k += 4)
							{
								_sum = _mm256_fmadd_pd(_mm256_load_pd(&data1[i * matrix1ActualCols + k]),
									_mm256_setr_pd(data2[k * matrix2ActualCols + j],
										data2[(k + 1) * matrix2ActualCols + j],
										data2[(k + 2) * matrix2ActualCols + j],
										data2[(k + 3) * matrix2ActualCols + j]),
									_sum);
							}
							__m128d vlow = _mm256_castpd256_pd128(_sum);
							__m128d vhigh = _mm256_extractf128_pd(_sum, 1);
							vlow = _mm_add_pd(vlow, vhigh);

							__m128d high64 = _mm_unpackhi_pd(vlow, vlow);
							double sum = _mm_cvtsd_f64(_mm_add_sd(vlow, high64));

							for (size_t k = matrix1FinalPosCols; k < matrix1Cols; k++)
							{
								sum += data1[i * matrix1ActualCols + k] * data2[k * matrix2ActualCols + j];
							}
							dataResult[j * matrix1Rows + i] = sum;
						}
					}
				}
				else
				{
					size_t matrix2FinalPosCols = matrix2.finalPosCols;
					size_t matrix1FinalPosCols = matrix1.finalPosCols;

					for (size_t j = 0; j < matrix2FinalPosCols; j += 4)
					{
						for (size_t i = 0; i < matrix1Rows; i++)
						{
							__m256d _sum = _mm256_setzero_pd();


							for (size_t k = 0; k < matrix1Cols; k++)
							{
								_sum = _mm256_fmadd_pd(_mm256_set1_pd(data1[i * matrix1ActualCols + k]),
									_mm256_load_pd(&data2[k * matrix2ActualCols + j]), _sum);
							}
							_mm256_store_pd(&dataResult[i * matrix2Cols + j], _sum);
						}
					}
					for (size_t j = matrix2FinalPosCols; j < matrix2Cols; j++)
					{
						for (size_t i = 0; i < matrix1Rows; i++)
						{
							__m256d _sum = _mm256_setzero_pd();

							for (size_t k = 0; k < matrix1FinalPosCols; k += 4)
							{
								_sum = _mm256_fmadd_pd(_mm256_load_pd(&data1[i * matrix1ActualCols + k]),
									_mm256_setr_pd(data2[k * matrix2ActualCols + j],
										data2[(k + 1) * matrix2ActualCols + j],
										data2[(k + 2) * matrix2ActualCols + j],
										data2[(k + 3) * matrix2ActualCols + j]), _sum);
							}

							__m128d vlow = _mm256_castpd256_pd128(_sum);
							__m128d vhigh = _mm256_extractf128_pd(_sum, 1);
							vlow = _mm_add_pd(vlow, vhigh);

							__m128d high64 = _mm_unpackhi_pd(vlow, vlow);
							double sum = _mm_cvtsd_f64(_mm_add_sd(vlow, high64));

							for (size_t k = matrix1FinalPosCols; k < matrix1Cols; k++)
							{
								sum += data1[i * matrix1ActualCols + k] * data2[k * matrix2ActualCols + j];
							}

							dataResult[i * matrix2Cols + j] = sum;
						}
					}
				}
			}
		}
		return result;
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

		matrix<float> result(matrix1Rows, matrix2Cols);

		float* dataResult = result._data;

		if constexpr (matrix1Transposed)
		{
			if constexpr (matrix2Transposed)
			{
				size_t matrix1ActualRows = matrix1.actualRows;
				size_t matrix2ActualRows = matrix2.actualRows;

				if constexpr (returnTransposed)
				{
					size_t matrix1FinalPosRows = matrix1.finalPosRows;
					size_t matrix1FinalPosCols = matrix1.finalPosCols;

					for (size_t i = 0; i < matrix1FinalPosRows; i += 8)
					{
						for (size_t j = 0; j < matrix2Cols; j++)
						{
							__m256 _sum = _mm256_setzero_ps();

							for (size_t k = 0; k < matrix1Cols; k++)
							{
								_sum = _mm256_fmadd_ps(_mm256_load_ps(&data1[k * matrix1ActualRows + i]),
									_mm256_set1_ps(data2[j * matrix2ActualRows + k]), _sum);
							}
							_mm256_store_ps(&dataResult[j * matrix1Rows + i], _sum);
						}
					}
					for (size_t i = matrix1FinalPosRows; i < matrix1Rows; i++)
					{
						for (size_t j = 0; j < matrix2Cols; j++)
						{
							__m256 _sum = _mm256_setzero_ps();

							for (size_t k = 0; k < matrix1FinalPosCols; k += 8)
							{
								_sum = _mm256_fmadd_ps(_mm256_setr_ps(data1[k * matrix1ActualRows + i],
									data1[(k + 1) * matrix1ActualRows + i],
									data1[(k + 2) * matrix1ActualRows + i],
									data1[(k + 3) * matrix1ActualRows + i],
									data1[(k + 4) * matrix1ActualRows + i],
									data1[(k + 5) * matrix1ActualRows + i],
									data1[(k + 6) * matrix1ActualRows + i],
									data1[(k + 7) * matrix1ActualRows + i]),
									_mm256_load_ps(&data2[j * matrix2ActualRows + k]),
									_sum);
							}

							__m256 _sum1 = _mm256_hadd_ps(_sum, _sum);
							__m256 _sum2 = _mm256_hadd_ps(_sum1, _sum1);

							__m128 lo128 = _mm256_castps256_ps128(_sum2);
							__m128 hi128 = _mm256_extractf128_ps(_sum2, 1);
							__m128 result128 = _mm_add_ps(lo128, hi128);
							float sum = _mm_cvtss_f32(result128);

							for (int k = matrix1FinalPosCols; k < matrix1Cols; k++)
							{
								sum += data1[k * matrix1ActualRows + i] * data2[j * matrix2ActualRows + k];
							}

							dataResult[j * matrix1Rows + i] = sum;
						}
					}
				}
				else
				{
					size_t matrix2FinalPosCols = matrix2.finalPosCols;
					size_t matrix1FinalPosCols = matrix1.finalPosCols;

					for (size_t j = 0; j < matrix2FinalPosCols; j += 8)
					{
						for (size_t i = 0; i < matrix1Rows; i++)
						{
							__m256 _sum = _mm256_setzero_ps();

							for (size_t k = 0; k < matrix1Cols; k++)
							{
								_sum = _mm256_fmadd_ps(_mm256_set1_ps(data1[k * matrix1ActualRows + i]),
									_mm256_setr_ps(data2[j * matrix2ActualRows + k],
										data2[(j + 1) * matrix2ActualRows + k],
										data2[(j + 2) * matrix2ActualRows + k],
										data2[(j + 3) * matrix2ActualRows + k],
										data2[(j + 4) * matrix2ActualRows + k],
										data2[(j + 5) * matrix2ActualRows + k],
										data2[(j + 6) * matrix2ActualRows + k],
										data2[(j + 7) * matrix2ActualRows + k]),
									_sum);
							}
							_mm256_store_ps(&dataResult[i * matrix2Cols + j], _sum);
						}
					}
					for (size_t j = matrix2FinalPosCols; j < matrix2Cols; j++)
					{
						for (size_t i = 0; i < matrix1Rows; i++)
						{
							__m256 _sum = _mm256_setzero_ps();

							for (size_t k = 0; k < matrix1FinalPosCols; k += 8)
							{
								_sum = _mm256_fmadd_ps(_mm256_setr_ps(data1[k * matrix1ActualRows + i],
									data1[(k + 1) * matrix1ActualRows + i],
									data1[(k + 2) * matrix1ActualRows + i],
									data1[(k + 3) * matrix1ActualRows + i],
									data1[(k + 4) * matrix1ActualRows + i],
									data1[(k + 5) * matrix1ActualRows + i],
									data1[(k + 6) * matrix1ActualRows + i],
									data1[(k + 7) * matrix1ActualRows + i]),
									_mm256_load_ps(&data2[j * matrix2ActualRows + k]),
									_sum);
							}
							__m256 _sum1 = _mm256_hadd_ps(_sum, _sum);
							__m256 _sum2 = _mm256_hadd_ps(_sum1, _sum1);

							__m128 lo128 = _mm256_castps256_ps128(_sum2);
							__m128 hi128 = _mm256_extractf128_ps(_sum2, 1);
							__m128 result128 = _mm_add_ps(lo128, hi128);
							float sum = _mm_cvtss_f32(result128);

							for (size_t k = matrix1FinalPosCols; k < matrix1Cols; k++)
							{
								sum += data1[k * matrix1ActualRows + i] * data2[j * matrix2ActualRows + k];
							}
							dataResult[i * matrix2Cols + j] = sum;
						}
					}
				}
			}
			else
			{
				size_t matrix1ActualRows = matrix1.actualRows;
				size_t matrix2ActualCols = matrix2.actualCols;

				if constexpr (returnTransposed)
				{
					size_t matrix1FinalPosRows = matrix1.finalPosRows;

					for (size_t i = 0; i < matrix1FinalPosRows; i += 8)
					{
						for (size_t j = 0; j < matrix2Cols; j++)
						{
							__m256 _sum = _mm256_setzero_ps();

							for (size_t k = 0; k < matrix1Cols; k++)
							{
								_sum = _mm256_fmadd_ps(_mm256_load_ps(&data1[k * matrix1ActualRows + i]),
									_mm256_set1_ps(data2[k * matrix2ActualCols + j]), _sum);
							}
							_mm256_store_ps(&dataResult[j * matrix1Rows + i], _sum);
						}
					}
					for (size_t i = matrix1FinalPosRows; i < matrix1Rows; i++)
					{
						for (size_t j = 0; j < matrix2Cols; j++)
						{
							float sum = 0.0f;
							for (int k = 0; k < matrix1Cols; k++)
							{
								sum += data1[k * matrix1ActualRows + i] * data2[k * matrix2ActualCols + j];
							}

							dataResult[j * matrix1Rows + i] = sum;
						}
					}
				}
				else
				{
					size_t matrix2FinalPosCols = matrix2.finalPosCols;

					for (size_t j = 0; j < matrix2FinalPosCols; j += 8)
					{
						for (size_t i = 0; i < matrix1Rows; i++)
						{
							__m256 _sum = _mm256_setzero_ps();

							for (size_t k = 0; k < matrix1Cols; k++)
							{
								_sum = _mm256_fmadd_ps(_mm256_set1_ps(data1[k * matrix1ActualRows + i]),
									_mm256_load_ps(&data2[k * matrix2ActualCols + j]),
									_sum);
							}
							_mm256_store_ps(&dataResult[i * matrix2Cols + j], _sum);
						}
					}
					for (size_t j = matrix2FinalPosCols; j < matrix2Cols; j++)
					{
						for (size_t i = 0; i < matrix1Rows; i++)
						{
							float sum = 0.0f;

							for (size_t k = 0; k < matrix1Cols; k++)
							{
								sum += data1[k * matrix1ActualRows + i] * data2[k * matrix2ActualCols + j];
							}

							dataResult[i * matrix2Cols + j] = sum;
						}
					}
				}
			}
		}
		else
		{
			if constexpr (matrix2Transposed)
			{
				size_t matrix1ActualCols = matrix1.actualCols;
				size_t matrix2ActualRows = matrix2.actualRows;

				if constexpr (returnTransposed)
				{
					size_t matrix1FinalPosRows = matrix1.finalPosRows;
					size_t matrix1FinalPosCols = matrix1.finalPosCols;

					for (size_t i = 0; i < matrix1FinalPosRows; i += 8)
					{
						for (size_t j = 0; j < matrix2Cols; j++)
						{
							__m256 _sum = _mm256_setzero_ps();

							for (size_t k = 0; k < matrix1Cols; k++)
							{
								_sum = _mm256_fmadd_ps(_mm256_setr_ps(data1[i * matrix1ActualCols + k],
									data1[(i + 1) * matrix1ActualCols + k],
									data1[(i + 2) * matrix1ActualCols + k],
									data1[(i + 3) * matrix1ActualCols + k],
									data1[(i + 4) * matrix1ActualCols + k],
									data1[(i + 5) * matrix1ActualCols + k],
									data1[(i + 6) * matrix1ActualCols + k],
									data1[(i + 7) * matrix1ActualCols + k]),
									_mm256_set1_ps(data2[j * matrix2ActualRows + k]), _sum);
							}
							_mm256_store_ps(&dataResult[j * matrix1Rows + i], _sum);
						}
					}
					for (size_t i = matrix1FinalPosRows; i < matrix1Rows; i++)
					{
						for (size_t j = 0; j < matrix2Cols; j++)
						{
							__m256 _sum = _mm256_setzero_ps();

							for (size_t k = 0; k < matrix1ActualCols; k += 8)
							{
								_sum = _mm256_fmadd_ps(_mm256_load_ps(&data1[i * matrix1ActualCols + k]),
									_mm256_load_ps(&data2[j * matrix2ActualRows + k]), _sum);
							}
							__m256 _sum1 = _mm256_hadd_ps(_sum, _sum);
							__m256 _sum2 = _mm256_hadd_ps(_sum1, _sum1);

							__m128 lo128 = _mm256_castps256_ps128(_sum2);
							__m128 hi128 = _mm256_extractf128_ps(_sum2, 1);
							__m128 result128 = _mm_add_ps(lo128, hi128);
							float sum = _mm_cvtss_f32(result128);

							for (size_t k = matrix1ActualCols; k < matrix1Cols; k++)
							{
								sum += data1[i * matrix1ActualCols + k] * data2[j * matrix2ActualRows + k];
							}
							dataResult[j * matrix1Rows + i] = sum;
						}
					}
				}
				else
				{
					size_t matrix2FinalPosCols = matrix2.finalPosCols;
					size_t matrix1FinalPosCols = matrix1.finalPosCols;

					for (size_t j = 0; j < matrix2FinalPosCols; j += 8)
					{
						for (size_t i = 0; i < matrix1Rows; i++)
						{
							__m256 _sum = _mm256_setzero_ps();

							for (size_t k = 0; k < matrix1Cols; k++)
							{
								_sum = _mm256_fmadd_ps(_mm256_set1_ps(data1[i * matrix1ActualCols + k]),
									_mm256_setr_ps(data2[j * matrix2ActualRows + k],
										data2[(j + 1) * matrix2ActualRows + k],
										data2[(j + 2) * matrix2ActualRows + k],
										data2[(j + 3) * matrix2ActualRows + k],
										data2[(j + 4) * matrix2ActualRows + k],
										data2[(j + 5) * matrix2ActualRows + k],
										data2[(j + 6) * matrix2ActualRows + k],
										data2[(j + 7) * matrix2ActualRows + k]), _sum);
							}
							_mm256_store_ps(&dataResult[i * matrix1Cols + j], _sum);
						}
					}
					for (size_t j = matrix2FinalPosCols; j < matrix2Cols; j++)
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

							__m128 lo128 = _mm256_castps256_ps128(_sum2);
							__m128 hi128 = _mm256_extractf128_ps(_sum2, 1);
							__m128 result128 = _mm_add_ps(lo128, hi128);
							float sum = _mm_cvtss_f32(result128);

							for (size_t k = matrix1FinalPosCols; k < matrix1Cols; k++)
							{
								sum += data1[i * matrix1ActualCols + k] * data2[j * matrix2ActualRows + k];
							}
							dataResult[i * matrix1Cols + j] = sum;
						}
					}
				}
			}
			else
			{
				size_t matrix1ActualCols = matrix1.actualCols;
				size_t matrix2ActualCols = matrix2.actualCols;

				if constexpr (returnTransposed)
				{
					size_t matrix1FinalPosRows = matrix1.finalPosRows;
					size_t matrix1FinalPosCols = matrix1.finalPosCols;

					for (size_t i = 0; i < matrix1FinalPosRows; i += 8)
					{
						for (size_t j = 0; j < matrix2Cols; j++)
						{
							__m256 _sum = _mm256_setzero_ps();

							for (size_t k = 0; k < matrix1Cols; k++)
							{
								_sum = _mm256_fmadd_ps(_mm256_setr_ps(data1[i * matrix1ActualCols + k],
									data1[(i + 1) * matrix1ActualCols + k],
									data1[(i + 2) * matrix1ActualCols + k],
									data1[(i + 3) * matrix1ActualCols + k],
									data1[(i + 4) * matrix1ActualCols + k],
									data1[(i + 5) * matrix1ActualCols + k],
									data1[(i + 6) * matrix1ActualCols + k],
									data1[(i + 7) * matrix1ActualCols + k]),
									_mm256_set1_ps(data2[k * matrix2ActualCols + j]), _sum);
							}
							_mm256_store_ps(&dataResult[j * matrix1Rows + i], _sum);
						}
					}
					for (size_t i = matrix1FinalPosRows; i < matrix1Rows; i++)
					{
						for (size_t j = 0; j < matrix2Cols; j++)
						{
							__m256 _sum = _mm256_setzero_ps();

							for (size_t k = 0; k < matrix1FinalPosCols; k += 8)
							{
								_sum = _mm256_fmadd_ps(_mm256_load_ps(&data1[i * matrix1ActualCols + k]),
									_mm256_setr_ps(data2[k * matrix2ActualCols + j],
										data2[(k + 1) * matrix2ActualCols + j],
										data2[(k + 2) * matrix2ActualCols + j],
										data2[(k + 3) * matrix2ActualCols + j],
										data2[(k + 4) * matrix2ActualCols + j],
										data2[(k + 5) * matrix2ActualCols + j],
										data2[(k + 6) * matrix2ActualCols + j],
										data2[(k + 7) * matrix2ActualCols + j]),
									_sum);
							}
							__m256 _sum1 = _mm256_hadd_ps(_sum, _sum);
							__m256 _sum2 = _mm256_hadd_ps(_sum1, _sum1);

							__m128 lo128 = _mm256_castps256_ps128(_sum2);
							__m128 hi128 = _mm256_extractf128_ps(_sum2, 1);
							__m128 result128 = _mm_add_ps(lo128, hi128);
							float sum = _mm_cvtss_f32(result128);

							for (size_t k = matrix1FinalPosCols; k < matrix1Cols; k++)
							{
								sum += data1[i * matrix1ActualCols + k] * data2[k * matrix2ActualCols + j];
							}
							dataResult[j * matrix1Rows + i] = sum;
						}
					}
				}
				else
				{
					size_t matrix2FinalPosCols = matrix2.finalPosCols;
					size_t matrix1FinalPosCols = matrix1.finalPosCols;

					for (size_t j = 0; j < matrix2FinalPosCols; j += 8)
					{
						for (size_t i = 0; i < matrix1Rows; i++)
						{
							__m256 _sum = _mm256_setzero_ps();


							for (size_t k = 0; k < matrix1Cols; k++)
							{
								_sum = _mm256_fmadd_ps(_mm256_set1_ps(data1[i * matrix1ActualCols + k]),
									_mm256_load_ps(&data2[k * matrix2ActualCols + j]), _sum);
							}
							_mm256_store_ps(&dataResult[i * matrix2Cols + j], _sum);
						}
					}
					for (size_t j = matrix2FinalPosCols; j < matrix2Cols; j++)
					{
						for (size_t i = 0; i < matrix1Rows; i++)
						{
							__m256 _sum = _mm256_setzero_ps();

							for (size_t k = 0; k < matrix1FinalPosCols; k += 8)
							{
								_sum = _mm256_fmadd_ps(_mm256_load_ps(&data1[i * matrix1ActualCols + k]),
									_mm256_setr_ps(data2[k * matrix2ActualCols + j],
										data2[(k + 1) * matrix2ActualCols + j],
										data2[(k + 2) * matrix2ActualCols + j],
										data2[(k + 3) * matrix2ActualCols + j],
										data2[(j + 4) * matrix2ActualCols + j],
										data2[(k + 5) * matrix2ActualCols + j],
										data2[(k + 6) * matrix2ActualCols + j],
										data2[(k + 7) * matrix2ActualCols + j]), _sum);
							}

							__m256 _sum1 = _mm256_hadd_ps(_sum, _sum);
							__m256 _sum2 = _mm256_hadd_ps(_sum1, _sum1);

							__m128 lo128 = _mm256_castps256_ps128(_sum2);
							__m128 hi128 = _mm256_extractf128_ps(_sum2, 1);
							__m128 result128 = _mm_add_ps(lo128, hi128);
							float sum = _mm_cvtss_f32(result128);

							for (size_t k = matrix1FinalPosCols; k < matrix1Cols; k++)
							{
								sum += data1[i * matrix1ActualCols + k] * data2[k * matrix2ActualCols + j];
							}

							dataResult[i * matrix2Cols + j] = sum;
						}
					}
				}
			}
		}
		return result;
	}

}