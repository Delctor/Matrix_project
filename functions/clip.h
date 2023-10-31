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
	template<bool returnTransposed = false, bool thisTransposed, bool thisContiguous>
	inline matrix<double> clip(matrix<double, thisTransposed, thisContiguous>& matrix1, double min, double max)

	{
		size_t rows = matrix1._rows;
		size_t cols = matrix1._cols;

		double* data1 = matrix1._data;

		matrix<double> result(rows, cols);

		double* dataResult = result._data;

		__m256d _min = _mm256_set1_pd(min);
		__m256d _max = _mm256_set1_pd(max);

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

						_mm256_store_pd(&dataResult[i], _mm256_min_pd(_mm256_max_pd(a, _min), _max));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = std::min(std::max(data1[i], min), max);
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

							_mm256_store_pd(&dataResult[j * rows + i], _mm256_min_pd(_mm256_max_pd(a, _min), _max));
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

							__m256d clip = _mm256_min_pd(_mm256_max_pd(a, _min), _max);

							__m128d val1 = _mm256_extractf128_pd(clip, 1);
							__m128d val2 = _mm256_castpd256_pd128(clip);

							_mm_store_sd(&dataResult[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

							_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
						}
						for (size_t j = finalPosCols; j < cols; j++)
						{
							dataResult[j * rows + i] = std::min(std::max(data1[j * matrix1ActualRows + i], min), max);
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

						_mm256_store_pd(&dataResult[i * cols + j], _mm256_min_pd(_mm256_max_pd(a, _min), _max));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						__m256d clip = _mm256_min_pd(_mm256_max_pd(a, _min), _max);

						__m128d val1 = _mm256_extractf128_pd(clip, 1);
						__m128d val2 = _mm256_castpd256_pd128(clip);

						_mm_store_sd(&dataResult[i * cols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

						_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						dataResult[i * cols + j] = std::min(std::max(data1[j * matrix1ActualRows + i], min), max);
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

						_mm256_store_pd(&dataResult[j * rows + i], _mm256_min_pd(_mm256_max_pd(a, _min), _max));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						__m256d clip = _mm256_min_pd(_mm256_max_pd(a, _min), _max);

						__m128d val1 = _mm256_extractf128_pd(clip, 1);
						__m128d val2 = _mm256_castpd256_pd128(clip);

						_mm_store_sd(&dataResult[j * rows + i], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

						_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						dataResult[j * rows + i] = std::min(std::max(data1[i * matrix1ActualCols + j], min), max);
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

						_mm256_store_pd(&dataResult[i], _mm256_min_pd(_mm256_max_pd(a, _min), _max));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = std::min(std::max(data1[i], min), max);
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

							_mm256_store_pd(&dataResult[i * cols + j], _mm256_min_pd(_mm256_max_pd(a, _min), _max));
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

							__m256d clip = _mm256_min_pd(_mm256_max_pd(a, _min), _max);

							__m128d val1 = _mm256_extractf128_pd(clip, 1);
							__m128d val2 = _mm256_castpd256_pd128(clip);

							_mm_store_sd(&dataResult[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

							_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
						}
						for (size_t i = finalPosRows; i < rows; i++)
						{
							dataResult[i * cols + j] = std::min(std::max(data1[i * matrix1ActualCols + j], min), max);
						}
					}
				}
			}
		}
		return result;
	}

	template<bool returnTransposed = false, bool thisTransposed, bool thisContiguous>
	inline matrix<float> clip(matrix<float, thisTransposed, thisContiguous>& matrix1, float min, float max)
	{
		size_t rows = matrix1._rows;
		size_t cols = matrix1._cols;

		float* data1 = matrix1._data;

		matrix<float> result(rows, cols);

		float* dataResult = result._data;

		__m256 _min = _mm256_set1_ps(min);
		__m256 _max = _mm256_set1_ps(max);

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

						_mm256_store_ps(&dataResult[i], _mm256_min_ps(_mm256_max_ps(a, _min), _max));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = std::min(std::max(data1[i], min), max);
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

							_mm256_store_ps(&dataResult[j * rows + i], _mm256_min_ps(_mm256_max_ps(a, _min), _max));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = std::min(std::max(data1[j * matrix1ActualRows + i], min), max);
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

						_mm256_store_ps(&dataResult[i * cols + j], _mm256_min_ps(_mm256_max_ps(a, _min), _max));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < rows; i++)
					{
						dataResult[i * cols + j] = std::min(std::max(data1[j * matrix1ActualRows + i], min), max);
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

						_mm256_store_ps(&dataResult[j * rows + i], _mm256_min_ps(_mm256_max_ps(a, _min), _max));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						dataResult[j * rows + i] = std::min(std::max(data1[i * matrix1ActualCols + j], min), max);
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

						_mm256_store_ps(&dataResult[i], _mm256_min_ps(_mm256_max_ps(a, _min), _max));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = std::min(std::max(data1[i], min), max);
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

							_mm256_store_ps(&dataResult[i * cols + j], _mm256_min_ps(_mm256_max_ps(a, _min), _max));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = std::min(std::max(data1[i * matrix1ActualCols + j], min), max);
						}
					}
				}
			}
		}
		return result;
	}

}