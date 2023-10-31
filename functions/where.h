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
	// Double

	inline vector<double> where(vector<uint8_t>& vector1, vector<double>& vector2, vector<double>& vector3)
	{
#ifdef _DEBUG
		if (vector1._size != vector2._size || vector2._size != vector3._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = vector1._size;

		size_t finalPos = vector1.finalPos;

		uint8_t* data1 = vector1._data;
		double* data2 = vector2._data;
		double* data3 = vector3._data;

		vector<double> result(size);

		double* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d mask = _mm256_castsi256_pd(_mm256_cvtepi8_epi64(_mm_castps_si128(_mm_load_ss(reinterpret_cast<float*>(&data1[i])))));

			_mm256_store_pd(&dataResult[i], _mm256_blendv_pd(_mm256_load_pd(&data3[i]), _mm256_load_pd(&data2[i]), mask));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] ? data2[i] : data3[i];
		}
		return result;
	}

	inline vector<double> where(vector<uint8_t>& vector1, double num1, double num2)
	{
		size_t size = vector1._size;

		uint8_t* data1 = vector1._data;

		__m256d _num1 = _mm256_set1_pd(num1);
		__m256d _num2 = _mm256_set1_pd(num2);

		size_t finalPos = (size / 4) * 4;

		vector<double> result(size);

		double* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&dataResult[i], _mm256_blendv_pd(_num2, _num1,
				_mm256_castsi256_pd(_mm256_cvtepi8_epi64(
					_mm_castps_si128(_mm_load_ss(reinterpret_cast<float*>(&data1[i])))))));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] ? num1 : num2;
		}
		return result;
	}

	inline vector<double> where(vector<uint8_t>& vector1, vector<double>& vector2, double num)
	{
#ifdef _DEBUG
		if (vector1._size != vector2._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = vector1._size;

		size_t finalPos = vector2.finalPos;

		uint8_t* data1 = vector1._data;

		double* data2 = vector2._data;

		__m256d b = _mm256_set1_pd(num);

		vector<double> result(size);

		double* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_load_pd(&data2[i]);
			_mm256_store_pd(&dataResult[i], _mm256_blendv_pd(b, a,
				_mm256_castsi256_pd(_mm256_cvtepi8_epi64(
					_mm_castps_si128(_mm_load_ss(reinterpret_cast<float*>(&data1[i])))))));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] ? data2[i] : num;
		}
		return result;
	}

	inline vector<double> where(vector<uint8_t>& vector1, double num, vector<double>& vector2)
	{
#ifdef _DEBUG
		if (vector1._size != vector2._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = vector1._size;

		size_t finalPos = vector2.finalPos;

		uint8_t* data1 = vector1._data;

		double* data2 = vector2._data;

		__m256d b = _mm256_set1_pd(num);

		vector<double> result(size);

		double* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_load_pd(&data2[i]);
			_mm256_store_pd(&dataResult[i], _mm256_blendv_pd(a, b,
				_mm256_castsi256_pd(_mm256_cvtepi8_epi64(
					_mm_castps_si128(_mm_load_ss(reinterpret_cast<float*>(&data1[i])))))));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] ? num : data2[i];
		}
		return result;
	}

	template<bool returnTransposed = false, bool matrx1Transposed, bool matrix1Contiguous>
	inline matrix<double> where(matrix<uint8_t, matrx1Transposed, matrix1Contiguous>& matrix1, double num1, double num2)
	{
		size_t rows = matrix1._rows;
		size_t cols = matrix1._cols;

		__m256d _num1 = _mm256_set1_pd(num1);
		__m256d _num2 = _mm256_set1_pd(num2);

		uint8_t* data1 = matrix1._data;

		matrix<double> result(rows, cols);

		double* dataResult = result._data;

		if constexpr (matrx1Transposed)
		{
			size_t actualRows = matrix1.actualRows;

			if constexpr (returnTransposed)
			{
				if constexpr (matrix1Contiguous)
				{
					size_t size = matrix1._size;
					size_t finalPosSize = matrix1.finalPosSize;
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_castsi256_pd(_mm256_cvtepi8_epi64(_mm_castps_si128(_mm_load_ss(reinterpret_cast<float*>(&data1[i])))));

						_mm256_store_pd(&dataResult[i], _mm256_blendv_pd(_num2, _num1, a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = data1[i] ? num1 : num2;
					}
				}
				else
				{
					size_t finalPosRows = matrix1.finalPosRows;

					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; i++)
						{
							__m256d a = _mm256_castsi256_pd(_mm256_cvtepi8_epi64(_mm_castps_si128(_mm_load_ss(reinterpret_cast<float*>(&data1[j * actualRows + i])))));
							_mm256_store_pd(&dataResult[j * rows + i], _mm256_blendv_pd(_num2, _num1, a));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < cols; i++)
						{
							dataResult[j * rows + i] = data1[j * actualRows + i] ? num1 : num2;
						}
					}
				}
			}
			else
			{
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < cols; i++)
					{
						dataResult[i * cols + j] = data1[j * actualRows + i] ? num1 : num2;
					}
				}
			}
		}
		else
		{
			size_t actualCols = matrix1.actualCols;

			if constexpr (returnTransposed)
			{
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < cols; i++)
					{
						dataResult[j * rows + i] = data1[i * actualCols + j] ? num1 : num2;
					}
				}
			}
			else
			{
				if constexpr (matrix1Contiguous)
				{
					size_t size = matrix1._size;
					size_t finalPosSize = matrix1.finalPosSize;
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_castsi256_pd(_mm256_cvtepi8_epi64(_mm_castps_si128(_mm_load_ss(reinterpret_cast<float*>(&data1[i])))));

						_mm256_store_pd(&dataResult[i], _mm256_blendv_pd(_num2, _num1, a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = data1[i] ? num1 : num2;
					}
				}
				else
				{
					size_t actualCols = matrix1.actualCols;
					size_t finalPosCols = matrix1.finalPosCols;
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_castsi256_pd(_mm256_cvtepi8_epi64(_mm_castps_si128(_mm_load_ss(reinterpret_cast<float*>(&data1[i * actualCols + j])))));
							_mm256_store_pd(&dataResult[i * cols + j], _mm256_blendv_pd(_num2, _num1, a));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[i * actualCols + j] ? num1 : num2;
						}
					}
				}
			}
		}
		return result;
	}

	// Float

	inline vector<float> where(vector<uint8_t>& vector1, vector<float>& vector2, vector<float>& vector3)
	{
#ifdef _DEBUG
		if (vector1._size != vector2._size || vector2._size != vector3._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = vector1._size;

		size_t finalPos = vector1.finalPos;

		uint8_t* data1 = vector1._data;
		float* data2 = vector2._data;
		float* data3 = vector3._data;

		vector<float> result(size);

		float* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256 mask = _mm256_castsi256_ps(_mm256_cvtepi8_epi32(_mm_castpd_si128(_mm_load_sd(reinterpret_cast<double*>(&data1[i])))));

			_mm256_store_ps(&dataResult[i], _mm256_blendv_ps(_mm256_load_ps(&data3[i]), _mm256_load_ps(&data2[i]), mask));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] ? data2[i] : data3[i];
		}
		return result;
	}

	inline vector<float> where(vector<uint8_t>& vector1, float num1, float num2)
	{
		size_t size = vector1._size;

		uint8_t* data1 = vector1._data;

		__m256 _num1 = _mm256_set1_ps(num1);
		__m256 _num2 = _mm256_set1_ps(num2);

		size_t finalPos = (size / 8) * 8;

		vector<float> result(size);

		float* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_store_ps(&dataResult[i], _mm256_blendv_ps(_num2, _num1,
				_mm256_castsi256_ps(_mm256_cvtepi8_epi32(
					_mm_castpd_si128(_mm_load_sd(reinterpret_cast<double*>(&data1[i])))))));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] ? num1 : num2;
		}
		return result;
	}

	inline vector<float> where(vector<uint8_t>& vector1, vector<float>& vector2, float num)
	{
#ifdef _DEBUG
		if (vector1._size != vector2._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = vector1._size;

		size_t finalPos = vector2.finalPos;

		uint8_t* data1 = vector1._data;

		float* data2 = vector2._data;

		__m256 b = _mm256_set1_ps(num);

		vector<float> result(size);

		float* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256 a = _mm256_load_ps(&data2[i]);
			_mm256_store_ps(&dataResult[i], _mm256_blendv_ps(b, a,
				_mm256_castsi256_ps(_mm256_cvtepi8_epi32(
					_mm_castpd_si128(_mm_load_sd(reinterpret_cast<double*>(&data1[i])))))));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] ? data2[i] : num;
		}
		return result;
	}

	inline vector<float> where(vector<uint8_t>& vector1, float num, vector<float>& vector2)
	{
#ifdef _DEBUG
		if (vector1._size != vector2._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = vector1._size;

		size_t finalPos = vector2.finalPos;

		uint8_t* data1 = vector1._data;

		float* data2 = vector2._data;

		__m256 b = _mm256_set1_ps(num);

		vector<float> result(size);

		float* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256 a = _mm256_load_ps(&data2[i]);
			_mm256_store_ps(&dataResult[i], _mm256_blendv_ps(a, b,
				_mm256_castsi256_ps(_mm256_cvtepi8_epi32(
					_mm_castpd_si128(_mm_load_sd(reinterpret_cast<double*>(&data1[i])))))));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] ? num : data2[i];
		}
		return result;
	}

	template<bool returnTransposed = false, bool matrx1Transposed, bool matrix1Contiguous>
	inline matrix<float> where(matrix<uint8_t, matrx1Transposed, matrix1Contiguous>& matrix1, float num1, float num2)
	{
		size_t rows = matrix1._rows;
		size_t cols = matrix1._cols;

		__m256 _num1 = _mm256_set1_ps(num1);
		__m256 _num2 = _mm256_set1_ps(num2);

		uint8_t* data1 = matrix1._data;

		matrix<float> result(rows, cols);

		float* dataResult = result._data;

		if constexpr (matrx1Transposed)
		{
			size_t actualRows = matrix1.actualRows;

			if constexpr (returnTransposed)
			{
				if constexpr (matrix1Contiguous)
				{
					size_t size = matrix1._size;
					size_t finalPosSize = matrix1.finalPosSize;
					for (size_t i = 0; i < finalPosSize; i += 8)
					{
						__m256 a = _mm256_castsi256_ps(_mm256_cvtepi8_epi32(_mm_castpd_si128(_mm_load_sd(reinterpret_cast<double*>(&data1[i])))));

						_mm256_store_ps(&dataResult[i], _mm256_blendv_ps(_num2, _num1, a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = data1[i] ? num1 : num2;
					}
				}
				else
				{
					size_t finalPosRows = matrix1.finalPosRows;

					for (size_t i = 0; i < finalPosRows; i += 8)
					{
						for (size_t j = 0; j < cols; i++)
						{
							__m256 a = _mm256_castsi256_ps(_mm256_cvtepi8_epi32(_mm_castpd_si128(_mm_load_sd(reinterpret_cast<double*>(&data1[j * actualRows + i])))));
							_mm256_store_ps(&dataResult[j * rows + i], _mm256_blendv_ps(_num2, _num1, a));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < cols; i++)
						{
							dataResult[j * rows + i] = data1[j * actualRows + i] ? num1 : num2;
						}
					}
				}
			}
			else
			{
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < cols; i++)
					{
						dataResult[i * cols + j] = data1[j * actualRows + i] ? num1 : num2;
					}
				}
			}
		}
		else
		{
			size_t actualCols = matrix1.actualCols;

			if constexpr (returnTransposed)
			{
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < cols; i++)
					{
						dataResult[j * rows + i] = data1[i * actualCols + j] ? num1 : num2;
					}
				}
			}
			else
			{
				if constexpr (matrix1Contiguous)
				{
					size_t size = matrix1._size;
					size_t finalPosSize = matrix1.finalPosSize;
					for (size_t i = 0; i < finalPosSize; i += 8)
					{
						__m256 a = _mm256_castsi256_ps(_mm256_cvtepi8_epi32(_mm_castpd_si128(_mm_load_sd(reinterpret_cast<double*>(&data1[i])))));

						_mm256_store_ps(&dataResult[i], _mm256_blendv_ps(_num2, _num1, a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = data1[i] ? num1 : num2;
					}
				}
				else
				{
					size_t actualCols = matrix1.actualCols;
					size_t finalPosCols = matrix1.finalPosCols;
					for (size_t j = 0; j < finalPosCols; j += 8)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256 a = _mm256_castsi256_ps(_mm256_cvtepi8_epi32(_mm_castpd_si128(_mm_load_sd(reinterpret_cast<double*>(&data1[i * actualCols + j])))));
							_mm256_store_ps(&dataResult[i * cols + j], _mm256_blendv_ps(_num2, _num1, a));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[i * actualCols + j] ? num1 : num2;
						}
					}
				}
			}
		}
		return result;
	}

	// Int

	inline vector<int> where(vector<uint8_t>& vector1, vector<int>& vector2, vector<int>& vector3)
	{
#ifdef _DEBUG
		if (vector1._size != vector2._size || vector2._size != vector3._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = vector1._size;

		size_t finalPos = vector1.finalPos;

		uint8_t* data1 = vector1._data;
		int* data2 = vector2._data;
		int* data3 = vector3._data;

		vector<int> result(size);

		int* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256 mask = _mm256_castsi256_ps(_mm256_cvtepi8_epi32(_mm_castpd_si128(_mm_load_sd(reinterpret_cast<double*>(&data1[i])))));

			_mm256_storeu_epi32(&dataResult[i], _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(_mm256_loadu_epi32(&data3[i])), _mm256_castsi256_ps(_mm256_loadu_epi32(&data2[i])), mask)));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] ? data2[i] : data3[i];
		}
		return result;
	}

	inline vector<int> where(vector<uint8_t>& vector1, int num1, int num2)

	{
		size_t size = vector1._size;

		uint8_t* data1 = vector1._data;

		__m256 _num1 = _mm256_castsi256_ps(_mm256_set1_epi32(num1));
		__m256 _num2 = _mm256_castsi256_ps(_mm256_set1_epi32(num2));

		size_t finalPos = (size / 8) * 8;

		vector<int> result(size);

		int* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_storeu_epi32(&dataResult[i], _mm256_castps_si256(_mm256_blendv_ps(_num2, _num1,
				_mm256_castsi256_ps(_mm256_cvtepi8_epi32(
					_mm_castpd_si128(_mm_load_sd(reinterpret_cast<double*>(&data1[i]))))))));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] ? num1 : num2;
		}
		return result;
	}

	inline vector<int> where(vector<uint8_t>& vector1, vector<int>& vector2, int num)

	{
#ifdef _DEBUG
		if (vector1._size != vector2._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = vector1._size;

		size_t finalPos = vector2.finalPos;

		uint8_t* data1 = vector1._data;

		int* data2 = vector2._data;

		__m256 b = _mm256_castsi256_ps(_mm256_set1_epi32(num));

		vector<int> result(size);

		int* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256 a = _mm256_castsi256_ps(_mm256_loadu_epi32(&data2[i]));
			_mm256_storeu_epi32(&dataResult[i], _mm256_castps_si256(_mm256_blendv_ps(b, a,
				_mm256_castsi256_ps(_mm256_cvtepi8_epi32(
					_mm_castpd_si128(_mm_load_sd(reinterpret_cast<double*>(&data1[i]))))))));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] ? data2[i] : num;
		}
		return result;
	}

	inline vector<int> where(vector<uint8_t>& vector1, int num, vector<int>& vector2)

	{
#ifdef _DEBUG
		if (vector1._size != vector2._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = vector1._size;

		size_t finalPos = vector2.finalPos;

		uint8_t* data1 = vector1._data;

		int* data2 = vector2._data;

		__m256 b = _mm256_castsi256_ps(_mm256_set1_epi32(num));

		vector<int> result(size);

		int* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256 a = _mm256_castsi256_ps(_mm256_loadu_epi32(&data2[i]));
			_mm256_storeu_epi32(&dataResult[i], _mm256_castps_si256(_mm256_blendv_ps(a, b,
				_mm256_castsi256_ps(_mm256_cvtepi8_epi32(
					_mm_castpd_si128(_mm_load_sd(reinterpret_cast<double*>(&data1[i]))))))));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] ? num : data2[i];
		}
		return result;
	}

	// uint64_t

	inline vector<uint64_t> where(vector<uint8_t>& vector1, vector<uint64_t>& vector2, vector<uint64_t>& vector3)

	{
#ifdef _DEBUG
		if (vector1._size != vector2._size || vector2._size != vector3._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = vector1._size;

		size_t finalPos = vector1.finalPos;

		uint8_t* data1 = vector1._data;
		uint64_t* data2 = vector2._data;
		uint64_t* data3 = vector3._data;

		vector<uint64_t> result(size);

		uint64_t* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d mask = _mm256_castsi256_pd(_mm256_cvtepi8_epi64(_mm_castps_si128(_mm_load_ss(reinterpret_cast<float*>(&data1[i])))));

			_mm256_storeu_epi64(&dataResult[i], _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(_mm256_loadu_epi64(&data3[i])), _mm256_castsi256_pd(_mm256_loadu_epi64(&data2[i])), mask)));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] ? data2[i] : data3[i];
		}
		return result;
	}

	inline vector<uint64_t> where(vector<uint8_t>& vector1, uint64_t num1, uint64_t num2)

	{
		size_t size = vector1._size;

		uint8_t* data1 = vector1._data;

		__m256d _num1 = _mm256_castsi256_pd(_mm256_set1_epi64x(num1));
		__m256d _num2 = _mm256_castsi256_pd(_mm256_set1_epi64x(num2));

		size_t finalPos = (size / 4) * 4;

		vector<uint64_t> result(size);

		uint64_t* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_storeu_epi64(&dataResult[i], _mm256_castpd_si256(_mm256_blendv_pd(_num2, _num1,
				_mm256_castsi256_pd(_mm256_cvtepi8_epi64(
					_mm_castps_si128(_mm_load_ss(reinterpret_cast<float*>(&data1[i]))))))));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] ? num1 : num2;
		}
		return result;
	}

	inline vector<uint64_t> where(vector<uint8_t>& vector1, vector<uint64_t>& vector2, uint64_t num)

	{
#ifdef _DEBUG
		if (vector1._size != vector2._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = vector1._size;

		size_t finalPos = vector2.finalPos;

		uint8_t* data1 = vector1._data;

		uint64_t* data2 = vector2._data;

		__m256d b = _mm256_castsi256_pd(_mm256_set1_epi64x(num));

		vector<uint64_t> result(size);

		uint64_t* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_castsi256_pd(_mm256_loadu_epi64(&data2[i]));
			_mm256_storeu_epi64(&dataResult[i], _mm256_castpd_si256(_mm256_blendv_pd(b, a,
				_mm256_castsi256_pd(_mm256_cvtepi8_epi64(
					_mm_castps_si128(_mm_load_ss(reinterpret_cast<float*>(&data1[i]))))))));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] ? data2[i] : num;
		}
		return result;
	}

	inline vector<uint64_t> where(vector<uint8_t>& vector1, uint64_t num, vector<uint64_t>& vector2)


	{
#ifdef _DEBUG
		if (vector1._size != vector2._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = vector1._size;

		size_t finalPos = vector2.finalPos;

		uint8_t* data1 = vector1._data;

		uint64_t* data2 = vector2._data;

		__m256d b = _mm256_castsi256_pd(_mm256_set1_epi64x(num));

		vector<uint64_t> result(size);

		uint64_t* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_castsi256_pd(_mm256_loadu_epi64(&data2[i]));
			_mm256_storeu_epi64(&dataResult[i], _mm256_castpd_si256(_mm256_blendv_pd(a, b,
				_mm256_castsi256_pd(_mm256_cvtepi8_epi64(
					_mm_castps_si128(_mm_load_ss(reinterpret_cast<float*>(&data1[i]))))))));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] ? num : data2[i];
		}
		return result;
	}

	// indices 

	inline vector<uint64_t> where(vector<uint8_t>& vector1)
	{
		size_t sizeVector1 = vector1._size;

		uint8_t* data1 = vector1._data;

		size_t sizeResult = vector1.count();

		vector<uint64_t> result(sizeResult);

		uint64_t* dataResult = result._data;

		for (size_t iVector1{ 0 }, iResult{ 0 }; iVector1 < sizeVector1; iVector1++)
		{
			if (data1[iVector1])
			{
				dataResult[iResult] = iVector1;
				iResult++;
			}
		}
		return result;
	}
}
