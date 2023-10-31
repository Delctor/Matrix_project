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
	template<bool useSteps = true, bool thisContiguous>
	inline matrix<double> randomGenerator(alge::matrix<double, false, thisContiguous>& bounds, size_t nRandoms)
	{
#ifdef _DEBUG
		if ((useSteps && bounds._cols < 3) || bounds._cols < 2) throw std::invalid_argument("Wrong dimensions");
#else
#endif

		size_t nParams = bounds._rows;
		matrix<double> result(nRandoms, nParams);

		size_t boundsActualCols = bounds.actualCols;

		double* data1 = bounds._data;

		double* dataResult = result._data;

		size_t finalPosCols = bounds.finalPosRows;
		size_t finalPosRows = (nRandoms / 4) * 4;

		masks_uint64_to_double;

		__m256i random;

		if constexpr (useSteps)
		{
			for (size_t j = 0; j < finalPosCols; j += 4)
			{
				__m256d _min = _mm256_setr_pd(data1[j * boundsActualCols],
					data1[(j + 1) * boundsActualCols],
					data1[(j + 2) * boundsActualCols],
					data1[(j + 3) * boundsActualCols]);
				__m256d _max = _mm256_setr_pd(data1[j * boundsActualCols + 1],
					data1[(j + 1) * boundsActualCols + 1],
					data1[(j + 2) * boundsActualCols + 1],
					data1[(j + 3) * boundsActualCols + 1]);
				__m256d _step = _mm256_setr_pd(data1[j * boundsActualCols + 2],
					data1[(j + 1) * boundsActualCols + 2],
					data1[(j + 2) * boundsActualCols + 2],
					data1[(j + 3) * boundsActualCols + 2]);
				__m256d _range = _mm256_sub_pd(_max, _min);
				__m256d divisor = _mm256_set1_pd(18446744073709551615.0);
				divisor = _mm256_div_pd(divisor, _range);

				for (size_t i = 0; i < nRandoms; i++)
				{
					random = _mm256_slli_epi64(__seeds__, 13);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					random = _mm256_srli_epi64(__seeds__, 10);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					random = _mm256_slli_epi64(__seeds__, 20);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					uint64_to_double(__seeds__);

					uint64ToDouble = _mm256_div_pd(uint64ToDouble, divisor);

					uint64ToDouble = _mm256_add_pd(_mm256_mul_pd(_mm256_round_pd(_mm256_div_pd(uint64ToDouble, _step), _MM_FROUND_TO_NEAREST_INT), _step), _min);

					_mm256_store_pd(&dataResult[i * nParams + j], uint64ToDouble);
				}
			}
			for (size_t j = finalPosCols; j < nParams; j++)
			{
				double min = data1[j * boundsActualCols];
				double max = data1[j * boundsActualCols + 1];
				double range = max - min;
				double step = data1[j * boundsActualCols + 2];
				__m256d _min = _mm256_set1_pd(min);
				__m256d _range = _mm256_set1_pd(range);
				__m256d _step = _mm256_set1_pd(step);
				__m256d divisor = _mm256_set1_pd(18446744073709551615.0 / range);

				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					random = _mm256_slli_epi64(__seeds__, 13);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					random = _mm256_srli_epi64(__seeds__, 10);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					random = _mm256_slli_epi64(__seeds__, 20);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					uint64_to_double(__seeds__);

					uint64ToDouble = _mm256_div_pd(uint64ToDouble, divisor);

					uint64ToDouble = _mm256_add_pd(_mm256_mul_pd(_mm256_round_pd(_mm256_div_pd(uint64ToDouble, _step), _MM_FROUND_TO_NEAREST_INT), _step), _min);

					__m128d val1 = _mm256_extractf128_pd(uint64ToDouble, 1);
					__m128d val2 = _mm256_castpd256_pd128(uint64ToDouble);

					_mm_store_sd(&dataResult[i * nParams + j], val2);
					val2 = _mm_shuffle_pd(val2, val2, 1);
					_mm_store_sd(&dataResult[(i + 1) * nParams + j], val2);

					_mm_store_sd(&dataResult[(i + 2) * nParams + j], val1);
					val1 = _mm_shuffle_pd(val1, val1, 1);
					_mm_store_sd(&dataResult[(i + 3) * nParams + j], val1);

				}
				for (size_t i = finalPosRows; i < nRandoms; i++)
				{
					random = _mm256_slli_epi64(__seeds__, 13);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					random = _mm256_srli_epi64(__seeds__, 10);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					random = _mm256_slli_epi64(__seeds__, 20);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					uint64_to_double(__seeds__);

					uint64ToDouble = _mm256_div_pd(uint64ToDouble, divisor);

					uint64ToDouble = _mm256_add_pd(_mm256_mul_pd(_mm256_round_pd(_mm256_div_pd(uint64ToDouble, _step), _MM_FROUND_TO_NEAREST_INT), _step), _min);

					_mm_store_sd(&dataResult[i * nParams + j], _mm256_castpd256_pd128(uint64ToDouble));
				}
			}
		}
		else
		{
			for (size_t j = 0; j < finalPosCols; j += 4)
			{
				__m256d _min = _mm256_setr_pd(data1[j * boundsActualCols],
					data1[(j + 1) * boundsActualCols],
					data1[(j + 2) * boundsActualCols],
					data1[(j + 3) * boundsActualCols]);
				__m256d _max = _mm256_setr_pd(data1[j * boundsActualCols + 1],
					data1[(j + 1) * boundsActualCols + 1],
					data1[(j + 2) * boundsActualCols + 1],
					data1[(j + 3) * boundsActualCols + 1]);
				__m256d _range = _mm256_sub_pd(_max, _min);
				__m256d divisor = _mm256_set1_pd(18446744073709551615.0);
				divisor = _mm256_div_pd(divisor, _range);

				for (size_t i = 0; i < nRandoms; i++)
				{
					random = _mm256_slli_epi64(__seeds__, 13);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					random = _mm256_srli_epi64(__seeds__, 10);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					random = _mm256_slli_epi64(__seeds__, 20);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					uint64_to_double(__seeds__);

					uint64ToDouble = _mm256_div_pd(uint64ToDouble, divisor);

					uint64ToDouble = _mm256_add_pd(uint64ToDouble, _min);

					_mm256_store_pd(&dataResult[i * nParams + j], uint64ToDouble);
				}
			}
			for (size_t j = finalPosCols; j < nParams; j++)
			{
				double min = data1[j * boundsActualCols];
				double max = data1[j * boundsActualCols + 1];
				double range = max - min;
				__m256d _min = _mm256_set1_pd(min);
				__m256d _range = _mm256_set1_pd(range);
				__m256d divisor = _mm256_set1_pd(18446744073709551615.0 / range);

				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					random = _mm256_slli_epi64(__seeds__, 13);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					random = _mm256_srli_epi64(__seeds__, 10);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					random = _mm256_slli_epi64(__seeds__, 20);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					uint64_to_double(__seeds__);

					uint64ToDouble = _mm256_div_pd(uint64ToDouble, divisor);

					uint64ToDouble = _mm256_add_pd(uint64ToDouble, _min);

					__m128d val1 = _mm256_extractf128_pd(uint64ToDouble, 1);
					__m128d val2 = _mm256_castpd256_pd128(uint64ToDouble);

					_mm_store_sd(&dataResult[i * nParams + j], val2);
					val2 = _mm_shuffle_pd(val2, val2, 1);
					_mm_store_sd(&dataResult[(i + 1) * nParams + j], val2);

					_mm_store_sd(&dataResult[(i + 2) * nParams + j], val1);
					val1 = _mm_shuffle_pd(val1, val1, 1);
					_mm_store_sd(&dataResult[(i + 3) * nParams + j], val1);

				}
				for (size_t i = finalPosRows; i < nRandoms; i++)
				{
					random = _mm256_slli_epi64(__seeds__, 13);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					random = _mm256_srli_epi64(__seeds__, 10);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					random = _mm256_slli_epi64(__seeds__, 20);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					uint64_to_double(__seeds__);

					uint64ToDouble = _mm256_div_pd(uint64ToDouble, divisor);

					uint64ToDouble = _mm256_add_pd(uint64ToDouble, _min);

					_mm_store_sd(&dataResult[i * nParams + j], _mm256_castpd256_pd128(uint64ToDouble));
				}
			}
		}
		return result;
	}

	template<bool useSteps = true, bool thisContiguous>
	inline matrix<float> randomGenerator(alge::matrix<float, false, thisContiguous>& bounds, size_t nRandoms)
	{
#ifdef _DEBUG
		if ((useSteps && bounds._cols < 3) || bounds._cols < 2) throw std::invalid_argument("Wrong dimensions");
#else
#endif

		size_t nParams = bounds._rows;
		matrix<float> result(nRandoms, nParams);

		size_t boundsActualCols = bounds.actualCols;

		float* data1 = bounds._data;

		float* dataResult = result._data;

		size_t finalPosCols = bounds.finalPosRows;
		size_t finalPosRows = (nRandoms / 8) * 8;

		__m256i random;

		if constexpr (useSteps)
		{
			for (size_t j = 0; j < finalPosCols; j += 8)
			{
				__m256 _min = _mm256_setr_ps(data1[j * boundsActualCols],
					data1[(j + 1) * boundsActualCols],
					data1[(j + 2) * boundsActualCols],
					data1[(j + 3) * boundsActualCols],
					data1[(j + 4) * boundsActualCols],
					data1[(j + 5) * boundsActualCols],
					data1[(j + 6) * boundsActualCols],
					data1[(j + 7) * boundsActualCols]);
				__m256 _max = _mm256_setr_ps(data1[j * boundsActualCols + 1],
					data1[(j + 1) * boundsActualCols + 1],
					data1[(j + 2) * boundsActualCols + 1],
					data1[(j + 3) * boundsActualCols + 1],
					data1[(j + 4) * boundsActualCols + 1],
					data1[(j + 5) * boundsActualCols + 1],
					data1[(j + 6) * boundsActualCols + 1],
					data1[(j + 7) * boundsActualCols + 1]);
				__m256 _step = _mm256_setr_ps(data1[j * boundsActualCols + 2],
					data1[(j + 1) * boundsActualCols + 2],
					data1[(j + 2) * boundsActualCols + 2],
					data1[(j + 3) * boundsActualCols + 2],
					data1[(j + 4) * boundsActualCols + 2],
					data1[(j + 5) * boundsActualCols + 2],
					data1[(j + 6) * boundsActualCols + 2],
					data1[(j + 7) * boundsActualCols + 2]);
				__m256 _range = _mm256_sub_ps(_max, _min);
				__m256 divisor = _mm256_set1_ps(4294967295.0f);
				divisor = _mm256_div_ps(divisor, _range);

				for (size_t i = 0; i < nRandoms; i++)
				{
					random = _mm256_slli_epi32(__seeds__, 6);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					random = _mm256_srli_epi32(__seeds__, 5);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					random = _mm256_slli_epi32(__seeds__, 10);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					uint32_to_float(__seeds__);

					uint32ToFloat = _mm256_div_ps(uint32ToFloat, divisor);

					uint32ToFloat = _mm256_add_ps(_mm256_mul_ps(_mm256_round_ps(_mm256_div_ps(uint32ToFloat, _step), _MM_FROUND_TO_NEAREST_INT), _step), _min);

					_mm256_store_ps(&dataResult[i * nParams + j], uint32ToFloat);
				}
			}
			for (size_t j = finalPosCols; j < nParams; j++)
			{
				float min = data1[j * boundsActualCols];
				float max = data1[j * boundsActualCols + 1];
				float range = max - min;
				float step = data1[j * boundsActualCols + 2];
				__m256 _min = _mm256_set1_ps(min);
				__m256 _range = _mm256_set1_ps(range);
				__m256 _step = _mm256_set1_ps(step);
				__m256 divisor = _mm256_set1_ps(4294967295.0f / range);

				for (size_t i = 0; i < nRandoms; i++)
				{
					random = _mm256_slli_epi32(__seeds__, 6);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					random = _mm256_srli_epi32(__seeds__, 5);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					random = _mm256_slli_epi32(__seeds__, 10);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					uint32_to_float(__seeds__);

					uint32ToFloat = _mm256_div_ps(uint32ToFloat, divisor);

					uint32ToFloat = _mm256_add_ps(_mm256_mul_ps(_mm256_round_ps(_mm256_div_ps(uint32ToFloat, _step), _MM_FROUND_TO_NEAREST_INT), _step), _min);

					_mm_store_ss(&dataResult[i * nParams + j], _mm256_castps256_ps128(uint32ToFloat));
				}
			}
		}
		else
		{
			for (size_t j = 0; j < finalPosCols; j += 8)
			{
				__m256 _min = _mm256_setr_ps(data1[j * boundsActualCols],
					data1[(j + 1) * boundsActualCols],
					data1[(j + 2) * boundsActualCols],
					data1[(j + 3) * boundsActualCols],
					data1[(j + 4) * boundsActualCols],
					data1[(j + 5) * boundsActualCols],
					data1[(j + 6) * boundsActualCols],
					data1[(j + 7) * boundsActualCols]);
				__m256 _max = _mm256_setr_ps(data1[j * boundsActualCols + 1],
					data1[(j + 1) * boundsActualCols + 1],
					data1[(j + 2) * boundsActualCols + 1],
					data1[(j + 3) * boundsActualCols + 1],
					data1[(j + 4) * boundsActualCols + 1],
					data1[(j + 5) * boundsActualCols + 1],
					data1[(j + 6) * boundsActualCols + 1],
					data1[(j + 7) * boundsActualCols + 1]);
				__m256 _range = _mm256_sub_ps(_max, _min);
				__m256 divisor = _mm256_set1_ps(4294967295.0f);
				divisor = _mm256_div_ps(divisor, _range);

				for (size_t i = 0; i < nRandoms; i++)
				{
					random = _mm256_slli_epi32(__seeds__, 6);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					random = _mm256_srli_epi32(__seeds__, 5);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					random = _mm256_slli_epi32(__seeds__, 10);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					uint32_to_float(__seeds__);

					uint32ToFloat = _mm256_div_ps(uint32ToFloat, divisor);

					uint32ToFloat = _mm256_add_ps(uint32ToFloat, _min);

					_mm256_store_ps(&dataResult[i * nParams + j], uint32ToFloat);
				}
			}
			for (size_t j = finalPosCols; j < nParams; j++)
			{
				float min = data1[j * boundsActualCols];
				float max = data1[j * boundsActualCols + 1];
				float range = max - min;
				__m256 _min = _mm256_set1_ps(min);
				__m256 _range = _mm256_set1_ps(range);
				__m256 divisor = _mm256_set1_ps(4294967295.0 / range);

				for (size_t i = 0; i < nRandoms; i++)
				{
					random = _mm256_slli_epi32(__seeds__, 6);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					random = _mm256_srli_epi32(__seeds__, 5);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					random = _mm256_slli_epi32(__seeds__, 10);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					uint32_to_float(__seeds__);

					uint32ToFloat = _mm256_div_ps(uint32ToFloat, divisor);

					uint32ToFloat = _mm256_add_ps(uint32ToFloat, _min);

					_mm_store_ss(&dataResult[i * nParams + j], _mm256_castps256_ps128(uint32ToFloat));
				}
			}
		}
		return result;
	}

}