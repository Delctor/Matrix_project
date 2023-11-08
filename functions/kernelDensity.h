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
	template<bool matrix1Contiguous, bool matrix2Contiguous>
	inline vector<double> kernelDensity(matrix<double, false, matrix1Contiguous>& samples, matrix<double, false, matrix2Contiguous>& x, double bandwidth)
	{
		double num = std::pow((1.0 / (bandwidth * 2.5066282746310002)), static_cast<double>(x._cols));
		__m256d _bandwidth = _mm256_set1_pd(bandwidth);
		__m256d _num = _mm256_set1_pd(num);
		
		__m256d _multiplier = _mm256_set1_pd(-0.5);

		vector<double> result(x._rows);

		double* dataResult = result._data;

		double samples_rows_d = static_cast<double>(samples._rows);

		__m256d _samples_rows_d = _mm256_set1_pd(samples_rows_d);

		double* data1 = samples._data;
		double* data2 = x._data;

		for (size_t param = 0; param < x.finalPosRows; param += 4)
		{
			__m256d _sum = _mm256_setzero_pd();

			for (size_t sample = 0; sample < samples._rows; sample++)
			{
				__m256d _sum2 = _mm256_setzero_pd();
				for (size_t col = 0; col < samples._cols; col++)
				{
					__m256d _diff = _mm256_div_pd(_mm256_sub_pd(_mm256_set1_pd(data1[sample * samples.actualCols + col]),
						_mm256_setr_pd(data2[param * x.actualCols + col], 
							data2[(param + 1) * x.actualCols + col],
							data2[(param + 2) * x.actualCols + col],
							data2[(param + 3) * x.actualCols + col])), _bandwidth);
					_sum2 = _mm256_add_pd(_sum2, _mm256_mul_pd(_diff, _diff));
				}
				_sum = _mm256_add_pd(_sum, _mm256_mul_pd(_mm256_exp_pd(_mm256_mul_pd(_sum2, _multiplier)), _num));
			}
			_mm256_store_pd(&dataResult[param], _mm256_div_pd(_sum, _samples_rows_d));
		}
		for (size_t param = x.finalPosRows; param < x._rows; param++)
		{
			double sum = 0.0;
			for (size_t sample = 0; sample < samples._rows; sample++)
			{
				double sum2 = 0.0;
				for (size_t col = 0; col < samples._cols; col++)
				{
					double diff = (data1[sample * samples.actualCols + col] - data2[param * x.actualCols + col]) / bandwidth;
					sum2 += diff * diff;
				}
				sum += std::exp(sum2 * -0.5) * num;
			}
			dataResult[param] = sum / samples_rows_d;
		}

		return result;
	}
	
	template<bool matrix1Contiguous, bool matrix2Contiguous>
	inline vector<float> kernelDensity(matrix<float, false, matrix1Contiguous>& samples, matrix<float, false, matrix2Contiguous>& x, float bandwidth)
	{
		float num = std::pow((1.0f / (bandwidth * 2.5066282746310002f)), static_cast<float>(x._cols));
		__m256 _bandwidth = _mm256_set1_ps(bandwidth);
		__m256 _num = _mm256_set1_ps(num);

		__m256 _multiplier = _mm256_set1_ps(-0.5f);

		vector<float> result(x._rows);

		float* dataResult = result._data;

		float samples_rows_f = static_cast<float>(samples._rows);

		__m256 _samples_rows_f = _mm256_set1_ps(samples_rows_f);

		float* data1 = samples._data;
		float* data2 = x._data;

		for (size_t param = 0; param < x.finalPosRows; param += 8)
		{
			__m256 _sum = _mm256_setzero_ps();

			for (size_t sample = 0; sample < samples._rows; sample++)
			{
				__m256 _sum2 = _mm256_setzero_ps();
				for (size_t col = 0; col < samples._cols; col++)
				{
					__m256 _diff = _mm256_div_ps(_mm256_sub_ps(_mm256_set1_ps(data1[sample * samples.actualCols + col]),
						_mm256_setr_ps(data2[param * x.actualCols + col],
							data2[(param + 1) * x.actualCols + col],
							data2[(param + 2) * x.actualCols + col],
							data2[(param + 3) * x.actualCols + col], 
							data2[(param + 4) * x.actualCols + col],
							data2[(param + 5) * x.actualCols + col],
							data2[(param + 6) * x.actualCols + col], 
							data2[(param + 7) * x.actualCols + col])), _bandwidth);
					_sum2 = _mm256_add_ps(_sum2, _mm256_mul_ps(_diff, _diff));
				}
				_sum = _mm256_add_ps(_sum, _mm256_mul_ps(_mm256_exp_ps(_mm256_mul_ps(_sum2, _multiplier)), _num));
			}
			_mm256_store_ps(&dataResult[param], _mm256_div_ps(_sum, _samples_rows_f));
		}
		for (size_t param = x.finalPosRows; param < x._rows; param++)
		{
			float sum = 0.0;
			for (size_t sample = 0; sample < samples._rows; sample++)
			{
				float sum2 = 0.0;
				for (size_t col = 0; col < samples._cols; col++)
				{
					float diff = (data1[sample * samples.actualCols + col] - data2[param * x.actualCols + col]) / bandwidth;
					sum2 += diff * diff;
				}
				sum += std::exp(sum2 * -0.5f) * num;
			}
			dataResult[param] = sum / samples_rows_f;
		}
		return result;
	}

}