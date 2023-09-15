#pragma once

#include <iostream>
#include <cmath>
#include <immintrin.h>

#define uint64_to_double(reg, mask) _mm256_sub_pd(_mm256_castsi256_pd(_mm256_or_si256(reg, _mm256_castpd_si256(mask))), mask);

#define int64_to_double(reg, mask) _mm256_sub_pd(_mm256_castsi256_pd(_mm256_add_epi64(reg, _mm256_castpd_si256(mask))), mask)

#define double_to_uint64(reg, mask) _mm256_xor_si256(_mm256_castpd_si256(_mm256_add_pd(reg, mask)),_mm256_castpd_si256(mask))

#define double_to_int64(reg, mask) _mm256_sub_epi64(_mm256_castpd_si256(_mm256_add_pd(reg, mask)),_mm256_castpd_si256(mask))


#define ud _mm256_set1_pd(0x0010000000000000)

#define id _mm256_set1_pd(0x0018000000000000)


#define True 0b11111111
#define False 0b00000000

namespace matrix
{
	template <typename T>
	class vector 
	{
		static_assert(std::is_same<T, double>::value || 
			std::is_same<T, float>::value ||
			std::is_same<T, int>::value || 
			std::is_same<T, uint64_t>::value || 
			std::is_same<T, int64_t>::value ||
			std::is_same<T, bool>::value
			,
			"");
	};
	
	template <typename T, bool tranposed = false, bool contiguous = true, bool call_destructor = true>
	class matrix
	{
		static_assert(std::is_same<T, double>::value ||
			std::is_same<T, float>::value ||
			std::is_same<T, int>::value ||
			std::is_same<T, uint64_t>::value ||
			std::is_same<T, int64_t>::value ||
			std::is_same<T, bool>::value
			,
			"");
	};

	//-----------------------------------

	template <>
	class vector<double>
	{
	public:
		vector(size_t size) : _data(new double[size]), _size(size) {}

		vector(double* data, size_t size) : _data(data), _size(size) {}

		friend class matrix<double>;


		friend class matrix<bool>;

		friend class vector<bool>;

		friend class vector<double>;

		friend class vector<uint64_t>;

		friend class vector<int>;

		double& operator[](size_t index)
		{
			double* data = this->_data;
			return data[index];
		}

		const double& operator[](size_t index) const
		{
			double* data = this->_data;
			return data[index];
		}

		// +

		vector<double> operator+(vector<double>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			double* data2 = other._data;

			vector<double> result(size);

			double* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_load_pd(&data2[i]);

				_mm256_store_pd(&data_result[i], _mm256_add_pd(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] + data2[i];
			}
			return result;
		}

		void operator+=(vector<double>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			double* data2 = other._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_load_pd(&data2[i]);

				_mm256_store_pd(&data1[i], _mm256_add_pd(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] = data1[i] + data2[i];
			}
		}

		vector<double> operator+(vector<float>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			float* data2 = other._data;

			vector<double> result(size);

			double* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_cvtps_pd(_mm_load_ps(&data2[i]));

				_mm256_store_pd(&data_result[i], _mm256_add_pd(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] + static_cast<double>(data2[i]);
			}
			return result;
		}

		void operator+=(vector<float>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			float* data2 = other._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_cvtps_pd(_mm_load_ps(&data2[i]));

				_mm256_store_pd(&data1[i], _mm256_add_pd(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] = data1[i] + static_cast<double>(data2[i]);
			}
		}

		vector<double> operator+(vector<uint64_t>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			uint64_t* data2 = other._data;

			vector<double> result(size);

			double* data_result = result._data;

			__m256d mask = ud;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = uint64_to_double(_mm256_loadu_epi64(&data2[i]), mask);

				_mm256_store_pd(&data_result[i], _mm256_add_pd(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] + static_cast<double>(data2[i]);
			}
			return result;
		}

		void operator+=(vector<uint64_t>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			uint64_t* data2 = other._data;

			__m256d mask = ud;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = uint64_to_double(_mm256_loadu_epi64(&data2[i]), mask);

				_mm256_store_pd(&data1[i], _mm256_add_pd(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] = data1[i] + static_cast<double>(data2[i]);
			}
		}

		vector<double> operator+(vector<int>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			int* data2 = other._data;

			vector<double> result(size);

			double* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_cvtepi32_pd(_mm_loadu_epi32(&data2[i]));

				_mm256_store_pd(&data_result[i], _mm256_add_pd(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] + static_cast<double>(data2[i]);
			}
			return result;
		}

		void operator+=(vector<int>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			int* data2 = other._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_cvtepi32_pd(_mm_loadu_epi32(&data2[i]));

				_mm256_store_pd(&data1[i], _mm256_add_pd(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] = data1[i] + static_cast<double>(data2[i]);
			}
		}

		// -

		vector<double> operator-(vector<double>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			double* data2 = other._data;

			vector<double> result(size);

			double* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_load_pd(&data2[i]);

				_mm256_store_pd(&data_result[i], _mm256_sub_pd(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] - data2[i];
			}
			return result;
		}

		void operator-=(vector<double>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			double* data2 = other._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_load_pd(&data2[i]);

				_mm256_store_pd(&data1[i], _mm256_sub_pd(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] = data1[i] - data2[i];
			}
		}

		vector<double> operator-(vector<float>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			float* data2 = other._data;

			vector<double> result(size);

			double* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_cvtps_pd(_mm_load_ps(&data2[i]));

				_mm256_store_pd(&data_result[i], _mm256_sub_pd(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] - static_cast<double>(data2[i]);
			}
			return result;
		}

		void operator-=(vector<float>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			float* data2 = other._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_cvtps_pd(_mm_load_ps(&data2[i]));

				_mm256_store_pd(&data1[i], _mm256_sub_pd(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] = data1[i] - static_cast<double>(data2[i]);
			}
		}

		vector<double> operator-(vector<uint64_t>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			uint64_t* data2 = other._data;

			vector<double> result(size);

			double* data_result = result._data;

			__m256d mask = ud;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = uint64_to_double(_mm256_loadu_epi64(&data2[i]), mask);

				_mm256_store_pd(&data_result[i], _mm256_sub_pd(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] - static_cast<double>(data2[i]);
			}
			return result;
		}

		void operator-=(vector<uint64_t>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			uint64_t* data2 = other._data;

			__m256d mask = ud;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = uint64_to_double(_mm256_loadu_epi64(&data2[i]), mask);

				_mm256_store_pd(&data1[i], _mm256_sub_pd(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] = data1[i] - static_cast<double>(data2[i]);
			}
		}

		vector<double> operator-(vector<int>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			int* data2 = other._data;

			vector<double> result(size);

			double* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_cvtepi32_pd(_mm_loadu_epi32(&data2[i]));

				_mm256_store_pd(&data_result[i], _mm256_sub_pd(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] - static_cast<double>(data2[i]);
			}
			return result;
		}

		void operator-=(vector<int>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			int* data2 = other._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_cvtepi32_pd(_mm_loadu_epi32(&data2[i]));

				_mm256_store_pd(&data1[i], _mm256_sub_pd(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] = data1[i] - static_cast<double>(data2[i]);
			}
		}

		// *

		vector<double> operator*(vector<double>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			double* data2 = other._data;

			vector<double> result(size);

			double* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_load_pd(&data2[i]);

				_mm256_store_pd(&data_result[i], _mm256_mul_pd(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] * data2[i];
			}
			return result;
		}

		void operator*=(vector<double>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			double* data2 = other._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_load_pd(&data2[i]);

				_mm256_store_pd(&data1[i], _mm256_mul_pd(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] = data1[i] * data2[i];
			}
		}

		vector<double> operator*(vector<float>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			float* data2 = other._data;

			vector<double> result(size);

			double* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_cvtps_pd(_mm_load_ps(&data2[i]));

				_mm256_store_pd(&data_result[i], _mm256_mul_pd(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] * static_cast<double>(data2[i]);
			}
			return result;
		}

		void operator*=(vector<float>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			float* data2 = other._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_cvtps_pd(_mm_load_ps(&data2[i]));

				_mm256_store_pd(&data1[i], _mm256_mul_pd(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] = data1[i] * static_cast<double>(data2[i]);
			}
		}

		vector<double> operator*(vector<uint64_t>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			uint64_t* data2 = other._data;

			vector<double> result(size);

			double* data_result = result._data;

			__m256d mask = ud;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = uint64_to_double(_mm256_loadu_epi64(&data2[i]), mask);

				_mm256_store_pd(&data_result[i], _mm256_mul_pd(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] * static_cast<double>(data2[i]);
			}
			return result;
		}

		void operator*=(vector<uint64_t>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			uint64_t* data2 = other._data;

			__m256d mask = ud;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = uint64_to_double(_mm256_loadu_epi64(&data2[i]), mask);

				_mm256_store_pd(&data1[i], _mm256_mul_pd(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] = data1[i] * static_cast<double>(data2[i]);
			}
		}

		vector<double> operator*(vector<int>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			int* data2 = other._data;

			vector<double> result(size);

			double* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_cvtepi32_pd(_mm_loadu_epi32(&data2[i]));

				_mm256_store_pd(&data_result[i], _mm256_mul_pd(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] * static_cast<double>(data2[i]);
			}
			return result;
		}

		void operator*=(vector<int>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			int* data2 = other._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_cvtepi32_pd(_mm_loadu_epi32(&data2[i]));

				_mm256_store_pd(&data1[i], _mm256_mul_pd(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] = data1[i] * static_cast<double>(data2[i]);
			}
		}


		// /

		vector<double> operator/(vector<double>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			double* data2 = other._data;

			vector<double> result(size);

			double* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_load_pd(&data2[i]);

				_mm256_store_pd(&data_result[i], _mm256_div_pd(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] / data2[i];
			}
			return result;
		}

		void operator/=(vector<double>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			double* data2 = other._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_load_pd(&data2[i]);

				_mm256_store_pd(&data1[i], _mm256_div_pd(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] = data1[i] / data2[i];
			}
		}

		vector<double> operator/(vector<float>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			float* data2 = other._data;

			vector<double> result(size);

			double* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_cvtps_pd(_mm_load_ps(&data2[i]));

				_mm256_store_pd(&data_result[i], _mm256_div_pd(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] / static_cast<double>(data2[i]);
			}
			return result;
		}

		void operator/=(vector<float>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			float* data2 = other._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_cvtps_pd(_mm_load_ps(&data2[i]));

				_mm256_store_pd(&data1[i], _mm256_div_pd(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] = data1[i] / static_cast<double>(data2[i]);
			}
		}

		vector<double> operator/(vector<uint64_t>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			uint64_t* data2 = other._data;

			vector<double> result(size);

			double* data_result = result._data;

			__m256d mask = ud;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = uint64_to_double(_mm256_loadu_epi64(&data2[i]), mask);

				_mm256_store_pd(&data_result[i], _mm256_div_pd(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] / static_cast<double>(data2[i]);
			}
			return result;
		}

		void operator/=(vector<uint64_t>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			uint64_t* data2 = other._data;

			__m256d mask = ud;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = uint64_to_double(_mm256_loadu_epi64(&data2[i]), mask);

				_mm256_store_pd(&data1[i], _mm256_div_pd(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] = data1[i] / static_cast<double>(data2[i]);
			}
		}

		vector<double> operator/(vector<int>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			int* data2 = other._data;

			vector<double> result(size);

			double* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_cvtepi32_pd(_mm_loadu_epi32(&data2[i]));

				_mm256_store_pd(&data_result[i], _mm256_div_pd(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] / static_cast<double>(data2[i]);
			}
			return result;
		}

		void operator/=(vector<int>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			int* data2 = other._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_cvtepi32_pd(_mm_loadu_epi32(&data2[i]));

				_mm256_store_pd(&data1[i], _mm256_div_pd(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] = data1[i] / static_cast<double>(data2[i]);
			}
		}

		// ==

		vector<bool> operator==(vector<double>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			double* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_load_pd(&data2[i]);

				__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_EQ_OQ));
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] == data2[i];
			}
			return result;
		}

		vector<bool> operator==(vector<float>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			float* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_cvtps_pd(_mm_load_ps(&data2[i]));

				__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_EQ_OQ));
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] == static_cast<double>(data2[i]);
			}
			return result;
		}

		vector<bool> operator==(vector<uint64_t>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			uint64_t* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			__m256d mask_cv = ud;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = uint64_to_double(_mm256_loadu_epi64(&data2[i]), mask_cv);

				__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_EQ_OQ));
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] == static_cast<double>(data2[i]);
			}
			return result;
		}

		vector<bool> operator==(vector<int>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			int* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_cvtepi32_pd(_mm_loadu_epi32(&data2[i]));

				__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_EQ_OQ));
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] == static_cast<double>(data2[i]);
			}
			return result;
		}

		// !=

		vector<bool> operator!=(vector<double>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			double* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_load_pd(&data2[i]);

				__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_NEQ_OQ));
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] != data2[i];
			}
			return result;
		}

		vector<bool> operator!=(vector<float>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			float* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_cvtps_pd(_mm_load_ps(&data2[i]));

				__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_NEQ_OQ));
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] != static_cast<double>(data2[i]);
			}
			return result;
		}

		vector<bool> operator!=(vector<uint64_t>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			uint64_t* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			__m256d mask_cv = ud;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = uint64_to_double(_mm256_loadu_epi64(&data2[i]), mask_cv);

				__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_NEQ_OQ));
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] != static_cast<double>(data2[i]);
			}
			return result;
		}

		vector<bool> operator!=(vector<int>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			int* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_cvtepi32_pd(_mm_loadu_epi32(&data2[i]));

				__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_NEQ_OQ));
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] != static_cast<double>(data2[i]);
			}
			return result;
		}

		// >

		vector<bool> operator>(vector<double>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			double* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_load_pd(&data2[i]);

				__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GT_OQ));
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] > data2[i];
			}
			return result;
		}

		vector<bool> operator>(vector<float>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			float* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_cvtps_pd(_mm_load_ps(&data2[i]));

				__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GT_OQ));
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] > static_cast<double>(data2[i]);
			}
			return result;
		}

		vector<bool> operator>(vector<uint64_t>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			uint64_t* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			__m256d mask_cv = ud;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = uint64_to_double(_mm256_loadu_epi64(&data2[i]), mask_cv);

				__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GT_OQ));
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] > static_cast<double>(data2[i]);
			}
			return result;
		}

		vector<bool> operator>(vector<int>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			int* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_cvtepi32_pd(_mm_loadu_epi32(&data2[i]));

				__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GT_OQ));
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] > static_cast<double>(data2[i]);
			}
			return result;
		}

		// >=

		vector<bool> operator>=(vector<double>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			double* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_load_pd(&data2[i]);

				__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GE_OQ));
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] >= data2[i];
			}
			return result;
		}

		vector<bool> operator>=(vector<float>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			float* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_cvtps_pd(_mm_load_ps(&data2[i]));

				__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GE_OQ));
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] >= static_cast<double>(data2[i]);
			}
			return result;
		}

		vector<bool> operator>=(vector<uint64_t>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			uint64_t* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			__m256d mask_cv = ud;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = uint64_to_double(_mm256_loadu_epi64(&data2[i]), mask_cv);

				__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GE_OQ));
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] >= static_cast<double>(data2[i]);
			}
			return result;
		}

		vector<bool> operator>=(vector<int>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			int* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_cvtepi32_pd(_mm_loadu_epi32(&data2[i]));

				__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GE_OQ));
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] >= static_cast<double>(data2[i]);
			}
			return result;
		}

		// <

		vector<bool> operator<(vector<double>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			double* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_load_pd(&data2[i]);

				__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LT_OQ));
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] < data2[i];
			}
			return result;
		}

		vector<bool> operator<(vector<float>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			float* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_cvtps_pd(_mm_load_ps(&data2[i]));

				__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LT_OQ));
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] < static_cast<float>(data2[i]);
			}
			return result;
		}

		vector<bool> operator<(vector<uint64_t>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			uint64_t* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			__m256d mask_cv = ud;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = uint64_to_double(_mm256_loadu_epi64(&data2[i]), mask_cv);

				__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LT_OQ));
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] < static_cast<double>(data2[i]);
			}
			return result;
		}

		vector<bool> operator<(vector<int>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			int* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_cvtepi32_pd(_mm_loadu_epi32(&data2[i]));

				__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LT_OQ));
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] < static_cast<float>(data2[i]);
			}
			return result;
		}

		// <=

		vector<bool> operator<=(vector<double>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			double* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_load_pd(&data2[i]);

				__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LE_OQ));
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] <= data2[i];
			}
			return result;
		}

		vector<bool> operator<=(vector<float>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			float* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_cvtps_pd(_mm_load_ps(&data2[i]));

				__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LE_OQ));
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] <= static_cast<float>(data2[i]);
			}
			return result;
		}

		vector<bool> operator<=(vector<uint64_t>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			uint64_t* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			__m256d mask_cv = ud;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = uint64_to_double(_mm256_loadu_epi64(&data2[i]), mask_cv);

				__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LE_OQ));
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] <= data2[i];
			}
			return result;
		}

		vector<bool> operator<=(vector<int>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;
			int* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);
				__m256d b = _mm256_cvtepi32_pd(_mm_loadu_epi32(&data2[i]));

				__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LE_OQ));
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] <= static_cast<float>(data2[i]);
			}
			return result;
		}

		// Functions

		// Pow

		vector<double> pow(double exponent)
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			vector<double> result(size);

			double* data_result = result._data;

			__m256d _exponet = _mm256_set1_pd(exponent);

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_store_pd(&data_result[i], _mm256_pow_pd(_mm256_load_pd(&data1[i]), _exponet));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = std::pow(data1[i], exponent);
			}
			return result;
		}

		vector<double> pow(vector<double>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			double* data2 = other._data;

			vector<double> result(size);

			double* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_store_pd(&data_result[i], _mm256_pow_pd(_mm256_load_pd(&data1[i]), _mm256_load_pd(&data2[i])));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = std::pow(data1[i], data2[i]);
			}
			return result;
		}

		vector<double> pow(vector<uint64_t>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			uint64_t* data2 = other._data;

			vector<double> result(size);

			double* data_result = result._data;
			
			__m256d mask = ud;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d _exponent = uint64_to_double(_mm256_loadu_epi64(&data2[i]), mask);
				_mm256_store_pd(&data_result[i], _mm256_pow_pd(_mm256_load_pd(&data1[i]), _exponent));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = std::pow(data1[i], static_cast<double>(data2[i]));
			}
			return result;
		}

		vector<double> pow(vector<int>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			int* data2 = other._data;

			vector<double> result(size);

			double* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d _exponent = _mm256_cvtepi32_pd(_mm_loadu_epi32(&data2[i]));
				_mm256_store_pd(&data_result[i], _mm256_pow_pd(_mm256_load_pd(&data1[i]), _exponent));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = std::pow(data1[i], static_cast<double>(data2[i]));
			}
			return result;
		}

		void self_pow(double exponent)
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			__m256d _exponet = _mm256_set1_pd(exponent);

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_store_pd(&data1[i], _mm256_pow_pd(_mm256_load_pd(&data1[i]), _exponet));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] = std::pow(data1[i], exponent);
			}
		}

		void self_pow(vector<double>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			double* data2 = other._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_store_pd(&data1[i], _mm256_pow_pd(_mm256_load_pd(&data1[i]), _mm256_load_pd(&data2[i])));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] = std::pow(data1[i], data2[i]);
			}
		}

		void self_pow(vector<uint64_t>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			uint64_t* data2 = other._data;

			__m256d mask = ud;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d _exponent = uint64_to_double(_mm256_loadu_epi64(&data2[i]), mask);
				_mm256_store_pd(&data1[i], _mm256_pow_pd(_mm256_load_pd(&data1[i]), _exponent));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] = std::pow(data1[i], static_cast<double>(data2[i]));
			}
		}

		void self_pow(vector<int>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			int* data2 = other._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d _exponent = _mm256_cvtepi32_pd(_mm_loadu_epi32(&data2[i]));
				_mm256_store_pd(&data1[i], _mm256_pow_pd(_mm256_load_pd(&data1[i]), _exponent));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] = std::pow(data1[i], static_cast<double>(data2[i]));
			}
		}

		// Root

		vector<double> root(double index)
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			vector<double> result(size);

			double* data_result = result._data;

			index = 1 / index;

			__m256d _index = _mm256_set1_pd(index);

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_store_pd(&data_result[i], _mm256_pow_pd(_mm256_load_pd(&data1[i]), _index));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = std::pow(data1[i], index);
			}
			return result;
		}

		vector<double> root(vector<double>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			double* data2 = other._data;

			vector<double> result(size);

			double* data_result = result._data;

			__m256d ones = _mm256_set1_pd(1.0);

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_store_pd(&data_result[i], _mm256_pow_pd(_mm256_load_pd(&data1[i]), _mm256_div_pd(ones, _mm256_load_pd(&data2[i]))));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = std::pow(data1[i], 1 / data2[i]);
			}
			return result;
		}

		vector<double> root(vector<uint64_t>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			uint64_t* data2 = other._data;

			vector<double> result(size);

			double* data_result = result._data;

			__m256d mask = ud;

			__m256d ones = _mm256_set1_pd(1.0);

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d _index = uint64_to_double(_mm256_loadu_epi64(&data2[i]), mask);
				_mm256_store_pd(&data_result[i], _mm256_pow_pd(_mm256_div_pd(ones, _mm256_load_pd(&data1[i])), _index));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = std::pow(data1[i], 1.0 / static_cast<double>(data2[i]));
			}
			return result;
		}

		vector<double> root(vector<int>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			int* data2 = other._data;

			vector<double> result(size);

			double* data_result = result._data;

			__m256d ones = _mm256_set1_pd(1.0);

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d _index = _mm256_cvtepi32_pd(_mm_loadu_epi32(&data2[i]));
				_mm256_store_pd(&data_result[i], _mm256_pow_pd(_mm256_div_pd(ones, _mm256_load_pd(&data1[i])), _index));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = std::pow(data1[i], 1 / static_cast<double>(data2[i]));
			}
			return result;
		}

		void self_root(double index)
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			index = 1 / index;

			__m256d _index = _mm256_set1_pd(index);

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_store_pd(&data1[i], _mm256_pow_pd(_mm256_load_pd(&data1[i]), _index));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] = std::pow(data1[i], index);
			}
		}

		void self_root(vector<double>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			double* data2 = other._data;

			__m256d ones = _mm256_set1_pd(1.0);

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_store_pd(&data1[i], _mm256_pow_pd(_mm256_load_pd(&data1[i]), _mm256_div_pd(ones, _mm256_load_pd(&data2[i]))));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] = std::pow(data1[i], 1 / data2[i]);
			}
		}

		void self_root(vector<uint64_t>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			uint64_t* data2 = other._data;

			__m256d mask = ud;

			__m256d ones = _mm256_set1_pd(1.0);

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d _index = uint64_to_double(_mm256_loadu_epi64(&data2[i]), mask);
				_mm256_store_pd(&data1[i], _mm256_pow_pd(_mm256_div_pd(ones, _mm256_load_pd(&data1[i])), _index));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] = std::pow(data1[i], 1.0 / static_cast<double>(data2[i]));
			}
		}

		void self_root(vector<int>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			int* data2 = other._data;

			__m256d ones = _mm256_set1_pd(1.0);

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d _index = _mm256_cvtepi32_pd(_mm_loadu_epi32(&data2[i]));
				_mm256_store_pd(&data1[i], _mm256_pow_pd(_mm256_div_pd(ones, _mm256_load_pd(&data1[i])), _index));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] = std::pow(data1[i], 1 / static_cast<double>(data2[i]));
			}
		}

		// Log

		vector<double> log()
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			vector<double> result(size);

			double* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_store_pd(&data_result[i], _mm256_log_pd(_mm256_load_pd(&data1[i])));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = std::log(data1[i]);
			}
			return result;
		}

		void self_log()
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_store_pd(&data1[i], _mm256_log_pd(_mm256_load_pd(&data1[i])));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] = std::log(data1[i]);
			}
		}

		// Log2

		vector<double> log2()
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			vector<double> result(size);

			double* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_store_pd(&data_result[i], _mm256_log2_pd(_mm256_load_pd(&data1[i])));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = std::log2(data1[i]);
			}
			return result;
		}

		void self_log2()
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_store_pd(&data1[i], _mm256_log2_pd(_mm256_load_pd(&data1[i])));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] = std::log2(data1[i]);
			}
		}

		// Log10

		vector<double> log10()
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			vector<double> result(size);

			double* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_store_pd(&data_result[i], _mm256_log10_pd(_mm256_load_pd(&data1[i])));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = std::log10(data1[i]);
			}
			return result;
		}

		void self_log10()
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_store_pd(&data1[i], _mm256_log10_pd(_mm256_load_pd(&data1[i])));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] = std::log10(data1[i]);
			}
		}

		// Exp

		vector<double> exp()
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			vector<double> result(size);

			double* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_store_pd(&data_result[i], _mm256_exp_pd(_mm256_load_pd(&data1[i])));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = std::exp(data1[i]);
			}
			return result;
		}

		void self_exp()
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_store_pd(&data1[i], _mm256_exp_pd(_mm256_load_pd(&data1[i])));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] = std::exp(data1[i]);
			}
		}

		// Exp2

		vector<double> exp2()
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			vector<double> result(size);

			double* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_store_pd(&data_result[i], _mm256_exp2_pd(_mm256_load_pd(&data1[i])));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = std::exp2(data1[i]);
			}
			return result;
		}

		void self_exp2()
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_store_pd(&data1[i], _mm256_exp2_pd(_mm256_load_pd(&data1[i])));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] = std::exp2(data1[i]);
			}
		}

		// Tan

		vector<double> tan()
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			vector<double> result(size);

			double* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_store_pd(&data_result[i], _mm256_tan_pd(_mm256_load_pd(&data1[i])));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = std::tan(data1[i]);
			}
			return result;
		}

		void self_tan()
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_store_pd(&data1[i], _mm256_tan_pd(_mm256_load_pd(&data1[i])));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] = std::tan(data1[i]);
			}
		}

		// Cos

		vector<double> cos()
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			vector<double> result(size);

			double* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_store_pd(&data_result[i], _mm256_cos_pd(_mm256_load_pd(&data1[i])));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = std::cos(data1[i]);
			}
			return result;
		}

		void self_cos()
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_store_pd(&data1[i], _mm256_cos_pd(_mm256_load_pd(&data1[i])));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] = std::cos(data1[i]);
			}
		}

		// Acos

		vector<double> acos()
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			vector<double> result(size);

			double* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_store_pd(&data_result[i], _mm256_acos_pd(_mm256_load_pd(&data1[i])));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = std::acos(data1[i]);
			}
			return result;
		}

		void self_acos()
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_store_pd(&data1[i], _mm256_acos_pd(_mm256_load_pd(&data1[i])));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] = std::acos(data1[i]);
			}
		}

		// Atan

		vector<double> atan()
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			vector<double> result(size);

			double* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_store_pd(&data_result[i], _mm256_atan_pd(_mm256_load_pd(&data1[i])));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = std::atan(data1[i]);
			}
			return result;
		}

		void self_atan()
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_store_pd(&data1[i], _mm256_atan_pd(_mm256_load_pd(&data1[i])));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] = std::atan(data1[i]);
			}
		}

		// Abs
		
		vector<double> abs()
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			vector<double> result(size);

			double* data_result = result._data;

			__m256d mask = _mm256_set1_pd(-0.0);

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_store_pd(&data_result[i], _mm256_andnot_pd(mask, _mm256_load_pd(&data1[i])));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = std::fabs(data1[i]);
			}
			return result;
		}
		
		void self_abs()
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			__m256d mask = _mm256_set1_pd(-0.0);

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_store_pd(&data1[i], _mm256_andnot_pd(mask, _mm256_load_pd(&data1[i])));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] = std::fabs(data1[i]);
			}
		}

		// Round

		vector<double> round()
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			vector<double> result(size);

			double* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_store_pd(&data_result[i], _mm256_round_pd(_mm256_load_pd(&data1[i]), _MM_FROUND_TO_NEAREST_INT));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = std::round(data1[i]);
			}
			return result;
		}

		void self_round()
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_store_pd(&data1[i], _mm256_round_pd(_mm256_load_pd(&data1[i]), _MM_FROUND_TO_NEAREST_INT));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] = std::round(data1[i]);
			}
		}

		// Floor

		vector<double> floor()
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			vector<double> result(size);

			double* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_store_pd(&data_result[i], _mm256_floor_pd(_mm256_load_pd(&data1[i])));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = std::floor(data1[i]);
			}
			return result;
		}

		void self_floor()
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_store_pd(&data1[i], _mm256_floor_pd(_mm256_load_pd(&data1[i])));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] = std::floor(data1[i]);
			}
		}

		// Ceil

		vector<double> floor()
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			vector<double> result(size);

			double* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_store_pd(&data_result[i], _mm256_ceil_pd(_mm256_load_pd(&data1[i])));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = std::ceil(data1[i]);
			}
			return result;
		}

		void self_floor()
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			double* data1 = this->_data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_store_pd(&data1[i], _mm256_ceil_pd(_mm256_load_pd(&data1[i])));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] = std::ceil(data1[i]);
			}
		}

		template <typename T>
		vector<T> cast()
		{
			size_t size = this->_size;

			vector<T> result(size);

			double* data1 = this->_data;

			T* data_result = result._data;

			size_t final_pos = (size / 4) * 4;

			if constexpr (std::is_same<T, uint64_t>::value)
			{
				__m256d mask = _mm256_set1_pd(ud);

				for (size_t i = 0; i < final_pos; i += 4)
				{
					_mm256_storeu_epi64(&data_result[i], double_to_uint64(_mm256_load_pd(&data1[i])));
				}
			}
			else if constexpr (std::is_same<T, int64_t>::value)
			{
				__m256d mask = _mm256_set1_pd(id);

				for (size_t i = 0; i < final_pos; i += 4)
				{
					_mm256_storeu_epi64(&data_result[i], double_to_int64(_mm256_load_pd(&data1[i])));
				}
				for (size_t i = final_pos; i < size; i++)
				{
					data_result[i] = static_cast<T>(data1[i]);
				}
			}
			else if constexpr (std::is_same<T, float>::value)
			{
				for (size_t i = 0; i < final_pos; i += 4)
				{
					_mm_store_ps(&data_result[i], _mm256_cvtpd_ps(_mm256_load_pd(&data1[i])));
				}
			}
			else if constexpr (std::is_same<T, int>::value)
			{
				for (size_t i = 0; i < final_pos; i += 4)
				{
					_mm256_storeu_epi32(&data_result[i], _mm256_cvttpd_epi32(_mm256_load_pd(&data1[i])));
				}
			}
			else if constexpr (std::is_same<T, bool>::value)
			{
				__m256d zero = _mm256_setzero_pd();

				for (size_t i = 0; i < final_pos; i += 4)
				{
					__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(_mm256_load_pd(&data1[i]), zero, _CMP_NEQ_OQ));

					__m128i mask1 = _mm256_castsi256_si128(mask);
					__m128i mask2 = _mm256_extracti128_si256(mask, 1);

					mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
					mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

					__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

					_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
				}
			}
			if constexpr (std::is_same<T, bool>::value)
			{
				for (size_t i = final_pos; i < size; i++)
				{
					data_result[i] = data1[i] == 0 ? False : True;
				}
			}
			else
			{
				for (size_t i = final_pos; i < size; i++)
				{
					data_result[i] = static_cast<T>(data1[i]);
				}
			}
			return data_result;
		}

	private:
		double* _data;
		size_t _size;
	};

	template <>
	class vector<float>
	{
	public:
		friend class vector<double>;
		friend class vector<uint64_t>;
		friend class vector<int>;

		vector(size_t size) : _size(size) {}
	private:
		float* _data;
		size_t _size;
	};

	template <>
	class vector<uint64_t>
	{
	public:
		vector(size_t size) : _data(new uint64_t[size]), _size(size) {}

		vector(uint64_t* data, size_t size) : _data(data), _size(size) {}

		friend class matrix<double>;

		friend class vector<double>;

		friend class vector<bool>;

		friend class vector<int>;

		uint64_t* data() { return this->_data; }

		size_t size() { return this->_size; }

		uint64_t& operator[](size_t index)
		{
			uint64_t* data = this->_data;
			return data[index];
		}

		const uint64_t& operator[](size_t index) const
		{
			uint64_t* data = this->_data;
			return data[index];
		}

		// +
		vector<uint64_t> operator+(vector<uint64_t>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			uint64_t* data2 = other._data;

			vector<uint64_t> result(size);

			uint64_t* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256i a = _mm256_loadu_epi64(&data1[i]);
				__m256i b = _mm256_loadu_epi64(&data2[i]);

				_mm256_storeu_epi64(&data_result[i], _mm256_add_epi64(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] + data2[i];
			}
			return result;
		}

		void operator+=(vector<uint64_t>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			uint64_t* data2 = other._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256i a = _mm256_loadu_epi64(&data1[i]);
				__m256i b = _mm256_loadu_epi64(&data2[i]);

				_mm256_storeu_epi64(&data1[i], _mm256_add_epi64(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] += data2[i];
			}
		}

		vector<double> operator+(vector<double>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			double* data2 = other._data;

			vector<double> result(size);

			double* data_result = result._data;

			__m256d mask = ud;
			double mask_d = 0x0010000000000000;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = uint64_to_double(_mm256_loadu_epi64(&data1[i]), mask);
				__m256d b = _mm256_load_pd(&data2[i]);

				_mm256_store_pd(&data_result[i], _mm256_add_pd(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				uint64_t num_i = data1[i] | reinterpret_cast<uint64_t&>(mask_d);
				double num_d = reinterpret_cast<double&>(num_i) - mask_d;
				data_result[i] = num_d + data2[i];
			}
			return result;
		}

		vector<uint64_t> operator+(vector<int>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			int* data2 = other._data;

			vector<uint64_t> result(size);

			uint64_t* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256i a = _mm256_loadu_epi64(&data1[i]);
				__m256i b = _mm256_cvtepi32_epi64(_mm_loadu_epi32(&data2[i]));
				_mm256_storeu_epi64(&data_result[i], _mm256_add_epi64(a, b));
			}
			for (size_t i = final_pos; i < size; i += 4)
			{
				data_result[i] = data1[i] + static_cast<uint64_t>(data2[i]);
			}
			return result;
		}

		void operator+=(vector<int>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			int* data2 = other._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256i a = _mm256_loadu_epi64(&data1[i]);
				__m256i b = _mm256_cvtepi32_epi64(_mm_loadu_epi32(&data2[i]));
				_mm256_storeu_epi64(&data1[i], _mm256_add_epi64(a, b));
			}
			for (size_t i = final_pos; i < size; i += 4)
			{
				data1[i] += static_cast<uint64_t>(data2[i]);
			}
		}

		vector<float> operator+(vector<float>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 8) * 8;

			uint64_t* data1 = this->_data;
			float* data2 = other._data;

			vector<float> result(size);

			float* data_result = result._data;

			__m256i indices = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);

			for (size_t i = 0; i < final_pos; i += 8)
			{
				__m256 a = _mm256_cvtepi32_ps(_mm256_insertf128_si256(_mm256_permutevar8x32_epi32(_mm256_loadu_epi64(&data1[i]), indices), _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_loadu_epi64(&data1[i + 4]), indices)), 1));
				
				__m256 b = _mm256_load_ps(&data2[i]);

				_mm256_store_ps(&data_result[i], _mm256_add_ps(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] = static_cast<float>(data1[i]) + data2[i];
			}
			return result;
		}

		// -

		vector<uint64_t> operator-(vector<uint64_t>&other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			uint64_t* data2 = other._data;

			vector<uint64_t> result(size);

			uint64_t* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256i a = _mm256_loadu_epi64(&data1[i]);
				__m256i b = _mm256_loadu_epi64(&data2[i]);

				_mm256_storeu_epi64(&data_result[i], _mm256_sub_epi64(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] - data2[i];
			}
			return result;
		}

		void operator-=(vector<uint64_t>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			uint64_t* data2 = other._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256i a = _mm256_loadu_epi64(&data1[i]);
				__m256i b = _mm256_loadu_epi64(&data2[i]);

				_mm256_storeu_epi64(&data1[i], _mm256_sub_epi64(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] -= data2[i];
			}
		}

		vector<double> operator-(vector<double>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			double* data2 = other._data;

			vector<double> result(size);

			double* data_result = result._data;

			__m256d mask = ud;

			double mask_d = 0x0010000000000000;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = uint64_to_double(_mm256_loadu_epi64(&data1[i]), mask);
				__m256d b = _mm256_load_pd(&data2[i]);

				_mm256_store_pd(&data_result[i], _mm256_sub_pd(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				uint64_t num_i = data1[i] | reinterpret_cast<uint64_t&>(mask_d);
				double num_d = reinterpret_cast<double&>(num_i) - mask_d;
				data_result[i] = num_d - data2[i];
			}
			return result;
		}

		vector<uint64_t> operator-(vector<int>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			int* data2 = other._data;

			vector<uint64_t> result(size);

			uint64_t* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256i a = _mm256_loadu_epi64(&data1[i]);
				__m256i b = _mm256_cvtepi32_epi64(_mm_loadu_epi32(&data2[i]));
				_mm256_storeu_epi64(&data_result[i], _mm256_sub_epi64(a, b));
			}
			for (size_t i = final_pos; i < size; i += 4)
			{
				data_result[i] = data1[i] - static_cast<uint64_t>(data2[i]);
			}
			return result;
		}

		void operator-=(vector<int>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			int* data2 = other._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256i a = _mm256_loadu_epi64(&data1[i]);
				__m256i b = _mm256_cvtepi32_epi64(_mm_loadu_epi32(&data2[i]));
				_mm256_storeu_epi64(&data1[i], _mm256_sub_epi64(a, b));
			}
			for (size_t i = final_pos; i < size; i += 4)
			{
				data1[i] -= static_cast<uint64_t>(data2[i]);
			}
		}

		vector<float> operator-(vector<float>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 8) * 8;

			uint64_t* data1 = this->_data;
			float* data2 = other._data;

			vector<float> result(size);

			float* data_result = result._data;

			__m256i indices = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);

			for (size_t i = 0; i < final_pos; i += 8)
			{
				__m256 a = _mm256_cvtepi32_ps(_mm256_insertf128_si256(_mm256_permutevar8x32_epi32(_mm256_loadu_epi64(&data1[i]), indices), _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_loadu_epi64(&data1[i + 4]), indices)), 1));

				__m256 b = _mm256_load_ps(&data2[i]);

				_mm256_store_ps(&data_result[i], _mm256_sub_ps(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] = static_cast<float>(data1[i]) - data2[i];
			}
			return result;
		}

		// *

		vector<uint64_t> operator*(vector<uint64_t>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			uint64_t* data2 = other._data;

			vector<uint64_t> result(size);

			uint64_t* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256i a = _mm256_loadu_epi64(&data1[i]);
				__m256i b = _mm256_loadu_epi64(&data2[i]);

				_mm256_storeu_epi64(&data_result[i], _mm256_mul_epu32(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] * data2[i];
			}
			return result;
		}

		void operator*=(vector<uint64_t>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			uint64_t* data2 = other._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256i a = _mm256_loadu_epi64(&data1[i]);
				__m256i b = _mm256_loadu_epi64(&data2[i]);

				_mm256_storeu_epi64(&data1[i], _mm256_mul_epu32(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] *= data2[i];
			}
		}

		vector<double> operator*(vector<double>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			double* data2 = other._data;

			vector<double> result(size);

			double* data_result = result._data;

			__m256d mask = ud;

			double mask_d = 0x0010000000000000;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = uint64_to_double(_mm256_loadu_epi64(&data1[i]), mask);
				__m256d b = _mm256_load_pd(&data2[i]);

				_mm256_store_pd(&data_result[i], _mm256_mul_pd(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				uint64_t num_i = data1[i] | reinterpret_cast<uint64_t&>(mask_d);
				double num_d = reinterpret_cast<double&>(num_i) - mask_d;
				data_result[i] = num_d * data2[i];

			}
			return result;
		}

		vector<uint64_t> operator*(vector<int>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			int* data2 = other._data;

			vector<uint64_t> result(size);

			uint64_t* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256i a = _mm256_loadu_epi64(&data1[i]);
				__m256i b = _mm256_cvtepi32_epi64(_mm_loadu_epi32(&data2[i]));
				_mm256_storeu_epi64(&data_result[i], _mm256_mul_epu32(a, b));
			}
			for (size_t i = final_pos; i < size; i += 4)
			{
				data_result[i] = data1[i] * static_cast<uint64_t>(data2[i]);
			}
			return result;
		}

		void operator*=(vector<int>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			int* data2 = other._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256i a = _mm256_loadu_epi64(&data1[i]);
				__m256i b = _mm256_cvtepi32_epi64(_mm_loadu_epi32(&data2[i]));
				_mm256_storeu_epi64(&data1[i], _mm256_mul_epu32(a, b));
			}
			for (size_t i = final_pos; i < size; i += 4)
			{
				data1[i] *= static_cast<uint64_t>(data2[i]);
			}
		}

		vector<float> operator*(vector<float>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 8) * 8;

			uint64_t* data1 = this->_data;
			float* data2 = other._data;

			vector<float> result(size);

			float* data_result = result._data;

			__m256i indices = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);

			for (size_t i = 0; i < final_pos; i += 8)
			{
				__m256 a = _mm256_cvtepi32_ps(_mm256_insertf128_si256(_mm256_permutevar8x32_epi32(_mm256_loadu_epi64(&data1[i]), indices), _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_loadu_epi64(&data1[i + 4]), indices)), 1));

				__m256 b = _mm256_load_ps(&data2[i]);

				_mm256_store_ps(&data_result[i], _mm256_mul_ps(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] = static_cast<float>(data1[i]) * data2[i];
			}
			return result;
		}

		// /

		vector<uint64_t> operator/(vector<uint64_t>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			uint64_t* data2 = other._data;

			vector<uint64_t> result(size);

			uint64_t* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256i a = _mm256_loadu_epi64(&data1[i]);
				__m256i b = _mm256_loadu_epi64(&data2[i]);

				_mm256_storeu_epi64(&data_result[i], _mm256_div_epi64(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] / data2[i];
			}
			return result;
		}

		void operator/=(vector<uint64_t>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			uint64_t* data2 = other._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256i a = _mm256_loadu_epi64(&data1[i]);
				__m256i b = _mm256_loadu_epi64(&data2[i]);

				_mm256_storeu_epi64(&data1[i], _mm256_div_epi64(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] /= data2[i];
			}
		}

		vector<double> operator/(vector<double>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			double* data2 = other._data;

			vector<double> result(size);

			double* data_result = result._data;

			__m256d mask = ud;

			double mask_d = 0x0010000000000000;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = uint64_to_double(_mm256_loadu_epi64(&data1[i]), mask);
				__m256d b = _mm256_load_pd(&data2[i]);

				_mm256_store_pd(&data_result[i], _mm256_div_pd(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				uint64_t num_i = data1[i] | reinterpret_cast<uint64_t&>(mask_d);
				double num_d = reinterpret_cast<double&>(num_i) - mask_d;
				data_result[i] = num_d / data2[i];
			}
			return result;
		}

		vector<uint64_t> operator/(vector<int>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			int* data2 = other._data;

			vector<uint64_t> result(size);

			uint64_t* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256i a = _mm256_loadu_epi64(&data1[i]);
				__m256i b = _mm256_cvtepi32_epi64(_mm_loadu_epi32(&data2[i]));
				_mm256_storeu_epi64(&data_result[i], _mm256_div_epi64(a, b));
			}
			for (size_t i = final_pos; i < size; i += 4)
			{
				data_result[i] = data1[i] / static_cast<uint64_t>(data2[i]);
			}
			return result;
		}

		void operator/=(vector<int>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			int* data2 = other._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256i a = _mm256_loadu_epi64(&data1[i]);
				__m256i b = _mm256_cvtepi32_epi64(_mm_loadu_epi32(&data2[i]));
				_mm256_storeu_epi64(&data1[i], _mm256_div_epi64(a, b));
			}
			for (size_t i = final_pos; i < size; i += 4)
			{
				data1[i] /= static_cast<uint64_t>(data2[i]);
			}
		}

		vector<float> operator/(vector<float>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 8) / 8;

			uint64_t* data1 = this->_data;
			float* data2 = other._data;

			vector<float> result(size);

			float* data_result = result._data;

			__m256i indices = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);

			for (size_t i = 0; i < final_pos; i += 8)
			{
				__m256 a = _mm256_cvtepi32_ps(_mm256_insertf128_si256(_mm256_permutevar8x32_epi32(_mm256_loadu_epi64(&data1[i]), indices), _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_loadu_epi64(&data1[i + 4]), indices)), 1));

				__m256 b = _mm256_load_ps(&data2[i]);

				_mm256_store_ps(&data_result[i], _mm256_div_ps(a, b));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data1[i] = static_cast<float>(data1[i]) / data2[i];
			}
			return result;
		}

		// ===

		vector<bool> operator==(vector<uint64_t>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			uint64_t* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256i a = _mm256_loadu_epi64(&data1[i]);
				__m256i b = _mm256_loadu_epi64(&data2[i]);
				
				__m256i mask = _mm256_cmpeq_epi64(a, b);

				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] == data2[i] ? True : False;
			}
			return result;
		}

		vector<bool> operator==(uint64_t num)
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;

			vector<bool> result(size);

			bool* data_result = result._data;

			__m256i b = _mm256_set1_epi64x(num);

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256i a = _mm256_loadu_epi64(&data1[i]);

				__m256i mask = _mm256_cmpeq_epi64(a, b);

				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] == num ? True : False;
			}
			return result;
		}

		vector<bool> operator==(vector<double>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			double* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			__m256d mask = ud;

			double mask_d = 0x0010000000000000;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = uint64_to_double(_mm256_loadu_epi64(&data1[i]), mask);
				__m256d b = _mm256_load_pd(&data2[i]);

				__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_EQ_OQ));
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				uint64_t num_i = data1[i] | reinterpret_cast<uint64_t&>(mask_d);
				double num_d = reinterpret_cast<double&>(num_i) - mask_d;
				data_result[i] = num_d == data2[i] ? True : False;
			}
			return result;
		}

		vector<bool> operator==(vector<int>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			int* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256i a = _mm256_loadu_epi64(&data1[i]);
				__m256i b = _mm256_cvtepi32_epi64(_mm_loadu_epi32(&data2[i]));

				__m256i mask = _mm256_cmpeq_epi64(a, b);
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] == data2[i] ? True : False;
			}
			return result;
		}

		vector<bool> operator==(vector<float>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 8) * 8;

			uint64_t* data1 = this->_data;
			float* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			__m256i indices = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);

			__m256i indices_mask = _mm256_setr_epi32(0, 7, 2, 3, 4, 5, 6, 1);

			for (size_t i = 0; i < final_pos; i += 8)
			{
				__m256i a1 = _mm256_permutevar8x32_epi32(_mm256_loadu_epi64(&data1[i]), indices);
				__m128i a2 = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_loadu_epi64(&data1[i + 4]), indices));
				__m256 a = _mm256_castsi256_ps(_mm256_inserti128_si256(a1, a2, 1));
				__m256 b = _mm256_load_ps(&data2[i]);

				__m256i mask = _mm256_castps_si256(_mm256_cmp_ps(a, b, _CMP_EQ_OQ));
				
				mask = _mm256_packs_epi32(mask, mask);
				mask = _mm256_packs_epi16(mask, mask);
				mask = _mm256_permutevar8x32_epi32(mask, indices_mask);

				_mm_store_sd(reinterpret_cast<double*>(&data_result[i]), _mm_castsi128_pd(_mm256_castsi256_si128(mask)));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = static_cast<float>(data1[i]) == data2[i] ? True : False;
			}
			return result;
		}

		// !=

		vector<bool> operator!=(vector<uint64_t>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			uint64_t* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			__m256i minus_ones = _mm256_set1_epi64x(-1);

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256i a = _mm256_loadu_epi64(&data1[i]);
				__m256i b = _mm256_loadu_epi64(&data2[i]);
				
				__m256i mask = _mm256_andnot_si256(_mm256_cmpeq_epi64(a, b), minus_ones);
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] != data2[i] ? True : False;
			}
			return result;
		}

		vector<bool> operator!=(uint64_t num)
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;

			vector<bool> result(size);

			bool* data_result = result._data;

			__m256i b = _mm256_set1_epi64x(num);

			__m256i minus_ones = _mm256_set1_epi64x(-1);

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256i a = _mm256_loadu_epi64(&data1[i]);

				__m256i mask = _mm256_andnot_si256((_mm256_cmpeq_epi64(a, b)), minus_ones);
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] != num ? True : False;
			}
			return result;
		}

		vector<bool> operator!=(vector<double>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			double* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			__m256d mask = ud;

			double mask_d = 0x0010000000000000;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = uint64_to_double(_mm256_loadu_epi64(&data1[i]), mask);
				__m256d b = _mm256_load_pd(&data2[i]);

				__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_NEQ_OQ));
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				uint64_t num_i = data1[i] | reinterpret_cast<uint64_t&>(mask_d);
				double num_d = reinterpret_cast<double&>(num_i) - mask_d;
				data_result[i] = num_d != data2[i] ? True : False;

			}
			return result;
		}

		vector<bool> operator!=(vector<int>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			int* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			__m256i minus_ones = _mm256_set1_epi64x(-1);

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256i a = _mm256_loadu_epi64(&data1[i]);
				__m256i b = _mm256_cvtepi32_epi64(_mm_loadu_epi32(&data2[i]));

				__m256i mask = _mm256_andnot_si256(_mm256_cmpeq_epi64(a, b), minus_ones);
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] != data2[i] ? True : False;
			}
			return result;
		}

		vector<bool> operator!=(vector<float>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 8) * 8;

			uint64_t* data1 = this->_data;
			float* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			__m256i indices = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);

			__m256i indices_mask = _mm256_setr_epi32(0, 7, 2, 3, 4, 5, 6, 1);

			for (size_t i = 0; i < final_pos; i += 8)
			{
				__m256i a1 = _mm256_permutevar8x32_epi32(_mm256_loadu_epi64(&data1[i]), indices);
				__m128i a2 = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_loadu_epi64(&data1[i + 4]), indices));
				__m256 a = _mm256_castsi256_ps(_mm256_inserti128_si256(a1, a2, 1));
				__m256 b = _mm256_load_ps(&data2[i]);

				__m256i mask = _mm256_castps_si256(_mm256_cmp_ps(a, b, _CMP_NEQ_OQ));

				mask = _mm256_packs_epi32(mask, mask);
				mask = _mm256_packs_epi16(mask, mask);
				mask = _mm256_permutevar8x32_epi32(mask, indices_mask);

				_mm_store_sd(reinterpret_cast<double*>(&data_result[i]), _mm_castsi128_pd(_mm256_castsi256_si128(mask)));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = static_cast<float>(data1[i]) != data2[i] ? True : False;
			}
			return result;
		}

		// >

		vector<bool> operator>(vector<uint64_t>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			uint64_t* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256i a = _mm256_loadu_epi64(&data1[i]);
				__m256i b = _mm256_loadu_epi64(&data2[i]);

				__m256i mask = _mm256_cmpgt_epi64(a, b);
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] > data2[i] ? True : False;
			}
			return result;
		}

		vector<bool> operator>(uint64_t num)
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;

			vector<bool> result(size);

			bool* data_result = result._data;

			__m256i b = _mm256_set1_epi64x(num);

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256i a = _mm256_loadu_epi64(&data1[i]);

				__m256i mask = _mm256_cmpgt_epi64(a, b);
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] > num ? True : False;
			}
			return result;
		}

		vector<bool> operator>(vector<double>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			double* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			__m256d mask = ud;

			double mask_d = 0x0010000000000000;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = uint64_to_double(_mm256_loadu_epi64(&data1[i]), mask);
				__m256d b = _mm256_load_pd(&data2[i]);

				__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GT_OQ));
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				uint64_t num_i = data1[i] | reinterpret_cast<uint64_t&>(mask_d);
				double num_d = reinterpret_cast<double&>(num_i) - mask_d;
				data_result[i] = num_d > data2[i] ? True : False;

			}
			return result;
		}

		vector<bool> operator>(vector<int>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			int* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256i a = _mm256_loadu_epi64(&data1[i]);
				__m256i b = _mm256_cvtepi32_epi64(_mm_loadu_epi32(&data2[i]));

				__m256i mask = _mm256_cmpgt_epi64(a, b);
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] > static_cast<uint64_t>(data2[i]) ? True : False;
			}
			return result;
		}

		vector<bool> operator>(vector<float>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 8) * 8;

			uint64_t* data1 = this->_data;
			float* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			__m256i indices = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);

			__m256i indices_mask = _mm256_setr_epi32(0, 7, 2, 3, 4, 5, 6, 1);

			for (size_t i = 0; i < final_pos; i += 8)
			{
				__m256i a1 = _mm256_permutevar8x32_epi32(_mm256_loadu_epi64(&data1[i]), indices);
				__m128i a2 = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_loadu_epi64(&data1[i + 4]), indices));
				__m256 a = _mm256_castsi256_ps(_mm256_inserti128_si256(a1, a2, 1));
				__m256 b = _mm256_load_ps(&data2[i]);

				__m256i mask = _mm256_castps_si256(_mm256_cmp_ps(a, b, _CMP_GT_OQ));

				mask = _mm256_packs_epi32(mask, mask);
				mask = _mm256_packs_epi16(mask, mask);
				mask = _mm256_permutevar8x32_epi32(mask, indices_mask);

				_mm_store_sd(reinterpret_cast<double*>(&data_result[i]), _mm_castsi128_pd(_mm256_castsi256_si128(mask)));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = static_cast<float>(data1[i]) > data2[i] ? True : False;
			}
			return result;
		}

		// < 

		vector<bool> operator<(vector<uint64_t>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			uint64_t* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			__m256i minus_ones = _mm256_set1_epi64x(-1);

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256i a = _mm256_loadu_epi64(&data1[i]);
				__m256i b = _mm256_loadu_epi64(&data2[i]);

				__m256i gt = _mm256_cmpgt_epi64(a, b);
				__m256i eq = _mm256_cmpeq_epi64(a, b);

				__m256i mask = _mm256_andnot_si256(gt, _mm256_andnot_si256(eq, minus_ones));
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] < data2[i] ? True : False;
			}
			return result;
		}

		vector<bool> operator<(uint64_t num)
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;

			vector<bool> result(size);

			bool* data_result = result._data;

			__m256i b = _mm256_set1_epi64x(num);

			__m256i minus_ones = _mm256_set1_epi64x(-1);

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256i a = _mm256_loadu_epi64(&data1[i]);

				__m256i gt = _mm256_cmpgt_epi64(a, b);
				__m256i eq = _mm256_cmpeq_epi64(a, b);

				__m256i mask = _mm256_andnot_si256(gt, _mm256_andnot_si256(eq, minus_ones));

				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] < num ? True : False;
			}
			return result;
		}

		vector<bool> operator<(vector<double>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			double* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			__m256d mask = ud;

			double mask_d = 0x0010000000000000;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = uint64_to_double(_mm256_loadu_epi64(&data1[i]), mask);
				__m256d b = _mm256_load_pd(&data2[i]);

				__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LT_OQ));
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				uint64_t num_i = data1[i] | reinterpret_cast<uint64_t&>(mask_d);
				double num_d = reinterpret_cast<double&>(num_i) - mask_d;
				data_result[i] = num_d < data2[i] ? True : False;

			}
			return result;
		}

		vector<bool> operator<(vector<int>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			int* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			__m256i minus_ones = _mm256_set1_epi64x(-1);

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256i a = _mm256_loadu_epi64(&data1[i]);
				__m256i b = _mm256_cvtepi32_epi64(_mm_loadu_epi32(&data2[i]));

				__m256i gt = _mm256_cmpgt_epi64(a, b);
				__m256i eq = _mm256_cmpeq_epi64(a, b);

				__m256i mask = _mm256_andnot_si256(gt, _mm256_andnot_si256(eq, minus_ones));
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] < static_cast<uint64_t>(data2[i]) ? True : False;
			}
			return result;
		}

		vector<bool> operator<(vector<float>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 8) * 8;

			uint64_t* data1 = this->_data;
			float* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			__m256i indices = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);

			__m256i indices_mask = _mm256_setr_epi32(0, 7, 2, 3, 4, 5, 6, 1);

			for (size_t i = 0; i < final_pos; i += 8)
			{
				__m256i a1 = _mm256_permutevar8x32_epi32(_mm256_loadu_epi64(&data1[i]), indices);
				__m128i a2 = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_loadu_epi64(&data1[i + 4]), indices));
				__m256 a = _mm256_castsi256_ps(_mm256_inserti128_si256(a1, a2, 1));
				__m256 b = _mm256_load_ps(&data2[i]);

				__m256i mask = _mm256_castps_si256(_mm256_cmp_ps(a, b, _CMP_LT_OQ));

				mask = _mm256_packs_epi32(mask, mask);
				mask = _mm256_packs_epi16(mask, mask);
				mask = _mm256_permutevar8x32_epi32(mask, indices_mask);

				_mm_store_sd(reinterpret_cast<double*>(&data_result[i]), _mm_castsi128_pd(_mm256_castsi256_si128(mask)));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = static_cast<float>(data1[i]) < data2[i] ? True : False;
			}
			return result;
		}

		// >=

		vector<bool> operator>=(vector<uint64_t>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			uint64_t* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256i a = _mm256_loadu_epi64(&data1[i]);
				__m256i b = _mm256_loadu_epi64(&data2[i]);

				__m256i gt = _mm256_cmpgt_epi64(a, b);
				__m256i eq = _mm256_cmpeq_epi64(a, b);

				__m256i mask = _mm256_or_si256(gt, eq);
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] >= data2[i] ? True : False;
			}
			return result;
		}

		vector<bool> operator>=(uint64_t num)
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;

			vector<bool> result(size);

			bool* data_result = result._data;

			__m256i b = _mm256_set1_epi64x(num);

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256i a = _mm256_loadu_epi64(&data1[i]);

				__m256i gt = _mm256_cmpgt_epi64(a, b);
				__m256i eq = _mm256_cmpeq_epi64(a, b);

				__m256i mask = _mm256_or_si256(gt, eq);
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] >= num ? True : False;
			}
			return result;
		}

		vector<bool> operator>=(vector<double>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			double* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			__m256d mask = ud;

			double mask_d = 0x0010000000000000;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = uint64_to_double(_mm256_loadu_epi64(&data1[i]), mask);
				__m256d b = _mm256_load_pd(&data2[i]);

				__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GE_OQ));
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				uint64_t num_i = data1[i] | reinterpret_cast<uint64_t&>(mask_d);
				double num_d = reinterpret_cast<double&>(num_i) - mask_d;
				data_result[i] = num_d >= data2[i] ? True : False;

			}
			return result;
		}

		vector<bool> operator>=(vector<int>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			int* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256i a = _mm256_loadu_epi64(&data1[i]);
				__m256i b = _mm256_cvtepi32_epi64(_mm_loadu_epi32(&data2[i]));

				__m256i gt = _mm256_cmpgt_epi64(a, b);
				__m256i eq = _mm256_cmpeq_epi64(a, b);

				__m256i mask = _mm256_or_si256(gt, eq);
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] >= data2[i] ? True : False;
			}
			return result;
		}

		vector<bool> operator>=(vector<float>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 8) * 8;

			uint64_t* data1 = this->_data;
			float* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			__m256i indices = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);

			for (size_t i = 0; i < final_pos; i += 8)
			{
				__m256 a = _mm256_cvtepi32_ps(_mm256_insertf128_si256(_mm256_permutevar8x32_epi32(_mm256_loadu_epi64(&data1[i]), indices), _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_loadu_epi64(&data1[i + 4]), indices)), 1));
				__m256 b = _mm256_load_ps(&data2[i]);

				__m256i mask = _mm256_castps_si256(_mm256_cmp_ps(a, b, _CMP_GE_OQ));
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = static_cast<float>(data1[i]) >= data2[i] ? True : False;
			}
			return result;
		}

		// <=

		vector<bool> operator<=(vector<uint64_t>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			uint64_t* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			__m256i minus_ones = _mm256_set1_epi64x(-1);

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256i a = _mm256_loadu_epi64(&data1[i]);
				__m256i b = _mm256_loadu_epi64(&data2[i]);

				__m256i mask = _mm256_andnot_si256(_mm256_cmpgt_epi64(a, b), minus_ones);
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] <= data2[i] ? True : False;
			}
			return result;
		}

		vector<bool> operator<=(uint64_t num)
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;

			vector<bool> result(size);

			bool* data_result = result._data;

			__m256i b = _mm256_set1_epi64x(num);

			__m256i minus_ones = _mm256_set1_epi64x(-1);

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256i a = _mm256_loadu_epi64(&data1[i]);
				
				__m256i mask = _mm256_andnot_si256(_mm256_cmpgt_epi64(a, b), minus_ones);
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] <= num ? True : False;
			}
			return result;
		}

		vector<bool> operator<=(vector<double>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			double* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			__m256d mask = ud;

			double mask_d = 0x0010000000000000;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = uint64_to_double(_mm256_loadu_epi64(&data1[i]), mask);
				__m256d b = _mm256_load_pd(&data2[i]);

				__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LE_OQ));
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				uint64_t num_i = data1[i] | reinterpret_cast<uint64_t&>(mask_d);
				double num_d = reinterpret_cast<double&>(num_i) - mask_d;
				data_result[i] = num_d <= data2[i] ? True : False;

			}
			return result;
		}

		vector<bool> operator<=(vector<int>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			int* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			__m256i minus_ones = _mm256_set1_epi64x(-1);

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256i a = _mm256_loadu_epi64(&data1[i]);
				__m256i b = _mm256_cvtepi32_epi64(_mm_loadu_epi32(&data2[i]));

				__m256i mask = _mm256_andnot_si256(_mm256_cmpgt_epi64(a, b), minus_ones);
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = data1[i] <= data2[i] ? True : False;
			}
			return result;
		}

		vector<bool> operator<=(vector<float>& other)
		{
#ifdef _DEBUG
			if (this->_size == other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;

			size_t final_pos = (size / 8) * 8;

			uint64_t* data1 = this->_data;
			float* data2 = other._data;

			vector<bool> result(size);

			bool* data_result = result._data;

			__m256i indices = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);

			for (size_t i = 0; i < final_pos; i += 8)
			{
				__m256 a = _mm256_cvtepi32_ps(_mm256_insertf128_si256(_mm256_permutevar8x32_epi32(_mm256_loadu_epi64(&data1[i]), indices), _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_loadu_epi64(&data1[i + 4]), indices)), 1));
				__m256 b = _mm256_load_ps(&data2[i]);

				__m256i mask = _mm256_castps_si256(_mm256_cmp_ps(a, b, _CMP_LE_OQ));
				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = static_cast<float>(data1[i]) <= data2[i] ? True : False;

			}
			return result;
		}

		// Functions

		// Pow

		vector<uint64_t> pow(uint64_t exponent)
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;

			vector<uint64_t> result(size);

			uint64_t* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256i result_pow = _mm256_set1_epi64x(1);
				__m256i base = _mm256_loadu_epi64(&data1[i]);
				uint64_t exp = exponent;
				while (exp > 0) {
					if (exp % 2 == 1) {
						result_pow = _mm256_mul_epu32(result_pow, base);
					}

					base = _mm256_mul_epu32(base, base);
					exp >>= 1;
				}
				_mm256_storeu_epi64(&data_result[i], result_pow);
			}
			for (size_t i = final_pos; i < size; i++)
			{
				uint64_t result_pow = 1;
				uint64_t base = data1[i];
				uint64_t exp = exponent;
				while (exp > 0) {
					if (exp % 2 == 1) {
						result_pow = result_pow * base;
					}

					base = base * base;
					exp >>= 1;
				}
				data_result[i] = result_pow;
			}
			return result;
		}

		vector<uint64_t> pow(vector<uint64_t>& other)
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			uint64_t* data2 = other._data;

			vector<uint64_t> result(size);

			uint64_t* data_result = result._data;

			__m256i one = _mm256_set1_epi64x(1);
			__m256i zero = _mm256_setzero_si256();

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256i result_pow = _mm256_set1_epi64x(1);

				__m256i base = _mm256_loadu_epi64(&data1[i]);
				__m256i exps = _mm256_loadu_epi64(&data2[i]);

				while (_mm256_movemask_epi8(_mm256_cmpgt_epi64(exps, zero))) {
					__m256i mask = _mm256_cmpeq_epi64(_mm256_and_si256(exps, one), one);
					result_pow = _mm256_blendv_epi8(result_pow, _mm256_mul_epu32(result_pow, base), mask);
					base = _mm256_mul_epu32(base, base);
					exps = _mm256_srli_epi64(exps, 1);
				}
				_mm256_storeu_epi64(&data_result[i], result_pow);
			}
			for (size_t i = final_pos; i < size; i++)
			{
				uint64_t result_pow = 1;
				uint64_t base = data1[i];
				uint64_t exp = data2[i];
				while (exp > 0) {
					if (exp % 2 == 1) {
						result_pow = result_pow * base;
					}

					base = base * base;
					exp >>= 1;
				}
				data_result[i] = result_pow;
			}
			return result;
		}

		vector<double> pow(vector<double>& other)
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			double* data2 = other._data;

			vector<double> result(size);

			double* data_result = result._data;

			__m256d mask = ud;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d base = uint64_to_double(_mm256_loadu_epi64(&data1[i]), mask);
				__m256d exps = _mm256_load_pd(&data2[i]);

				
				_mm256_store_pd(&data_result[i], _mm256_pow_pd(base, exps));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = std::pow(static_cast<double>(data1[i]), data2[i]);
			}
			return result;
		}

		vector<double> pow(double exponent)
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;

			vector<double> result(size);

			double* data_result = result._data;

			__m256d mask = ud;

			__m256d _exponent = _mm256_set1_pd(exponent);

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d base = uint64_to_double(_mm256_loadu_epi64(&data1[i]), mask);

				_mm256_store_pd(&data_result[i], _mm256_pow_pd(base, _exponent));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = std::pow(static_cast<double>(data1[i]), exponent);
			}
			return result;
		}

		vector<uint64_t> pow(vector<int>& other)
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			int* data2 = other._data;

			vector<uint64_t> result(size);

			uint64_t* data_result = result._data;

			__m256i one = _mm256_set1_epi64x(1);
			__m256i zero = _mm256_setzero_si256();

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256i result_pow = _mm256_set1_epi64x(1);

				__m256i base = _mm256_loadu_epi64(&data1[i]);
				__m256i exps = _mm256_cvtepi32_epi64(_mm_loadu_epi32(&data2[i]));

				while (_mm256_movemask_epi8(_mm256_cmpgt_epi64(exps, zero))) {
					__m256i mask = _mm256_cmpeq_epi64(_mm256_and_si256(exps, one), one);
					result_pow = _mm256_blendv_epi8(result_pow, _mm256_mul_epu32(result_pow, base), mask);
					base = _mm256_mul_epu32(base, base);
					exps = _mm256_srli_epi64(exps, 1);
				}
				_mm256_storeu_epi64(&data_result[i], result_pow);
			}
			for (size_t i = final_pos; i < size; i++)
			{
				uint64_t result_pow = 1;
				uint64_t base = data1[i];
				uint64_t exp = static_cast<uint64_t>(data2[i]);
				while (exp > 0) {
					if (exp % 2 == 1) {
						result_pow = result_pow * base;
					}

					base = base * base;
					exp >>= 1;
				}
				data_result[i] = result_pow;
			}
			return result;
		}

		void self_pow(uint64_t exponent)
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256i result_pow = _mm256_set1_epi64x(1);
				__m256i base = _mm256_loadu_epi64(&data1[i]);
				uint64_t exp = exponent;
				while (exp > 0) {
					if (exp % 2 == 1) {
						result_pow = _mm256_mul_epu32(result_pow, base);
					}

					base = _mm256_mul_epu32(base, base);
					exp >>= 1;
				}
				_mm256_storeu_epi64(&data1[i], result_pow);
			}
			for (size_t i = final_pos; i < size; i++)
			{
				uint64_t result_pow = 1;
				uint64_t base = data1[i];
				uint64_t exp = exponent;
				while (exp > 0) {
					if (exp % 2 == 1) {
						result_pow = result_pow * base;
					}

					base = base * base;
					exp >>= 1;
				}
				data1[i] = result_pow;
			}
		}

		void self_pow(vector<uint64_t>& other)
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			uint64_t* data2 = other._data;

			__m256i one = _mm256_set1_epi64x(1);
			__m256i zero = _mm256_setzero_si256();

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256i result_pow = _mm256_set1_epi64x(1);

				__m256i base = _mm256_loadu_epi64(&data1[i]);
				__m256i exps = _mm256_loadu_epi64(&data2[i]);

				while (_mm256_movemask_epi8(_mm256_cmpgt_epi64(exps, zero))) {
					__m256i mask = _mm256_cmpeq_epi64(_mm256_and_si256(exps, one), one);
					result_pow = _mm256_blendv_epi8(result_pow, _mm256_mul_epu32(result_pow, base), mask);
					base = _mm256_mul_epu32(base, base);
					exps = _mm256_srli_epi64(exps, 1);
				}
				_mm256_storeu_epi64(&data1[i], result_pow);
			}
			for (size_t i = final_pos; i < size; i++)
			{
				uint64_t result_pow = 1;
				uint64_t base = data1[i];
				uint64_t exp = data2[i];
				while (exp > 0) {
					if (exp % 2 == 1) {
						result_pow = result_pow * base;
					}

					base = base * base;
					exp >>= 1;
				}
				data1[i] = result_pow;
			}
		}

		void self_pow(vector<int>& other)
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			int* data2 = other._data;

			__m256i one = _mm256_set1_epi64x(1);
			__m256i zero = _mm256_setzero_si256();

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256i result_pow = _mm256_set1_epi64x(1);

				__m256i base = _mm256_loadu_epi64(&data1[i]);
				__m256i exps = _mm256_cvtepi32_epi64(_mm_loadu_epi32(&data2[i]));

				while (_mm256_movemask_epi8(_mm256_cmpgt_epi64(exps, zero))) {
					__m256i mask = _mm256_cmpeq_epi64(_mm256_and_si256(exps, one), one);
					result_pow = _mm256_blendv_epi8(result_pow, _mm256_mul_epu32(result_pow, base), mask);
					base = _mm256_mul_epu32(base, base);
					exps = _mm256_srli_epi64(exps, 1);
				}
				_mm256_storeu_epi64(&data1[i], result_pow);
			}
			for (size_t i = final_pos; i < size; i++)
			{
				uint64_t result_pow = 1;
				uint64_t base = data1[i];
				uint64_t exp = static_cast<uint64_t>(data2[i]);
				while (exp > 0) {
					if (exp % 2 == 1) {
						result_pow = result_pow * base;
					}

					base = base * base;
					exp >>= 1;
				}
				data1[i] = result_pow;
			}
		}

		// Root

		vector<double> root(double index)
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;

			vector<double> result(size);

			double* data_result = result._data;

			__m256d mask = ud;

			index = 1 / index;

			__m256d _indices = _mm256_set1_pd(index);

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d base = uint64_to_double(_mm256_loadu_epi64(&data1[i]), mask);

				_mm256_store_pd(&data_result[i], _mm256_pow_pd(base, _indices));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = std::pow(static_cast<double>(data1[i]), index);
			}
			return result;
		}

		vector<double> root(vector<uint64_t>& other)
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			uint64_t* data2 = other._data;

			vector<double> result(size);

			double* data_result = result._data;

			__m256d mask = _mm256_set1_pd(0x0010000000000000);

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d base = uint64_to_double(_mm256_loadu_epi64(&data1[i]), mask);
				__m256d exps = uint64_to_double(_mm256_loadu_epi64(&data2[i]), mask);


				_mm256_store_pd(&data_result[i], _mm256_pow_pd(base, exps));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = std::pow(static_cast<double>(data1[i]), static_cast<double>(data2[i]));
			}
			return result;
		}

		vector<double> root(vector<int>& other)
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			int* data2 = other._data;

			vector<double> result(size);

			double* data_result = result._data;

			__m256d mask = _mm256_set1_pd(0x0010000000000000);

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d base = uint64_to_double(_mm256_loadu_epi64(&data1[i]), mask);
				__m256d exps = _mm256_cvtepi32_pd(_mm_loadu_epi32(&data2[i]));


				_mm256_store_pd(&data_result[i], _mm256_pow_pd(base, exps));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = std::pow(static_cast<double>(data1[i]), static_cast<double>(data2[i]));
			}
			return result;
		}

		vector<double> sqrt()
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;

			vector<double> result(size);

			double* data_result = result._data;

			__m256d mask = ud;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d base = uint64_to_double(_mm256_loadu_epi64(&data1[i]), mask);

				_mm256_store_pd(&data_result[i], _mm256_sqrt_pd(base));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = std::sqrt(static_cast<double>(data1[i]));
			}
			return result;
		}

		// Log

		vector<double> log()
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;

			vector<double> result(size);

			double* data_result = result._data;

			__m256d mask = ud;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = uint64_to_double(_mm256_loadu_epi64(&data1[i]), mask);
				_mm256_store_pd(&data_result[i], _mm256_log_pd(a));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = std::log(static_cast<double>(data1[i]));
			}
			return result;
		}

		vector<double> log2()
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;

			vector<double> result(size);

			double* data_result = result._data;

			__m256d mask = ud;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = uint64_to_double(_mm256_loadu_epi64(&data1[i]), mask);
				_mm256_store_pd(&data_result[i], _mm256_log2_pd(a));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = std::log2(static_cast<double>(data1[i]));
			}
			return result;
		}

		vector<double> log10()
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;

			vector<double> result(size);

			double* data_result = result._data;

			__m256d mask = ud;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = uint64_to_double(_mm256_loadu_epi64(&data1[i]), mask);
				_mm256_store_pd(&data_result[i], _mm256_log10_pd(a));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = std::log10(static_cast<double>(data1[i]));
			}
			return result;
		}

		// Exp

		vector<double> exp()
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;

			vector<double> result(size);

			double* data_result = result._data;

			__m256d mask = ud;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = uint64_to_double(_mm256_loadu_epi64(&data1[i]), mask);
				_mm256_store_pd(&data_result[i], _mm256_exp_pd(a));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = std::exp(static_cast<double>(data1[i]));
			}
			return result;
		}

		vector<double> exp2()
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;

			vector<double> result(size);

			double* data_result = result._data;

			__m256d mask = ud;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = uint64_to_double(_mm256_loadu_epi64(&data1[i]), mask);
				_mm256_store_pd(&data_result[i], _mm256_exp2_pd(a));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = std::exp2(static_cast<double>(data1[i]));
			}
			return result;
		}

		// Trigonometric functions

		vector<double> cos()
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;

			vector<double> result(size);

			double* data_result = result._data;

			__m256d mask = ud;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = uint64_to_double(_mm256_loadu_epi64(&data1[i]), mask);
				_mm256_store_pd(&data_result[i], _mm256_cos_pd(a));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = std::cos(static_cast<double>(data1[i]));
			}
			return result;
		}

		vector<double> acos()
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;

			vector<double> result(size);

			double* data_result = result._data;

			__m256d mask = ud;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = uint64_to_double(_mm256_loadu_epi64(&data1[i]), mask);
				_mm256_store_pd(&data_result[i], _mm256_acos_pd(a));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = std::acos(static_cast<double>(data1[i]));
			}
			return result;
		}

		vector<double> tan()
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;

			vector<double> result(size);

			double* data_result = result._data;

			__m256d mask = ud;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = uint64_to_double(_mm256_loadu_epi64(&data1[i]), mask);
				_mm256_store_pd(&data_result[i], _mm256_tan_pd(a));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = std::tan(static_cast<double>(data1[i]));
			}
			return result;
		}

		vector<double> atan()
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;

			vector<double> result(size);

			double* data_result = result._data;

			__m256d mask = ud;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				__m256d a = uint64_to_double(_mm256_loadu_epi64(&data1[i]), mask);
				_mm256_store_pd(&data_result[i], _mm256_atan_pd(a));
			}
			for (size_t i = final_pos; i < size; i++)
			{
				data_result[i] = std::atan(static_cast<double>(data1[i]));
			}
			return result;
		}

		// <<

		vector<uint64_t> operator<<(int shift)
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;

			vector<uint64_t> result(size);

			uint64_t* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_storeu_epi64(&data_result[i], _mm256_slli_epi64(_mm256_loadu_epi64(&data1[i]), shift));
			}
			for (size_t i = final_pos; i < size; i += 4)
			{
				data_result[i] = data1[i] << shift;
			}
			return result;
		}

		vector<uint64_t> operator<<(vector<uint64_t> other)
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			uint64_t* data2 = other._data;

			vector<uint64_t> result(size);

			uint64_t* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_storeu_epi64(&data_result[i], _mm256_sllv_epi64(_mm256_loadu_epi64(&data1[i]), _mm256_loadu_epi64(&data2[i])));
			}
			for (size_t i = final_pos; i < size; i += 4)
			{
				data_result[i] = data1[i] << data2[i];
			}
			return result;
		}

		vector<uint64_t> operator<<(vector<int> other)
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			int* data2 = other._data;

			vector<uint64_t> result(size);

			uint64_t* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_storeu_epi64(&data_result[i], _mm256_sllv_epi64(_mm256_loadu_epi64(&data1[i]), _mm256_cvtepi32_epi64(_mm_loadu_epi32(&data2[i]))));
			}
			for (size_t i = final_pos; i < size; i += 4)
			{
				data_result[i] = data1[i] << data2[i];
			}
			return result;
		}

		void operator<<=(int shift)
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_storeu_epi64(&data1[i], _mm256_slli_epi64(_mm256_loadu_epi64(&data1[i]), shift));
			}
			for (size_t i = final_pos; i < size; i += 4)
			{
				data1[i] <<= shift;
			}
		}

		void operator<<=(vector<uint64_t> other)
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			uint64_t* data2 = other._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_storeu_epi64(&data1[i], _mm256_sllv_epi64(_mm256_loadu_epi64(&data1[i]), _mm256_loadu_epi64(&data2[i])));
			}
			for (size_t i = final_pos; i < size; i += 4)
			{
				data1[i] <<= data2[i];
			}
		}

		void operator<<=(vector<int> other)
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			int* data2 = other._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_storeu_epi64(&data1[i], _mm256_sllv_epi64(_mm256_loadu_epi64(&data1[i]), _mm256_cvtepi32_epi64(_mm_loadu_epi32(&data2[i]))));
			}
			for (size_t i = final_pos; i < size; i += 4)
			{
				data1[i] <<= data2[i];
			}
		}

		// >>

		vector<uint64_t> operator>>(int shift)
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;

			vector<uint64_t> result(size);

			uint64_t* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_storeu_epi64(&data_result[i], _mm256_srli_epi64(_mm256_loadu_epi64(&data1[i]), shift));
			}
			for (size_t i = final_pos; i < size; i += 4)
			{
				data_result[i] = data1[i] >> shift;
			}
			return result;
		}

		vector<uint64_t> operator>>(vector<uint64_t> other)
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			uint64_t* data2 = other._data;

			vector<uint64_t> result(size);

			uint64_t* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_storeu_epi64(&data_result[i], _mm256_srlv_epi64(_mm256_loadu_epi64(&data1[i]), _mm256_loadu_epi64(&data2[i])));
			}
			for (size_t i = final_pos; i < size; i += 4)
			{
				data_result[i] = data1[i] >> data2[i];
			}
			return result;
		}

		vector<uint64_t> operator>>(vector<int> other)
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			int* data2 = other._data;

			vector<uint64_t> result(size);

			uint64_t* data_result = result._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_storeu_epi64(&data_result[i], _mm256_srlv_epi64(_mm256_loadu_epi64(&data1[i]), _mm256_cvtepi32_epi64(_mm_loadu_epi32(&data2[i]))));
			}
			for (size_t i = final_pos; i < size; i += 4)
			{
				data_result[i] = data1[i] >> data2[i];
			}
			return result;
		}

		void operator>>=(int shift)
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_storeu_epi64(&data1[i], _mm256_srli_epi64(_mm256_loadu_epi64(&data1[i]), shift));
			}
			for (size_t i = final_pos; i < size; i += 4)
			{
				data1[i] >>= shift;
			}
		}

		void operator>>=(vector<uint64_t> other)
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			uint64_t* data2 = other._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_storeu_epi64(&data1[i], _mm256_srlv_epi64(_mm256_loadu_epi64(&data1[i]), _mm256_loadu_epi64(&data2[i])));
			}
			for (size_t i = final_pos; i < size; i += 4)
			{
				data1[i] >>= data2[i];
			}
		}

		void operator>>=(vector<int> other)
		{
			size_t size = this->_size;

			size_t final_pos = (size / 4) * 4;

			uint64_t* data1 = this->_data;
			int* data2 = other._data;

			for (size_t i = 0; i < final_pos; i += 4)
			{
				_mm256_storeu_epi64(&data1[i], _mm256_srlv_epi64(_mm256_loadu_epi64(&data1[i]), _mm256_cvtepi32_epi64(_mm_loadu_epi32(&data2[i]))));
			}
			for (size_t i = final_pos; i < size; i += 4)
			{
				data1[i] >>= data2[i];
			}
		}

	private:
		uint64_t* _data;
		size_t _size;

	};

	template <>
	class vector<int>
	{
	public:

		vector(size_t size) : _data(new int[size]), _size(size) {}

		vector(int* data, size_t size);

		friend class vector<uint64_t>;
		friend class vector<double>;
	private:
		int* _data;
		size_t _size;
	};

	template <>
	class vector<bool>
	{
	public:
		vector(size_t size);

		friend class vector<double>;

		friend class vector<uint64_t>;

		friend class vector<int>;

	private:
		bool* _data;
		size_t _size;
	};

	template <bool this_transposed, bool this_contiguous, bool this_call_destructor>
	class matrix<double, this_transposed, this_contiguous, this_call_destructor>
	{
	public:

		matrix(size_t rows, size_t cols) :
			_data(new double[rows * cols]),
			_rows(rows),
			_cols(cols),
			_size(rows* cols),
			actual_rows(rows),
			actual_cols(cols),
			contiguous(true),
			final_pos_rows((_rows / 4) * 4), 
			final_pos_cols((_cols / 4) * 4),
			final_pos_size((_size / 4) * 4) {}

		matrix(double* data, size_t rows, size_t cols, size_t actual_rows, size_t actual_cols) :
			_data(data),
			_rows(rows),
			_cols(cols),
			_size(rows* cols),
			actual_rows(actual_rows),
			actual_cols(actual_cols),
			contiguous(contiguous), 
			final_pos_rows((_rows / 4) * 4),
			final_pos_cols((_cols / 4) * 4),
			final_pos_size((_size / 4) * 4) {}

		~matrix() { if constexpr (this_call_destructor) delete[] this->_data; }

		//Friend classes

		friend class matrix<bool>;

		friend class vector<double>;

		friend class vector<uint64_t>;

		//----------------

		size_t rows() { return this->_rows; }

		size_t cols() { return this->_cols; }

		double* data() { return this->_data; }

		matrix<double, this_transposed, !this_transposed, false> row(size_t row)
		{
			if constexpr (this_transposed)
			{
				return matrix<double, true, false, false>(
					&this->_data[row],
					1,
					this->_cols,
					this->actual_rows,
					this->actual_cols);
			}
			else
			{
				return matrix<double, false, true, false>(
					&this->_data[row * this->actual_cols],
					1,
					this->_cols,
					this->actual_rows,
					this->actual_cols);
			}
		}

		matrix<double, this_transposed, this_transposed, false> col(size_t col)
		{
			if constexpr (this_transposed)
			{
				return matrix<double, true, true, false>(
					&this->_data[col * this->actual_rows],
					1,
					this->_cols,
					this->actual_rows,
					this->actual_cols);
			}
			else
			{
				return matrix<double, false, false, false>(
					&this->_data[col],
					1,
					this->_cols,
					this->actual_rows,
					this->actual_cols);
			}
		}

		matrix<double, !this_transposed, this_contiguous, false> tranpose()
		{
			if constexpr (this_transposed)
			{
				return matrix<double, false, this_contiguous, false>(
					this->_data,
					this->_rows,
					this->_cols,
					this->actual_cols,
					this->actual_rows
				);
			}
			else
			{
				return matrix<double, true, this_contiguous, false>(
					this->_data,
					this->_rows,
					this->_cols,
					this->actual_rows,
					this->actual_cols
				);
			}
		}

		template<bool block_contiguous = false>
		matrix<double, this_transposed, this_contiguous, false> block(size_t initial_row, size_t initial_col, size_t final_row, size_t final_col)
		{
			if constexpr (this_transposed)
			{
				return matrix<double, true, this_contiguous && block_contiguous, false>(
					&this->_data[initial_col * this->actual_rows + initial_row],
					final_row - initial_row,
					final_col - initial_col,
					final_row - initial_row,
					final_col - initial_col
				);
			}
			else
			{
				return matrix<double, false, this_contiguous && block_contiguous, false>(
					&this->_data[initial_row * this->actual_cols + initial_col],
					final_row - initial_row,
					final_col - initial_col,
					final_row - initial_row,
					final_col - initial_col
				);
			}
		}

		template<bool other_transposed, bool other_contiguous, bool call_destructor>
		friend std::ostream& operator<<(std::ostream& os, const matrix<double, other_transposed, other_contiguous, call_destructor>& matrix)
		{
			if constexpr (other_transposed)
			{
				for (size_t i = 0; i < matrix._rows; i++)
				{
					for (size_t j = 0; j < matrix._cols; j++)
					{
						std::cout << this->[j * matrix.actual_rows + i] << " ";
					}
					std::cout << std::endl;
				}
			}
			else
			{
				for (size_t i = 0; i < matrix._rows; i++)
				{
					for (size_t j = 0; j < matrix._cols; j++)
					{
						std::cout << this->[i * matrix.actual_cols + j] << " ";
					}
					std::cout << std::endl;
				}
			}
			return os;
		}

		double& operator()(size_t row, size_t col)
		{
			if constexpr (this_transposed)
			{
				return this->_data[col * this->actual_rows + row];
			}
			else
			{
				return this->_data[row * this->actual_cols + col];
			}
		}

		const double& operator()(size_t row, size_t col) const 
		{
			if constexpr (this_transposed)
			{
				return this->_data[col * this->actual_rows + row];
			}
			else
			{
				return this->_data[row * this->actual_cols + col];
			}
		}

		// +

		template<bool return_transposed = false, bool other_transposed, bool other_contiguous, bool call_destructor>
		matrix<double, return_transposed> operator+(const matrix<double, other_transposed, other_contiguous, call_destructor>& other)
		{
#ifdef _DEBUG
			if (other._cols != this->_cols || other._rows != this->_rows) throw std::invalid_argument("The dimensions of both matrices must be the same");
#else
#endif

			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;
			double* data2 = other._data;

			matrix<double, return_transposed, true> result(rows, cols);

			double* data_result = result._data;

			if constexpr (this_transposed)
			{
				if constexpr (other_transposed)
				{
					if constexpr (return_transposed)
					{
						if constexpr (this_contiguous && other_contiguous)
						{
							size_t size = this->_size;

							size_t final_pos_size = this->final_pos_size;

							for (size_t i = 0; i < final_pos_size; i += 4)
							{
								__m256d a = _mm256_load_pd(&data1[i]);
								__m256d b = _mm256_load_pd(&data2[i]);

								_mm256_store_pd(&data_result[i], _mm256_add_pd(a, b));
							}
							for (size_t i = final_pos_size; i < size; i++)
							{
								data_result[i] = data1[i] + data2[i];
							}
						}
						else
						{
							size_t final_pos_rows = this->final_pos_rows;
							size_t final_pos_cols = this->final_pos_cols;

							size_t rows1 = this->actual_rows;
							size_t rows2 = other.actual_rows;

							for (size_t i = 0; i < final_pos_rows; i += 4)
							{
								for (size_t j = 0; j < cols; j++)
								{
									__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
									__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);

									_mm256_store_pd(&data_result[j * rows + i], _mm256_add_pd(a, b));
								}
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								for (size_t j = 0; j < final_pos_cols; j += 4)
								{
									__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
										data1[(j + 1) * rows1 + i],
										data1[(j + 2) * rows1 + i],
										data1[(j + 3) * rows1 + i]);
									__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
										data2[(j + 1) * rows2 + i],
										data2[(j + 2) * rows2 + i],
										data2[(j + 3) * rows2 + i]);

									__m256d add = _mm256_add_pd(a, b);

									__m128d val1 = _mm256_extractf128_pd(add, 1);
									__m128d val2 = _mm256_castpd256_pd128(add);

									_mm_store_sd(&data_result[j * cols + i], val2);
									val2 = _mm_shuffle_pd(val2, val2, 1);
									_mm_store_sd(&data_result[(j + 1) * cols + i], val2);

									_mm_store_sd(&data_result[(j + 2) * cols + i], val1);
									val1 = _mm_shuffle_pd(val1, val1, 1);
									_mm_store_sd(&data_result[(j + 3) * cols + i], val1);
								}
								for (size_t j = final_pos_cols; j < cols; j++)
								{
									data_result[j * cols + i] = data1[j * rows1 + i] + data2[j * rows2 + i];
								}
							}
						}
					}
					else
					{
						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						size_t rows1 = this->actual_rows;
						size_t rows2 = other.actual_rows;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);
								__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
									data2[(j + 1) * rows2 + i],
									data2[(j + 2) * rows2 + i],
									data2[(j + 3) * rows2 + i]);

								_mm256_store_pd(&data_result[i * cols + j], _mm256_add_pd(a, b));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < final_pos_rows; i += 4)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
								__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);

								__m256d add = _mm256_add_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(add, 1);
								__m128d val2 = _mm256_castpd256_pd128(add);

								_mm_store_sd(&data_result[i * cols + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

								_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								data_result[i * cols + j] = data1[j * rows1 + i] + data2[j * rows2 + i];
							}
						}
					}
				}
				else
				{
					if constexpr (return_transposed)
					{
						size_t final_pos_rows = this->final_pos_rows;
						size_t final_pos_cols = this->final_pos_cols;

						size_t rows1 = this->actual_rows;
						size_t cols2 = other.actual_cols;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
								__m256d b = _mm256_setr_pd(data2[i * cols2 + j],
									data2[(i + 1) * cols2 + j],
									data2[(i + 2) * cols2 + j],
									data2[(i + 3) * cols2 + j]);

								_mm256_store_pd(&data_result[j * rows + i], _mm256_add_pd(a, b));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < final_pos_cols; j += 4)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);
								__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

								__m256d add = _mm256_add_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(add, 1);
								__m128d val2 = _mm256_castpd256_pd128(add);

								_mm_store_sd(&data_result[j * cols + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(j + 1) * cols + i], val2);

								_mm_store_sd(&data_result[(j + 2) * cols + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(j + 3) * cols + i], val1);
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								data_result[j * cols + i] = data1[j * rows1 + i] + data2[i * cols2 + j];
							}
						}
					}
					else
					{
						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						size_t rows1 = this->actual_rows;
						size_t cols2 = other.actual_cols;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);
								__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

								_mm256_store_pd(&data_result[i * cols + j], _mm256_add_pd(a, b));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < final_pos_rows; i += 4)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
								__m256d b = _mm256_setr_pd(data1[i * cols2 + j],
									data1[(i + 1) * cols2 + j],
									data1[(i + 2) * cols2 + j],
									data1[(i + 3) * cols2 + j]);

								__m256d add = _mm256_add_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(add, 1);
								__m128d val2 = _mm256_castpd256_pd128(add);

								_mm_store_sd(&data_result[i * cols + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

								_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								data_result[i * cols + j] = data1[j * rows1 + i] + data2[i * cols2 + j];
							}
						}
					}
				}
			}
			else
			{
				if constexpr (other_transposed)
				{
					size_t cols1 = this->actual_cols;
					size_t rows2 = other.actual_rows;

					if constexpr (return_transposed)
					{
						size_t final_pos_rows = this->final_pos_rows;
						size_t final_pos_cols = this->final_pos_cols;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);
								__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);
								_mm256_store_pd(&data_result[j * rows + i], _mm256_add_pd(a, b));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; final_pos_cols; j += 4)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
								__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
									data2[(j + 1) * rows2 + i],
									data2[(j + 2) * rows2 + i],
									data2[(j + 3) * rows2 + i]);

								__m256d add = _mm256_add_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(add, 1);
								__m128d val2 = _mm256_castpd256_pd128(add);

								_mm_store_sd(&data_result[j * rows + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(j + 1) * rows + i], val2);

								_mm_store_sd(&data_result[(j + 2) * rows + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(j + 3) * rows + i], val1);
							}
							for (size_t j = final_pos_cols; cols; j++)
							{
								data_result[j * rows + i] = data1[i * cols1 + j] + data2[j * rows2 + i];
							}
						}
					}
					else
					{
						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
								__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
									data2[(j + 1) * rows2 + i],
									data2[(j + 2) * rows2 + i],
									data2[(j + 3) * rows2 + i]);
								_mm256_store_pd(&data_result[i * cols + j], _mm256_add_pd(a, b));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < final_pos_rows; i += 4)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);
								__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);

								__m256d add = _mm256_add_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(add, 1);
								__m128d val2 = _mm256_castpd256_pd128(add);

								_mm_store_sd(&data_result[i * cols + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(i + 1) * rows + j], val2);

								_mm_store_sd(&data_result[(i + 2) * rows + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(i + 3) * rows + j], val1);
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								data_result[i * cols + j] = data1[i * cols1 + j] + data2[j * rows2 + i];
							}
						}
					}
				}
				else
				{
					if constexpr (return_transposed)
					{
						size_t cols1 = this->actual_cols;
						size_t cols2 = other.actual_cols;

						size_t final_pos_rows = this->final_pos_rows;
						size_t final_pos_cols = this->final_pos_cols;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < rows; j++)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);
								__m256d b = _mm256_setr_pd(data2[i * cols2 + j],
									data2[(i + 1) * cols2 + j],
									data2[(i + 2) * cols2 + j],
									data2[(i + 3) * cols2 + j]);
								_mm256_store_pd(&data_result[j * rows + i], _mm256_add_pd(a, b));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < final_pos_cols; j += 4)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
								__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

								__m256d add = _mm256_add_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(add, 1);
								__m128d val2 = _mm256_castpd256_pd128(add);

								_mm_store_sd(&data_result[j * rows + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(j + 1) * rows + i], val2);

								_mm_store_sd(&data_result[(j + 2) * rows + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(j + 3) * rows + i], val1);
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								data_result[j * rows + i] = data1[i * cols1 + j] + data2[i * cols2 + j];
							}
						}
					}
					else
					{
						if constexpr (this_contiguous && other_contiguous)
						{
							size_t final_pos_size = this->final_pos_size;
							size_t size = this->_size;

							for (size_t i = 0; i < final_pos_size; i += 4)
							{
								__m256d a = _mm256_load_pd(&data1[i]);
								__m256d b = _mm256_load_pd(&data2[i]);

								_mm256_store_pd(&data_result[i], _mm256_add_pd(a, b));
							}
							for (size_t i = final_pos_size; i < size; i++)
							{
								data_result[i] = data1[i] + data2[i];
							}
						}
						else
						{
							size_t cols1 = this->actual_cols;
							size_t cols2 = other.actual_cols;

							size_t final_pos_cols = this->final_pos_cols;
							size_t final_pos_rows = this->final_pos_rows;

							for (size_t j = 0; j < final_pos_cols; j += 4)
							{
								for (size_t i = 0; i < rows; i++)
								{
									__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
									__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

									_mm256_store_pd(&data_result[i * cols + j], _mm256_add_pd(a, b));
								}
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								for (size_t i = 0; i < final_pos_rows; i += 4)
								{
									__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
										data1[(i + 1) * cols1 + j],
										data1[(i + 2) * cols1 + j],
										data1[(i + 3) * cols1 + j]);
									__m256d b = _mm256_setr_pd(data2[i * cols2 + j],
										data2[(i + 1) * cols2 + j],
										data2[(i + 2) * cols2 + j],
										data2[(i + 3) * cols2 + j]);

									__m256d add = _mm256_add_pd(a, b);

									__m128d val1 = _mm256_extractf128_pd(add, 1);
									__m128d val2 = _mm256_castpd256_pd128(add);

									_mm_store_sd(&data_result[i * cols + j], val2);
									val2 = _mm_shuffle_pd(val2, val2, 1);
									_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

									_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
									val1 = _mm_shuffle_pd(val1, val1, 1);
									_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
								}
								for (size_t i = final_pos_rows; i < rows; i++)
								{
									data_result[i * cols + j] = data1[i * cols1 + j] + data2[i * cols2 + j];
								}
							}
						}
					}
				}
			}
			return result;
		}

		template<bool other_transposed, bool other_contiguous, bool call_destructor>
		void operator+=(const matrix<double, other_transposed, other_contiguous, call_destructor>& other)
		{
#ifdef _DEBUG
			if (other._cols != this->_cols || other._rows != this->_rows) throw std::invalid_argument("The dimensions of both matrices must be the same");
#else
#endif

			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;
			double* data2 = other._data;

			if constexpr (this_transposed)
			{
				if constexpr (other_transposed)
				{
					if constexpr (this_contiguous && other_contiguous)
					{
						size_t size = this->_size;

						size_t final_pos_size = this->final_pos_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);
							__m256d b = _mm256_load_pd(&data2[i]);

							_mm256_store_pd(&data1[i], _mm256_add_pd(a, b));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data1[i] += data2[i];
						}
					}
					else
					{
						size_t final_pos_rows = this->final_pos_rows;
						size_t final_pos_cols = this->final_pos_cols;

						size_t rows1 = this->actual_rows;
						size_t rows2 = other.actual_rows;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
								__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);

								_mm256_store_pd(&data1[j * rows1 + i], _mm256_add_pd(a, b));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < final_pos_cols; j += 4)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);
								__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
									data2[(j + 1) * rows2 + i],
									data2[(j + 2) * rows2 + i],
									data2[(j + 3) * rows2 + i]);

								__m256d add = _mm256_add_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(add, 1);
								__m128d val2 = _mm256_castpd256_pd128(add);

								_mm_store_sd(&data1[j * rows1 + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data1[(j + 1) * rows1 + i], val2);

								_mm_store_sd(&data1[(j + 2) * rows1 + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data1[(j + 3) * rows1 + i], val1);
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								data1[j * rows1 + i] += data2[j * rows2 + i];
							}
						}
					}
				}
				else
				{
					size_t final_pos_rows = this->final_pos_rows;
					size_t final_pos_cols = this->final_pos_cols;

					size_t rows1 = this->actual_rows;
					size_t cols2 = other.actual_cols;

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
							__m256d b = _mm256_setr_pd(data2[i * cols2 + j],
								data2[(i + 1) * cols2 + j],
								data2[(i + 2) * cols2 + j],
								data2[(i + 3) * cols2 + j]);

							_mm256_store_pd(&data1[j * rows1 + i], _mm256_add_pd(a, b));
						}
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
								data1[(j + 1) * rows1 + i],
								data1[(j + 2) * rows1 + i],
								data1[(j + 3) * rows1 + i]);
							__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

							__m256d add = _mm256_add_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(add, 1);
							__m128d val2 = _mm256_castpd256_pd128(add);

							_mm_store_sd(&data1[j * rows1 + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(j + 1) * rows1 + i], val2);

							_mm_store_sd(&data1[(j + 2) * rows1 + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(j + 3) * rows1 + i], val1);
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							data1[j * rows1 + i] += data2[i * cols2 + j];
						}
					}
				}
			}
			else
			{
				if constexpr (other_transposed)
				{
					size_t cols1 = this->actual_cols;
					size_t rows2 = other.actual_rows;

					size_t final_pos_cols = this->final_pos_cols;
					size_t final_pos_rows = this->final_pos_rows;

					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
							__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
								data2[(j + 1) * rows2 + i],
								data2[(j + 2) * rows2 + i],
								data2[(j + 3) * rows2 + i]);
							_mm256_store_pd(&data1[i * cols1 + j], _mm256_add_pd(a, b));
						}
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
								data1[(i + 1) * cols1 + j],
								data1[(i + 2) * cols1 + j],
								data1[(i + 3) * cols1 + j]);
							__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);

							__m256d add = _mm256_add_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(add, 1);
							__m128d val2 = _mm256_castpd256_pd128(add);

							_mm_store_sd(&data1[i * cols1 + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(i + 1) * cols1 + j], val2);

							_mm_store_sd(&data1[(i + 2) * cols1 + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(i + 3) * cols1 + j], val1);
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							data1[i * cols1 + j] += data2[j * rows2 + i];
						}
					}
				}
				else
				{
					if constexpr (this_contiguous && other_contiguous)
					{
						size_t final_pos_size = this->final_pos_size;
						size_t size = this->_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);
							__m256d b = _mm256_load_pd(&data2[i]);

							_mm256_store_pd(&data1[i], _mm256_add_pd(a, b));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data1[i] += data2[i];
						}
					}
					else
					{
						size_t cols1 = this->actual_cols;
						size_t cols2 = other.actual_cols;

						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
								__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

								_mm256_store_pd(&data1[i * cols1 + j], _mm256_add_pd(a, b));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < final_pos_rows; i += 4)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);
								__m256d b = _mm256_setr_pd(data2[i * cols2 + j],
									data2[(i + 1) * cols2 + j],
									data2[(i + 2) * cols2 + j],
									data2[(i + 3) * cols2 + j]);

								__m256d add = _mm256_add_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(add, 1);
								__m128d val2 = _mm256_castpd256_pd128(add);

								_mm_store_sd(&data1[i * cols1 + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data1[(i + 1) * cols1 + j], val2);

								_mm_store_sd(&data1[(i + 2) * cols1 + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data1[(i + 3) * cols1 + j], val1);
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								data1[i * cols1 + j] += data2[i * cols2 + j];
							}
						}
					}
				}
			}
		}

		template<bool return_transposed = false>
		matrix<double, return_transposed> operator+(double num)
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			matrix<double, return_transposed, true> result(rows, cols);

			double* data_result = result._data;

			__m256d b = _mm256_set1_pd(num);

			if constexpr (this_transposed)
			{
				if constexpr (return_transposed)
				{
					if constexpr (this_contiguous)
					{
						size_t size = this->_size;

						size_t final_pos_size = this->final_pos_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);

							_mm256_store_pd(&data_result[i], _mm256_add_pd(a, b));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = data1[i] + num;
						}
					}
					else
					{
						size_t final_pos_rows = this->final_pos_rows;
						size_t final_pos_cols = this->final_pos_cols;

						size_t rows1 = this->actual_rows;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

								_mm256_store_pd(&data_result[j * rows + i], _mm256_add_pd(a, b));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < final_pos_cols; j += 4)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);

								__m256d add = _mm256_add_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(add, 1);
								__m128d val2 = _mm256_castpd256_pd128(add);

								_mm_store_sd(&data_result[j * cols + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(j + 1) * cols + i], val2);

								_mm_store_sd(&data_result[(j + 2) * cols + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(j + 3) * cols + i], val1);
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								data_result[j * cols + i] = data1[j * rows1 + i] + num;
							}
						}
					}
				}
				else
				{
					size_t final_pos_cols = this->final_pos_cols;
					size_t final_pos_rows = this->final_pos_rows;

					size_t rows1 = this->actual_rows;

					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
								data1[(j + 1) * rows1 + i],
								data1[(j + 2) * rows1 + i],
								data1[(j + 3) * rows1 + i]);

							_mm256_store_pd(&data_result[i * cols + j], _mm256_add_pd(a, b));
						}
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

							__m256d add = _mm256_add_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(add, 1);
							__m128d val2 = _mm256_castpd256_pd128(add);

							_mm_store_sd(&data_result[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

							_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							data_result[i * cols + j] = data1[j * rows1 + i] + num;
						}
					}
				}
			}
			else
			{
				if constexpr (return_transposed)
				{
					size_t cols1 = this->actual_cols;

					size_t final_pos_rows = this->final_pos_rows;
					size_t final_pos_cols = this->final_pos_cols;

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						for (size_t j = 0; j < rows; j++)
						{
							__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
								data1[(i + 1) * cols1 + j],
								data1[(i + 2) * cols1 + j],
								data1[(i + 3) * cols1 + j]);

							_mm256_store_pd(&data_result[j * rows + i], _mm256_add_pd(a, b));
						}
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

							__m256d add = _mm256_add_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(add, 1);
							__m128d val2 = _mm256_castpd256_pd128(add);

							_mm_store_sd(&data_result[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data_result[(j + 1) * rows + i], val2);

							_mm_store_sd(&data_result[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data_result[(j + 3) * rows + i], val1);
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							data_result[j * rows + i] = data1[i * cols1 + j] + num;
						}
					}
				}
				else
				{
					if constexpr (this_contiguous)
					{
						size_t final_pos_size = this->final_pos_size;
						size_t size = this->_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);

							_mm256_store_pd(&data_result[i], _mm256_add_pd(a, b));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = data1[i] + num;
						}
					}
					else
					{
						size_t cols1 = this->actual_cols;

						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

								_mm256_store_pd(&data_result[i * cols + j], _mm256_add_pd(a, b));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < final_pos_rows; i += 4)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);

								__m256d add = _mm256_add_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(add, 1);
								__m128d val2 = _mm256_castpd256_pd128(add);

								_mm_store_sd(&data_result[i * cols + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

								_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								data_result[i * cols + j] = data1[i * cols1 + j] + num;
							}
						}
					}
				}
			}
			return result;
		}

		void operator+=(double num)
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			__m256d b = _mm256_set1_pd(num);

			if constexpr (this_transposed)
			{
				if constexpr (this_contiguous)
				{
					size_t size = this->_size;

					size_t final_pos_size = this->final_pos_size;

					for (size_t i = 0; i < final_pos_size; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&data1[i], _mm256_add_pd(a, b));
					}
					for (size_t i = final_pos_size; i < size; i++)
					{
						data1[i] += num;
					}
				}
				else
				{
					size_t final_pos_rows = this->final_pos_rows;
					size_t final_pos_cols = this->final_pos_cols;

					size_t rows1 = this->actual_rows;

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

							_mm256_store_pd(&data1[j * rows1 + i], _mm256_add_pd(a, b));
						}
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
								data1[(j + 1) * rows1 + i],
								data1[(j + 2) * rows1 + i],
								data1[(j + 3) * rows1 + i]);

							__m256d add = _mm256_add_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(add, 1);
							__m128d val2 = _mm256_castpd256_pd128(add);

							_mm_store_sd(&data1[j * rows1 + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(j + 1) * rows1 + i], val2);

							_mm_store_sd(&data1[(j + 2) * rows1 + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(j + 3) * rows1 + i], val1);
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							data1[j * rows1 + i] += num;
						}
					}
				}
			}
			else
			{
				if constexpr (this_contiguous)
				{
					size_t final_pos_size = this->final_pos_size;
					size_t size = this->_size;

					for (size_t i = 0; i < final_pos_size; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&data1[i], _mm256_add_pd(a, b));
					}
					for (size_t i = final_pos_size; i < size; i++)
					{
						data1[i] += num;
					}
				}
				else
				{
					size_t cols1 = this->actual_cols;

					size_t final_pos_cols = this->final_pos_cols;
					size_t final_pos_rows = this->final_pos_rows;

					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

							_mm256_store_pd(&data1[i * cols1 + j], _mm256_add_pd(a, b));
						}
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
								data1[(i + 1) * cols1 + j],
								data1[(i + 2) * cols1 + j],
								data1[(i + 3) * cols1 + j]);

							__m256d add = _mm256_add_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(add, 1);
							__m128d val2 = _mm256_castpd256_pd128(add);

							_mm_store_sd(&data1[i * cols1 + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(i + 1) * cols1 + j], val2);

							_mm_store_sd(&data1[(i + 2) * cols1 + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(i + 3) * cols1 + j], val1);
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							data1[i * cols1 + j] += num;
						}
					}
				}
			}
		}

		// -

		template<bool return_transposed = false, bool other_transposed, bool other_contiguous, bool call_destructor>
		matrix<double, return_transposed> operator-(const matrix<double, other_transposed, other_contiguous, call_destructor>& other)
		{
#ifdef _DEBUG
			if (other._cols != this->_cols || other._rows != this->_rows) throw std::invalid_argument("The dimensions of both matrices must be the same");
#else
#endif

			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;
			double* data2 = other._data;

			matrix<double, return_transposed, true> result(rows, cols);

			double* data_result = result._data;

			if constexpr (this_transposed)
			{
				if constexpr (other_transposed)
				{
					if constexpr (return_transposed)
					{
						if constexpr (this_contiguous && other_contiguous)
						{
							size_t size = this->_size;

							size_t final_pos_size = this->final_pos_size;

							for (size_t i = 0; i < final_pos_size; i -= 4)
							{
								__m256d a = _mm256_load_pd(&data1[i]);
								__m256d b = _mm256_load_pd(&data2[i]);

								_mm256_store_pd(&data_result[i], _mm256_sub_pd(a, b));
							}
							for (size_t i = final_pos_size; i < size; i++)
							{
								data_result[i] = data1[i] - data2[i];
							}
						}
						else
						{
							size_t final_pos_rows = this->final_pos_rows;
							size_t final_pos_cols = this->final_pos_cols;

							size_t rows1 = this->actual_rows;
							size_t rows2 = other.actual_rows;

							for (size_t i = 0; i < final_pos_rows; i -= 4)
							{
								for (size_t j = 0; j < cols; j++)
								{
									__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
									__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);

									_mm256_store_pd(&data_result[j * rows + i], _mm256_sub_pd(a, b));
								}
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								for (size_t j = 0; j < final_pos_cols; j -= 4)
								{
									__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
										data1[(j + 1) * rows1 + i],
										data1[(j + 2) * rows1 + i],
										data1[(j + 3) * rows1 + i]);
									__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
										data2[(j + 1) * rows2 + i],
										data2[(j + 2) * rows2 + i],
										data2[(j + 3) * rows2 + i]);

									__m256d sub = _mm256_sub_pd(a, b);

									__m128d val1 = _mm256_extractf128_pd(sub, 1);
									__m128d val2 = _mm256_castpd256_pd128(sub);

									_mm_store_sd(&data_result[j * cols + i], val2);
									val2 = _mm_shuffle_pd(val2, val2, 1);
									_mm_store_sd(&data_result[(j + 1) * cols + i], val2);

									_mm_store_sd(&data_result[(j + 2) * cols + i], val1);
									val1 = _mm_shuffle_pd(val1, val1, 1);
									_mm_store_sd(&data_result[(j + 3) * cols + i], val1);
								}
								for (size_t j = final_pos_cols; j < cols; j++)
								{
									data_result[j * cols + i] = data1[j * rows1 + i] - data2[j * rows2 + i];
								}
							}
						}
					}
					else
					{
						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						size_t rows1 = this->actual_rows;
						size_t rows2 = other.actual_rows;

						for (size_t j = 0; j < final_pos_cols; j -= 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);
								__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
									data2[(j + 1) * rows2 + i],
									data2[(j + 2) * rows2 + i],
									data2[(j + 3) * rows2 + i]);

								_mm256_store_pd(&data_result[i * cols + j], _mm256_sub_pd(a, b));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < final_pos_rows; i -= 4)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
								__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);

								__m256d sub = _mm256_sub_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(sub, 1);
								__m128d val2 = _mm256_castpd256_pd128(sub);

								_mm_store_sd(&data_result[i * cols + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

								_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								data_result[i * cols + j] = data1[j * rows1 + i] - data2[j * rows2 + i];
							}
						}
					}
				}
				else
				{
					if constexpr (return_transposed)
					{
						size_t final_pos_rows = this->final_pos_rows;
						size_t final_pos_cols = this->final_pos_cols;

						size_t rows1 = this->actual_rows;
						size_t cols2 = other.actual_cols;

						for (size_t i = 0; i < final_pos_rows; i -= 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
								__m256d b = _mm256_setr_pd(data2[i * cols2 + j],
									data2[(i + 1) * cols2 + j],
									data2[(i + 2) * cols2 + j],
									data2[(i + 3) * cols2 + j]);

								_mm256_store_pd(&data_result[j * rows + i], _mm256_sub_pd(a, b));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < final_pos_cols; j -= 4)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);
								__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

								__m256d sub = _mm256_sub_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(sub, 1);
								__m128d val2 = _mm256_castpd256_pd128(sub);

								_mm_store_sd(&data_result[j * cols + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(j + 1) * cols + i], val2);

								_mm_store_sd(&data_result[(j + 2) * cols + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(j + 3) * cols + i], val1);
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								data_result[j * cols + i] = data1[j * rows1 + i] - data2[i * cols2 + j];
							}
						}
					}
					else
					{
						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						size_t rows1 = this->actual_rows;
						size_t cols2 = other.actual_cols;

						for (size_t j = 0; j < final_pos_cols; j -= 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);
								__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

								_mm256_store_pd(&data_result[i * cols + j], _mm256_sub_pd(a, b));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < final_pos_rows; i -= 4)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
								__m256d b = _mm256_setr_pd(data1[i * cols2 + j],
									data1[(i + 1) * cols2 + j],
									data1[(i + 2) * cols2 + j],
									data1[(i + 3) * cols2 + j]);

								__m256d sub = _mm256_sub_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(sub, 1);
								__m128d val2 = _mm256_castpd256_pd128(sub);

								_mm_store_sd(&data_result[i * cols + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

								_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								data_result[i * cols + j] = data1[j * rows1 + i] - data2[i * cols2 + j];
							}
						}
					}
				}
			}
			else
			{
				if constexpr (other_transposed)
				{
					size_t cols1 = this->actual_cols;
					size_t rows2 = other.actual_rows;

					if constexpr (return_transposed)
					{
						size_t final_pos_rows = this->final_pos_rows;
						size_t final_pos_cols = this->final_pos_cols;

						for (size_t i = 0; i < final_pos_rows; i -= 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);
								__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);
								_mm256_store_pd(&data_result[j * rows + i], _mm256_sub_pd(a, b));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; final_pos_cols; j -= 4)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
								__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
									data2[(j + 1) * rows2 + i],
									data2[(j + 2) * rows2 + i],
									data2[(j + 3) * rows2 + i]);

								__m256d sub = _mm256_sub_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(sub, 1);
								__m128d val2 = _mm256_castpd256_pd128(sub);

								_mm_store_sd(&data_result[j * rows + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(j + 1) * rows + i], val2);

								_mm_store_sd(&data_result[(j + 2) * rows + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(j + 3) * rows + i], val1);
							}
							for (size_t j = final_pos_cols; cols; j++)
							{
								data_result[j * rows + i] = data1[i * cols1 + j] - data2[j * rows2 + i];
							}
						}
					}
					else
					{
						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						for (size_t j = 0; j < final_pos_cols; j -= 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
								__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
									data2[(j + 1) * rows2 + i],
									data2[(j + 2) * rows2 + i],
									data2[(j + 3) * rows2 + i]);
								_mm256_store_pd(&data_result[i * cols + j], _mm256_sub_pd(a, b));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < final_pos_rows; i -= 4)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);
								__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);

								__m256d sub = _mm256_sub_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(sub, 1);
								__m128d val2 = _mm256_castpd256_pd128(sub);

								_mm_store_sd(&data_result[i * cols + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(i + 1) * rows + j], val2);

								_mm_store_sd(&data_result[(i + 2) * rows + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(i + 3) * rows + j], val1);
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								data_result[i * cols + j] = data1[i * cols1 + j] - data2[j * rows2 + i];
							}
						}
					}
				}
				else
				{
					if constexpr (return_transposed)
					{
						size_t cols1 = this->actual_cols;
						size_t cols2 = other.actual_cols;

						size_t final_pos_rows = this->final_pos_rows;
						size_t final_pos_cols = this->final_pos_cols;

						for (size_t i = 0; i < final_pos_rows; i -= 4)
						{
							for (size_t j = 0; j < rows; j++)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);
								__m256d b = _mm256_setr_pd(data2[i * cols2 + j],
									data2[(i + 1) * cols2 + j],
									data2[(i + 2) * cols2 + j],
									data2[(i + 3) * cols2 + j]);
								_mm256_store_pd(&data_result[j * rows + i], _mm256_sub_pd(a, b));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < final_pos_cols; j -= 4)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
								__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

								__m256d sub = _mm256_sub_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(sub, 1);
								__m128d val2 = _mm256_castpd256_pd128(sub);

								_mm_store_sd(&data_result[j * rows + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(j + 1) * rows + i], val2);

								_mm_store_sd(&data_result[(j + 2) * rows + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(j + 3) * rows + i], val1);
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								data_result[j * rows + i] = data1[i * cols1 + j] - data2[i * cols2 + j];
							}
						}
					}
					else
					{
						if constexpr (this_contiguous && other_contiguous)
						{
							size_t final_pos_size = this->final_pos_size;
							size_t size = this->_size;

							for (size_t i = 0; i < final_pos_size; i -= 4)
							{
								__m256d a = _mm256_load_pd(&data1[i]);
								__m256d b = _mm256_load_pd(&data2[i]);

								_mm256_store_pd(&data_result[i], _mm256_sub_pd(a, b));
							}
							for (size_t i = final_pos_size; i < size; i++)
							{
								data_result[i] = data1[i] - data2[i];
							}
						}
						else
						{
							size_t cols1 = this->actual_cols;
							size_t cols2 = other.actual_cols;

							size_t final_pos_cols = this->final_pos_cols;
							size_t final_pos_rows = this->final_pos_rows;

							for (size_t j = 0; j < final_pos_cols; j -= 4)
							{
								for (size_t i = 0; i < rows; i++)
								{
									__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
									__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

									_mm256_store_pd(&data_result[i * cols + j], _mm256_sub_pd(a, b));
								}
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								for (size_t i = 0; i < final_pos_rows; i -= 4)
								{
									__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
										data1[(i + 1) * cols1 + j],
										data1[(i + 2) * cols1 + j],
										data1[(i + 3) * cols1 + j]);
									__m256d b = _mm256_setr_pd(data2[i * cols2 + j],
										data2[(i + 1) * cols2 + j],
										data2[(i + 2) * cols2 + j],
										data2[(i + 3) * cols2 + j]);

									__m256d sub = _mm256_sub_pd(a, b);

									__m128d val1 = _mm256_extractf128_pd(sub, 1);
									__m128d val2 = _mm256_castpd256_pd128(sub);

									_mm_store_sd(&data_result[i * cols + j], val2);
									val2 = _mm_shuffle_pd(val2, val2, 1);
									_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

									_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
									val1 = _mm_shuffle_pd(val1, val1, 1);
									_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
								}
								for (size_t i = final_pos_rows; i < rows; i++)
								{
									data_result[i * cols + j] = data1[i * cols1 + j] - data2[i * cols2 + j];
								}
							}
						}
					}
				}
			}
			return result;
		}

		template<bool other_transposed, bool other_contiguous, bool call_destructor>
		void operator-=(const matrix<double, other_transposed, other_contiguous, call_destructor>& other)
		{
#ifdef _DEBUG
			if (other._cols != this->_cols || other._rows != this->_rows) throw std::invalid_argument("The dimensions of both matrices must be the same");
#else
#endif

			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;
			double* data2 = other._data;

			if constexpr (this_transposed)
			{
				if constexpr (other_transposed)
				{
					if constexpr (this_contiguous && other_contiguous)
					{
						size_t size = this->_size;

						size_t final_pos_size = this->final_pos_size;

						for (size_t i = 0; i < final_pos_size; i -= 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);
							__m256d b = _mm256_load_pd(&data2[i]);

							_mm256_store_pd(&data1[i], _mm256_sub_pd(a, b));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data1[i] -= data2[i];
						}
					}
					else
					{
						size_t final_pos_rows = this->final_pos_rows;
						size_t final_pos_cols = this->final_pos_cols;

						size_t rows1 = this->actual_rows;
						size_t rows2 = other.actual_rows;

						for (size_t i = 0; i < final_pos_rows; i -= 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
								__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);

								_mm256_store_pd(&data1[j * rows1 + i], _mm256_sub_pd(a, b));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < final_pos_cols; j -= 4)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);
								__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
									data2[(j + 1) * rows2 + i],
									data2[(j + 2) * rows2 + i],
									data2[(j + 3) * rows2 + i]);

								__m256d sub = _mm256_sub_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(sub, 1);
								__m128d val2 = _mm256_castpd256_pd128(sub);

								_mm_store_sd(&data1[j * rows1 + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data1[(j + 1) * rows1 + i], val2);

								_mm_store_sd(&data1[(j + 2) * rows1 + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data1[(j + 3) * rows1 + i], val1);
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								data1[j * rows1 + i] -= data2[j * rows2 + i];
							}
						}
					}
				}
				else
				{
					size_t final_pos_rows = this->final_pos_rows;
					size_t final_pos_cols = this->final_pos_cols;

					size_t rows1 = this->actual_rows;
					size_t cols2 = other.actual_cols;

					for (size_t i = 0; i < final_pos_rows; i -= 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
							__m256d b = _mm256_setr_pd(data2[i * cols2 + j],
								data2[(i + 1) * cols2 + j],
								data2[(i + 2) * cols2 + j],
								data2[(i + 3) * cols2 + j]);

							_mm256_store_pd(&data1[j * rows1 + i], _mm256_sub_pd(a, b));
						}
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						for (size_t j = 0; j < final_pos_cols; j -= 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
								data1[(j + 1) * rows1 + i],
								data1[(j + 2) * rows1 + i],
								data1[(j + 3) * rows1 + i]);
							__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

							__m256d sub = _mm256_sub_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(sub, 1);
							__m128d val2 = _mm256_castpd256_pd128(sub);

							_mm_store_sd(&data1[j * rows1 + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(j + 1) * rows1 + i], val2);

							_mm_store_sd(&data1[(j + 2) * rows1 + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(j + 3) * rows1 + i], val1);
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							data1[j * rows1 + i] -= data2[i * cols2 + j];
						}
					}
				}
			}
			else
			{
				if constexpr (other_transposed)
				{
					size_t cols1 = this->actual_cols;
					size_t rows2 = other.actual_rows;

					size_t final_pos_cols = this->final_pos_cols;
					size_t final_pos_rows = this->final_pos_rows;

					for (size_t j = 0; j < final_pos_cols; j -= 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
							__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
								data2[(j + 1) * rows2 + i],
								data2[(j + 2) * rows2 + i],
								data2[(j + 3) * rows2 + i]);
							_mm256_store_pd(&data1[i * cols1 + j], _mm256_sub_pd(a, b));
						}
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						for (size_t i = 0; i < final_pos_rows; i -= 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
								data1[(i + 1) * cols1 + j],
								data1[(i + 2) * cols1 + j],
								data1[(i + 3) * cols1 + j]);
							__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);

							__m256d sub = _mm256_sub_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(sub, 1);
							__m128d val2 = _mm256_castpd256_pd128(sub);

							_mm_store_sd(&data1[i * cols1 + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(i + 1) * cols1 + j], val2);

							_mm_store_sd(&data1[(i + 2) * cols1 + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(i + 3) * cols1 + j], val1);
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							data1[i * cols1 + j] -= data2[j * rows2 + i];
						}
					}
				}
				else
				{
					if constexpr (this_contiguous && other_contiguous)
					{
						size_t final_pos_size = this->final_pos_size;
						size_t size = this->_size;

						for (size_t i = 0; i < final_pos_size; i -= 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);
							__m256d b = _mm256_load_pd(&data2[i]);

							_mm256_store_pd(&data1[i], _mm256_sub_pd(a, b));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data1[i] -= data2[i];
						}
					}
					else
					{
						size_t cols1 = this->actual_cols;
						size_t cols2 = other.actual_cols;

						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						for (size_t j = 0; j < final_pos_cols; j -= 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
								__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

								_mm256_store_pd(&data1[i * cols1 + j], _mm256_sub_pd(a, b));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < final_pos_rows; i -= 4)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);
								__m256d b = _mm256_setr_pd(data2[i * cols2 + j],
									data2[(i + 1) * cols2 + j],
									data2[(i + 2) * cols2 + j],
									data2[(i + 3) * cols2 + j]);

								__m256d sub = _mm256_sub_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(sub, 1);
								__m128d val2 = _mm256_castpd256_pd128(sub);

								_mm_store_sd(&data1[i * cols1 + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data1[(i + 1) * cols1 + j], val2);

								_mm_store_sd(&data1[(i + 2) * cols1 + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data1[(i + 3) * cols1 + j], val1);
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								data1[i * cols1 + j] -= data2[i * cols2 + j];
							}
						}
					}
				}
			}
		}

		template<bool return_transposed = false>
		matrix<double, return_transposed> operator-(double num)
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			matrix<double, return_transposed, true> result(rows, cols);

			double* data_result = result._data;

			__m256d b = _mm256_set1_pd(num);

			if constexpr (this_transposed)
			{
				if constexpr (return_transposed)
				{
					if constexpr (this_contiguous)
					{
						size_t size = this->_size;

						size_t final_pos_size = this->final_pos_size;

						for (size_t i = 0; i < final_pos_size; i -= 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);

							_mm256_store_pd(&data_result[i], _mm256_sub_pd(a, b));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = data1[i] - num;
						}
					}
					else
					{
						size_t final_pos_rows = this->final_pos_rows;
						size_t final_pos_cols = this->final_pos_cols;

						size_t rows1 = this->actual_rows;

						for (size_t i = 0; i < final_pos_rows; i -= 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

								_mm256_store_pd(&data_result[j * rows + i], _mm256_sub_pd(a, b));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < final_pos_cols; j -= 4)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);

								__m256d sub = _mm256_sub_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(sub, 1);
								__m128d val2 = _mm256_castpd256_pd128(sub);

								_mm_store_sd(&data_result[j * cols + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(j + 1) * cols + i], val2);

								_mm_store_sd(&data_result[(j + 2) * cols + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(j + 3) * cols + i], val1);
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								data_result[j * cols + i] = data1[j * rows1 + i] - num;
							}
						}
					}
				}
				else
				{
					size_t final_pos_cols = this->final_pos_cols;
					size_t final_pos_rows = this->final_pos_rows;

					size_t rows1 = this->actual_rows;

					for (size_t j = 0; j < final_pos_cols; j -= 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
								data1[(j + 1) * rows1 + i],
								data1[(j + 2) * rows1 + i],
								data1[(j + 3) * rows1 + i]);

							_mm256_store_pd(&data_result[i * cols + j], _mm256_sub_pd(a, b));
						}
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						for (size_t i = 0; i < final_pos_rows; i -= 4)
						{
							__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

							__m256d sub = _mm256_sub_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(sub, 1);
							__m128d val2 = _mm256_castpd256_pd128(sub);

							_mm_store_sd(&data_result[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

							_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							data_result[i * cols + j] = data1[j * rows1 + i] - num;
						}
					}
				}
			}
			else
			{
				if constexpr (return_transposed)
				{
					size_t cols1 = this->actual_cols;

					size_t final_pos_rows = this->final_pos_rows;
					size_t final_pos_cols = this->final_pos_cols;

					for (size_t i = 0; i < final_pos_rows; i -= 4)
					{
						for (size_t j = 0; j < rows; j++)
						{
							__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
								data1[(i + 1) * cols1 + j],
								data1[(i + 2) * cols1 + j],
								data1[(i + 3) * cols1 + j]);

							_mm256_store_pd(&data_result[j * rows + i], _mm256_sub_pd(a, b));
						}
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						for (size_t j = 0; j < final_pos_cols; j -= 4)
						{
							__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

							__m256d sub = _mm256_sub_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(sub, 1);
							__m128d val2 = _mm256_castpd256_pd128(sub);

							_mm_store_sd(&data_result[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data_result[(j + 1) * rows + i], val2);

							_mm_store_sd(&data_result[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data_result[(j + 3) * rows + i], val1);
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							data_result[j * rows + i] = data1[i * cols1 + j] - num;
						}
					}
				}
				else
				{
					if constexpr (this_contiguous)
					{
						size_t final_pos_size = this->final_pos_size;
						size_t size = this->_size;

						for (size_t i = 0; i < final_pos_size; i -= 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);

							_mm256_store_pd(&data_result[i], _mm256_sub_pd(a, b));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = data1[i] - num;
						}
					}
					else
					{
						size_t cols1 = this->actual_cols;

						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						for (size_t j = 0; j < final_pos_cols; j -= 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

								_mm256_store_pd(&data_result[i * cols + j], _mm256_sub_pd(a, b));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < final_pos_rows; i -= 4)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);

								__m256d sub = _mm256_sub_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(sub, 1);
								__m128d val2 = _mm256_castpd256_pd128(sub);

								_mm_store_sd(&data_result[i * cols + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

								_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								data_result[i * cols + j] = data1[i * cols1 + j] - num;
							}
						}
					}
				}
			}
			return result;
		}

		void operator-=(double num)
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			__m256d b = _mm256_set1_pd(num);

			if constexpr (this_transposed)
			{
				if constexpr (this_contiguous)
				{
					size_t size = this->_size;

					size_t final_pos_size = this->final_pos_size;

					for (size_t i = 0; i < final_pos_size; i -= 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&data1[i], _mm256_sub_pd(a, b));
					}
					for (size_t i = final_pos_size; i < size; i++)
					{
						data1[i] -= num;
					}
				}
				else
				{
					size_t final_pos_rows = this->final_pos_rows;
					size_t final_pos_cols = this->final_pos_cols;

					size_t rows1 = this->actual_rows;

					for (size_t i = 0; i < final_pos_rows; i -= 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

							_mm256_store_pd(&data1[j * rows1 + i], _mm256_sub_pd(a, b));
						}
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						for (size_t j = 0; j < final_pos_cols; j -= 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
								data1[(j + 1) * rows1 + i],
								data1[(j + 2) * rows1 + i],
								data1[(j + 3) * rows1 + i]);

							__m256d sub = _mm256_sub_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(sub, 1);
							__m128d val2 = _mm256_castpd256_pd128(sub);

							_mm_store_sd(&data1[j * rows1 + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(j + 1) * rows1 + i], val2);

							_mm_store_sd(&data1[(j + 2) * rows1 + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(j + 3) * rows1 + i], val1);
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							data1[j * rows1 + i] -= num;
						}
					}
				}
			}
			else
			{
				if constexpr (this_contiguous)
				{
					size_t final_pos_size = this->final_pos_size;
					size_t size = this->_size;

					for (size_t i = 0; i < final_pos_size; i -= 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&data1[i], _mm256_sub_pd(a, b));
					}
					for (size_t i = final_pos_size; i < size; i++)
					{
						data1[i] -= num;
					}
				}
				else
				{
					size_t cols1 = this->actual_cols;

					size_t final_pos_cols = this->final_pos_cols;
					size_t final_pos_rows = this->final_pos_rows;

					for (size_t j = 0; j < final_pos_cols; j -= 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

							_mm256_store_pd(&data1[i * cols1 + j], _mm256_sub_pd(a, b));
						}
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						for (size_t i = 0; i < final_pos_rows; i -= 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
								data1[(i + 1) * cols1 + j],
								data1[(i + 2) * cols1 + j],
								data1[(i + 3) * cols1 + j]);

							__m256d sub = _mm256_sub_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(sub, 1);
							__m128d val2 = _mm256_castpd256_pd128(sub);

							_mm_store_sd(&data1[i * cols1 + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(i + 1) * cols1 + j], val2);

							_mm_store_sd(&data1[(i + 2) * cols1 + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(i + 3) * cols1 + j], val1);
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							data1[i * cols1 + j] -= num;
						}
					}
				}
			}
		}

		// *

		template<bool return_transposed = false, bool other_transposed, bool other_contiguous, bool call_destructor>
		matrix<double, return_transposed> operator*(const matrix<double, other_transposed, other_contiguous, call_destructor>& other)
		{
#ifdef _DEBUG
			if (other._cols != this->_cols || other._rows != this->_rows) throw std::invalid_argument("The dimensions of both matrices must be the same");
#else
#endif

			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;
			double* data2 = other._data;

			matrix<double, return_transposed, true> result(rows, cols);

			double* data_result = result._data;

			if constexpr (this_transposed)
			{
				if constexpr (other_transposed)
				{
					if constexpr (return_transposed)
					{
						if constexpr (this_contiguous && other_contiguous)
						{
							size_t size = this->_size;

							size_t final_pos_size = this->final_pos_size;

							for (size_t i = 0; i < final_pos_size; i *= 4)
							{
								__m256d a = _mm256_load_pd(&data1[i]);
								__m256d b = _mm256_load_pd(&data2[i]);

								_mm256_store_pd(&data_result[i], _mm256_mul_pd(a, b));
							}
							for (size_t i = final_pos_size; i < size; i++)
							{
								data_result[i] = data1[i] * data2[i];
							}
						}
						else
						{
							size_t final_pos_rows = this->final_pos_rows;
							size_t final_pos_cols = this->final_pos_cols;

							size_t rows1 = this->actual_rows;
							size_t rows2 = other.actual_rows;

							for (size_t i = 0; i < final_pos_rows; i *= 4)
							{
								for (size_t j = 0; j < cols; j++)
								{
									__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
									__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);

									_mm256_store_pd(&data_result[j * rows + i], _mm256_mul_pd(a, b));
								}
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								for (size_t j = 0; j < final_pos_cols; j *= 4)
								{
									__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
										data1[(j + 1) * rows1 + i],
										data1[(j + 2) * rows1 + i],
										data1[(j + 3) * rows1 + i]);
									__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
										data2[(j + 1) * rows2 + i],
										data2[(j + 2) * rows2 + i],
										data2[(j + 3) * rows2 + i]);

									__m256d mul = _mm256_mul_pd(a, b);

									__m128d val1 = _mm256_extractf128_pd(mul, 1);
									__m128d val2 = _mm256_castpd256_pd128(mul);

									_mm_store_sd(&data_result[j * cols + i], val2);
									val2 = _mm_shuffle_pd(val2, val2, 1);
									_mm_store_sd(&data_result[(j + 1) * cols + i], val2);

									_mm_store_sd(&data_result[(j + 2) * cols + i], val1);
									val1 = _mm_shuffle_pd(val1, val1, 1);
									_mm_store_sd(&data_result[(j + 3) * cols + i], val1);
								}
								for (size_t j = final_pos_cols; j < cols; j++)
								{
									data_result[j * cols + i] = data1[j * rows1 + i] * data2[j * rows2 + i];
								}
							}
						}
					}
					else
					{
						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						size_t rows1 = this->actual_rows;
						size_t rows2 = other.actual_rows;

						for (size_t j = 0; j < final_pos_cols; j *= 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);
								__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
									data2[(j + 1) * rows2 + i],
									data2[(j + 2) * rows2 + i],
									data2[(j + 3) * rows2 + i]);

								_mm256_store_pd(&data_result[i * cols + j], _mm256_mul_pd(a, b));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < final_pos_rows; i *= 4)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
								__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);

								__m256d mul = _mm256_mul_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(mul, 1);
								__m128d val2 = _mm256_castpd256_pd128(mul);

								_mm_store_sd(&data_result[i * cols + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

								_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								data_result[i * cols + j] = data1[j * rows1 + i] * data2[j * rows2 + i];
							}
						}
					}
				}
				else
				{
					if constexpr (return_transposed)
					{
						size_t final_pos_rows = this->final_pos_rows;
						size_t final_pos_cols = this->final_pos_cols;

						size_t rows1 = this->actual_rows;
						size_t cols2 = other.actual_cols;

						for (size_t i = 0; i < final_pos_rows; i *= 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
								__m256d b = _mm256_setr_pd(data2[i * cols2 + j],
									data2[(i + 1) * cols2 + j],
									data2[(i + 2) * cols2 + j],
									data2[(i + 3) * cols2 + j]);

								_mm256_store_pd(&data_result[j * rows + i], _mm256_mul_pd(a, b));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < final_pos_cols; j *= 4)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);
								__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

								__m256d mul = _mm256_mul_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(mul, 1);
								__m128d val2 = _mm256_castpd256_pd128(mul);

								_mm_store_sd(&data_result[j * cols + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(j + 1) * cols + i], val2);

								_mm_store_sd(&data_result[(j + 2) * cols + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(j + 3) * cols + i], val1);
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								data_result[j * cols + i] = data1[j * rows1 + i] * data2[i * cols2 + j];
							}
						}
					}
					else
					{
						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						size_t rows1 = this->actual_rows;
						size_t cols2 = other.actual_cols;

						for (size_t j = 0; j < final_pos_cols; j *= 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);
								__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

								_mm256_store_pd(&data_result[i * cols + j], _mm256_mul_pd(a, b));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < final_pos_rows; i *= 4)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
								__m256d b = _mm256_setr_pd(data1[i * cols2 + j],
									data1[(i + 1) * cols2 + j],
									data1[(i + 2) * cols2 + j],
									data1[(i + 3) * cols2 + j]);

								__m256d mul = _mm256_mul_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(mul, 1);
								__m128d val2 = _mm256_castpd256_pd128(mul);

								_mm_store_sd(&data_result[i * cols + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

								_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								data_result[i * cols + j] = data1[j * rows1 + i] * data2[i * cols2 + j];
							}
						}
					}
				}
			}
			else
			{
				if constexpr (other_transposed)
				{
					size_t cols1 = this->actual_cols;
					size_t rows2 = other.actual_rows;

					if constexpr (return_transposed)
					{
						size_t final_pos_rows = this->final_pos_rows;
						size_t final_pos_cols = this->final_pos_cols;

						for (size_t i = 0; i < final_pos_rows; i *= 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);
								__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);
								_mm256_store_pd(&data_result[j * rows + i], _mm256_mul_pd(a, b));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; final_pos_cols; j *= 4)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
								__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
									data2[(j + 1) * rows2 + i],
									data2[(j + 2) * rows2 + i],
									data2[(j + 3) * rows2 + i]);

								__m256d mul = _mm256_mul_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(mul, 1);
								__m128d val2 = _mm256_castpd256_pd128(mul);

								_mm_store_sd(&data_result[j * rows + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(j + 1) * rows + i], val2);

								_mm_store_sd(&data_result[(j + 2) * rows + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(j + 3) * rows + i], val1);
							}
							for (size_t j = final_pos_cols; cols; j++)
							{
								data_result[j * rows + i] = data1[i * cols1 + j] * data2[j * rows2 + i];
							}
						}
					}
					else
					{
						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						for (size_t j = 0; j < final_pos_cols; j *= 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
								__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
									data2[(j + 1) * rows2 + i],
									data2[(j + 2) * rows2 + i],
									data2[(j + 3) * rows2 + i]);
								_mm256_store_pd(&data_result[i * cols + j], _mm256_mul_pd(a, b));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < final_pos_rows; i *= 4)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);
								__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);

								__m256d mul = _mm256_mul_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(mul, 1);
								__m128d val2 = _mm256_castpd256_pd128(mul);

								_mm_store_sd(&data_result[i * cols + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(i + 1) * rows + j], val2);

								_mm_store_sd(&data_result[(i + 2) * rows + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(i + 3) * rows + j], val1);
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								data_result[i * cols + j] = data1[i * cols1 + j] * data2[j * rows2 + i];
							}
						}
					}
				}
				else
				{
					if constexpr (return_transposed)
					{
						size_t cols1 = this->actual_cols;
						size_t cols2 = other.actual_cols;

						size_t final_pos_rows = this->final_pos_rows;
						size_t final_pos_cols = this->final_pos_cols;

						for (size_t i = 0; i < final_pos_rows; i *= 4)
						{
							for (size_t j = 0; j < rows; j++)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);
								__m256d b = _mm256_setr_pd(data2[i * cols2 + j],
									data2[(i + 1) * cols2 + j],
									data2[(i + 2) * cols2 + j],
									data2[(i + 3) * cols2 + j]);
								_mm256_store_pd(&data_result[j * rows + i], _mm256_mul_pd(a, b));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < final_pos_cols; j *= 4)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
								__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

								__m256d mul = _mm256_mul_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(mul, 1);
								__m128d val2 = _mm256_castpd256_pd128(mul);

								_mm_store_sd(&data_result[j * rows + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(j + 1) * rows + i], val2);

								_mm_store_sd(&data_result[(j + 2) * rows + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(j + 3) * rows + i], val1);
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								data_result[j * rows + i] = data1[i * cols1 + j] * data2[i * cols2 + j];
							}
						}
					}
					else
					{
						if constexpr (this_contiguous && other_contiguous)
						{
							size_t final_pos_size = this->final_pos_size;
							size_t size = this->_size;

							for (size_t i = 0; i < final_pos_size; i *= 4)
							{
								__m256d a = _mm256_load_pd(&data1[i]);
								__m256d b = _mm256_load_pd(&data2[i]);

								_mm256_store_pd(&data_result[i], _mm256_mul_pd(a, b));
							}
							for (size_t i = final_pos_size; i < size; i++)
							{
								data_result[i] = data1[i] * data2[i];
							}
						}
						else
						{
							size_t cols1 = this->actual_cols;
							size_t cols2 = other.actual_cols;

							size_t final_pos_cols = this->final_pos_cols;
							size_t final_pos_rows = this->final_pos_rows;

							for (size_t j = 0; j < final_pos_cols; j *= 4)
							{
								for (size_t i = 0; i < rows; i++)
								{
									__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
									__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

									_mm256_store_pd(&data_result[i * cols + j], _mm256_mul_pd(a, b));
								}
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								for (size_t i = 0; i < final_pos_rows; i *= 4)
								{
									__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
										data1[(i + 1) * cols1 + j],
										data1[(i + 2) * cols1 + j],
										data1[(i + 3) * cols1 + j]);
									__m256d b = _mm256_setr_pd(data2[i * cols2 + j],
										data2[(i + 1) * cols2 + j],
										data2[(i + 2) * cols2 + j],
										data2[(i + 3) * cols2 + j]);

									__m256d mul = _mm256_mul_pd(a, b);

									__m128d val1 = _mm256_extractf128_pd(mul, 1);
									__m128d val2 = _mm256_castpd256_pd128(mul);

									_mm_store_sd(&data_result[i * cols + j], val2);
									val2 = _mm_shuffle_pd(val2, val2, 1);
									_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

									_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
									val1 = _mm_shuffle_pd(val1, val1, 1);
									_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
								}
								for (size_t i = final_pos_rows; i < rows; i++)
								{
									data_result[i * cols + j] = data1[i * cols1 + j] * data2[i * cols2 + j];
								}
							}
						}
					}
				}
			}
			return result;
		}

		template<bool other_transposed, bool other_contiguous, bool call_destructor>
		void operator*=(const matrix<double, other_transposed, other_contiguous, call_destructor>& other)
		{
#ifdef _DEBUG
			if (other._cols != this->_cols || other._rows != this->_rows) throw std::invalid_argument("The dimensions of both matrices must be the same");
#else
#endif

			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;
			double* data2 = other._data;

			if constexpr (this_transposed)
			{
				if constexpr (other_transposed)
				{
					if constexpr (this_contiguous && other_contiguous)
					{
						size_t size = this->_size;

						size_t final_pos_size = this->final_pos_size;

						for (size_t i = 0; i < final_pos_size; i *= 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);
							__m256d b = _mm256_load_pd(&data2[i]);

							_mm256_store_pd(&data1[i], _mm256_mul_pd(a, b));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data1[i] *= data2[i];
						}
					}
					else
					{
						size_t final_pos_rows = this->final_pos_rows;
						size_t final_pos_cols = this->final_pos_cols;

						size_t rows1 = this->actual_rows;
						size_t rows2 = other.actual_rows;

						for (size_t i = 0; i < final_pos_rows; i *= 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
								__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);

								_mm256_store_pd(&data1[j * rows1 + i], _mm256_mul_pd(a, b));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < final_pos_cols; j *= 4)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);
								__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
									data2[(j + 1) * rows2 + i],
									data2[(j + 2) * rows2 + i],
									data2[(j + 3) * rows2 + i]);

								__m256d mul = _mm256_mul_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(mul, 1);
								__m128d val2 = _mm256_castpd256_pd128(mul);

								_mm_store_sd(&data1[j * rows1 + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data1[(j + 1) * rows1 + i], val2);

								_mm_store_sd(&data1[(j + 2) * rows1 + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data1[(j + 3) * rows1 + i], val1);
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								data1[j * rows1 + i] *= data2[j * rows2 + i];
							}
						}
					}
				}
				else
				{
					size_t final_pos_rows = this->final_pos_rows;
					size_t final_pos_cols = this->final_pos_cols;

					size_t rows1 = this->actual_rows;
					size_t cols2 = other.actual_cols;

					for (size_t i = 0; i < final_pos_rows; i *= 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
							__m256d b = _mm256_setr_pd(data2[i * cols2 + j],
								data2[(i + 1) * cols2 + j],
								data2[(i + 2) * cols2 + j],
								data2[(i + 3) * cols2 + j]);

							_mm256_store_pd(&data1[j * rows1 + i], _mm256_mul_pd(a, b));
						}
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						for (size_t j = 0; j < final_pos_cols; j *= 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
								data1[(j + 1) * rows1 + i],
								data1[(j + 2) * rows1 + i],
								data1[(j + 3) * rows1 + i]);
							__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

							__m256d mul = _mm256_mul_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(mul, 1);
							__m128d val2 = _mm256_castpd256_pd128(mul);

							_mm_store_sd(&data1[j * rows1 + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(j + 1) * rows1 + i], val2);

							_mm_store_sd(&data1[(j + 2) * rows1 + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(j + 3) * rows1 + i], val1);
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							data1[j * rows1 + i] *= data2[i * cols2 + j];
						}
					}
				}
			}
			else
			{
				if constexpr (other_transposed)
				{
					size_t cols1 = this->actual_cols;
					size_t rows2 = other.actual_rows;

					size_t final_pos_cols = this->final_pos_cols;
					size_t final_pos_rows = this->final_pos_rows;

					for (size_t j = 0; j < final_pos_cols; j *= 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
							__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
								data2[(j + 1) * rows2 + i],
								data2[(j + 2) * rows2 + i],
								data2[(j + 3) * rows2 + i]);
							_mm256_store_pd(&data1[i * cols1 + j], _mm256_mul_pd(a, b));
						}
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						for (size_t i = 0; i < final_pos_rows; i *= 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
								data1[(i + 1) * cols1 + j],
								data1[(i + 2) * cols1 + j],
								data1[(i + 3) * cols1 + j]);
							__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);

							__m256d mul = _mm256_mul_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(mul, 1);
							__m128d val2 = _mm256_castpd256_pd128(mul);

							_mm_store_sd(&data1[i * cols1 + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(i + 1) * cols1 + j], val2);

							_mm_store_sd(&data1[(i + 2) * cols1 + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(i + 3) * cols1 + j], val1);
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							data1[i * cols1 + j] *= data2[j * rows2 + i];
						}
					}
				}
				else
				{
					if constexpr (this_contiguous && other_contiguous)
					{
						size_t final_pos_size = this->final_pos_size;
						size_t size = this->_size;

						for (size_t i = 0; i < final_pos_size; i *= 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);
							__m256d b = _mm256_load_pd(&data2[i]);

							_mm256_store_pd(&data1[i], _mm256_mul_pd(a, b));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data1[i] *= data2[i];
						}
					}
					else
					{
						size_t cols1 = this->actual_cols;
						size_t cols2 = other.actual_cols;

						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						for (size_t j = 0; j < final_pos_cols; j *= 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
								__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

								_mm256_store_pd(&data1[i * cols1 + j], _mm256_mul_pd(a, b));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < final_pos_rows; i *= 4)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);
								__m256d b = _mm256_setr_pd(data2[i * cols2 + j],
									data2[(i + 1) * cols2 + j],
									data2[(i + 2) * cols2 + j],
									data2[(i + 3) * cols2 + j]);

								__m256d mul = _mm256_mul_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(mul, 1);
								__m128d val2 = _mm256_castpd256_pd128(mul);

								_mm_store_sd(&data1[i * cols1 + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data1[(i + 1) * cols1 + j], val2);

								_mm_store_sd(&data1[(i + 2) * cols1 + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data1[(i + 3) * cols1 + j], val1);
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								data1[i * cols1 + j] *= data2[i * cols2 + j];
							}
						}
					}
				}
			}
		}

		template<bool return_transposed = false>
		matrix<double, return_transposed> operator*(double num)
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			matrix<double, return_transposed, true> result(rows, cols);

			double* data_result = result._data;

			__m256d b = _mm256_set1_pd(num);

			if constexpr (this_transposed)
			{
				if constexpr (return_transposed)
				{
					if constexpr (this_contiguous)
					{
						size_t size = this->_size;

						size_t final_pos_size = this->final_pos_size;

						for (size_t i = 0; i < final_pos_size; i *= 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);

							_mm256_store_pd(&data_result[i], _mm256_mul_pd(a, b));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = data1[i] * num;
						}
					}
					else
					{
						size_t final_pos_rows = this->final_pos_rows;
						size_t final_pos_cols = this->final_pos_cols;

						size_t rows1 = this->actual_rows;

						for (size_t i = 0; i < final_pos_rows; i *= 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

								_mm256_store_pd(&data_result[j * rows + i], _mm256_mul_pd(a, b));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < final_pos_cols; j *= 4)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);

								__m256d mul = _mm256_mul_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(mul, 1);
								__m128d val2 = _mm256_castpd256_pd128(mul);

								_mm_store_sd(&data_result[j * cols + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(j + 1) * cols + i], val2);

								_mm_store_sd(&data_result[(j + 2) * cols + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(j + 3) * cols + i], val1);
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								data_result[j * cols + i] = data1[j * rows1 + i] * num;
							}
						}
					}
				}
				else
				{
					size_t final_pos_cols = this->final_pos_cols;
					size_t final_pos_rows = this->final_pos_rows;

					size_t rows1 = this->actual_rows;

					for (size_t j = 0; j < final_pos_cols; j *= 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
								data1[(j + 1) * rows1 + i],
								data1[(j + 2) * rows1 + i],
								data1[(j + 3) * rows1 + i]);

							_mm256_store_pd(&data_result[i * cols + j], _mm256_mul_pd(a, b));
						}
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						for (size_t i = 0; i < final_pos_rows; i *= 4)
						{
							__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

							__m256d mul = _mm256_mul_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(mul, 1);
							__m128d val2 = _mm256_castpd256_pd128(mul);

							_mm_store_sd(&data_result[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

							_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							data_result[i * cols + j] = data1[j * rows1 + i] * num;
						}
					}
				}
			}
			else
			{
				if constexpr (return_transposed)
				{
					size_t cols1 = this->actual_cols;

					size_t final_pos_rows = this->final_pos_rows;
					size_t final_pos_cols = this->final_pos_cols;

					for (size_t i = 0; i < final_pos_rows; i *= 4)
					{
						for (size_t j = 0; j < rows; j++)
						{
							__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
								data1[(i + 1) * cols1 + j],
								data1[(i + 2) * cols1 + j],
								data1[(i + 3) * cols1 + j]);

							_mm256_store_pd(&data_result[j * rows + i], _mm256_mul_pd(a, b));
						}
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						for (size_t j = 0; j < final_pos_cols; j *= 4)
						{
							__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

							__m256d mul = _mm256_mul_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(mul, 1);
							__m128d val2 = _mm256_castpd256_pd128(mul);

							_mm_store_sd(&data_result[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data_result[(j + 1) * rows + i], val2);

							_mm_store_sd(&data_result[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data_result[(j + 3) * rows + i], val1);
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							data_result[j * rows + i] = data1[i * cols1 + j] * num;
						}
					}
				}
				else
				{
					if constexpr (this_contiguous)
					{
						size_t final_pos_size = this->final_pos_size;
						size_t size = this->_size;

						for (size_t i = 0; i < final_pos_size; i *= 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);

							_mm256_store_pd(&data_result[i], _mm256_mul_pd(a, b));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = data1[i] * num;
						}
					}
					else
					{
						size_t cols1 = this->actual_cols;

						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						for (size_t j = 0; j < final_pos_cols; j *= 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

								_mm256_store_pd(&data_result[i * cols + j], _mm256_mul_pd(a, b));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < final_pos_rows; i *= 4)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);

								__m256d mul = _mm256_mul_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(mul, 1);
								__m128d val2 = _mm256_castpd256_pd128(mul);

								_mm_store_sd(&data_result[i * cols + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

								_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								data_result[i * cols + j] = data1[i * cols1 + j] * num;
							}
						}
					}
				}
			}
			return result;
		}

		void operator*=(double num)
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			__m256d b = _mm256_set1_pd(num);

			if constexpr (this_transposed)
			{
				if constexpr (this_contiguous)
				{
					size_t size = this->_size;

					size_t final_pos_size = this->final_pos_size;

					for (size_t i = 0; i < final_pos_size; i *= 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&data1[i], _mm256_mul_pd(a, b));
					}
					for (size_t i = final_pos_size; i < size; i++)
					{
						data1[i] *= num;
					}
				}
				else
				{
					size_t final_pos_rows = this->final_pos_rows;
					size_t final_pos_cols = this->final_pos_cols;

					size_t rows1 = this->actual_rows;

					for (size_t i = 0; i < final_pos_rows; i *= 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

							_mm256_store_pd(&data1[j * rows1 + i], _mm256_mul_pd(a, b));
						}
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						for (size_t j = 0; j < final_pos_cols; j *= 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
								data1[(j + 1) * rows1 + i],
								data1[(j + 2) * rows1 + i],
								data1[(j + 3) * rows1 + i]);

							__m256d mul = _mm256_mul_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(mul, 1);
							__m128d val2 = _mm256_castpd256_pd128(mul);

							_mm_store_sd(&data1[j * rows1 + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(j + 1) * rows1 + i], val2);

							_mm_store_sd(&data1[(j + 2) * rows1 + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(j + 3) * rows1 + i], val1);
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							data1[j * rows1 + i] *= num;
						}
					}
				}
			}
			else
			{
				if constexpr (this_contiguous)
				{
					size_t final_pos_size = this->final_pos_size;
					size_t size = this->_size;

					for (size_t i = 0; i < final_pos_size; i *= 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&data1[i], _mm256_mul_pd(a, b));
					}
					for (size_t i = final_pos_size; i < size; i++)
					{
						data1[i] *= num;
					}
				}
				else
				{
					size_t cols1 = this->actual_cols;

					size_t final_pos_cols = this->final_pos_cols;
					size_t final_pos_rows = this->final_pos_rows;

					for (size_t j = 0; j < final_pos_cols; j *= 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

							_mm256_store_pd(&data1[i * cols1 + j], _mm256_mul_pd(a, b));
						}
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						for (size_t i = 0; i < final_pos_rows; i *= 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
								data1[(i + 1) * cols1 + j],
								data1[(i + 2) * cols1 + j],
								data1[(i + 3) * cols1 + j]);

							__m256d mul = _mm256_mul_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(mul, 1);
							__m128d val2 = _mm256_castpd256_pd128(mul);

							_mm_store_sd(&data1[i * cols1 + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(i + 1) * cols1 + j], val2);

							_mm_store_sd(&data1[(i + 2) * cols1 + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(i + 3) * cols1 + j], val1);
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							data1[i * cols1 + j] *= num;
						}
					}
				}
			}
		}

		// /

		template<bool return_transposed = false, bool other_transposed, bool other_contiguous, bool call_destructor>
		matrix<double, return_transposed> operator/(const matrix<double, other_transposed, other_contiguous, call_destructor>& other)
		{
#ifdef _DEBUG
			if (other._cols != this->_cols || other._rows != this->_rows) throw std::invalid_argument("The dimensions of both matrices must be the same");
#else
#endif

			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;
			double* data2 = other._data;

			matrix<double, return_transposed, true> result(rows, cols);

			double* data_result = result._data;

			if constexpr (this_transposed)
			{
				if constexpr (other_transposed)
				{
					if constexpr (return_transposed)
					{
						if constexpr (this_contiguous && other_contiguous)
						{
							size_t size = this->_size;

							size_t final_pos_size = this->final_pos_size;

							for (size_t i = 0; i < final_pos_size; i += 4)
							{
								__m256d a = _mm256_load_pd(&data1[i]);
								__m256d b = _mm256_load_pd(&data2[i]);

								_mm256_store_pd(&data_result[i], _mm256_div_pd(a, b));
							}
							for (size_t i = final_pos_size; i < size; i++)
							{
								data_result[i] = data1[i] / data2[i];
							}
						}
						else
						{
							size_t final_pos_rows = this->final_pos_rows;
							size_t final_pos_cols = this->final_pos_cols;

							size_t rows1 = this->actual_rows;
							size_t rows2 = other.actual_rows;

							for (size_t i = 0; i < final_pos_rows; i += 4)
							{
								for (size_t j = 0; j < cols; j++)
								{
									__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
									__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);

									_mm256_store_pd(&data_result[j * rows + i], _mm256_div_pd(a, b));
								}
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								for (size_t j = 0; j < final_pos_cols; j += 4)
								{
									__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
										data1[(j + 1) * rows1 + i],
										data1[(j + 2) * rows1 + i],
										data1[(j + 3) * rows1 + i]);
									__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
										data2[(j + 1) * rows2 + i],
										data2[(j + 2) * rows2 + i],
										data2[(j + 3) * rows2 + i]);

									__m256d div = _mm256_div_pd(a, b);

									__m128d val1 = _mm256_extractf128_pd(div, 1);
									__m128d val2 = _mm256_castpd256_pd128(div);

									_mm_store_sd(&data_result[j * cols + i], val2);
									val2 = _mm_shuffle_pd(val2, val2, 1);
									_mm_store_sd(&data_result[(j + 1) * cols + i], val2);

									_mm_store_sd(&data_result[(j + 2) * cols + i], val1);
									val1 = _mm_shuffle_pd(val1, val1, 1);
									_mm_store_sd(&data_result[(j + 3) * cols + i], val1);
								}
								for (size_t j = final_pos_cols; j < cols; j++)
								{
									data_result[j * cols + i] = data1[j * rows1 + i] / data2[j * rows2 + i];
								}
							}
						}
					}
					else
					{
						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						size_t rows1 = this->actual_rows;
						size_t rows2 = other.actual_rows;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);
								__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
									data2[(j + 1) * rows2 + i],
									data2[(j + 2) * rows2 + i],
									data2[(j + 3) * rows2 + i]);

								_mm256_store_pd(&data_result[i * cols + j], _mm256_div_pd(a, b));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < final_pos_rows; i += 4)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
								__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);

								__m256d div = _mm256_div_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(div, 1);
								__m128d val2 = _mm256_castpd256_pd128(div);

								_mm_store_sd(&data_result[i * cols + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

								_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								data_result[i * cols + j] = data1[j * rows1 + i] / data2[j * rows2 + i];
							}
						}
					}
				}
				else
				{
					if constexpr (return_transposed)
					{
						size_t final_pos_rows = this->final_pos_rows;
						size_t final_pos_cols = this->final_pos_cols;

						size_t rows1 = this->actual_rows;
						size_t cols2 = other.actual_cols;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
								__m256d b = _mm256_setr_pd(data2[i * cols2 + j],
									data2[(i + 1) * cols2 + j],
									data2[(i + 2) * cols2 + j],
									data2[(i + 3) * cols2 + j]);

								_mm256_store_pd(&data_result[j * rows + i], _mm256_div_pd(a, b));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < final_pos_cols; j += 4)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);
								__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

								__m256d div = _mm256_div_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(div, 1);
								__m128d val2 = _mm256_castpd256_pd128(div);

								_mm_store_sd(&data_result[j * cols + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(j + 1) * cols + i], val2);

								_mm_store_sd(&data_result[(j + 2) * cols + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(j + 3) * cols + i], val1);
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								data_result[j * cols + i] = data1[j * rows1 + i] / data2[i * cols2 + j];
							}
						}
					}
					else
					{
						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						size_t rows1 = this->actual_rows;
						size_t cols2 = other.actual_cols;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);
								__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

								_mm256_store_pd(&data_result[i * cols + j], _mm256_div_pd(a, b));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < final_pos_rows; i += 4)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
								__m256d b = _mm256_setr_pd(data1[i * cols2 + j],
									data1[(i + 1) * cols2 + j],
									data1[(i + 2) * cols2 + j],
									data1[(i + 3) * cols2 + j]);

								__m256d div = _mm256_div_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(div, 1);
								__m128d val2 = _mm256_castpd256_pd128(div);

								_mm_store_sd(&data_result[i * cols + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

								_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								data_result[i * cols + j] = data1[j * rows1 + i] / data2[i * cols2 + j];
							}
						}
					}
				}
			}
			else
			{
				if constexpr (other_transposed)
				{
					size_t cols1 = this->actual_cols;
					size_t rows2 = other.actual_rows;

					if constexpr (return_transposed)
					{
						size_t final_pos_rows = this->final_pos_rows;
						size_t final_pos_cols = this->final_pos_cols;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);
								__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);
								_mm256_store_pd(&data_result[j * rows + i], _mm256_div_pd(a, b));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; final_pos_cols; j += 4)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
								__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
									data2[(j + 1) * rows2 + i],
									data2[(j + 2) * rows2 + i],
									data2[(j + 3) * rows2 + i]);

								__m256d div = _mm256_div_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(div, 1);
								__m128d val2 = _mm256_castpd256_pd128(div);

								_mm_store_sd(&data_result[j * rows + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(j + 1) * rows + i], val2);

								_mm_store_sd(&data_result[(j + 2) * rows + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(j + 3) * rows + i], val1);
							}
							for (size_t j = final_pos_cols; cols; j++)
							{
								data_result[j * rows + i] = data1[i * cols1 + j] / data2[j * rows2 + i];
							}
						}
					}
					else
					{
						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
								__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
									data2[(j + 1) * rows2 + i],
									data2[(j + 2) * rows2 + i],
									data2[(j + 3) * rows2 + i]);
								_mm256_store_pd(&data_result[i * cols + j], _mm256_div_pd(a, b));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < final_pos_rows; i += 4)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);
								__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);

								__m256d div = _mm256_div_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(div, 1);
								__m128d val2 = _mm256_castpd256_pd128(div);

								_mm_store_sd(&data_result[i * cols + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(i + 1) * rows + j], val2);

								_mm_store_sd(&data_result[(i + 2) * rows + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(i + 3) * rows + j], val1);
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								data_result[i * cols + j] = data1[i * cols1 + j] / data2[j * rows2 + i];
							}
						}
					}
				}
				else
				{
					if constexpr (return_transposed)
					{
						size_t cols1 = this->actual_cols;
						size_t cols2 = other.actual_cols;

						size_t final_pos_rows = this->final_pos_rows;
						size_t final_pos_cols = this->final_pos_cols;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < rows; j++)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);
								__m256d b = _mm256_setr_pd(data2[i * cols2 + j],
									data2[(i + 1) * cols2 + j],
									data2[(i + 2) * cols2 + j],
									data2[(i + 3) * cols2 + j]);
								_mm256_store_pd(&data_result[j * rows + i], _mm256_div_pd(a, b));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < final_pos_cols; j += 4)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
								__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

								__m256d div = _mm256_div_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(div, 1);
								__m128d val2 = _mm256_castpd256_pd128(div);

								_mm_store_sd(&data_result[j * rows + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(j + 1) * rows + i], val2);

								_mm_store_sd(&data_result[(j + 2) * rows + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(j + 3) * rows + i], val1);
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								data_result[j * rows + i] = data1[i * cols1 + j] / data2[i * cols2 + j];
							}
							}
						}
					else
					{
						if constexpr (this_contiguous && other_contiguous)
						{
							size_t final_pos_size = this->final_pos_size;
							size_t size = this->_size;

							for (size_t i = 0; i < final_pos_size; i += 4)
							{
								__m256d a = _mm256_load_pd(&data1[i]);
								__m256d b = _mm256_load_pd(&data2[i]);

								_mm256_store_pd(&data_result[i], _mm256_div_pd(a, b));
							}
							for (size_t i = final_pos_size; i < size; i++)
							{
								data_result[i] = data1[i] / data2[i];
							}
						}
						else
						{
							size_t cols1 = this->actual_cols;
							size_t cols2 = other.actual_cols;

							size_t final_pos_cols = this->final_pos_cols;
							size_t final_pos_rows = this->final_pos_rows;

							for (size_t j = 0; j < final_pos_cols; j += 4)
							{
								for (size_t i = 0; i < rows; i++)
								{
									__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
									__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

									_mm256_store_pd(&data_result[i * cols + j], _mm256_div_pd(a, b));
								}
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								for (size_t i = 0; i < final_pos_rows; i += 4)
								{
									__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
										data1[(i + 1) * cols1 + j],
										data1[(i + 2) * cols1 + j],
										data1[(i + 3) * cols1 + j]);
									__m256d b = _mm256_setr_pd(data2[i * cols2 + j],
										data2[(i + 1) * cols2 + j],
										data2[(i + 2) * cols2 + j],
										data2[(i + 3) * cols2 + j]);

									__m256d div = _mm256_div_pd(a, b);

									__m128d val1 = _mm256_extractf128_pd(div, 1);
									__m128d val2 = _mm256_castpd256_pd128(div);

									_mm_store_sd(&data_result[i * cols + j], val2);
									val2 = _mm_shuffle_pd(val2, val2, 1);
									_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

									_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
									val1 = _mm_shuffle_pd(val1, val1, 1);
									_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
								}
								for (size_t i = final_pos_rows; i < rows; i++)
								{
									data_result[i * cols + j] = data1[i * cols1 + j] / data2[i * cols2 + j];
								}
							}
						}
					}
					}
				}
			return result;
			}

		template<bool other_transposed, bool other_contiguous, bool call_destructor>
		void operator/=(const matrix<double, other_transposed, other_contiguous, call_destructor>& other)
		{
#ifdef _DEBUG
			if (other._cols != this->_cols || other._rows != this->_rows) throw std::invalid_argument("The dimensions of both matrices must be the same");
#else
#endif

			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;
			double* data2 = other._data;

			if constexpr (this_transposed)
			{
				if constexpr (other_transposed)
				{
					if constexpr (this_contiguous && other_contiguous)
					{
						size_t size = this->_size;

						size_t final_pos_size = this->final_pos_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);
							__m256d b = _mm256_load_pd(&data2[i]);

							_mm256_store_pd(&data1[i], _mm256_div_pd(a, b));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data1[i] /= data2[i];
						}
					}
					else
					{
						size_t final_pos_rows = this->final_pos_rows;
						size_t final_pos_cols = this->final_pos_cols;

						size_t rows1 = this->actual_rows;
						size_t rows2 = other.actual_rows;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
								__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);

								_mm256_store_pd(&data1[j * rows1 + i], _mm256_div_pd(a, b));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < final_pos_cols; j += 4)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);
								__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
									data2[(j + 1) * rows2 + i],
									data2[(j + 2) * rows2 + i],
									data2[(j + 3) * rows2 + i]);

								__m256d div = _mm256_div_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(div, 1);
								__m128d val2 = _mm256_castpd256_pd128(div);

								_mm_store_sd(&data1[j * rows1 + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data1[(j + 1) * rows1 + i], val2);

								_mm_store_sd(&data1[(j + 2) * rows1 + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data1[(j + 3) * rows1 + i], val1);
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								data1[j * rows1 + i] /= data2[j * rows2 + i];
							}
						}
					}
				}
				else
				{
					size_t final_pos_rows = this->final_pos_rows;
					size_t final_pos_cols = this->final_pos_cols;

					size_t rows1 = this->actual_rows;
					size_t cols2 = other.actual_cols;

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
							__m256d b = _mm256_setr_pd(data2[i * cols2 + j],
								data2[(i + 1) * cols2 + j],
								data2[(i + 2) * cols2 + j],
								data2[(i + 3) * cols2 + j]);

							_mm256_store_pd(&data1[j * rows1 + i], _mm256_div_pd(a, b));
						}
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
								data1[(j + 1) * rows1 + i],
								data1[(j + 2) * rows1 + i],
								data1[(j + 3) * rows1 + i]);
							__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

							__m256d div = _mm256_div_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(div, 1);
							__m128d val2 = _mm256_castpd256_pd128(div);

							_mm_store_sd(&data1[j * rows1 + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(j + 1) * rows1 + i], val2);

							_mm_store_sd(&data1[(j + 2) * rows1 + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(j + 3) * rows1 + i], val1);
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							data1[j * rows1 + i] /= data2[i * cols2 + j];
						}
					}
				}
			}
			else
			{
				if constexpr (other_transposed)
				{
					size_t cols1 = this->actual_cols;
					size_t rows2 = other.actual_rows;

					size_t final_pos_cols = this->final_pos_cols;
					size_t final_pos_rows = this->final_pos_rows;

					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
							__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
								data2[(j + 1) * rows2 + i],
								data2[(j + 2) * rows2 + i],
								data2[(j + 3) * rows2 + i]);
							_mm256_store_pd(&data1[i * cols1 + j], _mm256_div_pd(a, b));
						}
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
								data1[(i + 1) * cols1 + j],
								data1[(i + 2) * cols1 + j],
								data1[(i + 3) * cols1 + j]);
							__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);

							__m256d div = _mm256_div_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(div, 1);
							__m128d val2 = _mm256_castpd256_pd128(div);

							_mm_store_sd(&data1[i * cols1 + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(i + 1) * cols1 + j], val2);

							_mm_store_sd(&data1[(i + 2) * cols1 + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(i + 3) * cols1 + j], val1);
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							data1[i * cols1 + j] /= data2[j * rows2 + i];
						}
					}
				}
				else
				{
					if constexpr (this_contiguous && other_contiguous)
					{
						size_t final_pos_size = this->final_pos_size;
						size_t size = this->_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);
							__m256d b = _mm256_load_pd(&data2[i]);

							_mm256_store_pd(&data1[i], _mm256_div_pd(a, b));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data1[i] /= data2[i];
						}
					}
					else
					{
						size_t cols1 = this->actual_cols;
						size_t cols2 = other.actual_cols;

						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
								__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

								_mm256_store_pd(&data1[i * cols1 + j], _mm256_div_pd(a, b));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < final_pos_rows; i += 4)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);
								__m256d b = _mm256_setr_pd(data2[i * cols2 + j],
									data2[(i + 1) * cols2 + j],
									data2[(i + 2) * cols2 + j],
									data2[(i + 3) * cols2 + j]);

								__m256d div = _mm256_div_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(div, 1);
								__m128d val2 = _mm256_castpd256_pd128(div);

								_mm_store_sd(&data1[i * cols1 + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data1[(i + 1) * cols1 + j], val2);

								_mm_store_sd(&data1[(i + 2) * cols1 + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data1[(i + 3) * cols1 + j], val1);
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								data1[i * cols1 + j] /= data2[i * cols2 + j];
							}
						}
					}
				}
			}
		}

		template<bool return_transposed = false>
		matrix<double, return_transposed> operator/(double num)
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			matrix<double, return_transposed, true> result(rows, cols);

			double* data_result = result._data;

			__m256d b = _mm256_set1_pd(num);

			if constexpr (this_transposed)
			{
				if constexpr (return_transposed)
				{
					if constexpr (this_contiguous)
					{
						size_t size = this->_size;

						size_t final_pos_size = this->final_pos_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);

							_mm256_store_pd(&data_result[i], _mm256_div_pd(a, b));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = data1[i] / num;
						}
					}
					else
					{
						size_t final_pos_rows = this->final_pos_rows;
						size_t final_pos_cols = this->final_pos_cols;

						size_t rows1 = this->actual_rows;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

								_mm256_store_pd(&data_result[j * rows + i], _mm256_div_pd(a, b));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < final_pos_cols; j += 4)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);

								__m256d div = _mm256_div_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(div, 1);
								__m128d val2 = _mm256_castpd256_pd128(div);

								_mm_store_sd(&data_result[j * cols + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(j + 1) * cols + i], val2);

								_mm_store_sd(&data_result[(j + 2) * cols + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(j + 3) * cols + i], val1);
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								data_result[j * cols + i] = data1[j * rows1 + i] / num;
							}
						}
					}
				}
				else
				{
					size_t final_pos_cols = this->final_pos_cols;
					size_t final_pos_rows = this->final_pos_rows;

					size_t rows1 = this->actual_rows;

					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
								data1[(j + 1) * rows1 + i],
								data1[(j + 2) * rows1 + i],
								data1[(j + 3) * rows1 + i]);

							_mm256_store_pd(&data_result[i * cols + j], _mm256_div_pd(a, b));
						}
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

							__m256d div = _mm256_div_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(div, 1);
							__m128d val2 = _mm256_castpd256_pd128(div);

							_mm_store_sd(&data_result[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

							_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							data_result[i * cols + j] = data1[j * rows1 + i] / num;
						}
					}
				}
			}
			else
			{
				if constexpr (return_transposed)
				{
					size_t cols1 = this->actual_cols;

					size_t final_pos_rows = this->final_pos_rows;
					size_t final_pos_cols = this->final_pos_cols;

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						for (size_t j = 0; j < rows; j++)
						{
							__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
								data1[(i + 1) * cols1 + j],
								data1[(i + 2) * cols1 + j],
								data1[(i + 3) * cols1 + j]);

							_mm256_store_pd(&data_result[j * rows + i], _mm256_div_pd(a, b));
						}
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

							__m256d div = _mm256_div_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(div, 1);
							__m128d val2 = _mm256_castpd256_pd128(div);

							_mm_store_sd(&data_result[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data_result[(j + 1) * rows + i], val2);

							_mm_store_sd(&data_result[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data_result[(j + 3) * rows + i], val1);
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							data_result[j * rows + i] = data1[i * cols1 + j] / num;
						}
					}
				}
				else
				{
					if constexpr (this_contiguous)
					{
						size_t final_pos_size = this->final_pos_size;
						size_t size = this->_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);

							_mm256_store_pd(&data_result[i], _mm256_div_pd(a, b));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = data1[i] / num;
						}
					}
					else
					{
						size_t cols1 = this->actual_cols;

						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

								_mm256_store_pd(&data_result[i * cols + j], _mm256_div_pd(a, b));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < final_pos_rows; i += 4)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);

								__m256d div = _mm256_div_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(div, 1);
								__m128d val2 = _mm256_castpd256_pd128(div);

								_mm_store_sd(&data_result[i * cols + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

								_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								data_result[i * cols + j] = data1[i * cols1 + j] / num;
							}
						}
					}
				}
			}
			return result;
		}

		void operator/=(double num)
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			__m256d b = _mm256_set1_pd(num);

			if constexpr (this_transposed)
			{
				if constexpr (this_contiguous)
				{
					size_t size = this->_size;

					size_t final_pos_size = this->final_pos_size;

					for (size_t i = 0; i < final_pos_size; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&data1[i], _mm256_div_pd(a, b));
					}
					for (size_t i = final_pos_size; i < size; i++)
					{
						data1[i] /= num;
					}
				}
				else
				{
					size_t final_pos_rows = this->final_pos_rows;
					size_t final_pos_cols = this->final_pos_cols;

					size_t rows1 = this->actual_rows;

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

							_mm256_store_pd(&data1[j * rows1 + i], _mm256_div_pd(a, b));
						}
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
								data1[(j + 1) * rows1 + i],
								data1[(j + 2) * rows1 + i],
								data1[(j + 3) * rows1 + i]);

							__m256d div = _mm256_div_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(div, 1);
							__m128d val2 = _mm256_castpd256_pd128(div);

							_mm_store_sd(&data1[j * rows1 + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(j + 1) * rows1 + i], val2);

							_mm_store_sd(&data1[(j + 2) * rows1 + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(j + 3) * rows1 + i], val1);
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							data1[j * rows1 + i] /= num;
						}
					}
				}
			}
			else
			{
				if constexpr (this_contiguous)
				{
					size_t final_pos_size = this->final_pos_size;
					size_t size = this->_size;

					for (size_t i = 0; i < final_pos_size; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&data1[i], _mm256_div_pd(a, b));
					}
					for (size_t i = final_pos_size; i < size; i++)
					{
						data1[i] /= num;
					}
				}
				else
				{
					size_t cols1 = this->actual_cols;

					size_t final_pos_cols = this->final_pos_cols;
					size_t final_pos_rows = this->final_pos_rows;

					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

							_mm256_store_pd(&data1[i * cols1 + j], _mm256_div_pd(a, b));
						}
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
								data1[(i + 1) * cols1 + j],
								data1[(i + 2) * cols1 + j],
								data1[(i + 3) * cols1 + j]);

							__m256d div = _mm256_div_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(div, 1);
							__m128d val2 = _mm256_castpd256_pd128(div);

							_mm_store_sd(&data1[i * cols1 + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(i + 1) * cols1 + j], val2);

							_mm_store_sd(&data1[(i + 2) * cols1 + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(i + 3) * cols1 + j], val1);
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							data1[i * cols1 + j] /= num;
						}
					}
				}
			}
		}

		// ==

		template<bool return_transposed = false, bool other_transposed, bool other_contiguous, bool call_destructor>
		matrix<bool, return_transposed> operator==(const matrix<double, other_transposed, other_contiguous, call_destructor>& other)
		{
#ifdef _DEBUG
			if (other._cols != this->_cols || other._rows != this->_rows) throw std::invalid_argument("The dimensions of both matrices must be the same");
#else
#endif

			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;
			double* data2 = other._data;

			matrix<bool, return_transposed, true> result(rows, cols);

			bool* data_result = result._data;

			if constexpr (this_transposed)
			{
				if constexpr (other_transposed)
				{
					if constexpr (return_transposed)
					{
						if constexpr (this_contiguous && other_contiguous)
						{
							size_t size = this->_size;

							size_t final_pos_size = this->final_pos_size;

							for (size_t i = 0; i < final_pos_size; i += 4)
							{
								__m256d a = _mm256_load_pd(&data1[i]);
								__m256d b = _mm256_load_pd(&data2[i]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_EQ_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
							}
							for (size_t i = final_pos_size; i < size; i++)
							{
								data_result[i] = data1[i] == data2[i] ? True : False;
							}
						}
						else
						{
							size_t final_pos_rows = this->final_pos_rows;

							size_t rows1 = this->actual_rows;
							size_t rows2 = other.actual_rows;

							for (size_t i = 0; i < final_pos_rows; i += 4)
							{
								for (size_t j = 0; j < cols; j++)
								{
									__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
									__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);

									__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_EQ_OQ));

									__m128i mask1 = _mm256_castsi256_si128(mask);
									__m128i mask2 = _mm256_extracti128_si256(mask, 1);

									mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
									mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

									__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

									_mm_store_ss(reinterpret_cast<float*>(&data_result[j * rows + i]), _mm_castsi128_ps(mask_result));

								}
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								for (size_t j = 0; j < cols; j++)
								{
									data_result[j * cols + i] = data1[j * rows1 + i] == data2[j * rows2 + i] ? True : False;
								}
							}
						}
					}
					else
					{
						size_t rows1 = this->actual_rows;
						size_t rows2 = other.actual_rows;

						for (size_t i = 0; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								data_result[i * cols + j] = data1[j * rows1 + i] == data2[j * rows2 + i] ? True : False;
							}
						}
					}
				}
				else
				{
					if constexpr (return_transposed)
					{
						size_t final_pos_rows = this->final_pos_rows;

						size_t rows1 = this->actual_rows;
						size_t cols2 = other.actual_cols;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
								__m256d b = _mm256_setr_pd(data2[i * cols2 + j],
									data2[(i + 1) * cols2 + j],
									data2[(i + 2) * cols2 + j],
									data2[(i + 3) * cols2 + j]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_EQ_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[j * rows + i]), _mm_castsi128_ps(mask_result));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								data_result[j * cols + i] = data1[j * rows1 + i] == data2[i * cols2 + j] ? True : False;
							}
						}
					}
					else
					{
						size_t final_pos_cols = this->final_pos_cols;

						size_t rows1 = this->actual_rows;
						size_t cols2 = other.actual_cols;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);
								__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_EQ_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[i * cols + j]), _mm_castsi128_ps(mask_result));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < rows; i++)
							{
								data_result[i * cols + j] = data1[j * rows1 + i] == data2[i * cols2 + j] ? True : False;
							}
						}
					}
				}
			}
			else
			{
				if constexpr (other_transposed)
				{
					size_t cols1 = this->actual_cols;
					size_t rows2 = other.actual_rows;

					if constexpr (return_transposed)
					{
						size_t final_pos_rows = this->final_pos_rows;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);
								__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);
								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_EQ_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[j * rows + i]), _mm_castsi128_ps(mask_result));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; cols; j++)
							{
								data_result[j * rows + i] = data1[i * cols1 + j] == data2[j * rows2 + i] ? True : False;
							}
						}
					}
					else
					{
						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
								__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
									data2[(j + 1) * rows2 + i],
									data2[(j + 2) * rows2 + i],
									data2[(j + 3) * rows2 + i]);
								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_EQ_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[i * cols + j]), _mm_castsi128_ps(mask_result));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < rows; i++)
							{
								data_result[i * cols + j] = data1[i * cols1 + j] == data2[j * rows2 + i] ? True : False;
							}
						}
					}
				}
				else
				{
					if constexpr (return_transposed)
					{
						size_t cols1 = this->actual_cols;
						size_t cols2 = other.actual_cols;

						for (size_t i = 0; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								data_result[j * rows + i] = data1[i * cols1 + j] == data2[i * cols2 + j] ? True : False;
							}
						}
					}
					else
					{
						if constexpr (this_contiguous && other_contiguous)
						{
							size_t final_pos_size = this->final_pos_size;
							size_t size = this->_size;

							for (size_t i = 0; i < final_pos_size; i += 4)
							{
								__m256d a = _mm256_load_pd(&data1[i]);
								__m256d b = _mm256_load_pd(&data2[i]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_EQ_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
							}
							for (size_t i = final_pos_size; i < size; i++)
							{
								data_result[i] = data1[i] == data2[i] ? True : False;
							}
						}
						else
						{
							size_t cols1 = this->actual_cols;
							size_t cols2 = other.actual_cols;

							size_t final_pos_cols = this->final_pos_cols;

							for (size_t j = 0; j < final_pos_cols; j += 4)
							{
								for (size_t i = 0; i < rows; i++)
								{
									__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
									__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

									__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_EQ_OQ));

									__m128i mask1 = _mm256_castsi256_si128(mask);
									__m128i mask2 = _mm256_extracti128_si256(mask, 1);

									mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
									mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

									__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

									_mm_store_ss(reinterpret_cast<float*>(&data_result[i * cols + j]), _mm_castsi128_ps(mask_result));
								}
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								for (size_t i = 0; i < rows; i++)
								{
									data_result[i * cols + j] = data1[i * cols1 + j] == data2[i * cols2 + j] ? True : False;
								}
							}
						}
					}
				}
			}
			return result;
		}

		template<bool return_transposed = false>
		matrix<bool, return_transposed> operator==(double num)
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			matrix<bool, return_transposed, true> result(rows, cols);

			bool* data_result = result._data;

			__m256d b = _mm256_set1_pd(num);

			if constexpr (this_transposed)
			{
				if constexpr (return_transposed)
				{
					if constexpr (this_contiguous)
					{
						size_t size = this->_size;

						size_t final_pos_size = this->final_pos_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);

							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_EQ_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = data1[i] == num ? True : False;
						}
					}
					else
					{
						size_t final_pos_rows = this->final_pos_rows;

						size_t rows1 = this->actual_rows;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_EQ_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[j * rows + i]), _mm_castsi128_ps(mask_result));

							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								data_result[j * cols + i] = data1[j * rows1 + i] == num ? True : False;
							}
						}
					}
				}
				else
				{
					size_t rows1 = this->actual_rows;

					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							data_result[i * cols + j] = data1[j * rows1 + i] == num ? True : False;
						}
					}
				}
			}
			else
			{
				if constexpr (return_transposed)
				{
					size_t cols1 = this->actual_cols;

					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							data_result[j * rows + i] = data1[i * cols1 + j] == num ? True : False;
						}
					}
				}
				else
				{
					if constexpr (this_contiguous)
					{
						size_t final_pos_size = this->final_pos_size;
						size_t size = this->_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);

							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_EQ_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = data1[i] == num ? True : False;
						}
					}
					else
					{
						size_t cols1 = this->actual_cols;

						size_t final_pos_cols = this->final_pos_cols;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_EQ_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[i * cols + j]), _mm_castsi128_ps(mask_result));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < rows; i++)
							{
								data_result[i * cols + j] = data1[i * cols1 + j] == num ? True : False;
							}
						}
					}
				}
			}
			return result;
		}

		// !=

		template<bool return_transposed = false, bool other_transposed, bool other_contiguous, bool call_destructor>
		matrix<bool, return_transposed> operator!=(const matrix<double, other_transposed, other_contiguous, call_destructor>& other)
		{
#ifdef _DEBUG
			if (other._cols != this->_cols || other._rows != this->_rows) throw std::invalid_argument("The dimensions of both matrices must be the same");
#else
#endif

			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;
			double* data2 = other._data;

			matrix<bool, return_transposed, true> result(rows, cols);

			bool* data_result = result._data;

			if constexpr (this_transposed)
			{
				if constexpr (other_transposed)
				{
					if constexpr (return_transposed)
					{
						if constexpr (this_contiguous && other_contiguous)
						{
							size_t size = this->_size;

							size_t final_pos_size = this->final_pos_size;

							for (size_t i = 0; i < final_pos_size; i += 4)
							{
								__m256d a = _mm256_load_pd(&data1[i]);
								__m256d b = _mm256_load_pd(&data2[i]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_NEQ_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
							}
							for (size_t i = final_pos_size; i < size; i++)
							{
								data_result[i] = data1[i] != data2[i] ? True : False;
							}
						}
						else
						{
							size_t final_pos_rows = this->final_pos_rows;

							size_t rows1 = this->actual_rows;
							size_t rows2 = other.actual_rows;

							for (size_t i = 0; i < final_pos_rows; i += 4)
							{
								for (size_t j = 0; j < cols; j++)
								{
									__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
									__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);

									__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_NEQ_OQ));

									__m128i mask1 = _mm256_castsi256_si128(mask);
									__m128i mask2 = _mm256_extracti128_si256(mask, 1);

									mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
									mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

									__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

									_mm_store_ss(reinterpret_cast<float*>(&data_result[j * rows + i]), _mm_castsi128_ps(mask_result));

								}
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								for (size_t j = 0; j < cols; j++)
								{
									data_result[j * cols + i] = data1[j * rows1 + i] != data2[j * rows2 + i] ? True : False;
								}
							}
						}
					}
					else
					{
						size_t rows1 = this->actual_rows;
						size_t rows2 = other.actual_rows;

						for (size_t i = 0; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								data_result[i * cols + j] = data1[j * rows1 + i] != data2[j * rows2 + i] ? True : False;
							}
						}
					}
				}
				else
				{
					if constexpr (return_transposed)
					{
						size_t final_pos_rows = this->final_pos_rows;

						size_t rows1 = this->actual_rows;
						size_t cols2 = other.actual_cols;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
								__m256d b = _mm256_setr_pd(data2[i * cols2 + j],
									data2[(i + 1) * cols2 + j],
									data2[(i + 2) * cols2 + j],
									data2[(i + 3) * cols2 + j]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_NEQ_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[j * rows + i]), _mm_castsi128_ps(mask_result));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								data_result[j * cols + i] = data1[j * rows1 + i] != data2[i * cols2 + j] ? True : False;
							}
						}
					}
					else
					{
						size_t final_pos_cols = this->final_pos_cols;

						size_t rows1 = this->actual_rows;
						size_t cols2 = other.actual_cols;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);
								__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_NEQ_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[i * cols + j]), _mm_castsi128_ps(mask_result));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < rows; i++)
							{
								data_result[i * cols + j] = data1[j * rows1 + i] != data2[i * cols2 + j] ? True : False;
							}
						}
					}
				}
			}
			else
			{
				if constexpr (other_transposed)
				{
					size_t cols1 = this->actual_cols;
					size_t rows2 = other.actual_rows;

					if constexpr (return_transposed)
					{
						size_t final_pos_rows = this->final_pos_rows;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);
								__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);
								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_NEQ_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[j * rows + i]), _mm_castsi128_ps(mask_result));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; cols; j++)
							{
								data_result[j * rows + i] = data1[i * cols1 + j] != data2[j * rows2 + i] ? True : False;
							}
						}
					}
					else
					{
						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
								__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
									data2[(j + 1) * rows2 + i],
									data2[(j + 2) * rows2 + i],
									data2[(j + 3) * rows2 + i]);
								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_NEQ_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[i * cols + j]), _mm_castsi128_ps(mask_result));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < rows; i++)
							{
								data_result[i * cols + j] = data1[i * cols1 + j] != data2[j * rows2 + i] ? True : False;
							}
						}
					}
				}
				else
				{
					if constexpr (return_transposed)
					{
						size_t cols1 = this->actual_cols;
						size_t cols2 = other.actual_cols;

						for (size_t i = 0; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								data_result[j * rows + i] = data1[i * cols1 + j] != data2[i * cols2 + j] ? True : False;
							}
						}
					}
					else
					{
						if constexpr (this_contiguous && other_contiguous)
						{
							size_t final_pos_size = this->final_pos_size;
							size_t size = this->_size;

							for (size_t i = 0; i < final_pos_size; i += 4)
							{
								__m256d a = _mm256_load_pd(&data1[i]);
								__m256d b = _mm256_load_pd(&data2[i]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_NEQ_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
							}
							for (size_t i = final_pos_size; i < size; i++)
							{
								data_result[i] = data1[i] != data2[i] ? True : False;
							}
						}
						else
						{
							size_t cols1 = this->actual_cols;
							size_t cols2 = other.actual_cols;

							size_t final_pos_cols = this->final_pos_cols;

							for (size_t j = 0; j < final_pos_cols; j += 4)
							{
								for (size_t i = 0; i < rows; i++)
								{
									__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
									__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

									__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_NEQ_OQ));

									__m128i mask1 = _mm256_castsi256_si128(mask);
									__m128i mask2 = _mm256_extracti128_si256(mask, 1);

									mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
									mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

									__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

									_mm_store_ss(reinterpret_cast<float*>(&data_result[i * cols + j]), _mm_castsi128_ps(mask_result));
								}
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								for (size_t i = 0; i < rows; i++)
								{
									data_result[i * cols + j] = data1[i * cols1 + j] != data2[i * cols2 + j] ? True : False;
								}
							}
						}
					}
				}
			}
			return result;
		}

		template<bool return_transposed = false>
		matrix<bool, return_transposed> operator!=(double num)
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			matrix<bool, return_transposed, true> result(rows, cols);

			bool* data_result = result._data;

			__m256d b = _mm256_set1_pd(num);

			if constexpr (this_transposed)
			{
				if constexpr (return_transposed)
				{
					if constexpr (this_contiguous)
					{
						size_t size = this->_size;

						size_t final_pos_size = this->final_pos_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);

							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_NEQ_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = data1[i] != num ? True : False;
						}
					}
					else
					{
						size_t final_pos_rows = this->final_pos_rows;

						size_t rows1 = this->actual_rows;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_NEQ_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[j * rows + i]), _mm_castsi128_ps(mask_result));

							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								data_result[j * cols + i] = data1[j * rows1 + i] != num ? True : False;
							}
						}
					}
				}
				else
				{
					size_t rows1 = this->actual_rows;

					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							data_result[i * cols + j] = data1[j * rows1 + i] != num ? True : False;
						}
					}
				}
			}
			else
			{
				if constexpr (return_transposed)
				{
					size_t cols1 = this->actual_cols;

					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							data_result[j * rows + i] = data1[i * cols1 + j] != num ? True : False;
						}
					}
				}
				else
				{
					if constexpr (this_contiguous)
					{
						size_t final_pos_size = this->final_pos_size;
						size_t size = this->_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);

							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_NEQ_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = data1[i] != num ? True : False;
						}
					}
					else
					{
						size_t cols1 = this->actual_cols;

						size_t final_pos_cols = this->final_pos_cols;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_NEQ_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[i * cols + j]), _mm_castsi128_ps(mask_result));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < rows; i++)
							{
								data_result[i * cols + j] = data1[i * cols1 + j] != num ? True : False;
							}
						}
					}
				}
			}
			return result;
		}

		// >

		template<bool return_transposed = false, bool other_transposed, bool other_contiguous, bool call_destructor>
		matrix<bool, return_transposed> operator>(const matrix<double, other_transposed, other_contiguous, call_destructor>& other)
		{
#ifdef _DEBUG
			if (other._cols > this->_cols || other._rows > this->_rows) throw std::invalid_argument("The dimensions of both matrices must be the same");
#else
#endif

			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;
			double* data2 = other._data;

			matrix<bool, return_transposed, true> result(rows, cols);

			bool* data_result = result._data;

			if constexpr (this_transposed)
			{
				if constexpr (other_transposed)
				{
					if constexpr (return_transposed)
					{
						if constexpr (this_contiguous && other_contiguous)
						{
							size_t size = this->_size;

							size_t final_pos_size = this->final_pos_size;

							for (size_t i = 0; i < final_pos_size; i += 4)
							{
								__m256d a = _mm256_load_pd(&data1[i]);
								__m256d b = _mm256_load_pd(&data2[i]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GT_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
							}
							for (size_t i = final_pos_size; i < size; i++)
							{
								data_result[i] = data1[i] > data2[i] ? True : False;
							}
						}
						else
						{
							size_t final_pos_rows = this->final_pos_rows;

							size_t rows1 = this->actual_rows;
							size_t rows2 = other.actual_rows;

							for (size_t i = 0; i < final_pos_rows; i += 4)
							{
								for (size_t j = 0; j < cols; j++)
								{
									__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
									__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);

									__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GT_OQ));

									__m128i mask1 = _mm256_castsi256_si128(mask);
									__m128i mask2 = _mm256_extracti128_si256(mask, 1);

									mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
									mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

									__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

									_mm_store_ss(reinterpret_cast<float*>(&data_result[j * rows + i]), _mm_castsi128_ps(mask_result));

								}
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								for (size_t j = 0; j < cols; j++)
								{
									data_result[j * cols + i] = data1[j * rows1 + i] > data2[j * rows2 + i] ? True : False;
								}
							}
						}
					}
					else
					{
						size_t rows1 = this->actual_rows;
						size_t rows2 = other.actual_rows;

						for (size_t i = 0; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								data_result[i * cols + j] = data1[j * rows1 + i] > data2[j * rows2 + i] ? True : False;
						}
					}
				}
			}
				else
				{
					if constexpr (return_transposed)
					{
						size_t final_pos_rows = this->final_pos_rows;

						size_t rows1 = this->actual_rows;
						size_t cols2 = other.actual_cols;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
								__m256d b = _mm256_setr_pd(data2[i * cols2 + j],
									data2[(i + 1) * cols2 + j],
									data2[(i + 2) * cols2 + j],
									data2[(i + 3) * cols2 + j]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GT_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[j * rows + i]), _mm_castsi128_ps(mask_result));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								data_result[j * cols + i] = data1[j * rows1 + i] > data2[i * cols2 + j] ? True : False;
							}
						}
					}
					else
					{
						size_t final_pos_cols = this->final_pos_cols;

						size_t rows1 = this->actual_rows;
						size_t cols2 = other.actual_cols;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);
								__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GT_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[i * cols + j]), _mm_castsi128_ps(mask_result));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < rows; i++)
							{
								data_result[i * cols + j] = data1[j * rows1 + i] > data2[i * cols2 + j] ? True : False;
							}
						}
					}
				}
		}
			else
			{
				if constexpr (other_transposed)
				{
					size_t cols1 = this->actual_cols;
					size_t rows2 = other.actual_rows;

					if constexpr (return_transposed)
					{
						size_t final_pos_rows = this->final_pos_rows;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);
								__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);
								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GT_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[j * rows + i]), _mm_castsi128_ps(mask_result));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; cols; j++)
							{
								data_result[j * rows + i] = data1[i * cols1 + j] > data2[j * rows2 + i] ? True : False;
							}
						}
					}
					else
					{
						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
								__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
									data2[(j + 1) * rows2 + i],
									data2[(j + 2) * rows2 + i],
									data2[(j + 3) * rows2 + i]);
								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GT_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[i * cols + j]), _mm_castsi128_ps(mask_result));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < rows; i++)
							{
								data_result[i * cols + j] = data1[i * cols1 + j] > data2[j * rows2 + i] ? True : False;
							}
						}
					}
				}
				else
				{
					if constexpr (return_transposed)
					{
						size_t cols1 = this->actual_cols;
						size_t cols2 = other.actual_cols;

						for (size_t i = 0; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								data_result[j * rows + i] = data1[i * cols1 + j] > data2[i * cols2 + j] ? True : False;
							}
						}
					}
					else
					{
						if constexpr (this_contiguous && other_contiguous)
						{
							size_t final_pos_size = this->final_pos_size;
							size_t size = this->_size;

							for (size_t i = 0; i < final_pos_size; i += 4)
							{
								__m256d a = _mm256_load_pd(&data1[i]);
								__m256d b = _mm256_load_pd(&data2[i]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GT_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
							}
							for (size_t i = final_pos_size; i < size; i++)
							{
								data_result[i] = data1[i] > data2[i] ? True : False;
							}
						}
						else
						{
							size_t cols1 = this->actual_cols;
							size_t cols2 = other.actual_cols;

							size_t final_pos_cols = this->final_pos_cols;

							for (size_t j = 0; j < final_pos_cols; j += 4)
							{
								for (size_t i = 0; i < rows; i++)
								{
									__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
									__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

									__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GT_OQ));

									__m128i mask1 = _mm256_castsi256_si128(mask);
									__m128i mask2 = _mm256_extracti128_si256(mask, 1);

									mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
									mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

									__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

									_mm_store_ss(reinterpret_cast<float*>(&data_result[i * cols + j]), _mm_castsi128_ps(mask_result));
								}
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								for (size_t i = 0; i < rows; i++)
								{
									data_result[i * cols + j] = data1[i * cols1 + j] > data2[i * cols2 + j] ? True : False;
								}
							}
						}
					}
				}
			}
			return result;
		}

		template<bool return_transposed = false>
		matrix<bool, return_transposed> operator>(double num)
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			matrix<bool, return_transposed, true> result(rows, cols);

			bool* data_result = result._data;

			__m256d b = _mm256_set1_pd(num);

			if constexpr (this_transposed)
			{
				if constexpr (return_transposed)
				{
					if constexpr (this_contiguous)
					{
						size_t size = this->_size;

						size_t final_pos_size = this->final_pos_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);

							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GT_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = data1[i] > num ? True : False;
						}
					}
					else
					{
						size_t final_pos_rows = this->final_pos_rows;

						size_t rows1 = this->actual_rows;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GT_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[j * rows + i]), _mm_castsi128_ps(mask_result));

							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								data_result[j * cols + i] = data1[j * rows1 + i] > num ? True : False;
							}
						}
					}
				}
				else
				{
					size_t rows1 = this->actual_rows;

					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							data_result[i * cols + j] = data1[j * rows1 + i] > num ? True : False;
						}
					}
				}
			}
			else
			{
				if constexpr (return_transposed)
				{
					size_t cols1 = this->actual_cols;

					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							data_result[j * rows + i] = data1[i * cols1 + j] > num ? True : False;
						}
					}
				}
				else
				{
					if constexpr (this_contiguous)
					{
						size_t final_pos_size = this->final_pos_size;
						size_t size = this->_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);

							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GT_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = data1[i] > num ? True : False;
						}
					}
					else
					{
						size_t cols1 = this->actual_cols;

						size_t final_pos_cols = this->final_pos_cols;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GT_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[i * cols + j]), _mm_castsi128_ps(mask_result));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < rows; i++)
							{
								data_result[i * cols + j] = data1[i * cols1 + j] > num ? True : False;
							}
						}
					}
				}
			}
			return result;
		}

		// <

		template<bool return_transposed = false, bool other_transposed, bool other_contiguous, bool call_destructor>
		matrix<bool, return_transposed> operator<(const matrix<double, other_transposed, other_contiguous, call_destructor>& other)
		{
#ifdef _DEBUG
			if (other._cols < this->_cols || other._rows < this->_rows) throw std::invalid_argument("The dimensions of both matrices must be the same");
#else
#endif

			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;
			double* data2 = other._data;

			matrix<bool, return_transposed, true> result(rows, cols);

			bool* data_result = result._data;

			if constexpr (this_transposed)
			{
				if constexpr (other_transposed)
				{
					if constexpr (return_transposed)
					{
						if constexpr (this_contiguous && other_contiguous)
						{
							size_t size = this->_size;

							size_t final_pos_size = this->final_pos_size;

							for (size_t i = 0; i < final_pos_size; i += 4)
							{
								__m256d a = _mm256_load_pd(&data1[i]);
								__m256d b = _mm256_load_pd(&data2[i]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LT_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
							}
							for (size_t i = final_pos_size; i < size; i++)
							{
								data_result[i] = data1[i] < data2[i] ? True : False;
							}
						}
						else
						{
							size_t final_pos_rows = this->final_pos_rows;

							size_t rows1 = this->actual_rows;
							size_t rows2 = other.actual_rows;

							for (size_t i = 0; i < final_pos_rows; i += 4)
							{
								for (size_t j = 0; j < cols; j++)
								{
									__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
									__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);

									__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LT_OQ));

									__m128i mask1 = _mm256_castsi256_si128(mask);
									__m128i mask2 = _mm256_extracti128_si256(mask, 1);

									mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
									mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

									__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

									_mm_store_ss(reinterpret_cast<float*>(&data_result[j * rows + i]), _mm_castsi128_ps(mask_result));

								}
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								for (size_t j = 0; j < cols; j++)
								{
									data_result[j * cols + i] = data1[j * rows1 + i] < data2[j * rows2 + i] ? True : False;
								}
							}
						}
					}
					else
					{
						size_t rows1 = this->actual_rows;
						size_t rows2 = other.actual_rows;

						for (size_t i = 0; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								data_result[i * cols + j] = data1[j * rows1 + i] < data2[j * rows2 + i] ? True : False;
							}
						}
					}
				}
				else
				{
					if constexpr (return_transposed)
					{
						size_t final_pos_rows = this->final_pos_rows;

						size_t rows1 = this->actual_rows;
						size_t cols2 = other.actual_cols;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
								__m256d b = _mm256_setr_pd(data2[i * cols2 + j],
									data2[(i + 1) * cols2 + j],
									data2[(i + 2) * cols2 + j],
									data2[(i + 3) * cols2 + j]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LT_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[j * rows + i]), _mm_castsi128_ps(mask_result));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								data_result[j * cols + i] = data1[j * rows1 + i] < data2[i * cols2 + j] ? True : False;
							}
						}
					}
					else
					{
						size_t final_pos_cols = this->final_pos_cols;

						size_t rows1 = this->actual_rows;
						size_t cols2 = other.actual_cols;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);
								__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LT_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[i * cols + j]), _mm_castsi128_ps(mask_result));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < rows; i++)
							{
								data_result[i * cols + j] = data1[j * rows1 + i] < data2[i * cols2 + j] ? True : False;
							}
						}
					}
				}
			}
			else
			{
				if constexpr (other_transposed)
				{
					size_t cols1 = this->actual_cols;
					size_t rows2 = other.actual_rows;

					if constexpr (return_transposed)
					{
						size_t final_pos_rows = this->final_pos_rows;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);
								__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);
								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LT_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[j * rows + i]), _mm_castsi128_ps(mask_result));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; cols; j++)
							{
								data_result[j * rows + i] = data1[i * cols1 + j] < data2[j * rows2 + i] ? True : False;
							}
						}
					}
					else
					{
						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
								__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
									data2[(j + 1) * rows2 + i],
									data2[(j + 2) * rows2 + i],
									data2[(j + 3) * rows2 + i]);
								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LT_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[i * cols + j]), _mm_castsi128_ps(mask_result));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < rows; i++)
							{
								data_result[i * cols + j] = data1[i * cols1 + j] < data2[j * rows2 + i] ? True : False;
							}
						}
					}
				}
				else
				{
					if constexpr (return_transposed)
					{
						size_t cols1 = this->actual_cols;
						size_t cols2 = other.actual_cols;

						for (size_t i = 0; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								data_result[j * rows + i] = data1[i * cols1 + j] < data2[i * cols2 + j] ? True : False;
							}
						}
					}
					else
					{
						if constexpr (this_contiguous && other_contiguous)
						{
							size_t final_pos_size = this->final_pos_size;
							size_t size = this->_size;

							for (size_t i = 0; i < final_pos_size; i += 4)
							{
								__m256d a = _mm256_load_pd(&data1[i]);
								__m256d b = _mm256_load_pd(&data2[i]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LT_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
							}
							for (size_t i = final_pos_size; i < size; i++)
							{
								data_result[i] = data1[i] < data2[i] ? True : False;
							}
						}
						else
						{
							size_t cols1 = this->actual_cols;
							size_t cols2 = other.actual_cols;

							size_t final_pos_cols = this->final_pos_cols;

							for (size_t j = 0; j < final_pos_cols; j += 4)
							{
								for (size_t i = 0; i < rows; i++)
								{
									__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
									__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

									__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LT_OQ));

									__m128i mask1 = _mm256_castsi256_si128(mask);
									__m128i mask2 = _mm256_extracti128_si256(mask, 1);

									mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
									mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

									__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

									_mm_store_ss(reinterpret_cast<float*>(&data_result[i * cols + j]), _mm_castsi128_ps(mask_result));
								}
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								for (size_t i = 0; i < rows; i++)
								{
									data_result[i * cols + j] = data1[i * cols1 + j] < data2[i * cols2 + j] ? True : False;
								}
							}
						}
					}
				}
			}
			return result;
		}

		template<bool return_transposed = false>
		matrix<bool, return_transposed> operator<(double num)
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			matrix<bool, return_transposed, true> result(rows, cols);

			bool* data_result = result._data;

			__m256d b = _mm256_set1_pd(num);

			if constexpr (this_transposed)
			{
				if constexpr (return_transposed)
				{
					if constexpr (this_contiguous)
					{
						size_t size = this->_size;

						size_t final_pos_size = this->final_pos_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);

							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LT_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = data1[i] < num ? True : False;
						}
					}
					else
					{
						size_t final_pos_rows = this->final_pos_rows;

						size_t rows1 = this->actual_rows;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LT_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[j * rows + i]), _mm_castsi128_ps(mask_result));

							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								data_result[j * cols + i] = data1[j * rows1 + i] < num ? True : False;
							}
						}
					}
				}
				else
				{
					size_t rows1 = this->actual_rows;

					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							data_result[i * cols + j] = data1[j * rows1 + i] < num ? True : False;
						}
					}
				}
			}
			else
			{
				if constexpr (return_transposed)
				{
					size_t cols1 = this->actual_cols;

					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							data_result[j * rows + i] = data1[i * cols1 + j] < num ? True : False;
						}
					}
				}
				else
				{
					if constexpr (this_contiguous)
					{
						size_t final_pos_size = this->final_pos_size;
						size_t size = this->_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);

							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LT_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = data1[i] < num ? True : False;
						}
					}
					else
					{
						size_t cols1 = this->actual_cols;

						size_t final_pos_cols = this->final_pos_cols;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LT_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[i * cols + j]), _mm_castsi128_ps(mask_result));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < rows; i++)
							{
								data_result[i * cols + j] = data1[i * cols1 + j] < num ? True : False;
							}
						}
					}
				}
			}
			return result;
		}

		// >=

		template<bool return_transposed = false, bool other_transposed, bool other_contiguous, bool call_destructor>
		matrix<bool, return_transposed> operator>=(const matrix<double, other_transposed, other_contiguous, call_destructor>& other)
		{
#ifdef _DEBUG
			if (other._cols >= this->_cols || other._rows >= this->_rows) throw std::invalid_argument("The dimensions of both matrices must be the same");
#else
#endif

			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;
			double* data2 = other._data;

			matrix<bool, return_transposed, true> result(rows, cols);

			bool* data_result = result._data;

			if constexpr (this_transposed)
			{
				if constexpr (other_transposed)
				{
					if constexpr (return_transposed)
					{
						if constexpr (this_contiguous && other_contiguous)
						{
							size_t size = this->_size;

							size_t final_pos_size = this->final_pos_size;

							for (size_t i = 0; i < final_pos_size; i += 4)
							{
								__m256d a = _mm256_load_pd(&data1[i]);
								__m256d b = _mm256_load_pd(&data2[i]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GE_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
							}
							for (size_t i = final_pos_size; i < size; i++)
							{
								data_result[i] = data1[i] >= data2[i] ? True : False;
							}
						}
						else
						{
							size_t final_pos_rows = this->final_pos_rows;

							size_t rows1 = this->actual_rows;
							size_t rows2 = other.actual_rows;

							for (size_t i = 0; i < final_pos_rows; i += 4)
							{
								for (size_t j = 0; j < cols; j++)
								{
									__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
									__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);

									__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GE_OQ));

									__m128i mask1 = _mm256_castsi256_si128(mask);
									__m128i mask2 = _mm256_extracti128_si256(mask, 1);

									mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
									mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

									__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

									_mm_store_ss(reinterpret_cast<float*>(&data_result[j * rows + i]), _mm_castsi128_ps(mask_result));

								}
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								for (size_t j = 0; j < cols; j++)
								{
									data_result[j * cols + i] = data1[j * rows1 + i] >= data2[j * rows2 + i] ? True : False;
								}
							}
						}
					}
					else
					{
						size_t rows1 = this->actual_rows;
						size_t rows2 = other.actual_rows;

						for (size_t i = 0; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								data_result[i * cols + j] = data1[j * rows1 + i] >= data2[j * rows2 + i] ? True : False;
						}
					}
				}
			}
				else
				{
					if constexpr (return_transposed)
					{
						size_t final_pos_rows = this->final_pos_rows;

						size_t rows1 = this->actual_rows;
						size_t cols2 = other.actual_cols;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
								__m256d b = _mm256_setr_pd(data2[i * cols2 + j],
									data2[(i + 1) * cols2 + j],
									data2[(i + 2) * cols2 + j],
									data2[(i + 3) * cols2 + j]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GE_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[j * rows + i]), _mm_castsi128_ps(mask_result));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								data_result[j * cols + i] = data1[j * rows1 + i] >= data2[i * cols2 + j] ? True : False;
							}
						}
					}
					else
					{
						size_t final_pos_cols = this->final_pos_cols;

						size_t rows1 = this->actual_rows;
						size_t cols2 = other.actual_cols;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);
								__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GE_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[i * cols + j]), _mm_castsi128_ps(mask_result));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < rows; i++)
							{
								data_result[i * cols + j] = data1[j * rows1 + i] >= data2[i * cols2 + j] ? True : False;
							}
						}
					}
				}
		}
			else
			{
				if constexpr (other_transposed)
				{
					size_t cols1 = this->actual_cols;
					size_t rows2 = other.actual_rows;

					if constexpr (return_transposed)
					{
						size_t final_pos_rows = this->final_pos_rows;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);
								__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);
								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GE_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[j * rows + i]), _mm_castsi128_ps(mask_result));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; cols; j++)
							{
								data_result[j * rows + i] = data1[i * cols1 + j] >= data2[j * rows2 + i] ? True : False;
							}
						}
					}
					else
					{
						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
								__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
									data2[(j + 1) * rows2 + i],
									data2[(j + 2) * rows2 + i],
									data2[(j + 3) * rows2 + i]);
								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GE_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[i * cols + j]), _mm_castsi128_ps(mask_result));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < rows; i++)
							{
								data_result[i * cols + j] = data1[i * cols1 + j] >= data2[j * rows2 + i] ? True : False;
							}
						}
					}
				}
				else
				{
					if constexpr (return_transposed)
					{
						size_t cols1 = this->actual_cols;
						size_t cols2 = other.actual_cols;

						for (size_t i = 0; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								data_result[j * rows + i] = data1[i * cols1 + j] >= data2[i * cols2 + j] ? True : False;
							}
						}
					}
					else
					{
						if constexpr (this_contiguous && other_contiguous)
						{
							size_t final_pos_size = this->final_pos_size;
							size_t size = this->_size;

							for (size_t i = 0; i < final_pos_size; i += 4)
							{
								__m256d a = _mm256_load_pd(&data1[i]);
								__m256d b = _mm256_load_pd(&data2[i]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GE_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
							}
							for (size_t i = final_pos_size; i < size; i++)
							{
								data_result[i] = data1[i] >= data2[i] ? True : False;
							}
						}
						else
						{
							size_t cols1 = this->actual_cols;
							size_t cols2 = other.actual_cols;

							size_t final_pos_cols = this->final_pos_cols;

							for (size_t j = 0; j < final_pos_cols; j += 4)
							{
								for (size_t i = 0; i < rows; i++)
								{
									__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
									__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

									__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GE_OQ));

									__m128i mask1 = _mm256_castsi256_si128(mask);
									__m128i mask2 = _mm256_extracti128_si256(mask, 1);

									mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
									mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

									__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

									_mm_store_ss(reinterpret_cast<float*>(&data_result[i * cols + j]), _mm_castsi128_ps(mask_result));
								}
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								for (size_t i = 0; i < rows; i++)
								{
									data_result[i * cols + j] = data1[i * cols1 + j] >= data2[i * cols2 + j] ? True : False;
								}
							}
						}
					}
				}
			}
			return result;
		}

		template<bool return_transposed = false>
		matrix<bool, return_transposed> operator>=(double num)
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			matrix<bool, return_transposed, true> result(rows, cols);

			bool* data_result = result._data;

			__m256d b = _mm256_set1_pd(num);

			if constexpr (this_transposed)
			{
				if constexpr (return_transposed)
				{
					if constexpr (this_contiguous)
					{
						size_t size = this->_size;

						size_t final_pos_size = this->final_pos_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);

							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GE_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = data1[i] >= num ? True : False;
						}
					}
					else
					{
						size_t final_pos_rows = this->final_pos_rows;

						size_t rows1 = this->actual_rows;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GE_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[j * rows + i]), _mm_castsi128_ps(mask_result));

							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								data_result[j * cols + i] = data1[j * rows1 + i] >= num ? True : False;
							}
						}
					}
				}
				else
				{
					size_t rows1 = this->actual_rows;

					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							data_result[i * cols + j] = data1[j * rows1 + i] >= num ? True : False;
						}
					}
				}
			}
			else
			{
				if constexpr (return_transposed)
				{
					size_t cols1 = this->actual_cols;

					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							data_result[j * rows + i] = data1[i * cols1 + j] >= num ? True : False;
						}
					}
				}
				else
				{
					if constexpr (this_contiguous)
					{
						size_t final_pos_size = this->final_pos_size;
						size_t size = this->_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);

							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GE_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = data1[i] >= num ? True : False;
						}
					}
					else
					{
						size_t cols1 = this->actual_cols;

						size_t final_pos_cols = this->final_pos_cols;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GE_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[i * cols + j]), _mm_castsi128_ps(mask_result));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < rows; i++)
							{
								data_result[i * cols + j] = data1[i * cols1 + j] >= num ? True : False;
							}
						}
					}
				}
			}
			return result;
		}

		// <=

		template<bool return_transposed = false, bool other_transposed, bool other_contiguous, bool call_destructor>
		matrix<bool, return_transposed> operator<=(const matrix<double, other_transposed, other_contiguous, call_destructor>& other)
		{
#ifdef _DEBUG
			if (other._cols <= this->_cols || other._rows <= this->_rows) throw std::invalid_argument("The dimensions of both matrices must be the same");
#else
#endif

			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;
			double* data2 = other._data;

			matrix<bool, return_transposed, true> result(rows, cols);

			bool* data_result = result._data;

			if constexpr (this_transposed)
			{
				if constexpr (other_transposed)
				{
					if constexpr (return_transposed)
					{
						if constexpr (this_contiguous && other_contiguous)
						{
							size_t size = this->_size;

							size_t final_pos_size = this->final_pos_size;

							for (size_t i = 0; i < final_pos_size; i += 4)
							{
								__m256d a = _mm256_load_pd(&data1[i]);
								__m256d b = _mm256_load_pd(&data2[i]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LE_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
							}
							for (size_t i = final_pos_size; i < size; i++)
							{
								data_result[i] = data1[i] <= data2[i] ? True : False;
							}
						}
						else
						{
							size_t final_pos_rows = this->final_pos_rows;

							size_t rows1 = this->actual_rows;
							size_t rows2 = other.actual_rows;

							for (size_t i = 0; i < final_pos_rows; i += 4)
							{
								for (size_t j = 0; j < cols; j++)
								{
									__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
									__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);

									__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LE_OQ));

									__m128i mask1 = _mm256_castsi256_si128(mask);
									__m128i mask2 = _mm256_extracti128_si256(mask, 1);

									mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
									mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

									__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

									_mm_store_ss(reinterpret_cast<float*>(&data_result[j * rows + i]), _mm_castsi128_ps(mask_result));

								}
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								for (size_t j = 0; j < cols; j++)
								{
									data_result[j * cols + i] = data1[j * rows1 + i] <= data2[j * rows2 + i] ? True : False;
								}
							}
						}
					}
					else
					{
						size_t rows1 = this->actual_rows;
						size_t rows2 = other.actual_rows;

						for (size_t i = 0; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								data_result[i * cols + j] = data1[j * rows1 + i] <= data2[j * rows2 + i] ? True : False;
						}
					}
				}
			}
				else
				{
					if constexpr (return_transposed)
					{
						size_t final_pos_rows = this->final_pos_rows;

						size_t rows1 = this->actual_rows;
						size_t cols2 = other.actual_cols;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
								__m256d b = _mm256_setr_pd(data2[i * cols2 + j],
									data2[(i + 1) * cols2 + j],
									data2[(i + 2) * cols2 + j],
									data2[(i + 3) * cols2 + j]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LE_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[j * rows + i]), _mm_castsi128_ps(mask_result));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								data_result[j * cols + i] = data1[j * rows1 + i] <= data2[i * cols2 + j] ? True : False;
							}
						}
					}
					else
					{
						size_t final_pos_cols = this->final_pos_cols;

						size_t rows1 = this->actual_rows;
						size_t cols2 = other.actual_cols;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);
								__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LE_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[i * cols + j]), _mm_castsi128_ps(mask_result));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < rows; i++)
							{
								data_result[i * cols + j] = data1[j * rows1 + i] <= data2[i * cols2 + j] ? True : False;
							}
						}
					}
				}
		}
			else
			{
				if constexpr (other_transposed)
				{
					size_t cols1 = this->actual_cols;
					size_t rows2 = other.actual_rows;

					if constexpr (return_transposed)
					{
						size_t final_pos_rows = this->final_pos_rows;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);
								__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);
								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LE_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[j * rows + i]), _mm_castsi128_ps(mask_result));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; cols; j++)
							{
								data_result[j * rows + i] = data1[i * cols1 + j] <= data2[j * rows2 + i] ? True : False;
							}
						}
					}
					else
					{
						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
								__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
									data2[(j + 1) * rows2 + i],
									data2[(j + 2) * rows2 + i],
									data2[(j + 3) * rows2 + i]);
								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LE_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[i * cols + j]), _mm_castsi128_ps(mask_result));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < rows; i++)
							{
								data_result[i * cols + j] = data1[i * cols1 + j] <= data2[j * rows2 + i] ? True : False;
							}
						}
					}
				}
				else
				{
					if constexpr (return_transposed)
					{
						size_t cols1 = this->actual_cols;
						size_t cols2 = other.actual_cols;

						for (size_t i = 0; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								data_result[j * rows + i] = data1[i * cols1 + j] <= data2[i * cols2 + j] ? True : False;
							}
						}
					}
					else
					{
						if constexpr (this_contiguous && other_contiguous)
						{
							size_t final_pos_size = this->final_pos_size;
							size_t size = this->_size;

							for (size_t i = 0; i < final_pos_size; i += 4)
							{
								__m256d a = _mm256_load_pd(&data1[i]);
								__m256d b = _mm256_load_pd(&data2[i]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LE_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
							}
							for (size_t i = final_pos_size; i < size; i++)
							{
								data_result[i] = data1[i] <= data2[i] ? True : False;
							}
						}
						else
						{
							size_t cols1 = this->actual_cols;
							size_t cols2 = other.actual_cols;

							size_t final_pos_cols = this->final_pos_cols;

							for (size_t j = 0; j < final_pos_cols; j += 4)
							{
								for (size_t i = 0; i < rows; i++)
								{
									__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
									__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

									__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LE_OQ));

									__m128i mask1 = _mm256_castsi256_si128(mask);
									__m128i mask2 = _mm256_extracti128_si256(mask, 1);

									mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
									mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

									__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

									_mm_store_ss(reinterpret_cast<float*>(&data_result[i * cols + j]), _mm_castsi128_ps(mask_result));
								}
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								for (size_t i = 0; i < rows; i++)
								{
									data_result[i * cols + j] = data1[i * cols1 + j] <= data2[i * cols2 + j] ? True : False;
								}
							}
						}
					}
				}
			}
			return result;
		}

		template<bool return_transposed = false>
		matrix<bool, return_transposed> operator<=(double num)
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			matrix<bool, return_transposed, true> result(rows, cols);

			bool* data_result = result._data;

			__m256d b = _mm256_set1_pd(num);

			if constexpr (this_transposed)
			{
				if constexpr (return_transposed)
				{
					if constexpr (this_contiguous)
					{
						size_t size = this->_size;

						size_t final_pos_size = this->final_pos_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);

							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LE_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = data1[i] <= num ? True : False;
						}
					}
					else
					{
						size_t final_pos_rows = this->final_pos_rows;

						size_t rows1 = this->actual_rows;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LE_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[j * rows + i]), _mm_castsi128_ps(mask_result));

							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								data_result[j * cols + i] = data1[j * rows1 + i] <= num ? True : False;
							}
						}
					}
				}
				else
				{
					size_t rows1 = this->actual_rows;

					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							data_result[i * cols + j] = data1[j * rows1 + i] <= num ? True : False;
						}
					}
				}
			}
			else
			{
				if constexpr (return_transposed)
				{
					size_t cols1 = this->actual_cols;

					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							data_result[j * rows + i] = data1[i * cols1 + j] <= num ? True : False;
						}
					}
				}
				else
				{
					if constexpr (this_contiguous)
					{
						size_t final_pos_size = this->final_pos_size;
						size_t size = this->_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);

							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LE_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&data_result[i]), _mm_castsi128_ps(mask_result));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = data1[i] <= num ? True : False;
						}
					}
					else
					{
						size_t cols1 = this->actual_cols;

						size_t final_pos_cols = this->final_pos_cols;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LE_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i mask_result = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&data_result[i * cols + j]), _mm_castsi128_ps(mask_result));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < rows; i++)
							{
								data_result[i * cols + j] = data1[i * cols1 + j] <= num ? True : False;
							}
						}
					}
				}
			}
			return result;
		}

		// Functions

		template<bool return_transposed = false>
		matrix<double, return_transposed> exp()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			matrix<double, return_transposed> result(rows, cols);

			double* data_result = result._data;

			if constexpr (this_transposed)
			{
				if constexpr (return_transposed)
				{
					if constexpr (this_contiguous)
					{
						size_t final_pos_size = this->final_pos_size;
						size_t size = this->_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);

							_mm256_store_pd(&data_result[i]; _mm256_exp_pd(a));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = std::exp(data1[i]);
						}
					}
					else
					{
						size_t rows1 = this->actual_rows;

						size_t final_pos_rows = this->final_pos_rows;
						size_t final_pos_cols = this->final_pos_cols;
						
						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < final_pos_cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

								_mm256_store_pd(&data_result[j * rows + i], _mm256_exp_pd(a));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < final_pos_cols; j += 4)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i], 
									data1[(j + 1) * rows1 + i], 
									data1[(j + 2) * rows1 + i], 
									data1[(j + 3) * rows1 + i]);

								__m256d exp = _mm256_exp_pd(a);

								__m128d val1 = _mm256_extractf128_pd(exp, 1);
								__m128d val2 = _mm256_castpd256_pd128(exp);

								_mm_store_sd(&data_result[j * rows + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(j + 1) * rows + i], val2);

								_mm_store_sd(&data_result[(j + 2) * rows + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(j + 3) * rows + i], val1);
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								data_result[j * rows + i] = std::exp(data1[j * rows1 + i]);
							}
						}
					}
				}
				else
				{
					size_t rows1 = this->actual_rows;

					size_t final_pos_cols = this->final_pos_cols;
					size_t final_pos_rows = this->final_pos_rows;

					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_setr_pd(data1[j * rows1 + i], 
								data1[(j + 1) * rows1 + i], 
								data1[(j + 2) * rows1 + i], 
								data1[(j + 3) * rows1 + i]);
							_mm256_store_pd(&data_result[i * cols + j], _mm256_exp_pd(a));
						}
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

							__m256d exp = _mm256_exp_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data_result[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

							_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
						}
						for (size_t i = final_pos_rows; i < cols; i++)
						{
							data_result[i * cols + j] = std::exp(data1[j * rows1 + i]);
						}
					}
				}
			}
			else
			{
				if constexpr (return_transposed)
				{
					size_t cols1 = this->actual_cols;

					size_t final_pos_rows = this->final_pos_rows;
					size_t final_pos_cols = this->final_pos_cols;

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_setr_pd(data1[i * cols1 + j], 
								data1[(i + 1) * cols1 + j], 
								data1[(i + 2) * cols1 + j], 
								data1[(i + 3) * cols1 + j]);
							
							_mm256_store_pd(&data_result[j * rows + i], _mm256_exp_pd(a));
						}
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

							__m256d exp = _mm256_exp_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data_result[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data_result[(j + 1) * rows + i], val2);

							_mm_store_sd(&data_result[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data_result[(j + 3) * rows + i], val1);
						}
						for (size_t j = final_pos_cols; i < cols; j++)
						{
							data_result[j * rows + i] = std::exp(data1[i * cols1 + j]);
						}
					}
				}
				else
				{
					if constexpr (this_contiguous)
					{
						size_t size = this->_size;
						size_t final_pos_size = this->final_pos_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);
							_mm256_store_pd(&data_result[i], _mm256_exp_pd(a));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = std::exp(data1[i]);
						}
					}
					else
					{
						size_t cols1 = this->actual_cols;

						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						for (size_t j = 0; j < final_pos_cols; j += 4) 
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

								_mm256_store_pd(&data_result[i * cols + j], _mm256_exp_pd(a));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < final_pos_rows; i += 4)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j], 
									data1[(i + 2) * cols1 + j], 
									data1[(i + 3) * cols1 + j]);

								__m256d exp = _mm256_exp_pd(a);

								__m128d val1 = _mm256_extractf128_pd(exp, 1);
								__m128d val2 = _mm256_castpd256_pd128(exp);

								_mm_store_sd(&data_result[i * cols + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

								_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
							}
							for (size_t i = final_pos_rows; i++)
							{
								data_result[i * cols + j] = std::exp(data1[i * cols1 + j]);
							}
						}
					}
				}
			}
			return result;
		}

		void self_exp()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			if constexpr (this_transposed)
			{
				if constexpr (this_contiguous)
				{
					size_t final_pos_size = this->final_pos_size;
					size_t size = this->_size;

					for (size_t i = 0; i < final_pos_size; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&data1[i]; _mm256_exp_pd(a));
					}
					for (size_t i = final_pos_size; i < size; i++)
					{
						data1[i] = std::exp(data1[i]);
					}
				}
				else
				{
					size_t rows1 = this->actual_rows;

					size_t final_pos_rows = this->final_pos_rows;
					size_t final_pos_cols = this->final_pos_cols;

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						for (size_t j = 0; j < final_pos_cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

							_mm256_store_pd(&data1[j * rows1 + i], _mm256_exp_pd(a));
						}
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
								data1[(j + 1) * rows1 + i],
								data1[(j + 2) * rows1 + i],
								data1[(j + 3) * rows1 + i]);

							__m256d exp = _mm256_exp_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data1[j * rows1 + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(j + 1) * rows1 + i], val2);

							_mm_store_sd(&data1[(j + 2) * rows1 + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(j + 3) * rows1 + i], val1);
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							data1[j * rows1 + i] = std::exp(data1[j * rows1 + i]);
						}
					}
				}
			}
			else
			{
				if constexpr (this_contiguous)
				{
					size_t size = this->_size;
					size_t final_pos_size = this->final_pos_size;

					for (size_t i = 0; i < final_pos_size; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);
						_mm256_store_pd(&data1[i], _mm256_exp_pd(a));
					}
					for (size_t i = final_pos_size; i < size; i++)
					{
						data1[i] = std::exp(data1[i]);
					}
				}
				else
				{
					size_t cols1 = this->actual_cols;

					size_t final_pos_cols = this->final_pos_cols;
					size_t final_pos_rows = this->final_pos_rows;

					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

							_mm256_store_pd(&data1[i * cols1 + j], _mm256_exp_pd(a));
						}
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
								data1[(i + 1) * cols1 + j],
								data1[(i + 2) * cols1 + j],
								data1[(i + 3) * cols1 + j]);

							__m256d exp = _mm256_exp_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data1[i * cols1 + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(i + 1) * cols1 + j], val2);

							_mm_store_sd(&data1[(i + 2) * cols1 + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(i + 3) * cols1 + j], val1);
						}
						for (size_t i = final_pos_rows; i++)
						{
							data1[i * cols1 + j] = std::exp(data1[i * cols1 + j]);
						}
					}
				}
			}
		}

		template<bool return_transposed = false>
		matrix<double, return_transposed> exp2()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			matrix<double, return_transposed> result(rows, cols);

			double* data_result = result._data;

			if constexpr (this_transposed)
			{
				if constexpr (return_transposed)
				{
					if constexpr (this_contiguous)
					{
						size_t final_pos_size = this->final_pos_size;
						size_t size = this->_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);

							_mm256_store_pd(&data_result[i]; _mm256_exp2_pd(a));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = std::exp2(data1[i]);
						}
					}
					else
					{
						size_t rows1 = this->actual_rows;

						size_t final_pos_rows = this->final_pos_rows;
						size_t final_pos_cols = this->final_pos_cols;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < final_pos_cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

								_mm256_store_pd(&data_result[j * rows + i], _mm256_exp2_pd(a));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < final_pos_cols; j += 4)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);

								__m256d exp = _mm256_exp2_pd(a);

								__m128d val1 = _mm256_extractf128_pd(exp, 1);
								__m128d val2 = _mm256_castpd256_pd128(exp);

								_mm_store_sd(&data_result[j * rows + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(j + 1) * rows + i], val2);

								_mm_store_sd(&data_result[(j + 2) * rows + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(j + 3) * rows + i], val1);
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								data_result[j * rows + i] = std::exp2(data1[j * rows1 + i]);
							}
						}
					}
				}
				else
				{
					size_t rows1 = this->actual_rows;

					size_t final_pos_cols = this->final_pos_cols;
					size_t final_pos_rows = this->final_pos_rows;

					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
								data1[(j + 1) * rows1 + i],
								data1[(j + 2) * rows1 + i],
								data1[(j + 3) * rows1 + i]);
							_mm256_store_pd(&data_result[i * cols + j], _mm256_exp2_pd(a));
						}
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

							__m256d exp = _mm256_exp2_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data_result[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

							_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
						}
						for (size_t i = final_pos_rows; i < cols; i++)
						{
							data_result[i * cols + j] = std::exp2(data1[j * rows1 + i]);
						}
					}
				}
			}
			else
			{
				if constexpr (return_transposed)
				{
					size_t cols1 = this->actual_cols;

					size_t final_pos_rows = this->final_pos_rows;
					size_t final_pos_cols = this->final_pos_cols;

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
								data1[(i + 1) * cols1 + j],
								data1[(i + 2) * cols1 + j],
								data1[(i + 3) * cols1 + j]);

							_mm256_store_pd(&data_result[j * rows + i], _mm256_exp2_pd(a));
						}
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

							__m256d exp = _mm256_exp2_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data_result[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data_result[(j + 1) * rows + i], val2);

							_mm_store_sd(&data_result[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data_result[(j + 3) * rows + i], val1);
						}
						for (size_t j = final_pos_cols; i < cols; j++)
						{
							data_result[j * rows + i] = std::exp2(data1[i * cols1 + j]);
						}
					}
				}
				else
				{
					if constexpr (this_contiguous)
					{
						size_t size = this->_size;
						size_t final_pos_size = this->final_pos_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);
							_mm256_store_pd(&data_result[i], _mm256_exp2_pd(a));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = std::exp2(data1[i]);
						}
					}
					else
					{
						size_t cols1 = this->actual_cols;

						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

								_mm256_store_pd(&data_result[i * cols + j], _mm256_exp2_pd(a));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < final_pos_rows; i += 4)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);

								__m256d exp = _mm256_exp2_pd(a);

								__m128d val1 = _mm256_extractf128_pd(exp, 1);
								__m128d val2 = _mm256_castpd256_pd128(exp);

								_mm_store_sd(&data_result[i * cols + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

								_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
							}
							for (size_t i = final_pos_rows; i++)
							{
								data_result[i * cols + j] = std::exp2(data1[i * cols1 + j]);
							}
						}
					}
				}
			}
			return result;
		}

		void self_exp2()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			if constexpr (this_transposed)
			{
				if constexpr (this_contiguous)
				{
					size_t final_pos_size = this->final_pos_size;
					size_t size = this->_size;

					for (size_t i = 0; i < final_pos_size; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&data1[i]; _mm256_exp2_pd(a));
					}
					for (size_t i = final_pos_size; i < size; i++)
					{
						data1[i] = std::exp2(data1[i]);
					}
				}
				else
				{
					size_t rows1 = this->actual_rows;

					size_t final_pos_rows = this->final_pos_rows;
					size_t final_pos_cols = this->final_pos_cols;

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						for (size_t j = 0; j < final_pos_cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

							_mm256_store_pd(&data1[j * rows1 + i], _mm256_exp2_pd(a));
						}
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
								data1[(j + 1) * rows1 + i],
								data1[(j + 2) * rows1 + i],
								data1[(j + 3) * rows1 + i]);

							__m256d exp = _mm256_exp2_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data1[j * rows1 + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(j + 1) * rows1 + i], val2);

							_mm_store_sd(&data1[(j + 2) * rows1 + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(j + 3) * rows1 + i], val1);
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							data1[j * rows1 + i] = std::exp2(data1[j * rows1 + i]);
						}
					}
				}
			}
			else
			{
				if constexpr (this_contiguous)
				{
					size_t size = this->_size;
					size_t final_pos_size = this->final_pos_size;

					for (size_t i = 0; i < final_pos_size; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);
						_mm256_store_pd(&data1[i], _mm256_exp2_pd(a));
					}
					for (size_t i = final_pos_size; i < size; i++)
					{
						data1[i] = std::exp2(data1[i]);
					}
				}
				else
				{
					size_t cols1 = this->actual_cols;

					size_t final_pos_cols = this->final_pos_cols;
					size_t final_pos_rows = this->final_pos_rows;

					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

							_mm256_store_pd(&data1[i * cols1 + j], _mm256_exp2_pd(a));
						}
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
								data1[(i + 1) * cols1 + j],
								data1[(i + 2) * cols1 + j],
								data1[(i + 3) * cols1 + j]);

							__m256d exp = _mm256_exp2_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data1[i * cols1 + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(i + 1) * cols1 + j], val2);

							_mm_store_sd(&data1[(i + 2) * cols1 + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(i + 3) * cols1 + j], val1);
						}
						for (size_t i = final_pos_rows; i++)
						{
							data1[i * cols1 + j] = std::exp2(data1[i * cols1 + j]);
						}
					}
				}
			}
		}

		template<bool return_transposed = false>
		matrix<double, return_transposed> log()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			matrix<double, return_transposed> result(rows, cols);

			double* data_result = result._data;

			if constexpr (this_transposed)
			{
				if constexpr (return_transposed)
				{
					if constexpr (this_contiguous)
					{
						size_t final_pos_size = this->final_pos_size;
						size_t size = this->_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);

							_mm256_store_pd(&data_result[i]; _mm256_log_pd(a));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = std::log(data1[i]);
						}
					}
					else
					{
						size_t rows1 = this->actual_rows;

						size_t final_pos_rows = this->final_pos_rows;
						size_t final_pos_cols = this->final_pos_cols;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < final_pos_cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

								_mm256_store_pd(&data_result[j * rows + i], _mm256_log_pd(a));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < final_pos_cols; j += 4)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);

								__m256d exp = _mm256_log_pd(a);

								__m128d val1 = _mm256_extractf128_pd(exp, 1);
								__m128d val2 = _mm256_castpd256_pd128(exp);

								_mm_store_sd(&data_result[j * rows + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(j + 1) * rows + i], val2);

								_mm_store_sd(&data_result[(j + 2) * rows + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(j + 3) * rows + i], val1);
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								data_result[j * rows + i] = std::log(data1[j * rows1 + i]);
							}
						}
					}
				}
				else
				{
					size_t rows1 = this->actual_rows;

					size_t final_pos_cols = this->final_pos_cols;
					size_t final_pos_rows = this->final_pos_rows;

					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
								data1[(j + 1) * rows1 + i],
								data1[(j + 2) * rows1 + i],
								data1[(j + 3) * rows1 + i]);
							_mm256_store_pd(&data_result[i * cols + j], _mm256_log_pd(a));
						}
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

							__m256d exp = _mm256_log_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data_result[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

							_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
						}
						for (size_t i = final_pos_rows; i < cols; i++)
						{
							data_result[i * cols + j] = std::log(data1[j * rows1 + i]);
						}
					}
				}
			}
			else
			{
				if constexpr (return_transposed)
				{
					size_t cols1 = this->actual_cols;

					size_t final_pos_rows = this->final_pos_rows;
					size_t final_pos_cols = this->final_pos_cols;

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
								data1[(i + 1) * cols1 + j],
								data1[(i + 2) * cols1 + j],
								data1[(i + 3) * cols1 + j]);

							_mm256_store_pd(&data_result[j * rows + i], _mm256_log_pd(a));
						}
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

							__m256d exp = _mm256_log_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data_result[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data_result[(j + 1) * rows + i], val2);

							_mm_store_sd(&data_result[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data_result[(j + 3) * rows + i], val1);
						}
						for (size_t j = final_pos_cols; i < cols; j++)
						{
							data_result[j * rows + i] = std::log(data1[i * cols1 + j]);
						}
					}
				}
				else
				{
					if constexpr (this_contiguous)
					{
						size_t size = this->_size;
						size_t final_pos_size = this->final_pos_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);
							_mm256_store_pd(&data_result[i], _mm256_log_pd(a));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = std::log(data1[i]);
						}
					}
					else
					{
						size_t cols1 = this->actual_cols;

						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

								_mm256_store_pd(&data_result[i * cols + j], _mm256_log_pd(a));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < final_pos_rows; i += 4)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);

								__m256d exp = _mm256_log_pd(a);

								__m128d val1 = _mm256_extractf128_pd(exp, 1);
								__m128d val2 = _mm256_castpd256_pd128(exp);

								_mm_store_sd(&data_result[i * cols + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

								_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
							}
							for (size_t i = final_pos_rows; i++)
							{
								data_result[i * cols + j] = std::log(data1[i * cols1 + j]);
							}
						}
					}
				}
			}
			return result;
		}

		void self_log()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			if constexpr (this_transposed)
			{
				if constexpr (this_contiguous)
				{
					size_t final_pos_size = this->final_pos_size;
					size_t size = this->_size;

					for (size_t i = 0; i < final_pos_size; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&data1[i]; _mm256_log_pd(a));
					}
					for (size_t i = final_pos_size; i < size; i++)
					{
						data1[i] = std::log(data1[i]);
					}
				}
				else
				{
					size_t rows1 = this->actual_rows;

					size_t final_pos_rows = this->final_pos_rows;
					size_t final_pos_cols = this->final_pos_cols;

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						for (size_t j = 0; j < final_pos_cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

							_mm256_store_pd(&data1[j * rows1 + i], _mm256_log_pd(a));
						}
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
								data1[(j + 1) * rows1 + i],
								data1[(j + 2) * rows1 + i],
								data1[(j + 3) * rows1 + i]);

							__m256d exp = _mm256_log_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data1[j * rows1 + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(j + 1) * rows1 + i], val2);

							_mm_store_sd(&data1[(j + 2) * rows1 + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(j + 3) * rows1 + i], val1);
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							data1[j * rows1 + i] = std::log(data1[j * rows1 + i]);
						}
					}
				}
			}
			else
			{
				if constexpr (this_contiguous)
				{
					size_t size = this->_size;
					size_t final_pos_size = this->final_pos_size;

					for (size_t i = 0; i < final_pos_size; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);
						_mm256_store_pd(&data1[i], _mm256_log_pd(a));
					}
					for (size_t i = final_pos_size; i < size; i++)
					{
						data1[i] = std::log(data1[i]);
					}
				}
				else
				{
					size_t cols1 = this->actual_cols;

					size_t final_pos_cols = this->final_pos_cols;
					size_t final_pos_rows = this->final_pos_rows;

					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

							_mm256_store_pd(&data1[i * cols1 + j], _mm256_log_pd(a));
						}
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
								data1[(i + 1) * cols1 + j],
								data1[(i + 2) * cols1 + j],
								data1[(i + 3) * cols1 + j]);

							__m256d exp = _mm256_log_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data1[i * cols1 + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(i + 1) * cols1 + j], val2);

							_mm_store_sd(&data1[(i + 2) * cols1 + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(i + 3) * cols1 + j], val1);
						}
						for (size_t i = final_pos_rows; i++)
						{
							data1[i * cols1 + j] = std::log(data1[i * cols1 + j]);
						}
					}
				}
			}
		}
		
		template<bool return_transposed = false>
		matrix<double, return_transposed> log2()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			matrix<double, return_transposed> result(rows, cols);

			double* data_result = result._data;

			if constexpr (this_transposed)
			{
				if constexpr (return_transposed)
				{
					if constexpr (this_contiguous)
					{
						size_t final_pos_size = this->final_pos_size;
						size_t size = this->_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);

							_mm256_store_pd(&data_result[i]; _mm256_log2_pd(a));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = std::log2(data1[i]);
						}
					}
					else
					{
						size_t rows1 = this->actual_rows;

						size_t final_pos_rows = this->final_pos_rows;
						size_t final_pos_cols = this->final_pos_cols;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < final_pos_cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

								_mm256_store_pd(&data_result[j * rows + i], _mm256_log2_pd(a));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < final_pos_cols; j += 4)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);

								__m256d exp = _mm256_log2_pd(a);

								__m128d val1 = _mm256_extractf128_pd(exp, 1);
								__m128d val2 = _mm256_castpd256_pd128(exp);

								_mm_store_sd(&data_result[j * rows + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(j + 1) * rows + i], val2);

								_mm_store_sd(&data_result[(j + 2) * rows + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(j + 3) * rows + i], val1);
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								data_result[j * rows + i] = std::log2(data1[j * rows1 + i]);
							}
						}
					}
				}
				else
				{
					size_t rows1 = this->actual_rows;

					size_t final_pos_cols = this->final_pos_cols;
					size_t final_pos_rows = this->final_pos_rows;

					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
								data1[(j + 1) * rows1 + i],
								data1[(j + 2) * rows1 + i],
								data1[(j + 3) * rows1 + i]);
							_mm256_store_pd(&data_result[i * cols + j], _mm256_log2_pd(a));
						}
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

							__m256d exp = _mm256_log2_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data_result[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

							_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
						}
						for (size_t i = final_pos_rows; i < cols; i++)
						{
							data_result[i * cols + j] = std::log2(data1[j * rows1 + i]);
						}
					}
				}
			}
			else
			{
				if constexpr (return_transposed)
				{
					size_t cols1 = this->actual_cols;

					size_t final_pos_rows = this->final_pos_rows;
					size_t final_pos_cols = this->final_pos_cols;

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
								data1[(i + 1) * cols1 + j],
								data1[(i + 2) * cols1 + j],
								data1[(i + 3) * cols1 + j]);

							_mm256_store_pd(&data_result[j * rows + i], _mm256_log2_pd(a));
						}
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

							__m256d exp = _mm256_log2_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data_result[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data_result[(j + 1) * rows + i], val2);

							_mm_store_sd(&data_result[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data_result[(j + 3) * rows + i], val1);
						}
						for (size_t j = final_pos_cols; i < cols; j++)
						{
							data_result[j * rows + i] = std::log2(data1[i * cols1 + j]);
						}
					}
				}
				else
				{
					if constexpr (this_contiguous)
					{
						size_t size = this->_size;
						size_t final_pos_size = this->final_pos_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);
							_mm256_store_pd(&data_result[i], _mm256_log2_pd(a));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = std::log2(data1[i]);
						}
					}
					else
					{
						size_t cols1 = this->actual_cols;

						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

								_mm256_store_pd(&data_result[i * cols + j], _mm256_log2_pd(a));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < final_pos_rows; i += 4)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);

								__m256d exp = _mm256_log2_pd(a);

								__m128d val1 = _mm256_extractf128_pd(exp, 1);
								__m128d val2 = _mm256_castpd256_pd128(exp);

								_mm_store_sd(&data_result[i * cols + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

								_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
							}
							for (size_t i = final_pos_rows; i++)
							{
								data_result[i * cols + j] = std::log2(data1[i * cols1 + j]);
							}
						}
					}
				}
			}
			return result;
		}

		void self_log2()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			if constexpr (this_transposed)
			{
				if constexpr (this_contiguous)
				{
					size_t final_pos_size = this->final_pos_size;
					size_t size = this->_size;

					for (size_t i = 0; i < final_pos_size; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&data1[i]; _mm256_log2_pd(a));
					}
					for (size_t i = final_pos_size; i < size; i++)
					{
						data1[i] = std::log2(data1[i]);
					}
				}
				else
				{
					size_t rows1 = this->actual_rows;

					size_t final_pos_rows = this->final_pos_rows;
					size_t final_pos_cols = this->final_pos_cols;

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						for (size_t j = 0; j < final_pos_cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

							_mm256_store_pd(&data1[j * rows1 + i], _mm256_log2_pd(a));
						}
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
								data1[(j + 1) * rows1 + i],
								data1[(j + 2) * rows1 + i],
								data1[(j + 3) * rows1 + i]);

							__m256d exp = _mm256_log2_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data1[j * rows1 + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(j + 1) * rows1 + i], val2);

							_mm_store_sd(&data1[(j + 2) * rows1 + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(j + 3) * rows1 + i], val1);
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							data1[j * rows1 + i] = std::log2(data1[j * rows1 + i]);
						}
					}
				}
			}
			else
			{
				if constexpr (this_contiguous)
				{
					size_t size = this->_size;
					size_t final_pos_size = this->final_pos_size;

					for (size_t i = 0; i < final_pos_size; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);
						_mm256_store_pd(&data1[i], _mm256_log2_pd(a));
					}
					for (size_t i = final_pos_size; i < size; i++)
					{
						data1[i] = std::log2(data1[i]);
					}
				}
				else
				{
					size_t cols1 = this->actual_cols;

					size_t final_pos_cols = this->final_pos_cols;
					size_t final_pos_rows = this->final_pos_rows;

					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

							_mm256_store_pd(&data1[i * cols1 + j], _mm256_log2_pd(a));
						}
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
								data1[(i + 1) * cols1 + j],
								data1[(i + 2) * cols1 + j],
								data1[(i + 3) * cols1 + j]);

							__m256d exp = _mm256_log2_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data1[i * cols1 + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(i + 1) * cols1 + j], val2);

							_mm_store_sd(&data1[(i + 2) * cols1 + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(i + 3) * cols1 + j], val1);
						}
						for (size_t i = final_pos_rows; i++)
						{
							data1[i * cols1 + j] = std::log2(data1[i * cols1 + j]);
						}
					}
				}
			}
		}

		template<bool return_transposed = false>
		matrix<double, return_transposed> log10()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			matrix<double, return_transposed> result(rows, cols);

			double* data_result = result._data;

			if constexpr (this_transposed)
			{
				if constexpr (return_transposed)
				{
					if constexpr (this_contiguous)
					{
						size_t final_pos_size = this->final_pos_size;
						size_t size = this->_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);

							_mm256_store_pd(&data_result[i]; _mm256_log10_pd(a));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = std::log10(data1[i]);
						}
					}
					else
					{
						size_t rows1 = this->actual_rows;

						size_t final_pos_rows = this->final_pos_rows;
						size_t final_pos_cols = this->final_pos_cols;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < final_pos_cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

								_mm256_store_pd(&data_result[j * rows + i], _mm256_log10_pd(a));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < final_pos_cols; j += 4)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);

								__m256d exp = _mm256_log10_pd(a);

								__m128d val1 = _mm256_extractf128_pd(exp, 1);
								__m128d val2 = _mm256_castpd256_pd128(exp);

								_mm_store_sd(&data_result[j * rows + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(j + 1) * rows + i], val2);

								_mm_store_sd(&data_result[(j + 2) * rows + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(j + 3) * rows + i], val1);
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								data_result[j * rows + i] = std::log10(data1[j * rows1 + i]);
							}
						}
					}
				}
				else
				{
					size_t rows1 = this->actual_rows;

					size_t final_pos_cols = this->final_pos_cols;
					size_t final_pos_rows = this->final_pos_rows;

					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
								data1[(j + 1) * rows1 + i],
								data1[(j + 2) * rows1 + i],
								data1[(j + 3) * rows1 + i]);
							_mm256_store_pd(&data_result[i * cols + j], _mm256_log10_pd(a));
						}
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

							__m256d exp = _mm256_log10_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data_result[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

							_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
						}
						for (size_t i = final_pos_rows; i < cols; i++)
						{
							data_result[i * cols + j] = std::log10(data1[j * rows1 + i]);
						}
					}
				}
			}
			else
			{
				if constexpr (return_transposed)
				{
					size_t cols1 = this->actual_cols;

					size_t final_pos_rows = this->final_pos_rows;
					size_t final_pos_cols = this->final_pos_cols;

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
								data1[(i + 1) * cols1 + j],
								data1[(i + 2) * cols1 + j],
								data1[(i + 3) * cols1 + j]);

							_mm256_store_pd(&data_result[j * rows + i], _mm256_log10_pd(a));
						}
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

							__m256d exp = _mm256_log10_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data_result[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data_result[(j + 1) * rows + i], val2);

							_mm_store_sd(&data_result[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data_result[(j + 3) * rows + i], val1);
						}
						for (size_t j = final_pos_cols; i < cols; j++)
						{
							data_result[j * rows + i] = std::log10(data1[i * cols1 + j]);
						}
					}
				}
				else
				{
					if constexpr (this_contiguous)
					{
						size_t size = this->_size;
						size_t final_pos_size = this->final_pos_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);
							_mm256_store_pd(&data_result[i], _mm256_log10_pd(a));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = std::log10(data1[i]);
						}
					}
					else
					{
						size_t cols1 = this->actual_cols;

						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

								_mm256_store_pd(&data_result[i * cols + j], _mm256_log10_pd(a));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < final_pos_rows; i += 4)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);

								__m256d exp = _mm256_log10_pd(a);

								__m128d val1 = _mm256_extractf128_pd(exp, 1);
								__m128d val2 = _mm256_castpd256_pd128(exp);

								_mm_store_sd(&data_result[i * cols + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

								_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
							}
							for (size_t i = final_pos_rows; i++)
							{
								data_result[i * cols + j] = std::log10(data1[i * cols1 + j]);
							}
						}
					}
				}
			}
			return result;
		}

		void self_log10()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			if constexpr (this_transposed)
			{
				if constexpr (this_contiguous)
				{
					size_t final_pos_size = this->final_pos_size;
					size_t size = this->_size;

					for (size_t i = 0; i < final_pos_size; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&data1[i]; _mm256_log10_pd(a));
					}
					for (size_t i = final_pos_size; i < size; i++)
					{
						data1[i] = std::log10(data1[i]);
					}
				}
				else
				{
					size_t rows1 = this->actual_rows;

					size_t final_pos_rows = this->final_pos_rows;
					size_t final_pos_cols = this->final_pos_cols;

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						for (size_t j = 0; j < final_pos_cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

							_mm256_store_pd(&data1[j * rows1 + i], _mm256_log10_pd(a));
						}
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
								data1[(j + 1) * rows1 + i],
								data1[(j + 2) * rows1 + i],
								data1[(j + 3) * rows1 + i]);

							__m256d exp = _mm256_log10_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data1[j * rows1 + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(j + 1) * rows1 + i], val2);

							_mm_store_sd(&data1[(j + 2) * rows1 + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(j + 3) * rows1 + i], val1);
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							data1[j * rows1 + i] = std::log10(data1[j * rows1 + i]);
						}
					}
				}
			}
			else
			{
				if constexpr (this_contiguous)
				{
					size_t size = this->_size;
					size_t final_pos_size = this->final_pos_size;

					for (size_t i = 0; i < final_pos_size; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);
						_mm256_store_pd(&data1[i], _mm256_log10_pd(a));
					}
					for (size_t i = final_pos_size; i < size; i++)
					{
						data1[i] = std::log10(data1[i]);
					}
				}
				else
				{
					size_t cols1 = this->actual_cols;

					size_t final_pos_cols = this->final_pos_cols;
					size_t final_pos_rows = this->final_pos_rows;

					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

							_mm256_store_pd(&data1[i * cols1 + j], _mm256_log10_pd(a));
						}
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
								data1[(i + 1) * cols1 + j],
								data1[(i + 2) * cols1 + j],
								data1[(i + 3) * cols1 + j]);

							__m256d exp = _mm256_log10_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data1[i * cols1 + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(i + 1) * cols1 + j], val2);

							_mm_store_sd(&data1[(i + 2) * cols1 + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(i + 3) * cols1 + j], val1);
						}
						for (size_t i = final_pos_rows; i++)
						{
							data1[i * cols1 + j] = std::log10(data1[i * cols1 + j]);
						}
					}
				}
			}
		}

#define _mm256_abs_pd(a) _mm256_andnot_pd(mask, (a))

		template<bool return_transposed = false>
		matrix<double, return_transposed> abs()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			matrix<double, return_transposed> result(rows, cols);

			double* data_result = result._data;

			__m256d mask = _mm256_set1_pd(-0.0);

			if constexpr (this_transposed)
			{
				if constexpr (return_transposed)
				{
					if constexpr (this_contiguous)
					{
						size_t final_pos_size = this->final_pos_size;
						size_t size = this->_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);

							_mm256_store_pd(&data_result[i]; _mm256_abs_pd(a));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = std::fabs(data1[i]);
						}
					}
					else
					{
						size_t rows1 = this->actual_rows;

						size_t final_pos_rows = this->final_pos_rows;
						size_t final_pos_cols = this->final_pos_cols;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < final_pos_cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

								_mm256_store_pd(&data_result[j * rows + i], _mm256_abs_pd(a));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < final_pos_cols; j += 4)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);

								__m256d exp = _mm256_abs_pd(a);

								__m128d val1 = _mm256_extractf128_pd(exp, 1);
								__m128d val2 = _mm256_castpd256_pd128(exp);

								_mm_store_sd(&data_result[j * rows + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(j + 1) * rows + i], val2);

								_mm_store_sd(&data_result[(j + 2) * rows + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(j + 3) * rows + i], val1);
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								data_result[j * rows + i] = std::fabs(data1[j * rows1 + i]);
							}
						}
					}
				}
				else
				{
					size_t rows1 = this->actual_rows;

					size_t final_pos_cols = this->final_pos_cols;
					size_t final_pos_rows = this->final_pos_rows;

					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
								data1[(j + 1) * rows1 + i],
								data1[(j + 2) * rows1 + i],
								data1[(j + 3) * rows1 + i]);
							_mm256_store_pd(&data_result[i * cols + j], _mm256_abs_pd(a));
						}
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

							__m256d exp = _mm256_abs_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data_result[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

							_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
						}
						for (size_t i = final_pos_rows; i < cols; i++)
						{
							data_result[i * cols + j] = std::fabs(data1[j * rows1 + i]);
						}
					}
				}
			}
			else
			{
				if constexpr (return_transposed)
				{
					size_t cols1 = this->actual_cols;

					size_t final_pos_rows = this->final_pos_rows;
					size_t final_pos_cols = this->final_pos_cols;

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
								data1[(i + 1) * cols1 + j],
								data1[(i + 2) * cols1 + j],
								data1[(i + 3) * cols1 + j]);

							_mm256_store_pd(&data_result[j * rows + i], _mm256_abs_pd(a));
						}
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

							__m256d exp = _mm256_abs_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data_result[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data_result[(j + 1) * rows + i], val2);

							_mm_store_sd(&data_result[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data_result[(j + 3) * rows + i], val1);
						}
						for (size_t j = final_pos_cols; i < cols; j++)
						{
							data_result[j * rows + i] = std::fabs(data1[i * cols1 + j]);
						}
					}
				}
				else
				{
					if constexpr (this_contiguous)
					{
						size_t size = this->_size;
						size_t final_pos_size = this->final_pos_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);
							_mm256_store_pd(&data_result[i], _mm256_abs_pd(a));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = std::fabs(data1[i]);
						}
					}
					else
					{
						size_t cols1 = this->actual_cols;

						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

								_mm256_store_pd(&data_result[i * cols + j], _mm256_abs_pd(a));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < final_pos_rows; i += 4)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);

								__m256d exp = _mm256_abs_pd(a);

								__m128d val1 = _mm256_extractf128_pd(exp, 1);
								__m128d val2 = _mm256_castpd256_pd128(exp);

								_mm_store_sd(&data_result[i * cols + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

								_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
							}
							for (size_t i = final_pos_rows; i++)
							{
								data_result[i * cols + j] = std::fabs(data1[i * cols1 + j]);
							}
						}
					}
				}
			}
			return result;
		}

		void self_abs()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			if constexpr (this_transposed)
			{
				if constexpr (this_contiguous)
				{
					size_t final_pos_size = this->final_pos_size;
					size_t size = this->_size;

					for (size_t i = 0; i < final_pos_size; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&data1[i]; _mm256_abs_pd(a));
					}
					for (size_t i = final_pos_size; i < size; i++)
					{
						data1[i] = std::fabs(data1[i]);
					}
				}
				else
				{
					size_t rows1 = this->actual_rows;

					size_t final_pos_rows = this->final_pos_rows;
					size_t final_pos_cols = this->final_pos_cols;

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						for (size_t j = 0; j < final_pos_cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

							_mm256_store_pd(&data1[j * rows1 + i], _mm256_abs_pd(a));
						}
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
								data1[(j + 1) * rows1 + i],
								data1[(j + 2) * rows1 + i],
								data1[(j + 3) * rows1 + i]);

							__m256d exp = _mm256_abs_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data1[j * rows1 + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(j + 1) * rows1 + i], val2);

							_mm_store_sd(&data1[(j + 2) * rows1 + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(j + 3) * rows1 + i], val1);
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							data1[j * rows1 + i] = std::fabs(data1[j * rows1 + i]);
						}
					}
				}
			}
			else
			{
				if constexpr (this_contiguous)
				{
					size_t size = this->_size;
					size_t final_pos_size = this->final_pos_size;

					for (size_t i = 0; i < final_pos_size; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);
						_mm256_store_pd(&data1[i], _mm256_abs_pd(a));
					}
					for (size_t i = final_pos_size; i < size; i++)
					{
						data1[i] = std::fabs(data1[i]);
					}
				}
				else
				{
					size_t cols1 = this->actual_cols;

					size_t final_pos_cols = this->final_pos_cols;
					size_t final_pos_rows = this->final_pos_rows;

					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

							_mm256_store_pd(&data1[i * cols1 + j], _mm256_abs_pd(a));
						}
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
								data1[(i + 1) * cols1 + j],
								data1[(i + 2) * cols1 + j],
								data1[(i + 3) * cols1 + j]);

							__m256d exp = _mm256_abs_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data1[i * cols1 + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(i + 1) * cols1 + j], val2);

							_mm_store_sd(&data1[(i + 2) * cols1 + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(i + 3) * cols1 + j], val1);
						}
						for (size_t i = final_pos_rows; i++)
						{
							data1[i * cols1 + j] = std::fabs(data1[i * cols1 + j]);
						}
					}
				}
			}
		}
		
		template<bool return_transposed = false>
		matrix<double, return_transposed> cos()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			matrix<double, return_transposed> result(rows, cols);

			double* data_result = result._data;

			if constexpr (this_transposed)
			{
				if constexpr (return_transposed)
				{
					if constexpr (this_contiguous)
					{
						size_t final_pos_size = this->final_pos_size;
						size_t size = this->_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);

							_mm256_store_pd(&data_result[i]; _mm256_cos_pd(a));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = std::cos(data1[i]);
						}
					}
					else
					{
						size_t rows1 = this->actual_rows;

						size_t final_pos_rows = this->final_pos_rows;
						size_t final_pos_cols = this->final_pos_cols;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < final_pos_cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

								_mm256_store_pd(&data_result[j * rows + i], _mm256_cos_pd(a));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < final_pos_cols; j += 4)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);

								__m256d exp = _mm256_cos_pd(a);

								__m128d val1 = _mm256_extractf128_pd(exp, 1);
								__m128d val2 = _mm256_castpd256_pd128(exp);

								_mm_store_sd(&data_result[j * rows + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(j + 1) * rows + i], val2);

								_mm_store_sd(&data_result[(j + 2) * rows + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(j + 3) * rows + i], val1);
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								data_result[j * rows + i] = std::cos(data1[j * rows1 + i]);
							}
						}
					}
				}
				else
				{
					size_t rows1 = this->actual_rows;

					size_t final_pos_cols = this->final_pos_cols;
					size_t final_pos_rows = this->final_pos_rows;

					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
								data1[(j + 1) * rows1 + i],
								data1[(j + 2) * rows1 + i],
								data1[(j + 3) * rows1 + i]);
							_mm256_store_pd(&data_result[i * cols + j], _mm256_cos_pd(a));
						}
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

							__m256d exp = _mm256_cos_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data_result[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

							_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
						}
						for (size_t i = final_pos_rows; i < cols; i++)
						{
							data_result[i * cols + j] = std::cos(data1[j * rows1 + i]);
						}
					}
				}
			}
			else
			{
				if constexpr (return_transposed)
				{
					size_t cols1 = this->actual_cols;

					size_t final_pos_rows = this->final_pos_rows;
					size_t final_pos_cols = this->final_pos_cols;

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
								data1[(i + 1) * cols1 + j],
								data1[(i + 2) * cols1 + j],
								data1[(i + 3) * cols1 + j]);

							_mm256_store_pd(&data_result[j * rows + i], _mm256_cos_pd(a));
						}
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

							__m256d exp = _mm256_cos_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data_result[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data_result[(j + 1) * rows + i], val2);

							_mm_store_sd(&data_result[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data_result[(j + 3) * rows + i], val1);
						}
						for (size_t j = final_pos_cols; i < cols; j++)
						{
							data_result[j * rows + i] = std::cos(data1[i * cols1 + j]);
						}
					}
				}
				else
				{
					if constexpr (this_contiguous)
					{
						size_t size = this->_size;
						size_t final_pos_size = this->final_pos_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);
							_mm256_store_pd(&data_result[i], _mm256_cos_pd(a));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = std::cos(data1[i]);
						}
					}
					else
					{
						size_t cols1 = this->actual_cols;

						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

								_mm256_store_pd(&data_result[i * cols + j], _mm256_cos_pd(a));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < final_pos_rows; i += 4)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);

								__m256d exp = _mm256_cos_pd(a);

								__m128d val1 = _mm256_extractf128_pd(exp, 1);
								__m128d val2 = _mm256_castpd256_pd128(exp);

								_mm_store_sd(&data_result[i * cols + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

								_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
							}
							for (size_t i = final_pos_rows; i++)
							{
								data_result[i * cols + j] = std::cos(data1[i * cols1 + j]);
							}
						}
					}
				}
			}
			return result;
		}

		void self_cos()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			if constexpr (this_transposed)
			{
				if constexpr (this_contiguous)
				{
					size_t final_pos_size = this->final_pos_size;
					size_t size = this->_size;

					for (size_t i = 0; i < final_pos_size; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&data1[i]; _mm256_cos_pd(a));
					}
					for (size_t i = final_pos_size; i < size; i++)
					{
						data1[i] = std::cos(data1[i]);
					}
				}
				else
				{
					size_t rows1 = this->actual_rows;

					size_t final_pos_rows = this->final_pos_rows;
					size_t final_pos_cols = this->final_pos_cols;

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						for (size_t j = 0; j < final_pos_cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

							_mm256_store_pd(&data1[j * rows1 + i], _mm256_cos_pd(a));
						}
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
								data1[(j + 1) * rows1 + i],
								data1[(j + 2) * rows1 + i],
								data1[(j + 3) * rows1 + i]);

							__m256d exp = _mm256_cos_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data1[j * rows1 + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(j + 1) * rows1 + i], val2);

							_mm_store_sd(&data1[(j + 2) * rows1 + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(j + 3) * rows1 + i], val1);
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							data1[j * rows1 + i] = std::cos(data1[j * rows1 + i]);
						}
					}
				}
			}
			else
			{
				if constexpr (this_contiguous)
				{
					size_t size = this->_size;
					size_t final_pos_size = this->final_pos_size;

					for (size_t i = 0; i < final_pos_size; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);
						_mm256_store_pd(&data1[i], _mm256_cos_pd(a));
					}
					for (size_t i = final_pos_size; i < size; i++)
					{
						data1[i] = std::cos(data1[i]);
					}
				}
				else
				{
					size_t cols1 = this->actual_cols;

					size_t final_pos_cols = this->final_pos_cols;
					size_t final_pos_rows = this->final_pos_rows;

					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

							_mm256_store_pd(&data1[i * cols1 + j], _mm256_cos_pd(a));
						}
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
								data1[(i + 1) * cols1 + j],
								data1[(i + 2) * cols1 + j],
								data1[(i + 3) * cols1 + j]);

							__m256d exp = _mm256_cos_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data1[i * cols1 + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(i + 1) * cols1 + j], val2);

							_mm_store_sd(&data1[(i + 2) * cols1 + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(i + 3) * cols1 + j], val1);
						}
						for (size_t i = final_pos_rows; i++)
						{
							data1[i * cols1 + j] = std::cos(data1[i * cols1 + j]);
						}
					}
				}
			}
		}

		template<bool return_transposed = false>
		matrix<double, return_transposed> tan()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			matrix<double, return_transposed> result(rows, cols);

			double* data_result = result._data;

			if constexpr (this_transposed)
			{
				if constexpr (return_transposed)
				{
					if constexpr (this_contiguous)
					{
						size_t final_pos_size = this->final_pos_size;
						size_t size = this->_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);

							_mm256_store_pd(&data_result[i]; _mm256_tan_pd(a));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = std::tan(data1[i]);
						}
					}
					else
					{
						size_t rows1 = this->actual_rows;

						size_t final_pos_rows = this->final_pos_rows;
						size_t final_pos_cols = this->final_pos_cols;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < final_pos_cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

								_mm256_store_pd(&data_result[j * rows + i], _mm256_tan_pd(a));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < final_pos_cols; j += 4)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);

								__m256d exp = _mm256_tan_pd(a);

								__m128d val1 = _mm256_extractf128_pd(exp, 1);
								__m128d val2 = _mm256_castpd256_pd128(exp);

								_mm_store_sd(&data_result[j * rows + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(j + 1) * rows + i], val2);

								_mm_store_sd(&data_result[(j + 2) * rows + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(j + 3) * rows + i], val1);
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								data_result[j * rows + i] = std::tan(data1[j * rows1 + i]);
							}
						}
					}
				}
				else
				{
					size_t rows1 = this->actual_rows;

					size_t final_pos_cols = this->final_pos_cols;
					size_t final_pos_rows = this->final_pos_rows;

					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
								data1[(j + 1) * rows1 + i],
								data1[(j + 2) * rows1 + i],
								data1[(j + 3) * rows1 + i]);
							_mm256_store_pd(&data_result[i * cols + j], _mm256_tan_pd(a));
						}
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

							__m256d exp = _mm256_tan_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data_result[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

							_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
						}
						for (size_t i = final_pos_rows; i < cols; i++)
						{
							data_result[i * cols + j] = std::tan(data1[j * rows1 + i]);
						}
					}
				}
			}
			else
			{
				if constexpr (return_transposed)
				{
					size_t cols1 = this->actual_cols;

					size_t final_pos_rows = this->final_pos_rows;
					size_t final_pos_cols = this->final_pos_cols;

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
								data1[(i + 1) * cols1 + j],
								data1[(i + 2) * cols1 + j],
								data1[(i + 3) * cols1 + j]);

							_mm256_store_pd(&data_result[j * rows + i], _mm256_tan_pd(a));
						}
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

							__m256d exp = _mm256_tan_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data_result[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data_result[(j + 1) * rows + i], val2);

							_mm_store_sd(&data_result[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data_result[(j + 3) * rows + i], val1);
						}
						for (size_t j = final_pos_cols; i < cols; j++)
						{
							data_result[j * rows + i] = std::tan(data1[i * cols1 + j]);
						}
					}
				}
				else
				{
					if constexpr (this_contiguous)
					{
						size_t size = this->_size;
						size_t final_pos_size = this->final_pos_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);
							_mm256_store_pd(&data_result[i], _mm256_tan_pd(a));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = std::tan(data1[i]);
						}
					}
					else
					{
						size_t cols1 = this->actual_cols;

						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

								_mm256_store_pd(&data_result[i * cols + j], _mm256_tan_pd(a));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < final_pos_rows; i += 4)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);

								__m256d exp = _mm256_tan_pd(a);

								__m128d val1 = _mm256_extractf128_pd(exp, 1);
								__m128d val2 = _mm256_castpd256_pd128(exp);

								_mm_store_sd(&data_result[i * cols + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

								_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
							}
							for (size_t i = final_pos_rows; i++)
							{
								data_result[i * cols + j] = std::tan(data1[i * cols1 + j]);
							}
						}
					}
				}
			}
			return result;
		}

		void self_tan()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			if constexpr (this_transposed)
			{
				if constexpr (this_contiguous)
				{
					size_t final_pos_size = this->final_pos_size;
					size_t size = this->_size;

					for (size_t i = 0; i < final_pos_size; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&data1[i]; _mm256_tan_pd(a));
					}
					for (size_t i = final_pos_size; i < size; i++)
					{
						data1[i] = std::tan(data1[i]);
					}
				}
				else
				{
					size_t rows1 = this->actual_rows;

					size_t final_pos_rows = this->final_pos_rows;
					size_t final_pos_cols = this->final_pos_cols;

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						for (size_t j = 0; j < final_pos_cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

							_mm256_store_pd(&data1[j * rows1 + i], _mm256_tan_pd(a));
						}
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
								data1[(j + 1) * rows1 + i],
								data1[(j + 2) * rows1 + i],
								data1[(j + 3) * rows1 + i]);

							__m256d exp = _mm256_tan_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data1[j * rows1 + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(j + 1) * rows1 + i], val2);

							_mm_store_sd(&data1[(j + 2) * rows1 + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(j + 3) * rows1 + i], val1);
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							data1[j * rows1 + i] = std::tan(data1[j * rows1 + i]);
						}
					}
				}
			}
			else
			{
				if constexpr (this_contiguous)
				{
					size_t size = this->_size;
					size_t final_pos_size = this->final_pos_size;

					for (size_t i = 0; i < final_pos_size; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);
						_mm256_store_pd(&data1[i], _mm256_tan_pd(a));
					}
					for (size_t i = final_pos_size; i < size; i++)
					{
						data1[i] = std::tan(data1[i]);
					}
				}
				else
				{
					size_t cols1 = this->actual_cols;

					size_t final_pos_cols = this->final_pos_cols;
					size_t final_pos_rows = this->final_pos_rows;

					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

							_mm256_store_pd(&data1[i * cols1 + j], _mm256_tan_pd(a));
						}
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
								data1[(i + 1) * cols1 + j],
								data1[(i + 2) * cols1 + j],
								data1[(i + 3) * cols1 + j]);

							__m256d exp = _mm256_tan_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data1[i * cols1 + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(i + 1) * cols1 + j], val2);

							_mm_store_sd(&data1[(i + 2) * cols1 + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(i + 3) * cols1 + j], val1);
						}
						for (size_t i = final_pos_rows; i++)
						{
							data1[i * cols1 + j] = std::tan(data1[i * cols1 + j]);
						}
					}
				}
			}
		}

		template<bool return_transposed = false>
		matrix<double, return_transposed> acos()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			matrix<double, return_transposed> result(rows, cols);

			double* data_result = result._data;

			if constexpr (this_transposed)
			{
				if constexpr (return_transposed)
				{
					if constexpr (this_contiguous)
					{
						size_t final_pos_size = this->final_pos_size;
						size_t size = this->_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);

							_mm256_store_pd(&data_result[i]; _mm256_acos_pd(a));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = std::acos(data1[i]);
						}
					}
					else
					{
						size_t rows1 = this->actual_rows;

						size_t final_pos_rows = this->final_pos_rows;
						size_t final_pos_cols = this->final_pos_cols;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < final_pos_cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

								_mm256_store_pd(&data_result[j * rows + i], _mm256_acos_pd(a));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < final_pos_cols; j += 4)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);

								__m256d exp = _mm256_acos_pd(a);

								__m128d val1 = _mm256_extractf128_pd(exp, 1);
								__m128d val2 = _mm256_castpd256_pd128(exp);

								_mm_store_sd(&data_result[j * rows + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(j + 1) * rows + i], val2);

								_mm_store_sd(&data_result[(j + 2) * rows + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(j + 3) * rows + i], val1);
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								data_result[j * rows + i] = std::acos(data1[j * rows1 + i]);
							}
						}
					}
				}
				else
				{
					size_t rows1 = this->actual_rows;

					size_t final_pos_cols = this->final_pos_cols;
					size_t final_pos_rows = this->final_pos_rows;

					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
								data1[(j + 1) * rows1 + i],
								data1[(j + 2) * rows1 + i],
								data1[(j + 3) * rows1 + i]);
							_mm256_store_pd(&data_result[i * cols + j], _mm256_acos_pd(a));
						}
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

							__m256d exp = _mm256_acos_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data_result[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

							_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
						}
						for (size_t i = final_pos_rows; i < cols; i++)
						{
							data_result[i * cols + j] = std::acos(data1[j * rows1 + i]);
						}
					}
				}
			}
			else
			{
				if constexpr (return_transposed)
				{
					size_t cols1 = this->actual_cols;

					size_t final_pos_rows = this->final_pos_rows;
					size_t final_pos_cols = this->final_pos_cols;

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
								data1[(i + 1) * cols1 + j],
								data1[(i + 2) * cols1 + j],
								data1[(i + 3) * cols1 + j]);

							_mm256_store_pd(&data_result[j * rows + i], _mm256_acos_pd(a));
						}
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

							__m256d exp = _mm256_acos_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data_result[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data_result[(j + 1) * rows + i], val2);

							_mm_store_sd(&data_result[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data_result[(j + 3) * rows + i], val1);
						}
						for (size_t j = final_pos_cols; i < cols; j++)
						{
							data_result[j * rows + i] = std::acos(data1[i * cols1 + j]);
						}
					}
				}
				else
				{
					if constexpr (this_contiguous)
					{
						size_t size = this->_size;
						size_t final_pos_size = this->final_pos_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);
							_mm256_store_pd(&data_result[i], _mm256_acos_pd(a));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = std::acos(data1[i]);
						}
					}
					else
					{
						size_t cols1 = this->actual_cols;

						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

								_mm256_store_pd(&data_result[i * cols + j], _mm256_acos_pd(a));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < final_pos_rows; i += 4)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);

								__m256d exp = _mm256_acos_pd(a);

								__m128d val1 = _mm256_extractf128_pd(exp, 1);
								__m128d val2 = _mm256_castpd256_pd128(exp);

								_mm_store_sd(&data_result[i * cols + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

								_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
							}
							for (size_t i = final_pos_rows; i++)
							{
								data_result[i * cols + j] = std::acos(data1[i * cols1 + j]);
							}
						}
					}
				}
			}
			return result;
		}

		void self_acos()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			if constexpr (this_transposed)
			{
				if constexpr (this_contiguous)
				{
					size_t final_pos_size = this->final_pos_size;
					size_t size = this->_size;

					for (size_t i = 0; i < final_pos_size; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&data1[i]; _mm256_acos_pd(a));
					}
					for (size_t i = final_pos_size; i < size; i++)
					{
						data1[i] = std::acos(data1[i]);
					}
				}
				else
				{
					size_t rows1 = this->actual_rows;

					size_t final_pos_rows = this->final_pos_rows;
					size_t final_pos_cols = this->final_pos_cols;

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						for (size_t j = 0; j < final_pos_cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

							_mm256_store_pd(&data1[j * rows1 + i], _mm256_acos_pd(a));
						}
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
								data1[(j + 1) * rows1 + i],
								data1[(j + 2) * rows1 + i],
								data1[(j + 3) * rows1 + i]);

							__m256d exp = _mm256_acos_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data1[j * rows1 + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(j + 1) * rows1 + i], val2);

							_mm_store_sd(&data1[(j + 2) * rows1 + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(j + 3) * rows1 + i], val1);
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							data1[j * rows1 + i] = std::acos(data1[j * rows1 + i]);
						}
					}
				}
			}
			else
			{
				if constexpr (this_contiguous)
				{
					size_t size = this->_size;
					size_t final_pos_size = this->final_pos_size;

					for (size_t i = 0; i < final_pos_size; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);
						_mm256_store_pd(&data1[i], _mm256_acos_pd(a));
					}
					for (size_t i = final_pos_size; i < size; i++)
					{
						data1[i] = std::acos(data1[i]);
					}
				}
				else
				{
					size_t cols1 = this->actual_cols;

					size_t final_pos_cols = this->final_pos_cols;
					size_t final_pos_rows = this->final_pos_rows;

					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

							_mm256_store_pd(&data1[i * cols1 + j], _mm256_acos_pd(a));
						}
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
								data1[(i + 1) * cols1 + j],
								data1[(i + 2) * cols1 + j],
								data1[(i + 3) * cols1 + j]);

							__m256d exp = _mm256_acos_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data1[i * cols1 + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(i + 1) * cols1 + j], val2);

							_mm_store_sd(&data1[(i + 2) * cols1 + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(i + 3) * cols1 + j], val1);
						}
						for (size_t i = final_pos_rows; i++)
						{
							data1[i * cols1 + j] = std::acos(data1[i * cols1 + j]);
						}
					}
				}
			}
		}

		template<bool return_transposed = false>
		matrix<double, return_transposed> round()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			matrix<double, return_transposed> result(rows, cols);

			double* data_result = result._data;

			if constexpr (this_transposed)
			{
				if constexpr (return_transposed)
				{
					if constexpr (this_contiguous)
					{
						size_t final_pos_size = this->final_pos_size;
						size_t size = this->_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);

							_mm256_store_pd(&data_result[i]; _mm256_round_pd(a, _MM_FROUND_TO_NEAREST_INT));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = std::round(data1[i]);
						}
					}
					else
					{
						size_t rows1 = this->actual_rows;

						size_t final_pos_rows = this->final_pos_rows;
						size_t final_pos_cols = this->final_pos_cols;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < final_pos_cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

								_mm256_store_pd(&data_result[j * rows + i], _mm256_round_pd(a, _MM_FROUND_TO_NEAREST_INT));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < final_pos_cols; j += 4)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);

								__m256d exp = _mm256_round_pd(a, _MM_FROUND_TO_NEAREST_INT);

								__m128d val1 = _mm256_extractf128_pd(exp, 1);
								__m128d val2 = _mm256_castpd256_pd128(exp);

								_mm_store_sd(&data_result[j * rows + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(j + 1) * rows + i], val2);

								_mm_store_sd(&data_result[(j + 2) * rows + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(j + 3) * rows + i], val1);
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								data_result[j * rows + i] = std::round(data1[j * rows1 + i]);
							}
						}
					}
				}
				else
				{
					size_t rows1 = this->actual_rows;

					size_t final_pos_cols = this->final_pos_cols;
					size_t final_pos_rows = this->final_pos_rows;

					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
								data1[(j + 1) * rows1 + i],
								data1[(j + 2) * rows1 + i],
								data1[(j + 3) * rows1 + i]);
							_mm256_store_pd(&data_result[i * cols + j], _mm256_round_pd(a, _MM_FROUND_TO_NEAREST_INT));
						}
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

							__m256d exp = _mm256_round_pd(a, _MM_FROUND_TO_NEAREST_INT);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data_result[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

							_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
						}
						for (size_t i = final_pos_rows; i < cols; i++)
						{
							data_result[i * cols + j] = std::round(data1[j * rows1 + i]);
						}
					}
				}
			}
			else
			{
				if constexpr (return_transposed)
				{
					size_t cols1 = this->actual_cols;

					size_t final_pos_rows = this->final_pos_rows;
					size_t final_pos_cols = this->final_pos_cols;

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
								data1[(i + 1) * cols1 + j],
								data1[(i + 2) * cols1 + j],
								data1[(i + 3) * cols1 + j]);

							_mm256_store_pd(&data_result[j * rows + i], _mm256_round_pd(a, _MM_FROUND_TO_NEAREST_INT));
						}
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

							__m256d exp = _mm256_round_pd(a, _MM_FROUND_TO_NEAREST_INT);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data_result[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data_result[(j + 1) * rows + i], val2);

							_mm_store_sd(&data_result[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data_result[(j + 3) * rows + i], val1);
						}
						for (size_t j = final_pos_cols; i < cols; j++)
						{
							data_result[j * rows + i] = std::round(data1[i * cols1 + j]);
						}
					}
				}
				else
				{
					if constexpr (this_contiguous)
					{
						size_t size = this->_size;
						size_t final_pos_size = this->final_pos_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);
							_mm256_store_pd(&data_result[i], _mm256_round_pd(a, _MM_FROUND_TO_NEAREST_INT));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = std::round(data1[i]);
						}
					}
					else
					{
						size_t cols1 = this->actual_cols;

						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

								_mm256_store_pd(&data_result[i * cols + j], _mm256_round_pd(a, _MM_FROUND_TO_NEAREST_INT));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < final_pos_rows; i += 4)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);

								__m256d exp = _mm256_round_pd(a, _MM_FROUND_TO_NEAREST_INT);

								__m128d val1 = _mm256_extractf128_pd(exp, 1);
								__m128d val2 = _mm256_castpd256_pd128(exp);

								_mm_store_sd(&data_result[i * cols + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

								_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
							}
							for (size_t i = final_pos_rows; i++)
							{
								data_result[i * cols + j] = std::round(data1[i * cols1 + j]);
							}
						}
					}
				}
			}
			return result;
		}
		
		void self_round()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			if constexpr (this_transposed)
			{
				if constexpr (this_contiguous)
				{
					size_t final_pos_size = this->final_pos_size;
					size_t size = this->_size;

					for (size_t i = 0; i < final_pos_size; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&data1[i]; _mm256_round_pd(a, _MM_FROUND_TO_NEAREST_INT));
					}
					for (size_t i = final_pos_size; i < size; i++)
					{
						data1[i] = std::round(data1[i]);
					}
				}
				else
				{
					size_t rows1 = this->actual_rows;

					size_t final_pos_rows = this->final_pos_rows;
					size_t final_pos_cols = this->final_pos_cols;

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						for (size_t j = 0; j < final_pos_cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

							_mm256_store_pd(&data1[j * rows1 + i], _mm256_round_pd(a, _MM_FROUND_TO_NEAREST_INT));
						}
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
								data1[(j + 1) * rows1 + i],
								data1[(j + 2) * rows1 + i],
								data1[(j + 3) * rows1 + i]);

							__m256d exp = _mm256_round_pd(a, _MM_FROUND_TO_NEAREST_INT);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data1[j * rows1 + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(j + 1) * rows1 + i], val2);

							_mm_store_sd(&data1[(j + 2) * rows1 + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(j + 3) * rows1 + i], val1);
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							data1[j * rows1 + i] = std::round(data1[j * rows1 + i]);
						}
					}
				}
			}
			else
			{
				if constexpr (this_contiguous)
				{
					size_t size = this->_size;
					size_t final_pos_size = this->final_pos_size;

					for (size_t i = 0; i < final_pos_size; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);
						_mm256_store_pd(&data1[i], _mm256_round_pd(a, _MM_FROUND_TO_NEAREST_INT));
					}
					for (size_t i = final_pos_size; i < size; i++)
					{
						data1[i] = std::round(data1[i]);
					}
				}
				else
				{
					size_t cols1 = this->actual_cols;

					size_t final_pos_cols = this->final_pos_cols;
					size_t final_pos_rows = this->final_pos_rows;

					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

							_mm256_store_pd(&data1[i * cols1 + j], _mm256_round_pd(a, _MM_FROUND_TO_NEAREST_INT));
						}
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
								data1[(i + 1) * cols1 + j],
								data1[(i + 2) * cols1 + j],
								data1[(i + 3) * cols1 + j]);

							__m256d exp = _mm256_round_pd(a, _MM_FROUND_TO_NEAREST_INT);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data1[i * cols1 + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(i + 1) * cols1 + j], val2);

							_mm_store_sd(&data1[(i + 2) * cols1 + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(i + 3) * cols1 + j], val1);
						}
						for (size_t i = final_pos_rows; i++)
						{
							data1[i * cols1 + j] = std::round(data1[i * cols1 + j]);
						}
					}
				}
			}
		}

		template<bool return_transposed = false>
		matrix<double, return_transposed> floor()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			matrix<double, return_transposed> result(rows, cols);

			double* data_result = result._data;

			if constexpr (this_transposed)
			{
				if constexpr (return_transposed)
				{
					if constexpr (this_contiguous)
					{
						size_t final_pos_size = this->final_pos_size;
						size_t size = this->_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);

							_mm256_store_pd(&data_result[i]; _mm256_round_pd(a, _MM_FROUND_FLOOR));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = std::floor(data1[i]);
						}
					}
					else
					{
						size_t rows1 = this->actual_rows;

						size_t final_pos_rows = this->final_pos_rows;
						size_t final_pos_cols = this->final_pos_cols;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < final_pos_cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

								_mm256_store_pd(&data_result[j * rows + i], _mm256_round_pd(a, _MM_FROUND_FLOOR));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < final_pos_cols; j += 4)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);

								__m256d exp = _mm256_round_pd(a, _MM_FROUND_FLOOR);

								__m128d val1 = _mm256_extractf128_pd(exp, 1);
								__m128d val2 = _mm256_castpd256_pd128(exp);

								_mm_store_sd(&data_result[j * rows + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(j + 1) * rows + i], val2);

								_mm_store_sd(&data_result[(j + 2) * rows + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(j + 3) * rows + i], val1);
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								data_result[j * rows + i] = std::floor(data1[j * rows1 + i]);
							}
						}
					}
				}
				else
				{
					size_t rows1 = this->actual_rows;

					size_t final_pos_cols = this->final_pos_cols;
					size_t final_pos_rows = this->final_pos_rows;

					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
								data1[(j + 1) * rows1 + i],
								data1[(j + 2) * rows1 + i],
								data1[(j + 3) * rows1 + i]);
							_mm256_store_pd(&data_result[i * cols + j], _mm256_round_pd(a, _MM_FROUND_FLOOR));
						}
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

							__m256d exp = _mm256_round_pd(a, _MM_FROUND_FLOOR);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data_result[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

							_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
						}
						for (size_t i = final_pos_rows; i < cols; i++)
						{
							data_result[i * cols + j] = std::floor(data1[j * rows1 + i]);
						}
					}
				}
			}
			else
			{
				if constexpr (return_transposed)
				{
					size_t cols1 = this->actual_cols;

					size_t final_pos_rows = this->final_pos_rows;
					size_t final_pos_cols = this->final_pos_cols;

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
								data1[(i + 1) * cols1 + j],
								data1[(i + 2) * cols1 + j],
								data1[(i + 3) * cols1 + j]);

							_mm256_store_pd(&data_result[j * rows + i], _mm256_round_pd(a, _MM_FROUND_FLOOR));
						}
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

							__m256d exp = _mm256_round_pd(a, _MM_FROUND_FLOOR);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data_result[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data_result[(j + 1) * rows + i], val2);

							_mm_store_sd(&data_result[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data_result[(j + 3) * rows + i], val1);
						}
						for (size_t j = final_pos_cols; i < cols; j++)
						{
							data_result[j * rows + i] = std::floor(data1[i * cols1 + j]);
						}
					}
				}
				else
				{
					if constexpr (this_contiguous)
					{
						size_t size = this->_size;
						size_t final_pos_size = this->final_pos_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);
							_mm256_store_pd(&data_result[i], _mm256_round_pd(a, _MM_FROUND_FLOOR));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = std::floor(data1[i]);
						}
					}
					else
					{
						size_t cols1 = this->actual_cols;

						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

								_mm256_store_pd(&data_result[i * cols + j], _mm256_round_pd(a, _MM_FROUND_FLOOR));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < final_pos_rows; i += 4)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);

								__m256d exp = _mm256_round_pd(a, _MM_FROUND_FLOOR);

								__m128d val1 = _mm256_extractf128_pd(exp, 1);
								__m128d val2 = _mm256_castpd256_pd128(exp);

								_mm_store_sd(&data_result[i * cols + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

								_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
							}
							for (size_t i = final_pos_rows; i++)
							{
								data_result[i * cols + j] = std::floor(data1[i * cols1 + j]);
							}
						}
					}
				}
			}
			return result;
		}

		void self_floor()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			if constexpr (this_transposed)
			{
				if constexpr (this_contiguous)
				{
					size_t final_pos_size = this->final_pos_size;
					size_t size = this->_size;

					for (size_t i = 0; i < final_pos_size; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&data1[i]; _mm256_round_pd(a, _MM_FROUND_FLOOR));
					}
					for (size_t i = final_pos_size; i < size; i++)
					{
						data1[i] = std::floor(data1[i]);
					}
				}
				else
				{
					size_t rows1 = this->actual_rows;

					size_t final_pos_rows = this->final_pos_rows;
					size_t final_pos_cols = this->final_pos_cols;

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						for (size_t j = 0; j < final_pos_cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

							_mm256_store_pd(&data1[j * rows1 + i], _mm256_round_pd(a, _MM_FROUND_FLOOR));
						}
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
								data1[(j + 1) * rows1 + i],
								data1[(j + 2) * rows1 + i],
								data1[(j + 3) * rows1 + i]);

							__m256d exp = _mm256_round_pd(a, _MM_FROUND_FLOOR);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data1[j * rows1 + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(j + 1) * rows1 + i], val2);

							_mm_store_sd(&data1[(j + 2) * rows1 + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(j + 3) * rows1 + i], val1);
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							data1[j * rows1 + i] = std::floor(data1[j * rows1 + i]);
						}
					}
				}
			}
			else
			{
				if constexpr (this_contiguous)
				{
					size_t size = this->_size;
					size_t final_pos_size = this->final_pos_size;

					for (size_t i = 0; i < final_pos_size; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);
						_mm256_store_pd(&data1[i], _mm256_round_pd(a, _MM_FROUND_FLOOR));
					}
					for (size_t i = final_pos_size; i < size; i++)
					{
						data1[i] = std::floor(data1[i]);
					}
				}
				else
				{
					size_t cols1 = this->actual_cols;

					size_t final_pos_cols = this->final_pos_cols;
					size_t final_pos_rows = this->final_pos_rows;

					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

							_mm256_store_pd(&data1[i * cols1 + j], _mm256_round_pd(a, _MM_FROUND_FLOOR));
						}
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
								data1[(i + 1) * cols1 + j],
								data1[(i + 2) * cols1 + j],
								data1[(i + 3) * cols1 + j]);

							__m256d exp = _mm256_round_pd(a, _MM_FROUND_FLOOR);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&data1[i * cols1 + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(i + 1) * cols1 + j], val2);

							_mm_store_sd(&data1[(i + 2) * cols1 + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(i + 3) * cols1 + j], val1);
						}
						for (size_t i = final_pos_rows; i++)
						{
							data1[i * cols1 + j] = std::floor(data1[i * cols1 + j]);
						}
					}
				}
			}
		}

		// pow

		template<bool return_transposed = false>
		matrix<double, return_transposed> pow(double num)
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			matrix<double, return_transposed, true> result(rows, cols);

			double* data_result = result._data;

			__m256d b = _mm256_set1_pd(num);

			if constexpr (this_transposed)
			{
				if constexpr (return_transposed)
				{
					if constexpr (this_contiguous)
					{
						size_t size = this->_size;

						size_t final_pos_size = this->final_pos_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);

							_mm256_store_pd(&data_result[i], _mm256_pow_pd(a, b));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = std::pow(data1[i], num);
						}
					}
					else
					{
						size_t final_pos_rows = this->final_pos_rows;
						size_t final_pos_cols = this->final_pos_cols;

						size_t rows1 = this->actual_rows;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

								_mm256_store_pd(&data_result[j * rows + i], _mm256_pow_pd(a, b));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < final_pos_cols; j += 4)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);

								__m256d pow = _mm256_pow_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(pow, 1);
								__m128d val2 = _mm256_castpd256_pd128(pow);

								_mm_store_sd(&data_result[j * cols + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(j + 1) * cols + i], val2);

								_mm_store_sd(&data_result[(j + 2) * cols + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(j + 3) * cols + i], val1);
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								data_result[j * cols + i] = std::pow(data1[j * rows1 + i], num);
							}
						}
					}
				}
				else
				{
					size_t final_pos_cols = this->final_pos_cols;
					size_t final_pos_rows = this->final_pos_rows;

					size_t rows1 = this->actual_rows;

					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
								data1[(j + 1) * rows1 + i],
								data1[(j + 2) * rows1 + i],
								data1[(j + 3) * rows1 + i]);

							_mm256_store_pd(&data_result[i * cols + j], _mm256_pow_pd(a, b));
						}
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

							__m256d pow = _mm256_pow_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(pow, 1);
							__m128d val2 = _mm256_castpd256_pd128(pow);

							_mm_store_sd(&data_result[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

							_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							data_result[i * cols + j] = std::pow(data1[j * rows1 + i], num);
						}
					}
				}
			}
			else
			{
				if constexpr (return_transposed)
				{
					size_t cols1 = this->actual_cols;

					size_t final_pos_rows = this->final_pos_rows;
					size_t final_pos_cols = this->final_pos_cols;

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						for (size_t j = 0; j < rows; j++)
						{
							__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
								data1[(i + 1) * cols1 + j],
								data1[(i + 2) * cols1 + j],
								data1[(i + 3) * cols1 + j]);

							_mm256_store_pd(&data_result[j * rows + i], _mm256_pow_pd(a, b));
						}
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

							__m256d pow = _mm256_pow_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(pow, 1);
							__m128d val2 = _mm256_castpd256_pd128(pow);

							_mm_store_sd(&data_result[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data_result[(j + 1) * rows + i], val2);

							_mm_store_sd(&data_result[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data_result[(j + 3) * rows + i], val1);
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							data_result[j * rows + i] = std::pow(data1[i * cols1 + j], num);
						}
					}
				}
				else
				{
					if constexpr (this_contiguous)
					{
						size_t final_pos_size = this->final_pos_size;
						size_t size = this->_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);

							_mm256_store_pd(&data_result[i], _mm256_pow_pd(a, b));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = std::pow(data1[i], num);
						}
					}
					else
					{
						size_t cols1 = this->actual_cols;

						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

								_mm256_store_pd(&data_result[i * cols + j], _mm256_pow_pd(a, b));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < final_pos_rows; i += 4)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);

								__m256d pow = _mm256_pow_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(pow, 1);
								__m128d val2 = _mm256_castpd256_pd128(pow);

								_mm_store_sd(&data_result[i * cols + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

								_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								data_result[i * cols + j] = std::pow(data1[i * cols1 + j], num);
							}
						}
					}
				}
			}
			return result;
		}

		template<bool return_transposed = false, bool other_transposed, bool other_contiguous, bool call_destructor>
		matrix<double, return_transposed> pow(const matrix<double, other_transposed, other_contiguous, call_destructor>& other)
		{
#ifdef _DEBUG
			if (other._cols != this->_cols || other._rows != this->_rows) throw std::invalid_argument("The dimensions of both matrices must be the same");
#else
#endif

			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;
			double* data2 = other._data;

			matrix<double, return_transposed, true> result(rows, cols);

			double* data_result = result._data;

			if constexpr (this_transposed)
			{
				if constexpr (other_transposed)
				{
					if constexpr (return_transposed)
					{
						if constexpr (this_contiguous && other_contiguous)
						{
							size_t size = this->_size;

							size_t final_pos_size = this->final_pos_size;

							for (size_t i = 0; i < final_pos_size; i += 4)
							{
								__m256d a = _mm256_load_pd(&data1[i]);
								__m256d b = _mm256_load_pd(&data2[i]);

								_mm256_store_pd(&data_result[i], _mm256_pow_pd(a, b));
							}
							for (size_t i = final_pos_size; i < size; i++)
							{
								data_result[i] = std::pow(data1[i], data2[i]);
							}
						}
						else
						{
							size_t final_pos_rows = this->final_pos_rows;
							size_t final_pos_cols = this->final_pos_cols;

							size_t rows1 = this->actual_rows;
							size_t rows2 = other.actual_rows;

							for (size_t i = 0; i < final_pos_rows; i += 4)
							{
								for (size_t j = 0; j < cols; j++)
								{
									__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
									__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);

									_mm256_store_pd(&data_result[j * rows + i], _mm256_pow_pd(a, b));
								}
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								for (size_t j = 0; j < final_pos_cols; j += 4)
								{
									__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
										data1[(j + 1) * rows1 + i],
										data1[(j + 2) * rows1 + i],
										data1[(j + 3) * rows1 + i]);
									__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
										data2[(j + 1) * rows2 + i],
										data2[(j + 2) * rows2 + i],
										data2[(j + 3) * rows2 + i]);

									__m256d pow = _mm256_pow_pd(a, b);

									__m128d val1 = _mm256_extractf128_pd(pow, 1);
									__m128d val2 = _mm256_castpd256_pd128(pow);

									_mm_store_sd(&data_result[j * cols + i], val2);
									val2 = _mm_shuffle_pd(val2, val2, 1);
									_mm_store_sd(&data_result[(j + 1) * cols + i], val2);

									_mm_store_sd(&data_result[(j + 2) * cols + i], val1);
									val1 = _mm_shuffle_pd(val1, val1, 1);
									_mm_store_sd(&data_result[(j + 3) * cols + i], val1);
								}
								for (size_t j = final_pos_cols; j < cols; j++)
								{
									data_result[j * cols + i] = std::pow(data1[j * rows1 + i], data2[j * rows2 + i]);
								}
							}
						}
					}
					else
					{
						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						size_t rows1 = this->actual_rows;
						size_t rows2 = other.actual_rows;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);
								__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
									data2[(j + 1) * rows2 + i],
									data2[(j + 2) * rows2 + i],
									data2[(j + 3) * rows2 + i]);

								_mm256_store_pd(&data_result[i * cols + j], _mm256_pow_pd(a, b));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < final_pos_rows; i += 4)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
								__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);

								__m256d pow = _mm256_pow_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(pow, 1);
								__m128d val2 = _mm256_castpd256_pd128(pow);

								_mm_store_sd(&data_result[i * cols + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

								_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								data_result[i * cols + j] = std::pow(data1[j * rows1 + i], data2[j * rows2 + i]);
							}
						}
					}
				}
				else
				{
					if constexpr (return_transposed)
					{
						size_t final_pos_rows = this->final_pos_rows;
						size_t final_pos_cols = this->final_pos_cols;

						size_t rows1 = this->actual_rows;
						size_t cols2 = other.actual_cols;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
								__m256d b = _mm256_setr_pd(data2[i * cols2 + j],
									data2[(i + 1) * cols2 + j],
									data2[(i + 2) * cols2 + j],
									data2[(i + 3) * cols2 + j]);

								_mm256_store_pd(&data_result[j * rows + i], _mm256_pow_pd(a, b));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < final_pos_cols; j += 4)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);
								__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

								__m256d pow = _mm256_pow_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(pow, 1);
								__m128d val2 = _mm256_castpd256_pd128(pow);

								_mm_store_sd(&data_result[j * cols + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(j + 1) * cols + i], val2);

								_mm_store_sd(&data_result[(j + 2) * cols + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(j + 3) * cols + i], val1);
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								data_result[j * cols + i] = std::pow(data1[j * rows1 + i], data2[i * cols2 + j]);
							}
						}
					}
					else
					{
						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						size_t rows1 = this->actual_rows;
						size_t cols2 = other.actual_cols;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);
								__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

								_mm256_store_pd(&data_result[i * cols + j], _mm256_pow_pd(a, b));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < final_pos_rows; i += 4)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
								__m256d b = _mm256_setr_pd(data1[i * cols2 + j],
									data1[(i + 1) * cols2 + j],
									data1[(i + 2) * cols2 + j],
									data1[(i + 3) * cols2 + j]);

								__m256d pow = _mm256_pow_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(pow, 1);
								__m128d val2 = _mm256_castpd256_pd128(pow);

								_mm_store_sd(&data_result[i * cols + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

								_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								data_result[i * cols + j] = std::pow(data1[j * rows1 + i], data2[i * cols2 + j]);
							}
						}
					}
				}
			}
			else
			{
				if constexpr (other_transposed)
				{
					size_t cols1 = this->actual_cols;
					size_t rows2 = other.actual_rows;

					if constexpr (return_transposed)
					{
						size_t final_pos_rows = this->final_pos_rows;
						size_t final_pos_cols = this->final_pos_cols;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);
								__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);
								_mm256_store_pd(&data_result[j * rows + i], _mm256_pow_pd(a, b));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; final_pos_cols; j += 4)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
								__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
									data2[(j + 1) * rows2 + i],
									data2[(j + 2) * rows2 + i],
									data2[(j + 3) * rows2 + i]);

								__m256d pow = _mm256_pow_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(pow, 1);
								__m128d val2 = _mm256_castpd256_pd128(pow);

								_mm_store_sd(&data_result[j * rows + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(j + 1) * rows + i], val2);

								_mm_store_sd(&data_result[(j + 2) * rows + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(j + 3) * rows + i], val1);
							}
							for (size_t j = final_pos_cols; cols; j++)
							{
								data_result[j * rows + i] = std::pow(data1[i * cols1 + j], data2[j * rows2 + i]);
							}
						}
					}
					else
					{
						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
								__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
									data2[(j + 1) * rows2 + i],
									data2[(j + 2) * rows2 + i],
									data2[(j + 3) * rows2 + i]);
								_mm256_store_pd(&data_result[i * cols + j], _mm256_pow_pd(a, b));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < final_pos_rows; i += 4)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);
								__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);

								__m256d pow = _mm256_pow_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(pow, 1);
								__m128d val2 = _mm256_castpd256_pd128(pow);

								_mm_store_sd(&data_result[i * cols + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(i + 1) * rows + j], val2);

								_mm_store_sd(&data_result[(i + 2) * rows + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(i + 3) * rows + j], val1);
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								data_result[i * cols + j] = std::pow(data1[i * cols1 + j], data2[j * rows2 + i]);
							}
						}
					}
				}
				else
				{
					if constexpr (return_transposed)
					{
						size_t cols1 = this->actual_cols;
						size_t cols2 = other.actual_cols;

						size_t final_pos_rows = this->final_pos_rows;
						size_t final_pos_cols = this->final_pos_cols;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < rows; j++)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);
								__m256d b = _mm256_setr_pd(data2[i * cols2 + j],
									data2[(i + 1) * cols2 + j],
									data2[(i + 2) * cols2 + j],
									data2[(i + 3) * cols2 + j]);
								_mm256_store_pd(&data_result[j * rows + i], _mm256_pow_pd(a, b));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < final_pos_cols; j += 4)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
								__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

								__m256d pow = _mm256_pow_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(pow, 1);
								__m128d val2 = _mm256_castpd256_pd128(pow);

								_mm_store_sd(&data_result[j * rows + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(j + 1) * rows + i], val2);

								_mm_store_sd(&data_result[(j + 2) * rows + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(j + 3) * rows + i], val1);
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								data_result[j * rows + i] = std::pow(data1[i * cols1 + j], data2[i * cols2 + j]);
							}
						}
					}
					else
					{
						if constexpr (this_contiguous && other_contiguous)
						{
							size_t final_pos_size = this->final_pos_size;
							size_t size = this->_size;

							for (size_t i = 0; i < final_pos_size; i += 4)
							{
								__m256d a = _mm256_load_pd(&data1[i]);
								__m256d b = _mm256_load_pd(&data2[i]);

								_mm256_store_pd(&data_result[i], _mm256_pow_pd(a, b));
							}
							for (size_t i = final_pos_size; i < size; i++)
							{
								data_result[i] = std::pow(data1[i], data2[i]);
							}
						}
						else
						{
							size_t cols1 = this->actual_cols;
							size_t cols2 = other.actual_cols;

							size_t final_pos_cols = this->final_pos_cols;
							size_t final_pos_rows = this->final_pos_rows;

							for (size_t j = 0; j < final_pos_cols; j += 4)
							{
								for (size_t i = 0; i < rows; i++)
								{
									__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
									__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

									_mm256_store_pd(&data_result[i * cols + j], _mm256_pow_pd(a, b));
								}
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								for (size_t i = 0; i < final_pos_rows; i += 4)
								{
									__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
										data1[(i + 1) * cols1 + j],
										data1[(i + 2) * cols1 + j],
										data1[(i + 3) * cols1 + j]);
									__m256d b = _mm256_setr_pd(data2[i * cols2 + j],
										data2[(i + 1) * cols2 + j],
										data2[(i + 2) * cols2 + j],
										data2[(i + 3) * cols2 + j]);

									__m256d pow = _mm256_pow_pd(a, b);

									__m128d val1 = _mm256_extractf128_pd(pow, 1);
									__m128d val2 = _mm256_castpd256_pd128(pow);

									_mm_store_sd(&data_result[i * cols + j], val2);
									val2 = _mm_shuffle_pd(val2, val2, 1);
									_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

									_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
									val1 = _mm_shuffle_pd(val1, val1, 1);
									_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
								}
								for (size_t i = final_pos_rows; i < rows; i++)
								{
									data_result[i * cols + j] = std::pow(data1[i * cols1 + j], data2[i * cols2 + j]);
								}
							}
						}
					}
				}
			}
			return result;
		}

		void self_pow(double num)
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			__m256d b = _mm256_set1_pd(num);

			if constexpr (this_transposed)
			{
				if constexpr (this_contiguous)
				{
					size_t size = this->_size;

					size_t final_pos_size = this->final_pos_size;

					for (size_t i = 0; i < final_pos_size; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&data1[i], _mm256_pow_pd(a, b));
					}
					for (size_t i = final_pos_size; i < size; i++)
					{
						data1[i] = std::pow(data1[i], num);
					}
				}
				else
				{
					size_t final_pos_rows = this->final_pos_rows;
					size_t final_pos_cols = this->final_pos_cols;

					size_t rows1 = this->actual_rows;

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

							_mm256_store_pd(&data1[j * rows1 + i], _mm256_pow_pd(a, b));
						}
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
								data1[(j + 1) * rows1 + i],
								data1[(j + 2) * rows1 + i],
								data1[(j + 3) * rows1 + i]);

							__m256d pow = _mm256_pow_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(pow, 1);
							__m128d val2 = _mm256_castpd256_pd128(pow);

							_mm_store_sd(&data1[j * rows1 + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(j + 1) * rows1 + i], val2);

							_mm_store_sd(&data1[(j + 2) * rows1 + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(j + 3) * rows1 + i], val1);
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							data1[j * rows1 + i] = std::pow(data1[j * rows1 + i], num);
						}
					}
				}
			}
			else
			{
				if constexpr (this_contiguous)
				{
					size_t final_pos_size = this->final_pos_size;
					size_t size = this->_size;

					for (size_t i = 0; i < final_pos_size; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&data1[i], _mm256_pow_pd(a, b));
					}
					for (size_t i = final_pos_size; i < size; i++)
					{
						data1[i] = std::pow(data1[i], num);
					}
				}
				else
				{
					size_t cols1 = this->actual_cols;

					size_t final_pos_cols = this->final_pos_cols;
					size_t final_pos_rows = this->final_pos_rows;

					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

							_mm256_store_pd(&data1[i * cols1 + j], _mm256_pow_pd(a, b));
						}
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
								data1[(i + 1) * cols1 + j],
								data1[(i + 2) * cols1 + j],
								data1[(i + 3) * cols1 + j]);

							__m256d pow = _mm256_pow_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(pow, 1);
							__m128d val2 = _mm256_castpd256_pd128(pow);

							_mm_store_sd(&data1[i * cols1 + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(i + 1) * cols1 + j], val2);

							_mm_store_sd(&data1[(i + 2) * cols1 + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(i + 3) * cols1 + j], val1);
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							data1[i * cols1 + j] = std::pow(data1[i * cols1 + j], num);
						}
					}
				}
			}
		}

		template<bool other_transposed, bool other_contiguous, bool call_destructor>
		void self_pow(const matrix<double, other_transposed, other_contiguous, call_destructor>& other)
		{
#ifdef _DEBUG
			if (other._cols != this->_cols || other._rows != this->_rows) throw std::invalid_argument("The dimensions of both matrices must be the same");
#else
#endif

			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;
			double* data2 = other._data;

			if constexpr (this_transposed)
			{
				if constexpr (other_transposed)
				{
					if constexpr (this_contiguous && other_contiguous)
					{
						size_t size = this->_size;

						size_t final_pos_size = this->final_pos_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);
							__m256d b = _mm256_load_pd(&data2[i]);

							_mm256_store_pd(&data1[i], _mm256_pow_pd(a, b));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data1[i] = std::pow(data1[i], data2[i]);
						}
					}
					else
					{
						size_t final_pos_rows = this->final_pos_rows;
						size_t final_pos_cols = this->final_pos_cols;

						size_t rows1 = this->actual_rows;
						size_t rows2 = other.actual_rows;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
								__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);

								_mm256_store_pd(&data1[j * rows1 + i], _mm256_pow_pd(a, b));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < final_pos_cols; j += 4)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);
								__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
									data2[(j + 1) * rows2 + i],
									data2[(j + 2) * rows2 + i],
									data2[(j + 3) * rows2 + i]);

								__m256d pow = _mm256_pow_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(pow, 1);
								__m128d val2 = _mm256_castpd256_pd128(pow);

								_mm_store_sd(&data1[j * rows1 + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data1[(j + 1) * rows1 + i], val2);

								_mm_store_sd(&data1[(j + 2) * rows1 + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data1[(j + 3) * rows1 + i], val1);
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								data1[j * rows1 + i] = std::pow(data1[j * rows1 + i], data2[j * rows2 + i]);
							}
						}
					}
				}
				else
				{
					size_t final_pos_rows = this->final_pos_rows;
					size_t final_pos_cols = this->final_pos_cols;

					size_t rows1 = this->actual_rows;
					size_t cols2 = other.actual_cols;

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
							__m256d b = _mm256_setr_pd(data2[i * cols2 + j],
								data2[(i + 1) * cols2 + j],
								data2[(i + 2) * cols2 + j],
								data2[(i + 3) * cols2 + j]);

							_mm256_store_pd(&data1[j * rows1 + i], _mm256_pow_pd(a, b));
						}
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
								data1[(j + 1) * rows1 + i],
								data1[(j + 2) * rows1 + i],
								data1[(j + 3) * rows1 + i]);
							__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

							__m256d pow = _mm256_pow_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(pow, 1);
							__m128d val2 = _mm256_castpd256_pd128(pow);

							_mm_store_sd(&data1[j * rows1 + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(j + 1) * rows1 + i], val2);

							_mm_store_sd(&data1[(j + 2) * rows1 + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(j + 3) * rows1 + i], val1);
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							data1[j * rows1 + i] = std::pow(data1[j * rows1 + i], data2[i * cols2 + j]);
						}
					}
				}
			}
			else
			{
				if constexpr (other_transposed)
				{
					size_t cols1 = this->actual_cols;
					size_t rows2 = other.actual_rows;

					size_t final_pos_cols = this->final_pos_cols;
					size_t final_pos_rows = this->final_pos_rows;

					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
							__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
								data2[(j + 1) * rows2 + i],
								data2[(j + 2) * rows2 + i],
								data2[(j + 3) * rows2 + i]);
							_mm256_store_pd(&data1[i * cols1 + j], _mm256_pow_pd(a, b));
						}
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
								data1[(i + 1) * cols1 + j],
								data1[(i + 2) * cols1 + j],
								data1[(i + 3) * cols1 + j]);
							__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);

							__m256d pow = _mm256_pow_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(pow, 1);
							__m128d val2 = _mm256_castpd256_pd128(pow);

							_mm_store_sd(&data1[i * cols1 + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(i + 1) * cols1 + j], val2);

							_mm_store_sd(&data1[(i + 2) * cols1 + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(i + 3) * cols1 + j], val1);
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							data1[i * cols1 + j] = std::pow(data1[i * cols1 + j], data2[j * rows2 + i]);
						}
					}
				}
				else
				{
					if constexpr (this_contiguous && other_contiguous)
					{
						size_t final_pos_size = this->final_pos_size;
						size_t size = this->_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);
							__m256d b = _mm256_load_pd(&data2[i]);

							_mm256_store_pd(&data1[i], _mm256_pow_pd(a, b));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data1[i] = std::pow(data1[i], data2[i]);
						}
					}
					else
					{
						size_t cols1 = this->actual_cols;
						size_t cols2 = other.actual_cols;

						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
								__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

								_mm256_store_pd(&data1[i * cols1 + j], _mm256_pow_pd(a, b));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < final_pos_rows; i += 4)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);
								__m256d b = _mm256_setr_pd(data2[i * cols2 + j],
									data2[(i + 1) * cols2 + j],
									data2[(i + 2) * cols2 + j],
									data2[(i + 3) * cols2 + j]);

								__m256d pow = _mm256_pow_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(pow, 1);
								__m128d val2 = _mm256_castpd256_pd128(pow);

								_mm_store_sd(&data1[i * cols1 + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data1[(i + 1) * cols1 + j], val2);

								_mm_store_sd(&data1[(i + 2) * cols1 + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data1[(i + 3) * cols1 + j], val1);
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								data1[i * cols1 + j] = std::pow(data1[i * cols1 + j], data2[i * cols2 + j]);
							}
						}
					}
				}
			}
		}

		// root

		template<bool return_transposed = false>
		matrix<double, return_transposed> root(double num)
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			matrix<double, return_transposed, true> result(rows, cols);

			double* data_result = result._data;

			num = 1 / num;

			__m256d b = _mm256_set1_pd(num);

			if constexpr (this_transposed)
			{
				if constexpr (return_transposed)
				{
					if constexpr (this_contiguous)
					{
						size_t size = this->_size;

						size_t final_pos_size = this->final_pos_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);

							_mm256_store_pd(&data_result[i], _mm256_pow_pd(a, b));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = std::pow(data1[i], num);
						}
					}
					else
					{
						size_t final_pos_rows = this->final_pos_rows;
						size_t final_pos_cols = this->final_pos_cols;

						size_t rows1 = this->actual_rows;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

								_mm256_store_pd(&data_result[j * rows + i], _mm256_pow_pd(a, b));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < final_pos_cols; j += 4)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);

								__m256d pow = _mm256_pow_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(pow, 1);
								__m128d val2 = _mm256_castpd256_pd128(pow);

								_mm_store_sd(&data_result[j * cols + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(j + 1) * cols + i], val2);

								_mm_store_sd(&data_result[(j + 2) * cols + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(j + 3) * cols + i], val1);
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								data_result[j * cols + i] = std::pow(data1[j * rows1 + i], num);
							}
						}
					}
				}
				else
				{
					size_t final_pos_cols = this->final_pos_cols;
					size_t final_pos_rows = this->final_pos_rows;

					size_t rows1 = this->actual_rows;

					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
								data1[(j + 1) * rows1 + i],
								data1[(j + 2) * rows1 + i],
								data1[(j + 3) * rows1 + i]);

							_mm256_store_pd(&data_result[i * cols + j], _mm256_pow_pd(a, b));
						}
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

							__m256d pow = _mm256_pow_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(pow, 1);
							__m128d val2 = _mm256_castpd256_pd128(pow);

							_mm_store_sd(&data_result[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

							_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							data_result[i * cols + j] = std::pow(data1[j * rows1 + i], num);
						}
					}
				}
			}
			else
			{
				if constexpr (return_transposed)
				{
					size_t cols1 = this->actual_cols;

					size_t final_pos_rows = this->final_pos_rows;
					size_t final_pos_cols = this->final_pos_cols;

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						for (size_t j = 0; j < rows; j++)
						{
							__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
								data1[(i + 1) * cols1 + j],
								data1[(i + 2) * cols1 + j],
								data1[(i + 3) * cols1 + j]);

							_mm256_store_pd(&data_result[j * rows + i], _mm256_pow_pd(a, b));
						}
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

							__m256d pow = _mm256_pow_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(pow, 1);
							__m128d val2 = _mm256_castpd256_pd128(pow);

							_mm_store_sd(&data_result[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data_result[(j + 1) * rows + i], val2);

							_mm_store_sd(&data_result[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data_result[(j + 3) * rows + i], val1);
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							data_result[j * rows + i] = std::pow(data1[i * cols1 + j], num);
						}
					}
				}
				else
				{
					if constexpr (this_contiguous)
					{
						size_t final_pos_size = this->final_pos_size;
						size_t size = this->_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);

							_mm256_store_pd(&data_result[i], _mm256_pow_pd(a, b));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = std::pow(data1[i], num);
						}
					}
					else
					{
						size_t cols1 = this->actual_cols;

						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

								_mm256_store_pd(&data_result[i * cols + j], _mm256_pow_pd(a, b));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < final_pos_rows; i += 4)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);

								__m256d pow = _mm256_pow_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(pow, 1);
								__m128d val2 = _mm256_castpd256_pd128(pow);

								_mm_store_sd(&data_result[i * cols + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

								_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								data_result[i * cols + j] = std::pow(data1[i * cols1 + j], num);
							}
						}
					}
				}
			}
			return result;
		}

		template<bool return_transposed = false, bool other_transposed, bool other_contiguous, bool call_destructor>
		matrix<double, return_transposed> root(const matrix<double, other_transposed, other_contiguous, call_destructor>& other)
		{
#ifdef _DEBUG
			if (other._cols != this->_cols || other._rows != this->_rows) throw std::invalid_argument("The dimensions of both matrices must be the same");
#else
#endif

			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;
			double* data2 = other._data;

			matrix<double, return_transposed, true> result(rows, cols);

			double* data_result = result._data;

			__m256d one = _mm256_set1_pd(1.0);

			if constexpr (this_transposed)
			{
				if constexpr (other_transposed)
				{
					if constexpr (return_transposed)
					{
						if constexpr (this_contiguous && other_contiguous)
						{
							size_t size = this->_size;

							size_t final_pos_size = this->final_pos_size;

							for (size_t i = 0; i < final_pos_size; i += 4)
							{
								__m256d a = _mm256_load_pd(&data1[i]);
								__m256d b = _mm256_load_pd(&data2[i]);

								_mm256_store_pd(&data_result[i], _mm256_pow_pd(a, _mm256_div_pd(one, b)));
							}
							for (size_t i = final_pos_size; i < size; i++)
							{
								data_result[i] = std::pow(data1[i], 1.0 / data2[i]);
							}
						}
						else
						{
							size_t final_pos_rows = this->final_pos_rows;
							size_t final_pos_cols = this->final_pos_cols;

							size_t rows1 = this->actual_rows;
							size_t rows2 = other.actual_rows;

							for (size_t i = 0; i < final_pos_rows; i += 4)
							{
								for (size_t j = 0; j < cols; j++)
								{
									__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
									__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);

									_mm256_store_pd(&data_result[j * rows + i], _mm256_pow_pd(a, _mm256_div_pd(one, b)));
								}
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								for (size_t j = 0; j < final_pos_cols; j += 4)
								{
									__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
										data1[(j + 1) * rows1 + i],
										data1[(j + 2) * rows1 + i],
										data1[(j + 3) * rows1 + i]);
									__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
										data2[(j + 1) * rows2 + i],
										data2[(j + 2) * rows2 + i],
										data2[(j + 3) * rows2 + i]);

									__m256d pow = _mm256_pow_pd(a, _mm256_div_pd(one, b));

									__m128d val1 = _mm256_extractf128_pd(pow, 1);
									__m128d val2 = _mm256_castpd256_pd128(pow);

									_mm_store_sd(&data_result[j * cols + i], val2);
									val2 = _mm_shuffle_pd(val2, val2, 1);
									_mm_store_sd(&data_result[(j + 1) * cols + i], val2);

									_mm_store_sd(&data_result[(j + 2) * cols + i], val1);
									val1 = _mm_shuffle_pd(val1, val1, 1);
									_mm_store_sd(&data_result[(j + 3) * cols + i], val1);
								}
								for (size_t j = final_pos_cols; j < cols; j++)
								{
									data_result[j * cols + i] = std::pow(data1[j * rows1 + i], 1.0 / data2[j * rows2 + i]);
								}
							}
						}
					}
					else
					{
						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						size_t rows1 = this->actual_rows;
						size_t rows2 = other.actual_rows;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);
								__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
									data2[(j + 1) * rows2 + i],
									data2[(j + 2) * rows2 + i],
									data2[(j + 3) * rows2 + i]);

								_mm256_store_pd(&data_result[i * cols + j], _mm256_pow_pd(a, _mm256_div_pd(one, b)));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < final_pos_rows; i += 4)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
								__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);

								__m256d pow = _mm256_pow_pd(a, _mm256_div_pd(one, b));

								__m128d val1 = _mm256_extractf128_pd(pow, 1);
								__m128d val2 = _mm256_castpd256_pd128(pow);

								_mm_store_sd(&data_result[i * cols + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

								_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								data_result[i * cols + j] = std::pow(data1[j * rows1 + i], 1.0 / data2[j * rows2 + i]);
							}
						}
					}
				}
				else
				{
					if constexpr (return_transposed)
					{
						size_t final_pos_rows = this->final_pos_rows;
						size_t final_pos_cols = this->final_pos_cols;

						size_t rows1 = this->actual_rows;
						size_t cols2 = other.actual_cols;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
								__m256d b = _mm256_setr_pd(data2[i * cols2 + j],
									data2[(i + 1) * cols2 + j],
									data2[(i + 2) * cols2 + j],
									data2[(i + 3) * cols2 + j]);

								_mm256_store_pd(&data_result[j * rows + i], _mm256_pow_pd(a, _mm256_div_pd(one, b)));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < final_pos_cols; j += 4)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);
								__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

								__m256d pow = _mm256_pow_pd(a, _mm256_div_pd(one, b));

								__m128d val1 = _mm256_extractf128_pd(pow, 1);
								__m128d val2 = _mm256_castpd256_pd128(pow);

								_mm_store_sd(&data_result[j * cols + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(j + 1) * cols + i], val2);

								_mm_store_sd(&data_result[(j + 2) * cols + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(j + 3) * cols + i], val1);
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								data_result[j * cols + i] = std::pow(data1[j * rows1 + i], 1.0 / data2[i * cols2 + j]);
							}
						}
					}
					else
					{
						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						size_t rows1 = this->actual_rows;
						size_t cols2 = other.actual_cols;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);
								__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

								_mm256_store_pd(&data_result[i * cols + j], _mm256_pow_pd(a, _mm256_div_pd(one, b)));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < final_pos_rows; i += 4)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
								__m256d b = _mm256_setr_pd(data1[i * cols2 + j],
									data1[(i + 1) * cols2 + j],
									data1[(i + 2) * cols2 + j],
									data1[(i + 3) * cols2 + j]);

								__m256d pow = _mm256_pow_pd(a, _mm256_div_pd(one, b));

								__m128d val1 = _mm256_extractf128_pd(pow, 1);
								__m128d val2 = _mm256_castpd256_pd128(pow);

								_mm_store_sd(&data_result[i * cols + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

								_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								data_result[i * cols + j] = std::pow(data1[j * rows1 + i], 1.0 / data2[i * cols2 + j]);
							}
						}
					}
				}
			}
			else
			{
				if constexpr (other_transposed)
				{
					size_t cols1 = this->actual_cols;
					size_t rows2 = other.actual_rows;

					if constexpr (return_transposed)
					{
						size_t final_pos_rows = this->final_pos_rows;
						size_t final_pos_cols = this->final_pos_cols;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);
								__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);
								_mm256_store_pd(&data_result[j * rows + i], _mm256_pow_pd(a, _mm256_div_pd(one, b)));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; final_pos_cols; j += 4)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
								__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
									data2[(j + 1) * rows2 + i],
									data2[(j + 2) * rows2 + i],
									data2[(j + 3) * rows2 + i]);

								__m256d pow = _mm256_pow_pd(a, _mm256_div_pd(one, b));

								__m128d val1 = _mm256_extractf128_pd(pow, 1);
								__m128d val2 = _mm256_castpd256_pd128(pow);

								_mm_store_sd(&data_result[j * rows + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(j + 1) * rows + i], val2);

								_mm_store_sd(&data_result[(j + 2) * rows + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(j + 3) * rows + i], val1);
							}
							for (size_t j = final_pos_cols; cols; j++)
							{
								data_result[j * rows + i] = std::pow(data1[i * cols1 + j], 1.0 / data2[j * rows2 + i]);
							}
						}
					}
					else
					{
						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
								__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
									data2[(j + 1) * rows2 + i],
									data2[(j + 2) * rows2 + i],
									data2[(j + 3) * rows2 + i]);
								_mm256_store_pd(&data_result[i * cols + j], _mm256_pow_pd(a, _mm256_div_pd(one, b)));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < final_pos_rows; i += 4)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);
								__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);

								__m256d pow = _mm256_pow_pd(a, _mm256_div_pd(one, b));

								__m128d val1 = _mm256_extractf128_pd(pow, 1);
								__m128d val2 = _mm256_castpd256_pd128(pow);

								_mm_store_sd(&data_result[i * cols + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(i + 1) * rows + j], val2);

								_mm_store_sd(&data_result[(i + 2) * rows + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(i + 3) * rows + j], val1);
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								data_result[i * cols + j] = std::pow(data1[i * cols1 + j], 1.0 / data2[j * rows2 + i]);
							}
						}
					}
				}
				else
				{
					if constexpr (return_transposed)
					{
						size_t cols1 = this->actual_cols;
						size_t cols2 = other.actual_cols;

						size_t final_pos_rows = this->final_pos_rows;
						size_t final_pos_cols = this->final_pos_cols;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < rows; j++)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);
								__m256d b = _mm256_setr_pd(data2[i * cols2 + j],
									data2[(i + 1) * cols2 + j],
									data2[(i + 2) * cols2 + j],
									data2[(i + 3) * cols2 + j]);
								_mm256_store_pd(&data_result[j * rows + i], _mm256_pow_pd(a, _mm256_div_pd(one, b)));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < final_pos_cols; j += 4)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
								__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

								__m256d pow = _mm256_pow_pd(a, _mm256_div_pd(one, b));

								__m128d val1 = _mm256_extractf128_pd(pow, 1);
								__m128d val2 = _mm256_castpd256_pd128(pow);

								_mm_store_sd(&data_result[j * rows + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data_result[(j + 1) * rows + i], val2);

								_mm_store_sd(&data_result[(j + 2) * rows + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data_result[(j + 3) * rows + i], val1);
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								data_result[j * rows + i] = std::pow(data1[i * cols1 + j], 1.0 / data2[i * cols2 + j]);
							}
						}
					}
					else
					{
						if constexpr (this_contiguous && other_contiguous)
						{
							size_t final_pos_size = this->final_pos_size;
							size_t size = this->_size;

							for (size_t i = 0; i < final_pos_size; i += 4)
							{
								__m256d a = _mm256_load_pd(&data1[i]);
								__m256d b = _mm256_load_pd(&data2[i]);

								_mm256_store_pd(&data_result[i], _mm256_pow_pd(a, _mm256_div_pd(one, b)));
							}
							for (size_t i = final_pos_size; i < size; i++)
							{
								data_result[i] = std::pow(data1[i], 1 / data2[i]);
							}
						}
						else
						{
							size_t cols1 = this->actual_cols;
							size_t cols2 = other.actual_cols;

							size_t final_pos_cols = this->final_pos_cols;
							size_t final_pos_rows = this->final_pos_rows;

							for (size_t j = 0; j < final_pos_cols; j += 4)
							{
								for (size_t i = 0; i < rows; i++)
								{
									__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
									__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

									_mm256_store_pd(&data_result[i * cols + j], _mm256_pow_pd(a, _mm256_div_pd(one, b)));
								}
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								for (size_t i = 0; i < final_pos_rows; i += 4)
								{
									__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
										data1[(i + 1) * cols1 + j],
										data1[(i + 2) * cols1 + j],
										data1[(i + 3) * cols1 + j]);
									__m256d b = _mm256_setr_pd(data2[i * cols2 + j],
										data2[(i + 1) * cols2 + j],
										data2[(i + 2) * cols2 + j],
										data2[(i + 3) * cols2 + j]);

									__m256d pow = _mm256_pow_pd(a, _mm256_div_pd(one, b));

									__m128d val1 = _mm256_extractf128_pd(pow, 1);
									__m128d val2 = _mm256_castpd256_pd128(pow);

									_mm_store_sd(&data_result[i * cols + j], val2);
									val2 = _mm_shuffle_pd(val2, val2, 1);
									_mm_store_sd(&data_result[(i + 1) * cols + j], val2);

									_mm_store_sd(&data_result[(i + 2) * cols + j], val1);
									val1 = _mm_shuffle_pd(val1, val1, 1);
									_mm_store_sd(&data_result[(i + 3) * cols + j], val1);
								}
								for (size_t i = final_pos_rows; i < rows; i++)
								{
									data_result[i * cols + j] = std::pow(data1[i * cols1 + j], 1.0 / data2[i * cols2 + j]);
								}
							}
						}
					}
				}
			}
			return result;
		}

		void self_root(double num)
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			num = 1.0 / num;

			__m256d b = _mm256_set1_pd(num);

			if constexpr (this_transposed)
			{
				if constexpr (this_contiguous)
				{
					size_t size = this->_size;

					size_t final_pos_size = this->final_pos_size;

					for (size_t i = 0; i < final_pos_size; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&data1[i], _mm256_pow_pd(a, b));
					}
					for (size_t i = final_pos_size; i < size; i++)
					{
						data1[i] = std::pow(data1[i], num);
					}
				}
				else
				{
					size_t final_pos_rows = this->final_pos_rows;
					size_t final_pos_cols = this->final_pos_cols;

					size_t rows1 = this->actual_rows;

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

							_mm256_store_pd(&data1[j * rows1 + i], _mm256_pow_pd(a, b));
						}
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
								data1[(j + 1) * rows1 + i],
								data1[(j + 2) * rows1 + i],
								data1[(j + 3) * rows1 + i]);

							__m256d pow = _mm256_pow_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(pow, 1);
							__m128d val2 = _mm256_castpd256_pd128(pow);

							_mm_store_sd(&data1[j * rows1 + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(j + 1) * rows1 + i], val2);

							_mm_store_sd(&data1[(j + 2) * rows1 + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(j + 3) * rows1 + i], val1);
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							data1[j * rows1 + i] = std::pow(data1[j * rows1 + i], num);
						}
					}
				}
			}
			else
			{
				if constexpr (this_contiguous)
				{
					size_t final_pos_size = this->final_pos_size;
					size_t size = this->_size;

					for (size_t i = 0; i < final_pos_size; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&data1[i], _mm256_pow_pd(a, b));
					}
					for (size_t i = final_pos_size; i < size; i++)
					{
						data1[i] = std::pow(data1[i], num);
					}
				}
				else
				{
					size_t cols1 = this->actual_cols;

					size_t final_pos_cols = this->final_pos_cols;
					size_t final_pos_rows = this->final_pos_rows;

					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

							_mm256_store_pd(&data1[i * cols1 + j], _mm256_pow_pd(a, b));
						}
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
								data1[(i + 1) * cols1 + j],
								data1[(i + 2) * cols1 + j],
								data1[(i + 3) * cols1 + j]);

							__m256d pow = _mm256_pow_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(pow, 1);
							__m128d val2 = _mm256_castpd256_pd128(pow);

							_mm_store_sd(&data1[i * cols1 + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(i + 1) * cols1 + j], val2);

							_mm_store_sd(&data1[(i + 2) * cols1 + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(i + 3) * cols1 + j], val1);
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							data1[i * cols1 + j] = std::pow(data1[i * cols1 + j], num);
						}
					}
				}
			}
		}

		template<bool other_transposed, bool other_contiguous, bool call_destructor>
		void self_root(const matrix<double, other_transposed, other_contiguous, call_destructor>& other)
		{
#ifdef _DEBUG
			if (other._cols != this->_cols || other._rows != this->_rows) throw std::invalid_argument("The dimensions of both matrices must be the same");
#else
#endif

			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;
			double* data2 = other._data;

			__m256d one = _mm256_set1_pd(1.0);

			if constexpr (this_transposed)
			{
				if constexpr (other_transposed)
				{
					if constexpr (this_contiguous && other_contiguous)
					{
						size_t size = this->_size;

						size_t final_pos_size = this->final_pos_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);
							__m256d b = _mm256_load_pd(&data2[i]);

							_mm256_store_pd(&data1[i], _mm256_pow_pd(a, _mm256_div_pd(one, b)));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data1[i] = std::pow(data1[i], 1 / data2[i]);
						}
					}
					else
					{
						size_t final_pos_rows = this->final_pos_rows;
						size_t final_pos_cols = this->final_pos_cols;

						size_t rows1 = this->actual_rows;
						size_t rows2 = other.actual_rows;

						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
								__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);

								_mm256_store_pd(&data1[j * rows1 + i], _mm256_pow_pd(a, _mm256_div_pd(one, b)));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < final_pos_cols; j += 4)
							{
								__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
									data1[(j + 1) * rows1 + i],
									data1[(j + 2) * rows1 + i],
									data1[(j + 3) * rows1 + i]);
								__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
									data2[(j + 1) * rows2 + i],
									data2[(j + 2) * rows2 + i],
									data2[(j + 3) * rows2 + i]);

								__m256d pow = _mm256_pow_pd(a, _mm256_div_pd(one, b));

								__m128d val1 = _mm256_extractf128_pd(pow, 1);
								__m128d val2 = _mm256_castpd256_pd128(pow);

								_mm_store_sd(&data1[j * rows1 + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data1[(j + 1) * rows1 + i], val2);

								_mm_store_sd(&data1[(j + 2) * rows1 + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data1[(j + 3) * rows1 + i], val1);
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								data1[j * rows1 + i] = std::pow(data1[j * rows1 + i], 1.0 / data2[j * rows2 + i]);
							}
						}
					}
				}
				else
				{
					size_t final_pos_rows = this->final_pos_rows;
					size_t final_pos_cols = this->final_pos_cols;

					size_t rows1 = this->actual_rows;
					size_t cols2 = other.actual_cols;

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);
							__m256d b = _mm256_setr_pd(data2[i * cols2 + j],
								data2[(i + 1) * cols2 + j],
								data2[(i + 2) * cols2 + j],
								data2[(i + 3) * cols2 + j]);

							_mm256_store_pd(&data1[j * rows1 + i], _mm256_pow_pd(a, _mm256_div_pd(one, b)));
						}
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
								data1[(j + 1) * rows1 + i],
								data1[(j + 2) * rows1 + i],
								data1[(j + 3) * rows1 + i]);
							__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

							__m256d pow = _mm256_pow_pd(a, _mm256_div_pd(one, b));

							__m128d val1 = _mm256_extractf128_pd(pow, 1);
							__m128d val2 = _mm256_castpd256_pd128(pow);

							_mm_store_sd(&data1[j * rows1 + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(j + 1) * rows1 + i], val2);

							_mm_store_sd(&data1[(j + 2) * rows1 + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(j + 3) * rows1 + i], val1);
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							data1[j * rows1 + i] = std::pow(data1[j * rows1 + i], 1.0 / data2[i * cols2 + j]);
						}
					}
			}
		}
			else
			{
				if constexpr (other_transposed)
				{
					size_t cols1 = this->actual_cols;
					size_t rows2 = other.actual_rows;

					size_t final_pos_cols = this->final_pos_cols;
					size_t final_pos_rows = this->final_pos_rows;

					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
							__m256d b = _mm256_setr_pd(data2[j * rows2 + i],
								data2[(j + 1) * rows2 + i],
								data2[(j + 2) * rows2 + i],
								data2[(j + 3) * rows2 + i]);
							_mm256_store_pd(&data1[i * cols1 + j], _mm256_pow_pd(a, _mm256_div_pd(one, b)));
						}
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						for (size_t i = 0; i < final_pos_rows; i += 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
								data1[(i + 1) * cols1 + j],
								data1[(i + 2) * cols1 + j],
								data1[(i + 3) * cols1 + j]);
							__m256d b = _mm256_load_pd(&data2[j * rows2 + i]);

							__m256d pow = _mm256_pow_pd(a, _mm256_div_pd(one, b));

							__m128d val1 = _mm256_extractf128_pd(pow, 1);
							__m128d val2 = _mm256_castpd256_pd128(pow);

							_mm_store_sd(&data1[i * cols1 + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(i + 1) * cols1 + j], val2);

							_mm_store_sd(&data1[(i + 2) * cols1 + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(i + 3) * cols1 + j], val1);
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							data1[i * cols1 + j] = std::pow(data1[i * cols1 + j], 1.0 / data2[j * rows2 + i]);
						}
					}
				}
				else
				{
					if constexpr (this_contiguous && other_contiguous)
					{
						size_t final_pos_size = this->final_pos_size;
						size_t size = this->_size;

						for (size_t i = 0; i < final_pos_size; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);
							__m256d b = _mm256_load_pd(&data2[i]);

							_mm256_store_pd(&data1[i], _mm256_pow_pd(a, _mm256_div_pd(one, b)));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data1[i] = std::pow(data1[i], 1 / data2[i]);
						}
					}
					else
					{
						size_t cols1 = this->actual_cols;
						size_t cols2 = other.actual_cols;

						size_t final_pos_cols = this->final_pos_cols;
						size_t final_pos_rows = this->final_pos_rows;

						for (size_t j = 0; j < final_pos_cols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
								__m256d b = _mm256_load_pd(&data2[i * cols2 + j]);

								_mm256_store_pd(&data1[i * cols1 + j], _mm256_pow_pd(a, _mm256_div_pd(one, b)));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < final_pos_rows; i += 4)
							{
								__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
									data1[(i + 1) * cols1 + j],
									data1[(i + 2) * cols1 + j],
									data1[(i + 3) * cols1 + j]);
								__m256d b = _mm256_setr_pd(data2[i * cols2 + j],
									data2[(i + 1) * cols2 + j],
									data2[(i + 2) * cols2 + j],
									data2[(i + 3) * cols2 + j]);

								__m256d pow = _mm256_pow_pd(a, _mm256_div_pd(one, b));

								__m128d val1 = _mm256_extractf128_pd(pow, 1);
								__m128d val2 = _mm256_castpd256_pd128(pow);

								_mm_store_sd(&data1[i * cols1 + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&data1[(i + 1) * cols1 + j], val2);

								_mm_store_sd(&data1[(i + 2) * cols1 + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&data1[(i + 3) * cols1 + j], val1);
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								data1[i * cols1 + j] = std::pow(data1[i * cols1 + j], 1.0 / data2[i * cols2 + j]);
							}
						}
					}
				}
			}
		}

		// Mean 

		vector<double> mean_rowwise()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			vector<double> result(rows);

			double* data_result = result._data;

			double cols_d = static_cast<double>(cols);

			__m256d _cols = _mm256_set1_pd(cols_d);

			if constexpr (this_transposed)
			{
				size_t rows1 = this->actual_rows;

				size_t final_pos_rows = this->final_pos_rows;
				size_t final_pos_cols = this->final_pos_cols;

				for (size_t i = 0; i < final_pos_rows; i += 4)
				{
					__m256d _sum = _mm256_setzero_pd();
					for (size_t j = 0; j < cols; j++)
					{
						_sum = _mm256_add_pd(_sum, _mm256_load_pd(&data1[j * rows1 + i]));
					}
					_mm256_store_pd(&data_result[i], _mm256_div_pd(_sum, _cols));
				}
				for (size_t i = final_pos_rows; i < rows; i++)
				{
					__m256d _sum = _mm256_setzero_pd();
					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						_sum = _mm256_add_pd(_sum, _mm256_setr_pd(data1[j * rows1 + i], 
							data1[(j + 1) * rows1 + i], 
							data1[(j + 2) * rows1 + i], 
							data1[(j + 3) * rows1 + i]));
					}
					__m128d vlow = _mm256_castpd256_pd128(_sum);
					__m128d vhigh = _mm256_extractf128_pd(_sum, 1);
					vlow = _mm_add_pd(vlow, vhigh);

					__m128d high64 = _mm_unpackhi_pd(vlow, vlow);
					double sum = _mm_cvtsd_f64(_mm_add_sd(vlow, high64));
					
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						sum += data1[j * rows1 + i];
					}
					data_result[i] = sum / cols_d;
				}
			}
			else
			{
				size_t cols1 = this->actual_cols;

				size_t final_pos_rows = this->final_pos_rows;
				size_t final_pos_cols = this->final_pos_cols;

				for (size_t i = 0; i < final_pos_rows; i += 4)
				{
					__m256d _sum = _mm256_setzero_pd();
					for (size_t j = 0; j < cols; j++)
					{
						_sum = _mm256_add_pd(_sum, _mm256_setr_pd(data1[i * cols1 + j],
							data1[(i + 1) * cols1 + j],
							data1[(i + 2) * cols1 + j],
							data1[(i + 3) * cols1 + j]));
					}
					_mm256_store_pd(&data_result[i], _mm256_div_pd(_sum, _cols));
				}
				for (size_t i = final_pos_rows; i < rows; i++)
				{
					__m256d _sum = _mm256_setzero_pd();
					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						_sum = _mm256_add_pd(_sum, _mm256_load_pd(data1[i * cols1 + j]));
					}
					__m128d vlow = _mm256_castpd256_pd128(_sum);
					__m128d vhigh = _mm256_extractf128_pd(_sum, 1);
					vlow = _mm_add_pd(vlow, vhigh);

					__m128d high64 = _mm_unpackhi_pd(vlow, vlow);
					double sum = _mm_cvtsd_f64(_mm_add_sd(vlow, high64));

					for (size_t j = final_pos_cols; j < cols; j++)
					{
						sum += data1[i * cols1 + j];
					}
					data_result[i] = sum / cols_d;
				}
			}
			return result;
		}

		vector<double> mean_colwise()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			vector<double> result(cols);

			double* data_result = result._data;

			double rows_d = static_cast<double>(rows);

			__m256d _rows = _mm256_set1_pd(rows_d);

			if constexpr (this_transposed)
			{
				size_t rows1 = this->actual_rows;

				size_t final_pos_cols = this->final_pos_cols;
				size_t final_pos_rows = this->final_pos_rows;

				for (size_t j = 0; j < final_pos_cols; j += 4)
				{
					__m256d _sum = _mm256_setzero_pd();
					for (size_t i = 0; i < rows; i++)
					{
						_sum = _mm256_add_pd(_sum, _mm256_setr_pd(data1[j * rows1 + i],
							data1[(j + 1) * rows1 + i],
							data1[(j + 2) * rows1 + i],
							data1[(j + 3) * rows1 + i]));
					}
					_mm256_store_pd(&data_result[j], _mm256_div_pd(_sum, _rows));
				}
				for (size_t j = final_pos_cols; j < cols; j++)
				{
					__m256d _sum = _mm256_setzero_pd();

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						_sum = _mm256_add_pd(_sum, _mm256_load_pd(&data1[j * rows1 + i]));
					}
					__m128d vlow = _mm256_castpd256_pd128(_sum);
					__m128d vhigh = _mm256_extractf128_pd(_sum, 1);
					vlow = _mm_add_pd(vlow, vhigh);

					__m128d high64 = _mm_unpackhi_pd(vlow, vlow);
					double sum = _mm_cvtsd_f64(_mm_add_sd(vlow, high64));

					for (size_t i = final_pos_rows; i < rows; i++)
					{
						sum += data1[j * rows1 + i];
					}
					data_result[j] = sum / rows_d;
				}
			}
			else
			{
				size_t cols1 = this->actual_cols;

				size_t final_pos_cols = this->final_pos_cols;
				size_t final_pos_rows = this->final_pos_rows;

				for (size_t j = 0; j < final_pos_cols; j += 4)
				{
					__m256d _sum = _mm256_setzero_pd();
					for (size_t i = 0; i < rows; i++)
					{
						_sum = _mm256_add_pd(_sum, _mm256_load_pd(&data1[i * cols1 + j]));
					}
					_mm256_store_pd(&data_result[j], _mm256_div_pd(_sum, _rows));
				}
				for (size_t j = final_pos_cols; j < cols; j++)
				{
					__m256d _sum = _mm256_setzero_pd();

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						_sum = _mm256_add_pd(_sum, _mm256_setr_pd(data1[i * cols1 + j],
							data1[(i + 1) * cols1 + j],
							data1[(i + 2) * cols1 + j],
							data1[(i + 3) * cols1 + j]));
					}
					__m128d vlow = _mm256_castpd256_pd128(_sum);
					__m128d vhigh = _mm256_extractf128_pd(_sum, 1);
					vlow = _mm_add_pd(vlow, vhigh);

					__m128d high64 = _mm_unpackhi_pd(vlow, vlow);
					double sum = _mm_cvtsd_f64(_mm_add_sd(vlow, high64));

					for (size_t i = final_pos_rows; i < rows; i++)
					{
						sum += data1[i * cols1 + j];
					}
					data_result[j] = sum / rows_d;
				}
			}
			return result;
		}

		double mean_all()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			size_t size = this->_size;

			double* data1 = this->_data;

			__m256d _sum = _mm256_setzero_pd();
			double sum = 0;

			if constexpr (this_contiguous)
			{
				size_t final_pos_size = this->final_pos_size;

				for (size_t i = 0; i < final_pos_size; i += 4)
				{
					_sum = _mm256_add_pd(_sum, _mm256_load_pd(&data1[i]));
				}
				for (size_t i = final_pos_size; i < size; i++)
				{
					sum += data1[i];
				}
			}
			else if constexpr (this_transposed)
			{
				size_t rows1 = this->actual_rows;

				size_t final_pos_rows = this->final_pos_rows;
				size_t final_pos_cols = this->final_pos_cols;

				for (size_t i = 0; i < final_pos_rows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						_sum = _mm256_add_pd(_sum, _mm256_load_pd(&data1[j * rows1 + i]));
					}
				}
				for (size_t i = final_pos_rows; i < rows; i++)
				{
					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						_sum = _mm256_add_pd(_sum, _mm256_setr_pd(data1[j * rows1 + i], 
							data1[(j + 1) * rows1 + i], 
							data1[(j + 2) * rows1 + i],
							data1[(j + 3) * rows1 + i]));
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						sum += data1[j * rows1 + i];
					}
				}
			}
			else
			{
				size_t cols1 = this->actual_cols;

				size_t final_pos_cols = this->final_pos_cols;
				size_t final_pos_rows = this->final_pos_rows;

				for (size_t j = 0; j < final_pos_cols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						_sum = _mm256_add_pd(_sum, _mm256_load_pd(&data1[i * cols1 + j]));
					}
				}
				for (size_t j = final_pos_cols; j < cols; j++)
				{
					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						_sum = _mm256_add_pd(_sum, _mm256_setr_pd(data1[i * cols1 + j], 
							data1[(i + 1) * cols1 + j], 
							data1[(i + 2) * cols1 + j], 
							data1[(i + 3) * cols1 + j]));
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						sum += data1[i * cols1 + j];
					}
				}
			}

			__m128d vlow = _mm256_castpd256_pd128(_sum);
			__m128d vhigh = _mm256_extractf128_pd(_sum, 1);
			vlow = _mm_add_pd(vlow, vhigh);

			__m128d high64 = _mm_unpackhi_pd(vlow, vlow);
			sum += _mm_cvtsd_f64(_mm_add_sd(vlow, high64));

			return sum / static_cast<double>(size);
		}

		// Sum

		vector<double> sum_rowwise()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			vector<double> result(rows);

			double* data_result = result._data;

			if constexpr (this_transposed)
			{
				size_t rows1 = this->actual_rows;

				size_t final_pos_rows = this->final_pos_rows;
				size_t final_pos_cols = this->final_pos_cols;

				for (size_t i = 0; i < final_pos_rows; i += 4)
				{
					__m256d _sum = _mm256_setzero_pd();
					for (size_t j = 0; j < cols; j++)
					{
						_sum = _mm256_add_pd(_sum, _mm256_load_pd(&data1[j * rows1 + i]));
					}
					_mm256_store_pd(&data_result[i], _sum);
				}
				for (size_t i = final_pos_rows; i < rows; i++)
				{
					__m256d _sum = _mm256_setzero_pd();
					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						_sum = _mm256_add_pd(_sum, _mm256_setr_pd(data1[j * rows1 + i],
							data1[(j + 1) * rows1 + i],
							data1[(j + 2) * rows1 + i],
							data1[(j + 3) * rows1 + i]));
					}
					__m128d vlow = _mm256_castpd256_pd128(_sum);
					__m128d vhigh = _mm256_extractf128_pd(_sum, 1);
					vlow = _mm_add_pd(vlow, vhigh);

					__m128d high64 = _mm_unpackhi_pd(vlow, vlow);
					double sum = _mm_cvtsd_f64(_mm_add_sd(vlow, high64));

					for (size_t j = final_pos_cols; j < cols; j++)
					{
						sum += data1[j * rows1 + i];
					}
					data_result[i] = sum;
				}
			}
			else
			{
				size_t cols1 = this->actual_cols;

				size_t final_pos_rows = this->final_pos_rows;
				size_t final_pos_cols = this->final_pos_cols;

				for (size_t i = 0; i < final_pos_rows; i += 4)
				{
					__m256d _sum = _mm256_setzero_pd();
					for (size_t j = 0; j < cols; j++)
					{
						_sum = _mm256_add_pd(_sum, _mm256_setr_pd(data1[i * cols1 + j],
							data1[(i + 1) * cols1 + j],
							data1[(i + 2) * cols1 + j],
							data1[(i + 3) * cols1 + j]));
					}
					_mm256_store_pd(&data_result[i], _sum);
				}
				for (size_t i = final_pos_rows; i < rows; i++)
				{
					__m256d _sum = _mm256_setzero_pd();
					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						_sum = _mm256_add_pd(_sum, _mm256_load_pd(data1[i * cols1 + j]));
					}
					__m128d vlow = _mm256_castpd256_pd128(_sum);
					__m128d vhigh = _mm256_extractf128_pd(_sum, 1);
					vlow = _mm_add_pd(vlow, vhigh);

					__m128d high64 = _mm_unpackhi_pd(vlow, vlow);
					double sum = _mm_cvtsd_f64(_mm_add_sd(vlow, high64));

					for (size_t j = final_pos_cols; j < cols; j++)
					{
						sum += data1[i * cols1 + j];
					}
					data_result[i] = sum;
				}
			}
			return result;
		}

		vector<double> sum_colwise()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			vector<double> result(cols);

			double* data_result = result._data;

			if constexpr (this_transposed)
			{
				size_t rows1 = this->actual_rows;

				size_t final_pos_cols = this->final_pos_cols;
				size_t final_pos_rows = this->final_pos_rows;

				for (size_t j = 0; j < final_pos_cols; j += 4)
				{
					__m256d _sum = _mm256_setzero_pd();
					for (size_t i = 0; i < rows; i++)
					{
						_sum = _mm256_add_pd(_sum, _mm256_setr_pd(data1[j * rows1 + i],
							data1[(j + 1) * rows1 + i],
							data1[(j + 2) * rows1 + i],
							data1[(j + 3) * rows1 + i]));
					}
					_mm256_store_pd(&data_result[j], _sum);
				}
				for (size_t j = final_pos_cols; j < cols; j++)
				{
					__m256d _sum = _mm256_setzero_pd();

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						_sum = _mm256_add_pd(_sum, _mm256_load_pd(&data1[j * rows1 + i]));
					}
					__m128d vlow = _mm256_castpd256_pd128(_sum);
					__m128d vhigh = _mm256_extractf128_pd(_sum, 1);
					vlow = _mm_add_pd(vlow, vhigh);

					__m128d high64 = _mm_unpackhi_pd(vlow, vlow);
					double sum = _mm_cvtsd_f64(_mm_add_sd(vlow, high64));

					for (size_t i = final_pos_rows; i < rows; i++)
					{
						sum += data1[j * rows1 + i];
					}
					data_result[j] = sum;
				}
			}
			else
			{
				size_t cols1 = this->actual_cols;

				size_t final_pos_cols = this->final_pos_cols;
				size_t final_pos_rows = this->final_pos_rows;

				for (size_t j = 0; j < final_pos_cols; j += 4)
				{
					__m256d _sum = _mm256_setzero_pd();
					for (size_t i = 0; i < rows; i++)
					{
						_sum = _mm256_add_pd(_sum, _mm256_load_pd(&data1[i * cols1 + j]));
					}
					_mm256_store_pd(&data_result[j], _sum);
				}
				for (size_t j = final_pos_cols; j < cols; j++)
				{
					__m256d _sum = _mm256_setzero_pd();

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						_sum = _mm256_add_pd(_sum, _mm256_setr_pd(data1[i * cols1 + j],
							data1[(i + 1) * cols1 + j],
							data1[(i + 2) * cols1 + j],
							data1[(i + 3) * cols1 + j]));
					}
					__m128d vlow = _mm256_castpd256_pd128(_sum);
					__m128d vhigh = _mm256_extractf128_pd(_sum, 1);
					vlow = _mm_add_pd(vlow, vhigh);

					__m128d high64 = _mm_unpackhi_pd(vlow, vlow);
					double sum = _mm_cvtsd_f64(_mm_add_sd(vlow, high64));

					for (size_t i = final_pos_rows; i < rows; i++)
					{
						sum += data1[i * cols1 + j];
					}
					data_result[j] = sum;
				}
			}
			return result;
		}

		double sum_all()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			__m256d _sum = _mm256_setzero_pd();
			double sum = 0;

			if constexpr (this_contiguous)
			{
				size_t size = this->_size;
				size_t final_pos_size = this->final_pos_size;

				for (size_t i = 0; i < final_pos_size; i += 4)
				{
					_sum = _mm256_add_pd(_sum, _mm256_load_pd(&data1[i]));
				}
				for (size_t i = final_pos_size; i < size; i++)
				{
					sum += data1[i];
				}
			}
			else if constexpr (this_transposed)
			{
				size_t rows1 = this->actual_rows;

				size_t final_pos_rows = this->final_pos_rows;
				size_t final_pos_cols = this->final_pos_cols;

				for (size_t i = 0; i < final_pos_rows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						_sum = _mm256_add_pd(_sum, _mm256_load_pd(&data1[j * rows1 + i]));
					}
				}
				for (size_t i = final_pos_rows; i < rows; i++)
				{
					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						_sum = _mm256_add_pd(_sum, _mm256_setr_pd(data1[j * rows1 + i],
							data1[(j + 1) * rows1 + i],
							data1[(j + 2) * rows1 + i],
							data1[(j + 3) * rows1 + i]));
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						sum += data1[j * rows1 + i];
					}
				}
			}
			else
			{
				size_t cols1 = this->actual_cols;

				size_t final_pos_cols = this->final_pos_cols;
				size_t final_pos_rows = this->final_pos_rows;

				for (size_t j = 0; j < final_pos_cols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						_sum = _mm256_add_pd(_sum, _mm256_load_pd(&data1[i * cols1 + j]));
					}
				}
				for (size_t j = final_pos_cols; j < cols; j++)
				{
					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						_sum = _mm256_add_pd(_sum, _mm256_setr_pd(data1[i * cols1 + j],
							data1[(i + 1) * cols1 + j],
							data1[(i + 2) * cols1 + j],
							data1[(i + 3) * cols1 + j]));
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						sum += data1[i * cols1 + j];
					}
				}
			}

			__m128d vlow = _mm256_castpd256_pd128(_sum);
			__m128d vhigh = _mm256_extractf128_pd(_sum, 1);
			vlow = _mm_add_pd(vlow, vhigh);

			__m128d high64 = _mm_unpackhi_pd(vlow, vlow);
			sum += _mm_cvtsd_f64(_mm_add_sd(vlow, high64));

			return sum;
		}

		// Std

		vector<double> std_rowwise(double ddof = 0.0)
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			vector<double> result(rows);

			double* data_result = result._data;

			double cols_d = static_cast<double>(cols);

			__m256d _cols = _mm256_set1_pd(cols_d);
			__m256d _ddof = _mm256_set1_pd(ddof);

			if constexpr (this_transposed)
			{
				size_t rows1 = this->actual_rows;
				
				size_t final_pos_rows = this->final_pos_rows;
				size_t final_pos_cols = this->final_pos_cols;

				for (size_t i = 0; i < final_pos_rows; i += 4)
				{
					__m256d _sum = _mm256_setzero_pd();
					__m256d _sumSquare = _mm256_setzero_pd();
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

						_sum = _mm256_add_pd(_sum, a);
						_sumSquare = _mm256_add_pd(_sumSquare, _mm256_mul_pd(a, a));
					}
					__m256d variance = _mm256_div_pd(_mm256_sub_pd(_sumSquare, _mm256_div_pd(_mm256_mul_pd(_sum, _sum), _cols)), _mm256_sub_pd(_cols, _ddof));
					_mm256_store_pd(&data_result[i], _mm256_sqrt_pd(variance));
				}
				for (size_t i = final_pos_rows; i < rows; i++)
				{
					__m256d _sum = _mm256_setzero_pd();
					__m256d _sumSquare = _mm256_setzero_pd();
					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						__m256d a = _mm256_setr_pd(data1[j * rows1 + i], 
							data1[(j + 1) * rows1 + i], 
							data1[(j + 2) * rows1 + i],
							data1[(j + 3) * rows1 + i]);
						_sum = _mm256_add_pd(_sum, a);
						_sumSquare = _mm256_add_pd(_sumSquare, _mm256_mul_pd(a, a));
					}

					__m128d vlow = _mm256_castpd256_pd128(_sum);
					__m128d vhigh = _mm256_extractf128_pd(_sum, 1);
					vlow = _mm_add_pd(vlow, vhigh);

					__m128d high64 = _mm_unpackhi_pd(vlow, vlow);
					double sum = _mm_cvtsd_f64(_mm_add_sd(vlow, high64));
					//--
					__m128d vlow1 = _mm256_castpd256_pd128(_sumSquare);
					vhigh = _mm256_extractf128_pd(_sumSquare, 1);
					vlow1 = _mm_add_pd(vlow1, vhigh);

					high64 = _mm_unpackhi_pd(vlow1, vlow1);
					double sumSquare = _mm_cvtsd_f64(_mm_add_sd(vlow1, high64));

					for (size_t j = 0; j < cols; j++)
					{
						double data = data1[j * rows1 + i];
						sum += data;
						sumSquare += data * data;
					}
					double variance = (sumSquare - (sum * sum / cols_d)) / (cols_d - ddof);
					double std = std::sqrt(variance);
					data_result[i] = std;
				}
			}
			else
			{
				size_t cols1 = this->actual_cols;
				
				size_t final_pos_rows = this->final_pos_rows;
				size_t final_pos_cols = this->final_pos_cols;

				for (size_t i = 0; i < final_pos_rows; i += 4)
				{
					__m256d _sum = _mm256_setzero_pd();
					__m256d _sumSquare = _mm256_setzero_pd();
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_setr_pd(data1[i * cols1 + j], 
							data1[(i + 1) * cols1 + j], 
							data1[(i + 2) * cols1 + j],
							data1[(i + 3) * cols1 + j]);

						_sum = _mm256_add_pd(_sum, a);
						_sumSquare = _mm256_add_pd(_sumSquare, _mm256_mul_pd(a, a));
					}
					__m256d variance = _mm256_div_pd(_mm256_sub_pd(_sumSquare, _mm256_div_pd(_mm256_mul_pd(_sum, _sum), _cols)), _mm256_sub_pd(_cols, _ddof));
					_mm256_store_pd(&data_result[i], _mm256_sqrt_pd(variance));
				}
				for (size_t i = final_pos_rows; i < rows; i++)
				{
					__m256d _sum = _mm256_setzero_pd();
					__m256d _sumSquare = _mm256_setzero_pd();
					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
						_sum = _mm256_add_pd(_sum, a);
						_sumSquare = _mm256_add_pd(_sumSquare, _mm256_mul_pd(a, a));
					}

					__m128d vlow = _mm256_castpd256_pd128(_sum);
					__m128d vhigh = _mm256_extractf128_pd(_sum, 1);
					vlow = _mm_add_pd(vlow, vhigh);

					__m128d high64 = _mm_unpackhi_pd(vlow, vlow);
					double sum = _mm_cvtsd_f64(_mm_add_sd(vlow, high64));
					//--
					__m128d vlow1 = _mm256_castpd256_pd128(_sumSquare);
					vhigh = _mm256_extractf128_pd(_sumSquare, 1);
					vlow1 = _mm_add_pd(vlow1, vhigh);

					high64 = _mm_unpackhi_pd(vlow1, vlow1);
					double sumSquare = _mm_cvtsd_f64(_mm_add_sd(vlow1, high64));

					for (size_t j = 0; j < cols; j++)
					{
						double data = data1[i * cols1 + j];
						sum += data;
						sumSquare += data * data;
					}
					double variance = (sumSquare - (sum * sum / cols_d)) / (cols_d - ddof);
					double std = std::sqrt(variance);
					data_result[i] = std;
				}
			}
			return result;
		}

		vector<double> std_colwise(double ddof = 0.0)
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			vector<double> result(cols);

			double* data_result = result._data;

			double rows_d = static_cast<double>(rows);

			__m256d _rows = _mm256_set1_pd(rows_d);
			__m256d _ddof = _mm256_set1_pd(ddof);

			if constexpr (this_transposed)
			{
				size_t rows1 = this->actual_rows;

				size_t final_pos_cols = this->final_pos_cols;
				size_t final_pos_rows = this->final_pos_rows;

				for (size_t j = 0; j < final_pos_cols; j += 4)
				{
					__m256d _sum = _mm256_setzero_pd();
					__m256d _sumSquare = _mm256_setzero_pd();
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
							data1[(j + 1) * rows1 + i],
							data1[(j + 2) * rows1 + i],
							data1[(j + 3) * rows1 + i]);

						_sum = _mm256_add_pd(_sum, a);
						_sumSquare = _mm256_add_pd(_sumSquare, _mm256_mul_pd(a, a));
					}
					__m256d variance = _mm256_div_pd(_mm256_sub_pd(_sumSquare, _mm256_div_pd(_mm256_mul_pd(_sum, _sum), _rows)), _mm256_sub_pd(_rows, _ddof));
					_mm256_store_pd(&data_result[j], _mm256_sqrt_pd(variance));
				}
				for (size_t j = final_pos_cols; j < cols; j++)
				{
					__m256d _sum = _mm256_setzero_pd();
					__m256d _sumSquare = _mm256_setzero_pd();

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

						_sum = _mm256_add_pd(_sum, a);
						_sumSquare = _mm256_add_pd(_sumSquare, _mm256_mul_pd(a, a));
					}
					__m128d vlow = _mm256_castpd256_pd128(_sum);
					__m128d vhigh = _mm256_extractf128_pd(_sum, 1);
					vlow = _mm_add_pd(vlow, vhigh);

					__m128d high64 = _mm_unpackhi_pd(vlow, vlow);
					double sum = _mm_cvtsd_f64(_mm_add_sd(vlow, high64));
					//--
					__m128d vlow1 = _mm256_castpd256_pd128(_sumSquare);
					vhigh = _mm256_extractf128_pd(_sumSquare, 1);
					vlow1 = _mm_add_pd(vlow1, vhigh);

					high64 = _mm_unpackhi_pd(vlow1, vlow1);
					double sumSquare = _mm_cvtsd_f64(_mm_add_sd(vlow1, high64));
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						double data = data1[j * rows1 + i];
						sum += data;
						sumSquare += data * data;
					}
					double variance = (sumSquare - (sum * sum / rows_d)) / (rows_d - ddof);
					double std = std::sqrt(variance);
					data_result[j] = std;
				}
			}
			else
			{
				size_t cols1 = this->actual_cols;

				size_t final_pos_cols = this->final_pos_cols;
				size_t final_pos_rows = this->final_pos_rows;

				for (size_t j = 0; j < final_pos_cols; j += 4)
				{
					__m256d _sum = _mm256_setzero_pd();
					__m256d _sumSquare = _mm256_setzero_pd();
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

						_sum = _mm256_add_pd(_sum, a);
						_sumSquare = _mm256_add_pd(_sumSquare, _mm256_mul_pd(a, a));
					}
					__m256d variance = _mm256_div_pd(_mm256_sub_pd(_sumSquare, _mm256_div_pd(_mm256_mul_pd(_sum, _sum), _rows)), _mm256_sub_pd(_rows, _ddof));
					_mm256_store_pd(&data_result[j], _mm256_sqrt_pd(variance));
				}
				for (size_t j = final_pos_cols; j < cols; j++)
				{
					__m256d _sum = _mm256_setzero_pd();
					__m256d _sumSquare = _mm256_setzero_pd();

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
							data1[(i + 1) * cols1 + j],
							data1[(i + 2) * cols1 + j],
							data1[(i + 3) * cols1 + j]);

						_sum = _mm256_add_pd(_sum, a);
						_sumSquare = _mm256_add_pd(_sumSquare, _mm256_mul_pd(a, a));
					}
					__m128d vlow = _mm256_castpd256_pd128(_sum);
					__m128d vhigh = _mm256_extractf128_pd(_sum, 1);
					vlow = _mm_add_pd(vlow, vhigh);

					__m128d high64 = _mm_unpackhi_pd(vlow, vlow);
					double sum = _mm_cvtsd_f64(_mm_add_sd(vlow, high64));
					//--
					__m128d vlow1 = _mm256_castpd256_pd128(_sumSquare);
					vhigh = _mm256_extractf128_pd(_sumSquare, 1);
					vlow1 = _mm_add_pd(vlow1, vhigh);

					high64 = _mm_unpackhi_pd(vlow1, vlow1);
					double sumSquare = _mm_cvtsd_f64(_mm_add_sd(vlow1, high64));

					for (size_t i = final_pos_rows; i < rows; i++)
					{
						double data = data1[i * cols1 + j];
						sum += data;
						sumSquare += data * data;
					}
					double variance = (sumSquare - (sum * sum / rows_d)) / (rows_d - ddof);
					double std = std::sqrt(variance);
					data_result[j] = std;
				}
			}
			return result;
		}

		double std_all(double ddof = 0.0, double* mean = nullptr)
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			size_t size = this->_size;

			double size_d = static_cast<double>(size);

			double* data1 = this->_data;

			__m256d _sum = _mm256_setzero_pd();
			__m256d _sumSquare = _mm256_setzero_pd();

			double sum = 0.0;
			double sumSquare = 0.0;

			if constexpr (this_contiguous)
			{
				size_t size = this->_size;
				size_t final_pos_size = this->final_pos_size;

				for (size_t i = 0; i < final_pos_size; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);

					_sum = _mm256_add_pd(_sum, a);
					_sumSquare = _mm256_add_pd(_sumSquare, _mm256_mul_pd(a, a));
				}
				for (size_t i = final_pos_size; i < size; i++)
				{
					double data = data1[i];
					sum += data;
					sumSquare += data * data;
				}
			}
			else if constexpr (this_transposed)
			{
				size_t rows1 = this->actual_rows;

				size_t final_pos_rows = this->final_pos_rows;
				size_t final_pos_cols = this->final_pos_cols;

				for (size_t i = 0; i < final_pos_rows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

						_sum = _mm256_add_pd(_sum, a);
						_sumSquare = _mm256_add_pd(_sumSquare, _mm256_mul_pd(a, a));
					}
				}
				for (size_t i = final_pos_rows; i < rows; i++)
				{
					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
							data1[(j + 1) * rows1 + i],
							data1[(j + 2) * rows1 + i],
							data1[(j + 3) * rows1 + i]);

						_sum = _mm256_add_pd(_sum, a);
						_sumSquare = _mm256_add_pd(_sumSquare, _mm256_mul_pd(a, a));
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						double data = data1[j * rows1 + i];
						sum += data;
						sumSquare += data * data;
					}
				}
			}
			else
			{
				size_t cols1 = this->actual_cols;

				size_t final_pos_cols = this->final_pos_cols;
				size_t final_pos_rows = this->final_pos_rows;

				for (size_t j = 0; j < final_pos_cols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);
						_sum = _mm256_add_pd(_sum, a);
						_sumSquare = _mm256_add_pd(_sumSquare, _mm256_mul_pd(a, a));
					}
				}
				for (size_t j = final_pos_cols; j < cols; j++)
				{
					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
							data1[(i + 1) * cols1 + j],
							data1[(i + 2) * cols1 + j],
							data1[(i + 3) * cols1 + j]);
						_sum = _mm256_add_pd(_sum, a);
						_sumSquare = _mm256_add_pd(_sumSquare, _mm256_mul_pd(a, a));
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						double data = data1[i * cols1 + j];
						sum += data;
						sumSquare += data * data;
					}
				}
			}

			__m128d vlow = _mm256_castpd256_pd128(_sum);
			__m128d vhigh = _mm256_extractf128_pd(_sum, 1);
			vlow = _mm_add_pd(vlow, vhigh);

			__m128d high64 = _mm_unpackhi_pd(vlow, vlow);
			sum += _mm_cvtsd_f64(_mm_add_sd(vlow, high64));
			//--
			__m128d vlow1 = _mm256_castpd256_pd128(_sum);
			vhigh = _mm256_extractf128_pd(_sum, 1);
			vlow = _mm_add_pd(vlow1, vhigh);

			high64 = _mm_unpackhi_pd(vlow1, vlow1);
			sumSquare += _mm_cvtsd_f64(_mm_add_sd(vlow1, high64));

			if (mean != nullptr) *mean = sum / size_d;

			double variance = (sumSquare - (sum * sum / size_d)) / (size_d - ddof);
			double std = std::sqrt(variance);
			return std;
		}

		// Min

		vector<double> min_rowwise()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			vector<double> result(rows);

			double* data_result = result._data;

			if constexpr (this_transposed)
			{
				size_t rows1 = this->actual_rows;

				size_t final_pos_rows = this->final_pos_rows;
				size_t final_pos_cols = this->final_pos_cols;

				for (size_t i = 0; i < final_pos_rows; i += 4)
				{
					__m256d _min = _mm256_set1_pd(DBL_MAX);
					for (size_t j = 0; j < cols; j++)
					{
						_min = _mm256_min_pd(_min, _mm256_load_pd(&data1[j * rows1 + i]));
					}
					_mm256_store_pd(&data_result[i], _min);
				}
				for (size_t i = final_pos_rows; i < rows; i++)
				{
					__m256d _min = _mm256_set1_pd(DBL_MAX);
					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						_min = _mm256_min_pd(_min, _mm256_setr_pd(data1[j * rows1 + i],
							data1[(j + 1) * rows1 + i],
							data1[(j + 2) * rows1 + i],
							data1[(j + 3) * rows1 + i]));
					}
					__m256d tempMin = _mm256_permute2f128_pd(_min, _min, 0x01);
					_min = _mm256_min_pd(_min, tempMin);

					__m128d low = _mm256_castpd256_pd128(_min);
					__m128d high = _mm256_extractf128_pd(_min, 1);

					low = _mm_min_pd(low, high);
					double min = _mm_cvtsd_f64(low);

					for (size_t j = final_pos_cols; j < cols; j++)
					{
						double data = data1[j * rows1 + i];
						if (data < min) min = data;
					}
					data_result[i] = min;
				}
			}
			else
			{
				size_t cols1 = this->actual_cols;

				size_t final_pos_rows = this->final_pos_rows;
				size_t final_pos_cols = this->final_pos_cols;

				for (size_t i = 0; i < final_pos_rows; i += 4)
				{
					__m256d _min = _mm256_set1_pd(DBL_MAX);
					for (size_t j = 0; j < cols; j++)
					{
						_min = _mm256_min_pd(_min, _mm256_setr_pd(data1[i * cols1 + j],
							data1[(i + 1) * cols1 + j],
							data1[(i + 2) * cols1 + j],
							data1[(i + 3) * cols1 + j]));
					}
					_mm256_store_pd(&data_result[i], _min);
				}
				for (size_t i = final_pos_rows; i < rows; i++)
				{
					__m256d _min = _mm256_set1_pd(DBL_MAX);
					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						_min = _mm256_min_pd(_min, _mm256_load_pd(data1[i * cols1 + j]));
					}
					__m256d tempMin = _mm256_permute2f128_pd(_min, _min, 0x01);
					_min = _mm256_min_pd(_min, tempMin);

					__m128d low = _mm256_castpd256_pd128(_min);
					__m128d high = _mm256_extractf128_pd(_min, 1);

					low = _mm_min_pd(low, high);
					double min = _mm_cvtsd_f64(low);

					for (size_t j = final_pos_cols; j < cols; j++)
					{
						double data = data1[i * cols1 + j];
						if (data < min) min = data;
					}
					data_result[i] = min;
				}
			}
			return result;
		}

		vector<double> min_colwise()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			vector<double> result(cols);

			double* data_result = result._data;

			if constexpr (this_transposed)
			{
				size_t rows1 = this->actual_rows;

				size_t final_pos_cols = this->final_pos_cols;
				size_t final_pos_rows = this->final_pos_rows;

				for (size_t j = 0; j < final_pos_cols; j += 4)
				{
					__m256d _min = _mm256_set1_pd(DBL_MAX);
					for (size_t i = 0; i < rows; i++)
					{
						_min = _mm256_min_pd(_min, _mm256_setr_pd(data1[j * rows1 + i],
							data1[(j + 1) * rows1 + i],
							data1[(j + 2) * rows1 + i],
							data1[(j + 3) * rows1 + i]));
					}
					_mm256_store_pd(&data_result[j], _min);
				}
				for (size_t j = final_pos_cols; j < cols; j++)
				{
					__m256d _min = _mm256_set1_pd(DBL_MAX);

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						_min = _mm256_min_pd(_min, _mm256_load_pd(&data1[j * rows1 + i]));
					}
					__m256d tempMin = _mm256_permute2f128_pd(_min, _min, 0x01);
					_min = _mm256_min_pd(_min, tempMin);

					__m128d low = _mm256_castpd256_pd128(_min);
					__m128d high = _mm256_extractf128_pd(_min, 1);

					low = _mm_min_pd(low, high);
					double min = _mm_cvtsd_f64(low);

					for (size_t i = final_pos_rows; i < rows; i++)
					{
						double data = data1[j * rows1 + i];
						if (data < min) min = data;
					}
					data_result[j] = min;
				}
			}
			else
			{
				size_t cols1 = this->actual_cols;

				size_t final_pos_cols = this->final_pos_cols;
				size_t final_pos_rows = this->final_pos_rows;

				for (size_t j = 0; j < final_pos_cols; j += 4)
				{
					__m256d _min = _mm256_set1_pd(DBL_MAX);
					for (size_t i = 0; i < rows; i++)
					{
						_min = _mm256_min_pd(_min, _mm256_load_pd(&data1[i * cols1 + j]));
					}
					_mm256_store_pd(&data_result[j], _min);
				}
				for (size_t j = final_pos_cols; j < cols; j++)
				{
					__m256d _min = _mm256_set1_pd(DBL_MAX);

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						_min = _mm256_min_pd(_min, _mm256_setr_pd(data1[i * cols1 + j],
							data1[(i + 1) * cols1 + j],
							data1[(i + 2) * cols1 + j],
							data1[(i + 3) * cols1 + j]));
					}
					__m256d tempMin = _mm256_permute2f128_pd(_min, _min, 0x01);
					_min = _mm256_min_pd(_min, tempMin);

					__m128d low = _mm256_castpd256_pd128(_min);
					__m128d high = _mm256_extractf128_pd(_min, 1);

					low = _mm_min_pd(low, high);
					double min = _mm_cvtsd_f64(low);

					for (size_t i = final_pos_rows; i < rows; i++)
					{
						double data = data1[i * cols1 + j];
						if (data < min) min = data;
					}
					data_result[j] = min;
				}
			}
			return result;
		}

		double min_all()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			__m256d _min = _mm256_set1_pd(DBL_MAX);
			double min = DBL_MAX;

			if constexpr (this_contiguous)
			{
				size_t size = this->_size;
				size_t final_pos_size = this->final_pos_size;

				for (size_t i = 0; i < final_pos_size; i += 4)
				{
					_min = _mm256_min_pd(_min, _mm256_load_pd(&data1[i]));
				}
				for (size_t i = final_pos_size; i < size; i++)
				{
					double data = data1[i];
					if (data < min) min = data;
				}
			}
			else if constexpr (this_transposed)
			{
				size_t rows1 = this->actual_rows;

				size_t final_pos_rows = this->final_pos_rows;
				size_t final_pos_cols = this->final_pos_cols;

				for (size_t i = 0; i < final_pos_rows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						_min = _mm256_min_pd(_min, _mm256_load_pd(&data1[j * rows1 + i]));
					}
				}
				for (size_t i = final_pos_rows; i < rows; i++)
				{
					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						_min = _mm256_min_pd(_min, _mm256_setr_pd(data1[j * rows1 + i],
							data1[(j + 1) * rows1 + i],
							data1[(j + 2) * rows1 + i],
							data1[(j + 3) * rows1 + i]));
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						double data = data1[j * rows1 + i];
						if (data < min) min = data;
					}
				}
			}
			else
			{
				size_t cols1 = this->actual_cols;

				size_t final_pos_cols = this->final_pos_cols;
				size_t final_pos_rows = this->final_pos_rows;

				for (size_t j = 0; j < final_pos_cols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						_min = _mm256_min_pd(_min, _mm256_load_pd(&data1[i * cols1 + j]));
					}
				}
				for (size_t j = final_pos_cols; j < cols; j++)
				{
					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						_min = _mm256_min_pd(_min, _mm256_setr_pd(data1[i * cols1 + j],
							data1[(i + 1) * cols1 + j],
							data1[(i + 2) * cols1 + j],
							data1[(i + 3) * cols1 + j]));
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						double data = data1[i * cols1 + j];
						if (data < min) min = data;
					}
				}
			}

			__m256d tempMin = _mm256_permute2f128_pd(_min, _min, 0x01);
			_min = _mm256_min_pd(_min, tempMin);

			__m128d low = _mm256_castpd256_pd128(_min);
			__m128d high = _mm256_extractf128_pd(_min, 1);

			low = _mm_min_pd(low, high);
			double temp_min_d = _mm_cvtsd_f64(low);

			if (temp_min_d < min) min = temp_min_d;

			return min;
		}

		void argmin_all(size_t* row, size_t* col) 
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			__m256i four = _mm256_set1_epi64x(4);

			__m256d _min = _mm256_set1_pd(DBL_MAX);
			double min = DBL_MAX;

			if constexpr (this_contiguous)
			{
				size_t size = this->_size;

				size_t final_pos_size = this->final_pos_size;

				__m256i min_indices = _mm256_setr_epi64x(0, 1, 2, 4);
				size_t min_index = 0;

				__m256i indices = _mm256_setr_epi64x(0, 1, 2, 4);

				for (size_t i = 0; i < final_pos_size; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);

					int mask = _mm256_movemask_pd(_mm256_cmp_pd(a, _min, _CMP_LT_OQ));

					min_indices = _mm256_castpd_si256(_mm256_blend_pd(_mm256_castsi256_pd(min_indices), _mm256_castsi256_pd(indices), mask));

					_min = _mm256_blend_pd(_min, a, mask);

					indices = _mm256_add_epi64(indices, four);
				}
				for (size_t i = final_pos_size; i < size; i++)
				{
					double data = data1[i];
					if (min < data)
					{
						min = data;
						min_index = i;
					}
				}

				double mins_arr[4];
				size_t indices_arr[4];

				_mm256_store_pd(mins_arr, _min);
				_mm256_storeu_epi64(indices_arr, indices);

				for (size_t i = 0; i < 4; i++)
				{
					double element = mins_arr[i];
					if (element < min)
					{
						min = element;
						min_index = indices_arr[i];
					}
				}
				if constexpr (this_transposed)
				{
					*row = min_index % rows;
					*col = min_index / rows;
				}
				else
				{
					*row = min_index / cols;
					*col = min_index % cols;
				}
				
			}
			else if constexpr (this_transposed)
			{
				size_t rows1 = this->actual_rows;

				size_t final_pos_rows = this->final_pos_rows;

				__m256i _i = _mm256_set1_epi64x(0, 1, 2, 3);
				__m256i _j = _mm256_setzero_si256();

				__m256i one = _mm256_set1_epi64x(1);

				__m256i _i_min = _mm256_setzero_si256();
				__m256i _j_min = _mm256_setzero_si256();

				size_t row_index;
				size_t col_index;

				for (size_t i = 0; i < final_pos_rows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

						int mask = _mm256_movemask_pd(_mm256_cmp_pd(a, _min, _CMP_LT_OQ));

						_i_min = _mm256_castpd_si256(_mm256_blend_pd(_mm256_castsi256_pd(_i_min), _mm256_castsi256_pd(_i), mask));

						_j_min = _mm256_castpd_si256(_mm256_blend_pd(_mm256_castsi256_pd(_j_min), _mm256_castsi256_pd(_j), mask));

						_min = _mm256_blend_pd(_min, a, mask);

						_j = _mm256_add_epi64(_j, one)
					}
					_i = _mm256_add_epi64(_i, four);
				}
				for (size_t i = final_pos_rows; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						double data = data1[j * rows1 + i];
						if (data < min)
						{
							min = data;
							row_index = i;
							col_index = j;
						}
					}
				}

				double mins_arr[4];
				size_t i_arr[4];
				size_t j_arr[4];

				_mm256_store_pd(mins_arr, _min);
				_mm256_storeu_epi64(i_arr, _i);
				_mm256_storeu_epi64(j_arr, _j);

				for (size_t i = 0; i < 4; i++)
				{
					double element = mins_arr[i];
					if (element < min)
					{
						min = element;
						row_index = i_arr[i];
						col_index = j_arr[i];
					}
				}
				*row = row_index;
				*col = col_index;
			}
			else
			{
				size_t cols1 = this->actual_cols;

				size_t final_pos_cols = this->final_pos_cols;
				
				__m256i _i = _mm256_setzero_si256();
				__m256i _j = _mm256_set1_epi64x(0, 1, 2, 3);

				__m256i one = _mm256_set1_epi64x(1);

				__m256i _i_min = _mm256_setzero_si256();
				__m256i _j_min = _mm256_setzero_si256();

				size_t row_index;
				size_t col_index;

				for (size_t j = 0; j < final_pos_cols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

						int mask = _mm256_movemask_pd(_mm256_cmp_pd(a, _min, _CMP_LT_OQ));

						_i_min = _mm256_castpd_si256(_mm256_blend_pd(_mm256_castsi256_pd(_i_min), _mm256_castsi256_pd(_i), mask));

						_j_min = _mm256_castpd_si256(_mm256_blend_pd(_mm256_castsi256_pd(_j_min), _mm256_castsi256_pd(_j), mask));

						_min = _mm256_blend_pd(_min, a, mask);

						_i = _mm256_add_epi64(_i, one);
					}
					_j = _mm256_add_epi64(_j_min, four);
				}
				for (size_t j = final_pos_cols; j < cols; j++)
				{
					for (size_t i = 0; i < rows; i++)
					{
						double data = data1[i * cols1 + j];
						if (data < min)
						{
							min = data;
							row_index = i;
							col_index = j;
						}
					}
				}

				double mins_arr[4];
				size_t i_arr[4];
				size_t j_arr[4];

				_mm256_store_pd(mins_arr, _min);
				_mm256_storeu_epi64(i_arr, _i);
				_mm256_storeu_epi64(j_arr, _j);

				for (size_t i = 0; i < 4; i++)
				{
					double element = mins_arr[i];
					if (element < min)
					{
						min = element;
						row_index = i_arr[i];
						col_index = j_arr[i];
					}
				}
				*row = row_index;
				*col = col_index;
			}
		}

		vector<uint64_t> argmin_rowwise()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			vector<uint64_t> result(rows);

			uint64_t* data_result = result._data;

			if constexpr (this_transposed)
			{
				size_t rows1 = this->actual_rows;

				size_t final_pos_rows = this->final_pos_rows;

				__m256i one = _mm256_set1_epi64x(1);

				for (size_t i = 0; i < final_pos_rows; i += 4)
				{
					__m256d _min = _mm256_set1_pd(DBL_MAX);
					__m256i indices = _mm256_setzero_si256();
					__m256i min_indices = _mm256_setzero_si256();
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

						int mask = _mm256_movemask_pd(_mm256_cmp_pd(a, _min, _CMP_LT_OQ));

						min_indices = _mm256_castpd_si256(_mm256_blend_pd(_mm256_castsi256_pd(min_indices), _mm256_castsi256_pd(indices), mask));

						_min = _mm256_blend_pd(_min, a, mask);

						indices = _mm256_add_epi64(indices, one);
					}
					_mm256_storeu_epi64(&data_result[i], min_indices);
				}
				for (size_t i = final_pos_rows; i < rows; i++)
				{
					double min = DBL_MAX;
					size_t index;
					for (size_t j = 0; j < cols; j++)
					{
						double data = data1[j * rows1 + i];
						if (data < min)
						{
							min = data;
							index = j;
						}
					}
					data_result[i] = index;
				}
			}
			else
			{
				size_t cols1 = this->actual_cols;

				size_t final_pos_rows = this->final_pos_rows;

				__m256i one = _mm256_set1_epi64x(1);

				for (size_t i = 0; i < final_pos_rows; i += 4)
				{
					__m256d _min = _mm256_set1_pd(DBL_MAX);
					__m256i indices = _mm256_setzero_si256();
					__m256i min_indices = _mm256_setzero_si256();
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_setr_pd(data1[i * cols1 + j], 
							data1[(i + 1) * cols1 + j], 
							data1[(i + 2) * cols1 + j], 
							data1[(i + 3) * cols1 + j]);

						int mask = _mm256_movemask_pd(_mm256_cmp_pd(a, _min, _CMP_LT_OQ));

						min_indices = _mm256_castpd_si256(_mm256_blend_pd(_mm256_castsi256_pd(min_indices), _mm256_castsi256_pd(indices), mask));

						_min = _mm256_blend_pd(_min, a, mask);

						indices = _mm256_add_epi64(indices, one);
					}
					_mm256_storeu_epi64(&data_result[i], min_indices);
				}
				for (size_t i = final_pos_rows; i < rows; i++)
				{
					double min = DBL_MAX;
					size_t index;
					for (size_t j = 0; j < cols; j++)
					{
						double data = data1[i * cols1 + j];
						if (data < min)
						{
							min = data;
							index = j;
						}
					}
					data_result[i] = index;
				}
			}
			return result;
		}

		vector<uint64_t> argmin_colwise()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			vector<uint64_t> result(cols);

			uint64_t* data_result = result._data;

			if constexpr (this_transposed)
			{
				size_t rows1 = this->actual_rows;

				size_t final_pos_cols = this->final_pos_cols;

				__m256i one = _mm256_set1_epi64x(1);

				for (size_t j = 0; j < final_pos_cols; j += 4)
				{
					__m256d _min = _mm256_set1_pd(DBL_MAX);
					__m256i indices = _mm256_setzero_si256();
					__m256i min_indices = _mm256_setzero_si256();
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_setr_pd(data1[j * rows1 + i], 
							data1[(j + 1) * rows1 + i], 
							data1[(j + 2) * rows1 + i], 
							data1[(j + 3) * rows1 + i]);

						int mask = _mm256_movemask_pd(_mm256_cmp_pd(a, _min, _CMP_LT_OQ));

						min_indices = _mm256_castpd_si256(_mm256_blend_pd(_mm256_castsi256_pd(min_indices), _mm256_castsi256_pd(indices), mask));

						_min = _mm256_blend_pd(_min, a, mask);

						indices = _mm256_add_epi64(indices, one);
					}
					_mm256_storeu_epi64(&data_result[j], min_indices);
				}
				for (size_t j = final_pos_cols; j < cols; j++)
				{
					double min = DBL_MAX;
					size_t index;
					for (size_t i = 0; i < rows; i++)
					{
						double data = data1[j * rows1 + i];
						if (data < min)
						{
							min = data;
							index = i;
						}
					}
					data_result[j] = index;
				}
			}
			else
			{
				size_t cols1 = this->actual_cols;

				size_t final_pos_cols = this->final_pos_cols;

				__m256i one = _mm256_set1_epi64x(1);

				for (size_t j = 0; j < final_pos_cols; j += 4)
				{
					__m256d _min = _mm256_set1_pd(DBL_MAX);
					__m256i indices = _mm256_setzero_si256();
					__m256i min_indices = _mm256_setzero_si256();
					for (size_t i = 0; i < cols; i++)
					{
						__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

						int mask = _mm256_movemask_pd(_mm256_cmp_pd(a, _min, _CMP_LT_OQ));

						min_indices = _mm256_castpd_si256(_mm256_blend_pd(_mm256_castsi256_pd(min_indices), _mm256_castsi256_pd(indices), mask));

						_min = _mm256_blend_pd(_min, a, mask);

						indices = _mm256_add_epi64(indices, one);
					}
					_mm256_storeu_epi64(&data_result[j], min_indices);
				}
				for (size_t j = final_pos_cols; j < cols; j++)
				{
					double min = DBL_MAX;
					size_t index;
					for (size_t i = 0; i < rows; i++)
					{
						double data = data1[i * cols1 + j];
						if (data < min)
						{
							min = data;
							index = i;
						}
					}
					data_result[j] = index;
				}
			}
			return result;
		}

		// Max

		vector<double> max_rowwise()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			vector<double> result(rows);

			double* data_result = result._data;

			if constexpr (this_transposed)
			{
				size_t rows1 = this->actual_rows;

				size_t final_pos_rows = this->final_pos_rows;
				size_t final_pos_cols = this->final_pos_cols;

				for (size_t i = 0; i < final_pos_rows; i += 4)
				{
					__m256d _max = _mm256_set1_pd(DBL_MIN);
					for (size_t j = 0; j < cols; j++)
					{
						_max = _mm256_max_pd(_max, _mm256_load_pd(&data1[j * rows1 + i]));
					}
					_mm256_store_pd(&data_result[i], _max);
				}
				for (size_t i = final_pos_rows; i < rows; i++)
				{
					__m256d _max = _mm256_set1_pd(DBL_MIN);
					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						_max = _mm256_max_pd(_max, _mm256_setr_pd(data1[j * rows1 + i],
							data1[(j + 1) * rows1 + i],
							data1[(j + 2) * rows1 + i],
							data1[(j + 3) * rows1 + i]));
					}
					__m256d tempmax = _mm256_permute2f128_pd(_max, _max, 0x01);
					_max = _mm256_max_pd(_max, tempmax);

					__m128d low = _mm256_castpd256_pd128(_max);
					__m128d high = _mm256_extractf128_pd(_max, 1);

					low = _mm_max_pd(low, high);
					double max = _mm_cvtsd_f64(low);

					for (size_t j = final_pos_cols; j < cols; j++)
					{
						double data = data1[j * rows1 + i];
						if (data > max) max = data;
					}
					data_result[i] = max;
				}
			}
			else
			{
				size_t cols1 = this->actual_cols;

				size_t final_pos_rows = this->final_pos_rows;
				size_t final_pos_cols = this->final_pos_cols;

				for (size_t i = 0; i < final_pos_rows; i += 4)
				{
					__m256d _max = _mm256_set1_pd(DBL_MIN);
					for (size_t j = 0; j < cols; j++)
					{
						_max = _mm256_max_pd(_max, _mm256_setr_pd(data1[i * cols1 + j],
							data1[(i + 1) * cols1 + j],
							data1[(i + 2) * cols1 + j],
							data1[(i + 3) * cols1 + j]));
					}
					_mm256_store_pd(&data_result[i], _max);
				}
				for (size_t i = final_pos_rows; i < rows; i++)
				{
					__m256d _max = _mm256_set1_pd(DBL_MIN);
					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						_max = _mm256_max_pd(_max, _mm256_load_pd(data1[i * cols1 + j]));
					}
					__m256d tempmax = _mm256_permute2f128_pd(_max, _max, 0x01);
					_max = _mm256_max_pd(_max, tempmax);

					__m128d low = _mm256_castpd256_pd128(_max);
					__m128d high = _mm256_extractf128_pd(_max, 1);

					low = _mm_max_pd(low, high);
					double max = _mm_cvtsd_f64(low);

					for (size_t j = final_pos_cols; j < cols; j++)
					{
						double data = data1[i * cols1 + j];
						if (data > max) max = data;
					}
					data_result[i] = max;
				}
			}
			return result;
		}

		vector<double> max_colwise()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			vector<double> result(cols);

			double* data_result = result._data;

			if constexpr (this_transposed)
			{
				size_t rows1 = this->actual_rows;

				size_t final_pos_cols = this->final_pos_cols;
				size_t final_pos_rows = this->final_pos_rows;

				for (size_t j = 0; j < final_pos_cols; j += 4)
				{
					__m256d _max = _mm256_set1_pd(DBL_MIN);
					for (size_t i = 0; i < rows; i++)
					{
						_max = _mm256_max_pd(_max, _mm256_setr_pd(data1[j * rows1 + i],
							data1[(j + 1) * rows1 + i],
							data1[(j + 2) * rows1 + i],
							data1[(j + 3) * rows1 + i]));
					}
					_mm256_store_pd(&data_result[j], _max);
				}
				for (size_t j = final_pos_cols; j < cols; j++)
				{
					__m256d _max = _mm256_set1_pd(DBL_MIN);

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						_max = _mm256_max_pd(_max, _mm256_load_pd(&data1[j * rows1 + i]));
					}
					__m256d tempmax = _mm256_permute2f128_pd(_max, _max, 0x01);
					_max = _mm256_max_pd(_max, tempmax);

					__m128d low = _mm256_castpd256_pd128(_max);
					__m128d high = _mm256_extractf128_pd(_max, 1);

					low = _mm_max_pd(low, high);
					double max = _mm_cvtsd_f64(low);

					for (size_t i = final_pos_rows; i < rows; i++)
					{
						double data = data1[j * rows1 + i];
						if (data > max) max = data;
					}
					data_result[j] = max;
				}
			}
			else
			{
				size_t cols1 = this->actual_cols;

				size_t final_pos_cols = this->final_pos_cols;
				size_t final_pos_rows = this->final_pos_rows;

				for (size_t j = 0; j < final_pos_cols; j += 4)
				{
					__m256d _max = _mm256_set1_pd(DBL_MIN);
					for (size_t i = 0; i < rows; i++)
					{
						_max = _mm256_max_pd(_max, _mm256_load_pd(&data1[i * cols1 + j]));
					}
					_mm256_store_pd(&data_result[j], _max);
				}
				for (size_t j = final_pos_cols; j < cols; j++)
				{
					__m256d _max = _mm256_set1_pd(DBL_MIN);

					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						_max = _mm256_max_pd(_max, _mm256_setr_pd(data1[i * cols1 + j],
							data1[(i + 1) * cols1 + j],
							data1[(i + 2) * cols1 + j],
							data1[(i + 3) * cols1 + j]));
					}
					__m256d tempmax = _mm256_permute2f128_pd(_max, _max, 0x01);
					_max = _mm256_max_pd(_max, tempmax);

					__m128d low = _mm256_castpd256_pd128(_max);
					__m128d high = _mm256_extractf128_pd(_max, 1);

					low = _mm_max_pd(low, high);
					double max = _mm_cvtsd_f64(low);

					for (size_t i = final_pos_rows; i < rows; i++)
					{
						double data = data1[i * cols1 + j];
						if (data > max) max = data;
					}
					data_result[j] = max;
				}
			}
			return result;
		}

		double max_all()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			__m256d _max = _mm256_set1_pd(DBL_MIN);
			double max = DBL_MIN;

			if constexpr (this_contiguous)
			{
				size_t size = this->_size;
				size_t final_pos_size = this->final_pos_size;

				for (size_t i = 0; i < final_pos_size; i += 4)
				{
					_max = _mm256_max_pd(_max, _mm256_load_pd(&data1[i]));
				}
				for (size_t i = final_pos_size; i < size; i++)
				{
					double data = data1[i];
					if (data > max) max = data;
				}
			}
			else if constexpr (this_transposed)
			{
				size_t rows1 = this->actual_rows;

				size_t final_pos_rows = this->final_pos_rows;
				size_t final_pos_cols = this->final_pos_cols;

				for (size_t i = 0; i < final_pos_rows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						_max = _mm256_max_pd(_max, _mm256_load_pd(&data1[j * rows1 + i]));
					}
				}
				for (size_t i = final_pos_rows; i < rows; i++)
				{
					for (size_t j = 0; j < final_pos_cols; j += 4)
					{
						_max = _mm256_max_pd(_max, _mm256_setr_pd(data1[j * rows1 + i],
							data1[(j + 1) * rows1 + i],
							data1[(j + 2) * rows1 + i],
							data1[(j + 3) * rows1 + i]));
					}
					for (size_t j = final_pos_cols; j < cols; j++)
					{
						double data = data1[j * rows1 + i];
						if (data > max) max = data;
					}
				}
			}
			else
			{
				size_t cols1 = this->actual_cols;

				size_t final_pos_cols = this->final_pos_cols;
				size_t final_pos_rows = this->final_pos_rows;

				for (size_t j = 0; j < final_pos_cols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						_max = _mm256_max_pd(_max, _mm256_load_pd(&data1[i * cols1 + j]));
					}
				}
				for (size_t j = final_pos_cols; j < cols; j++)
				{
					for (size_t i = 0; i < final_pos_rows; i += 4)
					{
						_max = _mm256_max_pd(_max, _mm256_setr_pd(data1[i * cols1 + j],
							data1[(i + 1) * cols1 + j],
							data1[(i + 2) * cols1 + j],
							data1[(i + 3) * cols1 + j]));
					}
					for (size_t i = final_pos_rows; i < rows; i++)
					{
						double data = data1[i * cols1 + j];
						if (data > max) max = data;
					}
				}
			}

			__m256d tempmax = _mm256_permute2f128_pd(_max, _max, 0x01);
			_max = _mm256_max_pd(_max, tempmax);

			__m128d low = _mm256_castpd256_pd128(_max);
			__m128d high = _mm256_extractf128_pd(_max, 1);

			low = _mm_max_pd(low, high);
			double temp_max_d = _mm_cvtsd_f64(low);

			if (temp_max_d < max) max = temp_max_d;

			return max;
		}

		void argmax_all(size_t* row, size_t* col)
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			__m256i four = _mm256_set1_epi64x(4);

			__m256d _max = _mm256_set1_pd(DBL_MIN);
			double max = DBL_MIN;

			if constexpr (this_contiguous)
			{
				size_t size = this->_size;

				size_t final_pos_size = this->final_pos_size;

				__m256i max_indices = _mm256_setr_epi64x(0, 1, 2, 4);
				size_t max_index = 0;

				__m256i indices = _mm256_setr_epi64x(0, 1, 2, 4);

				for (size_t i = 0; i < final_pos_size; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);

					int mask = _mm256_movemask_pd(_mm256_cmp_pd(a, _max, _CMP_GT_OQ));

					max_indices = _mm256_castpd_si256(_mm256_blend_pd(_mm256_castsi256_pd(max_indices), _mm256_castsi256_pd(indices), mask));

					_max = _mm256_blend_pd(_max, a, mask);

					indices = _mm256_add_epi64(indices, four);
				}
				for (size_t i = final_pos_size; i < size; i++)
				{
					double data = data1[i];
					if (max < data)
					{
						max = data;
						max_index = i;
					}
				}

				double maxs_arr[4];
				size_t indices_arr[4];

				_mm256_store_pd(maxs_arr, _max);
				_mm256_storeu_epi64(indices_arr, indices);

				for (size_t i = 0; i < 4; i++)
				{
					double data = maxs_arr[i];
					if (data > max)
					{
						max = data;
						max_index = indices_arr[i];
					}
				}
				if constexpr (this_transposed)
				{
					*row = max_index % rows;
					*col = max_index / rows;
				}
				else
				{
					*row = max_index / cols;
					*col = max_index % cols;
				}

			}
			else if constexpr (this_transposed)
			{
				size_t rows1 = this->actual_rows;

				size_t final_pos_rows = this->final_pos_rows;

				__m256i _i = _mm256_set1_epi64x(0, 1, 2, 3);
				__m256i _j = _mm256_setzero_si256();

				__m256i one = _mm256_set1_epi64x(1);

				__m256i _i_max = _mm256_setzero_si256();
				__m256i _j_max = _mm256_setzero_si256();

				size_t row_index;
				size_t col_index;

				for (size_t i = 0; i < final_pos_rows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

						int mask = _mm256_movemask_pd(_mm256_cmp_pd(a, _max, _CMP_GT_OQ));

						_i_max = _mm256_castpd_si256(_mm256_blend_pd(_mm256_castsi256_pd(_i_max), _mm256_castsi256_pd(_i), mask));

						_j_max = _mm256_castpd_si256(_mm256_blend_pd(_mm256_castsi256_pd(_j_max), _mm256_castsi256_pd(_j), mask));

						_max = _mm256_blend_pd(_max, a, mask);

						_j = _mm256_add_epi64(_j, one)
					}
					_i = _mm256_add_epi64(_i, four);
				}
				for (size_t i = final_pos_rows; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						double data = data1[j * rows1 + i];
						if (data > max)
						{
							max = data;
							row_index = i;
							col_index = j;
						}
					}
				}

				double maxs_arr[4];
				size_t i_arr[4];
				size_t j_arr[4];

				_mm256_store_pd(maxs_arr, _max);
				_mm256_storeu_epi64(i_arr, _i);
				_mm256_storeu_epi64(j_arr, _j);

				for (size_t i = 0; i < 4; i++)
				{
					double data = maxs_arr[i];
					if (data > max)
					{
						max = data;
						row_index = i_arr[i];
						col_index = j_arr[i];
					}
				}
				*row = row_index;
				*col = col_index;
			}
			else
			{
				size_t cols1 = this->actual_cols;

				size_t final_pos_cols = this->final_pos_cols;

				__m256i _i = _mm256_setzero_si256();
				__m256i _j = _mm256_set1_epi64x(0, 1, 2, 3);

				__m256i one = _mm256_set1_epi64x(1);

				__m256i _i_max = _mm256_setzero_si256();
				__m256i _j_max = _mm256_setzero_si256();

				size_t row_index;
				size_t col_index;

				for (size_t j = 0; j < final_pos_cols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

						int mask = _mm256_movemask_pd(_mm256_cmp_pd(a, _max, _CMP_GT_OQ));

						_i_max = _mm256_castpd_si256(_mm256_blend_pd(_mm256_castsi256_pd(_i_max), _mm256_castsi256_pd(_i), mask));

						_j_max = _mm256_castpd_si256(_mm256_blend_pd(_mm256_castsi256_pd(_j_max), _mm256_castsi256_pd(_j), mask));

						_max = _mm256_blend_pd(_max, a, mask);

						_i = _mm256_add_epi64(_i, one);
					}
					_j = _mm256_add_epi64(_j_max, four);
				}
				for (size_t j = final_pos_cols; j < cols; j++)
				{
					for (size_t i = 0; i < rows; i++)
					{
						double data = data1[i * cols1 + j];
						if (data > max)
						{
							max = data;
							row_index = i;
							col_index = j;
						}
					}
				}

				double maxs_arr[4];
				size_t i_arr[4];
				size_t j_arr[4];

				_mm256_store_pd(maxs_arr, _max);
				_mm256_storeu_epi64(i_arr, _i);
				_mm256_storeu_epi64(j_arr, _j);

				for (size_t i = 0; i < 4; i++)
				{
					double data = maxs_arr[i];
					if (data > max)
					{
						max = data;
						row_index = i_arr[i];
						col_index = j_arr[i];
					}
				}
				*row = row_index;
				*col = col_index;
			}
		}

		vector<uint64_t> argmax_rowwise()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			vector<uint64_t> result(rows);

			uint64_t* data_result = result._data;

			if constexpr (this_transposed)
			{
				size_t rows1 = this->actual_rows;

				size_t final_pos_rows = this->final_pos_rows;

				__m256i one = _mm256_set1_epi64x(1);

				for (size_t i = 0; i < final_pos_rows; i += 4)
				{
					__m256d _max = _mm256_set1_pd(DBL_MIN);
					__m256i indices = _mm256_setzero_si256();
					__m256i max_indices = _mm256_setzero_si256();
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_load_pd(&data1[j * rows1 + i]);

						int mask = _mm256_movemask_pd(_mm256_cmp_pd(a, _max, _CMP_GT_OQ));

						max_indices = _mm256_castpd_si256(_mm256_blend_pd(_mm256_castsi256_pd(max_indices), _mm256_castsi256_pd(indices), mask));

						_max = _mm256_blend_pd(_max, a, mask);

						indices = _mm256_add_epi64(indices, one);
					}
					_mm256_storeu_epi64(&data_result[i], max_indices);
				}
				for (size_t i = final_pos_rows; i < rows; i++)
				{
					double max = DBL_MIN;
					size_t index;
					for (size_t j = 0; j < cols; j++)
					{
						double data = data1[j * rows1 + i];
						if (data > max)
						{
							max = data;
							index = j;
						}
					}
					data_result[i] = index;
				}
			}
			else
			{
				size_t cols1 = this->actual_cols;

				size_t final_pos_rows = this->final_pos_rows;

				__m256i one = _mm256_set1_epi64x(1);

				for (size_t i = 0; i < final_pos_rows; i += 4)
				{
					__m256d _max = _mm256_set1_pd(DBL_MIN);
					__m256i indices = _mm256_setzero_si256();
					__m256i max_indices = _mm256_setzero_si256();
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_setr_pd(data1[i * cols1 + j],
							data1[(i + 1) * cols1 + j],
							data1[(i + 2) * cols1 + j],
							data1[(i + 3) * cols1 + j]);

						int mask = _mm256_movemask_pd(_mm256_cmp_pd(a, _max, _CMP_GT_OQ));

						max_indices = _mm256_castpd_si256(_mm256_blend_pd(_mm256_castsi256_pd(max_indices), _mm256_castsi256_pd(indices), mask));

						_max = _mm256_blend_pd(_max, a, mask);

						indices = _mm256_add_epi64(indices, one);
					}
					_mm256_storeu_epi64(&data_result[i], max_indices);
				}
				for (size_t i = final_pos_rows; i < rows; i++)
				{
					double max = DBL_MIN;
					size_t index;
					for (size_t j = 0; j < cols; j++)
					{
						double data = data1[i * cols1 + j];
						if (data > max)
						{
							max = data;
							index = j;
						}
					}
					data_result[i] = index;
				}
			}
			return result;
		}

		vector<uint64_t> argmax_colwise()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;

			vector<uint64_t> result(cols);

			uint64_t* data_result = result._data;

			if constexpr (this_transposed)
			{
				size_t rows1 = this->actual_rows;

				size_t final_pos_cols = this->final_pos_cols;

				__m256i one = _mm256_set1_epi64x(1);

				for (size_t j = 0; j < final_pos_cols; j += 4)
				{
					__m256d _max = _mm256_set1_pd(DBL_MIN);
					__m256i indices = _mm256_setzero_si256();
					__m256i max_indices = _mm256_setzero_si256();
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_setr_pd(data1[j * rows1 + i],
							data1[(j + 1) * rows1 + i],
							data1[(j + 2) * rows1 + i],
							data1[(j + 3) * rows1 + i]);

						int mask = _mm256_movemask_pd(_mm256_cmp_pd(a, _max, _CMP_GT_OQ));

						max_indices = _mm256_castpd_si256(_mm256_blend_pd(_mm256_castsi256_pd(max_indices), _mm256_castsi256_pd(indices), mask));

						_max = _mm256_blend_pd(_max, a, mask);

						indices = _mm256_add_epi64(indices, one);
					}
					_mm256_storeu_epi64(&data_result[j], max_indices);
				}
				for (size_t j = final_pos_cols; j < cols; j++)
				{
					double max = DBL_MIN;
					size_t index;
					for (size_t i = 0; i < rows; i++)
					{
						double data = data1[j * rows1 + i];
						if (data > max)
						{
							max = data;
							index = i;
						}
					}
					data_result[j] = index;
				}
			}
			else
			{
				size_t cols1 = this->actual_cols;

				size_t final_pos_cols = this->final_pos_cols;

				__m256i one = _mm256_set1_epi64x(1);

				for (size_t j = 0; j < final_pos_cols; j += 4)
				{
					__m256d _max = _mm256_set1_pd(DBL_MIN);
					__m256i indices = _mm256_setzero_si256();
					__m256i max_indices = _mm256_setzero_si256();
					for (size_t i = 0; i < cols; i++)
					{
						__m256d a = _mm256_load_pd(&data1[i * cols1 + j]);

						int mask = _mm256_movemask_pd(_mm256_cmp_pd(a, _max, _CMP_GT_OQ));

						max_indices = _mm256_castpd_si256(_mm256_blend_pd(_mm256_castsi256_pd(max_indices), _mm256_castsi256_pd(indices), mask));

						_max = _mm256_blend_pd(_max, a, mask);

						indices = _mm256_add_epi64(indices, one);
					}
					_mm256_storeu_epi64(&data_result[j], max_indices);
				}
				for (size_t j = final_pos_cols; j < cols; j++)
				{
					double max = DBL_MIN;
					size_t index;
					for (size_t i = 0; i < rows; i++)
					{
						double data = data1[i * cols1 + j];
						if (data > max)
						{
							max = data;
							index = i;
						}
					}
					data_result[j] = index;
				}
			}
			return result;
		}

		// Dot
		// Missing
		template<bool return_transposed = false, bool other_transposed, bool other_contiguous, bool call_destructor>
		matrix<double, return_transposed> dot(matrix<double, other_transposed, other_contiguous, call_destructor>& other);

	private:
		double* _data;
		size_t _rows, _cols, _size;
		size_t actual_rows, actual_cols;
		size_t final_pos_rows, final_pos_cols, final_pos_size;
	};

	template <bool this_transposed, bool this_contiguous, bool call_destructor>
	class matrix<bool, this_transposed, this_contiguous, call_destructor>
	{
	public:
		matrix(size_t rows, size_t cols) : 
			_data(new uint8_t[rows * cols]), 
			_rows(rows), 
			_cols(cols), 
			actual_rows(rows), 
			actual_cols(cols), 
			final_pos_size((_size / 32) * 32),
			final_pos_rows((rows / 32) * 32),
			final_pos_cols((cols / 32) * 32), 
			final_pos_size_count((_size / 256) * 256),
			final_pos_rows_count((rows / 256) * 256),
			final_pos_cols_count((cols / 256) * 256) {}

		matrix(uint8_t* data, size_t rows, size_t cols, size_t actual_rows, size_t actual_cols) :
			_data(data),
			_rows(rows),
			_cols(cols),
			actual_rows(actual_rows),
			actual_cols(actual_cols),
			final_pos_size((_size / 32) * 32),
			final_pos_rows((rows / 32) * 32),
			final_pos_cols((cols / 32) * 32), 
			final_pos_size_count((_size / 256) * 256),
			final_pos_rows_count((rows / 256) * 256),
			final_pos_cols_count((cols / 256) * 256) {}

		friend class matrix<double, false, true>;
		friend class matrix<double, true, false>;
		friend class matrix<double, true, true>;
		friend class matrix<double, false, false>;

		friend class vector<uint64_t>;

		~matrix() { if constexpr (call_destructor) delete[] this->_data; }

		size_t rows() { return this->_rows; };

		size_t cols() { return this->_cols; };

		uint8_t* data() { return this->_data; };

		matrix<bool, this_transposed, !this_transposed, false> row(size_t row)
		{
			if constexpr (this_transposed)
			{
				return matrix<bool, true, false, false>(
					&this->_data[row],
					1,
					this->_cols,
					this->actual_rows,
					this->actual_cols);
			}
			else
			{
				return matrix<bool, false, true, false>(
					&this->_data[row * this->actual_cols],
					1,
					this->_cols,
					this->actual_rows,
					this->actual_cols);
			}
		}

		matrix<bool, this_transposed, this_transposed, false> col(size_t col)
		{
			if constexpr (this_transposed)
			{
				return matrix<bool, true, true, false>(
					&this->_data[col * this->actual_rows],
					1,
					this->_cols,
					this->actual_rows,
					this->actual_cols);
			}
			else
			{
				return matrix<bool, false, false, false>(
					&this->_data[col],
					1,
					this->_cols,
					this->actual_rows,
					this->actual_cols);
			}
		}

		matrix<bool, !this_transposed, this_contiguous, false> tranpose()
		{
			if constexpr (this_transposed)
			{
				return matrix<bool, false, this_contiguous, false>(
					this->_data,
					this->_rows,
					this->_cols,
					this->actual_cols,
					this->actual_rows
				);
			}
			else
			{
				return matrix<bool, true, this_contiguous>(
					this->_data,
					this->_rows,
					this->_cols,
					this->actual_rows,
					this->actual_cols
				);
			}
		}

		template<bool block_contiguous = false>
		matrix<bool, this_transposed, this_contiguous, false> block(size_t initial_row, size_t initial_col, size_t final_row, size_t final_col)
		{
			if constexpr (this_transposed)
			{
				return matrix<bool, true, this_contiguous && block_contiguous, false>(
					&this->_data[initial_col * this->actual_rows + initial_row],
					final_row - initial_row,
					final_col - initial_col,
					final_row - initial_row,
					final_col - initial_col
				);
			}
			else
			{
				return matrix<bool, false, this_contiguous && block_contiguous>(
					&this->_data[initial_row * this->actual_cols + initial_col],
					final_row - initial_row,
					final_col - initial_col,
					final_row - initial_row,
					final_col - initial_col
				);
			}
		}

		template<bool other_transposed, bool other_contiguous>
		friend std::ostream& operator<<(std::ostream& os, const matrix<bool, other_transposed, other_contiguous>& matrix)
		{
			if constexpr (other_transposed)
			{
				for (size_t i = 0; i < matrix._rows; i++)
				{
					for (size_t j = 0; j < matrix._cols; j++)
					{
						std::cout << this->[j * matrix.actual_rows + i] << " ";
					}
					std::cout << std::endl;
				}
			}
			else
			{
				for (size_t i = 0; i < matrix._rows; i++)
				{
					for (size_t j = 0; j < matrix._cols; j++)
					{
						std::cout << this->[i * matrix.actual_cols + j] << " ";
					}
					std::cout << std::endl;
				}
			}
			return os;
		}

		uint8_t& operator()(size_t row, size_t col)
		{
			if constexpr (this_transposed)
			{
				return this->_data[col * this->actual_rows + row]
			}
			else
			{
				return this->_data[row * this->actual_cols + col]
			}
		}

		const uint8_t& operator()(size_t row, size_t col) const
		{
			if constexpr (this_transposed)
			{
				return this->_data[col * this->actual_rows + row]
			}
			else
			{
				return this->_data[row * this->actual_cols + col]
			}
		}

		template<bool return_transposed = false, bool other_transposed, bool other_contiguous, bool call_destructor>
		matrix<bool, return_transposed, true> operator&&(matrix<bool, other_transposed, other_contiguous, call_destructor>& other)
		{
#ifdef _DEBUG
			if (other._cols != this->_cols || other._rows != this->_rows) throw std::invalid_argument("The dimensions of both matrices must be the same");
#else
#endif

			size_t rows = this->_rows;
			size_t cols = this->_cols;

			uint8_t* data1 = this->_data;
			uint8_t* data2 = other._data;

			matrix<bool, return_transposed, true> result(rows, cols);

			uint8_t* data_result = result._data;

			if constexpr (this_transposed)
			{
				if constexpr (other_transposed)
				{
					if constexpr (return_transposed)
					{
						if constexpr (this_contiguous && other_contiguous)
						{
							size_t size = this->_size;

							size_t final_pos_size = this->final_pos_size;

							for (size_t i = 0; i < final_pos_size; i += 32)
							{
								__m256i a = _mm256_loadu_epi8(&data1[i]);
								__m256i b = _mm256_loadu_epi8(&data2[i]);

								_mm256_storeu_epi8(&data_result[i], _mm256_and_si256(a, b));
							}
							for (size_t i = final_pos_size; i < size; i++)
							{
								data_result[i] = data1[i] & data2[i];
							}
						}
						else
						{
							size_t rows1 = this->actual_rows;
							size_t rows2 = other.actual_rows;

							size_t final_pos_rows = this->final_pos_rows;

							for (size_t i = 0; i < final_pos_rows; i += 32)
							{
								for (size_t j = 0; j < cols; j++)
								{
									__m256i a = _mm256_loadu_epi8(&data1[j * rows1 + i]);
									__m256i b = _mm256_loadu_epi8(&data2[j * rows2 + i]);

									_mm256_storeu_epi8(&data_result[j * rows + i], _mm256_and_si256(a, b));
								}
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								for (size_t j = 0; j < cols; j++)
								{
									data_result[j * rows + i] = data1[j * rows1 + i] & data2[j * rows2 + i];
								}
							}
						}
					}
					else
					{
						size_t rows1 = this->actual_rows;
						size_t rows2 = other.actual_rows;

						for (size_t i = 0; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								data_result[i * cols + j] = data1[j * rows1 + i] & data2[j * rows2 + i];
							}
						}
					}
				}
				else
				{
					if constexpr (return_transposed)
					{
						size_t rows1 = this->actual_rows;
						size_t cols2 = other.actual_cols;

						for (size_t i = 0; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								data_result[j * rows + i] = data1[j * rows1 + i] & data2[i * cols2 + j];
							}
						}
					}
					else
					{
						size_t rows1 = this->actual_rows;
						size_t cols2 = other.actual_cols;

						for (size_t j = 0; j < cols; j++)
						{
							for (size_t i = 0; i < rows; i++)
							{
								data_result[i * cols + j] = data1[j * rows1 + i] & data2[i * cols2 + j];
							}
						}
					}
				}
			}
			else
			{
				if constexpr (other_transposed)
				{
					if constexpr (return_transposed)
					{
						size_t cols1 = this->actual_cols;
						size_t rows2 = other.actual_rows;

						for (size_t i = 0; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								data_result[j * rows + i] = data1[i * cols1 + j] & data2[j * rows2 + i];
							}
						}
					}
					else
					{
						size_t cols1 = this->actual_cols;
						size_t rows2 = other.actual_rows;

						for (size_t i = 0; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								data_result[i * cols + j] = data1[i * cols1 + j] & data2[j * rows2 + i];
							}
						}
					}
				}
				else
				{
					if constexpr (return_transposed)
					{
						size_t cols1 = this->actual_cols;
						size_t cols2 = other.actual_cols;

						for (size_t i = 0; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								data_result[j * rows + i] = data1[i * cols1 + j] & data2[i * cols2 + j];
							}
						}
					}
					else
					{
						if constexpr (this_contiguous && other_contiguous)
						{
							size_t size = this->_size;

							size_t final_pos_size = this->final_pos_size;

							for (size_t i = 0; i < final_pos_size; i += 32)
							{
								__m256i a = _mm256_loadu_epi8(&data1[i]);
								__m256i b = _mm256_loadu_epi8(&data2[i]);

								_mm256_storeu_epi8(&data_result[i], _mm256_and_si256(a, b));
							}
							for (size_t i = final_pos_size; i < size; i++)
							{
								data_result[i] = data1[i] & data2[i];
							}
						}
						else
						{
							size_t cols1 = this->actual_cols;
							size_t cols2 = other.actual_cols;

							size_t final_pos_cols = this->final_pos_cols;

							for (size_t j = 0; j < final_pos_cols; j += 4)
							{
								for (size_t i = 0; i < rows; i++)
								{
									__m256i a = _mm256_loadu_epi8(&data1[i * cols1 + j]);
									__m256i b = _mm256_loadu_epi8(&data2[i * cols2 + j]);

									_mm256_storeu_epi8(&data_result[i * cols + j], _mm256_and_si256(a, b));
								}
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								for (size_t i = 0; i < rows; i++)
								{
									data_result[i * cols + j] = data1[i * cols1 + j] & data2[i * cols2 + j];
								}
							}
						}
					}
				}
			}
			return result;
		}

		template<bool return_transposed = false, bool other_transposed, bool other_contiguous, bool call_destructor>
		matrix<bool, return_transposed, true> operator||(matrix<bool, other_transposed, other_contiguous, call_destructor>& other)
		{
#ifdef _DEBUG
			if (other._cols != this->_cols || other._rows != this->_rows) throw std::invalid_argument("The dimensions of both matrices must be the same");
#else
#endif

			size_t rows = this->_rows;
			size_t cols = this->_cols;

			uint8_t* data1 = this->_data;
			uint8_t* data2 = other._data;

			matrix<bool, return_transposed, true> result(rows, cols);

			uint8_t* data_result = result._data;

			if constexpr (this_transposed)
			{
				if constexpr (other_transposed)
				{
					if constexpr (return_transposed)
					{
						if constexpr (this_contiguous || other_contiguous)
						{
							size_t size = this->_size;

							size_t final_pos_size = this->final_pos_size;

							for (size_t i = 0; i < final_pos_size; i += 32)
							{
								__m256i a = _mm256_loadu_epi8(&data1[i]);
								__m256i b = _mm256_loadu_epi8(&data2[i]);

								_mm256_storeu_epi8(&data_result[i], _mm256_or_si256(a, b));
							}
							for (size_t i = final_pos_size; i < size; i++)
							{
								data_result[i] = data1[i] | data2[i];
							}
						}
						else
						{
							size_t rows1 = this->actual_rows;
							size_t rows2 = other.actual_rows;

							size_t final_pos_rows = this->final_pos_rows;

							for (size_t i = 0; i < final_pos_rows; i += 32)
							{
								for (size_t j = 0; j < cols; j++)
								{
									__m256i a = _mm256_loadu_epi8(&data1[j * rows1 + i]);
									__m256i b = _mm256_loadu_epi8(&data2[j * rows2 + i]);

									_mm256_storeu_epi8(&data_result[j * rows + i], _mm256_or_si256(a, b));
								}
							}
							for (size_t i = final_pos_rows; i < rows; i++)
							{
								for (size_t j = 0; j < cols; j++)
								{
									data_result[j * rows + i] = data1[j * rows1 + i] | data2[j * rows2 + i];
								}
							}
						}
					}
					else
					{
						size_t rows1 = this->actual_rows;
						size_t rows2 = other.actual_rows;

						for (size_t i = 0; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								data_result[i * cols + j] = data1[j * rows1 + i] | data2[j * rows2 + i];
							}
						}
					}
				}
				else
				{
					if constexpr (return_transposed)
					{
						size_t rows1 = this->actual_rows;
						size_t cols2 = other.actual_cols;

						for (size_t i = 0; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								data_result[j * rows + i] = data1[j * rows1 + i] | data2[i * cols2 + j];
							}
						}
					}
					else
					{
						size_t rows1 = this->actual_rows;
						size_t cols2 = other.actual_cols;

						for (size_t j = 0; j < cols; j++)
						{
							for (size_t i = 0; i < rows; i++)
							{
								data_result[i * cols + j] = data1[j * rows1 + i] | data2[i * cols2 + j];
							}
						}
					}
				}
			}
			else
			{
				if constexpr (other_transposed)
				{
					if constexpr (return_transposed)
					{
						size_t cols1 = this->actual_cols;
						size_t rows2 = other.actual_rows;

						for (size_t i = 0; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								data_result[j * rows + i] = data1[i * cols1 + j] | data2[j * rows2 + i];
							}
						}
					}
					else
					{
						size_t cols1 = this->actual_cols;
						size_t rows2 = other.actual_rows;

						for (size_t i = 0; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								data_result[i * cols + j] = data1[i * cols1 + j] | data2[j * rows2 + i];
							}
						}
					}
				}
				else
				{
					if constexpr (return_transposed)
					{
						size_t cols1 = this->actual_cols;
						size_t cols2 = other.actual_cols;

						for (size_t i = 0; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								data_result[j * rows + i] = data1[i * cols1 + j] | data2[i * cols2 + j];
							}
						}
					}
					else
					{
						if constexpr (this_contiguous || other_contiguous)
						{
							size_t size = this->_size;

							size_t final_pos_size = this->final_pos_size;

							for (size_t i = 0; i < final_pos_size; i += 32)
							{
								__m256i a = _mm256_loadu_epi8(&data1[i]);
								__m256i b = _mm256_loadu_epi8(&data2[i]);

								_mm256_storeu_epi8(&data_result[i], _mm256_or_si256(a, b));
							}
							for (size_t i = final_pos_size; i < size; i++)
							{
								data_result[i] = data1[i] | data2[i];
							}
						}
						else
						{
							size_t cols1 = this->actual_cols;
							size_t cols2 = other.actual_cols;

							size_t final_pos_cols = this->final_pos_cols;

							for (size_t j = 0; j < final_pos_cols; j += 4)
							{
								for (size_t i = 0; i < rows; i++)
								{
									__m256i a = _mm256_loadu_epi8(&data1[i * cols1 + j]);
									__m256i b = _mm256_loadu_epi8(&data2[i * cols2 + j]);

									_mm256_storeu_epi8(&data_result[i * cols + j], _mm256_or_si256(a, b));
								}
							}
							for (size_t j = final_pos_cols; j < cols; j++)
							{
								for (size_t i = 0; i < rows; i++)
								{
									data_result[i * cols + j] = data1[i * cols1 + j] | data2[i * cols2 + j];
								}
							}
						}
					}
				}
			}
			return result;
		}

		template<bool return_transposed = false>
		matrix<bool, return_transposed, true> operator!()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			uint8_t* data1 = this->_data;

			matrix<bool, return_transposed, true> result(rows, cols);

			uint8_t* data_result = result._data;

			__m256d b = _mm256_set1_epi64x(-1);

			if constexpr (this_transposed)
			{
				if constexpr (return_transposed)
				{
					if constexpr (this_contiguous)
					{
						size_t size = this->_size;
						size_t final_pos_size = this->final_pos_size;

						for (size_t i = 0; i < final_pos_size; i += 32)
						{
							__m256d a = _mm256_loadu_epi8(&data1[i]);
							_mm256_storeu_epi8(&data_result[i], _mm256_andnot_si256(a, b));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = ~data1[i];
						}
					}
					else
					{
						size_t rows1 = this->actual_rows;

						size_t final_pos_rows = this->final_pos_rows;

						for (size_t i = 0; i < final_pos_rows; i += 32)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_loadu_epi8(&data1[j * rows1 + i]);
								_mm256_storeu_epi8(&data_result[j * rows + i], _mm256_andnot_si256(a, b));
							}
						}
						for (size_t i = final_pos_rows; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								data_result[j * rows + i] = ~data1[j * rows1 + i];
							}
						}
					}
				}
				else
				{
					size_t rows1 = this->actual_rows;

					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							data_result[i * cols + j] = ~data1[j * rows1 + i];
						}
					}
				}
			}
			else
			{
				if constexpr (return_transposed)
				{
					size_t cols1 = this->actual_cols;

					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							data_result[j * rows + i] = ~data1[i * cols1 + j];
						}
					}
				}
				else
				{
					if constexpr (this_contiguous)
					{
						size_t size = this->_size;
						size_t final_pos_size = this->final_pos_size;

						for (size_t i = 0; i < final_pos_size; i += 32)
						{
							__m256d a = _mm256_loadu_epi8(&data1[i]);
							_mm256_storeu_epi8(&data_result[i], _mm256_andnot_si256(a, b));
						}
						for (size_t i = final_pos_size; i < size; i++)
						{
							data_result[i] = ~data1[i];
						}
					}
					else
					{
						size_t cols1 = this->actual_rows;

						size_t final_pos_cols = this->final_pos_cols;

						for (size_t j = 0; j < final_pos_cols; j += 32)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_loadu_epi8(&data1[i * cols1 + j]);
								_mm256_storeu_epi8(&data_result[i * cols + j], _mm256_andnot_si256(a, b));
							}
						}
						for (size_t j = final_pos_cols; j < cols; j++)
						{
							for (size_t i = 0; i < rows; i++)
							{
								data_result[i * cols + j] = ~data1[i * cols1 + j];
							}
						}
					}
				}
			}
			return result;
		}

		void self_not()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			uint8_t* data1 = this->_data;

			__m256d b = _mm256_set1_epi64x(-1);

			if constexpr (this_contiguous)
			{
				size_t size = this->_size;
				size_t final_pos_size = this->final_pos_size;

				for (size_t i = 0; i < final_pos_size; i += 32)
				{
					__m256d a = _mm256_loadu_epi8(&data1[i]);
					_mm256_storeu_epi8(&data1[i], _mm256_andnot_si256(a, b));
				}
				for (size_t i = final_pos_size; i < size; i++)
				{
					data1[i] = ~data1[i];
				}
			}
			else if constexpr (this_transposed)
			{
				size_t rows1 = this->actual_rows;

				size_t final_pos_rows = this->final_pos_rows;

				for (size_t i = 0; i < final_pos_rows; i += 32)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_loadu_epi8(&data1[j * rows1 + i]);
						_mm256_storeu_epi8(&data1[j * rows1 + i], _mm256_andnot_si256(a, b));
					}
				}
				for (size_t i = final_pos_rows; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						data1[j * rows1 + i] = ~data1[j * rows1 + i];
					}
				}
			}
			else
			{
				size_t cols1 = this->actual_rows;

				size_t final_pos_cols = this->final_pos_cols;

				for (size_t j = 0; j < final_pos_cols; j += 32)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_loadu_epi8(&data1[i * cols1 + j]);
						_mm256_storeu_epi8(&data1[i * cols1 + j], _mm256_andnot_si256(a, b));
					}
				}
				for (size_t j = final_pos_cols; j < cols; j++)
				{
					for (size_t i = 0; i < rows; i++)
					{
						data1[i * cols1 + j] = ~data1[i * cols1 + j];
					}
				}
			}
		}

		size_t count_all()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			uint8_t* data1 = this->_data;

			size_t count = 0;

			int masks[8];

			__m256i mask1 = _mm256_set1_epi32(0x55555555);
			__m256i mask2 = _mm256_set1_epi32(0x33333333);
			__m256i mask3 = _mm256_set1_epi32(0x0F0F0F0F);
			__m256i mask4 = _mm256_set1_epi32(0x00FF00FF);
			__m256i mask5 = _mm256_set1_epi32(0x0000FFFF);

			if constexpr (this_contiguous)
			{
				size_t size = this->_size;

				size_t final_pos_size_count = this->final_pos_size_count;

				for (size_t i = 0; i < final_pos_size_count; i += 256)
				{
					masks[0] = _mm256_movemask_epi8(_mm256_loadu_epi8(data1));
					data1 += 32;
					masks[1] = _mm256_movemask_epi8(_mm256_loadu_epi8(data1));
					data1 += 32;
					masks[2] = _mm256_movemask_epi8(_mm256_loadu_epi8(data1));
					data1 += 32;
					masks[3] = _mm256_movemask_epi8(_mm256_loadu_epi8(data1));
					data1 += 32;
					masks[4] = _mm256_movemask_epi8(_mm256_loadu_epi8(data1));
					data1 += 32;
					masks[5] = _mm256_movemask_epi8(_mm256_loadu_epi8(data1));
					data1 += 32;
					masks[6] = _mm256_movemask_epi8(_mm256_loadu_epi8(data1));
					data1 += 32;
					masks[7] = _mm256_movemask_epi8(_mm256_loadu_epi8(data1));
					data1 += 32;

					__m256i masks_reg = _mm256_loadu_epi32(masks);

					//Get the number of bits that are 1

					masks_reg = _mm256_add_epi32(_mm256_and_si256(masks_reg, mask1), _mm256_and_si256(_mm256_srli_epi32(masks_reg, 1), mask1));
					masks_reg = _mm256_add_epi32(_mm256_and_si256(masks_reg, mask2), _mm256_and_si256(_mm256_srli_epi32(masks_reg, 2), mask2));
					masks_reg = _mm256_add_epi32(_mm256_and_si256(masks_reg, mask3), _mm256_and_si256(_mm256_srli_epi32(masks_reg, 4), mask3));
					masks_reg = _mm256_add_epi32(_mm256_and_si256(masks_reg, mask4), _mm256_and_si256(_mm256_srli_epi32(masks_reg, 8), mask4));
					masks_reg = _mm256_add_epi32(_mm256_srli_epi32(masks_reg, 16), _mm256_and_si256(masks_reg, mask5));

					__m256i a_hi = _mm256_permute2x128_si256(masks_reg, masks_reg, 1);
					masks_reg = _mm256_hadd_epi32(masks_reg, a_hi);
					masks_reg = _mm256_hadd_epi32(masks_reg, masks_reg);
					masks_reg = _mm256_hadd_epi32(masks_reg, masks_reg);

					count = _mm_cvtsi128_si32(_mm256_castsi256_si128(masks_reg));
				}
				for (size_t i = final_pos_size_count; i < size; i++)
				{
					if (*data1) count++;
					data1++;
				}
			}
			else if constexpr (this_transposed)
			{
				size_t rows1 = this->actual_rows;

				size_t extra_rows = rows1 - rows;

				size_t final_pos_rows_count = this->final_pos_rows_count;

				for (size_t j = 0; j < cols; j++)
				{
					for (size_t i = 0; i < final_pos_rows_count; i += 256)
					{
						masks[0] = _mm256_movemask_epi8(_mm256_loadu_epi8(data1));
						data1 += 32;
						masks[1] = _mm256_movemask_epi8(_mm256_loadu_epi8(data1));
						data1 += 32;
						masks[2] = _mm256_movemask_epi8(_mm256_loadu_epi8(data1));
						data1 += 32;
						masks[3] = _mm256_movemask_epi8(_mm256_loadu_epi8(data1));
						data1 += 32;
						masks[4] = _mm256_movemask_epi8(_mm256_loadu_epi8(data1));
						data1 += 32;
						masks[5] = _mm256_movemask_epi8(_mm256_loadu_epi8(data1));
						data1 += 32;
						masks[6] = _mm256_movemask_epi8(_mm256_loadu_epi8(data1));
						data1 += 32;
						masks[7] = _mm256_movemask_epi8(_mm256_loadu_epi8(data1));
						data1 += 32;

						__m256i masks_reg = _mm256_loadu_epi32(masks);

						//Get the number of bits that are 1

						masks_reg = _mm256_add_epi32(_mm256_and_si256(masks_reg, mask1), _mm256_and_si256(_mm256_srli_epi32(masks_reg, 1), mask1));
						masks_reg = _mm256_add_epi32(_mm256_and_si256(masks_reg, mask2), _mm256_and_si256(_mm256_srli_epi32(masks_reg, 2), mask2));
						masks_reg = _mm256_add_epi32(_mm256_and_si256(masks_reg, mask3), _mm256_and_si256(_mm256_srli_epi32(masks_reg, 4), mask3));
						masks_reg = _mm256_add_epi32(_mm256_and_si256(masks_reg, mask4), _mm256_and_si256(_mm256_srli_epi32(masks_reg, 8), mask4));
						masks_reg = _mm256_add_epi32(_mm256_srli_epi32(masks_reg, 16), _mm256_and_si256(masks_reg, mask5));

						__m256i a_hi = _mm256_permute2x128_si256(masks_reg, masks_reg, 1);
						masks_reg = _mm256_hadd_epi32(masks_reg, a_hi);
						masks_reg = _mm256_hadd_epi32(masks_reg, masks_reg);
						masks_reg = _mm256_hadd_epi32(masks_reg, masks_reg);

						count = _mm_cvtsi128_si32(_mm256_castsi256_si128(masks_reg));
					}
					for (size_t i = final_pos_rows_count; i < rows; i++)
					{
						if (*data1) count++;
						data1++;
					}
					data1 += extra_rows;
				}
			}
			else
			{
				size_t cols1 = this->actual_cols;

				size_t extra_cols = cols1 - cols;

				size_t final_pos_cols_count = this->final_pos_cols_count;

				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; i < final_pos_cols_count; i += 256)
					{
						masks[0] = _mm256_movemask_epi8(_mm256_loadu_epi8(data1));
						data1 += 32;
						masks[1] = _mm256_movemask_epi8(_mm256_loadu_epi8(data1));
						data1 += 32;
						masks[2] = _mm256_movemask_epi8(_mm256_loadu_epi8(data1));
						data1 += 32;
						masks[3] = _mm256_movemask_epi8(_mm256_loadu_epi8(data1));
						data1 += 32;
						masks[4] = _mm256_movemask_epi8(_mm256_loadu_epi8(data1));
						data1 += 32;
						masks[5] = _mm256_movemask_epi8(_mm256_loadu_epi8(data1));
						data1 += 32;
						masks[6] = _mm256_movemask_epi8(_mm256_loadu_epi8(data1));
						data1 += 32;
						masks[7] = _mm256_movemask_epi8(_mm256_loadu_epi8(data1));
						data1 += 32;

						__m256i masks_reg = _mm256_loadu_epi32(masks);

						//Get the number of bits that are 1

						masks_reg = _mm256_add_epi32(_mm256_and_si256(masks_reg, mask1), _mm256_and_si256(_mm256_srli_epi32(masks_reg, 1), mask1));
						masks_reg = _mm256_add_epi32(_mm256_and_si256(masks_reg, mask2), _mm256_and_si256(_mm256_srli_epi32(masks_reg, 2), mask2));
						masks_reg = _mm256_add_epi32(_mm256_and_si256(masks_reg, mask3), _mm256_and_si256(_mm256_srli_epi32(masks_reg, 4), mask3));
						masks_reg = _mm256_add_epi32(_mm256_and_si256(masks_reg, mask4), _mm256_and_si256(_mm256_srli_epi32(masks_reg, 8), mask4));
						masks_reg = _mm256_add_epi32(_mm256_srli_epi32(masks_reg, 16), _mm256_and_si256(masks_reg, mask5));

						__m256i a_hi = _mm256_permute2x128_si256(masks_reg, masks_reg, 1);
						masks_reg = _mm256_hadd_epi32(masks_reg, a_hi);
						masks_reg = _mm256_hadd_epi32(masks_reg, masks_reg);
						masks_reg = _mm256_hadd_epi32(masks_reg, masks_reg);

						count = _mm_cvtsi128_si32(_mm256_castsi256_si128(masks_reg));
					}
					for (size_t j = final_pos_cols_count; j < cols; j++)
					{
						if (*data1) count++;
						data1++;
					}
					data1 += extra_cols;
				}
			}
		}

		vector<uint64_t> count_colwise()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			uint8_t* data1 = this->_data;

			vector<uint64_t> result(cols);

			uint64_t* data_result = result._data;

			if constexpr (this_transposed)
			{
				int masks[8];

				__m256i mask1 = _mm256_set1_epi32(0x55555555);
				__m256i mask2 = _mm256_set1_epi32(0x33333333);
				__m256i mask3 = _mm256_set1_epi32(0x0F0F0F0F);
				__m256i mask4 = _mm256_set1_epi32(0x00FF00FF);
				__m256i mask5 = _mm256_set1_epi32(0x0000FFFF);

				size_t rows1 = this->actual_rows;

				size_t extra_rows = rows1 - rows;

				size_t final_pos_rows_count = this->final_pos_rows_count;

				for (size_t j = 0; j < cols; j++)
				{
					size_t count = 0;
					for (size_t i = 0; i < final_pos_rows_count; i += 256)
					{
						masks[0] = _mm256_movemask_epi8(_mm256_loadu_epi8(data1));
						data1 += 32;
						masks[1] = _mm256_movemask_epi8(_mm256_loadu_epi8(data1));
						data1 += 32;
						masks[2] = _mm256_movemask_epi8(_mm256_loadu_epi8(data1));
						data1 += 32;
						masks[3] = _mm256_movemask_epi8(_mm256_loadu_epi8(data1));
						data1 += 32;
						masks[4] = _mm256_movemask_epi8(_mm256_loadu_epi8(data1));
						data1 += 32;
						masks[5] = _mm256_movemask_epi8(_mm256_loadu_epi8(data1));
						data1 += 32;
						masks[6] = _mm256_movemask_epi8(_mm256_loadu_epi8(data1));
						data1 += 32;
						masks[7] = _mm256_movemask_epi8(_mm256_loadu_epi8(data1));
						data1 += 32;

						__m256i masks_reg = _mm256_loadu_epi32(masks);

						//Get the number of bits that are 1

						masks_reg = _mm256_add_epi32(_mm256_and_si256(masks_reg, mask1), _mm256_and_si256(_mm256_srli_epi32(masks_reg, 1), mask1));
						masks_reg = _mm256_add_epi32(_mm256_and_si256(masks_reg, mask2), _mm256_and_si256(_mm256_srli_epi32(masks_reg, 2), mask2));
						masks_reg = _mm256_add_epi32(_mm256_and_si256(masks_reg, mask3), _mm256_and_si256(_mm256_srli_epi32(masks_reg, 4), mask3));
						masks_reg = _mm256_add_epi32(_mm256_and_si256(masks_reg, mask4), _mm256_and_si256(_mm256_srli_epi32(masks_reg, 8), mask4));
						masks_reg = _mm256_add_epi32(_mm256_srli_epi32(masks_reg, 16), _mm256_and_si256(masks_reg, mask5));

						__m256i a_hi = _mm256_permute2x128_si256(masks_reg, masks_reg, 1);
						masks_reg = _mm256_hadd_epi32(masks_reg, a_hi);
						masks_reg = _mm256_hadd_epi32(masks_reg, masks_reg);
						masks_reg = _mm256_hadd_epi32(masks_reg, masks_reg);

						count = _mm_cvtsi128_si32(_mm256_castsi256_si128(masks_reg));
					}
					for (size_t i = final_pos_rows_count; i < rows; i++)
					{
						if (*data1) count++;
					}
					data_result[j] = count;
					data1 += extra_rows;
				}
			}
			else
			{
				size_t cols1 = this->actual_cols;

				for (size_t j = 0; j < cols; j++)
				{
					size_t count = 0;
					for (size_t i = 0; i < rows; i++)
					{
						if (data1[i * cols1 + j]) count++;
					}
					data_result[j] = count;
				}
			}
			return result;
		}

		vector<uint64_t> count_rowwise()
		{
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			uint8_t* data1 = this->_data;

			vector<uint64_t> result(rows);

			uint64_t* data_result = result._data;

			if constexpr (this_transposed)
			{
				size_t rows1 = this->actual_rows;

				for (size_t i = 0; i < rows; i++)
				{
					size_t count = 0;
					for (size_t j = 0; j < cols; j++)
					{
						if (data1[j * rows1 + i]) count++;
					}
					data_result[i] = count;
				}
			}
			else
			{
				int masks[8];

				__m256i mask1 = _mm256_set1_epi32(0x55555555);
				__m256i mask2 = _mm256_set1_epi32(0x33333333);
				__m256i mask3 = _mm256_set1_epi32(0x0F0F0F0F);
				__m256i mask4 = _mm256_set1_epi32(0x00FF00FF);
				__m256i mask5 = _mm256_set1_epi32(0x0000FFFF);

				size_t cols1 = this->actual_cols;

				size_t extra_cols = cols1 - cols;

				size_t final_pos_cols_count = this->final_pos_cols_count;

				for (size_t i = 0; i < rows; i++)
				{
					size_t count = 0;
					for (size_t j = 0; j < final_pos_cols_count; j += 256)
					{
						masks[0] = _mm256_movemask_epi8(_mm256_loadu_epi8(data1));
						data1 += 32;
						masks[1] = _mm256_movemask_epi8(_mm256_loadu_epi8(data1));
						data1 += 32;
						masks[2] = _mm256_movemask_epi8(_mm256_loadu_epi8(data1));
						data1 += 32;
						masks[3] = _mm256_movemask_epi8(_mm256_loadu_epi8(data1));
						data1 += 32;
						masks[4] = _mm256_movemask_epi8(_mm256_loadu_epi8(data1));
						data1 += 32;
						masks[5] = _mm256_movemask_epi8(_mm256_loadu_epi8(data1));
						data1 += 32;
						masks[6] = _mm256_movemask_epi8(_mm256_loadu_epi8(data1));
						data1 += 32;
						masks[7] = _mm256_movemask_epi8(_mm256_loadu_epi8(data1));
						data1 += 32;

						__m256i masks_reg = _mm256_loadu_epi32(masks);

						//Get the number of bits that are 1

						masks_reg = _mm256_add_epi32(_mm256_and_si256(masks_reg, mask1), _mm256_and_si256(_mm256_srli_epi32(masks_reg, 1), mask1));
						masks_reg = _mm256_add_epi32(_mm256_and_si256(masks_reg, mask2), _mm256_and_si256(_mm256_srli_epi32(masks_reg, 2), mask2));
						masks_reg = _mm256_add_epi32(_mm256_and_si256(masks_reg, mask3), _mm256_and_si256(_mm256_srli_epi32(masks_reg, 4), mask3));
						masks_reg = _mm256_add_epi32(_mm256_and_si256(masks_reg, mask4), _mm256_and_si256(_mm256_srli_epi32(masks_reg, 8), mask4));
						masks_reg = _mm256_add_epi32(_mm256_srli_epi32(masks_reg, 16), _mm256_and_si256(masks_reg, mask5));

						__m256i a_hi = _mm256_permute2x128_si256(masks_reg, masks_reg, 1);
						masks_reg = _mm256_hadd_epi32(masks_reg, a_hi);
						masks_reg = _mm256_hadd_epi32(masks_reg, masks_reg);
						masks_reg = _mm256_hadd_epi32(masks_reg, masks_reg);

						count = _mm_cvtsi128_si32(_mm256_castsi256_si128(masks_reg));
					}
					for (size_t j = final_pos_cols_count; j < cols; j++)
					{
						if (*data1) count++;
						data1++;
					}
					data_result[i] = count;
					data1 += extra_cols;
				}
			}
			return result;
		}

	private:

		uint8_t* _data;
		size_t _rows, _cols, _size;
		size_t actual_rows, actual_cols;
		size_t final_pos_size, final_pos_rows, final_pos_cols;
		size_t final_pos_size_count, final_pos_rows_count, final_pos_cols_count;
	};

//---------------------------------------------------------------------------

	

}
