#include "vectorDouble.h"

namespace alge
{
	// Constructor

	inline vector<double>::vector() :
		_data(nullptr),
		dataToDelete(nullptr),
		_size(0),
		finalPos(0),
		_capacity(0) {}

	inline vector<double>::vector(size_t size) :
		_data(new double[size]),
		dataToDelete(_data),
		_size(size),
		finalPos((size / 4) * 4),
		_capacity(size) {}

	inline vector<double>::vector(double* data, size_t size) :
		_data(data),
		dataToDelete(nullptr),
		_size(size),
		finalPos((size / 4) * 4),
		_capacity(size) {}

	inline vector<double>::vector(std::initializer_list<double> list)
	{
		this->_size = list.size();
		this->finalPos = (this->_size / 4) * 4;
		this->_data = new double[this->_size];
		this->dataToDelete = this->_data;
		this->_capacity = this->_size;

		for (size_t i = 0; i < this->_size; i++)
		{
			this->_data[i] = *(list.begin() + i);
		}
	}

	// Destructor

	inline vector<double>::~vector() { delete[] this->dataToDelete; }

	// Block

	inline vector<double> vector<double>::block(size_t initial, size_t final)
	{
		return vector<double>(
			&this->_data[initial],
			final - initial
		);
	}

	// Copy

	inline vector<double> vector<double>::copy()
	{
		size_t size = this->_size;

		vector<double> result;

		double* data1 = this->_data;

		double* dataResult = result._data;

		for (size_t i = 0; i < size; i++)
		{
			dataResult[i] = data1[i];
		}
		return result;
	}

	// =

	inline vector<double>& vector<double>::operator=(vector<double>& other)
	{
		if (this->_data == nullptr)
		{
			this->_data = other._data;
			other.dataToDelete = nullptr;
			this->dataToDelete = this->_data;
			this->_size = other._size;
			this->finalPos = other.finalPos;
		}
		else
		{
#ifdef _DEBUG
			if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
			size_t size = this->_size;
			double* data1 = this->_data;
			double* data2 = other._data;

			for (size_t i = 0; i < size; i++)
			{
				data1[i] = data2[i];
			}
		}
		return *this;
	}

	// Transfer

	inline void vector<double>::transfer(vector<double>& other)
	{
#ifdef _DEBUG
		if (other.dataToDelete == nullptr || (this->dataToDelete == nullptr && this->_data != nullptr)) throw std::invalid_argument("Error");
#else
#endif
		delete[] this->_data;

		this->_data = other._data;
		this->dataToDelete = other._data;
		other.dataToDelete = nullptr;
		this->_size = other._size;
		this->finalPos = other.finalPos;
	}

	// Set Constant

	inline void vector<double>::set_const(double num)
	{
		size_t size = this->_size;

		double* data1 = this->_data;

		for (size_t i = 0; i < size; i++)
		{
			data1[i] = num;
		}

	}

	inline double& vector<double>::operator[](size_t index)
	{
		double* data = this->_data;
		return data[index];
	}

	inline const double& vector<double>::operator[](size_t index) const
	{
		double* data = this->_data;
		return data[index];
	}

	inline vector<double> vector<double>::operator[](vector<uint64_t>& indices)
	{
		size_t size = indices._size;

		vector<double> result(size);

		double* data1 = this->_data;

		uint64_t* dataIndices = indices._data;

		double* dataResult = result._data;

		for (size_t i = 0; i < size; i++)
		{
			dataResult[i] = data1[dataIndices[i]];
		}
		return result;
	}

	inline double* vector<double>::data() { return this->_data; }

	inline size_t vector<double>::size() { return this->_size; }

	inline size_t vector<double>::capacity() { return this->_capacity; }

	template<bool reduceCapacity>
	inline void vector<double>::clear()
	{
		if constexpr (reduceCapacity)
		{
			this->_size = 0;
			this->_capacity = 0;
			this->finalPos = 0;
			delete[] this->dataToDelete;
			this->_data = nullptr;
			this->dataToDelete = nullptr;
		}
		else
		{
			this->_size = 0;
			this->_capacity = 0;
			this->finalPos = 0;
		}
	}

	inline void vector<double>::reserve(size_t newCapacity)
	{
		double* newData = new double[newCapacity];
		double* oldData = this->_data;

		this->_size = this->_size <= newCapacity ? this->_size : newCapacity;
		this->finalPos = (this->_size / 4) * 4;
		this->_capacity = newCapacity;
		for (size_t i = 0; i < this->_size; i++)
		{
			newData[i] = oldData[i];
		}
		delete[] this->dataToDelete;
		this->_data = newData;
		this->dataToDelete = newData;
	}

	inline void vector<double>::append(double num)
	{
		if (this->_capacity > this->_size)
		{
			this->_data[this->_size] = num;
			this->_size++;
		}
		else
		{
			size_t increase = this->_capacity / 2;
			increase = increase > 0 ? increase : 1;
			this->_capacity = this->_capacity + increase;
			double* newData = new double[this->_capacity];
			double* oldData = this->_data;
			for (size_t i = 0; i < this->_size; i++)
			{
				newData[i] = oldData[i];
			}
			newData[this->_size] = num;

			delete[] this->dataToDelete;
			this->_data = newData;
			this->dataToDelete = newData;
			this->_size++;
		}
		this->finalPos = (this->_size / 4) * 4;
	}

	inline void vector<double>::append(std::initializer_list<double> list)
	{
		size_t newSize = this->_size + list.size();
		if (this->_capacity >= newSize)
		{
			for (size_t i{ this->_size }, j{ 0 }; i < newSize; i++, j++)
			{
				this->_data[i] = *(list.begin() + j);
			}
			this->_size = newSize;
		}
		else
		{
			size_t sizeList = list.size();
			size_t increase = this->_capacity / 2;
			increase = increase >= sizeList ? increase : sizeList;

			this->_capacity = this->_capacity + increase;
			double* newData = new double[this->_capacity];
			double* oldData = this->_data;

			for (size_t i = 0; i < this->_size; i++)
			{
				newData[i] = oldData[i];
			}

			for (size_t i{ this->_size }, j{ 0 }; i < newSize; i++, j++)
			{
				newData[i] = *(list.begin() + j);
			}
			delete[] this->dataToDelete;
			this->_data = newData;
			this->dataToDelete = newData;
			this->_size = newSize;
		}
		this->finalPos = (this->_size / 4) * 4;
	}

	inline void vector<double>::append(vector<double>& other)
	{
		size_t newSize = this->_size + other._size;

		if (this->_capacity >= newSize)
		{
			for (size_t i{ this->_size }, j{ 0 }; i < newSize; i++, j++)
			{
				this->_data[i] = other._data[j];
			}
			this->_size = newSize;
		}
		else
		{
			size_t increase = this->_capacity / 2;
			increase = increase >= other._size ? increase : other._size;

			this->_capacity = this->_capacity + increase;
			double* newData = new double[this->_capacity];
			double* oldData = this->_data;

			for (size_t i = 0; i < this->_size; i++)
			{
				newData[i] = oldData[i];
			}

			for (size_t i{ this->_size }, j{ 0 }; i < newSize; i++, j++)
			{
				newData[i] = other._data[j];
			}
			delete[] this->dataToDelete;
			this->_data = newData;
			this->dataToDelete = newData;
			this->_size = newSize;
		}
		this->finalPos = (this->_size / 4) * 4;
	}

	inline void vector<double>::insert(double num, size_t index)
	{
		if (this->_capacity > this->_size)
		{
			double* data1 = this->_data;

			double tmp = num;
			double tmp2;
			for (size_t i = index; i < this->_size; i++)
			{
				tmp2 = data1[i];
				data1[i] = tmp;
				tmp = tmp2;
			}
			data1[index] = num;
			this->_size++;
		}
		else
		{
			size_t increase = this->_capacity / 2;
			increase = increase > 0 ? increase : 1;
			this->_capacity += increase;
			double* newData = new double[this->_capacity];
			double* oldData = this->_data;

			for (size_t i = 0; i < index; i++)
			{
				newData[i] = oldData[i];
			}
			for (size_t i = index; i < this->_size; i++)
			{
				newData[i + 1] = oldData[i];
			}
			newData[index] = num;

			delete[] this->dataToDelete;
			this->_data = newData;
			this->dataToDelete = newData;
			this->_size++;
		}
		this->finalPos = (this->_size / 4) * 4;
	}

	inline void vector<double>::erase(size_t index)
	{
		double* data1 = this->_data;
		this->_size--;
		this->finalPos = (this->_size / 4) * 4;
		if (this->dataToDelete == nullptr)
		{
			double* newData = new double[this->_size];

			for (size_t i = 0; i < index; i++)
			{
				newData[i] = data1[i];
			}
			for (size_t i = index; i < this->_size; i++)
			{
				newData[i] = data1[i + 1];
			}
			this->_data = newData;
			this->dataToDelete = newData;
		}
		else
		{
			for (size_t i = index; i < this->_size; i++)
			{
				data1[i] = data1[i + 1];
			}
		}
	}

	template<bool binarySearch>
	inline size_t vector<double>::find(double num)
	{
		if constexpr (binarySearch)
		{
			size_t left = 0;
			size_t right = this->_size - 1;

			while (left <= right)
			{
				size_t mid = left + (right - left) / 2;

				if (this->_data[mid] == num)
				{
					return mid;
				}
				else if (this->_data[mid] < num)
				{
					left = mid + 1;
				}
				else
				{
					right = mid - 1;
				}
			}
			return this->_size;
		}
		else
		{
			for (size_t i = 0; i < this->_size; i++)
			{
				if (this->_data[i] == num) return i;
			}
			return this->_size;
		}
	}

	// Rand

	inline void vector<double>::rand()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		__m256i random;

		masks_uint64_to_double;

		__m256d divisor = _mm256_set1_pd(18446744073709551615.0);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			random = _mm256_slli_epi64(__seeds__, 13);
			__seeds__ = _mm256_xor_si256(random, __seeds__);

			random = _mm256_srli_epi64(__seeds__, 10);
			__seeds__ = _mm256_xor_si256(random, __seeds__);

			random = _mm256_slli_epi64(__seeds__, 20);
			__seeds__ = _mm256_xor_si256(random, __seeds__);

			// uint64 to double

			uint64_to_double(__seeds__);

			_mm256_store_pd(&data1[i], _mm256_div_pd(uint64ToDouble, divisor));
		}

		for (size_t i = finalPos; i < size; i++)
		{
			random = _mm256_slli_epi64(__seeds__, 13);
			__seeds__ = _mm256_xor_si256(random, __seeds__);

			random = _mm256_srli_epi64(__seeds__, 10);
			__seeds__ = _mm256_xor_si256(random, __seeds__);

			random = _mm256_slli_epi64(__seeds__, 20);
			__seeds__ = _mm256_xor_si256(random, __seeds__);

			// uint64 to double

			uint64_to_double(__seeds__);

			_mm_store_sd(&data1[i], _mm256_castpd256_pd128(_mm256_div_pd(uint64ToDouble, divisor)));
		}
	}

	// neg

	inline vector<double> vector<double>::operator-()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		__m256d mask = _mm256_set1_pd(-0.0);

		vector<double> result(size);

		double* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&dataResult[i], _mm256_xor_pd(_mm256_load_pd(&data1[i]), mask));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = -data1[i];
		}
		return result;
	}

	inline void vector<double>::self_neg()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		__m256d mask = _mm256_set1_pd(-0.0);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&data1[i], _mm256_xor_pd(_mm256_load_pd(&data1[i]), mask));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = -data1[i];
		}
	}

	// +

	inline vector<double> vector<double>::operator+(vector<double>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;
		double* data2 = other._data;

		vector<double> result(size);

		double* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_load_pd(&data1[i]);
			__m256d b = _mm256_load_pd(&data2[i]);

			_mm256_store_pd(&dataResult[i], _mm256_add_pd(a, b));
		}

		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] + data2[i];
		}
		return result;
	}

	inline vector<double> vector<double>::operator+(double num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		vector<double> result(size);

		double* dataResult = result._data;

		__m256d b = _mm256_set1_pd(num);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_load_pd(&data1[i]);

			_mm256_store_pd(&dataResult[i], _mm256_add_pd(a, b));
		}

		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] + num;
		}
		return result;
	}


	inline void vector<double>::operator+=(vector<double>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;
		double* data2 = other._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_load_pd(&data1[i]);
			__m256d b = _mm256_load_pd(&data2[i]);

			_mm256_store_pd(&data1[i], _mm256_add_pd(a, b));
		}

		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = data1[i] + data2[i];
		}
	}

	inline void vector<double>::operator+=(double num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		__m256d b = _mm256_set1_pd(num);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_load_pd(&data1[i]);

			_mm256_store_pd(&data1[i], _mm256_add_pd(a, b));
		}

		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = data1[i] + num;
		}
	}

	// -


	inline vector<double> vector<double>::operator-(vector<double>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;
		double* data2 = other._data;

		vector<double> result(size);

		double* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_load_pd(&data1[i]);
			__m256d b = _mm256_load_pd(&data2[i]);

			_mm256_store_pd(&dataResult[i], _mm256_sub_pd(a, b));
		}

		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] - data2[i];
		}
		return result;
	}

	inline vector<double> vector<double>::operator-(double num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		vector<double> result(size);

		double* dataResult = result._data;

		__m256d b = _mm256_set1_pd(num);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_load_pd(&data1[i]);

			_mm256_store_pd(&dataResult[i], _mm256_sub_pd(a, b));
		}

		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] - num;
		}
		return result;
	}


	inline void vector<double>::operator-=(vector<double>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;
		double* data2 = other._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_load_pd(&data1[i]);
			__m256d b = _mm256_load_pd(&data2[i]);

			_mm256_store_pd(&data1[i], _mm256_sub_pd(a, b));
		}

		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = data1[i] - data2[i];
		}
	}

	inline void vector<double>::operator-=(double num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		__m256d b = _mm256_set1_pd(num);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_load_pd(&data1[i]);

			_mm256_store_pd(&data1[i], _mm256_sub_pd(a, b));
		}

		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = data1[i] - num;
		}
	}

	// *


	inline vector<double> vector<double>::operator*(vector<double>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;
		double* data2 = other._data;

		vector<double> result(size);

		double* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_load_pd(&data1[i]);
			__m256d b = _mm256_load_pd(&data2[i]);

			_mm256_store_pd(&dataResult[i], _mm256_mul_pd(a, b));
		}

		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] * data2[i];
		}
		return result;
	}

	inline vector<double> vector<double>::operator*(double num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		vector<double> result(size);

		double* dataResult = result._data;

		__m256d b = _mm256_set1_pd(num);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_load_pd(&data1[i]);

			_mm256_store_pd(&dataResult[i], _mm256_mul_pd(a, b));
		}

		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] * num;
		}
		return result;
	}


	inline void vector<double>::operator*=(vector<double>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;
		double* data2 = other._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_load_pd(&data1[i]);
			__m256d b = _mm256_load_pd(&data2[i]);

			_mm256_store_pd(&data1[i], _mm256_mul_pd(a, b));
		}

		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = data1[i] * data2[i];
		}
	}

	inline void vector<double>::operator*=(double num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		__m256d b = _mm256_set1_pd(num);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_load_pd(&data1[i]);

			_mm256_store_pd(&data1[i], _mm256_mul_pd(a, b));
		}

		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = data1[i] * num;
		}
	}

	// /


	inline vector<double> vector<double>::operator/(vector<double>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;
		double* data2 = other._data;

		vector<double> result(size);

		double* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_load_pd(&data1[i]);
			__m256d b = _mm256_load_pd(&data2[i]);

			_mm256_store_pd(&dataResult[i], _mm256_div_pd(a, b));
		}

		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] / data2[i];
		}
		return result;
	}

	inline vector<double> vector<double>::operator/(double num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		vector<double> result(size);

		double* dataResult = result._data;

		__m256d b = _mm256_set1_pd(num);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_load_pd(&data1[i]);

			_mm256_store_pd(&dataResult[i], _mm256_div_pd(a, b));
		}

		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] / num;
		}
		return result;
	}


	inline void vector<double>::operator/=(vector<double>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;
		double* data2 = other._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_load_pd(&data1[i]);
			__m256d b = _mm256_load_pd(&data2[i]);

			_mm256_store_pd(&data1[i], _mm256_div_pd(a, b));
		}

		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = data1[i] / data2[i];
		}
	}

	inline void vector<double>::operator/=(double num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		__m256d b = _mm256_set1_pd(num);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_load_pd(&data1[i]);

			_mm256_store_pd(&data1[i], _mm256_div_pd(a, b));
		}

		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = data1[i] / num;
		}
	}

	// ==

	inline vector<uint8_t> vector<double>::operator==(vector<double>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;
		double* data2 = other._data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_load_pd(&data1[i]);
			__m256d b = _mm256_load_pd(&data2[i]);

			__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_EQ_OQ));
			__m128i mask1 = _mm256_castsi256_si128(mask);
			__m128i mask2 = _mm256_extracti128_si256(mask, 1);

			mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
			mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

			__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

			_mm_store_ss(reinterpret_cast<float*>(&dataResult[i]), _mm_castsi128_ps(maskResult));
		}

		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] == data2[i] ? True : False;
		}
		return result;
	}

	inline vector<uint8_t> vector<double>::operator==(double num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		__m256d b = _mm256_set1_pd(num);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_load_pd(&data1[i]);

			__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_EQ_OQ));
			__m128i mask1 = _mm256_castsi256_si128(mask);
			__m128i mask2 = _mm256_extracti128_si256(mask, 1);

			mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
			mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

			__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

			_mm_store_ss(reinterpret_cast<float*>(&dataResult[i]), _mm_castsi128_ps(maskResult));
		}

		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] == num ? True : False;
		}
		return result;
	}

	// !=

	inline vector<uint8_t> vector<double>::operator!=(vector<double>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;
		double* data2 = other._data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_load_pd(&data1[i]);
			__m256d b = _mm256_load_pd(&data2[i]);

			__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_NEQ_OQ));
			__m128i mask1 = _mm256_castsi256_si128(mask);
			__m128i mask2 = _mm256_extracti128_si256(mask, 1);

			mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
			mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

			__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

			_mm_store_ss(reinterpret_cast<float*>(&dataResult[i]), _mm_castsi128_ps(maskResult));
		}

		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] != data2[i] ? True : False;
		}
		return result;
	}

	inline vector<uint8_t> vector<double>::operator!=(double num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		__m256d b = _mm256_set1_pd(num);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_load_pd(&data1[i]);

			__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_NEQ_OQ));
			__m128i mask1 = _mm256_castsi256_si128(mask);
			__m128i mask2 = _mm256_extracti128_si256(mask, 1);

			mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
			mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

			__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

			_mm_store_ss(reinterpret_cast<float*>(&dataResult[i]), _mm_castsi128_ps(maskResult));
		}

		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] != num ? True : False;
		}
		return result;
	}

	// >

	inline vector<uint8_t> vector<double>::operator>(vector<double>& other)
	{
#ifdef _DEBUG
		if (this->_size > other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;
		double* data2 = other._data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_load_pd(&data1[i]);
			__m256d b = _mm256_load_pd(&data2[i]);

			__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GT_OQ));
			__m128i mask1 = _mm256_castsi256_si128(mask);
			__m128i mask2 = _mm256_extracti128_si256(mask, 1);

			mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
			mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

			__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

			_mm_store_ss(reinterpret_cast<float*>(&dataResult[i]), _mm_castsi128_ps(maskResult));
		}

		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] > data2[i] ? True : False;
		}
		return result;
	}

	inline vector<uint8_t> vector<double>::operator>(double num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		__m256d b = _mm256_set1_pd(num);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_load_pd(&data1[i]);

			__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GT_OQ));
			__m128i mask1 = _mm256_castsi256_si128(mask);
			__m128i mask2 = _mm256_extracti128_si256(mask, 1);

			mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
			mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

			__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

			_mm_store_ss(reinterpret_cast<float*>(&dataResult[i]), _mm_castsi128_ps(maskResult));
		}

		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] > num ? True : False;
		}
		return result;
	}

	// >=


	inline vector<uint8_t> vector<double>::operator>=(vector<double>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;
		double* data2 = other._data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_load_pd(&data1[i]);
			__m256d b = _mm256_load_pd(&data2[i]);

			__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GE_OQ));
			__m128i mask1 = _mm256_castsi256_si128(mask);
			__m128i mask2 = _mm256_extracti128_si256(mask, 1);

			mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
			mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

			__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

			_mm_store_ss(reinterpret_cast<float*>(&dataResult[i]), _mm_castsi128_ps(maskResult));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] >= data2[i] ? True : False;
		}
		return result;
	}

	inline vector<uint8_t> vector<double>::operator>=(double num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		__m256d b = _mm256_set1_pd(num);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_load_pd(&data1[i]);

			__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GE_OQ));
			__m128i mask1 = _mm256_castsi256_si128(mask);
			__m128i mask2 = _mm256_extracti128_si256(mask, 1);

			mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
			mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

			__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

			_mm_store_ss(reinterpret_cast<float*>(&dataResult[i]), _mm_castsi128_ps(maskResult));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] >= num ? True : False;
		}
		return result;
	}

	// <


	inline vector<uint8_t> vector<double>::operator<(vector<double>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;
		double* data2 = other._data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_load_pd(&data1[i]);
			__m256d b = _mm256_load_pd(&data2[i]);

			__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LT_OQ));
			__m128i mask1 = _mm256_castsi256_si128(mask);
			__m128i mask2 = _mm256_extracti128_si256(mask, 1);

			mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
			mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

			__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

			_mm_store_ss(reinterpret_cast<float*>(&dataResult[i]), _mm_castsi128_ps(maskResult));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] < data2[i] ? True : False;
		}
		return result;
	}

	inline vector<uint8_t> vector<double>::operator<(double num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		__m256d b = _mm256_set1_pd(num);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_load_pd(&data1[i]);

			__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LT_OQ));
			__m128i mask1 = _mm256_castsi256_si128(mask);
			__m128i mask2 = _mm256_extracti128_si256(mask, 1);

			mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
			mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

			__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

			_mm_store_ss(reinterpret_cast<float*>(&dataResult[i]), _mm_castsi128_ps(maskResult));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] < num ? True : False;
		}
		return result;
	}

	// <=


	inline vector<uint8_t> vector<double>::operator<=(vector<double>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;
		double* data2 = other._data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_load_pd(&data1[i]);
			__m256d b = _mm256_load_pd(&data2[i]);

			__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LE_OQ));
			__m128i mask1 = _mm256_castsi256_si128(mask);
			__m128i mask2 = _mm256_extracti128_si256(mask, 1);

			mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
			mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

			__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

			_mm_store_ss(reinterpret_cast<float*>(&dataResult[i]), _mm_castsi128_ps(maskResult));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] <= data2[i] ? True : False;
		}
		return result;
	}

	inline vector<uint8_t> vector<double>::operator<=(double num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		__m256d b = _mm256_set1_pd(num);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_load_pd(&data1[i]);

			__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LE_OQ));
			__m128i mask1 = _mm256_castsi256_si128(mask);
			__m128i mask2 = _mm256_extracti128_si256(mask, 1);

			mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
			mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

			__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

			_mm_store_ss(reinterpret_cast<float*>(&dataResult[i]), _mm_castsi128_ps(maskResult));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] <= num ? True : False;
		}
		return result;
	}

	// Functions

	// Pow

	inline vector<double> vector<double>::pow(double exponent)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		vector<double> result(size);

		double* dataResult = result._data;

		__m256d _exponet = _mm256_set1_pd(exponent);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&dataResult[i], _mm256_pow_pd(_mm256_load_pd(&data1[i]), _exponet));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::pow(data1[i], exponent);
		}
		return result;
	}

	inline vector<double> vector<double>::pow(vector<double>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		double* data2 = other._data;

		vector<double> result(size);

		double* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&dataResult[i], _mm256_pow_pd(_mm256_load_pd(&data1[i]), _mm256_load_pd(&data2[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::pow(data1[i], data2[i]);
		}
		return result;
	}

	inline void vector<double>::self_pow(double exponent)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		__m256d _exponet = _mm256_set1_pd(exponent);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&data1[i], _mm256_pow_pd(_mm256_load_pd(&data1[i]), _exponet));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::pow(data1[i], exponent);
		}
	}

	inline void vector<double>::self_pow(vector<double>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		double* data2 = other._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&data1[i], _mm256_pow_pd(_mm256_load_pd(&data1[i]), _mm256_load_pd(&data2[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::pow(data1[i], data2[i]);
		}
	}

	// Root

	inline vector<double> vector<double>::root(double index)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		vector<double> result(size);

		double* dataResult = result._data;

		index = 1 / index;

		__m256d _index = _mm256_set1_pd(index);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&dataResult[i], _mm256_pow_pd(_mm256_load_pd(&data1[i]), _index));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::pow(data1[i], index);
		}
		return result;
	}

	inline vector<double> vector<double>::root(vector<double>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		double* data2 = other._data;

		vector<double> result(size);

		double* dataResult = result._data;

		__m256d ones = _mm256_set1_pd(1.0);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&dataResult[i], _mm256_pow_pd(_mm256_load_pd(&data1[i]), _mm256_div_pd(ones, _mm256_load_pd(&data2[i]))));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::pow(data1[i], 1 / data2[i]);
		}
		return result;
	}

	inline void vector<double>::self_root(double index)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		index = 1 / index;

		__m256d _index = _mm256_set1_pd(index);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&data1[i], _mm256_pow_pd(_mm256_load_pd(&data1[i]), _index));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::pow(data1[i], index);
		}
	}

	inline void vector<double>::self_root(vector<double>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		double* data2 = other._data;

		__m256d ones = _mm256_set1_pd(1.0);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&data1[i], _mm256_pow_pd(_mm256_load_pd(&data1[i]), _mm256_div_pd(ones, _mm256_load_pd(&data2[i]))));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::pow(data1[i], 1 / data2[i]);
		}
	}

	// Log

	inline vector<double> vector<double>::log()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		vector<double> result(size);

		double* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&dataResult[i], _mm256_log_pd(_mm256_load_pd(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::log(data1[i]);
		}
		return result;
	}

	inline void vector<double>::self_log()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&data1[i], _mm256_log_pd(_mm256_load_pd(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::log(data1[i]);
		}
	}

	// Log2

	inline vector<double> vector<double>::log2()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		vector<double> result(size);

		double* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&dataResult[i], _mm256_log2_pd(_mm256_load_pd(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::log2(data1[i]);
		}
		return result;
	}

	inline void vector<double>::self_log2()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&data1[i], _mm256_log2_pd(_mm256_load_pd(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::log2(data1[i]);
		}
	}

	// Log10

	inline vector<double> vector<double>::log10()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		vector<double> result(size);

		double* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&dataResult[i], _mm256_log10_pd(_mm256_load_pd(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::log10(data1[i]);
		}
		return result;
	}

	inline void vector<double>::self_log10()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&data1[i], _mm256_log10_pd(_mm256_load_pd(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::log10(data1[i]);
		}
	}

	// Exp

	inline vector<double> vector<double>::exp()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		vector<double> result(size);

		double* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&dataResult[i], _mm256_exp_pd(_mm256_load_pd(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::exp(data1[i]);
		}
		return result;
	}

	inline void vector<double>::self_exp()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&data1[i], _mm256_exp_pd(_mm256_load_pd(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::exp(data1[i]);
		}
	}

	// Exp2

	inline vector<double> vector<double>::exp2()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		vector<double> result(size);

		double* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&dataResult[i], _mm256_exp2_pd(_mm256_load_pd(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::exp2(data1[i]);
		}
		return result;
	}

	inline void vector<double>::self_exp2()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&data1[i], _mm256_exp2_pd(_mm256_load_pd(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::exp2(data1[i]);
		}
	}

	// Tan

	inline vector<double> vector<double>::tan()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		vector<double> result(size);

		double* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&dataResult[i], _mm256_tan_pd(_mm256_load_pd(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::tan(data1[i]);
		}
		return result;
	}

	inline void vector<double>::self_tan()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&data1[i], _mm256_tan_pd(_mm256_load_pd(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::tan(data1[i]);
		}
	}

	// Cos

	inline vector<double> vector<double>::cos()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		vector<double> result(size);

		double* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&dataResult[i], _mm256_cos_pd(_mm256_load_pd(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::cos(data1[i]);
		}
		return result;
	}

	inline void vector<double>::self_cos()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&data1[i], _mm256_cos_pd(_mm256_load_pd(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::cos(data1[i]);
		}
	}

	// Acos

	inline vector<double> vector<double>::acos()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		vector<double> result(size);

		double* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&dataResult[i], _mm256_acos_pd(_mm256_load_pd(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::acos(data1[i]);
		}
		return result;
	}

	inline void vector<double>::self_acos()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&data1[i], _mm256_acos_pd(_mm256_load_pd(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::acos(data1[i]);
		}
	}

	// Atan

	inline vector<double> vector<double>::atan()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		vector<double> result(size);

		double* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&dataResult[i], _mm256_atan_pd(_mm256_load_pd(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::atan(data1[i]);
		}
		return result;
	}

	inline void vector<double>::self_atan()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&data1[i], _mm256_atan_pd(_mm256_load_pd(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::atan(data1[i]);
		}
	}

	// Abs

	inline vector<double> vector<double>::abs()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		vector<double> result(size);

		double* dataResult = result._data;

		__m256d mask = _mm256_set1_pd(-0.0);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&dataResult[i], _mm256_andnot_pd(mask, _mm256_load_pd(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::fabs(data1[i]);
		}
		return result;
	}

	inline void vector<double>::self_abs()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		__m256d mask = _mm256_set1_pd(-0.0);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&data1[i], _mm256_andnot_pd(mask, _mm256_load_pd(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::fabs(data1[i]);
		}
	}

	// Round

	inline vector<double> vector<double>::round()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		vector<double> result(size);

		double* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&dataResult[i], _mm256_round_pd(_mm256_load_pd(&data1[i]), _MM_FROUND_TO_NEAREST_INT));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::round(data1[i]);
		}
		return result;
	}

	inline void vector<double>::self_round()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&data1[i], _mm256_round_pd(_mm256_load_pd(&data1[i]), _MM_FROUND_TO_NEAREST_INT));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::round(data1[i]);
		}
	}

	// Floor

	inline vector<double> vector<double>::floor()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		vector<double> result(size);

		double* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&dataResult[i], _mm256_floor_pd(_mm256_load_pd(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::floor(data1[i]);
		}
		return result;
	}

	inline void vector<double>::self_floor()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&data1[i], _mm256_floor_pd(_mm256_load_pd(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::floor(data1[i]);
		}
	}

	// Ceil

	inline vector<double> vector<double>::ceil()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		vector<double> result(size);

		double* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&dataResult[i], _mm256_ceil_pd(_mm256_load_pd(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::ceil(data1[i]);
		}
		return result;
	}

	inline void vector<double>::self_ceil()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&data1[i], _mm256_ceil_pd(_mm256_load_pd(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::ceil(data1[i]);
		}
	}

	// Max

	inline double vector<double>::max()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		__m256d _max = _mm256_set1_pd(DBL_MIN);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_max = _mm256_max_pd(_max, _mm256_load_pd(&data1[i]));
		}

		__m128d val1 = _mm256_castpd256_pd128(_max);

		__m128d val2 = _mm256_extractf128_pd(_max, 1);

		val1 = _mm_max_pd(val1, val2);

		val2 = _mm_permute_pd(val1, 0b01);

		val1 = _mm_max_pd(val1, val2);

		double max = _mm_cvtsd_f64(val1);

		for (size_t i = finalPos; i < size; i++)
		{
			double data = data1[i];
			if (data > max) max = data;
		}
		return max;
	}

	inline uint64_t vector<double>::argmax()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		__m256d _max = _mm256_set1_pd(DBL_MIN);

		double max = DBL_MIN;
		uint64_t indice = 0;

		__m256i indices = _mm256_setr_epi64x(0, 1, 2, 3);

		__m256i maxIndices = _mm256_setr_epi64x(0, 1, 2, 3);

		__m256i four = _mm256_set1_epi64x(4);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_load_pd(&data1[i]);

			__m256d mask = _mm256_cmp_pd(a, _max, _CMP_GT_OQ);

			maxIndices = _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(maxIndices), _mm256_castsi256_pd(indices), mask));

			_max = _mm256_blendv_pd(_max, a, mask);

			indices = _mm256_add_epi64(indices, four);
		}
		uint64_t maxIndicesArr[4];
		double maxArr[4];

		_mm256_storeu_epi64(maxIndicesArr, maxIndices);
		_mm256_store_pd(maxArr, _max);

		for (size_t i = 0; i < 4; i++)
		{
			double data = maxArr[i];
			if (data > max)
			{
				max = data;
				indice = maxIndicesArr[i];
			}
		}
		for (size_t i = finalPos; i < size; i++)
		{
			double data = data1[i];
			if (data > max)
			{
				max = data;
				indice = i;
			}
		}
		return indice;
	}

	// Min

	inline double vector<double>::min()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		__m256d _min = _mm256_set1_pd(DBL_MAX);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_min = _mm256_min_pd(_min, _mm256_load_pd(&data1[i]));
		}

		__m256d tempmin = _mm256_permute2f128_pd(_min, _min, 0x01);
		_min = _mm256_min_pd(_min, tempmin);

		__m128d low = _mm256_castpd256_pd128(_min);
		__m128d high = _mm256_extractf128_pd(_min, 1);

		low = _mm_min_pd(low, high);
		double min = _mm_cvtsd_f64(low);

		for (size_t i = finalPos; i < size; i++)
		{
			double data = data1[i];
			if (data > min) min = data;
		}
		return min;
	}

	inline uint64_t vector<double>::argmin()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		__m256d _min = _mm256_set1_pd(DBL_MAX);

		double min = DBL_MAX;
		uint64_t indice = 0;

		__m256i indices = _mm256_setr_epi64x(0, 1, 2, 3);

		__m256i minIndices = _mm256_setr_epi64x(0, 1, 2, 3);

		__m256i four = _mm256_set1_epi64x(4);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_load_pd(&data1[i]);

			__m256d mask = _mm256_cmp_pd(a, _min, _CMP_LT_OQ);

			minIndices = _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(minIndices), _mm256_castsi256_pd(indices), mask));

			_min = _mm256_blendv_pd(_min, a, mask);

			indices = _mm256_add_epi64(indices, four);
		}
		uint64_t minIndicesArr[4];
		double minArr[4];

		_mm256_storeu_epi64(minIndicesArr, minIndices);
		_mm256_store_pd(minArr, _min);

		for (size_t i = 0; i < 4; i++)
		{
			double data = minArr[i];
			if (data < min)
			{
				min = data;
				indice = minIndicesArr[i];
			}
		}
		for (size_t i = finalPos; i < size; i++)
		{
			double data = data1[i];
			if (data < min)
			{
				min = data;
				indice = i;
			}
		}
		return indice;
	}

	// Sum

	inline double vector<double>::sum()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		__m256d _sum = _mm256_setzero_pd();

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_sum = _mm256_add_pd(_sum, _mm256_load_pd(&data1[i]));
		}


		__m128d vlow = _mm256_castpd256_pd128(_sum);
		__m128d vhigh = _mm256_extractf128_pd(_sum, 1);
		vlow = _mm_add_pd(vlow, vhigh);

		__m128d high64 = _mm_unpackhi_pd(vlow, vlow);
		double sum = _mm_cvtsd_f64(_mm_add_sd(vlow, high64));

		for (size_t i = finalPos; i < size; i++)
		{
			sum += data1[i];
		}

		return sum;
	}

	// Mean

	inline double vector<double>::mean()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		__m256d _sum = _mm256_setzero_pd();

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_sum = _mm256_add_pd(_sum, _mm256_load_pd(&data1[i]));
		}


		__m128d vlow = _mm256_castpd256_pd128(_sum);
		__m128d vhigh = _mm256_extractf128_pd(_sum, 1);
		vlow = _mm_add_pd(vlow, vhigh);

		__m128d high64 = _mm_unpackhi_pd(vlow, vlow);
		double sum = _mm_cvtsd_f64(_mm_add_sd(vlow, high64));

		for (size_t i = finalPos; i < size; i++)
		{
			sum += data1[i];
		}

		return sum / static_cast<double>(size);
	}

	// Std

	inline double vector<double>::std(double ddof, double* mean)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		__m256d _sum = _mm256_setzero_pd();
		__m256d _sumSquare = _mm256_setzero_pd();

		double size_d = static_cast<double>(size);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_load_pd(&data1[i]);

			_sum = _mm256_add_pd(_sum, a);

			_sumSquare = _mm256_fmadd_pd(a, a, _sumSquare);
		}
		__m128d vlow = _mm256_castpd256_pd128(_sum);
		__m128d vhigh = _mm256_extractf128_pd(_sum, 1);
		vlow = _mm_add_pd(vlow, vhigh);

		__m128d high64 = _mm_unpackhi_pd(vlow, vlow);
		double sum = _mm_cvtsd_f64(_mm_add_sd(vlow, high64));
		//--
		vlow = _mm256_castpd256_pd128(_sumSquare);
		vhigh = _mm256_extractf128_pd(_sumSquare, 1);
		vlow = _mm_add_pd(vlow, vhigh);

		high64 = _mm_unpackhi_pd(vlow, vlow);
		double sumSquare = _mm_cvtsd_f64(_mm_add_sd(vlow, high64));

		for (size_t i = finalPos; i < size; i++)
		{
			double data = data1[i];
			sum += data;
			sumSquare += data * data;
		}
		if (mean != nullptr) *mean = sum / size_d;

		double variance = (sumSquare - (sum * sum / size_d)) / (size_d - ddof);
		double std = std::sqrt(variance);
		return std;
	}

	// Activation Functions

	// Tanh

	inline vector<double> vector<double>::tanh()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		vector<double> result(size);

		double* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&dataResult[i], _mm256_tanh_pd(_mm256_load_pd(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::tanh(data1[i]);
		}
		return result;
	}

	inline void vector<double>::self_tanh()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&data1[i], _mm256_tanh_pd(_mm256_load_pd(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::tanh(data1[i]);
		}
	}

	// Cosh

	inline vector<double> vector<double>::cosh()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		vector<double> result(size);

		double* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&dataResult[i], _mm256_cosh_pd(_mm256_load_pd(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::cosh(data1[i]);
		}
		return result;
	}

	inline void vector<double>::self_cosh()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&data1[i], _mm256_cosh_pd(_mm256_load_pd(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::cosh(data1[i]);
		}
	}

	// ReLU

	inline vector<double> vector<double>::relu()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		vector<double> result(size);

		double* dataResult = result._data;

		__m256d zero = _mm256_setzero_pd();

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&dataResult[i], _mm256_max_pd(zero, _mm256_load_pd(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::max(0.0, data1[i]);
		}
		return result;
	}

	inline void vector<double>::self_relu()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		__m256d zero = _mm256_setzero_pd();

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&data1[i], _mm256_max_pd(zero, _mm256_load_pd(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::max(0.0, data1[i]);
		}
	}

	// LReLU

	inline vector<double> vector<double>::lrelu()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		vector<double> result(size);

		double* dataResult = result._data;

		__m256d num = _mm256_set1_pd(0.01);

		__m256d zero = _mm256_setzero_pd();

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_load_pd(&data1[i]);

			_mm256_store_pd(&dataResult[i], _mm256_blendv_pd(_mm256_mul_pd(a, num), a, _mm256_cmp_pd(a, zero, _CMP_GT_OQ)));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] > 0.0 ? data1[i] : 0.01 * data1[i];
		}
		return result;
	}

	inline void vector<double>::self_lrelu()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		__m256d num = _mm256_set1_pd(0.01);

		__m256d zero = _mm256_setzero_pd();

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_load_pd(&data1[i]);

			_mm256_store_pd(&data1[i], _mm256_blendv_pd(_mm256_mul_pd(a, num), a, _mm256_cmp_pd(a, zero, _CMP_GT_OQ)));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = data1[i] > 0.0 ? data1[i] : 0.01 * data1[i];
		}
	}

	// Sigmoid

	inline vector<double> vector<double>::sigmoid()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		vector<double> result(size);

		double* dataResult = result._data;

		__m256d mask = _mm256_set1_pd(-0.0);

		__m256d one = _mm256_set1_pd(1.0);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_load_pd(&data1[i]);

			__m256d neg = _mm256_xor_pd(a, mask);

			__m256d sigmoid = _mm256_div_pd(one, _mm256_add_pd(_mm256_exp_pd(neg), one));

			_mm256_store_pd(&dataResult[i], sigmoid);
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = 1.0 / (1.0 + std::exp(-data1[i]));
		}
		return result;
	}

	inline void vector<double>::self_sigmoid()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		__m256d mask = _mm256_set1_pd(-0.0);

		__m256d one = _mm256_set1_pd(1.0);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256d a = _mm256_load_pd(&data1[i]);

			__m256d neg = _mm256_xor_pd(a, mask);

			__m256d sigmoid = _mm256_div_pd(one, _mm256_add_pd(_mm256_exp_pd(neg), one));

			_mm256_store_pd(&data1[i], sigmoid);
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = 1.0 / (1.0 + std::exp(-data1[i]));
		}
	}

	// Softplus

	inline vector<double> vector<double>::softplus()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		vector<double> result(size);

		double* dataResult = result._data;

		__m256d one = _mm256_set1_pd(1.0);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&dataResult[i], _mm256_log_pd(_mm256_add_pd(one, _mm256_exp_pd(_mm256_load_pd(&data1[i])))));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::log(1.0 + std::exp(data1[i]));
		}
		return result;
	}

	inline void vector<double>::self_softplus()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		double* data1 = this->_data;

		__m256d one = _mm256_set1_pd(1.0);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_pd(&data1[i], _mm256_log_pd(_mm256_add_pd(one, _mm256_exp_pd(_mm256_load_pd(&data1[i])))));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::log(1.0 + std::exp(data1[i]));
		}
	}

	// Softmax

	inline vector<double> vector<double>::softmax()
	{
		vector<double> exp = (this->vector<double>::operator-(this->max())).exp();

		return exp / exp.sum();
	}

	// Sort

	inline void vector<double>::sort()
	{
		std::sort(this->_data, this->_data + this->_size);
	}

	// Argsort

	inline vector<uint64_t> vector<double>::argsort()
	{
		vector<uint64_t> indices(this->_size);

		size_t* indicesData = indices._data;

		for (size_t i = 0; i < this->_size; i++) indicesData[i] = i;

		quicksort(indicesData, this->_data, 0, this->_size - 1);

		return indices;
	}

	// Cast

	template <typename T>
	inline vector<T> vector<double>::cast()
	{
		size_t size = this->_size;

		vector<T> result(size);

		double* data1 = this->_data;

		T* dataResult = result._data;

		size_t finalPos = this->finalPos;

		if constexpr (std::is_same<T, uint8_t>::value)
		{
			__m256d zero = _mm256_setzero_pd();

			for (size_t i = 0; i < finalPos; i += 4)
			{
				__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(_mm256_load_pd(&data1[i]), zero, _CMP_NEQ_OQ));

				__m128i mask1 = _mm256_castsi256_si128(mask);
				__m128i mask2 = _mm256_extracti128_si256(mask, 1);

				mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
				mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

				__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

				_mm_store_ss(reinterpret_cast<float*>(&dataResult[i]), _mm_castsi128_ps(maskResult));
			}
			for (size_t i = finalPos; i < size; i++)
			{
				dataResult[i] = data1[i] ? True : False;
			}
		}
		else if constexpr (std::is_same<T, float>::value)
		{
			for (size_t i = 0; i < finalPos; i += 4)
			{
				_mm_store_ps(&dataResult[i], _mm256_cvtpd_ps(_mm256_load_pd(&data1[i])));
			}
			for (size_t i = finalPos; i < size; i++)
			{
				dataResult[i] = static_cast<float>(data1[i]);
			}
		}
		else if constexpr (std::is_same<T, int>::value)
		{
			for (size_t i = 0; i < finalPos; i += 4)
			{
				_mm_storeu_epi32(&dataResult[i], _mm256_cvtpd_epi32(_mm256_load_pd(&data1[i])));
			}
			for (size_t i = finalPos; i < size; i++)
			{
				double data = data1[i];
				data += 6755399441055744.0;
				dataResult[i] = reinterpret_cast<int&>(data);
			}
		}
		else
		{
			for (size_t i = 0; i < size; i++)
			{
				dataResult[i] = static_cast<T>(data1[i]);
			}
		}
		return result;
	}

}
