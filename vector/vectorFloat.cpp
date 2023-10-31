#include "vectorFloat.h"

namespace alge
{
	inline vector<float>::vector() :
		_data(nullptr),
		dataToDelete(nullptr),
		_size(0),
		finalPos(0),
		_capacity(0) {}

	inline vector<float>::vector(size_t size) :
		_data(new float[size]),
		dataToDelete(_data),
		_size(size),
		finalPos((size / 8) * 8),
		_capacity(size) {}

	inline vector<float>::vector(float* data, size_t size) :
		_data(data),
		dataToDelete(nullptr),
		_size(size),
		finalPos((size / 8) * 8),
		_capacity(size) {}

	inline vector<float>::vector(std::initializer_list<float> list)
	{
		this->_size = list.size();
		this->finalPos = (this->_size / 8) * 8;
		this->_data = new float[this->_size];
		this->dataToDelete = this->_data;
		this->_capacity = this->_size;

		for (size_t i = 0; i < this->_size; i++)
		{
			this->_data[i] = *(list.begin() + i);
		}
	}

	inline vector<float>::~vector() { delete[] this->dataToDelete; }

	inline float& vector<float>::operator[](size_t index)
	{
		float* data = this->_data;
		return data[index];
	}

	inline const float& vector<float>::operator[](size_t index) const
	{
		float* data = this->_data;
		return data[index];
	}

	inline vector<float> vector<float>::operator[](vector<uint64_t>& indices)
	{
		size_t size = indices._size;

		vector<float> result(size);

		float* data1 = this->_data;

		uint64_t* dataIndices = indices._data;

		float* dataResult = result._data;

		for (size_t i = 0; i < size; i++)
		{
			dataResult[i] = data1[dataIndices[i]];
		}
		return result;
	}

	inline float* vector<float>::data() { return this->_data; }

	inline size_t vector<float>::capacity() { return this->_capacity; }

	template<bool reduceCapacity>
	inline void vector<float>::clear()
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

	inline void vector<float>::reserve(size_t newCapacity)
	{
		float* newData = new float[newCapacity];
		float* oldData = this->_data;

		this->_size = this->_size <= newCapacity ? this->_size : newCapacity;
		this->finalPos = (this->_size / 8) * 8;
		this->_capacity = newCapacity;
		for (size_t i = 0; i < this->_size; i++)
		{
			newData[i] = oldData[i];
		}
		delete[] this->dataToDelete;
		this->_data = newData;
		this->dataToDelete = newData;
	}

	inline void vector<float>::append(float num)
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
			float* newData = new float[this->_capacity];
			float* oldData = this->_data;
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
		this->finalPos = (this->_size / 8) * 8;
	}

	inline void vector<float>::append(std::initializer_list<float> list)
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
			float* newData = new float[this->_capacity];
			float* oldData = this->_data;

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
		this->finalPos = (this->_size / 8) * 8;
	}

	inline void vector<float>::append(vector<float>& other)
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
			float* newData = new float[this->_capacity];
			float* oldData = this->_data;

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
		this->finalPos = (this->_size / 8) * 8;
	}

	inline void vector<float>::insert(float num, size_t index)
	{
		if (this->_capacity > this->_size)
		{
			float* data1 = this->_data;

			float tmp = num;
			float tmp2;
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
			float* newData = new float[this->_capacity];
			float* oldData = this->_data;

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
		this->finalPos = (this->_size / 8) * 8;
	}

	inline void vector<float>::erase(size_t index)
	{
		float* data1 = this->_data;
		this->_size--;
		this->finalPos = (this->_size / 8) * 8;
		if (this->dataToDelete == nullptr)
		{
			float* newData = new float[this->_size];

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
	inline size_t vector<float>::find(float num)
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

	// Block

	inline vector<float> vector<float>::block(size_t initial, size_t final)
	{
		return vector<float>(
			&this->_data[initial],
			final - initial
		);
	}

	// Copy

	inline vector<float> vector<float>::copy()
	{
		size_t size = this->_size;

		vector<float> result(size);

		float* data1 = this->_data;

		float* dataResult = result._data;

		for (size_t i = 0; i < size; i++)
		{
			dataResult[i] = data1[i];
		}
		return result;
	}

	// =

	inline vector<float>& vector<float>::operator=(vector<float>& other)
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
			float* data1 = this->_data;
			float* data2 = other._data;

			for (size_t i = 0; i < size; i++)
			{
				data1[i] = data2[i];
			}
		}
		return *this;
	}

	inline void vector<float>::transfer(vector<float>& other)
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

	// neg

	inline vector<float> vector<float>::operator-()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		__m256 mask = _mm256_set1_ps(-0.0f);

		vector<float> result(size);

		float* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_store_ps(&dataResult[i], _mm256_xor_ps(_mm256_load_ps(&data1[i]), mask));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = -data1[i];
		}
		return result;
	}

	inline void vector<float>::self_neg()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		__m256 mask = _mm256_set1_ps(-0.0f);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_store_ps(&data1[i], _mm256_xor_ps(_mm256_load_ps(&data1[i]), mask));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = -data1[i];
		}
	}

	// Set Constant

	inline void vector<float>::set_const(float num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		for (size_t i = 0; i < size; i++)
		{
			data1[i] = num;
		}
	}

	// Rand

	inline void vector<float>::rand()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		__m256i random;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			random = _mm256_slli_epi32(__seeds__, 6);
			__seeds__ = _mm256_xor_si256(random, __seeds__);

			random = _mm256_srli_epi32(__seeds__, 5);
			__seeds__ = _mm256_xor_si256(random, __seeds__);

			random = _mm256_slli_epi32(__seeds__, 10);
			__seeds__ = _mm256_xor_si256(random, __seeds__);

			// uint32 to double

			uint32_to_float(__seeds__);

			_mm256_store_ps(&data1[i], uint32ToFloat);
		}
		for (size_t i = finalPos; i < size; i++)
		{
			random = _mm256_slli_epi32(__seeds__, 6);
			__seeds__ = _mm256_xor_si256(random, __seeds__);

			random = _mm256_srli_epi32(__seeds__, 5);
			__seeds__ = _mm256_xor_si256(random, __seeds__);

			random = _mm256_slli_epi64(__seeds__, 10);
			__seeds__ = _mm256_xor_si256(random, __seeds__);

			// uint32 to double

			uint32_to_float(__seeds__);

			_mm_store_ss(&data1[i], _mm256_castps256_ps128(uint32ToFloat));
		}
	}

	// +

	inline vector<float> vector<float>::operator+(vector<float>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;
		float* data2 = other._data;

		vector<float> result(size);

		float* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256 a = _mm256_load_ps(&data1[i]);
			__m256 b = _mm256_load_ps(&data2[i]);

			_mm256_store_ps(&dataResult[i], _mm256_add_ps(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] + data2[i];
		}
		return result;
	}

	inline vector<float> vector<float>::operator+(float num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		vector<float> result(size);

		float* dataResult = result._data;

		__m256 b = _mm256_set1_ps(num);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256 a = _mm256_load_ps(&data1[i]);

			_mm256_store_ps(&dataResult[i], _mm256_add_ps(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] + num;
		}
		return result;
	}

	inline void vector<float>::operator+=(vector<float>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;
		float* data2 = other._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256 a = _mm256_load_ps(&data1[i]);
			__m256 b = _mm256_load_ps(&data2[i]);

			_mm256_store_ps(&data1[i], _mm256_add_ps(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = data1[i] + data2[i];
		}
	}

	inline void vector<float>::operator+=(float num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		__m256 b = _mm256_set1_ps(num);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256 a = _mm256_load_ps(&data1[i]);

			_mm256_store_ps(&data1[i], _mm256_add_ps(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = data1[i] + num;
		}
	}

	// -

	inline vector<float> vector<float>::operator-(vector<float>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;
		float* data2 = other._data;

		vector<float> result(size);

		float* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256 a = _mm256_load_ps(&data1[i]);
			__m256 b = _mm256_load_ps(&data2[i]);

			_mm256_store_ps(&dataResult[i], _mm256_sub_ps(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] - data2[i];
		}
		return result;
	}

	inline vector<float> vector<float>::operator-(float num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		vector<float> result(size);

		float* dataResult = result._data;

		__m256 b = _mm256_set1_ps(num);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256 a = _mm256_load_ps(&data1[i]);

			_mm256_store_ps(&dataResult[i], _mm256_sub_ps(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] - num;
		}
		return result;
	}

	inline void vector<float>::operator-=(vector<float>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;
		float* data2 = other._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256 a = _mm256_load_ps(&data1[i]);
			__m256 b = _mm256_load_ps(&data2[i]);

			_mm256_store_ps(&data1[i], _mm256_sub_ps(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = data1[i] - data2[i];
		}
	}

	inline void vector<float>::operator-=(float num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		__m256 b = _mm256_set1_ps(num);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256 a = _mm256_load_ps(&data1[i]);

			_mm256_store_ps(&data1[i], _mm256_sub_ps(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = data1[i] - num;
		}
	}

	// *

	inline vector<float> vector<float>::operator*(vector<float>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;
		float* data2 = other._data;

		vector<float> result(size);

		float* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256 a = _mm256_load_ps(&data1[i]);
			__m256 b = _mm256_load_ps(&data2[i]);

			_mm256_store_ps(&dataResult[i], _mm256_mul_ps(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] * data2[i];
		}
		return result;
	}

	inline vector<float> vector<float>::operator*(float num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		vector<float> result(size);

		float* dataResult = result._data;

		__m256 b = _mm256_set1_ps(num);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256 a = _mm256_load_ps(&data1[i]);

			_mm256_store_ps(&dataResult[i], _mm256_mul_ps(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] * num;
		}
		return result;
	}

	inline void vector<float>::operator*=(vector<float>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;
		float* data2 = other._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256 a = _mm256_load_ps(&data1[i]);
			__m256 b = _mm256_load_ps(&data2[i]);

			_mm256_store_ps(&data1[i], _mm256_mul_ps(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = data1[i] * data2[i];
		}
	}

	inline void vector<float>::operator*=(float num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		__m256 b = _mm256_set1_ps(num);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256 a = _mm256_load_ps(&data1[i]);

			_mm256_store_ps(&data1[i], _mm256_mul_ps(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = data1[i] * num;
		}
	}

	// /

	inline vector<float> vector<float>::operator/(vector<float>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;
		float* data2 = other._data;

		vector<float> result(size);

		float* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256 a = _mm256_load_ps(&data1[i]);
			__m256 b = _mm256_load_ps(&data2[i]);

			_mm256_store_ps(&dataResult[i], _mm256_div_ps(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] / data2[i];
		}
		return result;
	}

	inline vector<float> vector<float>::operator/(float num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		vector<float> result(size);

		float* dataResult = result._data;

		__m256 b = _mm256_set1_ps(num);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256 a = _mm256_load_ps(&data1[i]);

			_mm256_store_ps(&dataResult[i], _mm256_div_ps(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] / num;
		}
		return result;
	}

	inline void vector<float>::operator/=(vector<float>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;
		float* data2 = other._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256 a = _mm256_load_ps(&data1[i]);
			__m256 b = _mm256_load_ps(&data2[i]);

			_mm256_store_ps(&data1[i], _mm256_div_ps(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = data1[i] / data2[i];
		}
	}

	inline void vector<float>::operator/=(float num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		__m256 b = _mm256_set1_ps(num);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256 a = _mm256_load_ps(&data1[i]);

			_mm256_store_ps(&data1[i], _mm256_div_ps(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = data1[i] / num;
		}
	}

	// ==

	inline vector<uint8_t> vector<float>::operator==(vector<float>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;
		float* data2 = other._data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		__m256i indices = _mm256_setr_epi32(0, 7, 2, 3, 4, 5, 6, 1);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256 a = _mm256_load_ps(&data1[i]);
			__m256 b = _mm256_load_ps(&data2[i]);

			__m256i mask = _mm256_castps_si256(_mm256_cmp_ps(a, b, _CMP_EQ_OQ));
			__m256i mask1 = _mm256_packs_epi32(mask, mask);
			__m256i mask2 = _mm256_packs_epi16(mask1, mask1);

			mask2 = _mm256_permutevar8x32_epi32(mask2, indices);

			_mm_store_sd(reinterpret_cast<double*>(&dataResult[i]), _mm_castsi128_pd(_mm256_castsi256_si128(mask2)));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] == data2[i] ? True : False;
		}
		return result;
	}

	inline vector<uint8_t> vector<float>::operator==(float num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		__m256i indices = _mm256_setr_epi32(0, 7, 2, 3, 4, 5, 6, 1);

		__m256 b = _mm256_set1_ps(num);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256 a = _mm256_load_ps(&data1[i]);

			__m256i mask = _mm256_castps_si256(_mm256_cmp_ps(a, b, _CMP_EQ_OQ));
			__m256i mask1 = _mm256_packs_epi32(mask, mask);
			__m256i mask2 = _mm256_packs_epi16(mask1, mask1);

			mask2 = _mm256_permutevar8x32_epi32(mask2, indices);

			_mm_store_sd(reinterpret_cast<double*>(&dataResult[i]), _mm_castsi128_pd(_mm256_castsi256_si128(mask2)));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] == num ? True : False;
		}
		return result;
	}

	// !=

	inline vector<uint8_t> vector<float>::operator!=(vector<float>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;
		float* data2 = other._data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		__m256i indices = _mm256_setr_epi32(0, 7, 2, 3, 4, 5, 6, 1);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256 a = _mm256_load_ps(&data1[i]);
			__m256 b = _mm256_load_ps(&data2[i]);

			__m256i mask = _mm256_castps_si256(_mm256_cmp_ps(a, b, _CMP_NEQ_OQ));
			__m256i mask1 = _mm256_packs_epi32(mask, mask);
			__m256i mask2 = _mm256_packs_epi16(mask1, mask1);

			mask2 = _mm256_permutevar8x32_epi32(mask2, indices);

			_mm_store_sd(reinterpret_cast<double*>(&dataResult[i]), _mm_castsi128_pd(_mm256_castsi256_si128(mask2)));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] != data2[i] ? True : False;
		}
		return result;
	}

	inline vector<uint8_t> vector<float>::operator!=(float num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		__m256i indices = _mm256_setr_epi32(0, 7, 2, 3, 4, 5, 6, 1);

		__m256 b = _mm256_set1_ps(num);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256 a = _mm256_load_ps(&data1[i]);

			__m256i mask = _mm256_castps_si256(_mm256_cmp_ps(a, b, _CMP_NEQ_OQ));
			__m256i mask1 = _mm256_packs_epi32(mask, mask);
			__m256i mask2 = _mm256_packs_epi16(mask1, mask1);

			mask2 = _mm256_permutevar8x32_epi32(mask2, indices);

			_mm_store_sd(reinterpret_cast<double*>(&dataResult[i]), _mm_castsi128_pd(_mm256_castsi256_si128(mask2)));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] != num ? True : False;
		}
		return result;
	}

	// >

	inline vector<uint8_t> vector<float>::operator>(vector<float>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;
		float* data2 = other._data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256 a = _mm256_load_ps(&data1[i]);
			__m256 b = _mm256_load_ps(&data2[i]);

			__m256i mask = _mm256_castps_si256(_mm256_cmp_ps(a, b, _CMP_GT_OQ));
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

	inline vector<uint8_t> vector<float>::operator>(float num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		__m256 b = _mm256_set1_ps(num);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256 a = _mm256_load_ps(&data1[i]);

			__m256i mask = _mm256_castps_si256(_mm256_cmp_ps(a, b, _CMP_GT_OQ));
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

	inline vector<uint8_t> vector<float>::operator>=(vector<float>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;
		float* data2 = other._data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256 a = _mm256_load_ps(&data1[i]);
			__m256 b = _mm256_load_ps(&data2[i]);

			__m256i mask = _mm256_castps_si256(_mm256_cmp_ps(a, b, _CMP_GE_OQ));
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

	inline vector<uint8_t> vector<float>::operator>=(float num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		__m256 b = _mm256_set1_ps(num);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256 a = _mm256_load_ps(&data1[i]);

			__m256i mask = _mm256_castps_si256(_mm256_cmp_ps(a, b, _CMP_GE_OQ));
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

	inline vector<uint8_t> vector<float>::operator<(vector<float>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;
		float* data2 = other._data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256 a = _mm256_load_ps(&data1[i]);
			__m256 b = _mm256_load_ps(&data2[i]);

			__m256i mask = _mm256_castps_si256(_mm256_cmp_ps(a, b, _CMP_LT_OQ));
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

	inline vector<uint8_t> vector<float>::operator<(float num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		__m256 b = _mm256_set1_ps(num);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256 a = _mm256_load_ps(&data1[i]);

			__m256i mask = _mm256_castps_si256(_mm256_cmp_ps(a, b, _CMP_LT_OQ));
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

	inline vector<uint8_t> vector<float>::operator<=(vector<float>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;
		float* data2 = other._data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256 a = _mm256_load_ps(&data1[i]);
			__m256 b = _mm256_load_ps(&data2[i]);

			__m256i mask = _mm256_castps_si256(_mm256_cmp_ps(a, b, _CMP_LE_OQ));
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

	inline vector<uint8_t> vector<float>::operator<=(float num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		__m256 b = _mm256_set1_ps(num);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256 a = _mm256_load_ps(&data1[i]);

			__m256i mask = _mm256_castps_si256(_mm256_cmp_ps(a, b, _CMP_LE_OQ));
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

	inline vector<float> vector<float>::pow(float exponent)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		vector<float> result(size);

		float* dataResult = result._data;

		__m256 _exponet = _mm256_set1_ps(exponent);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_store_ps(&dataResult[i], _mm256_pow_ps(_mm256_load_ps(&data1[i]), _exponet));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::pow(data1[i], exponent);
		}
		return result;
	}

	inline vector<float> vector<float>::pow(vector<float>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		float* data2 = other._data;

		vector<float> result(size);

		float* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_store_ps(&dataResult[i], _mm256_pow_ps(_mm256_load_ps(&data1[i]), _mm256_load_ps(&data2[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::pow(data1[i], data2[i]);
		}
		return result;
	}

	inline void vector<float>::self_pow(float exponent)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		__m256 _exponet = _mm256_set1_ps(exponent);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_store_ps(&data1[i], _mm256_pow_ps(_mm256_load_ps(&data1[i]), _exponet));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::pow(data1[i], exponent);
		}
	}

	inline void vector<float>::self_pow(vector<float>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		float* data2 = other._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_store_ps(&data1[i], _mm256_pow_ps(_mm256_load_ps(&data1[i]), _mm256_load_ps(&data2[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::pow(data1[i], data2[i]);
		}
	}

	// Root

	inline vector<float> vector<float>::root(float index)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		vector<float> result(size);

		float* dataResult = result._data;

		index = 1 / index;

		__m256 _index = _mm256_set1_ps(index);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_store_ps(&dataResult[i], _mm256_pow_ps(_mm256_load_ps(&data1[i]), _index));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::pow(data1[i], index);
		}
		return result;
	}

	inline vector<float> vector<float>::root(vector<float>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		float* data2 = other._data;

		vector<float> result(size);

		float* dataResult = result._data;

		__m256 ones = _mm256_set1_ps(1.0);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_store_ps(&dataResult[i], _mm256_pow_ps(_mm256_load_ps(&data1[i]), _mm256_div_ps(ones, _mm256_load_ps(&data2[i]))));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::pow(data1[i], 1 / data2[i]);
		}
		return result;
	}

	inline void vector<float>::self_root(float index)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		index = 1 / index;

		__m256 _index = _mm256_set1_ps(index);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_store_ps(&data1[i], _mm256_pow_ps(_mm256_load_ps(&data1[i]), _index));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::pow(data1[i], index);
		}
	}

	inline void vector<float>::self_root(vector<float>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		float* data2 = other._data;

		__m256 ones = _mm256_set1_ps(1.0);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_store_ps(&data1[i], _mm256_pow_ps(_mm256_load_ps(&data1[i]), _mm256_div_ps(ones, _mm256_load_ps(&data2[i]))));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::pow(data1[i], 1 / data2[i]);
		}
	}

	// Log

	inline vector<float> vector<float>::log()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		vector<float> result(size);

		float* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_store_ps(&dataResult[i], _mm256_log_ps(_mm256_load_ps(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::log(data1[i]);
		}
		return result;
	}

	inline void vector<float>::self_log()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_store_ps(&data1[i], _mm256_log_ps(_mm256_load_ps(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::log(data1[i]);
		}
	}

	// Log2

	inline vector<float> vector<float>::log2()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		vector<float> result(size);

		float* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_store_ps(&dataResult[i], _mm256_log2_ps(_mm256_load_ps(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::log2(data1[i]);
		}
		return result;
	}

	inline void vector<float>::self_log2()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_store_ps(&data1[i], _mm256_log2_ps(_mm256_load_ps(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::log2(data1[i]);
		}
	}

	// Log10

	inline vector<float> vector<float>::log10()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		vector<float> result(size);

		float* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_store_ps(&dataResult[i], _mm256_log10_ps(_mm256_load_ps(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::log10(data1[i]);
		}
		return result;
	}

	inline void vector<float>::self_log10()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_store_ps(&data1[i], _mm256_log10_ps(_mm256_load_ps(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::log10(data1[i]);
		}
	}

	// Exp

	inline vector<float> vector<float>::exp()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		vector<float> result(size);

		float* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_store_ps(&dataResult[i], _mm256_exp_ps(_mm256_load_ps(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::exp(data1[i]);
		}
		return result;
	}

	inline void vector<float>::self_exp()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_store_ps(&data1[i], _mm256_exp_ps(_mm256_load_ps(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::exp(data1[i]);
		}
	}

	// Exp2

	inline vector<float> vector<float>::exp2()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		vector<float> result(size);

		float* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_store_ps(&dataResult[i], _mm256_exp2_ps(_mm256_load_ps(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::exp2(data1[i]);
		}
		return result;
	}

	inline void vector<float>::self_exp2()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_store_ps(&data1[i], _mm256_exp2_ps(_mm256_load_ps(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::exp2(data1[i]);
		}
	}

	// Tan

	inline vector<float> vector<float>::tan()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		vector<float> result(size);

		float* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_store_ps(&dataResult[i], _mm256_tan_ps(_mm256_load_ps(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::tan(data1[i]);
		}
		return result;
	}

	inline void vector<float>::self_tan()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_store_ps(&data1[i], _mm256_tan_ps(_mm256_load_ps(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::tan(data1[i]);
		}
	}

	// Cos

	inline vector<float> vector<float>::cos()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		vector<float> result(size);

		float* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_store_ps(&dataResult[i], _mm256_cos_ps(_mm256_load_ps(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::cos(data1[i]);
		}
		return result;
	}

	inline void vector<float>::self_cos()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_store_ps(&data1[i], _mm256_cos_ps(_mm256_load_ps(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::cos(data1[i]);
		}
	}

	// Acos

	inline vector<float> vector<float>::acos()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		vector<float> result(size);

		float* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_store_ps(&dataResult[i], _mm256_acos_ps(_mm256_load_ps(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::acos(data1[i]);
		}
		return result;
	}

	inline void vector<float>::self_acos()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_store_ps(&data1[i], _mm256_acos_ps(_mm256_load_ps(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::acos(data1[i]);
		}
	}

	// Atan

	inline vector<float> vector<float>::atan()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		vector<float> result(size);

		float* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_store_ps(&dataResult[i], _mm256_atan_ps(_mm256_load_ps(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::atan(data1[i]);
		}
		return result;
	}

	inline void vector<float>::self_atan()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_store_ps(&data1[i], _mm256_atan_ps(_mm256_load_ps(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::atan(data1[i]);
		}
	}

	// Abs

	inline vector<float> vector<float>::abs()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		vector<float> result(size);

		float* dataResult = result._data;

		__m256 mask = _mm256_set1_ps(-0.0f);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_store_ps(&dataResult[i], _mm256_andnot_ps(mask, _mm256_load_ps(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::fabs(data1[i]);
		}
		return result;
	}

	inline void vector<float>::self_abs()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		__m256 mask = _mm256_set1_ps(-0.0f);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_store_ps(&data1[i], _mm256_andnot_ps(mask, _mm256_load_ps(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::fabs(data1[i]);
		}
	}

	// Round

	inline vector<float> vector<float>::round()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		vector<float> result(size);

		float* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_store_ps(&dataResult[i], _mm256_round_ps(_mm256_load_ps(&data1[i]), _MM_FROUND_TO_NEAREST_INT));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::round(data1[i]);
		}
		return result;
	}

	inline void vector<float>::self_round()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_store_ps(&data1[i], _mm256_round_ps(_mm256_load_ps(&data1[i]), _MM_FROUND_TO_NEAREST_INT));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::round(data1[i]);
		}
	}

	// Floor

	inline vector<float> vector<float>::floor()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		vector<float> result(size);

		float* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_store_ps(&dataResult[i], _mm256_floor_ps(_mm256_load_ps(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::floor(data1[i]);
		}
		return result;
	}

	inline void vector<float>::self_floor()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_store_ps(&data1[i], _mm256_floor_ps(_mm256_load_ps(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::floor(data1[i]);
		}
	}

	// Ceil

	inline vector<float> vector<float>::ceil()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		vector<float> result(size);

		float* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_store_ps(&dataResult[i], _mm256_ceil_ps(_mm256_load_ps(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::ceil(data1[i]);
		}
		return result;
	}

	inline void vector<float>::self_ceil()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_store_ps(&data1[i], _mm256_ceil_ps(_mm256_load_ps(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::ceil(data1[i]);
		}
	}

	// Max

	inline float vector<float>::max()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		__m256 _max = _mm256_set1_ps(FLT_MIN);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_max = _mm256_max_ps(_max, _mm256_load_ps(&data1[i]));
		}

		__m128 val1 = _mm256_castps256_ps128(_max);

		__m128 val2 = _mm256_extractf128_ps(_max, 1);

		val1 = _mm_max_ps(val1, val2);

		val2 = _mm_permute_ps(val1, 0b1110);

		val1 = _mm_max_ps(val1, val2);

		val2 = _mm_permute_ps(val1, 0b11100001);

		val1 = _mm_max_ps(val1, val2);

		float max = _mm_cvtss_f32(val1);

		for (size_t i = finalPos; i < size; i++)
		{
			float data = data1[i];
			if (data > max) max = data;
		}
		return max;
	}

	inline uint64_t vector<float>::argmax()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		__m128 _max = _mm_set1_ps(FLT_MIN);

		float max = FLT_MIN;
		uint64_t indice = 0;

		__m256i indices = _mm256_setr_epi64x(0, 1, 2, 3);

		__m256i maxIndices = _mm256_setr_epi64x(0, 1, 2, 3);

		__m256i four = _mm256_set1_epi64x(4);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m128 a = _mm_load_ps(&data1[i]);

			__m128 mask = _mm_cmp_ps(a, _max, _CMP_GT_OQ);

			maxIndices = _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(maxIndices), _mm256_castsi256_pd(indices), _mm256_castsi256_pd(_mm256_cvtepi32_epi64(_mm_castps_si128(mask)))));

			_max = _mm_blendv_ps(_max, a, mask);

			indices = _mm256_add_epi64(indices, four);
		}
		uint64_t maxIndicesArr[4];
		float maxArr[4];

		_mm256_storeu_epi64(maxIndicesArr, maxIndices);
		_mm_store_ps(maxArr, _max);

		for (size_t i = 0; i < 4; i++)
		{
			float data = maxArr[i];
			if (data > max)
			{
				max = data;
				indice = maxIndicesArr[i];
			}
		}
		for (size_t i = finalPos; i < size; i++)
		{
			float data = data1[i];
			if (data > max)
			{
				max = data;
				indice = i;
			}
		}
		return indice;
	}

	// Min

	inline float vector<float>::min()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		__m256 _min = _mm256_set1_ps(FLT_MIN);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_min = _mm256_min_ps(_min, _mm256_load_ps(&data1[i]));
		}

		__m128 val1 = _mm256_castps256_ps128(_min);

		__m128 val2 = _mm256_extractf128_ps(_min, 1);

		val1 = _mm_min_ps(val1, val2);

		val2 = _mm_permute_ps(val1, 0b1110);

		val1 = _mm_min_ps(val1, val2);

		val2 = _mm_permute_ps(val1, 0b11100001);

		val1 = _mm_min_ps(val1, val2);

		float min = _mm_cvtss_f32(val1);

		for (size_t i = finalPos; i < size; i++)
		{
			float data = data1[i];
			if (data > min) min = data;
		}
		return min;
	}

	inline uint64_t vector<float>::argmin()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		__m128 _min = _mm_set1_ps(FLT_MIN);

		float min = FLT_MIN;
		uint64_t indice = 0;

		__m256i indices = _mm256_setr_epi64x(0, 1, 2, 3);

		__m256i minIndices = _mm256_setr_epi64x(0, 1, 2, 3);

		__m256i four = _mm256_set1_epi64x(4);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m128 a = _mm_load_ps(&data1[i]);

			__m128 mask = _mm_cmp_ps(a, _min, _CMP_LT_OQ);

			minIndices = _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(minIndices), _mm256_castsi256_pd(indices), _mm256_castsi256_pd(_mm256_cvtepi32_epi64(_mm_castps_si128(mask)))));

			_min = _mm_blendv_ps(_min, a, mask);

			indices = _mm256_add_epi64(indices, four);
		}
		uint64_t minIndicesArr[4];
		float minArr[4];

		_mm256_storeu_epi64(minIndicesArr, minIndices);
		_mm_store_ps(minArr, _min);

		for (size_t i = 0; i < 4; i++)
		{
			float data = minArr[i];
			if (data < min)
			{
				min = data;
				indice = minIndicesArr[i];
			}
		}
		for (size_t i = finalPos; i < size; i++)
		{
			float data = data1[i];
			if (data < min)
			{
				min = data;
				indice = i;
			}
		}
		return indice;
	}

	// Sum

	inline float vector<float>::sum()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		__m256 _sum = _mm256_setzero_ps();

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_sum = _mm256_add_ps(_sum, _mm256_load_ps(&data1[i]));
		}


		__m256 _sum1 = _mm256_hadd_ps(_sum, _sum);
		__m256 _sum2 = _mm256_hadd_ps(_sum1, _sum1);

		__m128 lo128 = _mm256_castps256_ps128(_sum2);
		__m128 hi128 = _mm256_extractf128_ps(_sum2, 1);
		__m128 result128 = _mm_add_ps(lo128, hi128);

		float sum = _mm_cvtss_f32(result128);

		for (size_t i = finalPos; i < size; i++)
		{
			sum += data1[i];
		}

		return sum;
	}

	// Mean

	inline float vector<float>::mean()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		__m256 _sum = _mm256_setzero_ps();

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_sum = _mm256_add_ps(_sum, _mm256_load_ps(&data1[i]));
		}


		__m256 _sum1 = _mm256_hadd_ps(_sum, _sum);
		__m256 _sum2 = _mm256_hadd_ps(_sum1, _sum1);

		__m128 lo128 = _mm256_castps256_ps128(_sum2);
		__m128 hi128 = _mm256_extractf128_ps(_sum2, 1);
		__m128 result128 = _mm_add_ps(lo128, hi128);

		float sum = _mm_cvtss_f32(result128);

		for (size_t i = finalPos; i < size; i++)
		{
			sum += data1[i];
		}

		return sum / static_cast<float>(size);
	}

	// Std

	inline float vector<float>::std(float ddof, float* mean)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		__m256 _sum = _mm256_setzero_ps();
		__m256 _sumSquare = _mm256_setzero_ps();

		float size_f = static_cast<float>(size);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256 a = _mm256_load_ps(&data1[i]);

			_sum = _mm256_add_ps(_sum, a);
			_sumSquare = _mm256_add_ps(_sumSquare, _mm256_mul_ps(a, a));
		}

		__m256 _sum1 = _mm256_hadd_ps(_sum, _sum);
		__m256 _sum2 = _mm256_hadd_ps(_sum1, _sum1);

		__m128 lo128 = _mm256_castps256_ps128(_sum2);
		__m128 hi128 = _mm256_extractf128_ps(_sum2, 1);
		__m128 result128 = _mm_add_ps(lo128, hi128);

		float sum = _mm_cvtss_f32(result128);
		//--

		_sum1 = _mm256_hadd_ps(_sumSquare, _sumSquare);
		_sum2 = _mm256_hadd_ps(_sum1, _sum1);

		lo128 = _mm256_castps256_ps128(_sum2);
		hi128 = _mm256_extractf128_ps(_sum2, 1);
		result128 = _mm_add_ps(lo128, hi128);

		float sumSquare = _mm_cvtss_f32(result128);

		for (size_t i = finalPos; i < size; i++)
		{
			float data = data1[i];
			sum += data;
			sumSquare += data * data;
		}
		if (mean != nullptr) *mean = sum / size_f;

		float variance = (sumSquare - (sum * sum / size_f)) / (size_f - ddof);
		float std = std::sqrt(variance);
		return std;
	}

	// Activation Functions

	// Tanh

	inline vector<float> vector<float>::tanh()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		vector<float> result(size);

		float* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_ps(&dataResult[i], _mm256_tanh_ps(_mm256_load_ps(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::tanh(data1[i]);
		}
		return result;
	}

	inline void vector<float>::self_tanh()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_ps(&data1[i], _mm256_tanh_ps(_mm256_load_ps(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::tanh(data1[i]);
		}
	}

	// Cosh

	inline vector<float> vector<float>::cosh()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		vector<float> result(size);

		float* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_ps(&dataResult[i], _mm256_cosh_ps(_mm256_load_ps(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::cosh(data1[i]);
		}
		return result;
	}

	inline void vector<float>::self_cosh()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_ps(&data1[i], _mm256_cosh_ps(_mm256_load_ps(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::cosh(data1[i]);
		}
	}

	// ReLU

	inline vector<float> vector<float>::relu()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		vector<float> result(size);

		float* dataResult = result._data;

		__m256 zero = _mm256_setzero_ps();

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_ps(&dataResult[i], _mm256_max_ps(zero, _mm256_load_ps(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::max(0.0f, data1[i]);
		}
		return result;
	}

	inline void vector<float>::self_relu()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		__m256 zero = _mm256_setzero_ps();

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_ps(&data1[i], _mm256_max_ps(zero, _mm256_load_ps(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::max(0.0f, data1[i]);
		}
	}

	// LReLU

	inline vector<float> vector<float>::lrelu()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		vector<float> result(size);

		float* dataResult = result._data;

		__m256 num = _mm256_set1_ps(0.01f);

		__m256 zero = _mm256_setzero_ps();

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256 a = _mm256_load_ps(&data1[i]);

			_mm256_store_ps(&dataResult[i], _mm256_blendv_ps(_mm256_mul_ps(a, num), a, _mm256_cmp_ps(a, zero, _CMP_GT_OQ)));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] > 0.0f ? data1[i] : 0.01f * data1[i];
		}
		return result;
	}

	inline void vector<float>::self_lrelu()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		__m256 num = _mm256_set1_ps(0.01f);

		__m256 zero = _mm256_setzero_ps();

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256 a = _mm256_load_ps(&data1[i]);

			_mm256_store_ps(&data1[i], _mm256_blendv_ps(_mm256_mul_ps(a, num), a, _mm256_cmp_ps(a, zero, _CMP_GT_OQ)));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = data1[i] > 0.0f ? data1[i] : 0.01f * data1[i];
		}
	}

	// Sigmoid

	inline vector<float> vector<float>::sigmoid()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		vector<float> result(size);

		float* dataResult = result._data;

		__m256 mask = _mm256_set1_ps(-0.0f);

		__m256 one = _mm256_set1_ps(1.0f);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256 a = _mm256_load_ps(&data1[i]);

			__m256 neg = _mm256_xor_ps(a, mask);

			__m256 sigmoid = _mm256_div_ps(one, _mm256_add_ps(_mm256_exp_ps(neg), one));

			_mm256_store_ps(&dataResult[i], sigmoid);
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = 1.0f / (1.0f + std::exp(-data1[i]));
		}
		return result;
	}

	inline void vector<float>::self_sigmoid()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		__m256 mask = _mm256_set1_ps(-0.0f);

		__m256 one = _mm256_set1_ps(1.0f);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256 a = _mm256_load_ps(&data1[i]);

			__m256 neg = _mm256_xor_ps(a, mask);

			__m256 sigmoid = _mm256_div_ps(one, _mm256_add_ps(_mm256_exp_ps(neg), one));

			_mm256_store_ps(&data1[i], sigmoid);
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = 1.0f / (1.0f + std::exp(-data1[i]));
		}
	}

	// Softplus

	inline vector<float> vector<float>::softplus()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		vector<float> result(size);

		float* dataResult = result._data;

		__m256 one = _mm256_set1_ps(1.0f);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_ps(&dataResult[i], _mm256_log_ps(_mm256_add_ps(one, _mm256_exp_ps(_mm256_load_ps(&data1[i])))));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::log(1.0f + std::exp(data1[i]));
		}
		return result;
	}

	inline void vector<float>::self_softplus()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		float* data1 = this->_data;

		__m256 one = _mm256_set1_ps(1.0f);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_store_ps(&data1[i], _mm256_log_ps(_mm256_add_ps(one, _mm256_exp_ps(_mm256_load_ps(&data1[i])))));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::log(1.0f + std::exp(data1[i]));
		}
	}

	// Softmax

	inline vector<float> vector<float>::softmax()
	{
		vector<float> exp = ((*this) - this->max()).exp();

		return exp / exp.sum();
	}

	// Sort

	inline void vector<float>::sort()
	{
		std::sort(this->_data, this->_data + this->_size);
	}

	// Argsort

	inline vector<uint64_t> vector<float>::argsort()
	{
		vector<uint64_t> indices(this->_size);

		size_t* indicesData = indices._data;

		for (size_t i = 0; i < this->_size; i++) indicesData[i] = i;

		quicksort(indicesData, this->_data, 0, this->_size - 1);

		return indices;
	}

	// Cast

	template <typename T>
	inline vector<T> vector<float>::cast()
	{
		size_t size = this->_size;

		vector<T> result(size);

		float* data1 = this->_data;

		T* dataResult = result._data;

		size_t finalPos = this->finalPos;

		if constexpr (std::is_same<T, uint8_t>::value)
		{
			__m256 zero = _mm256_setzero_ps();
			__m256i indices = _mm256_setr_epi32(0, 7, 2, 3, 4, 5, 6, 1);

			for (size_t i = 0; i < finalPos; i += 8)
			{
				__m256i mask = _mm256_castps_si256(_mm256_cmp_ps(_mm256_load_ps(&data1[i]), zero, _CMP_NEQ_OQ));

				__m256i mask1 = _mm256_packs_epi32(mask, mask);
				__m256i mask2 = _mm256_packs_epi16(mask1, mask1);

				mask2 = _mm256_permutevar8x32_epi32(mask2, indices);

				_mm_store_sd(reinterpret_cast<double*>(&dataResult[i]), _mm_castsi128_pd(_mm256_castsi256_si128(mask2)));
			}
			for (size_t i = finalPos; i < size; i += 8)
			{
				dataResult[i] = data1[i] != 0.0f ? True : False;
			}
		}
		else if constexpr (std::is_same<T, int>::value)
		{
			for (size_t i = 0; i < finalPos; i += 8)
			{
				_mm256_storeu_epi32(&dataResult[i], _mm256_cvtps_epi32(_mm256_load_ps(&data1[i])));
			}
			for (size_t i = finalPos; i < size; i++)
			{
				dataResult[i] = static_cast<int>(data1[i]);
			}
		}
		else if constexpr (std::is_same<T, double>::value)
		{
			for (size_t i = 0; i < finalPos; i += 8)
			{
				_mm256_store_pd(&dataResult[i], _mm256_cvtps_pd(_mm_load_ps(&data1[i])));
			}
			for (size_t i = finalPos; i < size; i++)
			{
				dataResult[i] = static_cast<double>(data1[i]);
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