#include "vectorUint64_t.h"

namespace alge
{
	// Constructors

	inline vector<uint64_t>::vector() :
		_data(nullptr),
		dataToDelete(nullptr),
		_size(0),
		finalPos(0),
		_capacity(0) {}

	inline vector<uint64_t>::vector(size_t size) :
		_data(new uint64_t[size]),
		dataToDelete(_data),
		_size(size),
		finalPos((size / 4) * 4),
		_capacity(size) {}

	inline vector<uint64_t>::vector(uint64_t* data, size_t size) :
		_data(data),
		dataToDelete(nullptr),
		_size(size),
		finalPos((size / 4) * 4),
		_capacity(size) {}

	inline vector<uint64_t>::vector(std::initializer_list<uint64_t> list)
	{
		this->_size = list.size();
		this->finalPos = (this->_size / 4) * 4;
		this->_data = new uint64_t[this->_size];
		this->dataToDelete = this->_data;
		this->_capacity = this->_size;

		for (size_t i = 0; i < this->_size; i++)
		{
			this->_data[i] = *(list.begin() + i);
		}
	}

	// Destructor

	// Block

	inline vector<uint64_t> vector<uint64_t>::block(size_t initial, size_t final)
	{
		return vector<uint64_t>(
			&this->_data[initial],
			final - initial
		);
	}

	// Copy

	inline vector<uint64_t> vector<uint64_t>::copy()
	{
		size_t size = this->_size;

		vector<uint64_t> result(size);

		uint64_t* data1 = this->_data;

		uint64_t* dataResult = result._data;

		for (size_t i = 0; i < size; i++)
		{
			dataResult[i] = data1[i];
		}
		return result;
	}

	// = 

	inline vector<uint64_t>& vector<uint64_t>::operator=(vector<uint64_t>& other)
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
			uint64_t* data1 = this->_data;
			uint64_t* data2 = other._data;

			for (size_t i = 0; i < size; i++)
			{
				data1[i] = data2[i];
			}
		}
		return *this;
	}

	// Transfer

	inline void vector<uint64_t>::transfer(vector<uint64_t>& other)
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

	inline void vector<uint64_t>::set_const(uint64_t num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint64_t* data1 = this->_data;

		for (size_t i = 0; i < size; i++)
		{
			data1[i] = num;
		}
	}

	inline vector<uint64_t>::~vector() { delete[] this->dataToDelete; }

	inline uint64_t* vector<uint64_t>::data() { return this->_data; }

	inline size_t vector<uint64_t>::size() { return this->_size; }

	inline uint64_t& vector<uint64_t>::operator[](size_t index)
	{
		uint64_t* data = this->_data;
		return data[index];
	}

	inline const uint64_t& vector<uint64_t>::operator[](size_t index) const
	{
		uint64_t* data = this->_data;
		return data[index];
	}

	inline size_t vector<uint64_t>::capacity() { return this->_capacity; }

	template<bool reduceCapacity>
	inline void vector<uint64_t>::clear()
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

	inline void vector<uint64_t>::reserve(size_t newCapacity)
	{
		uint64_t* newData = new uint64_t[newCapacity];
		uint64_t* oldData = this->_data;

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

	inline void vector<uint64_t>::append(uint64_t num)
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
			uint64_t* newData = new uint64_t[this->_capacity];
			uint64_t* oldData = this->_data;
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

	inline void vector<uint64_t>::append(std::initializer_list<uint64_t> list)
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
			uint64_t* newData = new uint64_t[this->_capacity];
			uint64_t* oldData = this->_data;

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

	inline void vector<uint64_t>::append(vector<uint64_t>& other)
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
			uint64_t* newData = new uint64_t[this->_capacity];
			uint64_t* oldData = this->_data;

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

	inline void vector<uint64_t>::insert(uint64_t num, size_t index)
	{
		if (this->_capacity > this->_size)
		{
			uint64_t* data1 = this->_data;

			uint64_t tmp = num;
			uint64_t tmp2;
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
			uint64_t* newData = new uint64_t[this->_capacity];
			uint64_t* oldData = this->_data;

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

	inline void vector<uint64_t>::erase(size_t index)
	{
		uint64_t* data1 = this->_data;
		this->_size--;
		this->finalPos = (this->_size / 4) * 4;
		if (this->dataToDelete == nullptr)
		{
			uint64_t* newData = new uint64_t[this->_size];

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
	inline size_t vector<uint64_t>::find(uint64_t num)
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

	// +

	inline vector<uint64_t> vector<uint64_t>::operator+(vector<uint64_t>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint64_t* data1 = this->_data;
		uint64_t* data2 = other._data;

		vector<uint64_t> result(size);

		uint64_t* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256i a = _mm256_loadu_epi64(&data1[i]);
			__m256i b = _mm256_loadu_epi64(&data2[i]);

			_mm256_storeu_epi64(&dataResult[i], _mm256_add_epi64(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] + data2[i];
		}
		return result;
	}

	inline vector<uint64_t> vector<uint64_t>::operator+(uint64_t num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint64_t* data1 = this->_data;

		vector<uint64_t> result(size);

		uint64_t* dataResult = result._data;

		__m256i b = _mm256_set1_epi64x(num);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256i a = _mm256_loadu_epi64(&data1[i]);

			_mm256_storeu_epi64(&dataResult[i], _mm256_add_epi64(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] + num;
		}
		return result;
	}

	inline void vector<uint64_t>::operator+=(vector<uint64_t>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint64_t* data1 = this->_data;
		uint64_t* data2 = other._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256i a = _mm256_loadu_epi64(&data1[i]);
			__m256i b = _mm256_loadu_epi64(&data2[i]);

			_mm256_storeu_epi64(&data1[i], _mm256_add_epi64(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] += data2[i];
		}
	}

	inline void vector<uint64_t>::operator+=(uint64_t num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint64_t* data1 = this->_data;

		__m256i b = _mm256_set1_epi64x(num);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256i a = _mm256_loadu_epi64(&data1[i]);

			_mm256_storeu_epi64(&data1[i], _mm256_add_epi64(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] += num;
		}
	}

	// -

	inline vector<uint64_t> vector<uint64_t>::operator-(vector<uint64_t>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint64_t* data1 = this->_data;
		uint64_t* data2 = other._data;

		vector<uint64_t> result(size);

		uint64_t* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256i a = _mm256_loadu_epi64(&data1[i]);
			__m256i b = _mm256_loadu_epi64(&data2[i]);

			_mm256_storeu_epi64(&dataResult[i], _mm256_sub_epi64(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] - data2[i];
		}
		return result;
	}

	inline vector<uint64_t> vector<uint64_t>::operator-(uint64_t num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint64_t* data1 = this->_data;

		vector<uint64_t> result(size);

		uint64_t* dataResult = result._data;

		__m256i b = _mm256_set1_epi64x(num);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256i a = _mm256_loadu_epi64(&data1[i]);

			_mm256_storeu_epi64(&dataResult[i], _mm256_sub_epi64(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] - num;
		}
		return result;
	}

	inline void vector<uint64_t>::operator-=(vector<uint64_t>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint64_t* data1 = this->_data;
		uint64_t* data2 = other._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256i a = _mm256_loadu_epi64(&data1[i]);
			__m256i b = _mm256_loadu_epi64(&data2[i]);

			_mm256_storeu_epi64(&data1[i], _mm256_sub_epi64(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] -= data2[i];
		}
	}

	inline void vector<uint64_t>::operator-=(uint64_t num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint64_t* data1 = this->_data;

		__m256i b = _mm256_set1_epi64x(num);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256i a = _mm256_loadu_epi64(&data1[i]);

			_mm256_storeu_epi64(&data1[i], _mm256_sub_epi64(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] -= num;
		}
	}

	// *

	inline vector<uint64_t> vector<uint64_t>::operator*(vector<uint64_t>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint64_t* data1 = this->_data;
		uint64_t* data2 = other._data;

		vector<uint64_t> result(size);

		uint64_t* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256i a = _mm256_loadu_epi64(&data1[i]);
			__m256i b = _mm256_loadu_epi64(&data2[i]);

			_mm256_storeu_epi64(&dataResult[i], _mm256_mul_epu32(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] * data2[i];
		}
		return result;
	}

	inline vector<uint64_t> vector<uint64_t>::operator*(uint64_t num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint64_t* data1 = this->_data;

		vector<uint64_t> result(size);

		uint64_t* dataResult = result._data;

		__m256i b = _mm256_set1_epi64x(num);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256i a = _mm256_loadu_epi64(&data1[i]);

			_mm256_storeu_epi64(&dataResult[i], _mm256_mul_epu32(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] * num;
		}
		return result;
	}

	inline void vector<uint64_t>::operator*=(vector<uint64_t>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint64_t* data1 = this->_data;
		uint64_t* data2 = other._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256i a = _mm256_loadu_epi64(&data1[i]);
			__m256i b = _mm256_loadu_epi64(&data2[i]);

			_mm256_storeu_epi64(&data1[i], _mm256_mul_epu32(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] *= data2[i];
		}
	}

	inline void vector<uint64_t>::operator*=(uint64_t num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint64_t* data1 = this->_data;

		__m256i b = _mm256_set1_epi64x(num);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256i a = _mm256_loadu_epi64(&data1[i]);

			_mm256_storeu_epi64(&data1[i], _mm256_mul_epu32(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] *= num;
		}
	}

	// /

	inline vector<uint64_t> vector<uint64_t>::operator/(vector<uint64_t>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint64_t* data1 = this->_data;
		uint64_t* data2 = other._data;

		vector<uint64_t> result(size);

		uint64_t* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256i a = _mm256_loadu_epi64(&data1[i]);
			__m256i b = _mm256_loadu_epi64(&data2[i]);

			_mm256_storeu_epi64(&dataResult[i], _mm256_div_epi64(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] / data2[i];
		}
		return result;
	}

	inline vector<uint64_t> vector<uint64_t>::operator/(uint64_t num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint64_t* data1 = this->_data;

		vector<uint64_t> result(size);

		uint64_t* dataResult = result._data;

		__m256i b = _mm256_set1_epi64x(num);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256i a = _mm256_loadu_epi64(&data1[i]);

			_mm256_storeu_epi64(&dataResult[i], _mm256_div_epi64(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] / num;
		}
		return result;
	}

	inline void vector<uint64_t>::operator/=(vector<uint64_t>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint64_t* data1 = this->_data;
		uint64_t* data2 = other._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256i a = _mm256_loadu_epi64(&data1[i]);
			__m256i b = _mm256_loadu_epi64(&data2[i]);

			_mm256_storeu_epi64(&data1[i], _mm256_div_epi64(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] /= data2[i];
		}
	}

	inline void vector<uint64_t>::operator/=(uint64_t num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint64_t* data1 = this->_data;

		__m256i b = _mm256_set1_epi64x(num);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256i a = _mm256_loadu_epi64(&data1[i]);

			_mm256_storeu_epi64(&data1[i], _mm256_div_epi64(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] /= num;
		}
	}

	// ==

	inline vector<uint8_t> vector<uint64_t>::operator==(vector<uint64_t>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint64_t* data1 = this->_data;
		uint64_t* data2 = other._data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256i a = _mm256_loadu_epi64(&data1[i]);
			__m256i b = _mm256_loadu_epi64(&data2[i]);

			__m256i mask = _mm256_cmpeq_epi64(a, b);

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

	inline vector<uint8_t> vector<uint64_t>::operator==(uint64_t num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint64_t* data1 = this->_data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		__m256i b = _mm256_set1_epi64x(num);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256i a = _mm256_loadu_epi64(&data1[i]);

			__m256i mask = _mm256_cmpeq_epi64(a, b);

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

	inline vector<uint8_t> vector<uint64_t>::operator!=(vector<uint64_t>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint64_t* data1 = this->_data;
		uint64_t* data2 = other._data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		__m256i minus_ones = _mm256_set1_epi64x(-1);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256i a = _mm256_loadu_epi64(&data1[i]);
			__m256i b = _mm256_loadu_epi64(&data2[i]);

			__m256i mask = _mm256_andnot_si256(_mm256_cmpeq_epi64(a, b), minus_ones);
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

	inline vector<uint8_t> vector<uint64_t>::operator!=(uint64_t num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint64_t* data1 = this->_data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		__m256i b = _mm256_set1_epi64x(num);

		__m256i minus_ones = _mm256_set1_epi64x(-1);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256i a = _mm256_loadu_epi64(&data1[i]);

			__m256i mask = _mm256_andnot_si256((_mm256_cmpeq_epi64(a, b)), minus_ones);
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

	inline vector<uint8_t> vector<uint64_t>::operator>(vector<uint64_t>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint64_t* data1 = this->_data;
		uint64_t* data2 = other._data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256i a = _mm256_loadu_epi64(&data1[i]);
			__m256i b = _mm256_loadu_epi64(&data2[i]);

			__m256i mask = _mm256_cmpgt_epi64(a, b);
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

	inline vector<uint8_t> vector<uint64_t>::operator>(uint64_t num)
	{
		size_t size = this->_size;

		size_t finalPos = (size / 4) * 4;

		uint64_t* data1 = this->_data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		__m256i b = _mm256_set1_epi64x(num);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256i a = _mm256_loadu_epi64(&data1[i]);

			__m256i mask = _mm256_cmpgt_epi64(a, b);
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

	// < 

	inline vector<uint8_t> vector<uint64_t>::operator<(vector<uint64_t>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint64_t* data1 = this->_data;
		uint64_t* data2 = other._data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		__m256i minus_ones = _mm256_set1_epi64x(-1);

		for (size_t i = 0; i < finalPos; i += 4)
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

			__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

			_mm_store_ss(reinterpret_cast<float*>(&dataResult[i]), _mm_castsi128_ps(maskResult));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] < data2[i] ? True : False;
		}
		return result;
	}

	inline vector<uint8_t> vector<uint64_t>::operator<(uint64_t num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint64_t* data1 = this->_data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		__m256i b = _mm256_set1_epi64x(num);

		__m256i minus_ones = _mm256_set1_epi64x(-1);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256i a = _mm256_loadu_epi64(&data1[i]);

			__m256i gt = _mm256_cmpgt_epi64(a, b);
			__m256i eq = _mm256_cmpeq_epi64(a, b);

			__m256i mask = _mm256_andnot_si256(gt, _mm256_andnot_si256(eq, minus_ones));

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

	// >=

	inline vector<uint8_t> vector<uint64_t>::operator>=(vector<uint64_t>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint64_t* data1 = this->_data;
		uint64_t* data2 = other._data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
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

			__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

			_mm_store_ss(reinterpret_cast<float*>(&dataResult[i]), _mm_castsi128_ps(maskResult));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] >= data2[i] ? True : False;
		}
		return result;
	}

	inline vector<uint8_t> vector<uint64_t>::operator>=(uint64_t num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint64_t* data1 = this->_data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		__m256i b = _mm256_set1_epi64x(num);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256i a = _mm256_loadu_epi64(&data1[i]);

			__m256i gt = _mm256_cmpgt_epi64(a, b);
			__m256i eq = _mm256_cmpeq_epi64(a, b);

			__m256i mask = _mm256_or_si256(gt, eq);
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

	// <=

	inline vector<uint8_t> vector<uint64_t>::operator<=(vector<uint64_t>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint64_t* data1 = this->_data;
		uint64_t* data2 = other._data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		__m256i minus_ones = _mm256_set1_epi64x(-1);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256i a = _mm256_loadu_epi64(&data1[i]);
			__m256i b = _mm256_loadu_epi64(&data2[i]);

			__m256i mask = _mm256_andnot_si256(_mm256_cmpgt_epi64(a, b), minus_ones);
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

	inline vector<uint8_t> vector<uint64_t>::operator<=(uint64_t num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint64_t* data1 = this->_data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		__m256i b = _mm256_set1_epi64x(num);

		__m256i minus_ones = _mm256_set1_epi64x(-1);

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256i a = _mm256_loadu_epi64(&data1[i]);

			__m256i mask = _mm256_andnot_si256(_mm256_cmpgt_epi64(a, b), minus_ones);
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

	inline vector<uint64_t> vector<uint64_t>::pow(uint64_t exponent)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint64_t* data1 = this->_data;

		vector<uint64_t> result(size);

		uint64_t* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
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
			_mm256_storeu_epi64(&dataResult[i], result_pow);
		}
		for (size_t i = finalPos; i < size; i++)
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
			dataResult[i] = result_pow;
		}
		return result;
	}

	inline vector<uint64_t> vector<uint64_t>::pow(vector<uint64_t>& other)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint64_t* data1 = this->_data;
		uint64_t* data2 = other._data;

		vector<uint64_t> result(size);

		uint64_t* dataResult = result._data;

		__m256i one = _mm256_set1_epi64x(1);
		__m256i zero = _mm256_setzero_si256();

		for (size_t i = 0; i < finalPos; i += 4)
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
			_mm256_storeu_epi64(&dataResult[i], result_pow);
		}
		for (size_t i = finalPos; i < size; i++)
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
			dataResult[i] = result_pow;
		}
		return result;
	}

	inline void vector<uint64_t>::self_pow(uint64_t exponent)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint64_t* data1 = this->_data;

		for (size_t i = 0; i < finalPos; i += 4)
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
		for (size_t i = finalPos; i < size; i++)
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

	inline void vector<uint64_t>::self_pow(vector<uint64_t>& other)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint64_t* data1 = this->_data;
		uint64_t* data2 = other._data;

		__m256i one = _mm256_set1_epi64x(1);
		__m256i zero = _mm256_setzero_si256();

		for (size_t i = 0; i < finalPos; i += 4)
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
		for (size_t i = finalPos; i < size; i++)
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

	// <<

	inline vector<uint64_t> vector<uint64_t>::operator<<(int shift)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint64_t* data1 = this->_data;

		vector<uint64_t> result(size);

		uint64_t* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_storeu_epi64(&dataResult[i], _mm256_slli_epi64(_mm256_loadu_epi64(&data1[i]), shift));
		}
		for (size_t i = finalPos; i < size; i += 4)
		{
			dataResult[i] = data1[i] << shift;
		}
		return result;
	}

	inline vector<uint64_t> vector<uint64_t>::operator<<(vector<uint64_t>& other)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint64_t* data1 = this->_data;
		uint64_t* data2 = other._data;

		vector<uint64_t> result(size);

		uint64_t* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_storeu_epi64(&dataResult[i], _mm256_sllv_epi64(_mm256_loadu_epi64(&data1[i]), _mm256_loadu_epi64(&data2[i])));
		}
		for (size_t i = finalPos; i < size; i += 4)
		{
			dataResult[i] = data1[i] << data2[i];
		}
		return result;
	}

	inline void vector<uint64_t>::operator<<=(int shift)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint64_t* data1 = this->_data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_storeu_epi64(&data1[i], _mm256_slli_epi64(_mm256_loadu_epi64(&data1[i]), shift));
		}
		for (size_t i = finalPos; i < size; i += 4)
		{
			data1[i] <<= shift;
		}
	}

	inline void vector<uint64_t>::operator<<=(vector<uint64_t>& other)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint64_t* data1 = this->_data;
		uint64_t* data2 = other._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_storeu_epi64(&data1[i], _mm256_sllv_epi64(_mm256_loadu_epi64(&data1[i]), _mm256_loadu_epi64(&data2[i])));
		}
		for (size_t i = finalPos; i < size; i += 4)
		{
			data1[i] <<= data2[i];
		}
	}

	// >>

	inline vector<uint64_t> vector<uint64_t>::operator>>(int shift)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint64_t* data1 = this->_data;

		vector<uint64_t> result(size);

		uint64_t* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_storeu_epi64(&dataResult[i], _mm256_srli_epi64(_mm256_loadu_epi64(&data1[i]), shift));
		}
		for (size_t i = finalPos; i < size; i += 4)
		{
			dataResult[i] = data1[i] >> shift;
		}
		return result;
	}

	inline vector<uint64_t> vector<uint64_t>::operator>>(vector<uint64_t>& other)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint64_t* data1 = this->_data;
		uint64_t* data2 = other._data;

		vector<uint64_t> result(size);

		uint64_t* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_storeu_epi64(&dataResult[i], _mm256_srlv_epi64(_mm256_loadu_epi64(&data1[i]), _mm256_loadu_epi64(&data2[i])));
		}
		for (size_t i = finalPos; i < size; i += 4)
		{
			dataResult[i] = data1[i] >> data2[i];
		}
		return result;
	}

	inline void vector<uint64_t>::operator>>=(int shift)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint64_t* data1 = this->_data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_storeu_epi64(&data1[i], _mm256_srli_epi64(_mm256_loadu_epi64(&data1[i]), shift));
		}
		for (size_t i = finalPos; i < size; i += 4)
		{
			data1[i] >>= shift;
		}
	}

	inline void vector<uint64_t>::operator>>=(vector<uint64_t>& other)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint64_t* data1 = this->_data;
		uint64_t* data2 = other._data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			_mm256_storeu_epi64(&data1[i], _mm256_srlv_epi64(_mm256_loadu_epi64(&data1[i]), _mm256_loadu_epi64(&data2[i])));
		}
		for (size_t i = finalPos; i < size; i += 4)
		{
			data1[i] >>= data2[i];
		}
	}

	// Sort

	inline void vector<uint64_t>::sort()
	{
		std::sort(this->_data, this->_data + this->_size);
	}

	// Argsort

	inline vector<uint64_t> vector<uint64_t>::argsort()
	{
		vector<uint64_t> indices(this->_size);

		size_t* indicesData = indices._data;

		for (size_t i = 0; i < this->_size; i++) indicesData[i] = i;

		quicksort(indicesData, this->_data, 0, this->_size - 1);

		return indices;
	}

	// Cast

	template<typename T>
	inline vector<T> vector<uint64_t>::cast()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint64_t* data1 = this->_data;

		vector<T> result(size);

		T* dataResult = result._data;

		if constexpr (std::is_same<T, double>::value)
		{
			masks_uint64_to_double;
			for (size_t i = 0; i < finalPos; i += 4)
			{
				uint64_to_double(_mm256_loadu_epi64(&data1[i]));
				_mm256_store_pd(&dataResult[i], uint64ToDouble);
			}
			for (size_t i = finalPos; i < size; i++)
			{
				dataResult[i] = static_cast<double>(data1[i]);
			}
		}
		else if constexpr (std::is_same<T, int>::value)
		{
			__m256i indices = _mm256_setr_epi32(0, 2, 4, 6, 7, 5, 3, 1);

			for (size_t i = 0; i < finalPos; i += 4)
			{
				_mm_storeu_epi32(&dataResult[i], _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_loadu_epi64(&data1[i]), indices)));
			}
			for (size_t i = finalPos; i < size; i++)
			{
				dataResult[i] = static_cast<int>(data1[i]);
			}
		}
		else if constexpr (std::is_same<T, uint8_t>::value)
		{
			__m256d zero = _mm256_setzero_pd();

			for (size_t i = 0; i < finalPos; i += 4)
			{
				__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(_mm256_castsi256_pd(_mm256_loadu_epi64(&data1[i])), zero, _CMP_NEQ_OQ));

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