#include "vectorUint8_t.h"

namespace alge
{
	// Constructors

	inline vector<uint8_t>::vector() :
		_data(nullptr),
		dataToDelete(nullptr),
		_size(0), finalPos(0),
		finalPos256(0),
		_capacity(0) {}

	inline vector<uint8_t>::vector(size_t size) : 
		_data(new uint8_t[size]),
		dataToDelete(_data), _size(size),
		finalPos((size / 32) * 32),
		finalPos256((size / 256) * 256),
		_capacity(size) {}

	inline vector<uint8_t>::vector(uint8_t* data, size_t size) :
		_data(data),
		dataToDelete(nullptr),
		_size(size),
		finalPos((size / 32) * 32),
		finalPos256((size / 256) * 256),
		_capacity(size) {}

	inline vector<uint8_t>::vector(std::initializer_list<uint8_t> list)
	{
		this->_size = list.size();
		this->finalPos = (this->_size / 32) * 32;
		this->finalPos256 = (this->_size / 256) * 256;
		this->_data = new uint8_t[this->_size];
		this->dataToDelete = this->_data;
		this->_capacity = this->_size;

		for (size_t i = 0; i < this->_size; i++)
		{
			this->_data[i] = *(list.begin() + i);
		}
	}

	// Destructor

	inline vector<uint8_t>::~vector() { delete[] this->dataToDelete; }

	// Block

	inline vector<uint8_t> vector<uint8_t>::block(size_t initial, size_t final)
	{
		return vector<uint8_t>(
			&this->_data[initial],
			final - initial
		);
	}

	// Copy

	inline vector<uint8_t> vector<uint8_t>::copy()
	{
		size_t size = this->_size;

		vector<uint8_t> result(size);

		uint8_t* data1 = this->_data;

		uint8_t* dataResult = result._data;

		for (size_t i = 0; i < size; i++)
		{
			dataResult[i] = data1[i];
		}
		return result;
	}

	// Set Constant

	inline void vector<uint8_t>::set_const(uint8_t num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint8_t* data1 = this->_data;

		for (size_t i = 0; i < size; i++)
		{
			data1[i] = num;
		}
	}

	// =

	inline vector<uint8_t>& vector<uint8_t>::operator=(vector<uint8_t>& other)
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
			uint8_t* data1 = this->_data;
			uint8_t* data2 = other._data;

			for (size_t i = 0; i < size; i++)
			{
				data1[i] = data2[i];
			}
		}
		return *this;
	}

	// Transfer

	inline void vector<uint8_t>::transfer(vector<uint8_t>& other)
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
		this->finalPos256 = other.finalPos256;
	}

	// Utilities

	inline uint8_t* vector<uint8_t>::data() { return this->_data; }

	inline size_t vector<uint8_t>::size() { return this->_size; }

	inline uint8_t& vector<uint8_t>::operator[](size_t index){ return this->_data[index]; }

	inline const uint8_t& vector<uint8_t>::operator[](size_t index) const { return this->_data[index]; }

	inline vector<uint8_t> vector<uint8_t>::operator[](vector<uint64_t>& indices)
	{
		size_t size = indices._size;

		vector<uint8_t> result(size);

		uint8_t* data1 = this->_data;

		uint64_t* dataIndices = indices._data;

		uint8_t* dataResult = result._data;

		for (size_t i = 0; i < size; i++)
		{
			dataResult[i] = data1[dataIndices[i]];
		}
		return result;
	}

	inline size_t vector<uint8_t>::capacity() { return this->_capacity; }

	template<bool reduceCapacity>
	inline void vector<uint8_t>::clear()
	{
		if constexpr (reduceCapacity)
		{
			this->_size = 0;
			this->_capacity = 0;
			this->finalPos = 0;
			this->finalPos256 = 0;
			delete[] this->dataToDelete;
			this->_data = nullptr;
			this->dataToDelete = nullptr;
		}
		else
		{
			this->_size = 0;
			this->_capacity = 0;
			this->finalPos = 0;
			this->finalPos256 = 0;
		}
	}

	inline void vector<uint8_t>::reserve(size_t newCapacity)
	{
		uint8_t* newData = new uint8_t[newCapacity];
		uint8_t* oldData = this->_data;

		this->_size = this->_size <= newCapacity ? this->_size : newCapacity;
		this->finalPos = (this->_size / 32) * 32;
		this->finalPos256 = (this->_size / 256) * 256;
		this->_capacity = newCapacity;
		for (size_t i = 0; i < this->_size; i++)
		{
			newData[i] = oldData[i];
		}
		delete[] this->dataToDelete;
		this->_data = newData;
		this->dataToDelete = newData;
	}

	inline void vector<uint8_t>::append(uint8_t num)
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
			uint8_t* newData = new uint8_t[this->_capacity];
			uint8_t* oldData = this->_data;
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
		this->finalPos = (this->_size / 32) * 32;
		this->finalPos256 = (this->_size / 256) * 256;
	}

	inline void vector<uint8_t>::append(std::initializer_list<uint8_t> list)
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
			uint8_t* newData = new uint8_t[this->_capacity];
			uint8_t* oldData = this->_data;

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
		this->finalPos = (this->_size / 32) * 32;
		this->finalPos256 = (this->_size / 256) * 256;
	}

	inline void vector<uint8_t>::append(vector<uint8_t>& other)
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
			uint8_t* newData = new uint8_t[this->_capacity];
			uint8_t* oldData = this->_data;

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
		this->finalPos = (this->_size / 32) * 32;
		this->finalPos256 = (this->_size / 256) * 256;
	}

	inline void vector<uint8_t>::insert(uint8_t num, size_t index)
	{
		if (this->_capacity > this->_size)
		{
			uint8_t* data1 = this->_data;

			uint8_t tmp = num;
			uint8_t tmp2;
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
			uint8_t* newData = new uint8_t[this->_capacity];
			uint8_t* oldData = this->_data;

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
		this->finalPos = (this->_size / 32) * 32;
		this->finalPos256 = (this->_size / 256) * 256;
	}

	inline void vector<uint8_t>::erase(size_t index)
	{
		uint8_t* data1 = this->_data;
		this->_size--;
		this->finalPos = (this->_size / 32) * 32;
		this->finalPos256 = (this->_size / 256) * 256;
		if (this->dataToDelete == nullptr)
		{
			uint8_t* newData = new uint8_t[this->_size];

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

	// &&

	inline vector<uint8_t> vector<uint8_t>::operator&&(vector<uint8_t>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint8_t* data1 = this->_data;
		uint8_t* data2 = other._data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 32)
		{
			__m256i a = _mm256_loadu_epi8(&data1[i]);
			__m256i b = _mm256_loadu_epi8(&data2[i]);

			_mm256_storeu_epi64(&dataResult[i], _mm256_and_si256(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] & data2[i];
		}
		return result;
	}

	inline vector<uint8_t> vector<uint8_t>::operator&&(uint8_t num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint8_t* data1 = this->_data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		__m256i b = _mm256_set1_epi8(num);

		for (size_t i = 0; i < finalPos; i += 32)
		{
			__m256i a = _mm256_loadu_epi8(&data1[i]);

			_mm256_storeu_epi8(&dataResult[i], _mm256_and_si256(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] & num;
		}
		return result;
	}

	// ||

	inline vector<uint8_t> vector<uint8_t>::operator||(vector<uint8_t>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint8_t* data1 = this->_data;
		uint8_t* data2 = other._data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 32)
		{
			__m256i a = _mm256_loadu_epi8(&data1[i]);
			__m256i b = _mm256_loadu_epi8(&data2[i]);

			_mm256_storeu_epi64(&dataResult[i], _mm256_or_si256(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] | data2[i];
		}
		return result;
	}

	inline vector<uint8_t> vector<uint8_t>::operator||(uint8_t num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint8_t* data1 = this->_data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		__m256i b = _mm256_set1_epi8(num);

		for (size_t i = 0; i < finalPos; i += 32)
		{
			__m256i a = _mm256_loadu_epi8(&data1[i]);

			_mm256_storeu_epi8(&dataResult[i], _mm256_or_si256(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] | num;
		}
		return result;
	}

	// !

	inline vector<uint8_t> vector<uint8_t>::operator!()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint8_t* data1 = this->_data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		__m256i zero = _mm256_setzero_si256();

		for (size_t i = 0; i < finalPos; i += 32)
		{
			__m256i a = _mm256_loadu_epi8(&data1[i]);

			_mm256_storeu_epi8(&dataResult[i], _mm256_and_si256(a, zero));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] & 0;
		}
		return result;
	}

	inline void vector<uint8_t>::self_not()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		uint8_t* data1 = this->_data;

		__m256i zero = _mm256_setzero_si256();

		for (size_t i = 0; i < finalPos; i += 32)
		{
			__m256i a = _mm256_loadu_epi8(&data1[i]);

			_mm256_storeu_epi8(&data1[i], _mm256_and_si256(a, zero));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = data1[i] & 0;
		}
	}

	// Count

	inline uint64_t vector<uint8_t>::count()
	{
		size_t size = this->_size;

		size_t finalPos256 = this->finalPos256;

		uint8_t* data1 = this->_data;

		uint64_t sum = 0;

		for (size_t i = 0; i < finalPos256; i += 256)
		{
			/*
			I am using _mm256_movemask_epi8 to get the 32 bit most significant bit that
			I previously loaded with _mm256_loadu_epi8 and using _mm_popcnt_u32 to
			get the number of bits that are one
			*/
			sum += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[i])));
			sum += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[i + 32])));
			sum += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[i + 64])));
			sum += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[i + 96])));
			sum += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[i + 128])));
			sum += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[i + 160])));
			sum += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[i + 192])));
			sum += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[i + 224])));
		}
		for (size_t i = finalPos256; i < size; i++)
		{
			if (data1[i]) sum++;
		}
		return sum;
	}

	// sort

	inline void vector<uint8_t>::sort()
	{
		std::sort(this->_data, this->_data + this->_size);
	}

	// argsort

	inline vector<uint64_t> vector<uint8_t>::argsort()
	{
		vector<uint64_t> indices(this->_size);

		size_t* indicesData = indices._data;

		for (size_t i = 0; i < this->_size; i++) indicesData[i] = i;

		quicksort(indicesData, this->_data, 0, this->_size - 1);

		return indices;
	}

	// Cast

	template<typename T>
	inline vector<T> vector<uint8_t>::cast()
	{
		size_t size = this->_size;

		uint8_t* data1 = this->_data;

		vector<T> result(size);

		T* dataResult = result._data;

		if constexpr (std::is_same<T, double>::value)
		{
			size_t finalPos = (this->_size / 4) * 4;
			__m256d zero = _mm256_setzero_pd();
			__m256d one = _mm256_set1_pd(1.0);

			for (size_t i = 0; i < finalPos; i += 4)
			{
				_mm256_store_pd(&dataResult[i], _mm256_blendv_pd(zero, one, _mm256_castsi256_pd(_mm256_cvtepi8_epi64(_mm_loadu_epi8(&data1[i])))));
			}
			for (size_t i = finalPos; i < size; i++)
			{
				dataResult[i] = data1[i] ? 1.0 : 0.0;
			}
		}
		else if constexpr (std::is_same<T, float>::value)
		{
			size_t finalPos = (this->_size / 8) * 8;
			__m256 zero = _mm256_setzero_ps();
			__m256 one = _mm256_set1_ps(1.0f);

			for (size_t i = 0; i < finalPos; i += 8)
			{
				_mm256_store_ps(&dataResult[i], _mm256_blendv_ps(zero, one, _mm256_castsi256_ps(_mm256_cvtepi8_epi32(_mm_loadu_epi8(&data1[i])))));
			}
			for (size_t i = finalPos; i < size; i++)
			{
				dataResult[i] = data1[i] ? 1.0f : 0.0f;
			}
		}
		else if constexpr (std::is_same<T, uint64_t>::value)
		{
			size_t finalPos = (this->_size / 4) * 4;
			__m256i zero = _mm256_setzero_si256();
			__m256i one = _mm256_set1_epi64x(1);

			for (size_t i = 0; i < finalPos; i += 4)
			{
				_mm256_storeu_epi64(&dataResult[i], _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(zero), _mm256_castsi256_pd(one), _mm256_castsi256_pd(_mm256_cvtepi8_epi64(_mm_loadu_epi8(&data1[i]))))));
			}
			for (size_t i = finalPos; i < size; i++)
			{
				dataResult[i] = data1[i] ? 1 : 0;
			}
		}
		else if constexpr (std::is_same<T, int>::value)
		{
			size_t finalPos = (this->_size / 8) * 8;
			__m256i zero = _mm256_setzero_si256();
			__m256i one = _mm256_set1_epi32(1);

			for (size_t i = 0; i < finalPos; i += 8)
			{
				_mm256_storeu_epi32(&dataResult[i], _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(zero), _mm256_castsi256_ps(one), _mm256_castsi256_ps(_mm256_cvtepi8_epi32(_mm_loadu_epi8(&data1[i]))))));
			}
			for (size_t i = finalPos; i < size; i++)
			{
				dataResult[i] = data1[i] ? 1 : 0;
			}
		}
		return result;
	}

}
