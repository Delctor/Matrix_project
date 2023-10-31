#include "vectorInt.h"

namespace alge
{
	// Constructor

	inline vector<int>::vector() :
		_data(nullptr),
		dataToDelete(nullptr),
		_size(0),
		finalPos(0),
		_capacity(0) {}

	inline vector<int>::vector(size_t size) :
		_data(new int[size]),
		dataToDelete(_data),
		_size(size),
		finalPos((size / 8) * 8),
		_capacity(size) {}

	inline vector<int>::vector(int* data, size_t size) :
		_data(data),
		dataToDelete(nullptr),
		_size(size),
		finalPos((size / 8) * 8),
		_capacity(size) {}

	inline vector<int>::vector(std::initializer_list<int> list)
	{
		this->_size = list.size();
		this->finalPos = (this->_size / 4) * 4;
		this->_data = new int[this->_size];
		this->dataToDelete = this->_data;
		this->_capacity = this->_size;

		for (size_t i = 0; i < this->_size; i++)
		{
			this->_data[i] = *(list.begin() + i);
		}
	}

	// Destructor

	inline vector<int>::~vector() { delete[] this->dataToDelete; }

	// Block

	inline vector<int> vector<int>::block(size_t initial, size_t final)
	{
		return vector<int>(
			&this->_data[initial],
			final - initial
		);
	}

	// Copy

	inline vector<int> vector<int>::copy()
	{
		size_t size = this->_size;

		vector<int> result(size);

		int* data1 = this->_data;

		int* dataResult = result._data;

		for (size_t i = 0; i < size; i++)
		{
			dataResult[i] = data1[i];
		}
		return result;
	}

	// =

	inline vector<int>& vector<int>::operator=(vector<int>& other)
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
			int* data1 = this->_data;
			int* data2 = other._data;

			for (size_t i = 0; i < size; i++)
			{
				data1[i] = data2[i];
			}
		}
		return *this;
	}

	// Transfer

	inline void vector<int>::transfer(vector<int>& other)
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

	inline int* vector<int>::data() { return this->_data; }

	inline size_t vector<int>::size() { return this->_size; }

	inline int& vector<int>::operator[](size_t index)
	{
		int* data = this->_data;
		return data[index];
	}

	inline const int& vector<int>::operator[](size_t index) const
	{
		int* data = this->_data;
		return data[index];
	}

	inline vector<int> vector<int>::operator[](vector<uint64_t>& indices)
	{
		size_t size = indices._size;

		vector<int> result(size);

		int* data1 = this->_data;

		uint64_t* dataIndices = indices._data;

		int* dataResult = result._data;

		for (size_t i = 0; i < size; i++)
		{
			dataResult[i] = data1[dataIndices[i]];
		}
		return result;
	}

	inline size_t vector<int>::capacity() { return this->_capacity; }

	template<bool reduceCapacity>
	inline void vector<int>::clear()
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

	inline void vector<int>::reserve(size_t newCapacity)
	{
		int* newData = new int[newCapacity];
		int* oldData = this->_data;

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

	inline void vector<int>::append(int num)
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
			int* newData = new int[this->_capacity];
			int* oldData = this->_data;
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

	inline void vector<int>::append(std::initializer_list<int> list)
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
			int* newData = new int[this->_capacity];
			int* oldData = this->_data;

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

	inline void vector<int>::append(vector<int>& other)
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
			int* newData = new int[this->_capacity];
			int* oldData = this->_data;

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

	inline void vector<int>::insert(int num, size_t index)
	{
		if (this->_capacity > this->_size)
		{
			int* data1 = this->_data;

			int tmp = num;
			int tmp2;
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
			int* newData = new int[this->_capacity];
			int* oldData = this->_data;

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

	inline void vector<int>::erase(size_t index)
	{
		int* data1 = this->_data;
		this->_size--;
		this->finalPos = (this->_size / 8) * 8;
		if (this->dataToDelete == nullptr)
		{
			int* newData = new int[this->_size];

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
	inline size_t vector<int>::find(int num)
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

	// neg

	inline vector<int> vector<int>::operator-()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		int* data1 = this->_data;

		__m256i mask = _mm256_set1_epi32(-1);

		vector<int> result(size);

		int* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_storeu_epi32(&dataResult[i], _mm256_sign_epi32(_mm256_loadu_epi32(&data1[i]), mask));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = -data1[i];
		}
		return result;
	}

	inline void vector<int>::self_neg()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		int* data1 = this->_data;

		__m256i mask = _mm256_set1_epi32(0x80000000);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_storeu_epi32(&data1[i], _mm256_xor_si256(_mm256_loadu_epi32(&data1[i]), mask));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = -data1[i];
		}
	}

	// Set Constant

	inline void vector<int>::set_const(int num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		int* data1 = this->_data;

		for (size_t i = 0; i < size; i++)
		{
			data1[i] = num;
		}
	}

	// +

	inline vector<int> vector<int>::operator+(vector<int>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		int* data1 = this->_data;
		int* data2 = other._data;

		vector<int> result(size);

		int* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256i a = _mm256_loadu_epi32(&data1[i]);
			__m256i b = _mm256_loadu_epi32(&data2[i]);

			_mm256_storeu_epi32(&dataResult[i], _mm256_add_epi32(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] + data2[i];
		}
		return result;
	}

	inline vector<int> vector<int>::operator+(int num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		int* data1 = this->_data;

		vector<int> result(size);

		int* dataResult = result._data;

		__m256i b = _mm256_set1_epi64x(num);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256i a = _mm256_loadu_epi32(&data1[i]);

			_mm256_storeu_epi32(&dataResult[i], _mm256_add_epi32(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] + num;
		}
		return result;
	}

	inline void vector<int>::operator+=(vector<int>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		int* data1 = this->_data;
		int* data2 = other._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256i a = _mm256_loadu_epi32(&data1[i]);
			__m256i b = _mm256_loadu_epi32(&data2[i]);

			_mm256_storeu_epi32(&data1[i], _mm256_add_epi32(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] += data2[i];
		}
	}

	inline void vector<int>::operator+=(int num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		int* data1 = this->_data;

		__m256i b = _mm256_set1_epi32(num);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256i a = _mm256_loadu_epi32(&data1[i]);

			_mm256_storeu_epi32(&data1[i], _mm256_add_epi32(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] += num;
		}
	}

	// -

	inline vector<int> vector<int>::operator-(vector<int>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		int* data1 = this->_data;
		int* data2 = other._data;

		vector<int> result(size);

		int* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256i a = _mm256_loadu_epi32(&data1[i]);
			__m256i b = _mm256_loadu_epi32(&data2[i]);

			_mm256_storeu_epi32(&dataResult[i], _mm256_sub_epi32(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] - data2[i];
		}
		return result;
	}

	inline vector<int> vector<int>::operator-(int num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		int* data1 = this->_data;

		vector<int> result(size);

		int* dataResult = result._data;

		__m256i b = _mm256_set1_epi64x(num);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256i a = _mm256_loadu_epi32(&data1[i]);

			_mm256_storeu_epi32(&dataResult[i], _mm256_sub_epi32(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] - num;
		}
		return result;
	}

	inline void vector<int>::operator-=(vector<int>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		int* data1 = this->_data;
		int* data2 = other._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256i a = _mm256_loadu_epi32(&data1[i]);
			__m256i b = _mm256_loadu_epi32(&data2[i]);

			_mm256_storeu_epi32(&data1[i], _mm256_sub_epi32(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] -= data2[i];
		}
	}

	inline void vector<int>::operator-=(int num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		int* data1 = this->_data;

		__m256i b = _mm256_set1_epi32(num);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256i a = _mm256_loadu_epi32(&data1[i]);

			_mm256_storeu_epi32(&data1[i], _mm256_sub_epi32(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] -= num;
		}
	}

	// *

	inline vector<int> vector<int>::operator*(vector<int>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		int* data1 = this->_data;
		int* data2 = other._data;

		vector<int> result(size);

		int* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256i a = _mm256_loadu_epi32(&data1[i]);
			__m256i b = _mm256_loadu_epi32(&data2[i]);

			_mm256_storeu_epi32(&dataResult[i], _mm256_mul_epi32(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] * data2[i];
		}
		return result;
	}

	inline vector<int> vector<int>::operator*(int num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		int* data1 = this->_data;

		vector<int> result(size);

		int* dataResult = result._data;

		__m256i b = _mm256_set1_epi64x(num);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256i a = _mm256_loadu_epi32(&data1[i]);

			_mm256_storeu_epi32(&dataResult[i], _mm256_mul_epi32(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] * num;
		}
		return result;
	}

	inline void vector<int>::operator*=(vector<int>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		int* data1 = this->_data;
		int* data2 = other._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256i a = _mm256_loadu_epi32(&data1[i]);
			__m256i b = _mm256_loadu_epi32(&data2[i]);

			_mm256_storeu_epi32(&data1[i], _mm256_mul_epi32(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] *= data2[i];
		}
	}

	inline void vector<int>::operator*=(int num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		int* data1 = this->_data;

		__m256i b = _mm256_set1_epi32(num);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256i a = _mm256_loadu_epi32(&data1[i]);

			_mm256_storeu_epi32(&data1[i], _mm256_mul_epi32(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] *= num;
		}
	}

	// /

	inline vector<int> vector<int>::operator/(vector<int>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		int* data1 = this->_data;
		int* data2 = other._data;

		vector<int> result(size);

		int* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256i a = _mm256_loadu_epi32(&data1[i]);
			__m256i b = _mm256_loadu_epi32(&data2[i]);

			_mm256_storeu_epi32(&dataResult[i], _mm256_div_epi32(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] / data2[i];
		}
		return result;
	}

	inline vector<int> vector<int>::operator/(int num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		int* data1 = this->_data;

		vector<int> result(size);

		int* dataResult = result._data;

		__m256i b = _mm256_set1_epi64x(num);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256i a = _mm256_loadu_epi32(&data1[i]);

			_mm256_storeu_epi32(&dataResult[i], _mm256_div_epi32(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] / num;
		}
		return result;
	}

	inline void vector<int>::operator/=(vector<int>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		int* data1 = this->_data;
		int* data2 = other._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256i a = _mm256_loadu_epi32(&data1[i]);
			__m256i b = _mm256_loadu_epi32(&data2[i]);

			_mm256_storeu_epi32(&data1[i], _mm256_div_epi32(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] /= data2[i];
		}
	}

	inline void vector<int>::operator/=(int num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		int* data1 = this->_data;

		__m256i b = _mm256_set1_epi32(num);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256i a = _mm256_loadu_epi32(&data1[i]);

			_mm256_storeu_epi32(&data1[i], _mm256_div_epi32(a, b));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] /= num;
		}
	}

	// ==

	inline vector<uint8_t> vector<int>::operator==(vector<int>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		int* data1 = this->_data;
		int* data2 = other._data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		__m256i indices = _mm256_setr_epi32(0, 7, 2, 3, 4, 5, 6, 1);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256i a = _mm256_loadu_epi32(&data1[i]);
			__m256i b = _mm256_loadu_epi32(&data2[i]);

			__m256i mask = _mm256_cmpeq_epi32(a, b);

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

	inline vector<uint8_t> vector<int>::operator==(int num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		int* data1 = this->_data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		__m256i b = _mm256_set1_epi32(num);

		__m256i indices = _mm256_setr_epi32(0, 7, 2, 3, 4, 5, 6, 1);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256i a = _mm256_loadu_epi32(&data1[i]);

			__m256i mask = _mm256_cmpeq_epi32(a, b);

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

	inline vector<uint8_t> vector<int>::operator!=(vector<int>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		int* data1 = this->_data;
		int* data2 = other._data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		__m256i minus_ones = _mm256_set1_epi32(-1);

		__m256i indices = _mm256_setr_epi32(0, 7, 2, 3, 4, 5, 6, 1);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256i a = _mm256_loadu_epi32(&data1[i]);
			__m256i b = _mm256_loadu_epi32(&data2[i]);

			__m256i mask = _mm256_andnot_si256(_mm256_cmpeq_epi32(a, b), minus_ones);
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

	inline vector<uint8_t> vector<int>::operator!=(int num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		int* data1 = this->_data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		__m256i b = _mm256_set1_epi32(num);

		__m256i minus_ones = _mm256_set1_epi32(-1);

		__m256i indices = _mm256_setr_epi32(0, 7, 2, 3, 4, 5, 6, 1);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256i a = _mm256_loadu_epi32(&data1[i]);

			__m256i mask = _mm256_andnot_si256((_mm256_cmpeq_epi32(a, b)), minus_ones);
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

	inline vector<uint8_t> vector<int>::operator>(vector<int>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		int* data1 = this->_data;
		int* data2 = other._data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		__m256i indices = _mm256_setr_epi32(0, 7, 2, 3, 4, 5, 6, 1);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256i a = _mm256_loadu_epi32(&data1[i]);
			__m256i b = _mm256_loadu_epi32(&data2[i]);

			__m256i mask = _mm256_cmpgt_epi32(a, b);
			__m256i mask1 = _mm256_packs_epi32(mask, mask);
			__m256i mask2 = _mm256_packs_epi16(mask1, mask1);

			mask2 = _mm256_permutevar8x32_epi32(mask2, indices);

			_mm_store_sd(reinterpret_cast<double*>(&dataResult[i]), _mm_castsi128_pd(_mm256_castsi256_si128(mask2)));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] > data2[i] ? True : False;
		}
		return result;
	}

	inline vector<uint8_t> vector<int>::operator>(int num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		int* data1 = this->_data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		__m256i b = _mm256_set1_epi32(num);

		__m256i indices = _mm256_setr_epi32(0, 7, 2, 3, 4, 5, 6, 1);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256i a = _mm256_loadu_epi32(&data1[i]);

			__m256i mask = _mm256_cmpgt_epi32(a, b);
			__m256i mask1 = _mm256_packs_epi32(mask, mask);
			__m256i mask2 = _mm256_packs_epi16(mask1, mask1);

			mask2 = _mm256_permutevar8x32_epi32(mask2, indices);

			_mm_store_sd(reinterpret_cast<double*>(&dataResult[i]), _mm_castsi128_pd(_mm256_castsi256_si128(mask2)));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] > num ? True : False;
		}
		return result;
	}

	// < 

	inline vector<uint8_t> vector<int>::operator<(vector<int>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		int* data1 = this->_data;
		int* data2 = other._data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		__m256i minus_ones = _mm256_set1_epi32(-1);

		__m256i indices = _mm256_setr_epi32(0, 7, 2, 3, 4, 5, 6, 1);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256i a = _mm256_loadu_epi32(&data1[i]);
			__m256i b = _mm256_loadu_epi32(&data2[i]);

			__m256i gt = _mm256_cmpgt_epi32(a, b);
			__m256i eq = _mm256_cmpeq_epi32(a, b);

			__m256i mask = _mm256_andnot_si256(gt, _mm256_andnot_si256(eq, minus_ones));
			__m256i mask1 = _mm256_packs_epi32(mask, mask);
			__m256i mask2 = _mm256_packs_epi16(mask1, mask1);

			mask2 = _mm256_permutevar8x32_epi32(mask2, indices);

			_mm_store_sd(reinterpret_cast<double*>(&dataResult[i]), _mm_castsi128_pd(_mm256_castsi256_si128(mask2)));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] < data2[i] ? True : False;
		}
		return result;
	}

	inline vector<uint8_t> vector<int>::operator<(int num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		int* data1 = this->_data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		__m256i b = _mm256_set1_epi32(num);

		__m256i minus_ones = _mm256_set1_epi32(-1);

		__m256i indices = _mm256_setr_epi32(0, 7, 2, 3, 4, 5, 6, 1);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256i a = _mm256_loadu_epi32(&data1[i]);

			__m256i gt = _mm256_cmpgt_epi32(a, b);
			__m256i eq = _mm256_cmpeq_epi32(a, b);

			__m256i mask = _mm256_andnot_si256(gt, _mm256_andnot_si256(eq, minus_ones));

			__m256i mask1 = _mm256_packs_epi32(mask, mask);
			__m256i mask2 = _mm256_packs_epi16(mask1, mask1);

			mask2 = _mm256_permutevar8x32_epi32(mask2, indices);

			_mm_store_sd(reinterpret_cast<double*>(&dataResult[i]), _mm_castsi128_pd(_mm256_castsi256_si128(mask2)));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] < num ? True : False;
		}
		return result;
	}

	// >=

	inline vector<uint8_t> vector<int>::operator>=(vector<int>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		int* data1 = this->_data;
		int* data2 = other._data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		__m256i indices = _mm256_setr_epi32(0, 7, 2, 3, 4, 5, 6, 1);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256i a = _mm256_loadu_epi32(&data1[i]);
			__m256i b = _mm256_loadu_epi32(&data2[i]);

			__m256i gt = _mm256_cmpgt_epi32(a, b);
			__m256i eq = _mm256_cmpeq_epi32(a, b);

			__m256i mask = _mm256_or_si256(gt, eq);
			__m256i mask1 = _mm256_packs_epi32(mask, mask);
			__m256i mask2 = _mm256_packs_epi16(mask1, mask1);

			mask2 = _mm256_permutevar8x32_epi32(mask2, indices);

			_mm_store_sd(reinterpret_cast<double*>(&dataResult[i]), _mm_castsi128_pd(_mm256_castsi256_si128(mask2)));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] >= data2[i] ? True : False;
		}
		return result;
	}

	inline vector<uint8_t> vector<int>::operator>=(int num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		int* data1 = this->_data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		__m256i b = _mm256_set1_epi32(num);

		__m256i indices = _mm256_setr_epi32(0, 7, 2, 3, 4, 5, 6, 1);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256i a = _mm256_loadu_epi32(&data1[i]);

			__m256i gt = _mm256_cmpgt_epi32(a, b);
			__m256i eq = _mm256_cmpeq_epi32(a, b);

			__m256i mask = _mm256_or_si256(gt, eq);
			__m256i mask1 = _mm256_packs_epi32(mask, mask);
			__m256i mask2 = _mm256_packs_epi16(mask1, mask1);

			mask2 = _mm256_permutevar8x32_epi32(mask2, indices);

			_mm_store_sd(reinterpret_cast<double*>(&dataResult[i]), _mm_castsi128_pd(_mm256_castsi256_si128(mask2)));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] >= num ? True : False;
		}
		return result;
	}

	// <=

	inline vector<uint8_t> vector<int>::operator<=(vector<int>& other)
	{
#ifdef _DEBUG
		if (this->_size != other._size) throw std::invalid_argument("The dimensions of both vectors must be the same");
#else
#endif
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		int* data1 = this->_data;
		int* data2 = other._data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		__m256i minus_ones = _mm256_set1_epi32(-1);

		__m256i indices = _mm256_setr_epi32(0, 7, 2, 3, 4, 5, 6, 1);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256i a = _mm256_loadu_epi32(&data1[i]);
			__m256i b = _mm256_loadu_epi32(&data2[i]);

			__m256i mask = _mm256_andnot_si256(_mm256_cmpgt_epi32(a, b), minus_ones);
			__m256i mask1 = _mm256_packs_epi32(mask, mask);
			__m256i mask2 = _mm256_packs_epi16(mask1, mask1);

			mask2 = _mm256_permutevar8x32_epi32(mask2, indices);

			_mm_store_sd(reinterpret_cast<double*>(&dataResult[i]), _mm_castsi128_pd(_mm256_castsi256_si128(mask2)));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] <= data2[i] ? True : False;
		}
		return result;
	}

	inline vector<uint8_t> vector<int>::operator<=(int num)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		int* data1 = this->_data;

		vector<uint8_t> result(size);

		uint8_t* dataResult = result._data;

		__m256i b = _mm256_set1_epi32(num);

		__m256i minus_ones = _mm256_set1_epi32(-1);

		__m256i indices = _mm256_setr_epi32(0, 7, 2, 3, 4, 5, 6, 1);

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256i a = _mm256_loadu_epi32(&data1[i]);

			__m256i mask = _mm256_andnot_si256(_mm256_cmpgt_epi32(a, b), minus_ones);
			__m256i mask1 = _mm256_packs_epi32(mask, mask);
			__m256i mask2 = _mm256_packs_epi16(mask1, mask1);

			mask2 = _mm256_permutevar8x32_epi32(mask2, indices);

			_mm_store_sd(reinterpret_cast<double*>(&dataResult[i]), _mm_castsi128_pd(_mm256_castsi256_si128(mask2)));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = data1[i] <= num ? True : False;
		}
		return result;
	}

	// Pow

	inline vector<int> vector<int>::pow(int exponent)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		int* data1 = this->_data;

		vector<int> result(size);

		int* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			__m256i result_pow = _mm256_set1_epi32(1);
			__m256i base = _mm256_loadu_epi32(&data1[i]);
			int exp = exponent;
			while (exp > 0)
			{
				if (exp % 2 == 1)
				{
					result_pow = _mm256_mul_epi32(result_pow, base);
				}

				base = _mm256_mul_epi32(base, base);
				exp >>= 1;
			}
			_mm256_storeu_epi32(&dataResult[i], result_pow);
		}
		for (size_t i = finalPos; i < size; i++)
		{
			int result_pow = 1;
			int base = data1[i];
			int exp = exponent;
			while (exp > 0)
			{
				if (exp % 2 == 1)
				{
					result_pow = result_pow * base;
				}

				base = base * base;
				exp >>= 1;
			}
			dataResult[i] = result_pow;
		}
		return result;
	}

	inline vector<int> vector<int>::pow(vector<int>& other)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		int* data1 = this->_data;
		int* data2 = other._data;

		vector<int> result(size);

		int* dataResult = result._data;

		__m256i one = _mm256_set1_epi32(1);
		__m256i zero = _mm256_setzero_si256();

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256i result_pow = _mm256_set1_epi32(1);

			__m256i base = _mm256_loadu_epi32(&data1[i]);
			__m256i exps = _mm256_loadu_epi32(&data2[i]);

			while (_mm256_movemask_epi8(_mm256_cmpgt_epi32(exps, zero)))
			{
				__m256i mask = _mm256_cmpeq_epi32(_mm256_and_si256(exps, one), one);
				result_pow = _mm256_blendv_epi8(result_pow, _mm256_mul_epi32(result_pow, base), mask);
				base = _mm256_mul_epi32(base, base);
				exps = _mm256_srli_epi32(exps, 1);
			}
			_mm256_storeu_epi32(&dataResult[i], result_pow);
		}
		for (size_t i = finalPos; i < size; i++)
		{
			int result_pow = 1;
			int base = data1[i];
			int exp = data2[i];
			while (exp > 0)
			{
				if (exp % 2 == 1)
				{
					result_pow = result_pow * base;
				}

				base = base * base;
				exp >>= 1;
			}
			dataResult[i] = result_pow;
		}
		return result;
	}

	inline void vector<int>::self_pow(int exponent)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		int* data1 = this->_data;

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256i result_pow = _mm256_set1_epi32(1);
			__m256i base = _mm256_loadu_epi32(&data1[i]);
			int exp = exponent;
			while (exp > 0)
			{
				if (exp % 2 == 1)
				{
					result_pow = _mm256_mul_epi32(result_pow, base);
				}

				base = _mm256_mul_epi32(base, base);
				exp >>= 1;
			}
			_mm256_storeu_epi32(&data1[i], result_pow);
		}
		for (size_t i = finalPos; i < size; i++)
		{
			int result_pow = 1;
			int base = data1[i];
			int exp = exponent;
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

	inline void vector<int>::self_pow(vector<int>& other)
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		int* data1 = this->_data;
		int* data2 = other._data;

		__m256i one = _mm256_set1_epi32(1);
		__m256i zero = _mm256_setzero_si256();

		for (size_t i = 0; i < finalPos; i += 4)
		{
			__m256i result_pow = _mm256_set1_epi32(1);

			__m256i base = _mm256_loadu_epi32(&data1[i]);
			__m256i exps = _mm256_loadu_epi32(&data2[i]);

			while (_mm256_movemask_epi8(_mm256_cmpgt_epi32(exps, zero)))
			{
				__m256i mask = _mm256_cmpeq_epi32(_mm256_and_si256(exps, one), one);
				result_pow = _mm256_blendv_epi8(result_pow, _mm256_mul_epi32(result_pow, base), mask);
				base = _mm256_mul_epi32(base, base);
				exps = _mm256_srli_epi32(exps, 1);
			}
			_mm256_storeu_epi32(&data1[i], result_pow);
		}
		for (size_t i = finalPos; i < size; i++)
		{
			int result_pow = 1;
			int base = data1[i];
			int exp = data2[i];
			while (exp > 0)
			{
				if (exp % 2 == 1)
				{
					result_pow = result_pow * base;
				}

				base = base * base;
				exp >>= 1;
			}
			data1[i] = result_pow;
		}
	}

	// Abs

	inline vector<int> vector<int>::abs()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		int* data1 = this->_data;

		vector<int> result(size);

		int* dataResult = result._data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_storeu_epi32(&dataResult[i], _mm256_abs_epi32(_mm256_loadu_epi32(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			dataResult[i] = std::abs(data1[i]);
		}
		return result;
	}

	inline void vector<int>::self_abs()
	{
		size_t size = this->_size;

		size_t finalPos = this->finalPos;

		int* data1 = this->_data;

		for (size_t i = 0; i < finalPos; i += 8)
		{
			_mm256_storeu_epi32(&data1[i], _mm256_abs_epi32(_mm256_loadu_epi32(&data1[i])));
		}
		for (size_t i = finalPos; i < size; i++)
		{
			data1[i] = std::abs(data1[i]);
		}
	}

	// Sort

	inline void vector<int>::sort()
	{
		std::sort(this->_data, this->_data + this->_size);
	}

	// Argsort

	inline vector<uint64_t> vector<int>::argsort()
	{
		vector<uint64_t> indices(this->_size);

		size_t* indicesData = indices._data;

		for (size_t i = 0; i < this->_size; i++) indicesData[i] = i;

		quicksort(indicesData, this->_data, 0, this->_size - 1);

		return indices;
	}

	// Cast

	template<typename T>
	inline vector<T> vector<int>::cast()
	{
		size_t size = this->_size;

		int* data1 = this->_data;

		vector<T> result(size);

		T* dataResult = result._data;

		if constexpr (std::is_same<T, uint64_t>::value)
		{
			size_t finalPos = (this->_size / 4) * 4;
			for (size_t i = 0; i < finalPos; i += 4)
			{
				_mm256_storeu_epi64(&dataResult[i], _mm256_cvtepi32_epi64(_mm_loadu_epi32(&data1[i])));
			}
			for (size_t i = finalPos; i < size; i++)
			{
				dataResult[i] = static_cast<uint64_t>(data1[i]);
			}
		}
		else if constexpr (std::is_same<T, float>::value)
		{
			size_t finalPos = this->finalPos;
			for (size_t i = 0; i < finalPos; i += 8)
			{
				_mm256_storeu_ps(&dataResult[i], _mm256_cvtepi32_ps(_mm256_loadu_epi32(&data1[i])));
			}
			for (size_t i = finalPos; i < size; i++)
			{
				dataResult[i] = static_cast<float>(data1[i]);
			}
		}
		else if constexpr (std::is_same<T, double>::value)
		{
			size_t finalPos = (this->_size / 4) * 4;
			for (size_t i = 0; i < finalPos; i += 4)
			{
				_mm256_store_pd(&dataResult[i], _mm256_cvtepi32_pd(_mm_loadu_epi32(&data1[i])));
			}
			for (size_t i = finalPos; i < size; i++)
			{
				dataResult[i] = static_cast<double>(data1[i]);
			}
		}
		else if constexpr (std::is_same<T, uint8_t>::value)
		{
			size_t finalPos = this->finalPos;
			__m256 zero = _mm256_setzero_ps();
			__m256i indices = _mm256_setr_epi32(0, 7, 2, 3, 4, 5, 6, 1);
			for (size_t i = 0; i < finalPos; i += 8)
			{
				__m256i mask = _mm256_castps_si256(_mm256_cmp_ps(_mm256_castsi256_ps(_mm256_loadu_epi32(&data1[i])), zero, _CMP_NEQ_OQ));
				__m256i mask1 = _mm256_packs_epi32(mask, mask);
				__m256i mask2 = _mm256_packs_epi16(mask1, mask1);

				mask2 = _mm256_permutevar8x32_epi32(mask2, indices);

				_mm_store_sd(reinterpret_cast<double*>(&dataResult[i]), _mm_castsi128_pd(_mm256_castsi256_si128(mask2)));
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