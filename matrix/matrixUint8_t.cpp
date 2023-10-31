#include "matrixUint8_t.h"

namespace alge
{
	template <bool thisTransposed, bool thisContiguous>
	inline matrix<uint8_t, thisTransposed, thisContiguous>::matrix() :
		_data(nullptr),
		dataToDelete(nullptr),
		_rows(0),
		_cols(0),
		_size(0),
		actualRows(0),
		actualCols(0),
		finalPosSize(0),
		finalPosRows(0),
		finalPosCols(0),
		finalPosSize256(0),
		finalPosRows256(0),
		finalPosCols256(0) {}

	template <bool thisTransposed, bool thisContiguous>
	inline matrix<uint8_t, thisTransposed, thisContiguous>::matrix(size_t rows, size_t cols) :
		_data(new uint8_t[rows * cols]),
		dataToDelete(_data),
		_rows(rows),
		_cols(cols),
		_size(rows* cols),
		actualRows(rows),
		actualCols(cols),
		finalPosSize((_size / 32) * 32),
		finalPosRows((rows / 32) * 32),
		finalPosCols((cols / 32) * 32),
		finalPosSize256((_size / 256) * 256),
		finalPosRows256((rows / 256) * 256),
		finalPosCols256((cols / 256) * 256) {}

	template <bool thisTransposed, bool thisContiguous>
	inline matrix<uint8_t, thisTransposed, thisContiguous>::matrix(uint8_t* data, size_t rows, size_t cols, size_t actualRows, size_t actualCols) :
		_data(data),
		dataToDelete(nullptr),
		_rows(rows),
		_cols(cols),
		_size(rows* cols),
		actualRows(actualRows),
		actualCols(actualCols),
		finalPosSize((_size / 32) * 32),
		finalPosRows((rows / 32) * 32),
		finalPosCols((cols / 32) * 32),
		finalPosSize256((_size / 256) * 256),
		finalPosRows256((rows / 256) * 256),
		finalPosCols256((cols / 256) * 256) {}

	template <bool thisTransposed, bool thisContiguous>
	inline matrix<uint8_t, thisTransposed, thisContiguous>::matrix(std::initializer_list<std::initializer_list<double>> list)
	{
		this->_rows = list.size();
		this->_cols = (*list.begin()).size();
		this->actualRows = this->_rows;
		this->actualCols = this->_cols;
		this->_size = this->_rows * this->_cols;
		this->_data = new uint8_t[this->_size];
		this->dataToDelete = this->_data;
		this->finalPosRows = (this->_rows / 4) * 4;
		this->finalPosCols = (this->_cols / 4) * 4;
		this->finalPosSize = (this->_size / 4) * 4;
		this->_capacityRows = thisTransposed ? this->_cols : this->_rows;

		if constexpr (thisTransposed)
		{
			for (size_t i = 0; i < this->_rows; i++)
			{
				std::initializer_list<uint8_t> listI = *(list.begin() + i);
				for (size_t j = 0; j < this->_cols; j++)
				{
					this->_data[j * this->actualRows + i] = *(listI.begin() + j);
				}
			}
		}
		else
		{
			for (size_t i = 0; i < this->_rows; i++)
			{
				std::initializer_list<uint8_t> listI = *(list.begin() + i);
				for (size_t j = 0; j < this->_cols; j++)
				{
					this->_data[i * this->actualCols + j] = *(listI.begin() + j);
				}
			}
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline matrix<uint8_t, thisTransposed, thisContiguous>::~matrix() { delete[] this->dataToDelete; }

	template <bool thisTransposed, bool thisContiguous>
	inline size_t matrix<uint8_t, thisTransposed, thisContiguous>::rows() { return this->_rows; };

	template <bool thisTransposed, bool thisContiguous>
	inline size_t matrix<uint8_t, thisTransposed, thisContiguous>::cols() { return this->_cols; };

	template <bool thisTransposed, bool thisContiguous>
	inline uint8_t* matrix<uint8_t, thisTransposed, thisContiguous>::data() { return this->_data; };

	template <bool thisTransposed, bool thisContiguous>
	inline matrix<uint8_t, thisTransposed, thisContiguous> matrix<uint8_t, thisTransposed, thisContiguous>::row(size_t row)
	{
		if constexpr (thisTransposed)
		{
			return matrix<uint8_t, true, thisContiguous>(
				&this->_data[row],
				1,
				this->_cols,
				this->actualRows,
				this->actualCols);
		}
		else
		{
			return matrix<uint8_t, false, thisContiguous>(
				&this->_data[row * this->actualCols],
				1,
				this->_cols,
				this->actualRows,
				this->actualCols);
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline matrix<uint8_t, thisTransposed, thisContiguous> matrix<uint8_t, thisTransposed, thisContiguous>::col(size_t col)
	{
		if constexpr (thisTransposed)
		{
			return matrix<uint8_t, true, thisContiguous>(
				&this->_data[col * this->actualRows],
				this->_rows,
				1,
				this->actualRows,
				this->actualCols);
		}
		else
		{
			return matrix<uint8_t, false, thisContiguous>(
				&this->_data[col],
				this->_rows,
				1,
				this->actualRows,
				this->actualCols);
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline matrix<uint8_t, !thisTransposed, thisContiguous> matrix<uint8_t, thisTransposed, thisContiguous>::tranpose()
	{
		return matrix<uint8_t, !thisTransposed, thisContiguous>(
			this->_data,
			this->_cols,
			this->_rows,
			this->actualCols,
			this->actualRows
		);
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool blockContiguous>
	inline matrix<uint8_t, thisTransposed, thisContiguous && blockContiguous> matrix<uint8_t, thisTransposed, thisContiguous>::block(size_t initial_row, size_t initial_col, size_t final_row, size_t final_col)
	{
		if constexpr (thisTransposed)
		{
			return matrix<uint8_t, true, thisContiguous&& blockContiguous>(
				&this->_data[initial_col * this->actualRows + initial_row],
				final_row - initial_row,
				final_col - initial_col,
				final_row - initial_row,
				final_col - initial_col
			);
		}
		else
		{
			return matrix<uint8_t, false, thisContiguous&& blockContiguous>(
				&this->_data[initial_row * this->actualCols + initial_col],
				final_row - initial_row,
				final_col - initial_col,
				final_row - initial_row,
				final_col - initial_col
			);
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline uint8_t& matrix<uint8_t, thisTransposed, thisContiguous>::operator()(size_t row, size_t col)
	{
		if constexpr (thisTransposed)
		{
			return this->_data[col * this->actualRows + row];
		}
		else
		{
			return this->_data[row * this->actualCols + col];
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline const uint8_t& matrix<uint8_t, thisTransposed, thisContiguous>::operator()(size_t row, size_t col) const
	{
		if constexpr (thisTransposed)
		{
			return this->_data[col * this->actualRows + row];
		}
		else
		{
			return this->_data[row * this->actualCols + col];
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline size_t matrix<uint8_t, thisTransposed, thisContiguous>::capacity() { return this->_capacityRows; }

	template <bool thisTransposed, bool thisContiguous>
	template<bool reduceCapacity>
	inline void matrix<uint8_t, thisTransposed, thisContiguous>::clear()
	{
		if constexpr (reduceCapacity)
		{
			this->_size = 0;
			this->finalPosSize = 0;
			this->finalPosRows = 0;
			this->finalPosCols = 0;
			this->finalPosCols256 = 0;
			this->finalPosRows256 = 0;
			this->finalPosSize256 = 0;
			this->_rows = 0;
			this->_cols = 0;

			this->_capacityRows = 0;
			delete[] this->dataToDelete;
			this->_data = nullptr;
			this->dataToDelete = nullptr;
		}
		else
		{
			this->_size = 0;
			this->finalPosSize = 0;
			this->finalPosRows = 0;
			this->finalPosCols = 0;
			this->finalPosCols256 = 0;
			this->finalPosRows256 = 0;
			this->finalPosSize256 = 0;
			this->_rows = 0;
			this->_cols = 0;
		}

	}

	template <bool thisTransposed, bool thisContiguous>
	inline void matrix<uint8_t, thisTransposed, thisContiguous>::reserve(size_t newCapacity)
	{
		if constexpr (thisTransposed)
		{
			uint8_t* newData = new uint8_t[newCapacity * this->_rows];
			uint8_t* oldData = this->_data;

			this->_cols = this->_cols <= newCapacity ? this->_cols : newCapacity;

			this->_size = this->_rows * this->_cols;
			this->finalPosSize = (this->_size / 32) * 32;
			this->finalPosSize256 = (this->_size / 256) * 256;
			this->finalPoscols = (this->_cols / 32) * 32;
			this->finalPosCols256 = (this->_cols / 256) * 256;

			this->_capacityRows = newCapacity;

			for (size_t i = 0; i < this->_cols; i++)
			{
				for (size_t j = 0; j < this->_rows; j++)
				{
					newData[i * this->_rows + j] = oldData[i * this->actualRows + j];
				}
			}

			delete[] this->dataToDelete;
			this->_data = newData;
			this->dataToDelete = newData;
		}
		else
		{
			uint8_t* newData = new uint8_t[newCapacity * this->_cols];
			uint8_t* oldData = this->_data;

			this->_rows = this->_rows <= newCapacity ? this->_rows : newCapacity;

			this->_size = this->_rows * this->_cols;
			this->finalPosSize = (this->_size / 32) * 32;
			this->finalPosSize256 = (this->_size / 256) * 256;
			this->finalPosRows = (this->_rows / 32) * 32;
			this->finalPosRows256 = (this->_rows / 256) * 256;

			this->_capacityRows = newCapacity;

			for (size_t i = 0; i < this->_rows; i++)
			{
				for (size_t j = 0; j < this->_cols; j++)
				{
					newData[i * this->_cols + j] = oldData[i * this->actualCols + j];
				}
			}

			delete[] this->dataToDelete;
			this->_data = newData;
			this->dataToDelete = newData;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline void matrix<uint8_t, thisTransposed, thisContiguous>::append(std::initializer_list<std::initializer_list<uint8_t>> list)
	{
		size_t sizeList = list.size();

		if constexpr (thisTransposed)
		{
			size_t newCols = this->_cols + sizeList;

			if (this->_capacityRows >= newCols)
			{
				for (size_t i{ this->_cols }, i2{ 0 }; i < newCols; i++, i2++)
				{
					std::initializer_list<uint8_t> listI = *(list.begin() + i2);
					for (size_t j = 0; j < this->_rows; j++)
					{
						this->_data[i * this->actualRows + j] = *(listI.begin() + j);
					}
				}
			}
			else
			{
				size_t increase = this->_capacityRows / 2;
				increase = increase >= sizeList ? increase : sizeList;
				this->_capacityRows += increase;

				uint8_t* newData = new uint8_t[this->_capacityRows * this->_rows];
				uint8_t* oldData = this->_data;

				for (size_t i = 0; i < this->_cols; i++)
				{
					for (size_t j = 0; j < this->_rows; j++)
					{
						newData[i * this->_rows + j] = oldData[i * this->actualRows + j];
					}
				}
				for (size_t i{ this->_cols }, i2{ 0 }; i < newCols; i++, i2++)
				{
					std::initializer_list<uint8_t> listI = *(list.begin() + i2);
					for (size_t j = 0; j < this->_rows; j++)
					{
						newData[i * this->_rows + j] = *(listI.begin() + j);
					}
				}

				delete[] this->dataToDelete;

				this->_data = newData;
				this->dataToDelete = newData;
			}

			this->_cols = newCols;
			this->actualCols = newCols;
			this->_size = this->_rows * this->_cols;
			this->finalPosSize = (this->_size / 32) * 32;
			this->finalPosCols = (this->_cols / 32) * 32;
			this->finalPosSize256 = (this->_size / 256) * 256;
			this->finalPosCols256 = (this->_cols / 256) * 256;
		}
		else
		{
			size_t newRows = this->_rows + sizeList;

			if (this->_capacityRows >= newRows)
			{
				for (size_t i{ this->_rows }, i2{ 0 }; i < newRows; i++, i2++)
				{
					std::initializer_list<uint8_t> listI = *(list.begin() + i2);
					for (size_t j = 0; j < this->_cols; j++)
					{
						this->_data[i * this->actualCols + j] = *(listI.begin() + j);
					}
				}
			}
			else
			{
				size_t increase = this->_capacityRows / 2;
				increase = increase >= sizeList ? increase : sizeList;
				this->_capacityRows += increase;

				uint8_t* newData = new uint8_t[this->_capacityRows * this->_cols];
				uint8_t* oldData = this->_data;

				for (size_t i = 0; i < this->_rows; i++)
				{
					for (size_t j = 0; j < this->_cols; j++)
					{
						newData[i * this->_cols + j] = oldData[i * this->actualCols + j];
					}
				}
				for (size_t i{ this->_rows }, i2{ 0 }; i < newRows; i++, i2++)
				{
					std::initializer_list<uint8_t> listI = *(list.begin() + i2);
					for (size_t j = 0; j < this->_cols; j++)
					{
						newData[i * this->_cols + j] = *(listI.begin() + j);
					}
				}

				delete[] this->dataToDelete;

				this->_data = newData;
				this->dataToDelete = newData;
			}
			this->_rows = newRows;
			this->actualRows = newRows;
			this->_size = this->_rows * this->_cols;
			this->finalPosSize = (this->_size / 32) * 32;
			this->finalPosRows = (this->_rows / 32) * 32;
			this->finalPosSize256 = (this->_size / 256) * 256;
			this->finalPosRows256 = (this->_rows / 256) * 256;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool otherTransposed, bool otherContiguous>
	inline void matrix<uint8_t, thisTransposed, thisContiguous>::append(matrix<uint8_t, otherTransposed, otherContiguous>& other)
	{
		size_t sizeOther = other._rows;

		if constexpr (thisTransposed)
		{
#ifdef _DEBUG
			if (this->_rows != other._rows) throw std::invalid_argument("Error");
#else
#endif
			size_t newCols = this->_cols + sizeOther;

			if constexpr (otherTransposed)
			{
				if (this->_capacityRows >= newCols)
				{
					for (size_t i{ this->_cols }, i2{ 0 }; i < newCols; i++, i2++)
					{
						for (size_t j = 0; j < this->_rows; j++)
						{
							this->_data[i * this->actualRows + j] = other._data[i2 * other.actualRows + j];
						}
					}
				}
				else
				{
					size_t increase = this->_capacityRows / 2;
					increase = increase >= sizeOther ? increase : sizeOther;
					this->_capacityRows += increase;

					uint8_t* newData = new uint8_t[this->_capacityRows * this->_rows];
					uint8_t* oldData = this->_data;

					for (size_t i = 0; i < this->_cols; i++)
					{
						for (size_t j = 0; j < this->_rows; j++)
						{
							newData[i * this->_rows + j] = oldData[i * this->actualRows + j];
						}
					}
					for (size_t i{ this->_cols }, i2{ 0 }; i < newCols; i++, i2++)
					{
						for (size_t j = 0; j < this->_rows; j++)
						{
							newData[i * this->_rows + j] = other._data[i2 * other.actualRows + j];
						}
					}

					delete[] this->dataToDelete;

					this->_data = newData;
					this->dataToDelete = newData;
				}
			}
			else
			{
				if (this->_capacityRows >= newCols)
				{
					for (size_t i{ this->_cols }, i2{ 0 }; i < newCols; i++, i2++)
					{
						for (size_t j = 0; j < this->_rows; j++)
						{
							this->_data[i * this->actualRows + j] = other._data[j * other.actualCols + i2];
						}
					}
				}
				else
				{
					size_t increase = this->_capacityRows / 2;
					increase = increase >= sizeOther ? increase : sizeOther;
					this->_capacityRows += increase;

					uint8_t* newData = new uint8_t[this->_capacityRows * this->_rows];
					uint8_t* oldData = this->_data;

					for (size_t i = 0; i < this->_cols; i++)
					{
						for (size_t j = 0; j < this->_rows; j++)
						{
							newData[i * this->_rows + j] = oldData[i * this->actualRows + j];
						}
					}
					for (size_t i{ this->_cols }, i2{ 0 }; i < newCols; i++, i2++)
					{
						for (size_t j = 0; j < this->_rows; j++)
						{
							newData[i * this->_rows + j] = other._data[j * other.actualCols + i2];
						}
					}

					delete[] this->dataToDelete;

					this->_data = newData;
					this->dataToDelete = newData;
				}
			}

			this->_cols = newCols;
			this->actualCols = newCols;
			this->_size = this->_rows * this->_cols;
			this->finalPosSize = (this->_size / 32) * 32;
			this->finalPosCols = (this->_cols / 32) * 32;
			this->finalPosSize256 = (this->_size / 256) * 256;
			this->finalPosCols256 = (this->_cols / 256) * 256;
		}
		else
		{
#ifdef _DEBUG
			if (this->_cols != other._cols) throw std::invalid_argument("Error");
#else
#endif
			size_t newRows = this->_rows + sizeOther;

			if constexpr (otherTransposed)
			{
				if (this->_capacityRows >= newRows)
				{
					for (size_t i{ this->_rows }, i2{ 0 }; i < newRows; i++, i2++)
					{
						for (size_t j = 0; j < this->_cols; j++)
						{
							this->_data[i * this->actualCols + j] = other._data[j * other.actualRows + i2];
						}
					}
				}
				else
				{
					size_t increase = this->_capacityRows / 2;
					increase = increase >= sizeOther ? increase : sizeOther;
					this->_capacityRows += increase;

					uint8_t* newData = new uint8_t[this->_capacityRows * this->_cols];
					uint8_t* oldData = this->_data;

					for (size_t i = 0; i < this->_rows; i++)
					{
						for (size_t j = 0; j < this->_cols; j++)
						{
							newData[i * this->_cols + j] = oldData[i * this->actualCols + j];
						}
					}
					for (size_t i{ this->_rows }, i2{ 0 }; i < newRows; i++, i2++)
					{
						for (size_t j = 0; j < this->_cols; j++)
						{
							newData[i * this->_cols + j] = other._data[j * other.actualRows + i2];
						}
					}

					delete[] this->dataToDelete;

					this->_data = newData;
					this->dataToDelete = newData;
				}
			}
			else
			{
				if (this->_capacityRows >= newRows)
				{
					for (size_t i{ this->_rows }, i2{ 0 }; i < newRows; i++, i2++)
					{
						for (size_t j = 0; j < this->_cols; j++)
						{
							this->_data[i * this->actualCols + j] = other._data[i2 * other.actualCols + j];
						}
					}
				}
				else
				{
					size_t increase = this->_capacityRows / 2;
					increase = increase >= sizeOther ? increase : sizeOther;
					this->_capacityRows += increase;

					uint8_t* newData = new uint8_t[this->_capacityRows * this->_cols];
					uint8_t* oldData = this->_data;

					for (size_t i = 0; i < this->_rows; i++)
					{
						for (size_t j = 0; j < this->_cols; j++)
						{
							newData[i * this->_cols + j] = oldData[i * this->actualCols + j];
						}
					}
					for (size_t i{ this->_rows }, i2{ 0 }; i < newRows; i++, i2++)
					{
						for (size_t j = 0; j < this->_cols; j++)
						{
							newData[i * this->_cols + j] = other._data[i2 * other.actualCols + j];
						}
					}

					delete[] this->dataToDelete;

					this->_data = newData;
					this->dataToDelete = newData;
				}
			}

			this->_rows = newRows;
			this->actualRows = newRows;
			this->_size = this->_rows * this->_cols;
			this->finalPosSize = (this->_size / 32) * 32;
			this->finalPosRows = (this->_rows / 32) * 32;
			this->finalPosSize256 = (this->_size / 256) * 256;
			this->finalPosRows256 = (this->_rows / 256) * 256;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline void matrix<uint8_t, thisTransposed, thisContiguous>::erase(size_t index)
	{
		if constexpr (thisTransposed)
		{
			this->_cols--;
			this->actualCols--;
			this->_size = this->_rows * this->_cols;
			this->finalPosCols = (this->_cols / 32) * 32;
			this->finalPosSize = (this->_size / 32) * 32;
			this->finalPosCols256 = (this->_cols / 256) * 256;
			this->finalPosSize256 = (this->_size / 256) * 256;

			if (this->dataToDelete == nullptr)
			{
				uint8_t* newData = new uint8_t[this->_rows * this->_cols];
				uint8_t* oldData = this->_data;

				for (size_t i = 0; i < index; i++)
				{
					for (size_t j = 0; j < this->_cols; j++)
					{
						newData[i * this->_cols + j] = oldData[j * this->actualRows + i];
					}
				}
				for (size_t i{ index }, i2{ index + 1 }; i < this->_rows; i++, i2++)
				{
					for (size_t j = 0; j < this->_cols; j++)
					{
						newData[i * this->actualCols + j] = oldData[j * this->actualRows + i2];
					}
				}
				this->_data = newData;
				this->dataToDelete = newData;
			}
			else
			{
				for (size_t i{ index }, i2{ index + 1 }; i < this->_rows; i++, i2++)
				{
					for (size_t j = 0; j < this->_cols; j++)
					{
						this->_data[i * this->actualCols + j] = this->_data[j * this->actualRows + i2];
					}
				}

			}
		}
		else
		{
			this->_rows--;
			this->actualRows--;
			this->_size = this->_rows * this->_cols;
			this->finalPosRows = (this->_rows / 32) * 32;
			this->finalPosSize = (this->_size / 32) * 32;
			this->finalPosRows256 = (this->_rows / 256) * 256;
			this->finalPosSize256 = (this->_size / 256) * 256;

			if (this->dataToDelete == nullptr)
			{
				uint8_t* newData = new uint8_t[this->_rows * this->_cols];
				uint8_t* oldData = this->_data;

				for (size_t i = 0; i < index; i++)
				{
					for (size_t j = 0; j < this->_cols; j++)
					{
						newData[i * this->_cols + j] = oldData[i * this->actualCols + j];
					}
				}
				for (size_t i{ index }, i2{ index + 1 }; i < this->_rows; i++, i2++)
				{
					for (size_t j = 0; j < this->_cols; j++)
					{
						newData[i * this->actualCols + j] = oldData[i2 * this->actualCols + j];
					}
				}
				this->_data = newData;
				this->dataToDelete = newData;
			}
			else
			{
				for (size_t i{ index }, i2{ index + 1 }; i < this->_rows; i++, i2++)
				{
					for (size_t j = 0; j < this->_cols; j++)
					{
						this->_data[i * this->actualCols + j] = this->_data[i2 * this->actualCols + j];
					}
				}

			}
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline size_t matrix<uint8_t, thisTransposed, thisContiguous>::find(vector<uint8_t>& other)
	{
		if constexpr (thisTransposed)
		{
#ifdef _DEBUG
			if (this->_rows != other._size) throw std::invalid_argument("Error");
#else
#endif
			for (size_t j = 0; j < this->_cols; j++)
			{
				bool equal = true;
				for (size_t i = 0; i < this->_rows; i++)
				{
					if (this->_data[j * this->actualRows + i] != other._data[i]) equal = false;
				}
				if (equal) return j;
			}
		}
		else
		{
#ifdef _DEBUG
			if (this->_cols != other._size) throw std::invalid_argument("Error");
#else
#endif
			for (size_t i = 0; i < this->_rows; i++)
			{
				bool equal = true;
				for (size_t j = 0; j < this->_cols; j++)
				{
					if (this->_data[i * this->actualCols + j] != other._data[j]) equal = false;
				}
				if (equal) return i;
			}
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool otherTransposed, bool otherContiguous>
	inline vector<uint64_t> matrix<uint8_t, thisTransposed, thisContiguous>::find(matrix<uint8_t, otherTransposed, otherContiguous>& other)
	{
		if constexpr (thisTransposed)
		{
#ifdef _DEBUG
			if (this->_rows != other._rows) throw std::invalid_argument("Wrong dimension");
#else
#endif
			vector<uint64_t> result(other._cols);

			uint64_t* dataResult = result._data;

			for (size_t col = 0; col < other._cols; col++)
			{
				size_t index = this->_cols;
				for (size_t j = 0; j < this->_cols; j++)
				{
					bool equal = true;
					for (size_t i = 0; i < this->_rows; i++)
					{
						if constexpr (otherTransposed)
						{
							if (this->_data[j * this->actualRows + i] != other._data[col * other.actualRows + i]) equal = false;
						}
						else
						{
							if (this->_data[j * this->actualRows + i] != other._data[i * other.actualCols + col]) equal = false;
						}
					}
					if (equal)
					{
						index = j;
						break;
					}
				}
				dataResult[col] = index;
			}
			return result;
		}
		else
		{
#ifdef _DEBUG
			if (this->_cols != other._cols) throw std::invalid_argument("Wrong dimension");
#else
#endif
			vector<uint64_t> result(other._rows);

			uint64_t* dataResult = result._data;

			for (size_t row = 0; row < other._rows; row++)
			{
				size_t index = this->_rows;
				for (size_t i = 0; i < this->_rows; i++)
				{
					bool equal = true;
					for (size_t j = 0; j < this->_cols; j++)
					{
						if constexpr (otherTransposed)
						{
							if (this->_data[i * this->actualCols + j] != other._data[j * other.actualRows + row]) equal = false;
						}
						else
						{
							if (this->_data[i * this->actualCols + j] != other._data[row * other.actualCols + j]) equal = false;
						}
					}
					if (equal)
					{
						index = i;
						break;
					}
				}
				dataResult[row] = index;
			}
			return result;
		}
	}

	// Copy

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed>
	inline matrix<uint8_t> matrix<uint8_t, thisTransposed, thisContiguous>::copy()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		uint8_t* data1 = this->_data;

		matrix<uint8_t> result(rows, cols);

		uint8_t* dataResult = result._data;

		if constexpr (thisTransposed)
		{
			size_t actualRows = this->actualRows;
			if constexpr (returnTransposed)
			{
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						dataResult[j * rows + i] = data1[j * actualRows + i];
					}
				}
			}
			else
			{
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						dataResult[i * cols + j] = data1[j * actualRows + i];
					}
				}
			}
		}
		else
		{
			size_t actualCols = this->actualCols;
			if constexpr (returnTransposed)
			{
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						dataResult[j * rows + i] = data1[i * actualCols + j];
					}
				}
			}
			else
			{
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						dataResult[i * cols + j] = data1[i * actualCols + j];
					}
				}
			}
		}
		return result;
	}

	// = 

	template <bool thisTransposed, bool thisContiguous>
	template<bool otherTransposed, bool otherContiguous>
	inline matrix<uint8_t, thisTransposed, thisContiguous>& matrix<uint8_t, thisTransposed, thisContiguous>::operator=(matrix<uint8_t, otherTransposed, otherContiguous>& other)
	{
		if (this->_data == nullptr)
		{
#ifdef _DEBUG
			if (other.dataToDelete == nullptr) throw std::invalid_argument("Error");
#else
#endif
			this->_data = other._data;
			this->dataToDelete = this->_data;
			other.dataToDelete = nullptr;
			this->_size = other._size;
			this->_rows = other._rows;
			this->_cols = other._cols;
			this->finalPosCols = other.finalPosCols;
			this->finalPosRows = other.finalPosRows;
			this->finalPosSize = other.finalPosSize;
			this->finalPosCols256 = other.finalPosCols256;
			this->finalPosRows256 = other.finalPosRows256;
			this->finalPosActualSize256 = other.finalPosActualSize256;
		}
		else
		{
#ifdef _DEBUG
			if (other._cols != this->_cols || other._rows != this->_rows) throw std::invalid_argument("The dimensions of both matrices must be the same");
#else
#endif
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			uint8_t* data1 = this->_data;
			uint8_t* data2 = other._data;

			if constexpr (thisTransposed)
			{
				size_t matrix1ActualRows = this->actualRows;
				if constexpr (otherTransposed)
				{
					size_t matrix2ActualRows = other.actualRows;

					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							data1[j * matrix1ActualRows + i] = data2[j * matrix2ActualRows + i];
						}
					}
				}
				else
				{
					size_t matrix2ActualCols = other.actualCols;

					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							data1[j * matrix1ActualRows + i] = data2[i * matrix2ActualCols + j];
						}
					}
				}
			}
			else
			{
				size_t matrix1ActualCols = this->actualCols;
				if constexpr (otherTransposed)
				{
					size_t matrix2ActualRows = other.actualRows;

					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							data1[i * matrix1ActualCols + j] = data2[j * matrix2ActualRows + i];
						}
					}
				}
				else
				{
					size_t matrix2ActualCols = other.actualCols;

					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							data1[i * matrix1ActualCols + j] = data2[i * matrix2ActualCols + j];
						}
					}
				}
			}
		}
		return *this;
	}

	// Transfer

	template <bool thisTransposed, bool thisContiguous>
	template<bool otherContiguous>
	inline void matrix<uint8_t, thisTransposed, thisContiguous>::transfer(matrix<uint8_t, thisTransposed, otherContiguous>& other)
	{
#ifdef _DEBUG
		if (other.dataToDelete == nullptr || (this->dataToDelete == nullptr && this->_data != nullptr)) throw std::invalid_argument("Error");
#else
#endif
		delete[] this->_data;

		this->_data = other._data;
		this->dataToDelete = other._data;
		other.dataToDelete = nullptr;
		this->_cols = other._cols;
		this->_rows = other._rows;
		this->_size = other._size;
		this->actualCols = other.actualCols;
		this->actualRows = other.actualRows;
		this->finalPosCols = other.finalPosCols;
		this->finalPosRows = other.finalPosRows;
		this->finalPosSize = other.finalPosSize;
		this->finalPosCols256 = other.finalPosCols256;
		this->finalPosRows256 = other.finalPosRows256;
		this->finalPosSize256 = other.finalPosSize256;
	}

	// Set constant

	template <bool thisTransposed, bool thisContiguous>
	inline void matrix<uint8_t, thisTransposed, thisContiguous>::set_const(uint8_t num)
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		uint8_t* data1 = this->_data;

		if constexpr (thisTransposed)
		{
			size_t matrix1ActualRows = this->actualRows;
			for (size_t i = 0; i < rows; i++)
			{
				for (size_t j = 0; j < cols; j++)
				{
					data1[j * matrix1ActualRows + i] = num;
				}
			}
		}
		else
		{
			size_t matrix1ActualCols = this->actualCols;
			for (size_t i = 0; i < rows; i++)
			{
				for (size_t j = 0; j < cols; j++)
				{
					data1[i * matrix1ActualCols + j] = num;
				}
			}
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed, bool otherTransposed, bool otherContiguous>
	inline matrix<uint8_t> matrix<uint8_t, thisTransposed, thisContiguous>::operator&&(matrix<uint8_t, otherTransposed, otherContiguous>& other)
	{
#ifdef _DEBUG
		if (other._cols != this->_cols || other._rows != this->_rows) throw std::invalid_argument("The dimensions of both matrices must be the same");
#else
#endif

		size_t rows = this->_rows;
		size_t cols = this->_cols;

		uint8_t* data1 = this->_data;
		uint8_t* data2 = other._data;

		matrix<uint8_t> result(rows, cols);

		uint8_t* dataResult = result._data;

		if constexpr (thisTransposed)
		{
			if constexpr (otherTransposed)
			{
				if constexpr (returnTransposed)
				{
					if constexpr (thisContiguous && otherContiguous)
					{
						size_t size = this->_size;

						size_t finalPosSize = this->finalPosSize;

						for (size_t i = 0; i < finalPosSize; i += 32)
						{
							__m256i a = _mm256_loadu_epi8(&data1[i]);
							__m256i b = _mm256_loadu_epi8(&data2[i]);

							_mm256_storeu_epi8(&dataResult[i], _mm256_and_si256(a, b));
						}
						for (size_t i = finalPosSize; i < size; i++)
						{
							dataResult[i] = data1[i] & data2[i];
						}
					}
					else
					{
						size_t matrix1ActualRows = this->actualRows;
						size_t matrix2ActualRows = other.actualRows;

						size_t finalPosRows = this->finalPosRows;

						for (size_t i = 0; i < finalPosRows; i += 32)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256i a = _mm256_loadu_epi8(&data1[j * matrix1ActualRows + i]);
								__m256i b = _mm256_loadu_epi8(&data2[j * matrix2ActualRows + i]);

								_mm256_storeu_epi8(&dataResult[j * rows + i], _mm256_and_si256(a, b));
							}
						}
						for (size_t i = finalPosRows; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								dataResult[j * rows + i] = data1[j * matrix1ActualRows + i] & data2[j * matrix2ActualRows + i];
							}
						}
					}
				}
				else
				{
					size_t matrix1ActualRows = this->actualRows;
					size_t matrix2ActualRows = other.actualRows;

					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[i * cols + j] = data1[j * matrix1ActualRows + i] & data2[j * matrix2ActualRows + i];
						}
					}
				}
			}
			else
			{
				if constexpr (returnTransposed)
				{
					size_t matrix1ActualRows = this->actualRows;
					size_t matrix2ActualCols = other.actualCols;

					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = data1[j * matrix1ActualRows + i] & data2[i * matrix2ActualCols + j];
						}
					}
				}
				else
				{
					size_t matrix1ActualRows = this->actualRows;
					size_t matrix2ActualCols = other.actualCols;

					for (size_t j = 0; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[j * matrix1ActualRows + i] & data2[i * matrix2ActualCols + j];
						}
					}
				}
			}
		}
		else
		{
			if constexpr (otherTransposed)
			{
				if constexpr (returnTransposed)
				{
					size_t matrix1ActualCols = this->actualCols;
					size_t matrix2ActualRows = other.actualRows;

					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = data1[i * matrix1ActualCols + j] & data2[j * matrix2ActualRows + i];
						}
					}
				}
				else
				{
					size_t matrix1ActualCols = this->actualCols;
					size_t matrix2ActualRows = other.actualRows;

					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] & data2[j * matrix2ActualRows + i];
						}
					}
				}
			}
			else
			{
				if constexpr (returnTransposed)
				{
					size_t matrix1ActualCols = this->actualCols;
					size_t matrix2ActualCols = other.actualCols;

					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = data1[i * matrix1ActualCols + j] & data2[i * matrix2ActualCols + j];
						}
					}
				}
				else
				{
					if constexpr (thisContiguous && otherContiguous)
					{
						size_t size = this->_size;

						size_t finalPosSize = this->finalPosSize;

						for (size_t i = 0; i < finalPosSize; i += 32)
						{
							__m256i a = _mm256_loadu_epi8(&data1[i]);
							__m256i b = _mm256_loadu_epi8(&data2[i]);

							_mm256_storeu_epi8(&dataResult[i], _mm256_and_si256(a, b));
						}
						for (size_t i = finalPosSize; i < size; i++)
						{
							dataResult[i] = data1[i] & data2[i];

						}
					}
					else
					{
						size_t matrix1ActualCols = this->actualCols;
						size_t matrix2ActualCols = other.actualCols;

						size_t finalPosCols = this->finalPosCols;

						for (size_t j = 0; j < finalPosCols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256i a = _mm256_loadu_epi8(&data1[i * matrix1ActualCols + j]);
								__m256i b = _mm256_loadu_epi8(&data2[i * matrix2ActualCols + j]);

								_mm256_storeu_epi8(&dataResult[i * cols + j], _mm256_and_si256(a, b));
							}
						}
						for (size_t j = finalPosCols; j < cols; j++)
						{
							for (size_t i = 0; i < rows; i++)
							{
								dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] & data2[i * matrix2ActualCols + j];
							}
						}
					}
				}
			}
		}
		return result;
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed, bool otherTransposed, bool otherContiguous>
	inline matrix<uint8_t> matrix<uint8_t, thisTransposed, thisContiguous>::operator||(matrix<uint8_t, otherTransposed, otherContiguous>& other)
	{
#ifdef _DEBUG
		if (other._cols != this->_cols || other._rows != this->_rows) throw std::invalid_argument("The dimensions of both matrices must be the same");
#else
#endif

		size_t rows = this->_rows;
		size_t cols = this->_cols;

		uint8_t* data1 = this->_data;
		uint8_t* data2 = other._data;

		matrix<uint8_t> result(rows, cols);

		uint8_t* dataResult = result._data;

		if constexpr (thisTransposed)
		{
			if constexpr (otherTransposed)
			{
				if constexpr (returnTransposed)
				{
					if constexpr (thisContiguous || otherContiguous)
					{
						size_t size = this->_size;

						size_t finalPosSize = this->finalPosSize;

						for (size_t i = 0; i < finalPosSize; i += 32)
						{
							__m256i a = _mm256_loadu_epi8(&data1[i]);
							__m256i b = _mm256_loadu_epi8(&data2[i]);

							_mm256_storeu_epi8(&dataResult[i], _mm256_or_si256(a, b));
						}
						for (size_t i = finalPosSize; i < size; i++)
						{
							dataResult[i] = data1[i] | data2[i];
						}
					}
					else
					{
						size_t matrix1ActualRows = this->actualRows;
						size_t matrix2ActualRows = other.actualRows;

						size_t finalPosRows = this->finalPosRows;

						for (size_t i = 0; i < finalPosRows; i += 32)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256i a = _mm256_loadu_epi8(&data1[j * matrix1ActualRows + i]);
								__m256i b = _mm256_loadu_epi8(&data2[j * matrix2ActualRows + i]);

								_mm256_storeu_epi8(&dataResult[j * rows + i], _mm256_or_si256(a, b));
							}
						}
						for (size_t i = finalPosRows; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								dataResult[j * rows + i] = data1[j * matrix1ActualRows + i] | data2[j * matrix2ActualRows + i];
							}
						}
					}
				}
				else
				{
					size_t matrix1ActualRows = this->actualRows;
					size_t matrix2ActualRows = other.actualRows;

					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[i * cols + j] = data1[j * matrix1ActualRows + i] | data2[j * matrix2ActualRows + i];
						}
					}
				}
			}
			else
			{
				if constexpr (returnTransposed)
				{
					size_t matrix1ActualRows = this->actualRows;
					size_t matrix2ActualCols = other.actualCols;

					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = data1[j * matrix1ActualRows + i] | data2[i * matrix2ActualCols + j];
						}
					}
				}
				else
				{
					size_t matrix1ActualRows = this->actualRows;
					size_t matrix2ActualCols = other.actualCols;

					for (size_t j = 0; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[j * matrix1ActualRows + i] | data2[i * matrix2ActualCols + j];
						}
					}
				}
			}
		}
		else
		{
			if constexpr (otherTransposed)
			{
				if constexpr (returnTransposed)
				{
					size_t matrix1ActualCols = this->actualCols;
					size_t matrix2ActualRows = other.actualRows;

					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = data1[i * matrix1ActualCols + j] | data2[j * matrix2ActualRows + i];
						}
					}
				}
				else
				{
					size_t matrix1ActualCols = this->actualCols;
					size_t matrix2ActualRows = other.actualRows;

					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] | data2[j * matrix2ActualRows + i];
						}
					}
				}
			}
			else
			{
				if constexpr (returnTransposed)
				{
					size_t matrix1ActualCols = this->actualCols;
					size_t matrix2ActualCols = other.actualCols;

					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = data1[i * matrix1ActualCols + j] | data2[i * matrix2ActualCols + j];
						}
					}
				}
				else
				{
					if constexpr (thisContiguous || otherContiguous)
					{
						size_t size = this->_size;

						size_t finalPosSize = this->finalPosSize;

						for (size_t i = 0; i < finalPosSize; i += 32)
						{
							__m256i a = _mm256_loadu_epi8(&data1[i]);
							__m256i b = _mm256_loadu_epi8(&data2[i]);

							_mm256_storeu_epi8(&dataResult[i], _mm256_or_si256(a, b));
						}
						for (size_t i = finalPosSize; i < size; i++)
						{
							dataResult[i] = data1[i] | data2[i];
						}
					}
					else
					{
						size_t matrix1ActualCols = this->actualCols;
						size_t matrix2ActualCols = other.actualCols;

						size_t finalPosCols = this->finalPosCols;

						for (size_t j = 0; j < finalPosCols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256i a = _mm256_loadu_epi8(&data1[i * matrix1ActualCols + j]);
								__m256i b = _mm256_loadu_epi8(&data2[i * matrix2ActualCols + j]);

								_mm256_storeu_epi8(&dataResult[i * cols + j], _mm256_or_si256(a, b));
							}
						}
						for (size_t j = finalPosCols; j < cols; j++)
						{
							for (size_t i = 0; i < rows; i++)
							{
								dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] | data2[i * matrix2ActualCols + j];
							}
						}
					}
				}
			}
		}
		return result;
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed>
	inline matrix<uint8_t> matrix<uint8_t, thisTransposed, thisContiguous>::operator!()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		uint8_t* data1 = this->_data;

		matrix<uint8_t> result(rows, cols);

		uint8_t* dataResult = result._data;

		__m256i b = _mm256_set1_epi64x(-1);

		if constexpr (thisTransposed)
		{
			if constexpr (returnTransposed)
			{
				if constexpr (thisContiguous)
				{
					size_t size = this->_size;
					size_t finalPosSize = this->finalPosSize;

					for (size_t i = 0; i < finalPosSize; i += 32)
					{
						__m256i a = _mm256_loadu_epi8(&data1[i]);
						_mm256_storeu_epi8(&dataResult[i], _mm256_andnot_si256(a, b));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = ~data1[i];
					}
				}
				else
				{
					size_t matrix1ActualRows = this->actualRows;

					size_t finalPosRows = this->finalPosRows;

					for (size_t i = 0; i < finalPosRows; i += 32)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256i a = _mm256_loadu_epi8(&data1[j * matrix1ActualRows + i]);
							_mm256_storeu_epi8(&dataResult[j * rows + i], _mm256_andnot_si256(a, b));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = ~data1[j * matrix1ActualRows + i];
						}
					}
				}
			}
			else
			{
				size_t matrix1ActualRows = this->actualRows;

				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						dataResult[i * cols + j] = ~data1[j * matrix1ActualRows + i];
					}
				}
			}
		}
		else
		{
			if constexpr (returnTransposed)
			{
				size_t matrix1ActualCols = this->actualCols;

				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						dataResult[j * rows + i] = ~data1[i * matrix1ActualCols + j];
					}
				}
			}
			else
			{
				if constexpr (thisContiguous)
				{
					size_t size = this->_size;
					size_t finalPosSize = this->finalPosSize;

					for (size_t i = 0; i < finalPosSize; i += 32)
					{
						__m256i a = _mm256_loadu_epi8(&data1[i]);
						_mm256_storeu_epi8(&dataResult[i], _mm256_andnot_si256(a, b));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = ~data1[i];
					}
				}
				else
				{
					size_t matrix1ActualCols = this->actualRows;

					size_t finalPosCols = this->finalPosCols;

					for (size_t j = 0; j < finalPosCols; j += 32)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_loadu_epi8(&data1[i * matrix1ActualCols + j]);
							_mm256_storeu_epi8(&dataResult[i * cols + j], _mm256_andnot_si256(a, b));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = ~data1[i * matrix1ActualCols + j];
						}
					}
				}
			}
		}
		return result;
	}

	template <bool thisTransposed, bool thisContiguous>
	inline void matrix<uint8_t, thisTransposed, thisContiguous>::self_not()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		uint8_t* data1 = this->_data;

		__m256d b = _mm256_set1_epi64x(-1);

		if constexpr (thisContiguous)
		{
			size_t size = this->_size;
			size_t finalPosSize = this->finalPosSize;

			for (size_t i = 0; i < finalPosSize; i += 32)
			{
				__m256d a = _mm256_loadu_epi8(&data1[i]);
				_mm256_storeu_epi8(&data1[i], _mm256_andnot_si256(a, b));
			}
			for (size_t i = finalPosSize; i < size; i++)
			{
				data1[i] = ~data1[i];
			}
		}
		else if constexpr (thisTransposed)
		{
			size_t matrix1ActualRows = this->actualRows;

			size_t finalPosRows = this->finalPosRows;

			for (size_t i = 0; i < finalPosRows; i += 32)
			{
				for (size_t j = 0; j < cols; j++)
				{
					__m256d a = _mm256_loadu_epi8(&data1[j * matrix1ActualRows + i]);
					_mm256_storeu_epi8(&data1[j * matrix1ActualRows + i], _mm256_andnot_si256(a, b));
				}
			}
			for (size_t i = finalPosRows; i < rows; i++)
			{
				for (size_t j = 0; j < cols; j++)
				{
					data1[j * matrix1ActualRows + i] = ~data1[j * matrix1ActualRows + i];
				}
			}
		}
		else
		{
			size_t matrix1ActualCols = this->actualRows;

			size_t finalPosCols = this->finalPosCols;

			for (size_t j = 0; j < finalPosCols; j += 32)
			{
				for (size_t i = 0; i < rows; i++)
				{
					__m256d a = _mm256_loadu_epi8(&data1[i * matrix1ActualCols + j]);
					_mm256_storeu_epi8(&data1[i * matrix1ActualCols + j], _mm256_andnot_si256(a, b));
				}
			}
			for (size_t j = finalPosCols; j < cols; j++)
			{
				for (size_t i = 0; i < rows; i++)
				{
					data1[i * matrix1ActualCols + j] = ~data1[i * matrix1ActualCols + j];
				}
			}
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline size_t matrix<uint8_t, thisTransposed, thisContiguous>::count_all()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		uint8_t* data1 = this->_data;

		size_t count = 0;

		if constexpr (thisContiguous)
		{
			size_t size = this->_size;

			size_t finalPosSize256 = this->finalPosSize256;

			for (size_t i = 0; i < finalPosSize256; i += 256)
			{
				count += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[i])));
				count += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[i + 32])));
				count += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[i + 64])));
				count += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[i + 96])));
				count += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[i + 128])));
				count += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[i + 160])));
				count += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[i + 192])));
				count += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[i + 224])));
			}
			for (size_t i = finalPosSize256; i < size; i++)
			{
				if (data1[i]) count++;
			}
		}
		else if constexpr (thisTransposed)
		{
			size_t matrix1ActualRows = this->actualRows;

			size_t finalPosRows256 = this->finalPosRows256;

			for (size_t j = 0; j < cols; j++)
			{
				for (size_t i = 0; i < finalPosRows256; i += 256)
				{
					count += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[j * matrix1ActualRows + i])));
					count += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[j * matrix1ActualRows + i + 32])));
					count += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[j * matrix1ActualRows + i + 64])));
					count += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[j * matrix1ActualRows + i + 96])));
					count += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[j * matrix1ActualRows + i + 128])));
					count += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[j * matrix1ActualRows + i + 160])));
					count += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[j * matrix1ActualRows + i + 192])));
					count += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[j * matrix1ActualRows + i + 224])));
				}
				for (size_t i = finalPosRows256; i < rows; i++)
				{
					if (data1[j * matrix1ActualRows + i]) count++;
				}
			}
		}
		else
		{
			size_t matrix1ActualCols = this->actualCols;

			size_t extra_cols = matrix1ActualCols - cols;

			size_t finalPosCols256 = this->finalPosCols256;

			for (size_t i = 0; i < rows; i++)
			{
				for (size_t j = 0; j < finalPosCols256; j += 256)
				{
					count += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[i * matrix1ActualCols + j])));
					count += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[i * matrix1ActualCols + j + 32])));
					count += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[i * matrix1ActualCols + j + 64])));
					count += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[i * matrix1ActualCols + j + 96])));
					count += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[i * matrix1ActualCols + j + 128])));
					count += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[i * matrix1ActualCols + j + 160])));
					count += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[i * matrix1ActualCols + j + 192])));
					count += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[i * matrix1ActualCols + j + 224])));
				}
				for (size_t j = finalPosCols256; j < cols; j++)
				{
					if (data1[i * matrix1ActualCols + j]) count++;
				}
			}
		}
		return count;
	}

	template <bool thisTransposed, bool thisContiguous>
	inline vector<uint64_t> matrix<uint8_t, thisTransposed, thisContiguous>::count_colwise()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		uint8_t* data1 = this->_data;

		vector<uint64_t> result(cols);

		uint64_t* dataResult = result._data;

		if constexpr (thisTransposed)
		{
			size_t matrix1ActualRows = this->actualRows;

			size_t extra_rows = matrix1ActualRows - rows;

			size_t finalPosRows256 = this->finalPosRows256;

			for (size_t j = 0; j < cols; j++)
			{
				size_t count = 0;
				for (size_t i = 0; i < finalPosRows256; i += 256)
				{
					count += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[j * matrix1ActualRows + i])));
					count += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[j * matrix1ActualRows + i + 32])));
					count += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[j * matrix1ActualRows + i + 64])));
					count += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[j * matrix1ActualRows + i + 96])));
					count += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[j * matrix1ActualRows + i + 128])));
					count += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[j * matrix1ActualRows + i + 160])));
					count += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[j * matrix1ActualRows + i + 192])));
					count += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[j * matrix1ActualRows + i + 224])));
				}
				for (size_t i = finalPosRows256; i < rows; i++)
				{
					if (data1[j * matrix1ActualRows + i]) count++;
				}
				dataResult[j] = count;
			}
		}
		else
		{
			size_t matrix1ActualCols = this->actualCols;

			for (size_t j = 0; j < cols; j++)
			{
				size_t count = 0;
				for (size_t i = 0; i < rows; i++)
				{
					if (data1[i * matrix1ActualCols + j]) count++;
				}
				dataResult[j] = count;
			}
		}
		return result;
	}

	template <bool thisTransposed, bool thisContiguous>
	inline vector<uint64_t> matrix<uint8_t, thisTransposed, thisContiguous>::count_rowwise()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		uint8_t* data1 = this->_data;

		vector<uint64_t> result(rows);

		uint64_t* dataResult = result._data;

		if constexpr (thisTransposed)
		{
			size_t matrix1ActualRows = this->actualRows;

			for (size_t i = 0; i < rows; i++)
			{
				size_t count = 0;
				for (size_t j = 0; j < cols; j++)
				{
					if (data1[j * matrix1ActualRows + i]) count++;
				}
				dataResult[i] = count;
			}
		}
		else
		{

			size_t matrix1ActualCols = this->actualCols;

			size_t extra_cols = matrix1ActualCols - cols;

			size_t finalPosCols256 = this->finalPosCols256;

			for (size_t i = 0; i < rows; i++)
			{
				size_t count = 0;
				for (size_t j = 0; j < finalPosCols256; j += 256)
				{
					count += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[i * matrix1ActualCols + j])));
					count += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[i * matrix1ActualCols + j + 32])));
					count += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[i * matrix1ActualCols + j + 64])));
					count += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[i * matrix1ActualCols + j + 96])));
					count += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[i * matrix1ActualCols + j + 128])));
					count += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[i * matrix1ActualCols + j + 160])));
					count += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[i * matrix1ActualCols + j + 192])));
					count += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_loadu_epi8(&data1[i * matrix1ActualCols + j + 224])));
				}
				for (size_t j = finalPosCols256; j < cols; j++)
				{
					if (data1[i * matrix1ActualCols + j]) count++;
				}
				dataResult[i] = count;
			}
		}
		return result;
	}

	template <bool thisTransposed, bool thisContiguous>
	template<typename T>
	inline matrix<T> matrix<uint8_t, thisTransposed, thisContiguous>::cast()
	{
		size_t cols = this->_cols;
		size_t rows = this->_rows;

		uint8_t* data1 = this->_data;

		matrix<T> result(rows, cols);

		T* dataResult = result._data;

		if constexpr (sizeof(T) == 4)
		{
			if constexpr (thisTransposed)
			{
				uint32_t one = 0b1;
				uint32_t zero = 0b0;

				size_t actualRows = this->actualRows;

				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						dataResult[i * cols + j] = data1[j * actualRows + i] ? reinterpret_cast<T&>(one) : reinterpret_cast<T&>(zero);
					}
				}
			}
			else
			{
				__m256 _one = _mm256_set1_ps(1.0f);
				__m256 _zero = _mm256_setzero_si256();
				uint32_t one = 0b1;
				uint32_t zero = 0b0;

				size_t actualCols = this->actualCols;

				size_t finalPosCols = (this->_cols / 8) * 8;

				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 8)
					{
						__m256 mask = _mm256_castsi256_ps(_mm256_cvtepi8_epi32(_mm_loadu_epi8(&data1[i * actualCols + j])));

						_mm256_store_ps(reinterpret_cast<float*>(&dataResult[i * cols + j]), _mm256_blendv_ps(_zero, _one, mask));
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						dataResult[i * cols + j] = data1[i * actualCols + j] ? reinterpret_cast<T&>(one) : reinterpret_cast<T&>(zero);
					}
				}
			}
		}
		else if constexpr (sizeof(T) == 8)
		{
			if constexpr (thisTransposed)
			{
				uint64_t one = 0b1;
				uint64_t zero = 0b0;
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						dataResult[i * cols + j] = data1[j * actualRows + i] ? reinterpret_cast<T&>(one) : reinterpret_cast<T&>(zero);
					}
				}
			}
			else
			{
				__m256d _one = _mm256_set1_pd(1.0);
				__m256d _zero = _mm256_setzero_pd();
				uint64_t one = 0b1;
				uint64_t zero = 0b0;

				size_t finalPosCols = (this->_cols / 4) * 4;

				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d mask = _mm256_castsi256_pd(_mm256_cvtepi8_epi64(_mm_loadu_epi8(&data1[i * actualCols + j])));

						_mm256_store_pd(reinterpret_cast<double*>(&dataResult[i * cols + j]), _mm256_blendv_pd(_zero, _one, mask));
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						dataResult[i * cols + j] = data1[i * actualCols + j] ? reinterpret_cast<T&>(one) : reinterpret_cast<T&>(zero);
					}
				}
			}
		}
		return result;
	}

}