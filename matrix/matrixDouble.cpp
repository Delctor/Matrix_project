#include "matrixDouble.h"

namespace alge
{
	template <bool thisTransposed, bool thisContiguous>
	inline matrix<double, thisTransposed, thisContiguous>::matrix() :
		_data(nullptr),
		dataToDelete(nullptr),
		_rows(0),
		_cols(0),
		_size(0),
		actualRows(0),
		actualCols(0),
		finalPosRows(0),
		finalPosCols(0),
		finalPosSize(0),
		_capacityRows(0),
		transposed(thisTransposed) {}

	template <bool thisTransposed, bool thisContiguous>
	inline matrix<double, thisTransposed, thisContiguous>::matrix(size_t rows, size_t cols) :
		_data(new double[rows * cols]),
		dataToDelete(_data),
		_rows(rows),
		_cols(cols),
		_size(rows* cols),
		actualRows(rows),
		actualCols(cols),
		finalPosRows((_rows / 4) * 4),
		finalPosCols((_cols / 4) * 4),
		finalPosSize(((rows* cols) / 4) * 4),
		_capacityRows(thisTransposed ? cols : rows),
		transposed(thisTransposed) {}

	template <bool thisTransposed, bool thisContiguous>
	inline matrix<double, thisTransposed, thisContiguous>::matrix(double* data, size_t rows, size_t cols, size_t actualRows, size_t actualCols) :
		_data(data),
		dataToDelete(nullptr),
		_rows(rows),
		_cols(cols),
		_size(rows* cols),
		actualRows(actualRows),
		actualCols(actualCols),
		finalPosRows((_rows / 4) * 4),
		finalPosCols((_cols / 4) * 4),
		finalPosSize((_size / 4) * 4),
		_capacityRows(thisTransposed ? cols : rows),
		transposed(thisTransposed) {}

	template <bool thisTransposed, bool thisContiguous>
	inline matrix<double, thisTransposed, thisContiguous>::matrix(std::initializer_list<std::initializer_list<double>> list)
	{
		this->_rows = list.size();
		this->_cols = (*list.begin()).size();
		this->actualRows = this->_rows;
		this->actualCols = this->_cols;
		this->_size = this->_rows * this->_cols;
		this->_data = new double[this->_size];
		this->dataToDelete = this->_data;
		this->finalPosRows = (this->_rows / 4) * 4;
		this->finalPosCols = (this->_cols / 4) * 4;
		this->finalPosSize = (this->_size / 4) * 4;
		this->_capacityRows = thisTransposed;

		this->transposed = thisTransposed ? true : false;
		if constexpr (thisTransposed)
		{
			for (size_t i = 0; i < this->_rows; i++)
			{
				std::initializer_list<double> listI = *(list.begin() + i);
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
				std::initializer_list<double> listI = *(list.begin() + i);
				for (size_t j = 0; j < this->_cols; j++)
				{
					this->_data[i * this->actualCols + j] = *(listI.begin() + j);
				}
			}
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline matrix<double, thisTransposed, thisContiguous>::~matrix() { delete[] this->dataToDelete; }

	template <bool thisTransposed, bool thisContiguous>
	inline size_t matrix<double, thisTransposed, thisContiguous>::rows() { return this->_rows; }

	template <bool thisTransposed, bool thisContiguous>
	inline size_t matrix<double, thisTransposed, thisContiguous>::cols() { return this->_cols; }

	template <bool thisTransposed, bool thisContiguous>
	inline double* matrix<double, thisTransposed, thisContiguous>::data() { return this->_data; }

	template <bool thisTransposed, bool thisContiguous>
	inline matrix<double, thisTransposed, thisContiguous && !thisTransposed> matrix<double, thisTransposed, thisContiguous>::row(size_t row)
	{
		if constexpr (thisTransposed)
		{
			return matrix<double, true, false>(
				&this->_data[row],
				1,
				this->_cols,
				this->actualRows,
				this->actualCols);
		}
		else
		{
			return matrix<double, false, thisContiguous>(
				&this->_data[row * this->actualCols],
				1,
				this->_cols,
				this->actualRows,
				this->actualCols);
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline matrix<double, thisTransposed, thisContiguous && thisTransposed> matrix<double, thisTransposed, thisContiguous>::col(size_t col)
	{
		if constexpr (thisTransposed)
		{
			return matrix<double, true, thisContiguous>(
				&this->_data[col * this->actualRows],
				this->_rows,
				1,
				this->actualRows,
				this->actualCols);
		}
		else
		{
			return matrix<double, false, false>(
				&this->_data[col],
				this->_rows,
				1,
				this->actualRows,
				this->actualCols);
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline matrix<double, !thisTransposed, thisContiguous> matrix<double, thisTransposed, thisContiguous>::tranpose()
	{
		return matrix<double, !thisTransposed, thisContiguous>(
			this->_data,
			this->_cols,
			this->_rows,
			this->actualCols,
			this->actualRows
		);
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool blockContiguous>
	inline matrix<double, thisTransposed, thisContiguous && blockContiguous> matrix<double, thisTransposed, thisContiguous>::block(size_t initial_row, size_t initial_col, size_t final_row, size_t final_col)
	{
		if constexpr (thisTransposed)
		{
			return matrix<double, true, thisContiguous&& blockContiguous>(
				&this->_data[initial_col * this->actualRows + initial_row],
				final_row - initial_row,
				final_col - initial_col,
				final_row - initial_row,
				final_col - initial_col
			);
		}
		else
		{
			return matrix<double, false, thisContiguous&& blockContiguous>(
				&this->_data[initial_row * this->actualCols + initial_col],
				final_row - initial_row,
				final_col - initial_col,
				final_row - initial_row,
				final_col - initial_col
			);
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline double& matrix<double, thisTransposed, thisContiguous>::operator()(size_t row, size_t col)
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
	inline const double& matrix<double, thisTransposed, thisContiguous>::operator()(size_t row, size_t col) const
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
	inline size_t matrix<double, thisTransposed, thisContiguous>::capacity() { return this->_capacityRows; }

	template <bool thisTransposed, bool thisContiguous>
	template<bool reduceCapacity>
	inline void matrix<double, thisTransposed, thisContiguous>::clear()
	{
		if constexpr (reduceCapacity)
		{
			this->_size = 0;
			this->finalPosSize = 0;
			this->finalPosRows = 0;
			this->finalPosCols = 0;
			this->_rows = 0;
			this->_cols = 0;
			this->actualCols = 0;
			this->actualRows = 0;

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
			this->_rows = 0;
			this->_cols = 0;
			this->actualCols = 0;
			this->actualRows = 0;
			if (this->dataToDelete == nullptr)
			{
				if constexpr (thisTransposed) this->_data = new double[this->_capacityRows * this->_rows]; else this->_data = new double[this->_capacityRows * this->_cols];
				this->dataToDelete = this->_data;
			}
		}

	}

	template <bool thisTransposed, bool thisContiguous>
	inline void matrix<double, thisTransposed, thisContiguous>::reserve(size_t newCapacity)
	{
		if constexpr (thisTransposed)
		{
			double* newData = new double[newCapacity * this->_rows];
			double* oldData = this->_data;

			this->_cols = this->_cols <= newCapacity ? this->_cols : newCapacity;

			this->_size = this->_rows * this->_cols;
			this->finalPosSize = (this->_size / 4) * 4;
			this->finalPoscols = (this->_cols / 4) * 4;

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
			double* newData = new double[newCapacity * this->_cols];
			double* oldData = this->_data;

			this->_rows = this->_rows <= newCapacity ? this->_rows : newCapacity;

			this->_size = this->_rows * this->_cols;
			this->finalPosSize = (this->_size / 4) * 4;
			this->finalPosRows = (this->_rows / 4) * 4;

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
	inline void matrix<double, thisTransposed, thisContiguous>::append(std::initializer_list<std::initializer_list<double>> list)
	{
		size_t sizeList = list.size();

		if constexpr (thisTransposed)
		{
			size_t newCols = this->_cols + sizeList;

			if (this->_capacityRows >= newCols)
			{
				for (size_t i{ this->_cols }, i2{ 0 }; i < newCols; i++, i2++)
				{
					std::initializer_list<double> listI = *(list.begin() + i2);
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

				double* newData = new double[this->_capacityRows * this->_rows];
				double* oldData = this->_data;

				for (size_t i = 0; i < this->_cols; i++)
				{
					for (size_t j = 0; j < this->_rows; j++)
					{
						newData[i * this->_rows + j] = oldData[i * this->actualRows + j];
					}
				}
				for (size_t i{ this->_cols }, i2{ 0 }; i < newCols; i++, i2++)
				{
					std::initializer_list<double> listI = *(list.begin() + i2);
					for (size_t j = 0; j < this->_rows; j++)
					{
						newData[i * this->_rows + j] = *(listI.begin() + j);
					}
				}

				delete[] this->dataToDelete;

				this->actualCols = this->_cols;
				this->actualRows = this->_rows;

				this->_data = newData;
				this->dataToDelete = newData;
			}

			this->_cols = newCols;
			this->_size = this->_rows * this->_cols;
			this->finalPosSize = (this->_size / 4) * 4;
			this->finalPosCols = (this->_cols / 4) * 4;
		}
		else
		{
			size_t newRows = this->_rows + sizeList;

			if (this->_capacityRows >= newRows)
			{
				for (size_t i{ this->_rows }, i2{ 0 }; i < newRows; i++, i2++)
				{
					std::initializer_list<double> listI = *(list.begin() + i2);
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

				double* newData = new double[this->_capacityRows * this->_cols];
				double* oldData = this->_data;

				for (size_t i = 0; i < this->_rows; i++)
				{
					for (size_t j = 0; j < this->_cols; j++)
					{
						newData[i * this->_cols + j] = oldData[i * this->actualCols + j];
					}
				}
				for (size_t i{ this->_rows }, i2{ 0 }; i < newRows; i++, i2++)
				{
					std::initializer_list<double> listI = *(list.begin() + i2);
					for (size_t j = 0; j < this->_cols; j++)
					{
						newData[i * this->_cols + j] = *(listI.begin() + j);
					}
				}

				delete[] this->dataToDelete;

				this->actualCols = this->_cols;
				this->actualRows = this->_rows;

				this->_data = newData;
				this->dataToDelete = newData;
			}
			this->_rows = newRows;
			this->_size = this->_rows * this->_cols;
			this->finalPosSize = (this->_size / 4) * 4;
			this->finalPosRows = (this->_rows / 4) * 4;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool otherTransposed, bool otherContiguous>
	inline void matrix<double, thisTransposed, thisContiguous>::append(matrix<double, otherTransposed, otherContiguous>& other)
	{
		size_t sizeOther;
		if constexpr (otherTransposed)
		{
			sizeOther = other._cols;
		}
		else
		{
			sizeOther = other._rows;
		}

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

					double* newData = new double[this->_capacityRows * this->_rows];
					double* oldData = this->_data;

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

					this->actualCols = this->_cols;
					this->actualRows = this->_rows;

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

					double* newData = new double[this->_capacityRows * this->_rows];
					double* oldData = this->_data;

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

					this->actualCols = this->_cols;
					this->actualRows = this->_rows;

					this->_data = newData;
					this->dataToDelete = newData;
				}
			}

			this->_cols = newCols;
			this->_size = this->_rows * this->_cols;
			this->finalPosSize = (this->_size / 4) * 4;
			this->finalPosCols = (this->_cols / 4) * 4;
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

					double* newData = new double[this->_capacityRows * this->_cols];
					double* oldData = this->_data;

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

					this->actualCols = this->_cols;
					this->actualRows = this->_rows;

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

					double* newData = new double[this->_capacityRows * this->_cols];
					double* oldData = this->_data;

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

					this->actualCols = this->_cols;
					this->actualRows = this->_rows;

					this->_data = newData;
					this->dataToDelete = newData;
				}
			}

			this->_rows = newRows;
			this->_size = this->_rows * this->_cols;
			this->finalPosSize = (this->_size / 4) * 4;
			this->finalPosRows = (this->_rows / 4) * 4;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline void matrix<double, thisTransposed, thisContiguous>::erase(size_t index)
	{
		if constexpr (thisTransposed)
		{
			this->_cols--;
			this->actualCols--;
			this->_size = this->_rows * this->_cols;
			this->finalPosCols = (this->_cols / 4) * 4;
			this->finalPosSize = (this->_size / 4) * 4;

			if (this->dataToDelete == nullptr)
			{
				double* newData = new double[this->_rows * this->_cols];
				double* oldData = this->_data;

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
			this->finalPosRows = (this->_rows / 4) * 4;
			this->finalPosSize = (this->_size / 4) * 4;

			if (this->dataToDelete == nullptr)
			{
				double* newData = new double[this->_rows * this->_cols];
				double* oldData = this->_data;

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
	inline size_t matrix<double, thisTransposed, thisContiguous>::find(vector<double>& other)
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
	inline vector<uint64_t> matrix<double, thisTransposed, thisContiguous>::find(matrix<double, otherTransposed, otherContiguous>& other)
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

	template <bool thisTransposed, bool thisContiguous>
	inline void matrix<double, thisTransposed, thisContiguous>::insert(std::initializer_list<double> list, size_t index)
	{
		if constexpr (thisTransposed)
		{
#ifdef _DEBUG
			if (this->_rows != list.size()) throw std::invalid_argument("Error");
#else
#endif
			size_t newCols = this->_cols + 1;

			size_t increase = this->_capacityRows / 2;
			increase = increase > 0 ? increase : 1;
			this->_capacityRows += increase;

			double* newData = new double[this->_capacityRows * this->_rows];
			double* oldData = this->_data;

			for (size_t j = 0; j < index; j++)
			{
				for (size_t i = 0; i < this->_rows; i++)
				{
					newData[j * this->_rows + i] = oldData[j * this->_rows + i];
				}
			}
			for (size_t j{ index }, j2{ index + 1 }; j < this->_cols; j++, j2++)
			{
				for (size_t i = 0; i < this->_rows; i++)
				{
					newData[j2 * this->_rows + i] = oldData[j * this->_rows + i];
				}
			}

			for (size_t i = 0; i < this->_rows; i++)
			{
				newData[index * this->_rows + i] = *(list.begin() + i);
			}

			delete[] this->dataToDelete;
			this->_data = newData;
			this->dataToDelete = newData;

			this->_cols = newCols;
			this->_size = this->_rows * this->_cols;
			this->actualCols = this->_cols;
			this->actualRows = this->_rows;
			this->finalPosCols = (this->_cols / 4) * 4;
			this->finalPosSize = (this->_size / 4) * 4;
		}
		else
		{
#ifdef _DEBUG
			if (this->_cols != list.size()) throw std::invalid_argument("Error");
#else
#endif
			size_t newRows = this->_rows + 1;

			if (this->_capacityRows >= newRows)
			{
				double* tmp = new double[this->_cols];
				double* tmp2 = new double[this->_cols];
				for (size_t j = 0; j < this->_cols; j++) tmp[j] = *(list.begin() + j);
				for (size_t i = index; i < this->_rows; i++)
				{
					for (size_t j = 0; j < this->_cols; j++)
					{
						tmp2[j] = this->_data[i * this->actualCols + j];
						this->_data[i * this->actualCols + j] = tmp[j];
						tmp[j] = tmp2[j];
					}
				}
				delete[] tmp, tmp2;
			}
			else
			{
				size_t increase = this->_capacityRows / 2;
				increase = increase > 0 ? increase : 1;
				this->_capacityRows += increase;

				double* newData = new double[this->_capacityRows * this->_cols];
				double* oldData = this->_data;

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
						newData[i2 * this->_cols + j] = oldData[i * this->actualCols + j];
					}
				}
				for (size_t j = 0; j < this->_cols; j++)
				{
					newData[index * this->_cols + j] = *(list.begin() + j);
				}
				delete[] this->dataToDelete;
				this->_data = newData;
				this->dataToDelete = newData;
			}
			this->_rows = newRows;
			this->actualCols = this->_cols;
			this->actualRows = this->_rows;
			this->finalPosRows = (this->_rows / 4) * 4;
			this->_size = this->_rows * this->_cols;
			this->finalPosSize = (this->_size / 4) * 4;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline void matrix<double, thisTransposed, thisContiguous>::insert(vector<double>& vector1, size_t index)
	{
		if constexpr (thisTransposed)
		{
#ifdef _DEBUG
			if (this->_rows != vector1._size) throw std::invalid_argument("Error");
#else
#endif
			size_t newCols = this->_cols + 1;

			size_t increase = this->_capacityRows / 2;
			increase = increase > 0 ? increase : 1;
			this->_capacityRows += increase;

			double* newData = new double[this->_capacityRows * this->_rows];
			double* oldData = this->_data;

			for (size_t j = 0; j < index; j++)
			{
				for (size_t i = 0; i < this->_rows; i++)
				{
					newData[j * this->_rows + i] = oldData[j * this->_rows + i];
				}
			}
			for (size_t j{ index }, j2{ index + 1 }; j < this->_cols; j++, j2++)
			{
				for (size_t i = 0; i < this->_rows; i++)
				{
					newData[j2 * this->_rows + i] = oldData[j * this->_rows + i];
				}
			}

			for (size_t i = 0; i < this->_rows; i++)
			{
				newData[index * this->_rows + i] = vector1._data[i];
			}

			delete[] this->dataToDelete;
			this->_data = newData;
			this->dataToDelete = newData;

			this->_cols = newCols;
			this->_size = this->_rows * this->_cols;
			this->actualCols = this->_cols;
			this->actualRows = this->_rows;
			this->finalPosCols = (this->_cols / 4) * 4;
			this->finalPosSize = (this->_size / 4) * 4;
		}
		else
		{
#ifdef _DEBUG
			if (this->_cols != vector1._size) throw std::invalid_argument("Error");
#else
#endif
			size_t newRows = this->_rows + 1;

			if (this->_capacityRows >= newRows)
			{
				double* tmp = new double[this->_cols];
				double* tmp2 = new double[this->_cols];
				for (size_t j = 0; j < this->_cols; j++) tmp[j] = vector1._data[j];
				for (size_t i = index; i < this->_rows; i++)
				{
					for (size_t j = 0; j < this->_cols; j++)
					{
						tmp2[j] = this->_data[i * this->actualCols + j];
						this->_data[i * this->actualCols + j] = tmp[j];
						tmp[j] = tmp2[j];
					}
				}
				delete[] tmp, tmp2;
			}
			else
			{
				size_t increase = this->_capacityRows / 2;
				increase = increase > 0 ? increase : 1;
				this->_capacityRows += increase;

				double* newData = new double[this->_capacityRows * this->_cols];
				double* oldData = this->_data;

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
						newData[i2 * this->_cols + j] = oldData[i * this->actualCols + j];
					}
				}
				for (size_t j = 0; j < this->_cols; j++)
				{
					newData[index * this->_cols + j] = vector1._data[j];
				}
				delete[] this->dataToDelete;
				this->_data = newData;
				this->dataToDelete = newData;
			}
			this->_rows = newRows;
			this->actualCols = this->_cols;
			this->actualRows = this->_rows;
			this->finalPosRows = (this->_rows / 4) * 4;
			this->_size = this->_rows * this->_cols;
			this->finalPosSize = (this->_size / 4) * 4;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool otherTransposed, bool otherContiguous>
	inline void matrix<double, thisTransposed, thisContiguous>::insert(matrix<double, otherTransposed, otherContiguous>& matrix1, size_t index)
	{
		if constexpr (thisTransposed)
		{
#ifdef _DEBUG
			if ((this->_rows != matrix1._rows) || (matrix1._cols > 1)) throw std::invalid_argument("Error");
#else
#endif
			size_t newCols = this->_cols + 1;

			size_t increase = this->_capacityRows / 2;
			increase = increase > 0 ? increase : 1;
			this->_capacityRows += increase;

			double* newData = new double[this->_capacityRows * this->_rows];
			double* oldData = this->_data;

			for (size_t j = 0; j < index; j++)
			{
				for (size_t i = 0; i < this->_rows; i++)
				{
					newData[j * this->_rows + i] = oldData[j * this->_rows + i];
				}
			}
			for (size_t j{ index }, j2{ index + 1 }; j < this->_cols; j++, j2++)
			{
				for (size_t i = 0; i < this->_rows; i++)
				{
					newData[j2 * this->_rows + i] = oldData[j * this->_rows + i];
				}
			}

			for (size_t i = 0; i < this->_rows; i++)
			{
				if constexpr (otherTransposed)
				{
					newData[index * this->_rows + i] = matrix1._data[i];
				}
				else
				{
					newData[index * this->_rows + i] = matrix1._data[i * this->actualCols];
				}
			}

			delete[] this->dataToDelete;
			this->_data = newData;
			this->dataToDelete = newData;

			this->_cols = newCols;
			this->_size = this->_rows * this->_cols;
			this->actualCols = this->_cols;
			this->actualRows = this->_rows;
			this->finalPosCols = (this->_cols / 4) * 4;
			this->finalPosSize = (this->_size / 4) * 4;
		}
		else
		{
#ifdef _DEBUG
			if ((this->_cols != matrix1._cols) || (matrix1._rows > 1)) throw std::invalid_argument("Error");
#else
#endif
			size_t newRows = this->_rows + 1;

			if (this->_capacityRows >= newRows)
			{
				double* tmp = new double[this->_cols];
				double* tmp2 = new double[this->_cols];
				for (size_t j = 0; j < this->_cols; j++)
				{
					if constexpr (otherTransposed)
					{
						tmp[j] = matrix1._data[j * this->actualRows];
					}
					else
					{
						tmp[j] = matrix1._data[j];
					}
				}
				for (size_t i = index; i < this->_rows; i++)
				{
					for (size_t j = 0; j < this->_cols; j++)
					{
						tmp2[j] = this->_data[i * this->actualCols + j];
						this->_data[i * this->actualCols + j] = tmp[j];
						tmp[j] = tmp2[j];
					}
				}
				delete[] tmp, tmp2;
			}
			else
			{
				size_t increase = this->_capacityRows / 2;
				increase = increase > 0 ? increase : 1;
				this->_capacityRows += increase;

				double* newData = new double[this->_capacityRows * this->_cols];
				double* oldData = this->_data;

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
						newData[i2 * this->_cols + j] = oldData[i * this->actualCols + j];
					}
				}
				for (size_t j = 0; j < this->_cols; j++)
				{
					if constexpr (otherTransposed)
					{
						newData[index * this->_cols + j] = matrix1._data[j * this->actualRows];
					}
					else
					{
						newData[index * this->_cols + j] = matrix1._data[j];
					}
				}
				delete[] this->dataToDelete;
				this->_data = newData;
				this->dataToDelete = newData;
			}
			this->_rows = newRows;
			this->actualCols = this->_cols;
			this->actualRows = this->_rows;
			this->finalPosRows = (this->_rows / 4) * 4;
			this->_size = this->_rows * this->_cols;
			this->finalPosSize = (this->_size / 4) * 4;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	template <bool otherContiguous>
	inline vector<uint8_t> matrix<double, thisTransposed, thisContiguous>::in(matrix<double, false, otherContiguous>& other)
	{
		double* data1 = this->_data;
		double* data2 = other._data;

		vector<uint8_t> result(this->_rows);

		uint8_t* dataResult = result._data;

		for (size_t i = 0; i < this->_rows; i++)
		{
			uint8_t boolean = False;
			for (size_t j = 0; j < other._rows; j++)
			{
				bool equalRow = true;
				for (size_t k = 0; k < other.finalPosCols; k += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i * this->actualCols + k]);
					__m256d b = _mm256_load_pd(&data2[j * other.actualCols + k]);

					if (_mm256_movemask_pd(_mm256_cmp_pd(a, b, _CMP_NEQ_OQ)))
					{
						equalRow = false;
						break;
					}
				}
				for (size_t k = other.finalPosCols; k < other._cols; k++)
				{
					if (data1[i * this->actualCols + k] != data2[j * other.actualCols + k])
					{
						equalRow = false;
						break;
					}
				}
				if (equalRow)
				{
					boolean = True;
					break;
				}
			}
			dataResult[i] = boolean;
		}
		return result;
	}

	// Copy

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed>
	inline matrix<double> matrix<double, thisTransposed, thisContiguous>::copy()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		matrix<double> result(rows, cols);

		double* dataResult = result._data;

		size_t actualRows = this->actualRows;
		size_t actualCols = this->actualCols;

		if constexpr (returnTransposed)
		{
			matrix<double> result(cols, rows);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
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
						dataResult[j * rows + i] = data1[i * actualCols + j];
					}
				}
			}
			return result;
		}
		else
		{
			matrix<double> result(rows, cols);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						dataResult[i * cols + j] = data1[j * actualRows + i];
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
			return result;
		}
	}

	// =

	template <bool thisTransposed, bool thisContiguous>
	template<bool otherTransposed, bool otherContiguous>
	inline matrix<double, thisTransposed, thisContiguous>& matrix<double, thisTransposed, thisContiguous>::operator=(matrix<double, otherTransposed, otherContiguous>& other)
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
		}
		else
		{
#ifdef _DEBUG
			if (other._cols != this->_cols || other._rows != this->_rows) throw std::invalid_argument("The dimensions of both matrices must be the same");
#else
#endif
			size_t rows = this->_rows;
			size_t cols = this->_cols;

			double* data1 = this->_data;
			double* data2 = other._data;

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
	inline void matrix<double, thisTransposed, thisContiguous>::transfer(matrix<double, thisTransposed, otherContiguous>& other)
	{
#ifdef _DEBUG
		if (other.dataToDelete == nullptr)
		{
			std::cerr << "You can not transfer data from a matrix that does not own its data" << std::endl;
			exit(1);
		}
#else
#endif
		delete[] this->dataToDelete;

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
		this->_capacityRows = other._capacityRows;
	}

	// neg

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed>
	inline matrix<double> matrix<double, thisTransposed, thisContiguous>::operator-()

	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		__m256d b = _mm256_set1_pd(-0.0);

		size_t finalPosSize = this->finalPosSize;
		size_t size = this->_size;
		
		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix1ActualCols = this->actualCols;

		if constexpr (returnTransposed)
		{
			matrix<double> result(cols, rows);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&dataResult[i], _mm256_xor_pd(a, b));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = -data1[i];
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

							_mm256_store_pd(&dataResult[j * rows + i], _mm256_xor_pd(a, b));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = -data1[j * matrix1ActualRows + i];
						}
					}
				}
			}
			else
			{
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						dataResult[j * rows + i] = -data1[i * matrix1ActualCols + j];
					}
				}
			}
			return result;
		}
		else
		{
			matrix<double> result(rows, cols);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				for (size_t j = 0; j < cols; j++)
				{
					for (size_t i = 0; i < rows; i++)
					{
						dataResult[i * cols + j] = -data1[j * matrix1ActualRows + i];
					}
				}
			}
			else
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&dataResult[i], _mm256_xor_pd(a, b));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = -data1[i];
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

							_mm256_store_pd(&dataResult[i * cols + j], _mm256_xor_pd(a, b));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = -data1[i * matrix1ActualCols + j];
						}
					}
				}
			}
			return result;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline void matrix<double, thisTransposed, thisContiguous>::self_neg()

	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		__m256d b = _mm256_set1_pd(-0.0);

		if constexpr (thisTransposed)
		{
			if constexpr (thisContiguous)
			{
				size_t size = this->_size;

				size_t finalPosSize = this->finalPosSize;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);

					_mm256_store_pd(&data1[i], _mm256_xor_pd(a, b));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] = -data1[i];
				}
			}
			else
			{
				size_t finalPosRows = this->finalPosRows;
				size_t finalPosCols = this->finalPosCols;

				size_t matrix1ActualRows = this->actualRows;

				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						_mm256_store_pd(&data1[j * matrix1ActualRows + i], _mm256_xor_pd(a, b));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						data1[j * matrix1ActualRows + i] = -data1[j * matrix1ActualRows + i];
					}
				}
			}
		}
		else
		{
			if constexpr (thisContiguous)
			{
				size_t finalPosSize = this->finalPosSize;
				size_t size = this->_size;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);

					_mm256_store_pd(&data1[i], _mm256_xor_pd(a, b));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] = -data1[i];
				}
			}
			else
			{
				size_t matrix1ActualCols = this->actualCols;

				size_t finalPosCols = this->finalPosCols;
				size_t finalPosRows = this->finalPosRows;

				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						_mm256_store_pd(&data1[i * matrix1ActualCols + j], _mm256_xor_pd(a, b));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < rows; i++)
					{
						data1[i * matrix1ActualCols + j] = -data1[i * matrix1ActualCols + j];
					}
				}
			}
		}
	}

	// Set constant

	template <bool thisTransposed, bool thisContiguous>
	inline void matrix<double, thisTransposed, thisContiguous>::set_const(double num)
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		if constexpr (thisContiguous)
		{
			size_t size = this->_size;
			for (size_t i = 0; i < size; i++)
			{
				data1[i] = num;
			}
		}
		else if constexpr (thisTransposed)
		{
			size_t matrix1ActualRows = this->actualRows;
			for (size_t i = 0; i < rows; i++)
			{
				for (size_t j = 0; i < cols; j++)
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
				for (size_t j = 0; i < cols; j++)
				{
					data1[i * matrix1ActualCols + j] = num;
				}
			}
		}
	}

	// Rand

	template <bool thisTransposed, bool thisContiguous>
	inline void matrix<double, thisTransposed, thisContiguous>::rand()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		size_t matrix1FinalPosRows = this->finalPosRows;
		size_t matrix1FinalPosCols = this->finalPosRows;

		size_t matrix1ActualCols = this->actualCols;
		size_t matrix1ActualRows = this->actualRows;

		double* data1 = this->_data;

		__m256i random;

		masks_uint64_to_double;

		__m256d divisor = _mm256_set1_pd(18446744073709551615.0);

		if constexpr (thisContiguous)
		{
			size_t size = this->_size;
			size_t finalPosSize = this->finalPosSize;

			for (size_t i = 0; i < finalPosSize; i += 4)
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
			for (size_t i = finalPosSize; i < size; i++)
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
		else if constexpr (thisTransposed)
		{

			for (size_t i = 0; i < matrix1FinalPosRows; i += 4)
			{
				for (size_t j = 0; j < cols; j++)
				{
					random = _mm256_slli_epi64(__seeds__, 13);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					random = _mm256_srli_epi64(__seeds__, 10);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					random = _mm256_slli_epi64(__seeds__, 20);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					// uint64 to double

					uint64_to_double(__seeds__);

					_mm256_store_pd(&data1[j * matrix1ActualRows + i], _mm256_div_pd(uint64ToDouble, divisor));
				}
			}
			for (size_t i = matrix1FinalPosRows; i < rows; i++)
			{
				for (size_t j = 0; j < matrix1FinalPosCols; j += 4)
				{
					random = _mm256_slli_epi64(__seeds__, 13);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					random = _mm256_srli_epi64(__seeds__, 10);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					random = _mm256_slli_epi64(__seeds__, 20);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					uint64_to_double(__seeds__);

					uint64ToDouble = _mm256_div_pd(uint64ToDouble, divisor);

					__m128d val1 = _mm256_extractf128_pd(uint64ToDouble, 1);
					__m128d val2 = _mm256_castpd256_pd128(uint64ToDouble);

					_mm_store_sd(&data1[j * matrix1ActualRows + i], val2);
					val2 = _mm_shuffle_pd(val2, val2, 1);
					_mm_store_sd(&data1[(j + 1) * matrix1ActualRows + i], val2);

					_mm_store_sd(&data1[(j + 2) * matrix1ActualRows + i], val1);
					val1 = _mm_shuffle_pd(val1, val1, 1);
					_mm_store_sd(&data1[(j + 3) * matrix1ActualRows + i], val1);
				}
				for (size_t j = matrix1FinalPosCols; j < cols; j++)
				{
					random = _mm256_slli_epi64(__seeds__, 13);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					random = _mm256_srli_epi64(__seeds__, 10);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					random = _mm256_slli_epi64(__seeds__, 20);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					// uint64 to double

					uint64_to_double(__seeds__);

					_mm_store_sd(&data1[j * matrix1ActualRows + i], _mm256_castpd256_pd128(_mm256_div_pd(uint64ToDouble, divisor)));
				}
			}
		}
		else
		{
			for (size_t j = 0; j < matrix1FinalPosCols; j += 4)
			{
				for (size_t i = 0; i < rows; i++)
				{
					random = _mm256_slli_epi64(__seeds__, 13);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					random = _mm256_srli_epi64(__seeds__, 10);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					random = _mm256_slli_epi64(__seeds__, 20);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					// uint64 to double

					uint64_to_double(__seeds__);

					_mm256_store_pd(&data1[i * matrix1ActualCols + j], _mm256_div_pd(uint64ToDouble, divisor));
				}
			}
			for (size_t j = matrix1FinalPosCols; j < cols; j++)
			{
				for (size_t i = 0; i < matrix1FinalPosRows; i += 4)
				{
					random = _mm256_slli_epi64(__seeds__, 13);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					random = _mm256_srli_epi64(__seeds__, 10);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					random = _mm256_slli_epi64(__seeds__, 20);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					// uint64 to double

					uint64_to_double(__seeds__);

					uint64ToDouble = _mm256_div_pd(uint64ToDouble, divisor);

					__m128d val1 = _mm256_extractf128_pd(uint64ToDouble, 1);
					__m128d val2 = _mm256_castpd256_pd128(uint64ToDouble);

					_mm_store_sd(&data1[i * matrix1ActualCols + j], val2);
					val2 = _mm_shuffle_pd(val2, val2, 1);
					_mm_store_sd(&data1[(i + 1) * matrix1ActualCols + j], val2);

					_mm_store_sd(&data1[(i + 2) * matrix1ActualCols + j], val1);
					val1 = _mm_shuffle_pd(val1, val1, 1);
					_mm_store_sd(&data1[(i + 3) * matrix1ActualCols + j], val1);
				}
				for (size_t i = matrix1FinalPosRows; i < rows; i++)
				{
					random = _mm256_slli_epi64(__seeds__, 13);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					random = _mm256_srli_epi64(__seeds__, 10);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					random = _mm256_slli_epi64(__seeds__, 20);
					__seeds__ = _mm256_xor_si256(random, __seeds__);

					// uint64 to double

					uint64_to_double(__seeds__);

					_mm_store_sd(&data1[i * matrix1ActualCols + j], _mm256_castpd256_pd128(_mm256_div_pd(uint64ToDouble, divisor)));
				}
			}
		}
	}

	// Identity

	template <bool thisTransposed, bool thisContiguous>
	inline void matrix<double, thisTransposed, thisContiguous>::identity()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		if constexpr (thisTransposed)
		{
			size_t matrix1ActualRows = this->actualRows;

			for (size_t i = 0; i < rows; i++)
			{
				for (size_t j = 0; j < cols; j++)
				{
					if (i == j)
					{
						data1[j * matrix1ActualRows + i] = 1.0;
					}
					else
					{
						data1[j * matrix1ActualRows + i] = 0.0;
					}
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
					if (i == j)
					{
						data1[i * matrix1ActualCols + j] = 1.0;
					}
					else
					{
						data1[i * matrix1ActualCols + j] = 0.0;
					}
				}
			}
		}
	}

	// +

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed, bool otherTransposed, bool otherContiguous>
	inline matrix<double> matrix<double, thisTransposed, thisContiguous>::operator+(const matrix<double, otherTransposed, otherContiguous>& other)
	{
#ifdef _DEBUG
		if (other._cols != this->_cols || other._rows != this->_rows)
		{
			std::cerr << "The dimensions of both matrices must be the same " << std::endl;
			std::cerr << "Matrix 1: " << std::endl;
			std::cerr << "Columns: " << this->_cols << std::endl;
			std::cerr << "Rows: " << this->_rows << std::endl;
			std::cerr << std::endl;
			std::cerr << "Matrix 2: " << std::endl;
			std::cerr << "Columns: " << other._cols << std::endl;
			std::cerr << "Rows: " << other._rows << std::endl;
			exit(1);
		}
#else
#endif

		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;
		double* data2 = other._data;

		size_t size = this->_size;

		size_t finalPosSize = this->finalPosSize;

		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix2ActualRows = other.actualRows;
		size_t matrix1ActualCols = this->actualCols;
		size_t matrix2ActualCols = other.actualCols;
		
		if constexpr (returnTransposed)
		{
			matrix<double> result(cols, rows);
			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (otherTransposed)
				{
					if constexpr (thisContiguous && otherContiguous)
					{
						for (size_t i = 0; i < finalPosSize; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);
							__m256d b = _mm256_load_pd(&data2[i]);

							_mm256_store_pd(&dataResult[i], _mm256_add_pd(a, b));
						}
						for (size_t i = finalPosSize; i < size; i++)
						{
							dataResult[i] = data1[i] + data2[i];
						}
					}
					else
					{
						for (size_t i = 0; i < finalPosRows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);
								__m256d b = _mm256_load_pd(&data2[j * matrix2ActualRows + i]);

								_mm256_store_pd(&dataResult[j * rows + i], _mm256_add_pd(a, b));
							}
						}
						for (size_t i = finalPosRows; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								dataResult[j * rows + i] = data1[j * matrix1ActualRows + i] + data2[j * matrix2ActualRows + i];
							}
						}
					}
				}
				else
				{
					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = data1[j * matrix1ActualRows + i] + data2[i * matrix2ActualCols + j];
						}
					}
				}
			}
			else
			{
				if constexpr (otherTransposed)
				{
					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = data1[i * matrix1ActualCols + j] + data2[j * matrix2ActualRows + i];
						}
					}
				}
				else
				{
					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < finalPosCols; j += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
							__m256d b = _mm256_load_pd(&data2[i * matrix2ActualCols + j]);

							__m256d add = _mm256_add_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(add, 1);
							__m128d val2 = _mm256_castpd256_pd128(add);

							_mm_store_sd(&dataResult[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

							_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
						}
						for (size_t j = finalPosCols; j < cols; j++)
						{
							dataResult[j * rows + i] = data1[i * matrix1ActualCols + j] + data2[i * matrix2ActualCols + j];
						}
					}
				}
			}
			return result;
		}
		else
		{
			matrix<double> result(rows, cols);
			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (otherTransposed)
				{
					for (size_t j = 0; j < cols; j++)
					{
						for (size_t i = 0; i < finalPosRows; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);
							__m256d b = _mm256_load_pd(&data2[j * matrix2ActualRows + i]);

							__m256d add = _mm256_add_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(add, 1);
							__m128d val2 = _mm256_castpd256_pd128(add);

							_mm_store_sd(&dataResult[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

							_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
						}
						for (size_t i = finalPosRows; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[j * matrix1ActualRows + i] + data2[j * matrix2ActualRows + i];
						}
					}
				}
				else
				{
					for (size_t j = 0; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[j * matrix1ActualRows + i] + data2[i * matrix2ActualCols + j];
						}
					}
				}
			}
			else
			{
				if constexpr (otherTransposed)
				{
					for (size_t j = 0; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] + data2[j * matrix2ActualRows + i];
						}
					}
				}
				else
				{
					if constexpr (thisContiguous && otherContiguous)
					{
						for (size_t i = 0; i < finalPosSize; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);
							__m256d b = _mm256_load_pd(&data2[i]);

							_mm256_store_pd(&dataResult[i], _mm256_add_pd(a, b));
						}
						for (size_t i = finalPosSize; i < size; i++)
						{
							dataResult[i] = data1[i] + data2[i];
						}
					}
					else
					{
						size_t matrix1ActualCols = this->actualCols;
						size_t matrix2ActualCols = other.actualCols;

						size_t finalPosCols = this->finalPosCols;
						size_t finalPosRows = this->finalPosRows;

						for (size_t j = 0; j < finalPosCols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
								__m256d b = _mm256_load_pd(&data2[i * matrix2ActualCols + j]);

								_mm256_store_pd(&dataResult[i * cols + j], _mm256_add_pd(a, b));
							}
						}
						for (size_t j = finalPosCols; j < cols; j++)
						{
							for (size_t i = 0; i < rows; i++)
							{
								dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] + data2[i * matrix2ActualCols + j];
							}
						}
					}
				}
			}
			return result;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool otherTransposed, bool otherContiguous>
	inline void matrix<double, thisTransposed, thisContiguous>::operator+=(const matrix<double, otherTransposed, otherContiguous>& other)
	{
#ifdef _DEBUG
		if (other._cols != this->_cols || other._rows != this->_rows)
		{
			std::cerr << "The dimensions of both matrices must be the same " << std::endl;
			std::cerr << "Matrix 1: " << std::endl;
			std::cerr << "Columns: " << this->_cols << std::endl;
			std::cerr << "Rows: " << this->_rows << std::endl;
			std::cerr << std::endl;
			std::cerr << "Matrix 2: " << std::endl;
			std::cerr << "Columns: " << other._cols << std::endl;
			std::cerr << "Rows: " << other._rows << std::endl;
			exit(1);
		}
#else
#endif

		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;
		double* data2 = other._data;

		if constexpr (thisTransposed)
		{
			if constexpr (otherTransposed)
			{
				if constexpr (thisContiguous && otherContiguous)
				{
					size_t size = this->_size;

					size_t finalPosSize = this->finalPosSize;

					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);
						__m256d b = _mm256_load_pd(&data2[i]);

						_mm256_store_pd(&data1[i], _mm256_add_pd(a, b));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						data1[i] += data2[i];
					}
				}
				else
				{
					size_t finalPosRows = this->finalPosRows;
					size_t finalPosCols = this->finalPosCols;

					size_t matrix1ActualRows = this->actualRows;
					size_t matrix2ActualRows = other.actualRows;

					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);
							__m256d b = _mm256_load_pd(&data2[j * matrix2ActualRows + i]);

							_mm256_store_pd(&data1[j * matrix1ActualRows + i], _mm256_add_pd(a, b));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							data1[j * matrix1ActualRows + i] += data2[j * matrix2ActualRows + i];
						}
					}
				}
			}
			else
			{
				size_t finalPosRows = this->finalPosRows;
				size_t finalPosCols = this->finalPosCols;

				size_t matrix1ActualRows = this->actualRows;
				size_t matrix2ActualCols = other.actualCols;

				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						data1[j * matrix1ActualRows + i] += data2[i * matrix2ActualCols + j];
					}
				}
			}
		}
		else
		{
			if constexpr (otherTransposed)
			{
				size_t matrix1ActualCols = this->actualCols;
				size_t matrix2ActualRows = other.actualRows;

				size_t finalPosCols = this->finalPosCols;
				size_t finalPosRows = this->finalPosRows;

				for (size_t j = 0; j < cols; j++)
				{
					for (size_t i = 0; i < rows; i++)
					{
						data1[i * matrix1ActualCols + j] += data2[j * matrix2ActualRows + i];
					}
				}
			}
			else
			{
				if constexpr (thisContiguous && otherContiguous)
				{
					size_t finalPosSize = this->finalPosSize;
					size_t size = this->_size;

					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);
						__m256d b = _mm256_load_pd(&data2[i]);

						_mm256_store_pd(&data1[i], _mm256_add_pd(a, b));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						data1[i] += data2[i];
					}
				}
				else
				{
					size_t matrix1ActualCols = this->actualCols;
					size_t matrix2ActualCols = other.actualCols;

					size_t finalPosCols = this->finalPosCols;
					size_t finalPosRows = this->finalPosRows;

					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
							__m256d b = _mm256_load_pd(&data2[i * matrix2ActualCols + j]);

							_mm256_store_pd(&data1[i * matrix1ActualCols + j], _mm256_add_pd(a, b));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							data1[i * matrix1ActualCols + j] += data2[i * matrix2ActualCols + j];
						}
					}
				}
			}
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed>
	inline matrix<double> matrix<double, thisTransposed, thisContiguous>::operator+(double num)
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		size_t size = this->_size;

		double* data1 = this->_data;

		__m256d b = _mm256_set1_pd(num);

		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;
		size_t finalPosSize = this->finalPosSize;

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix1ActualCols = this->actualCols;

		if constexpr (returnTransposed)
		{
			matrix<double> result(cols, rows);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&dataResult[i], _mm256_add_pd(a, b));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = data1[i] + num;
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

							_mm256_store_pd(&dataResult[j * rows + i], _mm256_add_pd(a, b));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = data1[j * matrix1ActualRows + i] + num;
						}
					}
				}
			}
			else
			{
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						__m256d add = _mm256_add_pd(a, b);

						__m128d val1 = _mm256_extractf128_pd(add, 1);
						__m128d val2 = _mm256_castpd256_pd128(add);

						_mm_store_sd(&dataResult[j * rows + i], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

						_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						dataResult[j * rows + i] = data1[i * matrix1ActualCols + j] + num;
					}
				}
			}
			return result;
		}
		else
		{
			matrix<double> result(rows, cols);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				for (size_t j = 0; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						__m256d add = _mm256_add_pd(a, b);

						__m128d val1 = _mm256_extractf128_pd(add, 1);
						__m128d val2 = _mm256_castpd256_pd128(add);

						_mm_store_sd(&dataResult[i * cols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

						_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						dataResult[i * cols + j] = data1[j * matrix1ActualRows + i] + num;
					}
				}
			}
			else
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&dataResult[i], _mm256_add_pd(a, b));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = data1[i] + num;
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

							_mm256_store_pd(&dataResult[i * cols + j], _mm256_add_pd(a, b));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] + num;
						}
					}
				}
			}
			return result;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline void matrix<double, thisTransposed, thisContiguous>::operator+=(double num)
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		__m256d b = _mm256_set1_pd(num);

		if constexpr (thisTransposed)
		{
			if constexpr (thisContiguous)
			{
				size_t size = this->_size;

				size_t finalPosSize = this->finalPosSize;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);

					_mm256_store_pd(&data1[i], _mm256_add_pd(a, b));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] += num;
				}
			}
			else
			{
				size_t finalPosRows = this->finalPosRows;
				size_t finalPosCols = this->finalPosCols;

				size_t matrix1ActualRows = this->actualRows;

				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						_mm256_store_pd(&data1[j * matrix1ActualRows + i], _mm256_add_pd(a, b));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						data1[j * matrix1ActualRows + i] += num;
					}
				}
			}
		}
		else
		{
			if constexpr (thisContiguous)
			{
				size_t finalPosSize = this->finalPosSize;
				size_t size = this->_size;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);

					_mm256_store_pd(&data1[i], _mm256_add_pd(a, b));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] += num;
				}
			}
			else
			{
				size_t matrix1ActualCols = this->actualCols;

				size_t finalPosCols = this->finalPosCols;
				size_t finalPosRows = this->finalPosRows;

				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						_mm256_store_pd(&data1[i * matrix1ActualCols + j], _mm256_add_pd(a, b));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < rows; i++)
					{
						data1[i * matrix1ActualCols + j] += num;
					}
				}
			}
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed>
	inline matrix<double> matrix<double, thisTransposed, thisContiguous>::operator+(const vector<double>& other)
	{
#ifdef _DEBUG
		if (this->_cols != other._size)
		{
			std::cerr << "The number of columns must match the number of elements in the vector " << this->_cols << " != " << other._size << std::endl;
			exit(1);
		}
#else
#endif
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		size_t actualRows = this->actualRows;
		size_t actualCols = this->actualCols;

		if constexpr (returnTransposed)
		{
			matrix<double> result(cols, rows);
			double* data1 = this->_data;
			double* data2 = other._data;

			double* dataResult = result._data;

			size_t finalPosCols = this->finalPosCols;
			size_t finalPosRows = this->finalPosRows;

			if constexpr (thisTransposed)
			{
				for (size_t j = 0; j < cols; j++)
				{
					__m256d b = _mm256_broadcast_sd(&data2[j]);
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[j * actualRows + i]);
						_mm256_store_pd(&dataResult[j * rows + i], _mm256_add_pd(a, b));
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						dataResult[j * rows + i] = data1[j * actualRows + i] + data2[j];
					}
				}
			}
			else
			{
				for (size_t j = 0; j < cols; j++)
				{
					for (size_t i = 0; i < rows; i++)
					{
						dataResult[j * rows + i] = data1[i * actualCols + j] + data2[j];
					}
				}
			}

			return result;
		}
		else
		{
			matrix<double> result(rows, cols);
			double* data1 = this->_data;
			double* data2 = other._data;

			double* dataResult = result._data;

			size_t finalPosCols = this->finalPosCols;
			size_t finalPosRows = this->finalPosRows;

			if constexpr (thisTransposed)
			{
				for (size_t j = 0; j < cols; j++)
				{
					__m256d b = _mm256_broadcast_sd(&data2[j]);
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[j * actualRows + i]);

						__m256d add = _mm256_add_pd(a, b);

						__m128d val1 = _mm256_extractf128_pd(add, 1);
						__m128d val2 = _mm256_castpd256_pd128(add);

						_mm_store_sd(&dataResult[i * cols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

						_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						dataResult[i * cols + j] = data1[j * actualRows + i] + data2[j];
					}
				}
			}
			else
			{
				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					__m256d b = _mm256_load_pd(&data2[j]);
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_load_pd(&data1[i * actualCols + j]);
						_mm256_store_pd(&dataResult[i * cols + j], _mm256_add_pd(a, b));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < rows; i++)
					{
						dataResult[i * cols + j] = data1[i * actualCols + j] + data2[j];
					}
				}
			}

			return result;
		}

	}

	template <bool thisTransposed, bool thisContiguous>
	inline void  matrix<double, thisTransposed, thisContiguous>::operator+=(const vector<double>& other)
	{
#ifdef _DEBUG
		if (this->_cols != other._size)
		{
			std::cerr << "The number of columns must match the number of elements in the vector " << this->_cols << " != " << other._size << std::endl;
			exit(1);
		}
#else
#endif
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;
		double* data2 = other._data;

		size_t finalPosCols = this->finalPosCols;
		size_t finalPosRows = this->finalPosRows;

		if constexpr (thisTransposed)
		{
			size_t actualRows = this->actualRows;

			for (size_t j = 0; j < cols; j++)
			{
				__m256d b = _mm256_broadcast_sd(&data2[j]);
				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[j * actualRows + i]);
					_mm256_store_pd(&data1[j * rows + i], _mm256_add_pd(a, b));
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					data1[j * rows + i] += data2[j];
				}
			}
		}
		else
		{
			size_t actualCols = this->actualCols;

			for (size_t j = 0; j < finalPosCols; j += 4)
			{
				__m256d b = _mm256_load_pd(&data2[j]);
				for (size_t i = 0; i < rows; i++)
				{
					__m256d a = _mm256_load_pd(&data1[i * actualCols + j]);
					_mm256_store_pd(&data1[i * cols + j], _mm256_add_pd(a, b));
				}
			}
			for (size_t j = finalPosCols; j < cols; j++)
			{
				for (size_t i = 0; i < rows; i++)
				{
					data1[i * cols + j] += data2[j];
				}
			}
		}
	}

	// -

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed, bool otherTransposed, bool otherContiguous>
	inline matrix<double> matrix<double, thisTransposed, thisContiguous>::operator-(const matrix<double, otherTransposed, otherContiguous>& other)
	{
#ifdef _DEBUG
		if (other._cols != this->_cols || other._rows != this->_rows)
		{
			std::cerr << "The dimensions of both matrices must be the same " << std::endl;
			std::cerr << "Matrix 1: " << std::endl;
			std::cerr << "Columns: " << this->_cols << std::endl;
			std::cerr << "Rows: " << this->_rows << std::endl;
			std::cerr << std::endl;
			std::cerr << "Matrix 2: " << std::endl;
			std::cerr << "Columns: " << other._cols << std::endl;
			std::cerr << "Rows: " << other._rows << std::endl;
			exit(1);
	}
#else
#endif

		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;
		double* data2 = other._data;

		size_t size = this->_size;

		size_t finalPosSize = this->finalPosSize;

		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix2ActualRows = other.actualRows;
		size_t matrix1ActualCols = this->actualCols;
		size_t matrix2ActualCols = other.actualCols;

		if constexpr (returnTransposed)
		{
			matrix<double> result(cols, rows);
			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (otherTransposed)
				{
					if constexpr (thisContiguous && otherContiguous)
					{
						for (size_t i = 0; i < finalPosSize; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);
							__m256d b = _mm256_load_pd(&data2[i]);

							_mm256_store_pd(&dataResult[i], _mm256_sub_pd(a, b));
						}
						for (size_t i = finalPosSize; i < size; i++)
						{
							dataResult[i] = data1[i] - data2[i];
						}
					}
					else
					{
						for (size_t i = 0; i < finalPosRows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);
								__m256d b = _mm256_load_pd(&data2[j * matrix2ActualRows + i]);

								_mm256_store_pd(&dataResult[j * rows + i], _mm256_sub_pd(a, b));
							}
						}
						for (size_t i = finalPosRows; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								dataResult[j * rows + i] = data1[j * matrix1ActualRows + i] - data2[j * matrix2ActualRows + i];
							}
						}
					}
				}
				else
				{
					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = data1[j * matrix1ActualRows + i] - data2[i * matrix2ActualCols + j];
						}
					}
				}
			}
			else
			{
				if constexpr (otherTransposed)
				{
					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = data1[i * matrix1ActualCols + j] - data2[j * matrix2ActualRows + i];
						}
					}
				}
				else
				{
					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < finalPosCols; j += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
							__m256d b = _mm256_load_pd(&data2[i * matrix2ActualCols + j]);

							__m256d sub = _mm256_sub_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(sub, 1);
							__m128d val2 = _mm256_castpd256_pd128(sub);

							_mm_store_sd(&dataResult[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

							_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
						}
						for (size_t j = finalPosCols; j < cols; j++)
						{
							dataResult[j * rows + i] = data1[i * matrix1ActualCols + j] - data2[i * matrix2ActualCols + j];
						}
					}
				}
			}
			return result;
		}
		else
		{
			matrix<double> result(rows, cols);
			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (otherTransposed)
				{
					for (size_t j = 0; j < cols; j++)
					{
						for (size_t i = 0; i < finalPosRows; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);
							__m256d b = _mm256_load_pd(&data2[j * matrix2ActualRows + i]);

							__m256d sub = _mm256_sub_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(sub, 1);
							__m128d val2 = _mm256_castpd256_pd128(sub);

							_mm_store_sd(&dataResult[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

							_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
						}
						for (size_t i = finalPosRows; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[j * matrix1ActualRows + i] - data2[j * matrix2ActualRows + i];
						}
					}
				}
				else
				{
					for (size_t j = 0; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[j * matrix1ActualRows + i] - data2[i * matrix2ActualCols + j];
						}
					}
				}
			}
			else
			{
				if constexpr (otherTransposed)
				{
					for (size_t j = 0; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] - data2[j * matrix2ActualRows + i];
						}
					}
				}
				else
				{
					if constexpr (thisContiguous && otherContiguous)
					{
						for (size_t i = 0; i < finalPosSize; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);
							__m256d b = _mm256_load_pd(&data2[i]);

							_mm256_store_pd(&dataResult[i], _mm256_sub_pd(a, b));
						}
						for (size_t i = finalPosSize; i < size; i++)
						{
							dataResult[i] = data1[i] - data2[i];
						}
					}
					else
					{
						size_t matrix1ActualCols = this->actualCols;
						size_t matrix2ActualCols = other.actualCols;

						size_t finalPosCols = this->finalPosCols;
						size_t finalPosRows = this->finalPosRows;

						for (size_t j = 0; j < finalPosCols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
								__m256d b = _mm256_load_pd(&data2[i * matrix2ActualCols + j]);

								_mm256_store_pd(&dataResult[i * cols + j], _mm256_sub_pd(a, b));
							}
						}
						for (size_t j = finalPosCols; j < cols; j++)
						{
							for (size_t i = 0; i < rows; i++)
							{
								dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] - data2[i * matrix2ActualCols + j];
							}
						}
					}
				}
			}
			return result;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool otherTransposed, bool otherContiguous>
	inline void matrix<double, thisTransposed, thisContiguous>::operator-=(const matrix<double, otherTransposed, otherContiguous>& other)
	{
#ifdef _DEBUG
		if (other._cols != this->_cols || other._rows != this->_rows)
		{
			std::cerr << "The dimensions of both matrices must be the same " << std::endl;
			std::cerr << "Matrix 1: " << std::endl;
			std::cerr << "Columns: " << this->_cols << std::endl;
			std::cerr << "Rows: " << this->_rows << std::endl;
			std::cerr << std::endl;
			std::cerr << "Matrix 2: " << std::endl;
			std::cerr << "Columns: " << other._cols << std::endl;
			std::cerr << "Rows: " << other._rows << std::endl;
			exit(1);
		}
#else
#endif

		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;
		double* data2 = other._data;

		if constexpr (thisTransposed)
		{
			if constexpr (otherTransposed)
			{
				if constexpr (thisContiguous && otherContiguous)
				{
					size_t size = this->_size;

					size_t finalPosSize = this->finalPosSize;

					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);
						__m256d b = _mm256_load_pd(&data2[i]);

						_mm256_store_pd(&data1[i], _mm256_sub_pd(a, b));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						data1[i] -= data2[i];
					}
				}
				else
				{
					size_t finalPosRows = this->finalPosRows;
					size_t finalPosCols = this->finalPosCols;

					size_t matrix1ActualRows = this->actualRows;
					size_t matrix2ActualRows = other.actualRows;

					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);
							__m256d b = _mm256_load_pd(&data2[j * matrix2ActualRows + i]);

							_mm256_store_pd(&data1[j * matrix1ActualRows + i], _mm256_sub_pd(a, b));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							data1[j * matrix1ActualRows + i] -= data2[j * matrix2ActualRows + i];
						}
					}
				}
			}
			else
			{
				size_t finalPosRows = this->finalPosRows;
				size_t finalPosCols = this->finalPosCols;

				size_t matrix1ActualRows = this->actualRows;
				size_t matrix2ActualCols = other.actualCols;

				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						data1[j * matrix1ActualRows + i] -= data2[i * matrix2ActualCols + j];
					}
				}
			}
		}
		else
		{
			if constexpr (otherTransposed)
			{
				size_t matrix1ActualCols = this->actualCols;
				size_t matrix2ActualRows = other.actualRows;

				size_t finalPosCols = this->finalPosCols;
				size_t finalPosRows = this->finalPosRows;

				for (size_t j = 0; j < cols; j++)
				{
					for (size_t i = 0; i < rows; i++)
					{
						data1[i * matrix1ActualCols + j] -= data2[j * matrix2ActualRows + i];
					}
				}
			}
			else
			{
				if constexpr (thisContiguous && otherContiguous)
				{
					size_t finalPosSize = this->finalPosSize;
					size_t size = this->_size;

					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);
						__m256d b = _mm256_load_pd(&data2[i]);

						_mm256_store_pd(&data1[i], _mm256_sub_pd(a, b));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						data1[i] -= data2[i];
					}
				}
				else
				{
					size_t matrix1ActualCols = this->actualCols;
					size_t matrix2ActualCols = other.actualCols;

					size_t finalPosCols = this->finalPosCols;
					size_t finalPosRows = this->finalPosRows;

					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
							__m256d b = _mm256_load_pd(&data2[i * matrix2ActualCols + j]);

							_mm256_store_pd(&data1[i * matrix1ActualCols + j], _mm256_sub_pd(a, b));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							data1[i * matrix1ActualCols + j] -= data2[i * matrix2ActualCols + j];
						}
					}
				}
			}
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed>
	inline matrix<double> matrix<double, thisTransposed, thisContiguous>::operator-(double num)
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		size_t size = this->_size;

		double* data1 = this->_data;

		__m256d b = _mm256_set1_pd(num);

		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;
		size_t finalPosSize = this->finalPosSize;

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix1ActualCols = this->actualCols;

		if constexpr (returnTransposed)
		{
			matrix<double> result(cols, rows);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&dataResult[i], _mm256_sub_pd(a, b));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = data1[i] - num;
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

							_mm256_store_pd(&dataResult[j * rows + i], _mm256_sub_pd(a, b));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = data1[j * matrix1ActualRows + i] - num;
						}
					}
				}
			}
			else
			{
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						__m256d sub = _mm256_sub_pd(a, b);

						__m128d val1 = _mm256_extractf128_pd(sub, 1);
						__m128d val2 = _mm256_castpd256_pd128(sub);

						_mm_store_sd(&dataResult[j * rows + i], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

						_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						dataResult[j * rows + i] = data1[i * matrix1ActualCols + j] - num;
					}
				}
			}
			return result;
		}
		else
		{
			matrix<double> result(rows, cols);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				for (size_t j = 0; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						__m256d sub = _mm256_sub_pd(a, b);

						__m128d val1 = _mm256_extractf128_pd(sub, 1);
						__m128d val2 = _mm256_castpd256_pd128(sub);

						_mm_store_sd(&dataResult[i * cols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

						_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						dataResult[i * cols + j] = data1[j * matrix1ActualRows + i] - num;
					}
				}
			}
			else
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&dataResult[i], _mm256_sub_pd(a, b));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = data1[i] - num;
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

							_mm256_store_pd(&dataResult[i * cols + j], _mm256_sub_pd(a, b));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] - num;
						}
					}
				}
			}
			return result;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline void matrix<double, thisTransposed, thisContiguous>::operator-=(double num)
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		__m256d b = _mm256_set1_pd(num);

		if constexpr (thisTransposed)
		{
			if constexpr (thisContiguous)
			{
				size_t size = this->_size;

				size_t finalPosSize = this->finalPosSize;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);

					_mm256_store_pd(&data1[i], _mm256_sub_pd(a, b));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] -= num;
				}
			}
			else
			{
				size_t finalPosRows = this->finalPosRows;
				size_t finalPosCols = this->finalPosCols;

				size_t matrix1ActualRows = this->actualRows;

				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						_mm256_store_pd(&data1[j * matrix1ActualRows + i], _mm256_sub_pd(a, b));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						data1[j * matrix1ActualRows + i] -= num;
					}
				}
			}
		}
		else
		{
			if constexpr (thisContiguous)
			{
				size_t finalPosSize = this->finalPosSize;
				size_t size = this->_size;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);

					_mm256_store_pd(&data1[i], _mm256_sub_pd(a, b));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] -= num;
				}
			}
			else
			{
				size_t matrix1ActualCols = this->actualCols;

				size_t finalPosCols = this->finalPosCols;
				size_t finalPosRows = this->finalPosRows;

				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						_mm256_store_pd(&data1[i * matrix1ActualCols + j], _mm256_sub_pd(a, b));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < rows; i++)
					{
						data1[i * matrix1ActualCols + j] -= num;
					}
				}
			}
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed>
	inline matrix<double> matrix<double, thisTransposed, thisContiguous>::operator-(const vector<double>& other)
	{
#ifdef _DEBUG
		if (this->_cols != other._size)
		{
			std::cerr << "The number of columns must match the number of elements in the vector " << this->_cols << " != " << other._size << std::endl;
			exit(1);
		}
#else
#endif
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		size_t actualRows = this->actualRows;
		size_t actualCols = this->actualCols;

		if constexpr (returnTransposed)
		{
			matrix<double> result(cols, rows);
			double* data1 = this->_data;
			double* data2 = other._data;

			double* dataResult = result._data;

			size_t finalPosCols = this->finalPosCols;
			size_t finalPosRows = this->finalPosRows;

			if constexpr (thisTransposed)
			{
				for (size_t j = 0; j < cols; j++)
				{
					__m256d b = _mm256_broadcast_sd(&data2[j]);
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[j * actualRows + i]);
						_mm256_store_pd(&dataResult[j * rows + i], _mm256_sub_pd(a, b));
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						dataResult[j * rows + i] = data1[j * actualRows + i] - data2[j];
					}
				}
			}
			else
			{
				for (size_t j = 0; j < cols; j++)
				{
					for (size_t i = 0; i < rows; i++)
					{
						dataResult[j * rows + i] = data1[i * actualCols + j] - data2[j];
					}
				}
			}

			return result;
		}
		else
		{
			matrix<double> result(rows, cols);
			double* data1 = this->_data;
			double* data2 = other._data;

			double* dataResult = result._data;

			size_t finalPosCols = this->finalPosCols;
			size_t finalPosRows = this->finalPosRows;

			if constexpr (thisTransposed)
			{
				for (size_t j = 0; j < cols; j++)
				{
					__m256d b = _mm256_broadcast_sd(&data2[j]);
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[j * actualRows + i]);

						__m256d sub = _mm256_sub_pd(a, b);

						__m128d val1 = _mm256_extractf128_pd(sub, 1);
						__m128d val2 = _mm256_castpd256_pd128(sub);

						_mm_store_sd(&dataResult[i * cols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

						_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						dataResult[i * cols + j] = data1[j * actualRows + i] - data2[j];
					}
				}
			}
			else
			{
				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					__m256d b = _mm256_load_pd(&data2[j]);
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_load_pd(&data1[i * actualCols + j]);
						_mm256_store_pd(&dataResult[i * cols + j], _mm256_sub_pd(a, b));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < rows; i++)
					{
						dataResult[i * cols + j] = data1[i * actualCols + j] - data2[j];
					}
				}
			}

			return result;
		}

	}

	template <bool thisTransposed, bool thisContiguous>
	inline void  matrix<double, thisTransposed, thisContiguous>::operator-=(const vector<double>& other)
	{
#ifdef _DEBUG
		if (this->_cols != other._size)
		{
			std::cerr << "The number of columns must match the number of elements in the vector " << this->_cols << " != " << other._size << std::endl;
			exit(1);
		}
#else
#endif
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;
		double* data2 = other._data;

		size_t finalPosCols = this->finalPosCols;
		size_t finalPosRows = this->finalPosRows;

		if constexpr (thisTransposed)
		{
			size_t actualRows = this->actualRows;

			for (size_t j = 0; j < cols; j++)
			{
				__m256d b = _mm256_broadcast_sd(&data2[j]);
				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[j * actualRows + i]);
					_mm256_store_pd(&data1[j * rows + i], _mm256_sub_pd(a, b));
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					data1[j * rows + i] -= data2[j];
				}
			}
		}
		else
		{
			size_t actualCols = this->actualCols;

			for (size_t j = 0; j < finalPosCols; j += 4)
			{
				__m256d b = _mm256_load_pd(&data2[j]);
				for (size_t i = 0; i < rows; i++)
				{
					__m256d a = _mm256_load_pd(&data1[i * actualCols + j]);
					_mm256_store_pd(&data1[i * cols + j], _mm256_sub_pd(a, b));
				}
			}
			for (size_t j = finalPosCols; j < cols; j++)
			{
				for (size_t i = 0; i < rows; i++)
				{
					data1[i * cols + j] -= data2[j];
				}
			}
		}
	}

	// *

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed, bool otherTransposed, bool otherContiguous>
	inline matrix<double> matrix<double, thisTransposed, thisContiguous>::operator*(const matrix<double, otherTransposed, otherContiguous>& other)
	{
#ifdef _DEBUG
		if (other._cols != this->_cols || other._rows != this->_rows)
		{
			std::cerr << "The dimensions of both matrices must be the same " << std::endl;
			std::cerr << "Matrix 1: " << std::endl;
			std::cerr << "Columns: " << this->_cols << std::endl;
			std::cerr << "Rows: " << this->_rows << std::endl;
			std::cerr << std::endl;
			std::cerr << "Matrix 2: " << std::endl;
			std::cerr << "Columns: " << other._cols << std::endl;
			std::cerr << "Rows: " << other._rows << std::endl;
			exit(1);
	}
#else
#endif

		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;
		double* data2 = other._data;

		size_t size = this->_size;

		size_t finalPosSize = this->finalPosSize;

		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix2ActualRows = other.actualRows;
		size_t matrix1ActualCols = this->actualCols;
		size_t matrix2ActualCols = other.actualCols;

		if constexpr (returnTransposed)
		{
			matrix<double> result(cols, rows);
			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (otherTransposed)
				{
					if constexpr (thisContiguous && otherContiguous)
					{
						for (size_t i = 0; i < finalPosSize; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);
							__m256d b = _mm256_load_pd(&data2[i]);

							_mm256_store_pd(&dataResult[i], _mm256_mul_pd(a, b));
						}
						for (size_t i = finalPosSize; i < size; i++)
						{
							dataResult[i] = data1[i] * data2[i];
						}
					}
					else
					{
						for (size_t i = 0; i < finalPosRows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);
								__m256d b = _mm256_load_pd(&data2[j * matrix2ActualRows + i]);

								_mm256_store_pd(&dataResult[j * rows + i], _mm256_mul_pd(a, b));
							}
						}
						for (size_t i = finalPosRows; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								dataResult[j * rows + i] = data1[j * matrix1ActualRows + i] * data2[j * matrix2ActualRows + i];
							}
						}
					}
				}
				else
				{
					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = data1[j * matrix1ActualRows + i] * data2[i * matrix2ActualCols + j];
						}
					}
				}
			}
			else
			{
				if constexpr (otherTransposed)
				{
					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = data1[i * matrix1ActualCols + j] * data2[j * matrix2ActualRows + i];
						}
					}
				}
				else
				{
					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < finalPosCols; j += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
							__m256d b = _mm256_load_pd(&data2[i * matrix2ActualCols + j]);

							__m256d mul = _mm256_mul_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(mul, 1);
							__m128d val2 = _mm256_castpd256_pd128(mul);

							_mm_store_sd(&dataResult[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

							_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
						}
						for (size_t j = finalPosCols; j < cols; j++)
						{
							dataResult[j * rows + i] = data1[i * matrix1ActualCols + j] * data2[i * matrix2ActualCols + j];
						}
					}
				}
			}
			return result;
		}
		else
		{
			matrix<double> result(rows, cols);
			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (otherTransposed)
				{
					for (size_t j = 0; j < cols; j++)
					{
						for (size_t i = 0; i < finalPosRows; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);
							__m256d b = _mm256_load_pd(&data2[j * matrix2ActualRows + i]);

							__m256d mul = _mm256_mul_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(mul, 1);
							__m128d val2 = _mm256_castpd256_pd128(mul);

							_mm_store_sd(&dataResult[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

							_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
						}
						for (size_t i = finalPosRows; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[j * matrix1ActualRows + i] * data2[j * matrix2ActualRows + i];
						}
					}
				}
				else
				{
					for (size_t j = 0; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[j * matrix1ActualRows + i] * data2[i * matrix2ActualCols + j];
						}
					}
				}
			}
			else
			{
				if constexpr (otherTransposed)
				{
					for (size_t j = 0; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] * data2[j * matrix2ActualRows + i];
						}
					}
				}
				else
				{
					if constexpr (thisContiguous && otherContiguous)
					{
						for (size_t i = 0; i < finalPosSize; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);
							__m256d b = _mm256_load_pd(&data2[i]);

							_mm256_store_pd(&dataResult[i], _mm256_mul_pd(a, b));
						}
						for (size_t i = finalPosSize; i < size; i++)
						{
							dataResult[i] = data1[i] * data2[i];
						}
					}
					else
					{
						size_t matrix1ActualCols = this->actualCols;
						size_t matrix2ActualCols = other.actualCols;

						size_t finalPosCols = this->finalPosCols;
						size_t finalPosRows = this->finalPosRows;

						for (size_t j = 0; j < finalPosCols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
								__m256d b = _mm256_load_pd(&data2[i * matrix2ActualCols + j]);

								_mm256_store_pd(&dataResult[i * cols + j], _mm256_mul_pd(a, b));
							}
						}
						for (size_t j = finalPosCols; j < cols; j++)
						{
							for (size_t i = 0; i < rows; i++)
							{
								dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] * data2[i * matrix2ActualCols + j];
							}
						}
					}
				}
			}
			return result;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool otherTransposed, bool otherContiguous>
	inline void matrix<double, thisTransposed, thisContiguous>::operator*=(const matrix<double, otherTransposed, otherContiguous>& other)
	{
#ifdef _DEBUG
		if (other._cols != this->_cols || other._rows != this->_rows)
		{
			std::cerr << "The dimensions of both matrices must be the same " << std::endl;
			std::cerr << "Matrix 1: " << std::endl;
			std::cerr << "Columns: " << this->_cols << std::endl;
			std::cerr << "Rows: " << this->_rows << std::endl;
			std::cerr << std::endl;
			std::cerr << "Matrix 2: " << std::endl;
			std::cerr << "Columns: " << other._cols << std::endl;
			std::cerr << "Rows: " << other._rows << std::endl;
			exit(1);
		}
#else
#endif

		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;
		double* data2 = other._data;

		if constexpr (thisTransposed)
		{
			if constexpr (otherTransposed)
			{
				if constexpr (thisContiguous && otherContiguous)
				{
					size_t size = this->_size;

					size_t finalPosSize = this->finalPosSize;

					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);
						__m256d b = _mm256_load_pd(&data2[i]);

						_mm256_store_pd(&data1[i], _mm256_mul_pd(a, b));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						data1[i] *= data2[i];
					}
				}
				else
				{
					size_t finalPosRows = this->finalPosRows;
					size_t finalPosCols = this->finalPosCols;

					size_t matrix1ActualRows = this->actualRows;
					size_t matrix2ActualRows = other.actualRows;

					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);
							__m256d b = _mm256_load_pd(&data2[j * matrix2ActualRows + i]);

							_mm256_store_pd(&data1[j * matrix1ActualRows + i], _mm256_mul_pd(a, b));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							data1[j * matrix1ActualRows + i] *= data2[j * matrix2ActualRows + i];
						}
					}
				}
			}
			else
			{
				size_t finalPosRows = this->finalPosRows;
				size_t finalPosCols = this->finalPosCols;

				size_t matrix1ActualRows = this->actualRows;
				size_t matrix2ActualCols = other.actualCols;

				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						data1[j * matrix1ActualRows + i] *= data2[i * matrix2ActualCols + j];
					}
				}
			}
		}
		else
		{
			if constexpr (otherTransposed)
			{
				size_t matrix1ActualCols = this->actualCols;
				size_t matrix2ActualRows = other.actualRows;

				size_t finalPosCols = this->finalPosCols;
				size_t finalPosRows = this->finalPosRows;

				for (size_t j = 0; j < cols; j++)
				{
					for (size_t i = 0; i < rows; i++)
					{
						data1[i * matrix1ActualCols + j] *= data2[j * matrix2ActualRows + i];
					}
				}
			}
			else
			{
				if constexpr (thisContiguous && otherContiguous)
				{
					size_t finalPosSize = this->finalPosSize;
					size_t size = this->_size;

					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);
						__m256d b = _mm256_load_pd(&data2[i]);

						_mm256_store_pd(&data1[i], _mm256_mul_pd(a, b));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						data1[i] *= data2[i];
					}
				}
				else
				{
					size_t matrix1ActualCols = this->actualCols;
					size_t matrix2ActualCols = other.actualCols;

					size_t finalPosCols = this->finalPosCols;
					size_t finalPosRows = this->finalPosRows;

					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
							__m256d b = _mm256_load_pd(&data2[i * matrix2ActualCols + j]);

							_mm256_store_pd(&data1[i * matrix1ActualCols + j], _mm256_mul_pd(a, b));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							data1[i * matrix1ActualCols + j] *= data2[i * matrix2ActualCols + j];
						}
					}
				}
			}
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed>
	inline matrix<double> matrix<double, thisTransposed, thisContiguous>::operator*(double num)
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		size_t size = this->_size;

		double* data1 = this->_data;

		__m256d b = _mm256_set1_pd(num);

		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;
		size_t finalPosSize = this->finalPosSize;

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix1ActualCols = this->actualCols;

		if constexpr (returnTransposed)
		{
			matrix<double> result(cols, rows);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&dataResult[i], _mm256_mul_pd(a, b));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = data1[i] * num;
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

							_mm256_store_pd(&dataResult[j * rows + i], _mm256_mul_pd(a, b));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = data1[j * matrix1ActualRows + i] * num;
						}
					}
				}
			}
			else
			{
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						__m256d mul = _mm256_mul_pd(a, b);

						__m128d val1 = _mm256_extractf128_pd(mul, 1);
						__m128d val2 = _mm256_castpd256_pd128(mul);

						_mm_store_sd(&dataResult[j * rows + i], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

						_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						dataResult[j * rows + i] = data1[i * matrix1ActualCols + j] * num;
					}
				}
			}
			return result;
		}
		else
		{
			matrix<double> result(rows, cols);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				for (size_t j = 0; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						__m256d mul = _mm256_mul_pd(a, b);

						__m128d val1 = _mm256_extractf128_pd(mul, 1);
						__m128d val2 = _mm256_castpd256_pd128(mul);

						_mm_store_sd(&dataResult[i * cols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

						_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						dataResult[i * cols + j] = data1[j * matrix1ActualRows + i] * num;
					}
				}
			}
			else
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&dataResult[i], _mm256_mul_pd(a, b));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = data1[i] * num;
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

							_mm256_store_pd(&dataResult[i * cols + j], _mm256_mul_pd(a, b));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] * num;
						}
					}
				}
			}
			return result;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline void matrix<double, thisTransposed, thisContiguous>::operator*=(double num)
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		__m256d b = _mm256_set1_pd(num);

		if constexpr (thisTransposed)
		{
			if constexpr (thisContiguous)
			{
				size_t size = this->_size;

				size_t finalPosSize = this->finalPosSize;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);

					_mm256_store_pd(&data1[i], _mm256_mul_pd(a, b));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] *= num;
				}
			}
			else
			{
				size_t finalPosRows = this->finalPosRows;
				size_t finalPosCols = this->finalPosCols;

				size_t matrix1ActualRows = this->actualRows;

				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						_mm256_store_pd(&data1[j * matrix1ActualRows + i], _mm256_mul_pd(a, b));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						data1[j * matrix1ActualRows + i] *= num;
					}
				}
			}
		}
		else
		{
			if constexpr (thisContiguous)
			{
				size_t finalPosSize = this->finalPosSize;
				size_t size = this->_size;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);

					_mm256_store_pd(&data1[i], _mm256_mul_pd(a, b));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] *= num;
				}
			}
			else
			{
				size_t matrix1ActualCols = this->actualCols;

				size_t finalPosCols = this->finalPosCols;
				size_t finalPosRows = this->finalPosRows;

				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						_mm256_store_pd(&data1[i * matrix1ActualCols + j], _mm256_mul_pd(a, b));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < rows; i++)
					{
						data1[i * matrix1ActualCols + j] *= num;
					}
				}
			}
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed>
	inline matrix<double> matrix<double, thisTransposed, thisContiguous>::operator*(const vector<double>& other)
	{
#ifdef _DEBUG
		if (this->_cols != other._size)
		{
			std::cerr << "The number of columns must match the number of elements in the vector " << this->_cols << " != " << other._size << std::endl;
			exit(1);
		}
#else
#endif
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		size_t actualRows = this->actualRows;
		size_t actualCols = this->actualCols;

		if constexpr (returnTransposed)
		{
			matrix<double> result(cols, rows);
			double* data1 = this->_data;
			double* data2 = other._data;

			double* dataResult = result._data;

			size_t finalPosCols = this->finalPosCols;
			size_t finalPosRows = this->finalPosRows;

			if constexpr (thisTransposed)
			{
				for (size_t j = 0; j < cols; j++)
				{
					__m256d b = _mm256_broadcast_sd(&data2[j]);
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[j * actualRows + i]);
						_mm256_store_pd(&dataResult[j * rows + i], _mm256_mul_pd(a, b));
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						dataResult[j * rows + i] = data1[j * actualRows + i] * data2[j];
					}
				}
			}
			else
			{
				for (size_t j = 0; j < cols; j++)
				{
					for (size_t i = 0; i < rows; i++)
					{
						dataResult[j * rows + i] = data1[i * actualCols + j] * data2[j];
					}
				}
			}

			return result;
		}
		else
		{
			matrix<double> result(rows, cols);
			double* data1 = this->_data;
			double* data2 = other._data;

			double* dataResult = result._data;

			size_t finalPosCols = this->finalPosCols;
			size_t finalPosRows = this->finalPosRows;

			if constexpr (thisTransposed)
			{
				for (size_t j = 0; j < cols; j++)
				{
					__m256d b = _mm256_broadcast_sd(&data2[j]);
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[j * actualRows + i]);

						__m256d mul = _mm256_mul_pd(a, b);

						__m128d val1 = _mm256_extractf128_pd(mul, 1);
						__m128d val2 = _mm256_castpd256_pd128(mul);

						_mm_store_sd(&dataResult[i * cols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

						_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						dataResult[i * cols + j] = data1[j * actualRows + i] * data2[j];
					}
				}
			}
			else
			{
				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					__m256d b = _mm256_load_pd(&data2[j]);
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_load_pd(&data1[i * actualCols + j]);
						_mm256_store_pd(&dataResult[i * cols + j], _mm256_mul_pd(a, b));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < rows; i++)
					{
						dataResult[i * cols + j] = data1[i * actualCols + j] * data2[j];
					}
				}
			}

			return result;
		}

	}

	template <bool thisTransposed, bool thisContiguous>
	inline void  matrix<double, thisTransposed, thisContiguous>::operator*=(const vector<double>& other)
	{
#ifdef _DEBUG
		if (this->_cols != other._size)
		{
			std::cerr << "The number of columns must match the number of elements in the vector " << this->_cols << " != " << other._size << std::endl;
			exit(1);
		}
#else
#endif
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;
		double* data2 = other._data;

		size_t finalPosCols = this->finalPosCols;
		size_t finalPosRows = this->finalPosRows;

		if constexpr (thisTransposed)
		{
			size_t actualRows = this->actualRows;

			for (size_t j = 0; j < cols; j++)
			{
				__m256d b = _mm256_broadcast_sd(&data2[j]);
				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[j * actualRows + i]);
					_mm256_store_pd(&data1[j * rows + i], _mm256_mul_pd(a, b));
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					data1[j * rows + i] *= data2[j];
				}
			}
		}
		else
		{
			size_t actualCols = this->actualCols;

			for (size_t j = 0; j < finalPosCols; j += 4)
			{
				__m256d b = _mm256_load_pd(&data2[j]);
				for (size_t i = 0; i < rows; i++)
				{
					__m256d a = _mm256_load_pd(&data1[i * actualCols + j]);
					_mm256_store_pd(&data1[i * cols + j], _mm256_mul_pd(a, b));
				}
			}
			for (size_t j = finalPosCols; j < cols; j++)
			{
				for (size_t i = 0; i < rows; i++)
				{
					data1[i * cols + j] *= data2[j];
				}
			}
		}
	}

	// /

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed, bool otherTransposed, bool otherContiguous>
	inline matrix<double> matrix<double, thisTransposed, thisContiguous>::operator/(const matrix<double, otherTransposed, otherContiguous>& other)
	{
#ifdef _DEBUG
		if (other._cols != this->_cols || other._rows != this->_rows)
		{
			std::cerr << "The dimensions of both matrices must be the same " << std::endl;
			std::cerr << "Matrix 1: " << std::endl;
			std::cerr << "Columns: " << this->_cols << std::endl;
			std::cerr << "Rows: " << this->_rows << std::endl;
			std::cerr << std::endl;
			std::cerr << "Matrix 2: " << std::endl;
			std::cerr << "Columns: " << other._cols << std::endl;
			std::cerr << "Rows: " << other._rows << std::endl;
			exit(1);
	}
#else
#endif

		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;
		double* data2 = other._data;

		size_t size = this->_size;

		size_t finalPosSize = this->finalPosSize;

		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix2ActualRows = other.actualRows;
		size_t matrix1ActualCols = this->actualCols;
		size_t matrix2ActualCols = other.actualCols;

		if constexpr (returnTransposed)
		{
			matrix<double> result(cols, rows);
			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (otherTransposed)
				{
					if constexpr (thisContiguous && otherContiguous)
					{
						for (size_t i = 0; i < finalPosSize; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);
							__m256d b = _mm256_load_pd(&data2[i]);

							_mm256_store_pd(&dataResult[i], _mm256_div_pd(a, b));
						}
						for (size_t i = finalPosSize; i < size; i++)
						{
							dataResult[i] = data1[i] / data2[i];
						}
					}
					else
					{
						for (size_t i = 0; i < finalPosRows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);
								__m256d b = _mm256_load_pd(&data2[j * matrix2ActualRows + i]);

								_mm256_store_pd(&dataResult[j * rows + i], _mm256_div_pd(a, b));
							}
						}
						for (size_t i = finalPosRows; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								dataResult[j * rows + i] = data1[j * matrix1ActualRows + i] / data2[j * matrix2ActualRows + i];
							}
						}
					}
				}
				else
				{
					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = data1[j * matrix1ActualRows + i] / data2[i * matrix2ActualCols + j];
						}
					}
				}
			}
			else
			{
				if constexpr (otherTransposed)
				{
					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = data1[i * matrix1ActualCols + j] / data2[j * matrix2ActualRows + i];
						}
					}
				}
				else
				{
					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < finalPosCols; j += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
							__m256d b = _mm256_load_pd(&data2[i * matrix2ActualCols + j]);

							__m256d div = _mm256_div_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(div, 1);
							__m128d val2 = _mm256_castpd256_pd128(div);

							_mm_store_sd(&dataResult[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

							_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
						}
						for (size_t j = finalPosCols; j < cols; j++)
						{
							dataResult[j * rows + i] = data1[i * matrix1ActualCols + j] / data2[i * matrix2ActualCols + j];
						}
					}
				}
			}
			return result;
		}
		else
		{
			matrix<double> result(rows, cols);
			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (otherTransposed)
				{
					for (size_t j = 0; j < cols; j++)
					{
						for (size_t i = 0; i < finalPosRows; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);
							__m256d b = _mm256_load_pd(&data2[j * matrix2ActualRows + i]);

							__m256d div = _mm256_div_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(div, 1);
							__m128d val2 = _mm256_castpd256_pd128(div);

							_mm_store_sd(&dataResult[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

							_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
						}
						for (size_t i = finalPosRows; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[j * matrix1ActualRows + i] / data2[j * matrix2ActualRows + i];
						}
					}
				}
				else
				{
					for (size_t j = 0; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[j * matrix1ActualRows + i] / data2[i * matrix2ActualCols + j];
						}
					}
				}
			}
			else
			{
				if constexpr (otherTransposed)
				{
					for (size_t j = 0; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] / data2[j * matrix2ActualRows + i];
						}
					}
				}
				else
				{
					if constexpr (thisContiguous && otherContiguous)
					{
						for (size_t i = 0; i < finalPosSize; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);
							__m256d b = _mm256_load_pd(&data2[i]);

							_mm256_store_pd(&dataResult[i], _mm256_div_pd(a, b));
						}
						for (size_t i = finalPosSize; i < size; i++)
						{
							dataResult[i] = data1[i] / data2[i];
						}
					}
					else
					{
						size_t matrix1ActualCols = this->actualCols;
						size_t matrix2ActualCols = other.actualCols;

						size_t finalPosCols = this->finalPosCols;
						size_t finalPosRows = this->finalPosRows;

						for (size_t j = 0; j < finalPosCols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
								__m256d b = _mm256_load_pd(&data2[i * matrix2ActualCols + j]);

								_mm256_store_pd(&dataResult[i * cols + j], _mm256_div_pd(a, b));
							}
						}
						for (size_t j = finalPosCols; j < cols; j++)
						{
							for (size_t i = 0; i < rows; i++)
							{
								dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] / data2[i * matrix2ActualCols + j];
							}
						}
					}
				}
			}
			return result;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool otherTransposed, bool otherContiguous>
	inline void matrix<double, thisTransposed, thisContiguous>::operator/=(const matrix<double, otherTransposed, otherContiguous>& other)
	{
#ifdef _DEBUG
		if (other._cols != this->_cols || other._rows != this->_rows)
		{
			std::cerr << "The dimensions of both matrices must be the same " << std::endl;
			std::cerr << "Matrix 1: " << std::endl;
			std::cerr << "Columns: " << this->_cols << std::endl;
			std::cerr << "Rows: " << this->_rows << std::endl;
			std::cerr << std::endl;
			std::cerr << "Matrix 2: " << std::endl;
			std::cerr << "Columns: " << other._cols << std::endl;
			std::cerr << "Rows: " << other._rows << std::endl;
			exit(1);
		}
#else
#endif

		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;
		double* data2 = other._data;

		if constexpr (thisTransposed)
		{
			if constexpr (otherTransposed)
			{
				if constexpr (thisContiguous && otherContiguous)
				{
					size_t size = this->_size;

					size_t finalPosSize = this->finalPosSize;

					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);
						__m256d b = _mm256_load_pd(&data2[i]);

						_mm256_store_pd(&data1[i], _mm256_div_pd(a, b));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						data1[i] /= data2[i];
					}
				}
				else
				{
					size_t finalPosRows = this->finalPosRows;
					size_t finalPosCols = this->finalPosCols;

					size_t matrix1ActualRows = this->actualRows;
					size_t matrix2ActualRows = other.actualRows;

					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);
							__m256d b = _mm256_load_pd(&data2[j * matrix2ActualRows + i]);

							_mm256_store_pd(&data1[j * matrix1ActualRows + i], _mm256_div_pd(a, b));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							data1[j * matrix1ActualRows + i] /= data2[j * matrix2ActualRows + i];
						}
					}
				}
			}
			else
			{
				size_t finalPosRows = this->finalPosRows;
				size_t finalPosCols = this->finalPosCols;

				size_t matrix1ActualRows = this->actualRows;
				size_t matrix2ActualCols = other.actualCols;

				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						data1[j * matrix1ActualRows + i] /= data2[i * matrix2ActualCols + j];
					}
				}
			}
		}
		else
		{
			if constexpr (otherTransposed)
			{
				size_t matrix1ActualCols = this->actualCols;
				size_t matrix2ActualRows = other.actualRows;

				size_t finalPosCols = this->finalPosCols;
				size_t finalPosRows = this->finalPosRows;

				for (size_t j = 0; j < cols; j++)
				{
					for (size_t i = 0; i < rows; i++)
					{
						data1[i * matrix1ActualCols + j] /= data2[j * matrix2ActualRows + i];
					}
				}
			}
			else
			{
				if constexpr (thisContiguous && otherContiguous)
				{
					size_t finalPosSize = this->finalPosSize;
					size_t size = this->_size;

					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);
						__m256d b = _mm256_load_pd(&data2[i]);

						_mm256_store_pd(&data1[i], _mm256_div_pd(a, b));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						data1[i] /= data2[i];
					}
				}
				else
				{
					size_t matrix1ActualCols = this->actualCols;
					size_t matrix2ActualCols = other.actualCols;

					size_t finalPosCols = this->finalPosCols;
					size_t finalPosRows = this->finalPosRows;

					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
							__m256d b = _mm256_load_pd(&data2[i * matrix2ActualCols + j]);

							_mm256_store_pd(&data1[i * matrix1ActualCols + j], _mm256_div_pd(a, b));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							data1[i * matrix1ActualCols + j] /= data2[i * matrix2ActualCols + j];
						}
					}
				}
			}
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed>
	inline matrix<double> matrix<double, thisTransposed, thisContiguous>::operator/(double num)
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		size_t size = this->_size;

		double* data1 = this->_data;

		__m256d b = _mm256_set1_pd(num);

		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;
		size_t finalPosSize = this->finalPosSize;

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix1ActualCols = this->actualCols;

		if constexpr (returnTransposed)
		{
			matrix<double> result(cols, rows);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&dataResult[i], _mm256_div_pd(a, b));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = data1[i] / num;
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

							_mm256_store_pd(&dataResult[j * rows + i], _mm256_div_pd(a, b));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = data1[j * matrix1ActualRows + i] / num;
						}
					}
				}
			}
			else
			{
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						__m256d div = _mm256_div_pd(a, b);

						__m128d val1 = _mm256_extractf128_pd(div, 1);
						__m128d val2 = _mm256_castpd256_pd128(div);

						_mm_store_sd(&dataResult[j * rows + i], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

						_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						dataResult[j * rows + i] = data1[i * matrix1ActualCols + j] / num;
					}
				}
			}
			return result;
		}
		else
		{
			matrix<double> result(rows, cols);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				for (size_t j = 0; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						__m256d div = _mm256_div_pd(a, b);

						__m128d val1 = _mm256_extractf128_pd(div, 1);
						__m128d val2 = _mm256_castpd256_pd128(div);

						_mm_store_sd(&dataResult[i * cols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

						_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						dataResult[i * cols + j] = data1[j * matrix1ActualRows + i] / num;
					}
				}
			}
			else
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&dataResult[i], _mm256_div_pd(a, b));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = data1[i] / num;
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

							_mm256_store_pd(&dataResult[i * cols + j], _mm256_div_pd(a, b));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] / num;
						}
					}
				}
			}
			return result;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline void matrix<double, thisTransposed, thisContiguous>::operator/=(double num)
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		__m256d b = _mm256_set1_pd(num);

		if constexpr (thisTransposed)
		{
			if constexpr (thisContiguous)
			{
				size_t size = this->_size;

				size_t finalPosSize = this->finalPosSize;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);

					_mm256_store_pd(&data1[i], _mm256_div_pd(a, b));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] /= num;
				}
			}
			else
			{
				size_t finalPosRows = this->finalPosRows;
				size_t finalPosCols = this->finalPosCols;

				size_t matrix1ActualRows = this->actualRows;

				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						_mm256_store_pd(&data1[j * matrix1ActualRows + i], _mm256_div_pd(a, b));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						data1[j * matrix1ActualRows + i] /= num;
					}
				}
			}
		}
		else
		{
			if constexpr (thisContiguous)
			{
				size_t finalPosSize = this->finalPosSize;
				size_t size = this->_size;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);

					_mm256_store_pd(&data1[i], _mm256_div_pd(a, b));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] /= num;
				}
			}
			else
			{
				size_t matrix1ActualCols = this->actualCols;

				size_t finalPosCols = this->finalPosCols;
				size_t finalPosRows = this->finalPosRows;

				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						_mm256_store_pd(&data1[i * matrix1ActualCols + j], _mm256_div_pd(a, b));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < rows; i++)
					{
						data1[i * matrix1ActualCols + j] /= num;
					}
				}
			}
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed>
	inline matrix<double> matrix<double, thisTransposed, thisContiguous>::operator/(const vector<double>& other)
	{
#ifdef _DEBUG
		if (this->_cols != other._size)
		{
			std::cerr << "The number of columns must match the number of elements in the vector " << this->_cols << " != " << other._size << std::endl;
			exit(1);
		}
#else
#endif
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		size_t actualRows = this->actualRows;
		size_t actualCols = this->actualCols;

		if constexpr (returnTransposed)
		{
			matrix<double> result(cols, rows);
			double* data1 = this->_data;
			double* data2 = other._data;

			double* dataResult = result._data;

			size_t finalPosCols = this->finalPosCols;
			size_t finalPosRows = this->finalPosRows;

			if constexpr (thisTransposed)
			{
				for (size_t j = 0; j < cols; j++)
				{
					__m256d b = _mm256_broadcast_sd(&data2[j]);
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[j * actualRows + i]);
						_mm256_store_pd(&dataResult[j * rows + i], _mm256_div_pd(a, b));
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						dataResult[j * rows + i] = data1[j * actualRows + i] / data2[j];
					}
				}
			}
			else
			{
				for (size_t j = 0; j < cols; j++)
				{
					for (size_t i = 0; i < rows; i++)
					{
						dataResult[j * rows + i] = data1[i * actualCols + j] / data2[j];
					}
				}
			}

			return result;
		}
		else
		{
			matrix<double> result(rows, cols);
			double* data1 = this->_data;
			double* data2 = other._data;

			double* dataResult = result._data;

			size_t finalPosCols = this->finalPosCols;
			size_t finalPosRows = this->finalPosRows;

			if constexpr (thisTransposed)
			{
				for (size_t j = 0; j < cols; j++)
				{
					__m256d b = _mm256_broadcast_sd(&data2[j]);
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[j * actualRows + i]);

						__m256d div = _mm256_div_pd(a, b);

						__m128d val1 = _mm256_extractf128_pd(div, 1);
						__m128d val2 = _mm256_castpd256_pd128(div);

						_mm_store_sd(&dataResult[i * cols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

						_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						dataResult[i * cols + j] = data1[j * actualRows + i] / data2[j];
					}
				}
			}
			else
			{
				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					__m256d b = _mm256_load_pd(&data2[j]);
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_load_pd(&data1[i * actualCols + j]);
						_mm256_store_pd(&dataResult[i * cols + j], _mm256_div_pd(a, b));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < rows; i++)
					{
						dataResult[i * cols + j] = data1[i * actualCols + j] / data2[j];
					}
				}
			}

			return result;
		}

	}

	template <bool thisTransposed, bool thisContiguous>
	inline void  matrix<double, thisTransposed, thisContiguous>::operator/=(const vector<double>& other)
	{
#ifdef _DEBUG
		if (this->_cols != other._size)
		{
			std::cerr << "The number of columns must match the number of elements in the vector " << this->_cols << " != " << other._size << std::endl;
			exit(1);
		}
#else
#endif
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;
		double* data2 = other._data;

		size_t finalPosCols = this->finalPosCols;
		size_t finalPosRows = this->finalPosRows;

		if constexpr (thisTransposed)
		{
			size_t actualRows = this->actualRows;

			for (size_t j = 0; j < cols; j++)
			{
				__m256d b = _mm256_broadcast_sd(&data2[j]);
				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[j * actualRows + i]);
					_mm256_store_pd(&data1[j * rows + i], _mm256_div_pd(a, b));
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					data1[j * rows + i] /= data2[j];
				}
			}
		}
		else
		{
			size_t actualCols = this->actualCols;

			for (size_t j = 0; j < finalPosCols; j += 4)
			{
				__m256d b = _mm256_load_pd(&data2[j]);
				for (size_t i = 0; i < rows; i++)
				{
					__m256d a = _mm256_load_pd(&data1[i * actualCols + j]);
					_mm256_store_pd(&data1[i * cols + j], _mm256_div_pd(a, b));
				}
			}
			for (size_t j = finalPosCols; j < cols; j++)
			{
				for (size_t i = 0; i < rows; i++)
				{
					data1[i * cols + j] /= data2[j];
				}
			}
		}
	}

	// ==

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed, bool otherTransposed, bool otherContiguous>
	inline matrix<uint8_t> matrix<double, thisTransposed, thisContiguous>::operator==(const matrix<double, otherTransposed, otherContiguous>& other)
	{
#ifdef _DEBUG
		if (other._cols != this->_cols || other._rows != this->_rows) throw std::invalid_argument("The dimensions of both matrices must be the same");
#else
#endif

		size_t rows = this->_rows;
		size_t cols = this->_cols;

		size_t size = this->_size;

		size_t finalPosSize = this->finalPosSize;

		double* data1 = this->_data;
		double* data2 = other._data;

		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix2ActualRows = other.actualRows;
		size_t matrix1ActualCols = this->actualCols;
		size_t matrix2ActualCols = other.actualCols;

		if constexpr (returnTransposed)
		{
			matrix<uint8_t> result(cols, rows);

			uint8_t* dataResult = result._data;
			if constexpr (thisTransposed)
			{
				if constexpr (otherTransposed)
				{
					if constexpr (thisContiguous && otherContiguous)
					{
						for (size_t i = 0; i < finalPosSize; i += 4)
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
						for (size_t i = finalPosSize; i < size; i++)
						{
							dataResult[i] = data1[i] == data2[i] ? True : False;
						}
					}
					else
					{
						for (size_t i = 0; i < finalPosRows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);
								__m256d b = _mm256_load_pd(&data2[j * matrix2ActualRows + i]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_EQ_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&dataResult[j * rows + i]), _mm_castsi128_ps(maskResult));

							}
						}
						for (size_t i = finalPosRows; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								dataResult[j * rows + i] = data1[j * matrix1ActualRows + i] == data2[j * matrix2ActualRows + i] ? True : False;
							}
						}
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);
							__m256d b = _mm256_setr_pd(data2[i * matrix2ActualCols + j],
								data2[(i + 1) * matrix2ActualCols + j],
								data2[(i + 2) * matrix2ActualCols + j],
								data2[(i + 3) * matrix2ActualCols + j]);

							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_EQ_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&dataResult[j * rows + i]), _mm_castsi128_ps(maskResult));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = data1[j * matrix1ActualRows + i] == data2[i * matrix2ActualCols + j] ? True : False;
						}
					}
				}
			}
			else
			{
				if constexpr (otherTransposed)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
							__m256d b = _mm256_setr_pd(data2[j * matrix2ActualRows + i],
								data2[(j + 1) * matrix2ActualRows + i],
								data2[(j + 2) * matrix2ActualRows + i],
								data2[(j + 3) * matrix2ActualRows + i]);
							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_EQ_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&dataResult[i * cols + j]), _mm_castsi128_ps(maskResult));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] == data2[j * matrix2ActualRows + i] ? True : False;
						}
					}
				}
				else
				{
					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = data1[i * matrix1ActualCols + j] == data2[i * matrix2ActualCols + j] ? True : False;
						}
					}
				}
			}
			return result;
		}
		else
		{
			matrix<uint8_t> result(rows, cols);

			uint8_t* dataResult = result._data;
			if constexpr (thisTransposed)
			{
				if constexpr (otherTransposed)
				{
					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[i * cols + j] = data1[j * matrix1ActualRows + i] == data2[j * matrix2ActualRows + i] ? True : False;
						}
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
								data1[(j + 1) * matrix1ActualRows + i],
								data1[(j + 2) * matrix1ActualRows + i],
								data1[(j + 3) * matrix1ActualRows + i]);
							__m256d b = _mm256_load_pd(&data2[i * matrix2ActualCols + j]);

							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_EQ_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&dataResult[i * cols + j]), _mm_castsi128_ps(maskResult));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[j * matrix1ActualRows + i] == data2[i * matrix2ActualCols + j] ? True : False;
						}
					}
				}
			}
			else
			{
				if constexpr (otherTransposed)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
							__m256d b = _mm256_setr_pd(data2[j * matrix2ActualRows + i],
								data2[(j + 1) * matrix2ActualRows + i],
								data2[(j + 2) * matrix2ActualRows + i],
								data2[(j + 3) * matrix2ActualRows + i]);
							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_EQ_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&dataResult[i * cols + j]), _mm_castsi128_ps(maskResult));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] == data2[j * matrix2ActualRows + i] ? True : False;
						}
					}
				}
				else
				{
					if constexpr (thisContiguous && otherContiguous)
					{
						for (size_t i = 0; i < finalPosSize; i += 4)
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
						for (size_t i = finalPosSize; i < size; i++)
						{
							dataResult[i] = data1[i] == data2[i] ? True : False;
						}
					}
					else
					{
						for (size_t j = 0; j < finalPosCols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
								__m256d b = _mm256_load_pd(&data2[i * matrix2ActualCols + j]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_EQ_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&dataResult[i * cols + j]), _mm_castsi128_ps(maskResult));
							}
						}
						for (size_t j = finalPosCols; j < cols; j++)
						{
							for (size_t i = 0; i < rows; i++)
							{
								dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] == data2[i * matrix2ActualCols + j] ? True : False;
							}
						}
					}
				}
			}
			return result;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed>
	inline matrix<uint8_t> matrix<double, thisTransposed, thisContiguous>::operator==(double num)
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;

		size_t size = this->_size;

		size_t finalPosSize = this->finalPosSize;

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix1ActualCols = this->actualCols;

		__m256d b = _mm256_set1_pd(num);

		if constexpr (returnTransposed)
		{
			matrix<uint8_t> result(cols, rows);

			uint8_t* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
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
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = data1[i] == num ? True : False;
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_EQ_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&dataResult[j * rows + i]), _mm_castsi128_ps(maskResult));

						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = data1[j * matrix1ActualRows + i] == num ? True : False;
						}
					}
				}
			}
			else
			{
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						dataResult[j * rows + i] = data1[i * matrix1ActualCols + j] == num ? True : False;
					}
				}
			}
			return result;
		}
		else
		{
			matrix<uint8_t> result(rows, cols);

			uint8_t* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						dataResult[i * cols + j] = data1[j * matrix1ActualRows + i] == num ? True : False;
					}
				}
			}
			else
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
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
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = data1[i] == num ? True : False;
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_EQ_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&dataResult[i * cols + j]), _mm_castsi128_ps(maskResult));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] == num ? True : False;
						}
					}
				}
			}
			return result;
		}
	}

	// !=

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed, bool otherTransposed, bool otherContiguous>
	inline matrix<uint8_t> matrix<double, thisTransposed, thisContiguous>::operator!=(const matrix<double, otherTransposed, otherContiguous>& other)
	{
#ifdef _DEBUG
		if (other._cols != this->_cols || other._rows != this->_rows) throw std::invalid_argument("The dimensions of both matrices must be the same");
#else
#endif

		size_t rows = this->_rows;
		size_t cols = this->_cols;

		size_t size = this->_size;

		size_t finalPosSize = this->finalPosSize;

		double* data1 = this->_data;
		double* data2 = other._data;

		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix2ActualRows = other.actualRows;
		size_t matrix1ActualCols = this->actualCols;
		size_t matrix2ActualCols = other.actualCols;

		if constexpr (returnTransposed)
		{
			matrix<uint8_t> result(cols, rows);

			uint8_t* dataResult = result._data;
			if constexpr (thisTransposed)
			{
				if constexpr (otherTransposed)
				{
					if constexpr (thisContiguous && otherContiguous)
					{
						for (size_t i = 0; i < finalPosSize; i += 4)
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
						for (size_t i = finalPosSize; i < size; i++)
						{
							dataResult[i] = data1[i] != data2[i] ? True : False;
						}
					}
					else
					{
						for (size_t i = 0; i < finalPosRows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);
								__m256d b = _mm256_load_pd(&data2[j * matrix2ActualRows + i]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_NEQ_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&dataResult[j * rows + i]), _mm_castsi128_ps(maskResult));

							}
						}
						for (size_t i = finalPosRows; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								dataResult[j * rows + i] = data1[j * matrix1ActualRows + i] != data2[j * matrix2ActualRows + i] ? True : False;
							}
						}
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);
							__m256d b = _mm256_setr_pd(data2[i * matrix2ActualCols + j],
								data2[(i + 1) * matrix2ActualCols + j],
								data2[(i + 2) * matrix2ActualCols + j],
								data2[(i + 3) * matrix2ActualCols + j]);

							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_NEQ_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&dataResult[j * rows + i]), _mm_castsi128_ps(maskResult));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = data1[j * matrix1ActualRows + i] != data2[i * matrix2ActualCols + j] ? True : False;
						}
					}
				}
			}
			else
			{
				if constexpr (otherTransposed)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
							__m256d b = _mm256_setr_pd(data2[j * matrix2ActualRows + i],
								data2[(j + 1) * matrix2ActualRows + i],
								data2[(j + 2) * matrix2ActualRows + i],
								data2[(j + 3) * matrix2ActualRows + i]);
							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_NEQ_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&dataResult[i * cols + j]), _mm_castsi128_ps(maskResult));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] != data2[j * matrix2ActualRows + i] ? True : False;
						}
					}
				}
				else
				{
					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = data1[i * matrix1ActualCols + j] != data2[i * matrix2ActualCols + j] ? True : False;
						}
					}
				}
			}
			return result;
		}
		else
		{
			matrix<uint8_t> result(rows, cols);

			uint8_t* dataResult = result._data;
			if constexpr (thisTransposed)
			{
				if constexpr (otherTransposed)
				{
					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[i * cols + j] = data1[j * matrix1ActualRows + i] != data2[j * matrix2ActualRows + i] ? True : False;
						}
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
								data1[(j + 1) * matrix1ActualRows + i],
								data1[(j + 2) * matrix1ActualRows + i],
								data1[(j + 3) * matrix1ActualRows + i]);
							__m256d b = _mm256_load_pd(&data2[i * matrix2ActualCols + j]);

							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_NEQ_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&dataResult[i * cols + j]), _mm_castsi128_ps(maskResult));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[j * matrix1ActualRows + i] != data2[i * matrix2ActualCols + j] ? True : False;
						}
					}
				}
			}
			else
			{
				if constexpr (otherTransposed)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
							__m256d b = _mm256_setr_pd(data2[j * matrix2ActualRows + i],
								data2[(j + 1) * matrix2ActualRows + i],
								data2[(j + 2) * matrix2ActualRows + i],
								data2[(j + 3) * matrix2ActualRows + i]);
							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_NEQ_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&dataResult[i * cols + j]), _mm_castsi128_ps(maskResult));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] != data2[j * matrix2ActualRows + i] ? True : False;
						}
					}
				}
				else
				{
					if constexpr (thisContiguous && otherContiguous)
					{
						for (size_t i = 0; i < finalPosSize; i += 4)
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
						for (size_t i = finalPosSize; i < size; i++)
						{
							dataResult[i] = data1[i] != data2[i] ? True : False;
						}
					}
					else
					{
						for (size_t j = 0; j < finalPosCols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
								__m256d b = _mm256_load_pd(&data2[i * matrix2ActualCols + j]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_NEQ_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&dataResult[i * cols + j]), _mm_castsi128_ps(maskResult));
							}
						}
						for (size_t j = finalPosCols; j < cols; j++)
						{
							for (size_t i = 0; i < rows; i++)
							{
								dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] != data2[i * matrix2ActualCols + j] ? True : False;
							}
						}
					}
				}
			}
			return result;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed>
	inline matrix<uint8_t> matrix<double, thisTransposed, thisContiguous>::operator!=(double num)
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;

		size_t size = this->_size;

		size_t finalPosSize = this->finalPosSize;

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix1ActualCols = this->actualCols;

		__m256d b = _mm256_set1_pd(num);

		if constexpr (returnTransposed)
		{
			matrix<uint8_t> result(cols, rows);

			uint8_t* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
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
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = data1[i] != num ? True : False;
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_NEQ_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&dataResult[j * rows + i]), _mm_castsi128_ps(maskResult));

						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = data1[j * matrix1ActualRows + i] != num ? True : False;
						}
					}
				}
			}
			else
			{
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						dataResult[j * rows + i] = data1[i * matrix1ActualCols + j] != num ? True : False;
					}
				}
			}
			return result;
		}
		else
		{
			matrix<uint8_t> result(rows, cols);

			uint8_t* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						dataResult[i * cols + j] = data1[j * matrix1ActualRows + i] != num ? True : False;
					}
				}
			}
			else
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
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
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = data1[i] != num ? True : False;
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_NEQ_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&dataResult[i * cols + j]), _mm_castsi128_ps(maskResult));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] != num ? True : False;
						}
					}
				}
			}
			return result;
		}
	}

	// >

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed, bool otherTransposed, bool otherContiguous>
	inline matrix<uint8_t> matrix<double, thisTransposed, thisContiguous>::operator>(const matrix<double, otherTransposed, otherContiguous>& other)
	{
#ifdef _DEBUG
		if (other._cols > this->_cols || other._rows > this->_rows) throw std::invalid_argument("The dimensions of both matrices must be the same");
#else
#endif

		size_t rows = this->_rows;
		size_t cols = this->_cols;

		size_t size = this->_size;

		size_t finalPosSize = this->finalPosSize;

		double* data1 = this->_data;
		double* data2 = other._data;

		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix2ActualRows = other.actualRows;
		size_t matrix1ActualCols = this->actualCols;
		size_t matrix2ActualCols = other.actualCols;

		if constexpr (returnTransposed)
		{
			matrix<uint8_t> result(cols, rows);

			uint8_t* dataResult = result._data;
			if constexpr (thisTransposed)
			{
				if constexpr (otherTransposed)
				{
					if constexpr (thisContiguous && otherContiguous)
					{
						for (size_t i = 0; i < finalPosSize; i += 4)
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
						for (size_t i = finalPosSize; i < size; i++)
						{
							dataResult[i] = data1[i] > data2[i] ? True : False;
						}
					}
					else
					{
						for (size_t i = 0; i < finalPosRows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);
								__m256d b = _mm256_load_pd(&data2[j * matrix2ActualRows + i]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GT_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&dataResult[j * rows + i]), _mm_castsi128_ps(maskResult));

							}
						}
						for (size_t i = finalPosRows; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								dataResult[j * rows + i] = data1[j * matrix1ActualRows + i] > data2[j * matrix2ActualRows + i] ? True : False;
							}
						}
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);
							__m256d b = _mm256_setr_pd(data2[i * matrix2ActualCols + j],
								data2[(i + 1) * matrix2ActualCols + j],
								data2[(i + 2) * matrix2ActualCols + j],
								data2[(i + 3) * matrix2ActualCols + j]);

							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GT_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&dataResult[j * rows + i]), _mm_castsi128_ps(maskResult));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = data1[j * matrix1ActualRows + i] > data2[i * matrix2ActualCols + j] ? True : False;
						}
					}
				}
			}
			else
			{
				if constexpr (otherTransposed)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
							__m256d b = _mm256_setr_pd(data2[j * matrix2ActualRows + i],
								data2[(j + 1) * matrix2ActualRows + i],
								data2[(j + 2) * matrix2ActualRows + i],
								data2[(j + 3) * matrix2ActualRows + i]);
							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GT_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&dataResult[i * cols + j]), _mm_castsi128_ps(maskResult));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] > data2[j * matrix2ActualRows + i] ? True : False;
						}
					}
				}
				else
				{
					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = data1[i * matrix1ActualCols + j] > data2[i * matrix2ActualCols + j] ? True : False;
						}
					}
				}
			}
			return result;
		}
		else
		{
			matrix<uint8_t> result(rows, cols);

			uint8_t* dataResult = result._data;
			if constexpr (thisTransposed)
			{
				if constexpr (otherTransposed)
				{
					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[i * cols + j] = data1[j * matrix1ActualRows + i] > data2[j * matrix2ActualRows + i] ? True : False;
						}
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
								data1[(j + 1) * matrix1ActualRows + i],
								data1[(j + 2) * matrix1ActualRows + i],
								data1[(j + 3) * matrix1ActualRows + i]);
							__m256d b = _mm256_load_pd(&data2[i * matrix2ActualCols + j]);

							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GT_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&dataResult[i * cols + j]), _mm_castsi128_ps(maskResult));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[j * matrix1ActualRows + i] > data2[i * matrix2ActualCols + j] ? True : False;
						}
					}
				}
			}
			else
			{
				if constexpr (otherTransposed)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
							__m256d b = _mm256_setr_pd(data2[j * matrix2ActualRows + i],
								data2[(j + 1) * matrix2ActualRows + i],
								data2[(j + 2) * matrix2ActualRows + i],
								data2[(j + 3) * matrix2ActualRows + i]);
							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GT_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&dataResult[i * cols + j]), _mm_castsi128_ps(maskResult));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] > data2[j * matrix2ActualRows + i] ? True : False;
						}
					}
				}
				else
				{
					if constexpr (thisContiguous && otherContiguous)
					{
						for (size_t i = 0; i < finalPosSize; i += 4)
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
						for (size_t i = finalPosSize; i < size; i++)
						{
							dataResult[i] = data1[i] > data2[i] ? True : False;
						}
					}
					else
					{
						for (size_t j = 0; j < finalPosCols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
								__m256d b = _mm256_load_pd(&data2[i * matrix2ActualCols + j]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GT_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&dataResult[i * cols + j]), _mm_castsi128_ps(maskResult));
							}
						}
						for (size_t j = finalPosCols; j < cols; j++)
						{
							for (size_t i = 0; i < rows; i++)
							{
								dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] > data2[i * matrix2ActualCols + j] ? True : False;
							}
						}
					}
				}
			}
			return result;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed>
	inline matrix<uint8_t> matrix<double, thisTransposed, thisContiguous>::operator>(double num)
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;

		size_t size = this->_size;

		size_t finalPosSize = this->finalPosSize;

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix1ActualCols = this->actualCols;

		__m256d b = _mm256_set1_pd(num);

		if constexpr (returnTransposed)
		{
			matrix<uint8_t> result(cols, rows);

			uint8_t* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
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
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = data1[i] > num ? True : False;
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GT_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&dataResult[j * rows + i]), _mm_castsi128_ps(maskResult));

						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = data1[j * matrix1ActualRows + i] > num ? True : False;
						}
					}
				}
			}
			else
			{
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						dataResult[j * rows + i] = data1[i * matrix1ActualCols + j] > num ? True : False;
					}
				}
			}
			return result;
		}
		else
		{
			matrix<uint8_t> result(rows, cols);

			uint8_t* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						dataResult[i * cols + j] = data1[j * matrix1ActualRows + i] > num ? True : False;
					}
				}
			}
			else
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
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
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = data1[i] > num ? True : False;
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GT_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&dataResult[i * cols + j]), _mm_castsi128_ps(maskResult));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] > num ? True : False;
						}
					}
				}
			}
			return result;
		}
	}

	// <

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed, bool otherTransposed, bool otherContiguous>
	inline matrix<uint8_t> matrix<double, thisTransposed, thisContiguous>::operator<(const matrix<double, otherTransposed, otherContiguous>& other)
	{
#ifdef _DEBUG
		if (other._cols < this->_cols || other._rows < this->_rows) throw std::invalid_argument("The dimensions of both matrices must be the same");
#else
#endif

		size_t rows = this->_rows;
		size_t cols = this->_cols;

		size_t size = this->_size;

		size_t finalPosSize = this->finalPosSize;

		double* data1 = this->_data;
		double* data2 = other._data;

		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix2ActualRows = other.actualRows;
		size_t matrix1ActualCols = this->actualCols;
		size_t matrix2ActualCols = other.actualCols;

		if constexpr (returnTransposed)
		{
			matrix<uint8_t> result(cols, rows);

			uint8_t* dataResult = result._data;
			if constexpr (thisTransposed)
			{
				if constexpr (otherTransposed)
				{
					if constexpr (thisContiguous && otherContiguous)
					{
						for (size_t i = 0; i < finalPosSize; i += 4)
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
						for (size_t i = finalPosSize; i < size; i++)
						{
							dataResult[i] = data1[i] < data2[i] ? True : False;
						}
					}
					else
					{
						for (size_t i = 0; i < finalPosRows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);
								__m256d b = _mm256_load_pd(&data2[j * matrix2ActualRows + i]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LT_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&dataResult[j * rows + i]), _mm_castsi128_ps(maskResult));

							}
						}
						for (size_t i = finalPosRows; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								dataResult[j * rows + i] = data1[j * matrix1ActualRows + i] < data2[j * matrix2ActualRows + i] ? True : False;
							}
						}
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);
							__m256d b = _mm256_setr_pd(data2[i * matrix2ActualCols + j],
								data2[(i + 1) * matrix2ActualCols + j],
								data2[(i + 2) * matrix2ActualCols + j],
								data2[(i + 3) * matrix2ActualCols + j]);

							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LT_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&dataResult[j * rows + i]), _mm_castsi128_ps(maskResult));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = data1[j * matrix1ActualRows + i] < data2[i * matrix2ActualCols + j] ? True : False;
						}
					}
				}
			}
			else
			{
				if constexpr (otherTransposed)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
							__m256d b = _mm256_setr_pd(data2[j * matrix2ActualRows + i],
								data2[(j + 1) * matrix2ActualRows + i],
								data2[(j + 2) * matrix2ActualRows + i],
								data2[(j + 3) * matrix2ActualRows + i]);
							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LT_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&dataResult[i * cols + j]), _mm_castsi128_ps(maskResult));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] < data2[j * matrix2ActualRows + i] ? True : False;
						}
					}
				}
				else
				{
					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = data1[i * matrix1ActualCols + j] < data2[i * matrix2ActualCols + j] ? True : False;
						}
					}
				}
			}
			return result;
		}
		else
		{
			matrix<uint8_t> result(rows, cols);

			uint8_t* dataResult = result._data;
			if constexpr (thisTransposed)
			{
				if constexpr (otherTransposed)
				{
					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[i * cols + j] = data1[j * matrix1ActualRows + i] < data2[j * matrix2ActualRows + i] ? True : False;
						}
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
								data1[(j + 1) * matrix1ActualRows + i],
								data1[(j + 2) * matrix1ActualRows + i],
								data1[(j + 3) * matrix1ActualRows + i]);
							__m256d b = _mm256_load_pd(&data2[i * matrix2ActualCols + j]);

							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LT_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&dataResult[i * cols + j]), _mm_castsi128_ps(maskResult));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[j * matrix1ActualRows + i] < data2[i * matrix2ActualCols + j] ? True : False;
						}
					}
				}
			}
			else
			{
				if constexpr (otherTransposed)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
							__m256d b = _mm256_setr_pd(data2[j * matrix2ActualRows + i],
								data2[(j + 1) * matrix2ActualRows + i],
								data2[(j + 2) * matrix2ActualRows + i],
								data2[(j + 3) * matrix2ActualRows + i]);
							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LT_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&dataResult[i * cols + j]), _mm_castsi128_ps(maskResult));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] < data2[j * matrix2ActualRows + i] ? True : False;
						}
					}
				}
				else
				{
					if constexpr (thisContiguous && otherContiguous)
					{
						for (size_t i = 0; i < finalPosSize; i += 4)
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
						for (size_t i = finalPosSize; i < size; i++)
						{
							dataResult[i] = data1[i] < data2[i] ? True : False;
						}
					}
					else
					{
						for (size_t j = 0; j < finalPosCols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
								__m256d b = _mm256_load_pd(&data2[i * matrix2ActualCols + j]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LT_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&dataResult[i * cols + j]), _mm_castsi128_ps(maskResult));
							}
						}
						for (size_t j = finalPosCols; j < cols; j++)
						{
							for (size_t i = 0; i < rows; i++)
							{
								dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] < data2[i * matrix2ActualCols + j] ? True : False;
							}
						}
					}
				}
			}
			return result;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed>
	inline matrix<uint8_t> matrix<double, thisTransposed, thisContiguous>::operator<(double num)
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;

		size_t size = this->_size;

		size_t finalPosSize = this->finalPosSize;

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix1ActualCols = this->actualCols;

		__m256d b = _mm256_set1_pd(num);

		if constexpr (returnTransposed)
		{
			matrix<uint8_t> result(cols, rows);

			uint8_t* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
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
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = data1[i] < num ? True : False;
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LT_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&dataResult[j * rows + i]), _mm_castsi128_ps(maskResult));

						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = data1[j * matrix1ActualRows + i] < num ? True : False;
						}
					}
				}
			}
			else
			{
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						dataResult[j * rows + i] = data1[i * matrix1ActualCols + j] < num ? True : False;
					}
				}
			}
			return result;
		}
		else
		{
			matrix<uint8_t> result(rows, cols);

			uint8_t* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						dataResult[i * cols + j] = data1[j * matrix1ActualRows + i] < num ? True : False;
					}
				}
			}
			else
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
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
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = data1[i] < num ? True : False;
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LT_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&dataResult[i * cols + j]), _mm_castsi128_ps(maskResult));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] < num ? True : False;
						}
					}
				}
			}
			return result;
		}
	}

	// >=

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed, bool otherTransposed, bool otherContiguous>
	inline matrix<uint8_t> matrix<double, thisTransposed, thisContiguous>::operator>=(const matrix<double, otherTransposed, otherContiguous>& other)
	{
#ifdef _DEBUG
		if (other._cols >= this->_cols || other._rows >= this->_rows) throw std::invalid_argument("The dimensions of both matrices must be the same");
#else
#endif

		size_t rows = this->_rows;
		size_t cols = this->_cols;

		size_t size = this->_size;

		size_t finalPosSize = this->finalPosSize;

		double* data1 = this->_data;
		double* data2 = other._data;

		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix2ActualRows = other.actualRows;
		size_t matrix1ActualCols = this->actualCols;
		size_t matrix2ActualCols = other.actualCols;

		if constexpr (returnTransposed)
		{
			matrix<uint8_t> result(cols, rows);

			uint8_t* dataResult = result._data;
			if constexpr (thisTransposed)
			{
				if constexpr (otherTransposed)
				{
					if constexpr (thisContiguous && otherContiguous)
					{
						for (size_t i = 0; i < finalPosSize; i += 4)
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
						for (size_t i = finalPosSize; i < size; i++)
						{
							dataResult[i] = data1[i] >= data2[i] ? True : False;
						}
					}
					else
					{
						for (size_t i = 0; i < finalPosRows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);
								__m256d b = _mm256_load_pd(&data2[j * matrix2ActualRows + i]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GE_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&dataResult[j * rows + i]), _mm_castsi128_ps(maskResult));

							}
						}
						for (size_t i = finalPosRows; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								dataResult[j * rows + i] = data1[j * matrix1ActualRows + i] >= data2[j * matrix2ActualRows + i] ? True : False;
							}
						}
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);
							__m256d b = _mm256_setr_pd(data2[i * matrix2ActualCols + j],
								data2[(i + 1) * matrix2ActualCols + j],
								data2[(i + 2) * matrix2ActualCols + j],
								data2[(i + 3) * matrix2ActualCols + j]);

							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GE_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&dataResult[j * rows + i]), _mm_castsi128_ps(maskResult));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = data1[j * matrix1ActualRows + i] >= data2[i * matrix2ActualCols + j] ? True : False;
						}
					}
				}
			}
			else
			{
				if constexpr (otherTransposed)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
							__m256d b = _mm256_setr_pd(data2[j * matrix2ActualRows + i],
								data2[(j + 1) * matrix2ActualRows + i],
								data2[(j + 2) * matrix2ActualRows + i],
								data2[(j + 3) * matrix2ActualRows + i]);
							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GE_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&dataResult[i * cols + j]), _mm_castsi128_ps(maskResult));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] >= data2[j * matrix2ActualRows + i] ? True : False;
						}
					}
				}
				else
				{
					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = data1[i * matrix1ActualCols + j] >= data2[i * matrix2ActualCols + j] ? True : False;
						}
					}
				}
			}
			return result;
		}
		else
		{
			matrix<uint8_t> result(rows, cols);

			uint8_t* dataResult = result._data;
			if constexpr (thisTransposed)
			{
				if constexpr (otherTransposed)
				{
					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[i * cols + j] = data1[j * matrix1ActualRows + i] >= data2[j * matrix2ActualRows + i] ? True : False;
						}
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
								data1[(j + 1) * matrix1ActualRows + i],
								data1[(j + 2) * matrix1ActualRows + i],
								data1[(j + 3) * matrix1ActualRows + i]);
							__m256d b = _mm256_load_pd(&data2[i * matrix2ActualCols + j]);

							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GE_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&dataResult[i * cols + j]), _mm_castsi128_ps(maskResult));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[j * matrix1ActualRows + i] >= data2[i * matrix2ActualCols + j] ? True : False;
						}
					}
				}
			}
			else
			{
				if constexpr (otherTransposed)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
							__m256d b = _mm256_setr_pd(data2[j * matrix2ActualRows + i],
								data2[(j + 1) * matrix2ActualRows + i],
								data2[(j + 2) * matrix2ActualRows + i],
								data2[(j + 3) * matrix2ActualRows + i]);
							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GE_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&dataResult[i * cols + j]), _mm_castsi128_ps(maskResult));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] >= data2[j * matrix2ActualRows + i] ? True : False;
						}
					}
				}
				else
				{
					if constexpr (thisContiguous && otherContiguous)
					{
						for (size_t i = 0; i < finalPosSize; i += 4)
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
						for (size_t i = finalPosSize; i < size; i++)
						{
							dataResult[i] = data1[i] >= data2[i] ? True : False;
						}
					}
					else
					{
						for (size_t j = 0; j < finalPosCols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
								__m256d b = _mm256_load_pd(&data2[i * matrix2ActualCols + j]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GE_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&dataResult[i * cols + j]), _mm_castsi128_ps(maskResult));
							}
						}
						for (size_t j = finalPosCols; j < cols; j++)
						{
							for (size_t i = 0; i < rows; i++)
							{
								dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] >= data2[i * matrix2ActualCols + j] ? True : False;
							}
						}
					}
				}
			}
			return result;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed>
	inline matrix<uint8_t> matrix<double, thisTransposed, thisContiguous>::operator>=(double num)
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;

		size_t size = this->_size;

		size_t finalPosSize = this->finalPosSize;

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix1ActualCols = this->actualCols;

		__m256d b = _mm256_set1_pd(num);

		if constexpr (returnTransposed)
		{
			matrix<uint8_t> result(cols, rows);

			uint8_t* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
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
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = data1[i] >= num ? True : False;
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GE_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&dataResult[j * rows + i]), _mm_castsi128_ps(maskResult));

						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = data1[j * matrix1ActualRows + i] >= num ? True : False;
						}
					}
				}
			}
			else
			{
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						dataResult[j * rows + i] = data1[i * matrix1ActualCols + j] >= num ? True : False;
					}
				}
			}
			return result;
		}
		else
		{
			matrix<uint8_t> result(rows, cols);

			uint8_t* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						dataResult[i * cols + j] = data1[j * matrix1ActualRows + i] >= num ? True : False;
					}
				}
			}
			else
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
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
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = data1[i] >= num ? True : False;
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GE_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&dataResult[i * cols + j]), _mm_castsi128_ps(maskResult));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] >= num ? True : False;
						}
					}
				}
			}
			return result;
		}
	}

	// <=

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed, bool otherTransposed, bool otherContiguous>
	inline matrix<uint8_t> matrix<double, thisTransposed, thisContiguous>::operator<=(const matrix<double, otherTransposed, otherContiguous>& other)
	{
#ifdef _DEBUG
		if (other._cols <= this->_cols || other._rows <= this->_rows) throw std::invalid_argument("The dimensions of both matrices must be the same");
#else
#endif

		size_t rows = this->_rows;
		size_t cols = this->_cols;

		size_t size = this->_size;

		size_t finalPosSize = this->finalPosSize;

		double* data1 = this->_data;
		double* data2 = other._data;

		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix2ActualRows = other.actualRows;
		size_t matrix1ActualCols = this->actualCols;
		size_t matrix2ActualCols = other.actualCols;

		if constexpr (returnTransposed)
		{
			matrix<uint8_t> result(cols, rows);

			uint8_t* dataResult = result._data;
			if constexpr (thisTransposed)
			{
				if constexpr (otherTransposed)
				{
					if constexpr (thisContiguous && otherContiguous)
					{
						for (size_t i = 0; i < finalPosSize; i += 4)
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
						for (size_t i = finalPosSize; i < size; i++)
						{
							dataResult[i] = data1[i] <= data2[i] ? True : False;
						}
					}
					else
					{
						for (size_t i = 0; i < finalPosRows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);
								__m256d b = _mm256_load_pd(&data2[j * matrix2ActualRows + i]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LE_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&dataResult[j * rows + i]), _mm_castsi128_ps(maskResult));

							}
						}
						for (size_t i = finalPosRows; i < rows; i++)
						{
							for (size_t j = 0; j < cols; j++)
							{
								dataResult[j * rows + i] = data1[j * matrix1ActualRows + i] <= data2[j * matrix2ActualRows + i] ? True : False;
							}
						}
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);
							__m256d b = _mm256_setr_pd(data2[i * matrix2ActualCols + j],
								data2[(i + 1) * matrix2ActualCols + j],
								data2[(i + 2) * matrix2ActualCols + j],
								data2[(i + 3) * matrix2ActualCols + j]);

							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LE_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&dataResult[j * rows + i]), _mm_castsi128_ps(maskResult));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = data1[j * matrix1ActualRows + i] <= data2[i * matrix2ActualCols + j] ? True : False;
						}
					}
				}
			}
			else
			{
				if constexpr (otherTransposed)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
							__m256d b = _mm256_setr_pd(data2[j * matrix2ActualRows + i],
								data2[(j + 1) * matrix2ActualRows + i],
								data2[(j + 2) * matrix2ActualRows + i],
								data2[(j + 3) * matrix2ActualRows + i]);
							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LE_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&dataResult[i * cols + j]), _mm_castsi128_ps(maskResult));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] <= data2[j * matrix2ActualRows + i] ? True : False;
						}
					}
				}
				else
				{
					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = data1[i * matrix1ActualCols + j] <= data2[i * matrix2ActualCols + j] ? True : False;
						}
					}
				}
			}
			return result;
		}
		else
		{
			matrix<uint8_t> result(rows, cols);

			uint8_t* dataResult = result._data;
			if constexpr (thisTransposed)
			{
				if constexpr (otherTransposed)
				{
					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[i * cols + j] = data1[j * matrix1ActualRows + i] <= data2[j * matrix2ActualRows + i] ? True : False;
						}
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
								data1[(j + 1) * matrix1ActualRows + i],
								data1[(j + 2) * matrix1ActualRows + i],
								data1[(j + 3) * matrix1ActualRows + i]);
							__m256d b = _mm256_load_pd(&data2[i * matrix2ActualCols + j]);

							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LE_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&dataResult[i * cols + j]), _mm_castsi128_ps(maskResult));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[j * matrix1ActualRows + i] <= data2[i * matrix2ActualCols + j] ? True : False;
						}
					}
				}
			}
			else
			{
				if constexpr (otherTransposed)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
							__m256d b = _mm256_setr_pd(data2[j * matrix2ActualRows + i],
								data2[(j + 1) * matrix2ActualRows + i],
								data2[(j + 2) * matrix2ActualRows + i],
								data2[(j + 3) * matrix2ActualRows + i]);
							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LE_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&dataResult[i * cols + j]), _mm_castsi128_ps(maskResult));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] <= data2[j * matrix2ActualRows + i] ? True : False;
						}
					}
				}
				else
				{
					if constexpr (thisContiguous && otherContiguous)
					{
						for (size_t i = 0; i < finalPosSize; i += 4)
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
						for (size_t i = finalPosSize; i < size; i++)
						{
							dataResult[i] = data1[i] <= data2[i] ? True : False;
						}
					}
					else
					{
						for (size_t j = 0; j < finalPosCols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
								__m256d b = _mm256_load_pd(&data2[i * matrix2ActualCols + j]);

								__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LE_OQ));

								__m128i mask1 = _mm256_castsi256_si128(mask);
								__m128i mask2 = _mm256_extracti128_si256(mask, 1);

								mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
								mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

								__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

								_mm_store_ss(reinterpret_cast<float*>(&dataResult[i * cols + j]), _mm_castsi128_ps(maskResult));
							}
						}
						for (size_t j = finalPosCols; j < cols; j++)
						{
							for (size_t i = 0; i < rows; i++)
							{
								dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] <= data2[i * matrix2ActualCols + j] ? True : False;
							}
						}
					}
				}
			}
			return result;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed>
	inline matrix<uint8_t> matrix<double, thisTransposed, thisContiguous>::operator<=(double num)
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;

		size_t size = this->_size;

		size_t finalPosSize = this->finalPosSize;

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix1ActualCols = this->actualCols;

		__m256d b = _mm256_set1_pd(num);

		if constexpr (returnTransposed)
		{
			matrix<uint8_t> result(cols, rows);

			uint8_t* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
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
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = data1[i] <= num ? True : False;
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LE_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&dataResult[j * rows + i]), _mm_castsi128_ps(maskResult));

						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = data1[j * matrix1ActualRows + i] <= num ? True : False;
						}
					}
				}
			}
			else
			{
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						dataResult[j * rows + i] = data1[i * matrix1ActualCols + j] <= num ? True : False;
					}
				}
			}
			return result;
		}
		else
		{
			matrix<uint8_t> result(rows, cols);

			uint8_t* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						dataResult[i * cols + j] = data1[j * matrix1ActualRows + i] <= num ? True : False;
					}
				}
			}
			else
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
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
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = data1[i] <= num ? True : False;
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

							__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LE_OQ));

							__m128i mask1 = _mm256_castsi256_si128(mask);
							__m128i mask2 = _mm256_extracti128_si256(mask, 1);

							mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
							mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

							__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

							_mm_store_ss(reinterpret_cast<float*>(&dataResult[i * cols + j]), _mm_castsi128_ps(maskResult));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] <= num ? True : False;
						}
					}
				}
			}
			return result;
		}
	}

	// Functions

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed>
	inline matrix<double> matrix<double, thisTransposed, thisContiguous>::exp()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		size_t finalPosSize = this->finalPosSize;
		size_t size = this->_size;

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix1ActualCols = this->actualCols;

		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;

		if constexpr (returnTransposed)
		{
			matrix<double> result(cols, rows);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&dataResult[i], _mm256_exp_pd(a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = std::exp(data1[i]);
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

							_mm256_store_pd(&dataResult[j * rows + i], _mm256_exp_pd(a));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < finalPosCols; j += 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
								data1[(j + 1) * matrix1ActualRows + i],
								data1[(j + 2) * matrix1ActualRows + i],
								data1[(j + 3) * matrix1ActualRows + i]);

							__m256d exp = _mm256_exp_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&dataResult[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

							_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
						}
						for (size_t j = finalPosCols; j < cols; j++)
						{
							dataResult[j * rows + i] = std::exp(data1[j * matrix1ActualRows + i]);
						}
					}
				}
			}
			else
			{
				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
							data1[(i + 1) * matrix1ActualCols + j],
							data1[(i + 2) * matrix1ActualCols + j],
							data1[(i + 3) * matrix1ActualCols + j]);

						_mm256_store_pd(&dataResult[j * rows + i], _mm256_exp_pd(a));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						__m256d exp = _mm256_exp_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&dataResult[j * rows + i], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

						_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						dataResult[j * rows + i] = std::exp(data1[i * matrix1ActualCols + j]);
					}
				}
			}
			return result;
		}
		else
		{
			matrix<double> result(rows, cols);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
							data1[(j + 1) * matrix1ActualRows + i],
							data1[(j + 2) * matrix1ActualRows + i],
							data1[(j + 3) * matrix1ActualRows + i]);
						_mm256_store_pd(&dataResult[i * cols + j], _mm256_exp_pd(a));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						__m256d exp = _mm256_exp_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&dataResult[i * cols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

						_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						dataResult[i * cols + j] = std::exp(data1[j * matrix1ActualRows + i]);
					}
				}
			}
			else
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);
						_mm256_store_pd(&dataResult[i], _mm256_exp_pd(a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = std::exp(data1[i]);
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

							_mm256_store_pd(&dataResult[i * cols + j], _mm256_exp_pd(a));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < finalPosRows; i += 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
								data1[(i + 1) * matrix1ActualCols + j],
								data1[(i + 2) * matrix1ActualCols + j],
								data1[(i + 3) * matrix1ActualCols + j]);

							__m256d exp = _mm256_exp_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&dataResult[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

							_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
						}
						for (size_t i = finalPosRows; i < rows; i++)
						{
							dataResult[i * cols + j] = std::exp(data1[i * matrix1ActualCols + j]);
						}
					}
				}
			}
			return result;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline void matrix<double, thisTransposed, thisContiguous>::self_exp()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		if constexpr (thisTransposed)
		{
			if constexpr (thisContiguous)
			{
				size_t finalPosSize = this->finalPosSize;
				size_t size = this->_size;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);

					_mm256_store_pd(&data1[i], _mm256_exp_pd(a));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] = std::exp(data1[i]);
				}
			}
			else
			{
				size_t matrix1ActualRows = this->actualRows;

				size_t finalPosRows = this->finalPosRows;
				size_t finalPosCols = this->finalPosCols;

				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						_mm256_store_pd(&data1[j * matrix1ActualRows + i], _mm256_exp_pd(a));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
							data1[(j + 1) * matrix1ActualRows + i],
							data1[(j + 2) * matrix1ActualRows + i],
							data1[(j + 3) * matrix1ActualRows + i]);

						__m256d exp = _mm256_exp_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&data1[j * matrix1ActualRows + i], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&data1[(j + 1) * matrix1ActualRows + i], val2);

						_mm_store_sd(&data1[(j + 2) * matrix1ActualRows + i], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&data1[(j + 3) * matrix1ActualRows + i], val1);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						data1[j * matrix1ActualRows + i] = std::exp(data1[j * matrix1ActualRows + i]);
					}
				}
			}
		}
		else
		{
			if constexpr (thisContiguous)
			{
				size_t size = this->_size;
				size_t finalPosSize = this->finalPosSize;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);
					_mm256_store_pd(&data1[i], _mm256_exp_pd(a));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] = std::exp(data1[i]);
				}
			}
			else
			{
				size_t matrix1ActualCols = this->actualCols;

				size_t finalPosCols = this->finalPosCols;
				size_t finalPosRows = this->finalPosRows;

				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						_mm256_store_pd(&data1[i * matrix1ActualCols + j], _mm256_exp_pd(a));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
							data1[(i + 1) * matrix1ActualCols + j],
							data1[(i + 2) * matrix1ActualCols + j],
							data1[(i + 3) * matrix1ActualCols + j]);

						__m256d exp = _mm256_exp_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&data1[i * matrix1ActualCols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&data1[(i + 1) * matrix1ActualCols + j], val2);

						_mm_store_sd(&data1[(i + 2) * matrix1ActualCols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&data1[(i + 3) * matrix1ActualCols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						data1[i * matrix1ActualCols + j] = std::exp(data1[i * matrix1ActualCols + j]);
					}
				}
			}
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed>
	inline matrix<double> matrix<double, thisTransposed, thisContiguous>::exp2()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		size_t finalPosSize = this->finalPosSize;
		size_t size = this->_size;

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix1ActualCols = this->actualCols;

		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;

		if constexpr (returnTransposed)
		{
			matrix<double> result(cols, rows);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&dataResult[i], _mm256_exp2_pd(a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = std::exp2(data1[i]);
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

							_mm256_store_pd(&dataResult[j * rows + i], _mm256_exp2_pd(a));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < finalPosCols; j += 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
								data1[(j + 1) * matrix1ActualRows + i],
								data1[(j + 2) * matrix1ActualRows + i],
								data1[(j + 3) * matrix1ActualRows + i]);

							__m256d exp = _mm256_exp2_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&dataResult[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

							_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
						}
						for (size_t j = finalPosCols; j < cols; j++)
						{
							dataResult[j * rows + i] = std::exp2(data1[j * matrix1ActualRows + i]);
						}
					}
				}
			}
			else
			{
				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
							data1[(i + 1) * matrix1ActualCols + j],
							data1[(i + 2) * matrix1ActualCols + j],
							data1[(i + 3) * matrix1ActualCols + j]);

						_mm256_store_pd(&dataResult[j * rows + i], _mm256_exp2_pd(a));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						__m256d exp = _mm256_exp2_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&dataResult[j * rows + i], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

						_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						dataResult[j * rows + i] = std::exp2(data1[i * matrix1ActualCols + j]);
					}
				}
			}
			return result;
		}
		else
		{
			matrix<double> result(rows, cols);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
							data1[(j + 1) * matrix1ActualRows + i],
							data1[(j + 2) * matrix1ActualRows + i],
							data1[(j + 3) * matrix1ActualRows + i]);
						_mm256_store_pd(&dataResult[i * cols + j], _mm256_exp2_pd(a));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						__m256d exp = _mm256_exp2_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&dataResult[i * cols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

						_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						dataResult[i * cols + j] = std::exp2(data1[j * matrix1ActualRows + i]);
					}
				}
			}
			else
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);
						_mm256_store_pd(&dataResult[i], _mm256_exp2_pd(a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = std::exp2(data1[i]);
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

							_mm256_store_pd(&dataResult[i * cols + j], _mm256_exp2_pd(a));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < finalPosRows; i += 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
								data1[(i + 1) * matrix1ActualCols + j],
								data1[(i + 2) * matrix1ActualCols + j],
								data1[(i + 3) * matrix1ActualCols + j]);

							__m256d exp = _mm256_exp2_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&dataResult[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

							_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
						}
						for (size_t i = finalPosRows; i < rows; i++)
						{
							dataResult[i * cols + j] = std::exp2(data1[i * matrix1ActualCols + j]);
						}
					}
				}
			}
			return result;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline void matrix<double, thisTransposed, thisContiguous>::self_exp2()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		if constexpr (thisTransposed)
		{
			if constexpr (thisContiguous)
			{
				size_t finalPosSize = this->finalPosSize;
				size_t size = this->_size;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);

					_mm256_store_pd(&data1[i], _mm256_exp2_pd(a));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] = std::exp2(data1[i]);
				}
			}
			else
			{
				size_t matrix1ActualRows = this->actualRows;

				size_t finalPosRows = this->finalPosRows;
				size_t finalPosCols = this->finalPosCols;

				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						_mm256_store_pd(&data1[j * matrix1ActualRows + i], _mm256_exp2_pd(a));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
							data1[(j + 1) * matrix1ActualRows + i],
							data1[(j + 2) * matrix1ActualRows + i],
							data1[(j + 3) * matrix1ActualRows + i]);

						__m256d exp = _mm256_exp2_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&data1[j * matrix1ActualRows + i], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&data1[(j + 1) * matrix1ActualRows + i], val2);

						_mm_store_sd(&data1[(j + 2) * matrix1ActualRows + i], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&data1[(j + 3) * matrix1ActualRows + i], val1);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						data1[j * matrix1ActualRows + i] = std::exp2(data1[j * matrix1ActualRows + i]);
					}
				}
			}
		}
		else
		{
			if constexpr (thisContiguous)
			{
				size_t size = this->_size;
				size_t finalPosSize = this->finalPosSize;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);
					_mm256_store_pd(&data1[i], _mm256_exp2_pd(a));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] = std::exp2(data1[i]);
				}
			}
			else
			{
				size_t matrix1ActualCols = this->actualCols;

				size_t finalPosCols = this->finalPosCols;
				size_t finalPosRows = this->finalPosRows;

				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						_mm256_store_pd(&data1[i * matrix1ActualCols + j], _mm256_exp2_pd(a));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
							data1[(i + 1) * matrix1ActualCols + j],
							data1[(i + 2) * matrix1ActualCols + j],
							data1[(i + 3) * matrix1ActualCols + j]);

						__m256d exp = _mm256_exp2_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&data1[i * matrix1ActualCols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&data1[(i + 1) * matrix1ActualCols + j], val2);

						_mm_store_sd(&data1[(i + 2) * matrix1ActualCols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&data1[(i + 3) * matrix1ActualCols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						data1[i * matrix1ActualCols + j] = std::exp2(data1[i * matrix1ActualCols + j]);
					}
				}
			}
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed>
	inline matrix<double> matrix<double, thisTransposed, thisContiguous>::log()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		size_t finalPosSize = this->finalPosSize;
		size_t size = this->_size;

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix1ActualCols = this->actualCols;

		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;

		if constexpr (returnTransposed)
		{
			matrix<double> result(cols, rows);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&dataResult[i], _mm256_log_pd(a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = std::log(data1[i]);
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

							_mm256_store_pd(&dataResult[j * rows + i], _mm256_log_pd(a));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < finalPosCols; j += 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
								data1[(j + 1) * matrix1ActualRows + i],
								data1[(j + 2) * matrix1ActualRows + i],
								data1[(j + 3) * matrix1ActualRows + i]);

							__m256d exp = _mm256_log_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&dataResult[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

							_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
						}
						for (size_t j = finalPosCols; j < cols; j++)
						{
							dataResult[j * rows + i] = std::log(data1[j * matrix1ActualRows + i]);
						}
					}
				}
			}
			else
			{
				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
							data1[(i + 1) * matrix1ActualCols + j],
							data1[(i + 2) * matrix1ActualCols + j],
							data1[(i + 3) * matrix1ActualCols + j]);

						_mm256_store_pd(&dataResult[j * rows + i], _mm256_log_pd(a));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						__m256d exp = _mm256_log_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&dataResult[j * rows + i], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

						_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						dataResult[j * rows + i] = std::log(data1[i * matrix1ActualCols + j]);
					}
				}
			}
			return result;
		}
		else
		{
			matrix<double> result(rows, cols);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
							data1[(j + 1) * matrix1ActualRows + i],
							data1[(j + 2) * matrix1ActualRows + i],
							data1[(j + 3) * matrix1ActualRows + i]);
						_mm256_store_pd(&dataResult[i * cols + j], _mm256_log_pd(a));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						__m256d exp = _mm256_log_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&dataResult[i * cols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

						_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						dataResult[i * cols + j] = std::log(data1[j * matrix1ActualRows + i]);
					}
				}
			}
			else
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);
						_mm256_store_pd(&dataResult[i], _mm256_log_pd(a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = std::log(data1[i]);
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

							_mm256_store_pd(&dataResult[i * cols + j], _mm256_log_pd(a));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < finalPosRows; i += 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
								data1[(i + 1) * matrix1ActualCols + j],
								data1[(i + 2) * matrix1ActualCols + j],
								data1[(i + 3) * matrix1ActualCols + j]);

							__m256d exp = _mm256_log_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&dataResult[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

							_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
						}
						for (size_t i = finalPosRows; i < rows; i++)
						{
							dataResult[i * cols + j] = std::log(data1[i * matrix1ActualCols + j]);
						}
					}
				}
			}
			return result;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline void matrix<double, thisTransposed, thisContiguous>::self_log()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		if constexpr (thisTransposed)
		{
			if constexpr (thisContiguous)
			{
				size_t finalPosSize = this->finalPosSize;
				size_t size = this->_size;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);

					_mm256_store_pd(&data1[i], _mm256_log_pd(a));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] = std::log(data1[i]);
				}
			}
			else
			{
				size_t matrix1ActualRows = this->actualRows;

				size_t finalPosRows = this->finalPosRows;
				size_t finalPosCols = this->finalPosCols;

				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						_mm256_store_pd(&data1[j * matrix1ActualRows + i], _mm256_log_pd(a));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
							data1[(j + 1) * matrix1ActualRows + i],
							data1[(j + 2) * matrix1ActualRows + i],
							data1[(j + 3) * matrix1ActualRows + i]);

						__m256d exp = _mm256_log_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&data1[j * matrix1ActualRows + i], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&data1[(j + 1) * matrix1ActualRows + i], val2);

						_mm_store_sd(&data1[(j + 2) * matrix1ActualRows + i], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&data1[(j + 3) * matrix1ActualRows + i], val1);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						data1[j * matrix1ActualRows + i] = std::log(data1[j * matrix1ActualRows + i]);
					}
				}
			}
		}
		else
		{
			if constexpr (thisContiguous)
			{
				size_t size = this->_size;
				size_t finalPosSize = this->finalPosSize;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);
					_mm256_store_pd(&data1[i], _mm256_log_pd(a));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] = std::log(data1[i]);
				}
			}
			else
			{
				size_t matrix1ActualCols = this->actualCols;

				size_t finalPosCols = this->finalPosCols;
				size_t finalPosRows = this->finalPosRows;

				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						_mm256_store_pd(&data1[i * matrix1ActualCols + j], _mm256_log_pd(a));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
							data1[(i + 1) * matrix1ActualCols + j],
							data1[(i + 2) * matrix1ActualCols + j],
							data1[(i + 3) * matrix1ActualCols + j]);

						__m256d exp = _mm256_log_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&data1[i * matrix1ActualCols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&data1[(i + 1) * matrix1ActualCols + j], val2);

						_mm_store_sd(&data1[(i + 2) * matrix1ActualCols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&data1[(i + 3) * matrix1ActualCols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						data1[i * matrix1ActualCols + j] = std::log(data1[i * matrix1ActualCols + j]);
					}
				}
			}
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed>
	inline matrix<double> matrix<double, thisTransposed, thisContiguous>::log2()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		size_t finalPosSize = this->finalPosSize;
		size_t size = this->_size;

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix1ActualCols = this->actualCols;

		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;

		if constexpr (returnTransposed)
		{
			matrix<double> result(cols, rows);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&dataResult[i], _mm256_log2_pd(a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = std::log2(data1[i]);
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

							_mm256_store_pd(&dataResult[j * rows + i], _mm256_log2_pd(a));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < finalPosCols; j += 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
								data1[(j + 1) * matrix1ActualRows + i],
								data1[(j + 2) * matrix1ActualRows + i],
								data1[(j + 3) * matrix1ActualRows + i]);

							__m256d exp = _mm256_log2_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&dataResult[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

							_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
						}
						for (size_t j = finalPosCols; j < cols; j++)
						{
							dataResult[j * rows + i] = std::log2(data1[j * matrix1ActualRows + i]);
						}
					}
				}
			}
			else
			{
				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
							data1[(i + 1) * matrix1ActualCols + j],
							data1[(i + 2) * matrix1ActualCols + j],
							data1[(i + 3) * matrix1ActualCols + j]);

						_mm256_store_pd(&dataResult[j * rows + i], _mm256_log2_pd(a));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						__m256d exp = _mm256_log2_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&dataResult[j * rows + i], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

						_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						dataResult[j * rows + i] = std::log2(data1[i * matrix1ActualCols + j]);
					}
				}
			}
			return result;
		}
		else
		{
			matrix<double> result(rows, cols);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
							data1[(j + 1) * matrix1ActualRows + i],
							data1[(j + 2) * matrix1ActualRows + i],
							data1[(j + 3) * matrix1ActualRows + i]);
						_mm256_store_pd(&dataResult[i * cols + j], _mm256_log2_pd(a));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						__m256d exp = _mm256_log2_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&dataResult[i * cols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

						_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						dataResult[i * cols + j] = std::log2(data1[j * matrix1ActualRows + i]);
					}
				}
			}
			else
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);
						_mm256_store_pd(&dataResult[i], _mm256_log2_pd(a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = std::log2(data1[i]);
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

							_mm256_store_pd(&dataResult[i * cols + j], _mm256_log2_pd(a));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < finalPosRows; i += 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
								data1[(i + 1) * matrix1ActualCols + j],
								data1[(i + 2) * matrix1ActualCols + j],
								data1[(i + 3) * matrix1ActualCols + j]);

							__m256d exp = _mm256_log2_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&dataResult[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

							_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
						}
						for (size_t i = finalPosRows; i < rows; i++)
						{
							dataResult[i * cols + j] = std::log2(data1[i * matrix1ActualCols + j]);
						}
					}
				}
			}
			return result;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline void matrix<double, thisTransposed, thisContiguous>::self_log2()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		if constexpr (thisTransposed)
		{
			if constexpr (thisContiguous)
			{
				size_t finalPosSize = this->finalPosSize;
				size_t size = this->_size;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);

					_mm256_store_pd(&data1[i], _mm256_log2_pd(a));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] = std::log2(data1[i]);
				}
			}
			else
			{
				size_t matrix1ActualRows = this->actualRows;

				size_t finalPosRows = this->finalPosRows;
				size_t finalPosCols = this->finalPosCols;

				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						_mm256_store_pd(&data1[j * matrix1ActualRows + i], _mm256_log2_pd(a));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
							data1[(j + 1) * matrix1ActualRows + i],
							data1[(j + 2) * matrix1ActualRows + i],
							data1[(j + 3) * matrix1ActualRows + i]);

						__m256d exp = _mm256_log2_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&data1[j * matrix1ActualRows + i], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&data1[(j + 1) * matrix1ActualRows + i], val2);

						_mm_store_sd(&data1[(j + 2) * matrix1ActualRows + i], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&data1[(j + 3) * matrix1ActualRows + i], val1);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						data1[j * matrix1ActualRows + i] = std::log2(data1[j * matrix1ActualRows + i]);
					}
				}
			}
		}
		else
		{
			if constexpr (thisContiguous)
			{
				size_t size = this->_size;
				size_t finalPosSize = this->finalPosSize;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);
					_mm256_store_pd(&data1[i], _mm256_log2_pd(a));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] = std::log2(data1[i]);
				}
			}
			else
			{
				size_t matrix1ActualCols = this->actualCols;

				size_t finalPosCols = this->finalPosCols;
				size_t finalPosRows = this->finalPosRows;

				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						_mm256_store_pd(&data1[i * matrix1ActualCols + j], _mm256_log2_pd(a));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
							data1[(i + 1) * matrix1ActualCols + j],
							data1[(i + 2) * matrix1ActualCols + j],
							data1[(i + 3) * matrix1ActualCols + j]);

						__m256d exp = _mm256_log2_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&data1[i * matrix1ActualCols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&data1[(i + 1) * matrix1ActualCols + j], val2);

						_mm_store_sd(&data1[(i + 2) * matrix1ActualCols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&data1[(i + 3) * matrix1ActualCols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						data1[i * matrix1ActualCols + j] = std::log2(data1[i * matrix1ActualCols + j]);
					}
				}
			}
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed>
	inline matrix<double> matrix<double, thisTransposed, thisContiguous>::log10()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		size_t finalPosSize = this->finalPosSize;
		size_t size = this->_size;

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix1ActualCols = this->actualCols;

		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;

		if constexpr (returnTransposed)
		{
			matrix<double> result(cols, rows);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&dataResult[i], _mm256_log10_pd(a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = std::log10(data1[i]);
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

							_mm256_store_pd(&dataResult[j * rows + i], _mm256_log10_pd(a));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < finalPosCols; j += 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
								data1[(j + 1) * matrix1ActualRows + i],
								data1[(j + 2) * matrix1ActualRows + i],
								data1[(j + 3) * matrix1ActualRows + i]);

							__m256d exp = _mm256_log10_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&dataResult[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

							_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
						}
						for (size_t j = finalPosCols; j < cols; j++)
						{
							dataResult[j * rows + i] = std::log10(data1[j * matrix1ActualRows + i]);
						}
					}
				}
			}
			else
			{
				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
							data1[(i + 1) * matrix1ActualCols + j],
							data1[(i + 2) * matrix1ActualCols + j],
							data1[(i + 3) * matrix1ActualCols + j]);

						_mm256_store_pd(&dataResult[j * rows + i], _mm256_log10_pd(a));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						__m256d exp = _mm256_log10_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&dataResult[j * rows + i], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

						_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						dataResult[j * rows + i] = std::log10(data1[i * matrix1ActualCols + j]);
					}
				}
			}
			return result;
		}
		else
		{
			matrix<double> result(rows, cols);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
							data1[(j + 1) * matrix1ActualRows + i],
							data1[(j + 2) * matrix1ActualRows + i],
							data1[(j + 3) * matrix1ActualRows + i]);
						_mm256_store_pd(&dataResult[i * cols + j], _mm256_log10_pd(a));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						__m256d exp = _mm256_log10_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&dataResult[i * cols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

						_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						dataResult[i * cols + j] = std::log10(data1[j * matrix1ActualRows + i]);
					}
				}
			}
			else
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);
						_mm256_store_pd(&dataResult[i], _mm256_log10_pd(a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = std::log10(data1[i]);
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

							_mm256_store_pd(&dataResult[i * cols + j], _mm256_log10_pd(a));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < finalPosRows; i += 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
								data1[(i + 1) * matrix1ActualCols + j],
								data1[(i + 2) * matrix1ActualCols + j],
								data1[(i + 3) * matrix1ActualCols + j]);

							__m256d exp = _mm256_log10_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&dataResult[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

							_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
						}
						for (size_t i = finalPosRows; i < rows; i++)
						{
							dataResult[i * cols + j] = std::log10(data1[i * matrix1ActualCols + j]);
						}
					}
				}
			}
			return result;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline void matrix<double, thisTransposed, thisContiguous>::self_log10()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		if constexpr (thisTransposed)
		{
			if constexpr (thisContiguous)
			{
				size_t finalPosSize = this->finalPosSize;
				size_t size = this->_size;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);

					_mm256_store_pd(&data1[i], _mm256_log10_pd(a));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] = std::log10(data1[i]);
				}
			}
			else
			{
				size_t matrix1ActualRows = this->actualRows;

				size_t finalPosRows = this->finalPosRows;
				size_t finalPosCols = this->finalPosCols;

				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						_mm256_store_pd(&data1[j * matrix1ActualRows + i], _mm256_log10_pd(a));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
							data1[(j + 1) * matrix1ActualRows + i],
							data1[(j + 2) * matrix1ActualRows + i],
							data1[(j + 3) * matrix1ActualRows + i]);

						__m256d exp = _mm256_log10_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&data1[j * matrix1ActualRows + i], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&data1[(j + 1) * matrix1ActualRows + i], val2);

						_mm_store_sd(&data1[(j + 2) * matrix1ActualRows + i], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&data1[(j + 3) * matrix1ActualRows + i], val1);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						data1[j * matrix1ActualRows + i] = std::log10(data1[j * matrix1ActualRows + i]);
					}
				}
			}
		}
		else
		{
			if constexpr (thisContiguous)
			{
				size_t size = this->_size;
				size_t finalPosSize = this->finalPosSize;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);
					_mm256_store_pd(&data1[i], _mm256_log10_pd(a));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] = std::log10(data1[i]);
				}
			}
			else
			{
				size_t matrix1ActualCols = this->actualCols;

				size_t finalPosCols = this->finalPosCols;
				size_t finalPosRows = this->finalPosRows;

				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						_mm256_store_pd(&data1[i * matrix1ActualCols + j], _mm256_log10_pd(a));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
							data1[(i + 1) * matrix1ActualCols + j],
							data1[(i + 2) * matrix1ActualCols + j],
							data1[(i + 3) * matrix1ActualCols + j]);

						__m256d exp = _mm256_log10_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&data1[i * matrix1ActualCols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&data1[(i + 1) * matrix1ActualCols + j], val2);

						_mm_store_sd(&data1[(i + 2) * matrix1ActualCols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&data1[(i + 3) * matrix1ActualCols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						data1[i * matrix1ActualCols + j] = std::log10(data1[i * matrix1ActualCols + j]);
					}
				}
			}
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed>
	inline matrix<double> matrix<double, thisTransposed, thisContiguous>::abs()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		size_t finalPosSize = this->finalPosSize;
		size_t size = this->_size;

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix1ActualCols = this->actualCols;

		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;

		__m256d mask = _mm256_set1_pd(-0.0);

		if constexpr (returnTransposed)
		{
			matrix<double> result(cols, rows);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&dataResult[i], _mm256_abs_pd(a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = std::fabs(data1[i]);
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

							_mm256_store_pd(&dataResult[j * rows + i], _mm256_abs_pd(a));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = std::fabs(data1[j * matrix1ActualRows + i]);
						}
					}
				}
			}
			else
			{
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						dataResult[j * rows + i] = std::fabs(data1[i * matrix1ActualCols + j]);
					}
				}
			}
			return result;
		}
		else
		{
			matrix<double> result(rows, cols);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				for (size_t j = 0; j < cols; j++)
				{
					for (size_t i = 0; i < rows; i++)
					{
						dataResult[i * cols + j] = std::fabs(data1[j * matrix1ActualRows + i]);
					}
				}
			}
			else
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);
						_mm256_store_pd(&dataResult[i], _mm256_abs_pd(a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = std::fabs(data1[i]);
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

							_mm256_store_pd(&dataResult[i * cols + j], _mm256_abs_pd(a));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = std::fabs(data1[i * matrix1ActualCols + j]);
						}
					}
				}
			}
			return result;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline void matrix<double, thisTransposed, thisContiguous>::self_abs()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		__m256d mask = _mm256_set1_pd(-0.0);

		if constexpr (thisTransposed)
		{
			if constexpr (thisContiguous)
			{
				size_t finalPosSize = this->finalPosSize;
				size_t size = this->_size;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);

					_mm256_store_pd(&data1[i], _mm256_abs_pd(a));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] = std::fabs(data1[i]);
				}
			}
			else
			{
				size_t matrix1ActualRows = this->actualRows;

				size_t finalPosRows = this->finalPosRows;
				size_t finalPosCols = this->finalPosCols;

				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					for (size_t j = 0; j < finalPosCols; j++)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						_mm256_store_pd(&data1[j * matrix1ActualRows + i], _mm256_abs_pd(a));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
							data1[(j + 1) * matrix1ActualRows + i],
							data1[(j + 2) * matrix1ActualRows + i],
							data1[(j + 3) * matrix1ActualRows + i]);

						__m256d exp = _mm256_abs_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&data1[j * matrix1ActualRows + i], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&data1[(j + 1) * matrix1ActualRows + i], val2);

						_mm_store_sd(&data1[(j + 2) * matrix1ActualRows + i], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&data1[(j + 3) * matrix1ActualRows + i], val1);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						data1[j * matrix1ActualRows + i] = std::fabs(data1[j * matrix1ActualRows + i]);
					}
				}
			}
		}
		else
		{
			if constexpr (thisContiguous)
			{
				size_t size = this->_size;
				size_t finalPosSize = this->finalPosSize;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);
					_mm256_store_pd(&data1[i], _mm256_abs_pd(a));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] = std::fabs(data1[i]);
				}
			}
			else
			{
				size_t matrix1ActualCols = this->actualCols;

				size_t finalPosCols = this->finalPosCols;
				size_t finalPosRows = this->finalPosRows;

				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						_mm256_store_pd(&data1[i * matrix1ActualCols + j], _mm256_abs_pd(a));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
							data1[(i + 1) * matrix1ActualCols + j],
							data1[(i + 2) * matrix1ActualCols + j],
							data1[(i + 3) * matrix1ActualCols + j]);

						__m256d exp = _mm256_abs_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&data1[i * matrix1ActualCols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&data1[(i + 1) * matrix1ActualCols + j], val2);

						_mm_store_sd(&data1[(i + 2) * matrix1ActualCols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&data1[(i + 3) * matrix1ActualCols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						data1[i * matrix1ActualCols + j] = std::fabs(data1[i * matrix1ActualCols + j]);
					}
				}
			}
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed>
	inline matrix<double> matrix<double, thisTransposed, thisContiguous>::cos()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		size_t finalPosSize = this->finalPosSize;
		size_t size = this->_size;

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix1ActualCols = this->actualCols;

		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;

		if constexpr (returnTransposed)
		{
			matrix<double> result(cols, rows);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&dataResult[i], _mm256_cos_pd(a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = std::cos(data1[i]);
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

							_mm256_store_pd(&dataResult[j * rows + i], _mm256_cos_pd(a));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < finalPosCols; j += 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
								data1[(j + 1) * matrix1ActualRows + i],
								data1[(j + 2) * matrix1ActualRows + i],
								data1[(j + 3) * matrix1ActualRows + i]);

							__m256d exp = _mm256_cos_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&dataResult[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

							_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
						}
						for (size_t j = finalPosCols; j < cols; j++)
						{
							dataResult[j * rows + i] = std::cos(data1[j * matrix1ActualRows + i]);
						}
					}
				}
			}
			else
			{
				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
							data1[(i + 1) * matrix1ActualCols + j],
							data1[(i + 2) * matrix1ActualCols + j],
							data1[(i + 3) * matrix1ActualCols + j]);

						_mm256_store_pd(&dataResult[j * rows + i], _mm256_cos_pd(a));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						__m256d exp = _mm256_cos_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&dataResult[j * rows + i], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

						_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						dataResult[j * rows + i] = std::cos(data1[i * matrix1ActualCols + j]);
					}
				}
			}
			return result;
		}
		else
		{
			matrix<double> result(rows, cols);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
							data1[(j + 1) * matrix1ActualRows + i],
							data1[(j + 2) * matrix1ActualRows + i],
							data1[(j + 3) * matrix1ActualRows + i]);
						_mm256_store_pd(&dataResult[i * cols + j], _mm256_cos_pd(a));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						__m256d exp = _mm256_cos_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&dataResult[i * cols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

						_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						dataResult[i * cols + j] = std::cos(data1[j * matrix1ActualRows + i]);
					}
				}
			}
			else
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);
						_mm256_store_pd(&dataResult[i], _mm256_cos_pd(a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = std::cos(data1[i]);
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

							_mm256_store_pd(&dataResult[i * cols + j], _mm256_cos_pd(a));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < finalPosRows; i += 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
								data1[(i + 1) * matrix1ActualCols + j],
								data1[(i + 2) * matrix1ActualCols + j],
								data1[(i + 3) * matrix1ActualCols + j]);

							__m256d exp = _mm256_cos_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&dataResult[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

							_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
						}
						for (size_t i = finalPosRows; i < rows; i++)
						{
							dataResult[i * cols + j] = std::cos(data1[i * matrix1ActualCols + j]);
						}
					}
				}
			}
			return result;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline void matrix<double, thisTransposed, thisContiguous>::self_cos()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		if constexpr (thisTransposed)
		{
			if constexpr (thisContiguous)
			{
				size_t finalPosSize = this->finalPosSize;
				size_t size = this->_size;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);

					_mm256_store_pd(&data1[i], _mm256_cos_pd(a));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] = std::cos(data1[i]);
				}
			}
			else
			{
				size_t matrix1ActualRows = this->actualRows;

				size_t finalPosRows = this->finalPosRows;
				size_t finalPosCols = this->finalPosCols;

				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						_mm256_store_pd(&data1[j * matrix1ActualRows + i], _mm256_cos_pd(a));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
							data1[(j + 1) * matrix1ActualRows + i],
							data1[(j + 2) * matrix1ActualRows + i],
							data1[(j + 3) * matrix1ActualRows + i]);

						__m256d exp = _mm256_cos_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&data1[j * matrix1ActualRows + i], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&data1[(j + 1) * matrix1ActualRows + i], val2);

						_mm_store_sd(&data1[(j + 2) * matrix1ActualRows + i], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&data1[(j + 3) * matrix1ActualRows + i], val1);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						data1[j * matrix1ActualRows + i] = std::cos(data1[j * matrix1ActualRows + i]);
					}
				}
			}
		}
		else
		{
			if constexpr (thisContiguous)
			{
				size_t size = this->_size;
				size_t finalPosSize = this->finalPosSize;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);
					_mm256_store_pd(&data1[i], _mm256_cos_pd(a));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] = std::cos(data1[i]);
				}
			}
			else
			{
				size_t matrix1ActualCols = this->actualCols;

				size_t finalPosCols = this->finalPosCols;
				size_t finalPosRows = this->finalPosRows;

				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						_mm256_store_pd(&data1[i * matrix1ActualCols + j], _mm256_cos_pd(a));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
							data1[(i + 1) * matrix1ActualCols + j],
							data1[(i + 2) * matrix1ActualCols + j],
							data1[(i + 3) * matrix1ActualCols + j]);

						__m256d exp = _mm256_cos_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&data1[i * matrix1ActualCols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&data1[(i + 1) * matrix1ActualCols + j], val2);

						_mm_store_sd(&data1[(i + 2) * matrix1ActualCols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&data1[(i + 3) * matrix1ActualCols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						data1[i * matrix1ActualCols + j] = std::cos(data1[i * matrix1ActualCols + j]);
					}
				}
			}
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed>
	inline matrix<double> matrix<double, thisTransposed, thisContiguous>::tan()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		size_t finalPosSize = this->finalPosSize;
		size_t size = this->_size;

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix1ActualCols = this->actualCols;

		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;

		if constexpr (returnTransposed)
		{
			matrix<double> result(cols, rows);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&dataResult[i], _mm256_tan_pd(a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = std::tan(data1[i]);
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

							_mm256_store_pd(&dataResult[j * rows + i], _mm256_tan_pd(a));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < finalPosCols; j += 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
								data1[(j + 1) * matrix1ActualRows + i],
								data1[(j + 2) * matrix1ActualRows + i],
								data1[(j + 3) * matrix1ActualRows + i]);

							__m256d exp = _mm256_tan_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&dataResult[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

							_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
						}
						for (size_t j = finalPosCols; j < cols; j++)
						{
							dataResult[j * rows + i] = std::tan(data1[j * matrix1ActualRows + i]);
						}
					}
				}
			}
			else
			{
				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
							data1[(i + 1) * matrix1ActualCols + j],
							data1[(i + 2) * matrix1ActualCols + j],
							data1[(i + 3) * matrix1ActualCols + j]);

						_mm256_store_pd(&dataResult[j * rows + i], _mm256_tan_pd(a));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						__m256d exp = _mm256_tan_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&dataResult[j * rows + i], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

						_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						dataResult[j * rows + i] = std::tan(data1[i * matrix1ActualCols + j]);
					}
				}
			}
			return result;
		}
		else
		{
			matrix<double> result(rows, cols);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
							data1[(j + 1) * matrix1ActualRows + i],
							data1[(j + 2) * matrix1ActualRows + i],
							data1[(j + 3) * matrix1ActualRows + i]);
						_mm256_store_pd(&dataResult[i * cols + j], _mm256_tan_pd(a));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						__m256d exp = _mm256_tan_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&dataResult[i * cols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

						_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						dataResult[i * cols + j] = std::tan(data1[j * matrix1ActualRows + i]);
					}
				}
			}
			else
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);
						_mm256_store_pd(&dataResult[i], _mm256_tan_pd(a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = std::tan(data1[i]);
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

							_mm256_store_pd(&dataResult[i * cols + j], _mm256_tan_pd(a));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < finalPosRows; i += 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
								data1[(i + 1) * matrix1ActualCols + j],
								data1[(i + 2) * matrix1ActualCols + j],
								data1[(i + 3) * matrix1ActualCols + j]);

							__m256d exp = _mm256_tan_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&dataResult[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

							_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
						}
						for (size_t i = finalPosRows; i < rows; i++)
						{
							dataResult[i * cols + j] = std::tan(data1[i * matrix1ActualCols + j]);
						}
					}
				}
			}
			return result;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline void matrix<double, thisTransposed, thisContiguous>::self_tan()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		if constexpr (thisTransposed)
		{
			if constexpr (thisContiguous)
			{
				size_t finalPosSize = this->finalPosSize;
				size_t size = this->_size;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);

					_mm256_store_pd(&data1[i], _mm256_tan_pd(a));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] = std::tan(data1[i]);
				}
			}
			else
			{
				size_t matrix1ActualRows = this->actualRows;

				size_t finalPosRows = this->finalPosRows;
				size_t finalPosCols = this->finalPosCols;

				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						_mm256_store_pd(&data1[j * matrix1ActualRows + i], _mm256_tan_pd(a));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
							data1[(j + 1) * matrix1ActualRows + i],
							data1[(j + 2) * matrix1ActualRows + i],
							data1[(j + 3) * matrix1ActualRows + i]);

						__m256d exp = _mm256_tan_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&data1[j * matrix1ActualRows + i], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&data1[(j + 1) * matrix1ActualRows + i], val2);

						_mm_store_sd(&data1[(j + 2) * matrix1ActualRows + i], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&data1[(j + 3) * matrix1ActualRows + i], val1);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						data1[j * matrix1ActualRows + i] = std::tan(data1[j * matrix1ActualRows + i]);
					}
				}
			}
		}
		else
		{
			if constexpr (thisContiguous)
			{
				size_t size = this->_size;
				size_t finalPosSize = this->finalPosSize;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);
					_mm256_store_pd(&data1[i], _mm256_tan_pd(a));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] = std::tan(data1[i]);
				}
			}
			else
			{
				size_t matrix1ActualCols = this->actualCols;

				size_t finalPosCols = this->finalPosCols;
				size_t finalPosRows = this->finalPosRows;

				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						_mm256_store_pd(&data1[i * matrix1ActualCols + j], _mm256_tan_pd(a));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
							data1[(i + 1) * matrix1ActualCols + j],
							data1[(i + 2) * matrix1ActualCols + j],
							data1[(i + 3) * matrix1ActualCols + j]);

						__m256d exp = _mm256_tan_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&data1[i * matrix1ActualCols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&data1[(i + 1) * matrix1ActualCols + j], val2);

						_mm_store_sd(&data1[(i + 2) * matrix1ActualCols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&data1[(i + 3) * matrix1ActualCols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						data1[i * matrix1ActualCols + j] = std::tan(data1[i * matrix1ActualCols + j]);
					}
				}
			}
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed>
	inline matrix<double> matrix<double, thisTransposed, thisContiguous>::acos()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		size_t finalPosSize = this->finalPosSize;
		size_t size = this->_size;

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix1ActualCols = this->actualCols;

		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;

		if constexpr (returnTransposed)
		{
			matrix<double> result(cols, rows);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&dataResult[i], _mm256_acos_pd(a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = std::acos(data1[i]);
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

							_mm256_store_pd(&dataResult[j * rows + i], _mm256_acos_pd(a));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < finalPosCols; j += 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
								data1[(j + 1) * matrix1ActualRows + i],
								data1[(j + 2) * matrix1ActualRows + i],
								data1[(j + 3) * matrix1ActualRows + i]);

							__m256d exp = _mm256_acos_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&dataResult[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

							_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
						}
						for (size_t j = finalPosCols; j < cols; j++)
						{
							dataResult[j * rows + i] = std::acos(data1[j * matrix1ActualRows + i]);
						}
					}
				}
			}
			else
			{
				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
							data1[(i + 1) * matrix1ActualCols + j],
							data1[(i + 2) * matrix1ActualCols + j],
							data1[(i + 3) * matrix1ActualCols + j]);

						_mm256_store_pd(&dataResult[j * rows + i], _mm256_acos_pd(a));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						__m256d exp = _mm256_acos_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&dataResult[j * rows + i], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

						_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						dataResult[j * rows + i] = std::acos(data1[i * matrix1ActualCols + j]);
					}
				}
			}
			return result;
		}
		else
		{
			matrix<double> result(rows, cols);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
							data1[(j + 1) * matrix1ActualRows + i],
							data1[(j + 2) * matrix1ActualRows + i],
							data1[(j + 3) * matrix1ActualRows + i]);
						_mm256_store_pd(&dataResult[i * cols + j], _mm256_acos_pd(a));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						__m256d exp = _mm256_acos_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&dataResult[i * cols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

						_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						dataResult[i * cols + j] = std::acos(data1[j * matrix1ActualRows + i]);
					}
				}
			}
			else
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);
						_mm256_store_pd(&dataResult[i], _mm256_acos_pd(a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = std::acos(data1[i]);
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

							_mm256_store_pd(&dataResult[i * cols + j], _mm256_acos_pd(a));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < finalPosRows; i += 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
								data1[(i + 1) * matrix1ActualCols + j],
								data1[(i + 2) * matrix1ActualCols + j],
								data1[(i + 3) * matrix1ActualCols + j]);

							__m256d exp = _mm256_acos_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&dataResult[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

							_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
						}
						for (size_t i = finalPosRows; i < rows; i++)
						{
							dataResult[i * cols + j] = std::acos(data1[i * matrix1ActualCols + j]);
						}
					}
				}
			}
			return result;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline void matrix<double, thisTransposed, thisContiguous>::self_acos()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		if constexpr (thisTransposed)
		{
			if constexpr (thisContiguous)
			{
				size_t finalPosSize = this->finalPosSize;
				size_t size = this->_size;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);

					_mm256_store_pd(&data1[i], _mm256_acos_pd(a));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] = std::acos(data1[i]);
				}
			}
			else
			{
				size_t matrix1ActualRows = this->actualRows;

				size_t finalPosRows = this->finalPosRows;
				size_t finalPosCols = this->finalPosCols;

				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						_mm256_store_pd(&data1[j * matrix1ActualRows + i], _mm256_acos_pd(a));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
							data1[(j + 1) * matrix1ActualRows + i],
							data1[(j + 2) * matrix1ActualRows + i],
							data1[(j + 3) * matrix1ActualRows + i]);

						__m256d exp = _mm256_acos_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&data1[j * matrix1ActualRows + i], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&data1[(j + 1) * matrix1ActualRows + i], val2);

						_mm_store_sd(&data1[(j + 2) * matrix1ActualRows + i], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&data1[(j + 3) * matrix1ActualRows + i], val1);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						data1[j * matrix1ActualRows + i] = std::acos(data1[j * matrix1ActualRows + i]);
					}
				}
			}
		}
		else
		{
			if constexpr (thisContiguous)
			{
				size_t size = this->_size;
				size_t finalPosSize = this->finalPosSize;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);
					_mm256_store_pd(&data1[i], _mm256_acos_pd(a));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] = std::acos(data1[i]);
				}
			}
			else
			{
				size_t matrix1ActualCols = this->actualCols;

				size_t finalPosCols = this->finalPosCols;
				size_t finalPosRows = this->finalPosRows;

				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						_mm256_store_pd(&data1[i * matrix1ActualCols + j], _mm256_acos_pd(a));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
							data1[(i + 1) * matrix1ActualCols + j],
							data1[(i + 2) * matrix1ActualCols + j],
							data1[(i + 3) * matrix1ActualCols + j]);

						__m256d exp = _mm256_acos_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&data1[i * matrix1ActualCols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&data1[(i + 1) * matrix1ActualCols + j], val2);

						_mm_store_sd(&data1[(i + 2) * matrix1ActualCols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&data1[(i + 3) * matrix1ActualCols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						data1[i * matrix1ActualCols + j] = std::acos(data1[i * matrix1ActualCols + j]);
					}
				}
			}
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed>
	inline matrix<double> matrix<double, thisTransposed, thisContiguous>::round()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		size_t finalPosSize = this->finalPosSize;
		size_t size = this->_size;

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix1ActualCols = this->actualCols;

		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;

		if constexpr (returnTransposed)
		{
			matrix<double> result(cols, rows);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&dataResult[i], _mm256_round_pd(a, _MM_FROUND_TO_NEAREST_INT));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = std::round(data1[i]);
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

							_mm256_store_pd(&dataResult[j * rows + i], _mm256_round_pd(a, _MM_FROUND_TO_NEAREST_INT));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = std::round(data1[j * matrix1ActualRows + i]);
						}
					}
				}
			}
			else
			{
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						__m256d exp = _mm256_round_pd(a, _MM_FROUND_TO_NEAREST_INT);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&dataResult[j * rows + i], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

						_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						dataResult[j * rows + i] = std::round(data1[i * matrix1ActualCols + j]);
					}
				}
			}
			return result;
		}
		else
		{
			matrix<double> result(rows, cols);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				for (size_t j = 0; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						__m256d exp = _mm256_round_pd(a, _MM_FROUND_TO_NEAREST_INT);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&dataResult[i * cols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

						_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						dataResult[i * cols + j] = std::round(data1[j * matrix1ActualRows + i]);
					}
				}
			}
			else
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);
						_mm256_store_pd(&dataResult[i], _mm256_round_pd(a, _MM_FROUND_TO_NEAREST_INT));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = std::round(data1[i]);
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

							_mm256_store_pd(&dataResult[i * cols + j], _mm256_round_pd(a, _MM_FROUND_TO_NEAREST_INT));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = std::round(data1[i * matrix1ActualCols + j]);
						}
					}
				}
			}
			return result;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline void matrix<double, thisTransposed, thisContiguous>::self_round()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		if constexpr (thisTransposed)
		{
			if constexpr (thisContiguous)
			{
				size_t finalPosSize = this->finalPosSize;
				size_t size = this->_size;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);

					_mm256_store_pd(&data1[i], _mm256_round_pd(a, _MM_FROUND_TO_NEAREST_INT));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] = std::round(data1[i]);
				}
			}
			else
			{
				size_t matrix1ActualRows = this->actualRows;

				size_t finalPosRows = this->finalPosRows;
				size_t finalPosCols = this->finalPosCols;

				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						_mm256_store_pd(&data1[j * matrix1ActualRows + i], _mm256_round_pd(a, _MM_FROUND_TO_NEAREST_INT));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						data1[j * matrix1ActualRows + i] = std::round(data1[j * matrix1ActualRows + i]);
					}
				}
			}
		}
		else
		{
			if constexpr (thisContiguous)
			{
				size_t size = this->_size;
				size_t finalPosSize = this->finalPosSize;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);
					_mm256_store_pd(&data1[i], _mm256_round_pd(a, _MM_FROUND_TO_NEAREST_INT));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] = std::round(data1[i]);
				}
			}
			else
			{
				size_t matrix1ActualCols = this->actualCols;

				size_t finalPosCols = this->finalPosCols;
				size_t finalPosRows = this->finalPosRows;

				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						_mm256_store_pd(&data1[i * matrix1ActualCols + j], _mm256_round_pd(a, _MM_FROUND_TO_NEAREST_INT));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < rows; i++)
					{
						data1[i * matrix1ActualCols + j] = std::round(data1[i * matrix1ActualCols + j]);
					}
				}
			}
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed>
	inline matrix<double> matrix<double, thisTransposed, thisContiguous>::floor()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		size_t finalPosSize = this->finalPosSize;
		size_t size = this->_size;

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix1ActualCols = this->actualCols;

		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;

		if constexpr (returnTransposed)
		{
			matrix<double> result(cols, rows);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&dataResult[i], _mm256_floor_pd(a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = std::floor(data1[i]);
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

							_mm256_store_pd(&dataResult[j * rows + i], _mm256_floor_pd(a));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = std::floor(data1[j * matrix1ActualRows + i]);
						}
					}
				}
			}
			else
			{
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						__m256d exp = _mm256_floor_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&dataResult[j * rows + i], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

						_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						dataResult[j * rows + i] = std::floor(data1[i * matrix1ActualCols + j]);
					}
				}
			}
			return result;
		}
		else
		{
			matrix<double> result(rows, cols);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				for (size_t j = 0; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						__m256d exp = _mm256_floor_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&dataResult[i * cols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

						_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						dataResult[i * cols + j] = std::floor(data1[j * matrix1ActualRows + i]);
					}
				}
			}
			else
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);
						_mm256_store_pd(&dataResult[i], _mm256_floor_pd(a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = std::floor(data1[i]);
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

							_mm256_store_pd(&dataResult[i * cols + j], _mm256_floor_pd(a));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = std::floor(data1[i * matrix1ActualCols + j]);
						}
					}
				}
			}
			return result;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline void matrix<double, thisTransposed, thisContiguous>::self_floor()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		if constexpr (thisTransposed)
		{
			if constexpr (thisContiguous)
			{
				size_t finalPosSize = this->finalPosSize;
				size_t size = this->_size;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);

					_mm256_store_pd(&data1[i], _mm256_floor_pd(a));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] = std::floor(data1[i]);
				}
			}
			else
			{
				size_t matrix1ActualRows = this->actualRows;

				size_t finalPosRows = this->finalPosRows;
				size_t finalPosCols = this->finalPosCols;

				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						_mm256_store_pd(&data1[j * matrix1ActualRows + i], _mm256_floor_pd(a));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						data1[j * matrix1ActualRows + i] = std::floor(data1[j * matrix1ActualRows + i]);
					}
				}
			}
		}
		else
		{
			if constexpr (thisContiguous)
			{
				size_t size = this->_size;
				size_t finalPosSize = this->finalPosSize;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);
					_mm256_store_pd(&data1[i], _mm256_floor_pd(a));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] = std::floor(data1[i]);
				}
			}
			else
			{
				size_t matrix1ActualCols = this->actualCols;

				size_t finalPosCols = this->finalPosCols;
				size_t finalPosRows = this->finalPosRows;

				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						_mm256_store_pd(&data1[i * matrix1ActualCols + j], _mm256_floor_pd(a));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < rows; i++)
					{
						data1[i * matrix1ActualCols + j] = std::floor(data1[i * matrix1ActualCols + j]);
					}
				}
			}
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed>
	inline matrix<double> matrix<double, thisTransposed, thisContiguous>::ceil()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		size_t finalPosSize = this->finalPosSize;
		size_t size = this->_size;

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix1ActualCols = this->actualCols;

		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;

		if constexpr (returnTransposed)
		{
			matrix<double> result(cols, rows);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&dataResult[i], _mm256_ceil_pd(a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = std::ceil(data1[i]);
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

							_mm256_store_pd(&dataResult[j * rows + i], _mm256_ceil_pd(a));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = std::ceil(data1[j * matrix1ActualRows + i]);
						}
					}
				}
			}
			else
			{
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						__m256d exp = _mm256_ceil_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&dataResult[j * rows + i], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

						_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						dataResult[j * rows + i] = std::ceil(data1[i * matrix1ActualCols + j]);
					}
				}
			}
			return result;
		}
		else
		{
			matrix<double> result(rows, cols);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				for (size_t j = 0; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						__m256d exp = _mm256_ceil_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&dataResult[i * cols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

						_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						dataResult[i * cols + j] = std::ceil(data1[j * matrix1ActualRows + i]);
					}
				}
			}
			else
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);
						_mm256_store_pd(&dataResult[i], _mm256_ceil_pd(a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = std::ceil(data1[i]);
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

							_mm256_store_pd(&dataResult[i * cols + j], _mm256_ceil_pd(a));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = std::ceil(data1[i * matrix1ActualCols + j]);
						}
					}
				}
			}
			return result;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline void matrix<double, thisTransposed, thisContiguous>::self_ceil()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		if constexpr (thisTransposed)
		{
			if constexpr (thisContiguous)
			{
				size_t finalPosSize = this->finalPosSize;
				size_t size = this->_size;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);

					_mm256_store_pd(&data1[i], _mm256_ceil_pd(a));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] = std::ceil(data1[i]);
				}
			}
			else
			{
				size_t matrix1ActualRows = this->actualRows;

				size_t finalPosRows = this->finalPosRows;
				size_t finalPosCols = this->finalPosCols;

				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						_mm256_store_pd(&data1[j * matrix1ActualRows + i], _mm256_ceil_pd(a));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						data1[j * matrix1ActualRows + i] = std::ceil(data1[j * matrix1ActualRows + i]);
					}
				}
			}
		}
		else
		{
			if constexpr (thisContiguous)
			{
				size_t size = this->_size;
				size_t finalPosSize = this->finalPosSize;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);
					_mm256_store_pd(&data1[i], _mm256_ceil_pd(a));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] = std::ceil(data1[i]);
				}
			}
			else
			{
				size_t matrix1ActualCols = this->actualCols;

				size_t finalPosCols = this->finalPosCols;
				size_t finalPosRows = this->finalPosRows;

				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						_mm256_store_pd(&data1[i * matrix1ActualCols + j], _mm256_ceil_pd(a));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < rows; i++)
					{
						data1[i * matrix1ActualCols + j] = std::ceil(data1[i * matrix1ActualCols + j]);
					}
				}
			}
		}
	}

	// pow

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed>
	inline matrix<double> matrix<double, thisTransposed, thisContiguous>::pow(double num)
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		size_t size = this->_size;

		double* data1 = this->_data;

		__m256d b = _mm256_set1_pd(num);

		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;
		size_t finalPosSize = this->finalPosSize;

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix1ActualCols = this->actualCols;

		if constexpr (returnTransposed)
		{
			matrix<double> result(cols, rows);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&dataResult[i], _mm256_pow_pd(a, b));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = std::pow(data1[i], num);
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

							_mm256_store_pd(&dataResult[j * rows + i], _mm256_pow_pd(a, b));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < finalPosCols; j += 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
								data1[(j + 1) * matrix1ActualRows + i],
								data1[(j + 2) * matrix1ActualRows + i],
								data1[(j + 3) * matrix1ActualRows + i]);

							__m256d pow = _mm256_pow_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(pow, 1);
							__m128d val2 = _mm256_castpd256_pd128(pow);

							_mm_store_sd(&dataResult[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

							_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
						}
						for (size_t j = finalPosCols; j < cols; j++)
						{
							dataResult[j * rows + i] = std::pow(data1[j * matrix1ActualRows + i], num);
						}
					}
				}
			}
			else
			{
				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
							data1[(i + 1) * matrix1ActualCols + j],
							data1[(i + 2) * matrix1ActualCols + j],
							data1[(i + 3) * matrix1ActualCols + j]);

						_mm256_store_pd(&dataResult[j * rows + i], _mm256_pow_pd(a, b));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						__m256d pow = _mm256_pow_pd(a, b);

						__m128d val1 = _mm256_extractf128_pd(pow, 1);
						__m128d val2 = _mm256_castpd256_pd128(pow);

						_mm_store_sd(&dataResult[j * rows + i], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

						_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						dataResult[j * rows + i] = std::pow(data1[i * matrix1ActualCols + j], num);
					}
				}
			}
			return result;
		}
		else
		{
			matrix<double> result(rows, cols);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
							data1[(j + 1) * matrix1ActualRows + i],
							data1[(j + 2) * matrix1ActualRows + i],
							data1[(j + 3) * matrix1ActualRows + i]);

						_mm256_store_pd(&dataResult[i * cols + j], _mm256_pow_pd(a, b));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						__m256d pow = _mm256_pow_pd(a, b);

						__m128d val1 = _mm256_extractf128_pd(pow, 1);
						__m128d val2 = _mm256_castpd256_pd128(pow);

						_mm_store_sd(&dataResult[i * cols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

						_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						dataResult[i * cols + j] = std::pow(data1[j * matrix1ActualRows + i], num);
					}
				}
			}
			else
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&dataResult[i], _mm256_pow_pd(a, b));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = std::pow(data1[i], num);
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

							_mm256_store_pd(&dataResult[i * cols + j], _mm256_pow_pd(a, b));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < finalPosRows; i += 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
								data1[(i + 1) * matrix1ActualCols + j],
								data1[(i + 2) * matrix1ActualCols + j],
								data1[(i + 3) * matrix1ActualCols + j]);

							__m256d pow = _mm256_pow_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(pow, 1);
							__m128d val2 = _mm256_castpd256_pd128(pow);

							_mm_store_sd(&dataResult[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

							_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
						}
						for (size_t i = finalPosRows; i < rows; i++)
						{
							dataResult[i * cols + j] = std::pow(data1[i * matrix1ActualCols + j], num);
						}
					}
				}
			}
			return result;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed, bool otherTransposed, bool otherContiguous>
	inline matrix<double> matrix<double, thisTransposed, thisContiguous>::pow(const matrix<double, otherTransposed, otherContiguous>& other)
	{
#ifdef _DEBUG
		if (other._cols != this->_cols || other._rows != this->_rows) throw std::invalid_argument("The dimensions of both matrices must be the same");
#else
#endif

		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;
		double* data2 = other._data;

		size_t size = this->_size;

		size_t finalPosSize = this->finalPosSize;

		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix2ActualRows = other.actualRows;
		size_t matrix1ActualCols = this->actualCols;
		size_t matrix2ActualCols = other.actualCols;

		if constexpr (returnTransposed)
		{
			matrix<double> result(cols, rows);
			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (otherTransposed)
				{
					if constexpr (thisContiguous && otherContiguous)
					{
						for (size_t i = 0; i < finalPosSize; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);
							__m256d b = _mm256_load_pd(&data2[i]);

							_mm256_store_pd(&dataResult[i], _mm256_pow_pd(a, b));
						}
						for (size_t i = finalPosSize; i < size; i++)
						{
							dataResult[i] = std::pow(data1[i], data2[i]);
						}
					}
					else
					{
						for (size_t i = 0; i < finalPosRows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);
								__m256d b = _mm256_load_pd(&data2[j * matrix2ActualRows + i]);

								_mm256_store_pd(&dataResult[j * rows + i], _mm256_pow_pd(a, b));
							}
						}
						for (size_t i = finalPosRows; i < rows; i++)
						{
							for (size_t j = 0; j < finalPosCols; j += 4)
							{
								__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
									data1[(j + 1) * matrix1ActualRows + i],
									data1[(j + 2) * matrix1ActualRows + i],
									data1[(j + 3) * matrix1ActualRows + i]);
								__m256d b = _mm256_setr_pd(data2[j * matrix2ActualRows + i],
									data2[(j + 1) * matrix2ActualRows + i],
									data2[(j + 2) * matrix2ActualRows + i],
									data2[(j + 3) * matrix2ActualRows + i]);

								__m256d pow = _mm256_pow_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(pow, 1);
								__m128d val2 = _mm256_castpd256_pd128(pow);

								_mm_store_sd(&dataResult[j * rows + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

								_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
							}
							for (size_t j = finalPosCols; j < cols; j++)
							{
								dataResult[j * rows + i] = std::pow(data1[j * matrix1ActualRows + i], data2[j * matrix2ActualRows + i]);
							}
						}
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);
							__m256d b = _mm256_setr_pd(data2[i * matrix2ActualCols + j],
								data2[(i + 1) * matrix2ActualCols + j],
								data2[(i + 2) * matrix2ActualCols + j],
								data2[(i + 3) * matrix2ActualCols + j]);

							_mm256_store_pd(&dataResult[j * rows + i], _mm256_pow_pd(a, b));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < finalPosCols; j += 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
								data1[(j + 1) * matrix1ActualRows + i],
								data1[(j + 2) * matrix1ActualRows + i],
								data1[(j + 3) * matrix1ActualRows + i]);
							__m256d b = _mm256_load_pd(&data2[i * matrix2ActualCols + j]);

							__m256d pow = _mm256_pow_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(pow, 1);
							__m128d val2 = _mm256_castpd256_pd128(pow);

							_mm_store_sd(&dataResult[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

							_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
						}
						for (size_t j = finalPosCols; j < cols; j++)
						{
							dataResult[j * rows + i] = std::pow(data1[j * matrix1ActualRows + i], data2[i * matrix2ActualCols + j]);
						}
					}
				}
			}
			else
			{
				if constexpr (otherTransposed)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
								data1[(i + 1) * matrix1ActualCols + j],
								data1[(i + 2) * matrix1ActualCols + j],
								data1[(i + 3) * matrix1ActualCols + j]);
							__m256d b = _mm256_load_pd(&data2[j * matrix2ActualRows + i]);
							_mm256_store_pd(&dataResult[j * rows + i], _mm256_pow_pd(a, b));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < finalPosCols; j += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
							__m256d b = _mm256_setr_pd(data2[j * matrix2ActualRows + i],
								data2[(j + 1) * matrix2ActualRows + i],
								data2[(j + 2) * matrix2ActualRows + i],
								data2[(j + 3) * matrix2ActualRows + i]);

							__m256d pow = _mm256_pow_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(pow, 1);
							__m128d val2 = _mm256_castpd256_pd128(pow);

							_mm_store_sd(&dataResult[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

							_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
						}
						for (size_t j = finalPosCols; j < cols; j++)
						{
							dataResult[j * rows + i] = std::pow(data1[i * matrix1ActualCols + j], data2[j * matrix2ActualRows + i]);
						}
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
								data1[(i + 1) * matrix1ActualCols + j],
								data1[(i + 2) * matrix1ActualCols + j],
								data1[(i + 3) * matrix1ActualCols + j]);
							__m256d b = _mm256_setr_pd(data2[i * matrix2ActualCols + j],
								data2[(i + 1) * matrix2ActualCols + j],
								data2[(i + 2) * matrix2ActualCols + j],
								data2[(i + 3) * matrix2ActualCols + j]);
							_mm256_store_pd(&dataResult[j * rows + i], _mm256_pow_pd(a, b));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < finalPosCols; j += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
							__m256d b = _mm256_load_pd(&data2[i * matrix2ActualCols + j]);

							__m256d pow = _mm256_pow_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(pow, 1);
							__m128d val2 = _mm256_castpd256_pd128(pow);

							_mm_store_sd(&dataResult[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

							_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
						}
						for (size_t j = finalPosCols; j < cols; j++)
						{
							dataResult[j * rows + i] = std::pow(data1[i * matrix1ActualCols + j], data2[i * matrix2ActualCols + j]);
						}
					}
				}
			}
			return result;
		}
		else
		{
			matrix<double> result(rows, cols);
			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (otherTransposed)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
								data1[(j + 1) * matrix1ActualRows + i],
								data1[(j + 2) * matrix1ActualRows + i],
								data1[(j + 3) * matrix1ActualRows + i]);
							__m256d b = _mm256_setr_pd(data2[j * matrix2ActualRows + i],
								data2[(j + 1) * matrix2ActualRows + i],
								data2[(j + 2) * matrix2ActualRows + i],
								data2[(j + 3) * matrix2ActualRows + i]);

							_mm256_store_pd(&dataResult[i * cols + j], _mm256_pow_pd(a, b));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < finalPosRows; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);
							__m256d b = _mm256_load_pd(&data2[j * matrix2ActualRows + i]);

							__m256d pow = _mm256_pow_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(pow, 1);
							__m128d val2 = _mm256_castpd256_pd128(pow);

							_mm_store_sd(&dataResult[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

							_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
						}
						for (size_t i = finalPosRows; i < rows; i++)
						{
							dataResult[i * cols + j] = std::pow(data1[j * matrix1ActualRows + i], data2[j * matrix2ActualRows + i]);
						}
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
								data1[(j + 1) * matrix1ActualRows + i],
								data1[(j + 2) * matrix1ActualRows + i],
								data1[(j + 3) * matrix1ActualRows + i]);
							__m256d b = _mm256_load_pd(&data2[i * matrix2ActualCols + j]);

							_mm256_store_pd(&dataResult[i * cols + j], _mm256_pow_pd(a, b));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < finalPosRows; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);
							__m256d b = _mm256_setr_pd(data2[i * matrix2ActualCols + j],
								data2[(i + 1) * matrix2ActualCols + j],
								data2[(i + 2) * matrix2ActualCols + j],
								data2[(i + 3) * matrix2ActualCols + j]);

							__m256d pow = _mm256_pow_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(pow, 1);
							__m128d val2 = _mm256_castpd256_pd128(pow);

							_mm_store_sd(&dataResult[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

							_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
						}
						for (size_t i = finalPosRows; i < rows; i++)
						{
							dataResult[i * cols + j] = std::pow(data1[j * matrix1ActualRows + i], data2[i * matrix2ActualCols + j]);
						}
					}
				}
			}
			else
			{
				if constexpr (otherTransposed)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
							__m256d b = _mm256_setr_pd(data2[j * matrix2ActualRows + i],
								data2[(j + 1) * matrix2ActualRows + i],
								data2[(j + 2) * matrix2ActualRows + i],
								data2[(j + 3) * matrix2ActualRows + i]);
							_mm256_store_pd(&dataResult[i * cols + j], _mm256_pow_pd(a, b));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < finalPosRows; i += 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
								data1[(i + 1) * matrix1ActualCols + j],
								data1[(i + 2) * matrix1ActualCols + j],
								data1[(i + 3) * matrix1ActualCols + j]);
							__m256d b = _mm256_load_pd(&data2[j * matrix2ActualRows + i]);

							__m256d pow = _mm256_pow_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(pow, 1);
							__m128d val2 = _mm256_castpd256_pd128(pow);

							_mm_store_sd(&dataResult[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

							_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
						}
						for (size_t i = finalPosRows; i < rows; i++)
						{
							dataResult[i * cols + j] = std::pow(data1[i * matrix1ActualCols + j], data2[j * matrix2ActualRows + i]);
						}
					}
				}
				else
				{
					if constexpr (thisContiguous && otherContiguous)
					{
						for (size_t i = 0; i < finalPosSize; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);
							__m256d b = _mm256_load_pd(&data2[i]);

							_mm256_store_pd(&dataResult[i], _mm256_pow_pd(a, b));
						}
						for (size_t i = finalPosSize; i < size; i++)
						{
							dataResult[i] = std::pow(data1[i], data2[i]);
						}
					}
					else
					{
						size_t matrix1ActualCols = this->actualCols;
						size_t matrix2ActualCols = other.actualCols;

						size_t finalPosCols = this->finalPosCols;
						size_t finalPosRows = this->finalPosRows;

						for (size_t j = 0; j < finalPosCols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
								__m256d b = _mm256_load_pd(&data2[i * matrix2ActualCols + j]);

								_mm256_store_pd(&dataResult[i * cols + j], _mm256_pow_pd(a, b));
							}
						}
						for (size_t j = finalPosCols; j < cols; j++)
						{
							for (size_t i = 0; i < finalPosRows; i += 4)
							{
								__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
									data1[(i + 1) * matrix1ActualCols + j],
									data1[(i + 2) * matrix1ActualCols + j],
									data1[(i + 3) * matrix1ActualCols + j]);
								__m256d b = _mm256_setr_pd(data2[i * matrix2ActualCols + j],
									data2[(i + 1) * matrix2ActualCols + j],
									data2[(i + 2) * matrix2ActualCols + j],
									data2[(i + 3) * matrix2ActualCols + j]);

								__m256d pow = _mm256_pow_pd(a, b);

								__m128d val1 = _mm256_extractf128_pd(pow, 1);
								__m128d val2 = _mm256_castpd256_pd128(pow);

								_mm_store_sd(&dataResult[i * cols + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

								_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
							}
							for (size_t i = finalPosRows; i < rows; i++)
							{
								dataResult[i * cols + j] = std::pow(data1[i * matrix1ActualCols + j], data2[i * matrix2ActualCols + j]);
							}
						}
					}
				}
			}
			return result;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline void matrix<double, thisTransposed, thisContiguous>::self_pow(double num)
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		__m256d b = _mm256_set1_pd(num);

		if constexpr (thisTransposed)
		{
			if constexpr (thisContiguous)
			{
				size_t size = this->_size;

				size_t finalPosSize = this->finalPosSize;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);

					_mm256_store_pd(&data1[i], _mm256_pow_pd(a, b));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] = std::pow(data1[i], num);
				}
			}
			else
			{
				size_t finalPosRows = this->finalPosRows;
				size_t finalPosCols = this->finalPosCols;

				size_t matrix1ActualRows = this->actualRows;

				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						_mm256_store_pd(&data1[j * matrix1ActualRows + i], _mm256_pow_pd(a, b));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
							data1[(j + 1) * matrix1ActualRows + i],
							data1[(j + 2) * matrix1ActualRows + i],
							data1[(j + 3) * matrix1ActualRows + i]);

						__m256d pow = _mm256_pow_pd(a, b);

						__m128d val1 = _mm256_extractf128_pd(pow, 1);
						__m128d val2 = _mm256_castpd256_pd128(pow);

						_mm_store_sd(&data1[j * matrix1ActualRows + i], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&data1[(j + 1) * matrix1ActualRows + i], val2);

						_mm_store_sd(&data1[(j + 2) * matrix1ActualRows + i], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&data1[(j + 3) * matrix1ActualRows + i], val1);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						data1[j * matrix1ActualRows + i] = std::pow(data1[j * matrix1ActualRows + i], num);
					}
				}
			}
		}
		else
		{
			if constexpr (thisContiguous)
			{
				size_t finalPosSize = this->finalPosSize;
				size_t size = this->_size;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);

					_mm256_store_pd(&data1[i], _mm256_pow_pd(a, b));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] = std::pow(data1[i], num);
				}
			}
			else
			{
				size_t matrix1ActualCols = this->actualCols;

				size_t finalPosCols = this->finalPosCols;
				size_t finalPosRows = this->finalPosRows;

				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						_mm256_store_pd(&data1[i * matrix1ActualCols + j], _mm256_pow_pd(a, b));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
							data1[(i + 1) * matrix1ActualCols + j],
							data1[(i + 2) * matrix1ActualCols + j],
							data1[(i + 3) * matrix1ActualCols + j]);

						__m256d pow = _mm256_pow_pd(a, b);

						__m128d val1 = _mm256_extractf128_pd(pow, 1);
						__m128d val2 = _mm256_castpd256_pd128(pow);

						_mm_store_sd(&data1[i * matrix1ActualCols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&data1[(i + 1) * matrix1ActualCols + j], val2);

						_mm_store_sd(&data1[(i + 2) * matrix1ActualCols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&data1[(i + 3) * matrix1ActualCols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						data1[i * matrix1ActualCols + j] = std::pow(data1[i * matrix1ActualCols + j], num);
					}
				}
			}
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool otherTransposed, bool otherContiguous>
	inline void matrix<double, thisTransposed, thisContiguous>::self_pow(const matrix<double, otherTransposed, otherContiguous>& other)
	{
#ifdef _DEBUG
		if (other._cols != this->_cols || other._rows != this->_rows) throw std::invalid_argument("The dimensions of both matrices must be the same");
#else
#endif

		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;
		double* data2 = other._data;

		if constexpr (thisTransposed)
		{
			if constexpr (otherTransposed)
			{
				if constexpr (thisContiguous && otherContiguous)
				{
					size_t size = this->_size;

					size_t finalPosSize = this->finalPosSize;

					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);
						__m256d b = _mm256_load_pd(&data2[i]);

						_mm256_store_pd(&data1[i], _mm256_pow_pd(a, b));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						data1[i] = std::pow(data1[i], data2[i]);
					}
				}
				else
				{
					size_t finalPosRows = this->finalPosRows;
					size_t finalPosCols = this->finalPosCols;

					size_t matrix1ActualRows = this->actualRows;
					size_t matrix2ActualRows = other.actualRows;

					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);
							__m256d b = _mm256_load_pd(&data2[j * matrix2ActualRows + i]);

							_mm256_store_pd(&data1[j * matrix1ActualRows + i], _mm256_pow_pd(a, b));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < finalPosCols; j += 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
								data1[(j + 1) * matrix1ActualRows + i],
								data1[(j + 2) * matrix1ActualRows + i],
								data1[(j + 3) * matrix1ActualRows + i]);
							__m256d b = _mm256_setr_pd(data2[j * matrix2ActualRows + i],
								data2[(j + 1) * matrix2ActualRows + i],
								data2[(j + 2) * matrix2ActualRows + i],
								data2[(j + 3) * matrix2ActualRows + i]);

							__m256d pow = _mm256_pow_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(pow, 1);
							__m128d val2 = _mm256_castpd256_pd128(pow);

							_mm_store_sd(&data1[j * matrix1ActualRows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(j + 1) * matrix1ActualRows + i], val2);

							_mm_store_sd(&data1[(j + 2) * matrix1ActualRows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(j + 3) * matrix1ActualRows + i], val1);
						}
						for (size_t j = finalPosCols; j < cols; j++)
						{
							data1[j * matrix1ActualRows + i] = std::pow(data1[j * matrix1ActualRows + i], data2[j * matrix2ActualRows + i]);
						}
					}
				}
			}
			else
			{
				size_t finalPosRows = this->finalPosRows;
				size_t finalPosCols = this->finalPosCols;

				size_t matrix1ActualRows = this->actualRows;
				size_t matrix2ActualCols = other.actualCols;

				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);
						__m256d b = _mm256_setr_pd(data2[i * matrix2ActualCols + j],
							data2[(i + 1) * matrix2ActualCols + j],
							data2[(i + 2) * matrix2ActualCols + j],
							data2[(i + 3) * matrix2ActualCols + j]);

						_mm256_store_pd(&data1[j * matrix1ActualRows + i], _mm256_pow_pd(a, b));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
							data1[(j + 1) * matrix1ActualRows + i],
							data1[(j + 2) * matrix1ActualRows + i],
							data1[(j + 3) * matrix1ActualRows + i]);
						__m256d b = _mm256_load_pd(&data2[i * matrix2ActualCols + j]);

						__m256d pow = _mm256_pow_pd(a, b);

						__m128d val1 = _mm256_extractf128_pd(pow, 1);
						__m128d val2 = _mm256_castpd256_pd128(pow);

						_mm_store_sd(&data1[j * matrix1ActualRows + i], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&data1[(j + 1) * matrix1ActualRows + i], val2);

						_mm_store_sd(&data1[(j + 2) * matrix1ActualRows + i], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&data1[(j + 3) * matrix1ActualRows + i], val1);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						data1[j * matrix1ActualRows + i] = std::pow(data1[j * matrix1ActualRows + i], data2[i * matrix2ActualCols + j]);
					}
				}
			}
		}
		else
		{
			if constexpr (otherTransposed)
			{
				size_t matrix1ActualCols = this->actualCols;
				size_t matrix2ActualRows = other.actualRows;

				size_t finalPosCols = this->finalPosCols;
				size_t finalPosRows = this->finalPosRows;

				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
						__m256d b = _mm256_setr_pd(data2[j * matrix2ActualRows + i],
							data2[(j + 1) * matrix2ActualRows + i],
							data2[(j + 2) * matrix2ActualRows + i],
							data2[(j + 3) * matrix2ActualRows + i]);
						_mm256_store_pd(&data1[i * matrix1ActualCols + j], _mm256_pow_pd(a, b));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
							data1[(i + 1) * matrix1ActualCols + j],
							data1[(i + 2) * matrix1ActualCols + j],
							data1[(i + 3) * matrix1ActualCols + j]);
						__m256d b = _mm256_load_pd(&data2[j * matrix2ActualRows + i]);

						__m256d pow = _mm256_pow_pd(a, b);

						__m128d val1 = _mm256_extractf128_pd(pow, 1);
						__m128d val2 = _mm256_castpd256_pd128(pow);

						_mm_store_sd(&data1[i * matrix1ActualCols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&data1[(i + 1) * matrix1ActualCols + j], val2);

						_mm_store_sd(&data1[(i + 2) * matrix1ActualCols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&data1[(i + 3) * matrix1ActualCols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						data1[i * matrix1ActualCols + j] = std::pow(data1[i * matrix1ActualCols + j], data2[j * matrix2ActualRows + i]);
					}
				}
			}
			else
			{
				if constexpr (thisContiguous && otherContiguous)
				{
					size_t finalPosSize = this->finalPosSize;
					size_t size = this->_size;

					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);
						__m256d b = _mm256_load_pd(&data2[i]);

						_mm256_store_pd(&data1[i], _mm256_pow_pd(a, b));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						data1[i] = std::pow(data1[i], data2[i]);
					}
				}
				else
				{
					size_t matrix1ActualCols = this->actualCols;
					size_t matrix2ActualCols = other.actualCols;

					size_t finalPosCols = this->finalPosCols;
					size_t finalPosRows = this->finalPosRows;

					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
							__m256d b = _mm256_load_pd(&data2[i * matrix2ActualCols + j]);

							_mm256_store_pd(&data1[i * matrix1ActualCols + j], _mm256_pow_pd(a, b));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < finalPosRows; i += 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
								data1[(i + 1) * matrix1ActualCols + j],
								data1[(i + 2) * matrix1ActualCols + j],
								data1[(i + 3) * matrix1ActualCols + j]);
							__m256d b = _mm256_setr_pd(data2[i * matrix2ActualCols + j],
								data2[(i + 1) * matrix2ActualCols + j],
								data2[(i + 2) * matrix2ActualCols + j],
								data2[(i + 3) * matrix2ActualCols + j]);

							__m256d pow = _mm256_pow_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(pow, 1);
							__m128d val2 = _mm256_castpd256_pd128(pow);

							_mm_store_sd(&data1[i * matrix1ActualCols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(i + 1) * matrix1ActualCols + j], val2);

							_mm_store_sd(&data1[(i + 2) * matrix1ActualCols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(i + 3) * matrix1ActualCols + j], val1);
						}
						for (size_t i = finalPosRows; i < rows; i++)
						{
							data1[i * matrix1ActualCols + j] = std::pow(data1[i * matrix1ActualCols + j], data2[i * matrix2ActualCols + j]);
						}
					}
				}
			}
		}
	}

	// root

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed>
	inline matrix<double> matrix<double, thisTransposed, thisContiguous>::root(double num)
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		size_t size = this->_size;

		double* data1 = this->_data;

		num = 1 / num;

		__m256d b = _mm256_set1_pd(num);

		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;
		size_t finalPosSize = this->finalPosSize;

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix1ActualCols = this->actualCols;

		if constexpr (returnTransposed)
		{
			matrix<double> result(cols, rows);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&dataResult[i], _mm256_pow_pd(a, b));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = std::pow(data1[i], num);
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

							_mm256_store_pd(&dataResult[j * rows + i], _mm256_pow_pd(a, b));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < finalPosCols; j += 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
								data1[(j + 1) * matrix1ActualRows + i],
								data1[(j + 2) * matrix1ActualRows + i],
								data1[(j + 3) * matrix1ActualRows + i]);

							__m256d pow = _mm256_pow_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(pow, 1);
							__m128d val2 = _mm256_castpd256_pd128(pow);

							_mm_store_sd(&dataResult[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

							_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
						}
						for (size_t j = finalPosCols; j < cols; j++)
						{
							dataResult[j * rows + i] = std::pow(data1[j * matrix1ActualRows + i], num);
						}
					}
				}
			}
			else
			{
				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
							data1[(i + 1) * matrix1ActualCols + j],
							data1[(i + 2) * matrix1ActualCols + j],
							data1[(i + 3) * matrix1ActualCols + j]);

						_mm256_store_pd(&dataResult[j * rows + i], _mm256_pow_pd(a, b));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						__m256d pow = _mm256_pow_pd(a, b);

						__m128d val1 = _mm256_extractf128_pd(pow, 1);
						__m128d val2 = _mm256_castpd256_pd128(pow);

						_mm_store_sd(&dataResult[j * rows + i], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

						_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						dataResult[j * rows + i] = std::pow(data1[i * matrix1ActualCols + j], num);
					}
				}
			}
			return result;
		}
		else
		{
			matrix<double> result(rows, cols);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
							data1[(j + 1) * matrix1ActualRows + i],
							data1[(j + 2) * matrix1ActualRows + i],
							data1[(j + 3) * matrix1ActualRows + i]);

						_mm256_store_pd(&dataResult[i * cols + j], _mm256_pow_pd(a, b));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						__m256d pow = _mm256_pow_pd(a, b);

						__m128d val1 = _mm256_extractf128_pd(pow, 1);
						__m128d val2 = _mm256_castpd256_pd128(pow);

						_mm_store_sd(&dataResult[i * cols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

						_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						dataResult[i * cols + j] = std::pow(data1[j * matrix1ActualRows + i], num);
					}
				}
			}
			else
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&dataResult[i], _mm256_pow_pd(a, b));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = std::pow(data1[i], num);
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

							_mm256_store_pd(&dataResult[i * cols + j], _mm256_pow_pd(a, b));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < finalPosRows; i += 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
								data1[(i + 1) * matrix1ActualCols + j],
								data1[(i + 2) * matrix1ActualCols + j],
								data1[(i + 3) * matrix1ActualCols + j]);

							__m256d pow = _mm256_pow_pd(a, b);

							__m128d val1 = _mm256_extractf128_pd(pow, 1);
							__m128d val2 = _mm256_castpd256_pd128(pow);

							_mm_store_sd(&dataResult[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

							_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
						}
						for (size_t i = finalPosRows; i < rows; i++)
						{
							dataResult[i * cols + j] = std::pow(data1[i * matrix1ActualCols + j], num);
						}
					}
				}
			}
			return result;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed, bool otherTransposed, bool otherContiguous>
	inline matrix<double> matrix<double, thisTransposed, thisContiguous>::root(const matrix<double, otherTransposed, otherContiguous>& other)
	{
#ifdef _DEBUG
		if (other._cols != this->_cols || other._rows != this->_rows) throw std::invalid_argument("The dimensions of both matrices must be the same");
#else
#endif

		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;
		double* data2 = other._data;

		size_t size = this->_size;

		size_t finalPosSize = this->finalPosSize;

		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix2ActualRows = other.actualRows;
		size_t matrix1ActualCols = this->actualCols;
		size_t matrix2ActualCols = other.actualCols;

		__m256d one = _mm256_set1_pd(1.0);

		if constexpr (returnTransposed)
		{
			matrix<double> result(cols, rows);
			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (otherTransposed)
				{
					if constexpr (thisContiguous && otherContiguous)
					{
						for (size_t i = 0; i < finalPosSize; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);
							__m256d b = _mm256_load_pd(&data2[i]);

							_mm256_store_pd(&dataResult[i], _mm256_pow_pd(a, _mm256_div_pd(one, b)));
						}
						for (size_t i = finalPosSize; i < size; i++)
						{
							dataResult[i] = std::pow(data1[i], 1.0 / data2[i]);
						}
					}
					else
					{
						for (size_t i = 0; i < finalPosRows; i += 4)
						{
							for (size_t j = 0; j < cols; j++)
							{
								__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);
								__m256d b = _mm256_load_pd(&data2[j * matrix2ActualRows + i]);

								_mm256_store_pd(&dataResult[j * rows + i], _mm256_pow_pd(a, _mm256_div_pd(one, b)));
							}
						}
						for (size_t i = finalPosRows; i < rows; i++)
						{
							for (size_t j = 0; j < finalPosCols; j += 4)
							{
								__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
									data1[(j + 1) * matrix1ActualRows + i],
									data1[(j + 2) * matrix1ActualRows + i],
									data1[(j + 3) * matrix1ActualRows + i]);
								__m256d b = _mm256_setr_pd(data2[j * matrix2ActualRows + i],
									data2[(j + 1) * matrix2ActualRows + i],
									data2[(j + 2) * matrix2ActualRows + i],
									data2[(j + 3) * matrix2ActualRows + i]);

								__m256d pow = _mm256_pow_pd(a, _mm256_div_pd(one, b));

								__m128d val1 = _mm256_extractf128_pd(pow, 1);
								__m128d val2 = _mm256_castpd256_pd128(pow);

								_mm_store_sd(&dataResult[j * rows + i], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

								_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
							}
							for (size_t j = finalPosCols; j < cols; j++)
							{
								dataResult[j * rows + i] = std::pow(data1[j * matrix1ActualRows + i], 1.0 / data2[j * matrix2ActualRows + i]);
							}
						}
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);
							__m256d b = _mm256_setr_pd(data2[i * matrix2ActualCols + j],
								data2[(i + 1) * matrix2ActualCols + j],
								data2[(i + 2) * matrix2ActualCols + j],
								data2[(i + 3) * matrix2ActualCols + j]);

							_mm256_store_pd(&dataResult[j * rows + i], _mm256_pow_pd(a, _mm256_div_pd(one, b)));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < finalPosCols; j += 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
								data1[(j + 1) * matrix1ActualRows + i],
								data1[(j + 2) * matrix1ActualRows + i],
								data1[(j + 3) * matrix1ActualRows + i]);
							__m256d b = _mm256_load_pd(&data2[i * matrix2ActualCols + j]);

							__m256d pow = _mm256_pow_pd(a, _mm256_div_pd(one, b));

							__m128d val1 = _mm256_extractf128_pd(pow, 1);
							__m128d val2 = _mm256_castpd256_pd128(pow);

							_mm_store_sd(&dataResult[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

							_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
						}
						for (size_t j = finalPosCols; j < cols; j++)
						{
							dataResult[j * rows + i] = std::pow(data1[j * matrix1ActualRows + i], 1.0 / data2[i * matrix2ActualCols + j]);
						}
					}
				}
			}
			else
			{
				if constexpr (otherTransposed)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
								data1[(i + 1) * matrix1ActualCols + j],
								data1[(i + 2) * matrix1ActualCols + j],
								data1[(i + 3) * matrix1ActualCols + j]);
							__m256d b = _mm256_load_pd(&data2[j * matrix2ActualRows + i]);
							_mm256_store_pd(&dataResult[j * rows + i], _mm256_pow_pd(a, _mm256_div_pd(one, b)));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < finalPosCols; j += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
							__m256d b = _mm256_setr_pd(data2[j * matrix2ActualRows + i],
								data2[(j + 1) * matrix2ActualRows + i],
								data2[(j + 2) * matrix2ActualRows + i],
								data2[(j + 3) * matrix2ActualRows + i]);

							__m256d pow = _mm256_pow_pd(a, _mm256_div_pd(one, b));

							__m128d val1 = _mm256_extractf128_pd(pow, 1);
							__m128d val2 = _mm256_castpd256_pd128(pow);

							_mm_store_sd(&dataResult[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

							_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
						}
						for (size_t j = finalPosCols; j < cols; j++)
						{
							dataResult[j * rows + i] = std::pow(data1[i * matrix1ActualCols + j], 1.0 / data2[j * matrix2ActualRows + i]);
						}
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
								data1[(i + 1) * matrix1ActualCols + j],
								data1[(i + 2) * matrix1ActualCols + j],
								data1[(i + 3) * matrix1ActualCols + j]);
							__m256d b = _mm256_setr_pd(data2[i * matrix2ActualCols + j],
								data2[(i + 1) * matrix2ActualCols + j],
								data2[(i + 2) * matrix2ActualCols + j],
								data2[(i + 3) * matrix2ActualCols + j]);
							_mm256_store_pd(&dataResult[j * rows + i], _mm256_pow_pd(a, _mm256_div_pd(one, b)));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < finalPosCols; j += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
							__m256d b = _mm256_load_pd(&data2[i * matrix2ActualCols + j]);

							__m256d pow = _mm256_pow_pd(a, _mm256_div_pd(one, b));

							__m128d val1 = _mm256_extractf128_pd(pow, 1);
							__m128d val2 = _mm256_castpd256_pd128(pow);

							_mm_store_sd(&dataResult[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

							_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
						}
						for (size_t j = finalPosCols; j < cols; j++)
						{
							dataResult[j * rows + i] = std::pow(data1[i * matrix1ActualCols + j], 1.0 / data2[i * matrix2ActualCols + j]);
						}
					}
				}
			}
			return result;
		}
		else
		{
			matrix<double> result(rows, cols);
			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (otherTransposed)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
								data1[(j + 1) * matrix1ActualRows + i],
								data1[(j + 2) * matrix1ActualRows + i],
								data1[(j + 3) * matrix1ActualRows + i]);
							__m256d b = _mm256_setr_pd(data2[j * matrix2ActualRows + i],
								data2[(j + 1) * matrix2ActualRows + i],
								data2[(j + 2) * matrix2ActualRows + i],
								data2[(j + 3) * matrix2ActualRows + i]);

							_mm256_store_pd(&dataResult[i * cols + j], _mm256_pow_pd(a, _mm256_div_pd(one, b)));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < finalPosRows; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);
							__m256d b = _mm256_load_pd(&data2[j * matrix2ActualRows + i]);

							__m256d pow = _mm256_pow_pd(a, _mm256_div_pd(one, b));

							__m128d val1 = _mm256_extractf128_pd(pow, 1);
							__m128d val2 = _mm256_castpd256_pd128(pow);

							_mm_store_sd(&dataResult[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

							_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
						}
						for (size_t i = finalPosRows; i < rows; i++)
						{
							dataResult[i * cols + j] = std::pow(data1[j * matrix1ActualRows + i], 1.0 / data2[j * matrix2ActualRows + i]);
						}
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
								data1[(j + 1) * matrix1ActualRows + i],
								data1[(j + 2) * matrix1ActualRows + i],
								data1[(j + 3) * matrix1ActualRows + i]);
							__m256d b = _mm256_load_pd(&data2[i * matrix2ActualCols + j]);

							_mm256_store_pd(&dataResult[i * cols + j], _mm256_pow_pd(a, _mm256_div_pd(one, b)));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < finalPosRows; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);
							__m256d b = _mm256_setr_pd(data2[i * matrix2ActualCols + j],
								data2[(i + 1) * matrix2ActualCols + j],
								data2[(i + 2) * matrix2ActualCols + j],
								data2[(i + 3) * matrix2ActualCols + j]);

							__m256d pow = _mm256_pow_pd(a, _mm256_div_pd(one, b));

							__m128d val1 = _mm256_extractf128_pd(pow, 1);
							__m128d val2 = _mm256_castpd256_pd128(pow);

							_mm_store_sd(&dataResult[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

							_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
						}
						for (size_t i = finalPosRows; i < rows; i++)
						{
							dataResult[i * cols + j] = std::pow(data1[j * matrix1ActualRows + i], 1.0 / data2[i * matrix2ActualCols + j]);
						}
					}
				}
			}
			else
			{
				if constexpr (otherTransposed)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
							__m256d b = _mm256_setr_pd(data2[j * matrix2ActualRows + i],
								data2[(j + 1) * matrix2ActualRows + i],
								data2[(j + 2) * matrix2ActualRows + i],
								data2[(j + 3) * matrix2ActualRows + i]);
							_mm256_store_pd(&dataResult[i * cols + j], _mm256_pow_pd(a, _mm256_div_pd(one, b)));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < finalPosRows; i += 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
								data1[(i + 1) * matrix1ActualCols + j],
								data1[(i + 2) * matrix1ActualCols + j],
								data1[(i + 3) * matrix1ActualCols + j]);
							__m256d b = _mm256_load_pd(&data2[j * matrix2ActualRows + i]);

							__m256d pow = _mm256_pow_pd(a, _mm256_div_pd(one, b));

							__m128d val1 = _mm256_extractf128_pd(pow, 1);
							__m128d val2 = _mm256_castpd256_pd128(pow);

							_mm_store_sd(&dataResult[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

							_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
						}
						for (size_t i = finalPosRows; i < rows; i++)
						{
							dataResult[i * cols + j] = std::pow(data1[i * matrix1ActualCols + j], 1.0 / data2[j * matrix2ActualRows + i]);
						}
					}
				}
				else
				{
					if constexpr (thisContiguous && otherContiguous)
					{
						for (size_t i = 0; i < finalPosSize; i += 4)
						{
							__m256d a = _mm256_load_pd(&data1[i]);
							__m256d b = _mm256_load_pd(&data2[i]);

							_mm256_store_pd(&dataResult[i], _mm256_pow_pd(a, _mm256_div_pd(one, b)));
						}
						for (size_t i = finalPosSize; i < size; i++)
						{
							dataResult[i] = std::pow(data1[i], 1.0 / data2[i]);
						}
					}
					else
					{
						size_t matrix1ActualCols = this->actualCols;
						size_t matrix2ActualCols = other.actualCols;

						size_t finalPosCols = this->finalPosCols;
						size_t finalPosRows = this->finalPosRows;

						for (size_t j = 0; j < finalPosCols; j += 4)
						{
							for (size_t i = 0; i < rows; i++)
							{
								__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
								__m256d b = _mm256_load_pd(&data2[i * matrix2ActualCols + j]);

								_mm256_store_pd(&dataResult[i * cols + j], _mm256_pow_pd(a, _mm256_div_pd(one, b)));
							}
						}
						for (size_t j = finalPosCols; j < cols; j++)
						{
							for (size_t i = 0; i < finalPosRows; i += 4)
							{
								__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
									data1[(i + 1) * matrix1ActualCols + j],
									data1[(i + 2) * matrix1ActualCols + j],
									data1[(i + 3) * matrix1ActualCols + j]);
								__m256d b = _mm256_setr_pd(data2[i * matrix2ActualCols + j],
									data2[(i + 1) * matrix2ActualCols + j],
									data2[(i + 2) * matrix2ActualCols + j],
									data2[(i + 3) * matrix2ActualCols + j]);

								__m256d pow = _mm256_pow_pd(a, _mm256_div_pd(one, b));

								__m128d val1 = _mm256_extractf128_pd(pow, 1);
								__m128d val2 = _mm256_castpd256_pd128(pow);

								_mm_store_sd(&dataResult[i * cols + j], val2);
								val2 = _mm_shuffle_pd(val2, val2, 1);
								_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

								_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
								val1 = _mm_shuffle_pd(val1, val1, 1);
								_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
							}
							for (size_t i = finalPosRows; i < rows; i++)
							{
								dataResult[i * cols + j] = std::pow(data1[i * matrix1ActualCols + j], 1.0 / data2[i * matrix2ActualCols + j]);
							}
						}
					}
				}
			}
			return result;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline void matrix<double, thisTransposed, thisContiguous>::self_root(double num)
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		num = 1.0 / num;

		__m256d b = _mm256_set1_pd(num);

		if constexpr (thisTransposed)
		{
			if constexpr (thisContiguous)
			{
				size_t size = this->_size;

				size_t finalPosSize = this->finalPosSize;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);

					_mm256_store_pd(&data1[i], _mm256_pow_pd(a, b));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] = std::pow(data1[i], num);
				}
			}
			else
			{
				size_t finalPosRows = this->finalPosRows;
				size_t finalPosCols = this->finalPosCols;

				size_t matrix1ActualRows = this->actualRows;

				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						_mm256_store_pd(&data1[j * matrix1ActualRows + i], _mm256_pow_pd(a, b));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
							data1[(j + 1) * matrix1ActualRows + i],
							data1[(j + 2) * matrix1ActualRows + i],
							data1[(j + 3) * matrix1ActualRows + i]);

						__m256d pow = _mm256_pow_pd(a, b);

						__m128d val1 = _mm256_extractf128_pd(pow, 1);
						__m128d val2 = _mm256_castpd256_pd128(pow);

						_mm_store_sd(&data1[j * matrix1ActualRows + i], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&data1[(j + 1) * matrix1ActualRows + i], val2);

						_mm_store_sd(&data1[(j + 2) * matrix1ActualRows + i], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&data1[(j + 3) * matrix1ActualRows + i], val1);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						data1[j * matrix1ActualRows + i] = std::pow(data1[j * matrix1ActualRows + i], num);
					}
				}
			}
		}
		else
		{
			if constexpr (thisContiguous)
			{
				size_t finalPosSize = this->finalPosSize;
				size_t size = this->_size;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);

					_mm256_store_pd(&data1[i], _mm256_pow_pd(a, b));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] = std::pow(data1[i], num);
				}
			}
			else
			{
				size_t matrix1ActualCols = this->actualCols;

				size_t finalPosCols = this->finalPosCols;
				size_t finalPosRows = this->finalPosRows;

				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						_mm256_store_pd(&data1[i * matrix1ActualCols + j], _mm256_pow_pd(a, b));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
							data1[(i + 1) * matrix1ActualCols + j],
							data1[(i + 2) * matrix1ActualCols + j],
							data1[(i + 3) * matrix1ActualCols + j]);

						__m256d pow = _mm256_pow_pd(a, b);

						__m128d val1 = _mm256_extractf128_pd(pow, 1);
						__m128d val2 = _mm256_castpd256_pd128(pow);

						_mm_store_sd(&data1[i * matrix1ActualCols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&data1[(i + 1) * matrix1ActualCols + j], val2);

						_mm_store_sd(&data1[(i + 2) * matrix1ActualCols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&data1[(i + 3) * matrix1ActualCols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						data1[i * matrix1ActualCols + j] = std::pow(data1[i * matrix1ActualCols + j], num);
					}
				}
			}
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	template<bool otherTransposed, bool otherContiguous>
	inline void matrix<double, thisTransposed, thisContiguous>::self_root(const matrix<double, otherTransposed, otherContiguous>& other)
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

		if constexpr (thisTransposed)
		{
			if constexpr (otherTransposed)
			{
				if constexpr (thisContiguous && otherContiguous)
				{
					size_t size = this->_size;

					size_t finalPosSize = this->finalPosSize;

					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);
						__m256d b = _mm256_load_pd(&data2[i]);

						_mm256_store_pd(&data1[i], _mm256_pow_pd(a, _mm256_div_pd(one, b)));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						data1[i] = std::pow(data1[i], 1 / data2[i]);
					}
				}
				else
				{
					size_t finalPosRows = this->finalPosRows;
					size_t finalPosCols = this->finalPosCols;

					size_t matrix1ActualRows = this->actualRows;
					size_t matrix2ActualRows = other.actualRows;

					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);
							__m256d b = _mm256_load_pd(&data2[j * matrix2ActualRows + i]);

							_mm256_store_pd(&data1[j * matrix1ActualRows + i], _mm256_pow_pd(a, _mm256_div_pd(one, b)));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < finalPosCols; j += 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
								data1[(j + 1) * matrix1ActualRows + i],
								data1[(j + 2) * matrix1ActualRows + i],
								data1[(j + 3) * matrix1ActualRows + i]);
							__m256d b = _mm256_setr_pd(data2[j * matrix2ActualRows + i],
								data2[(j + 1) * matrix2ActualRows + i],
								data2[(j + 2) * matrix2ActualRows + i],
								data2[(j + 3) * matrix2ActualRows + i]);

							__m256d pow = _mm256_pow_pd(a, _mm256_div_pd(one, b));

							__m128d val1 = _mm256_extractf128_pd(pow, 1);
							__m128d val2 = _mm256_castpd256_pd128(pow);

							_mm_store_sd(&data1[j * matrix1ActualRows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(j + 1) * matrix1ActualRows + i], val2);

							_mm_store_sd(&data1[(j + 2) * matrix1ActualRows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(j + 3) * matrix1ActualRows + i], val1);
						}
						for (size_t j = finalPosCols; j < cols; j++)
						{
							data1[j * matrix1ActualRows + i] = std::pow(data1[j * matrix1ActualRows + i], 1.0 / data2[j * matrix2ActualRows + i]);
						}
					}
				}
			}
			else
			{
				size_t finalPosRows = this->finalPosRows;
				size_t finalPosCols = this->finalPosCols;

				size_t matrix1ActualRows = this->actualRows;
				size_t matrix2ActualCols = other.actualCols;

				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);
						__m256d b = _mm256_setr_pd(data2[i * matrix2ActualCols + j],
							data2[(i + 1) * matrix2ActualCols + j],
							data2[(i + 2) * matrix2ActualCols + j],
							data2[(i + 3) * matrix2ActualCols + j]);

						_mm256_store_pd(&data1[j * matrix1ActualRows + i], _mm256_pow_pd(a, _mm256_div_pd(one, b)));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
							data1[(j + 1) * matrix1ActualRows + i],
							data1[(j + 2) * matrix1ActualRows + i],
							data1[(j + 3) * matrix1ActualRows + i]);
						__m256d b = _mm256_load_pd(&data2[i * matrix2ActualCols + j]);

						__m256d pow = _mm256_pow_pd(a, _mm256_div_pd(one, b));

						__m128d val1 = _mm256_extractf128_pd(pow, 1);
						__m128d val2 = _mm256_castpd256_pd128(pow);

						_mm_store_sd(&data1[j * matrix1ActualRows + i], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&data1[(j + 1) * matrix1ActualRows + i], val2);

						_mm_store_sd(&data1[(j + 2) * matrix1ActualRows + i], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&data1[(j + 3) * matrix1ActualRows + i], val1);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						data1[j * matrix1ActualRows + i] = std::pow(data1[j * matrix1ActualRows + i], 1.0 / data2[i * matrix2ActualCols + j]);
					}
				}
			}
		}
		else
		{
			if constexpr (otherTransposed)
			{
				size_t matrix1ActualCols = this->actualCols;
				size_t matrix2ActualRows = other.actualRows;

				size_t finalPosCols = this->finalPosCols;
				size_t finalPosRows = this->finalPosRows;

				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
						__m256d b = _mm256_setr_pd(data2[j * matrix2ActualRows + i],
							data2[(j + 1) * matrix2ActualRows + i],
							data2[(j + 2) * matrix2ActualRows + i],
							data2[(j + 3) * matrix2ActualRows + i]);
						_mm256_store_pd(&data1[i * matrix1ActualCols + j], _mm256_pow_pd(a, _mm256_div_pd(one, b)));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
							data1[(i + 1) * matrix1ActualCols + j],
							data1[(i + 2) * matrix1ActualCols + j],
							data1[(i + 3) * matrix1ActualCols + j]);
						__m256d b = _mm256_load_pd(&data2[j * matrix2ActualRows + i]);

						__m256d pow = _mm256_pow_pd(a, _mm256_div_pd(one, b));

						__m128d val1 = _mm256_extractf128_pd(pow, 1);
						__m128d val2 = _mm256_castpd256_pd128(pow);

						_mm_store_sd(&data1[i * matrix1ActualCols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&data1[(i + 1) * matrix1ActualCols + j], val2);

						_mm_store_sd(&data1[(i + 2) * matrix1ActualCols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&data1[(i + 3) * matrix1ActualCols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						data1[i * matrix1ActualCols + j] = std::pow(data1[i * matrix1ActualCols + j], 1.0 / data2[j * matrix2ActualRows + i]);
					}
				}
			}
			else
			{
				if constexpr (thisContiguous && otherContiguous)
				{
					size_t finalPosSize = this->finalPosSize;
					size_t size = this->_size;

					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);
						__m256d b = _mm256_load_pd(&data2[i]);

						_mm256_store_pd(&data1[i], _mm256_pow_pd(a, _mm256_div_pd(one, b)));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						data1[i] = std::pow(data1[i], 1 / data2[i]);
					}
				}
				else
				{
					size_t matrix1ActualCols = this->actualCols;
					size_t matrix2ActualCols = other.actualCols;

					size_t finalPosCols = this->finalPosCols;
					size_t finalPosRows = this->finalPosRows;

					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
							__m256d b = _mm256_load_pd(&data2[i * matrix2ActualCols + j]);

							_mm256_store_pd(&data1[i * matrix1ActualCols + j], _mm256_pow_pd(a, _mm256_div_pd(one, b)));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < finalPosRows; i += 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
								data1[(i + 1) * matrix1ActualCols + j],
								data1[(i + 2) * matrix1ActualCols + j],
								data1[(i + 3) * matrix1ActualCols + j]);
							__m256d b = _mm256_setr_pd(data2[i * matrix2ActualCols + j],
								data2[(i + 1) * matrix2ActualCols + j],
								data2[(i + 2) * matrix2ActualCols + j],
								data2[(i + 3) * matrix2ActualCols + j]);

							__m256d pow = _mm256_pow_pd(a, _mm256_div_pd(one, b));

							__m128d val1 = _mm256_extractf128_pd(pow, 1);
							__m128d val2 = _mm256_castpd256_pd128(pow);

							_mm_store_sd(&data1[i * matrix1ActualCols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&data1[(i + 1) * matrix1ActualCols + j], val2);

							_mm_store_sd(&data1[(i + 2) * matrix1ActualCols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&data1[(i + 3) * matrix1ActualCols + j], val1);
						}
						for (size_t i = finalPosRows; i < rows; i++)
						{
							data1[i * matrix1ActualCols + j] = std::pow(data1[i * matrix1ActualCols + j], 1.0 / data2[i * matrix2ActualCols + j]);
						}
					}
				}
			}
		}
	}

	// Mean 

	template <bool thisTransposed, bool thisContiguous>
	inline vector<double> matrix<double, thisTransposed, thisContiguous>::mean_rowwise()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		vector<double> result(rows);

		double* dataResult = result._data;

		double cols_d = static_cast<double>(cols);

		__m256d _cols = _mm256_set1_pd(cols_d);

		if constexpr (thisTransposed)
		{
			size_t matrix1ActualRows = this->actualRows;

			size_t finalPosRows = this->finalPosRows;
			size_t finalPosCols = this->finalPosCols;

			for (size_t i = 0; i < finalPosRows; i += 4)
			{
				__m256d _sum = _mm256_setzero_pd();
				for (size_t j = 0; j < cols; j++)
				{
					_sum = _mm256_add_pd(_sum, _mm256_load_pd(&data1[j * matrix1ActualRows + i]));
				}
				_mm256_store_pd(&dataResult[i], _mm256_div_pd(_sum, _cols));
			}
			for (size_t i = finalPosRows; i < rows; i++)
			{
				double sum = 0.0;
				for (size_t j = 0; j < cols; j++)
				{
					sum += data1[j * matrix1ActualRows + i];
				}
				dataResult[i] = sum / cols_d;
			}
		}
		else
		{
			size_t matrix1ActualCols = this->actualCols;

			size_t finalPosRows = this->finalPosRows;
			size_t finalPosCols = this->finalPosCols;

			for (size_t i = 0; i < rows; i++)
			{
				__m256d _sum = _mm256_setzero_pd();
				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					_sum = _mm256_add_pd(_sum, _mm256_load_pd(data1[i * matrix1ActualCols + j]));
				}
				__m256d hadd = _mm256_hadd_pd(_sum, _sum);
				__m128d high = _mm256_extractf128_pd(hadd, 1);
				__m128d low = _mm256_castpd256_pd128(hadd);
				long long sumLong = _mm_extract_epi64(_mm_castpd_si128(_mm_add_pd(high, low)), 0);

				double sum = reinterpret_cast<double&>(sumLong);

				for (size_t j = finalPosCols; j < cols; j++)
				{
					sum += data1[i * matrix1ActualCols + j];
				}
				dataResult[i] = sum / cols_d;
			}
		}
		return result;
	}

	template <bool thisTransposed, bool thisContiguous>
	inline vector<double> matrix<double, thisTransposed, thisContiguous>::mean_colwise()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		vector<double> result(cols);

		double* dataResult = result._data;

		double rows_d = static_cast<double>(rows);

		__m256d _rows = _mm256_set1_pd(rows_d);

		if constexpr (thisTransposed)
		{
			size_t matrix1ActualRows = this->actualRows;

			size_t finalPosCols = this->finalPosCols;
			size_t finalPosRows = this->finalPosRows;

			for (size_t j = 0; j < cols; j++)
			{
				__m256d _sum = _mm256_setzero_pd();

				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					_sum = _mm256_add_pd(_sum, _mm256_load_pd(&data1[j * matrix1ActualRows + i]));
				}
				__m256d hadd = _mm256_hadd_pd(_sum, _sum);
				__m128d high = _mm256_extractf128_pd(hadd, 1);
				__m128d low = _mm256_castpd256_pd128(hadd);
				long long sumLong = _mm_extract_epi64(_mm_castpd_si128(_mm_add_pd(high, low)), 0);

				double sum = reinterpret_cast<double&>(sumLong);

				for (size_t i = finalPosRows; i < rows; i++)
				{
					sum += data1[j * matrix1ActualRows + i];
				}
				dataResult[j] = sum / rows_d;
			}
		}
		else
		{
			size_t matrix1ActualCols = this->actualCols;

			size_t finalPosCols = this->finalPosCols;
			size_t finalPosRows = this->finalPosRows;

			for (size_t j = 0; j < finalPosCols; j += 4)
			{
				__m256d _sum = _mm256_setzero_pd();
				for (size_t i = 0; i < rows; i++)
				{
					_sum = _mm256_add_pd(_sum, _mm256_load_pd(&data1[i * matrix1ActualCols + j]));
				}
				_mm256_store_pd(&dataResult[j], _mm256_div_pd(_sum, _rows));
			}
			for (size_t j = finalPosCols; j < cols; j++)
			{
				double sum = 0.0;

				for (size_t i = 0; i < rows; i++)
				{
					sum += data1[i * matrix1ActualCols + j];
				}
				dataResult[j] = sum / rows_d;
			}
		}
		return result;
	}

	template <bool thisTransposed, bool thisContiguous>
	inline double matrix<double, thisTransposed, thisContiguous>::mean_all()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		size_t size = this->_size;

		double* data1 = this->_data;

		__m256d _sum = _mm256_setzero_pd();
		double sum = 0;

		if constexpr (thisContiguous)
		{
			size_t finalPosSize = this->finalPosSize;

			for (size_t i = 0; i < finalPosSize; i += 4)
			{
				_sum = _mm256_add_pd(_sum, _mm256_load_pd(&data1[i]));
			}
			for (size_t i = finalPosSize; i < size; i++)
			{
				sum += data1[i];
			}
		}
		else if constexpr (thisTransposed)
		{
			size_t matrix1ActualRows = this->actualRows;

			size_t finalPosRows = this->finalPosRows;
			size_t finalPosCols = this->finalPosCols;

			for (size_t i = 0; i < finalPosRows; i += 4)
			{
				for (size_t j = 0; j < cols; j++)
				{
					_sum = _mm256_add_pd(_sum, _mm256_load_pd(&data1[j * matrix1ActualRows + i]));
				}
			}
			for (size_t i = finalPosRows; i < rows; i++)
			{
				for (size_t j = 0; j < cols; j++)
				{
					sum += data1[j * matrix1ActualRows + i];
				}
			}
		}
		else
		{
			size_t matrix1ActualCols = this->actualCols;

			size_t finalPosCols = this->finalPosCols;
			size_t finalPosRows = this->finalPosRows;

			for (size_t j = 0; j < finalPosCols; j += 4)
			{
				for (size_t i = 0; i < rows; i++)
				{
					_sum = _mm256_add_pd(_sum, _mm256_load_pd(&data1[i * matrix1ActualCols + j]));
				}
			}
			for (size_t j = finalPosCols; j < cols; j++)
			{
				for (size_t i = 0; i < rows; i++)
				{
					sum += data1[i * matrix1ActualCols + j];
				}
			}
		}

		__m256d hadd = _mm256_hadd_pd(_sum, _sum);
		__m128d high = _mm256_extractf128_pd(hadd, 1);
		__m128d low = _mm256_castpd256_pd128(hadd);
		long long sumLong = _mm_extract_epi64(_mm_castpd_si128(_mm_add_pd(high, low)), 0);

		double sum = reinterpret_cast<double&>(sumLong);

		return sum / static_cast<double>(size);
	}

	template <bool thisTransposed, bool thisContiguous>
	template<char axis>
	inline std::conditional<axis == 'a', double, vector<double>> matrix<double, thisTransposed, thisContiguous>::mean()
	{
		if constexpr (axis != 'a' && axis != 'r' && axis != 'c')
		{
			std::cerr << "Valid parameters for the axis are 'a', 'r', 'c'" << std::endl;
			exit(1);
		}

		if constexpr (axis == 'a')
		{
			return this->mean_all();
		}
		else if constexpr (axis == 'r')
		{
			return this->mean_rowwise();
		}
		else if constexpr (axis == 'c')
		{
			return this->mean_colwise();
		}
	}

	// Sum

	template <bool thisTransposed, bool thisContiguous>
	inline vector<double> matrix<double, thisTransposed, thisContiguous>::sum_rowwise()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		vector<double> result(rows);

		double* dataResult = result._data;

		double cols_d = static_cast<double>(cols);

		__m256d _cols = _mm256_set1_pd(cols_d);

		if constexpr (thisTransposed)
		{
			size_t matrix1ActualRows = this->actualRows;

			size_t finalPosRows = this->finalPosRows;
			size_t finalPosCols = this->finalPosCols;

			for (size_t i = 0; i < finalPosRows; i += 4)
			{
				__m256d _sum = _mm256_setzero_pd();
				for (size_t j = 0; j < cols; j++)
				{
					_sum = _mm256_add_pd(_sum, _mm256_load_pd(&data1[j * matrix1ActualRows + i]));
				}
				_mm256_store_pd(&dataResult[i], _sum);
			}
			for (size_t i = finalPosRows; i < rows; i++)
			{
				double sum = 0.0;
				for (size_t j = 0; j < cols; j++)
				{
					sum += data1[j * matrix1ActualRows + i];
				}
				dataResult[i] = sum;
			}
		}
		else
		{
			size_t matrix1ActualCols = this->actualCols;

			size_t finalPosRows = this->finalPosRows;
			size_t finalPosCols = this->finalPosCols;

			for (size_t i = 0; i < rows; i++)
			{
				__m256d _sum = _mm256_setzero_pd();
				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					_sum = _mm256_add_pd(_sum, _mm256_load_pd(data1[i * matrix1ActualCols + j]));
				}
				__m256d hadd = _mm256_hadd_pd(_sum, _sum);
				__m128d high = _mm256_extractf128_pd(hadd, 1);
				__m128d low = _mm256_castpd256_pd128(hadd);
				long long sumLong = _mm_extract_epi64(_mm_castpd_si128(_mm_add_pd(high, low)), 0);

				double sum = reinterpret_cast<double&>(sumLong);

				for (size_t j = finalPosCols; j < cols; j++)
				{
					sum += data1[i * matrix1ActualCols + j];
				}
				dataResult[i] = sum;
			}
		}
		return result;
	}

	template <bool thisTransposed, bool thisContiguous>
	inline vector<double> matrix<double, thisTransposed, thisContiguous>::sum_colwise()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		vector<double> result(cols);

		double* dataResult = result._data;

		if constexpr (thisTransposed)
		{
			size_t matrix1ActualRows = this->actualRows;

			size_t finalPosCols = this->finalPosCols;
			size_t finalPosRows = this->finalPosRows;

			for (size_t j = 0; j < cols; j++)
			{
				__m256d _sum = _mm256_setzero_pd();

				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					_sum = _mm256_add_pd(_sum, _mm256_load_pd(&data1[j * matrix1ActualRows + i]));
				}
				__m256d hadd = _mm256_hadd_pd(_sum, _sum);
				__m128d high = _mm256_extractf128_pd(hadd, 1);
				__m128d low = _mm256_castpd256_pd128(hadd);
				long long sumLong = _mm_extract_epi64(_mm_castpd_si128(_mm_add_pd(high, low)), 0);

				double sum = reinterpret_cast<double&>(sumLong);

				for (size_t i = finalPosRows; i < rows; i++)
				{
					sum += data1[j * matrix1ActualRows + i];
				}
				dataResult[j] = sum;
			}
		}
		else
		{
			size_t matrix1ActualCols = this->actualCols;

			size_t finalPosCols = this->finalPosCols;
			size_t finalPosRows = this->finalPosRows;

			for (size_t j = 0; j < finalPosCols; j += 4)
			{
				__m256d _sum = _mm256_setzero_pd();
				for (size_t i = 0; i < rows; i++)
				{
					_sum = _mm256_add_pd(_sum, _mm256_load_pd(&data1[i * matrix1ActualCols + j]));
				}
				_mm256_store_pd(&dataResult[j], _sum);
			}
			for (size_t j = finalPosCols; j < cols; j++)
			{
				double sum = 0.0;

				for (size_t i = 0; i < rows; i++)
				{
					sum += data1[i * matrix1ActualCols + j];
				}
				dataResult[j] = sum;
			}
		}
		return result;
	}

	template <bool thisTransposed, bool thisContiguous>
	inline double matrix<double, thisTransposed, thisContiguous>::sum_all()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		size_t size = this->_size;

		double* data1 = this->_data;

		__m256d _sum = _mm256_setzero_pd();
		double sum = 0;

		if constexpr (thisContiguous)
		{
			size_t finalPosSize = this->finalPosSize;

			for (size_t i = 0; i < finalPosSize; i += 4)
			{
				_sum = _mm256_add_pd(_sum, _mm256_load_pd(&data1[i]));
			}
			for (size_t i = finalPosSize; i < size; i++)
			{
				sum += data1[i];
			}
		}
		else if constexpr (thisTransposed)
		{
			size_t matrix1ActualRows = this->actualRows;

			size_t finalPosRows = this->finalPosRows;
			size_t finalPosCols = this->finalPosCols;

			for (size_t i = 0; i < finalPosRows; i += 4)
			{
				for (size_t j = 0; j < cols; j++)
				{
					_sum = _mm256_add_pd(_sum, _mm256_load_pd(&data1[j * matrix1ActualRows + i]));
				}
			}
			for (size_t i = finalPosRows; i < rows; i++)
			{
				for (size_t j = 0; j < cols; j++)
				{
					sum += data1[j * matrix1ActualRows + i];
				}
			}
		}
		else
		{
			size_t matrix1ActualCols = this->actualCols;

			size_t finalPosCols = this->finalPosCols;
			size_t finalPosRows = this->finalPosRows;

			for (size_t j = 0; j < finalPosCols; j += 4)
			{
				for (size_t i = 0; i < rows; i++)
				{
					_sum = _mm256_add_pd(_sum, _mm256_load_pd(&data1[i * matrix1ActualCols + j]));
				}
			}
			for (size_t j = finalPosCols; j < cols; j++)
			{
				for (size_t i = 0; i < rows; i++)
				{
					sum += data1[i * matrix1ActualCols + j];
				}
			}
		}

		__m256d hadd = _mm256_hadd_pd(_sum, _sum);
		__m128d high = _mm256_extractf128_pd(hadd, 1);
		__m128d low = _mm256_castpd256_pd128(hadd);
		long long sumLong = _mm_extract_epi64(_mm_castpd_si128(_mm_add_pd(high, low)), 0);

		sum += reinterpret_cast<double&>(sumLong);

		return sum;
	}

	template <bool thisTransposed, bool thisContiguous>
	template<char axis>
	inline std::conditional<axis == 'a', double, vector<double>> matrix<double, thisTransposed, thisContiguous>::sum()
	{
		if constexpr (axis != 'a' && axis != 'r' && axis != 'c')
		{
			std::cerr << "Valid parameters for the axis are 'a', 'r', 'c'" << std::endl;
			exit(1);
		}

		if constexpr (axis == 'a')
		{
			return this->sum_all();
		}
		else if constexpr (axis == 'r')
		{
			return this->sum_rowwise();
		}
		else if constexpr (axis == 'c')
		{
			return this->sum_colwise();
		}
	}

	// Std

	template <bool thisTransposed, bool thisContiguous>
	inline vector<double> matrix<double, thisTransposed, thisContiguous>::std_rowwise(double ddof)
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		vector<double> result(rows);

		double* dataResult = result._data;

		double cols_d = static_cast<double>(cols);

		__m256d _cols = _mm256_set1_pd(cols_d);
		__m256d _ddof = _mm256_set1_pd(ddof);

		if constexpr (thisTransposed)
		{
			size_t matrix1ActualRows = this->actualRows;

			size_t finalPosRows = this->finalPosRows;
			size_t finalPosCols = this->finalPosCols;

			for (size_t i = 0; i < finalPosRows; i += 4)
			{
				__m256d _sum = _mm256_setzero_pd();
				__m256d _sumSquare = _mm256_setzero_pd();
				for (size_t j = 0; j < cols; j++)
				{
					__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

					_sum = _mm256_add_pd(_sum, a);
					_sumSquare = _mm256_fmadd_pd(a, a, _sumSquare);
				}
				__m256d variance = _mm256_div_pd(_mm256_sub_pd(_sumSquare, _mm256_div_pd(_mm256_mul_pd(_sum, _sum), _cols)), _mm256_sub_pd(_cols, _ddof));
				_mm256_store_pd(&dataResult[i], _mm256_sqrt_pd(variance));
			}
			for (size_t i = finalPosRows; i < rows; i++)
			{
				double sum = 0.0;
				double sumSquare = 0.0;

				for (size_t j = finalPosCols; j < cols; j++)
				{
					double data = data1[j * matrix1ActualRows + i];
					sum += data;
					sumSquare += data * data;
				}
				double variance = (sumSquare - (sum * sum / cols_d)) / (cols_d - ddof);
				double std = std::sqrt(variance);
				dataResult[i] = std;
			}
		}
		else
		{
			size_t matrix1ActualCols = this->actualCols;

			size_t finalPosRows = this->finalPosRows;
			size_t finalPosCols = this->finalPosCols;

			for (size_t i = finalPosRows; i < rows; i++)
			{
				__m256d _sum = _mm256_setzero_pd();
				__m256d _sumSquare = _mm256_setzero_pd();
				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
					_sum = _mm256_add_pd(_sum, a);
					_sumSquare = _mm256_fmadd_pd(a, a, _sumSquare);
				}

				__m256d hadd = _mm256_hadd_pd(_sum, _sum);
				__m128d high = _mm256_extractf128_pd(hadd, 1);
				__m128d low = _mm256_castpd256_pd128(hadd);
				long long sumLong = _mm_extract_epi64(_mm_castpd_si128(_mm_add_pd(high, low)), 0);

				double sum = reinterpret_cast<double&>(sumLong);
				//--
				hadd = _mm256_hadd_pd(_sumSquare, _sumSquare);
				high = _mm256_extractf128_pd(hadd, 1);
				low = _mm256_castpd256_pd128(hadd);
				long long sumSquareLong = _mm_extract_epi64(_mm_castpd_si128(_mm_add_pd(high, low)), 0);

				double sumSquare = reinterpret_cast<double&>(sumSquareLong);

				for (size_t j = finalPosCols; j < cols; j++)
				{
					double data = data1[i * matrix1ActualCols + j];
					sum += data;
					sumSquare += data * data;
				}
				double variance = (sumSquare - (sum * sum / cols_d)) / (cols_d - ddof);
				double std = std::sqrt(variance);
				dataResult[i] = std;
			}
		}
		return result;
	}

	template <bool thisTransposed, bool thisContiguous>
	inline vector<double> matrix<double, thisTransposed, thisContiguous>::std_colwise(double ddof)
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		vector<double> result(cols);

		double* dataResult = result._data;

		double rows_d = static_cast<double>(rows);

		__m256d _rows = _mm256_set1_pd(rows_d);
		__m256d _ddof = _mm256_set1_pd(ddof);

		if constexpr (thisTransposed)
		{
			size_t matrix1ActualRows = this->actualRows;

			size_t finalPosCols = this->finalPosCols;
			size_t finalPosRows = this->finalPosRows;

			for (size_t j = finalPosCols; j < cols; j++)
			{
				__m256d _sum = _mm256_setzero_pd();
				__m256d _sumSquare = _mm256_setzero_pd();

				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

					_sum = _mm256_add_pd(_sum, a);
					_sumSquare = _mm256_fmadd_pd(a, a, _sumSquare);
				}
				__m256d hadd = _mm256_hadd_pd(_sum, _sum);
				__m128d high = _mm256_extractf128_pd(hadd, 1);
				__m128d low = _mm256_castpd256_pd128(hadd);
				long long sumLong = _mm_extract_epi64(_mm_castpd_si128(_mm_add_pd(high, low)), 0);

				double sum = reinterpret_cast<double&>(sumLong);
				//--
				hadd = _mm256_hadd_pd(_sumSquare, _sumSquare);
				high = _mm256_extractf128_pd(hadd, 1);
				low = _mm256_castpd256_pd128(hadd);
				long long sumSquareLong = _mm_extract_epi64(_mm_castpd_si128(_mm_add_pd(high, low)), 0);

				double sumSquare = reinterpret_cast<double&>(sumSquareLong);
				for (size_t i = finalPosRows; i < rows; i++)
				{
					double data = data1[j * matrix1ActualRows + i];
					sum += data;
					sumSquare += data * data;
				}
				double variance = (sumSquare - (sum * sum / rows_d)) / (rows_d - ddof);
				double std = std::sqrt(variance);
				dataResult[j] = std;
			}
		}
		else
		{
			size_t matrix1ActualCols = this->actualCols;

			size_t finalPosCols = this->finalPosCols;
			size_t finalPosRows = this->finalPosRows;

			for (size_t j = 0; j < finalPosCols; j += 4)
			{
				__m256d _sum = _mm256_setzero_pd();
				__m256d _sumSquare = _mm256_setzero_pd();
				for (size_t i = 0; i < rows; i++)
				{
					__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

					_sum = _mm256_add_pd(_sum, a);
					_sumSquare = _mm256_fmadd_pd(a, a, _sumSquare);
				}
				__m256d variance = _mm256_div_pd(_mm256_sub_pd(_sumSquare, _mm256_div_pd(_mm256_mul_pd(_sum, _sum), _rows)), _mm256_sub_pd(_rows, _ddof));
				_mm256_store_pd(&dataResult[j], _mm256_sqrt_pd(variance));
			}
			for (size_t j = finalPosCols; j < cols; j++)
			{
				double sum = 0.0;
				double sumSquare = 0.0;

				for (size_t i = finalPosRows; i < rows; i++)
				{
					double data = data1[i * matrix1ActualCols + j];
					sum += data;
					sumSquare += data * data;
				}
				double variance = (sumSquare - (sum * sum / rows_d)) / (rows_d - ddof);
				double std = std::sqrt(variance);
				dataResult[j] = std;
			}
		}
		return result;
	}

	template <bool thisTransposed, bool thisContiguous>
	inline double matrix<double, thisTransposed, thisContiguous>::std_all(double ddof, double* mean)
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

		if constexpr (thisContiguous)
		{
			size_t size = this->_size;
			size_t finalPosSize = this->finalPosSize;

			for (size_t i = 0; i < finalPosSize; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);

				_sum = _mm256_add_pd(_sum, a);
				_sumSquare = _mm256_fmadd_pd(a, a, _sumSquare);
			}
			for (size_t i = finalPosSize; i < size; i++)
			{
				double data = data1[i];
				sum += data;
				sumSquare += data * data;
			}
		}
		else if constexpr (thisTransposed)
		{
			size_t matrix1ActualRows = this->actualRows;

			size_t finalPosRows = this->finalPosRows;
			size_t finalPosCols = this->finalPosCols;

			for (size_t i = 0; i < finalPosRows; i += 4)
			{
				for (size_t j = 0; j < cols; j++)
				{
					__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

					_sum = _mm256_add_pd(_sum, a);
					_sumSquare = _mm256_fmadd_pd(a, a, _sumSquare);
				}
			}
			for (size_t i = finalPosRows; i < rows; i++)
			{
				for (size_t j = 0; j < cols; j++)
				{
					double data = data1[j * matrix1ActualRows + i];
					sum += data;
					sumSquare += data * data;
				}
			}
		}
		else
		{
			size_t matrix1ActualCols = this->actualCols;

			size_t finalPosCols = this->finalPosCols;
			size_t finalPosRows = this->finalPosRows;

			for (size_t j = 0; j < finalPosCols; j += 4)
			{
				for (size_t i = 0; i < rows; i++)
				{
					__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);
					_sum = _mm256_add_pd(_sum, a);
					_sumSquare = _mm256_fmadd_pd(a, a, _sumSquare);
				}
			}
			for (size_t j = finalPosCols; j < cols; j++)
			{
				for (size_t i = 0; i < rows; i++)
				{
					double data = data1[i * matrix1ActualCols + j];
					sum += data;
					sumSquare += data * data;
				}
			}
		}

		__m256d hadd = _mm256_hadd_pd(_sum, _sum);
		__m128d high = _mm256_extractf128_pd(hadd, 1);
		__m128d low = _mm256_castpd256_pd128(hadd);
		long long sumLong = _mm_extract_epi64(_mm_castpd_si128(_mm_add_pd(high, low)), 0);

		sum += reinterpret_cast<double&>(sumLong);
		//--
		hadd = _mm256_hadd_pd(_sumSquare, _sumSquare);
		high = _mm256_extractf128_pd(hadd, 1);
		low = _mm256_castpd256_pd128(hadd);
		long long sumSquareLong = _mm_extract_epi64(_mm_castpd_si128(_mm_add_pd(high, low)), 0);

		sumSquare += reinterpret_cast<double&>(sumSquareLong);

		if (mean != nullptr) *mean = sum / size_d;

		double variance = (sumSquare - (sum * sum / size_d)) / (size_d - ddof);
		double std = std::sqrt(variance);
		return std;
	}

	template <bool thisTransposed, bool thisContiguous>
	template<char axis>
	inline std::conditional<axis == 'a', double, vector<double>> matrix<double, thisTransposed, thisContiguous>::std(double ddof, double* mean)
	{
		if constexpr (axis != 'a' && axis != 'r' && axis != 'c')
		{
			std::cerr << "Valid parameters for the axis are 'a', 'r', 'c'" << std::endl;
			exit(1);
		}

		if constexpr (axis == 'a')
		{
			return this->std_all(ddof, mean);
		}
		else if constexpr (axis == 'r')
		{
			return this->std_rowwise(ddof);
		}
		else if constexpr (axis == 'c')
		{
			return this->std_colwise(ddof);
		}
	}

	// Min

	template <bool thisTransposed, bool thisContiguous>
	inline vector<double> matrix<double, thisTransposed, thisContiguous>::min_rowwise()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		vector<double> result(rows);

		double* dataResult = result._data;

		if constexpr (thisTransposed)
		{
			size_t matrix1ActualRows = this->actualRows;

			size_t finalPosRows = this->finalPosRows;
			size_t finalPosCols = this->finalPosCols;

			for (size_t i = 0; i < finalPosRows; i += 4)
			{
				__m256d _min = _mm256_set1_pd(DBL_MAX);
				for (size_t j = 0; j < cols; j++)
				{
					_min = _mm256_min_pd(_min, _mm256_load_pd(&data1[j * matrix1ActualRows + i]));
				}
				_mm256_store_pd(&dataResult[i], _min);
			}
			for (size_t i = finalPosRows; i < rows; i++)
			{
				double min = DBL_MAX;

				for (size_t j = 0; j < cols; j++)
				{
					double data = data1[j * matrix1ActualRows + i];
					if (data < min) min = data;
				}
				dataResult[i] = min;
			}
		}
		else
		{
			size_t matrix1ActualCols = this->actualCols;

			size_t finalPosRows = this->finalPosRows;
			size_t finalPosCols = this->finalPosCols;

			for (size_t i = 0; i < rows; i++)
			{
				__m256d _min = _mm256_set1_pd(DBL_MAX);
				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					_min = _mm256_min_pd(_min, _mm256_load_pd(data1[i * matrix1ActualCols + j]));
				}
				__m256d tempMin = _mm256_permute2f128_pd(_min, _min, 0x01);
				_min = _mm256_min_pd(_min, tempMin);

				__m128d low = _mm256_castpd256_pd128(_min);
				__m128d high = _mm256_extractf128_pd(_min, 1);

				low = _mm_min_pd(low, high);
				long long minLong = _mm_extract_epi64(low, 0);
				double min = reinterpret_cast<double&>(minLong);

				for (size_t j = finalPosCols; j < cols; j++)
				{
					double data = data1[i * matrix1ActualCols + j];
					if (data < min) min = data;
				}
				dataResult[i] = min;
			}
		}
		return result;
	}

	template <bool thisTransposed, bool thisContiguous>
	inline vector<double> matrix<double, thisTransposed, thisContiguous>::min_colwise()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		vector<double> result(cols);

		double* dataResult = result._data;

		if constexpr (thisTransposed)
		{
			size_t matrix1ActualRows = this->actualRows;

			size_t finalPosCols = this->finalPosCols;
			size_t finalPosRows = this->finalPosRows;

			for (size_t j = 0; j < cols; j++)
			{
				__m256d _min = _mm256_set1_pd(DBL_MAX);

				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					_min = _mm256_min_pd(_min, _mm256_load_pd(&data1[j * matrix1ActualRows + i]));
				}
				__m256d tempMin = _mm256_permute2f128_pd(_min, _min, 0x01);
				_min = _mm256_min_pd(_min, tempMin);

				__m128d low = _mm256_castpd256_pd128(_min);
				__m128d high = _mm256_extractf128_pd(_min, 1);

				low = _mm_min_pd(low, high);
				long long minLong = _mm_extract_epi64(low, 0);
				double min = reinterpret_cast<double&>(minLong);

				for (size_t i = finalPosRows; i < rows; i++)
				{
					double data = data1[j * matrix1ActualRows + i];
					if (data < min) min = data;
				}
				dataResult[j] = min;
			}
		}
		else
		{
			size_t matrix1ActualCols = this->actualCols;

			size_t finalPosCols = this->finalPosCols;
			size_t finalPosRows = this->finalPosRows;

			for (size_t j = 0; j < finalPosCols; j += 4)
			{
				__m256d _min = _mm256_set1_pd(DBL_MAX);
				for (size_t i = 0; i < rows; i++)
				{
					_min = _mm256_min_pd(_min, _mm256_load_pd(&data1[i * matrix1ActualCols + j]));
				}
				_mm256_store_pd(&dataResult[j], _min);
			}
			for (size_t j = finalPosCols; j < cols; j++)
			{
				double min = DBL_MAX;

				for (size_t i = 0; i < rows; i++)
				{
					double data = data1[i * matrix1ActualCols + j];
					if (data < min) min = data;
				}
				dataResult[j] = min;
			}
		}
		return result;
	}

	template <bool thisTransposed, bool thisContiguous>
	inline double matrix<double, thisTransposed, thisContiguous>::min_all()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		__m256d _min = _mm256_set1_pd(DBL_MAX);
		double min = DBL_MAX;

		if constexpr (thisContiguous)
		{
			size_t size = this->_size;
			size_t finalPosSize = this->finalPosSize;

			for (size_t i = 0; i < finalPosSize; i += 4)
			{
				_min = _mm256_min_pd(_min, _mm256_load_pd(&data1[i]));
			}
			for (size_t i = finalPosSize; i < size; i++)
			{
				double data = data1[i];
				if (data < min) min = data;
			}
		}
		else if constexpr (thisTransposed)
		{
			size_t matrix1ActualRows = this->actualRows;

			size_t finalPosRows = this->finalPosRows;
			size_t finalPosCols = this->finalPosCols;

			for (size_t i = 0; i < finalPosRows; i += 4)
			{
				for (size_t j = 0; j < cols; j++)
				{
					_min = _mm256_min_pd(_min, _mm256_load_pd(&data1[j * matrix1ActualRows + i]));
				}
			}
			for (size_t i = finalPosRows; i < rows; i++)
			{
				for (size_t j = 0; j < cols; j++)
				{
					double data = data1[j * matrix1ActualRows + i];
					if (data < min) min = data;
				}
			}
		}
		else
		{
			size_t matrix1ActualCols = this->actualCols;

			size_t finalPosCols = this->finalPosCols;
			size_t finalPosRows = this->finalPosRows;

			for (size_t j = 0; j < finalPosCols; j += 4)
			{
				for (size_t i = 0; i < rows; i++)
				{
					_min = _mm256_min_pd(_min, _mm256_load_pd(&data1[i * matrix1ActualCols + j]));
				}
			}
			for (size_t j = finalPosCols; j < cols; j++)
			{
				for (size_t i = 0; i < rows; i++)
				{
					double data = data1[i * matrix1ActualCols + j];
					if (data < min) min = data;
				}
			}
		}

		__m256d tempMin = _mm256_permute2f128_pd(_min, _min, 0x01);
		_min = _mm256_min_pd(_min, tempMin);

		__m128d low = _mm256_castpd256_pd128(_min);
		__m128d high = _mm256_extractf128_pd(_min, 1);

		low = _mm_min_pd(low, high);
		long long temp_min_long_long = _mm_extract_epi64(low, 0);
		double temp_min_d = reinterpret_cast<double&>(temp_min_long_long);

		if (temp_min_d < min) min = temp_min_d;

		return min;
	}

	template <bool thisTransposed, bool thisContiguous>
	template<char axis>
	inline std::conditional<axis == 'a', double, vector<double>> matrix<double, thisTransposed, thisContiguous>::min()
	{
		if constexpr (axis != 'a' && axis != 'r' && axis != 'c')
		{
			std::cerr << "Valid parameters for the axis are 'a', 'r', 'c'" << std::endl;
			exit(1);
		}

		if constexpr (axis == 'a')
		{
			return this->min_all();
		}
		else if constexpr (axis == 'r')
		{
			return this->min_rowwsie();
		}
		else if constexpr (axis == 'c')
		{
			return this->min_colwise();
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline void matrix<double, thisTransposed, thisContiguous>::argmin_all(size_t* row, size_t* col)
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		__m256i four = _mm256_set1_epi64x(4);

		__m256d _min = _mm256_set1_pd(DBL_MAX);
		double min = DBL_MAX;

		if constexpr (thisContiguous)
		{
			size_t size = this->_size;

			size_t finalPosSize = this->finalPosSize;

			__m256i min_indices = _mm256_setr_epi64x(0, 1, 2, 4);
			size_t min_index = 0;

			__m256i indices = _mm256_setr_epi64x(0, 1, 2, 4);

			for (size_t i = 0; i < finalPosSize; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);

				__m256d mask = _mm256_cmp_pd(a, _min, _CMP_LT_OQ);

				min_indices = _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(min_indices), _mm256_castsi256_pd(indices), mask));

				_min = _mm256_blendv_pd(_min, a, mask);

				indices = _mm256_add_epi64(indices, four);
			}
			for (size_t i = finalPosSize; i < size; i++)
			{
				double data = data1[i];
				if (data < min)
				{
					min = data;
					min_index = i;
				}
			}

			double mins_arr[4];
			size_t indices_arr[4];

			_mm256_store_pd(mins_arr, _min);
			_mm256_storeu_epi64(indices_arr, min_indices);

			for (size_t i = 0; i < 4; i++)
			{
				double element = mins_arr[i];
				if (element < min)
				{
					min = element;
					min_index = indices_arr[i];
				}
			}
			if constexpr (thisTransposed)
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
		else if constexpr (thisTransposed)
		{
			size_t matrix1ActualRows = this->actualRows;

			size_t finalPosRows = this->finalPosRows;

			__m256i _i = _mm256_set1_epi64x(0, 1, 2, 3);
			__m256i _j = _mm256_setzero_si256();

			__m256i one = _mm256_set1_epi64x(1);

			__m256i _i_min = _mm256_setzero_si256();
			__m256i _j_min = _mm256_setzero_si256();

			size_t row_index;
			size_t col_index;

			for (size_t i = 0; i < finalPosRows; i += 4)
			{
				for (size_t j = 0; j < cols; j++)
				{
					__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

					int mask = _mm256_movemask_pd(_mm256_cmp_pd(a, _min, _CMP_LT_OQ));

					_i_min = _mm256_castpd_si256(_mm256_blend_pd(_mm256_castsi256_pd(_i_min), _mm256_castsi256_pd(_i), mask));

					_j_min = _mm256_castpd_si256(_mm256_blend_pd(_mm256_castsi256_pd(_j_min), _mm256_castsi256_pd(_j), mask));

					_min = _mm256_blend_pd(_min, a, mask);

					_j = _mm256_add_epi64(_j, one);
				}
				_i = _mm256_add_epi64(_i, four);
			}
			for (size_t i = finalPosRows; i < rows; i++)
			{
				for (size_t j = 0; j < cols; j++)
				{
					double data = data1[j * matrix1ActualRows + i];
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
			size_t matrix1ActualCols = this->actualCols;

			size_t finalPosCols = this->finalPosCols;

			__m256i _i = _mm256_setzero_si256();
			__m256i _j = _mm256_set1_epi64x(0, 1, 2, 3);

			__m256i one = _mm256_set1_epi64x(1);

			__m256i _i_min = _mm256_setzero_si256();
			__m256i _j_min = _mm256_setzero_si256();

			size_t row_index;
			size_t col_index;

			for (size_t j = 0; j < finalPosCols; j += 4)
			{
				for (size_t i = 0; i < rows; i++)
				{
					__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

					int mask = _mm256_movemask_pd(_mm256_cmp_pd(a, _min, _CMP_LT_OQ));

					_i_min = _mm256_castpd_si256(_mm256_blend_pd(_mm256_castsi256_pd(_i_min), _mm256_castsi256_pd(_i), mask));

					_j_min = _mm256_castpd_si256(_mm256_blend_pd(_mm256_castsi256_pd(_j_min), _mm256_castsi256_pd(_j), mask));

					_min = _mm256_blend_pd(_min, a, mask);

					_i = _mm256_add_epi64(_i, one);
				}
				_j = _mm256_add_epi64(_j_min, four);
			}
			for (size_t j = finalPosCols; j < cols; j++)
			{
				for (size_t i = 0; i < rows; i++)
				{
					double data = data1[i * matrix1ActualCols + j];
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

	template <bool thisTransposed, bool thisContiguous>
	inline vector<uint64_t> matrix<double, thisTransposed, thisContiguous>::argmin_rowwise()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		vector<uint64_t> result(rows);

		uint64_t* dataResult = result._data;

		if constexpr (thisTransposed)
		{
			size_t matrix1ActualRows = this->actualRows;

			size_t finalPosRows = this->finalPosRows;

			__m256i one = _mm256_set1_epi64x(1);

			for (size_t i = 0; i < finalPosRows; i += 4)
			{
				__m256d _min = _mm256_set1_pd(DBL_MAX);
				__m256i indices = _mm256_setzero_si256();
				__m256i min_indices = _mm256_setzero_si256();
				for (size_t j = 0; j < cols; j++)
				{
					__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

					__m256d mask = _mm256_cmp_pd(a, _min, _CMP_LT_OQ);

					min_indices = _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(min_indices), _mm256_castsi256_pd(indices), mask));

					_min = _mm256_blendv_pd(_min, a, mask);

					indices = _mm256_add_epi64(indices, one);
				}
				_mm256_storeu_epi64(&dataResult[i], min_indices);
			}
			for (size_t i = finalPosRows; i < rows; i++)
			{
				double min = DBL_MAX;
				size_t index;
				for (size_t j = 0; j < cols; j++)
				{
					double data = data1[j * matrix1ActualRows + i];
					if (data < min)
					{
						min = data;
						index = j;
					}
				}
				dataResult[i] = index;
			}
		}
		else
		{
			size_t matrix1ActualCols = this->actualCols;

			size_t finalPosRows = this->finalPosRows;

			for (size_t i = 0; i < rows; i++)
			{
				double min = DBL_MAX;
				size_t index;
				for (size_t j = 0; j < cols; j++)
				{
					double data = data1[i * matrix1ActualCols + j];
					if (data < min)
					{
						min = data;
						index = j;
					}
				}
				dataResult[i] = index;
			}
		}
		return result;
	}

	template <bool thisTransposed, bool thisContiguous>
	inline vector<uint64_t> matrix<double, thisTransposed, thisContiguous>::argmin_colwise()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		vector<uint64_t> result(cols);

		uint64_t* dataResult = result._data;

		if constexpr (thisTransposed)
		{
			size_t matrix1ActualRows = this->actualRows;

			size_t finalPosCols = this->finalPosCols;

			for (size_t j = 0; j < cols; j++)
			{
				double min = DBL_MAX;
				size_t index;
				for (size_t i = 0; i < rows; i++)
				{
					double data = data1[j * matrix1ActualRows + i];
					if (data < min)
					{
						min = data;
						index = i;
					}
				}
				dataResult[j] = index;
			}
		}
		else
		{
			size_t matrix1ActualCols = this->actualCols;

			size_t finalPosCols = this->finalPosCols;

			__m256i one = _mm256_set1_epi64x(1);

			for (size_t j = 0; j < finalPosCols; j += 4)
			{
				__m256d _min = _mm256_set1_pd(DBL_MAX);
				__m256i indices = _mm256_setzero_si256();
				__m256i min_indices = _mm256_setzero_si256();
				for (size_t i = 0; i < cols; i++)
				{
					__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

					__m256d mask = _mm256_cmp_pd(a, _min, _CMP_LT_OQ);

					min_indices = _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(min_indices), _mm256_castsi256_pd(indices), mask));

					_min = _mm256_blendv_pd(_min, a, mask);

					indices = _mm256_add_epi64(indices, one);
				}
				_mm256_storeu_epi64(&dataResult[j], min_indices);
			}
			for (size_t j = finalPosCols; j < cols; j++)
			{
				double min = DBL_MAX;
				size_t index;
				for (size_t i = 0; i < rows; i++)
				{
					double data = data1[i * matrix1ActualCols + j];
					if (data < min)
					{
						min = data;
						index = i;
					}
				}
				dataResult[j] = index;
			}
		}
		return result;
	}

	template <bool thisTransposed, bool thisContiguous>
	template<char axis>
	inline std::conditional<axis == 'a', double, vector<double>> matrix<double, thisTransposed, thisContiguous>::argmin(size_t* row, size_t* col)
	{
		if constexpr (axis != 'a' && axis != 'r' && axis != 'c')
		{
			std::cerr << "Valid parameters for the axis are 'a', 'r', 'c'" << std::endl;
			exit(1);
		}

		if constexpr (axis == 'a')
		{
			return this->argmin_all(row, col);
		}
		else if constexpr (axis == 'r')
		{
			return this->argmin_rowwsie();
		}
		else if constexpr (axis == 'c')
		{
			return this->argmin_colwise();
		}
	}

	// Max

	template <bool thisTransposed, bool thisContiguous>
	inline vector<double> matrix<double, thisTransposed, thisContiguous>::max_rowwise()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		vector<double> result(rows);

		double* dataResult = result._data;

		if constexpr (thisTransposed)
		{
			size_t matrix1ActualRows = this->actualRows;

			size_t finalPosRows = this->finalPosRows;
			size_t finalPosCols = this->finalPosCols;

			for (size_t i = 0; i < finalPosRows; i += 4)
			{
				__m256d _max = _mm256_set1_pd(DBL_MIN);
				for (size_t j = 0; j < cols; j++)
				{
					_max = _mm256_max_pd(_max, _mm256_load_pd(&data1[j * matrix1ActualRows + i]));
				}
				_mm256_store_pd(&dataResult[i], _max);
			}
			for (size_t i = finalPosRows; i < rows; i++)
			{
				double max = DBL_MIN;

				for (size_t j = 0; j < cols; j++)
				{
					double data = data1[j * matrix1ActualRows + i];
					if (data > max) max = data;
				}
				dataResult[i] = max;
			}
		}
		else
		{
			size_t matrix1ActualCols = this->actualCols;

			size_t finalPosRows = this->finalPosRows;
			size_t finalPosCols = this->finalPosCols;

			for (size_t i = 0; i < rows; i++)
			{
				__m256d _max = _mm256_set1_pd(DBL_MIN);
				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					_max = _mm256_max_pd(_max, _mm256_load_pd(data1[i * matrix1ActualCols + j]));
				}
				__m256d tempmax = _mm256_permute2f128_pd(_max, _max, 0x01);
				_max = _mm256_max_pd(_max, tempmax);

				__m128d low = _mm256_castpd256_pd128(_max);
				__m128d high = _mm256_extractf128_pd(_max, 1);

				low = _mm_max_pd(low, high);
				long long maxLong = _mm_extract_epi64(low, 0);
				double max = reinterpret_cast<double&>(maxLong);

				for (size_t j = finalPosCols; j < cols; j++)
				{
					double data = data1[i * matrix1ActualCols + j];
					if (data > max) max = data;
				}
				dataResult[i] = max;
			}
		}
		return result;
	}

	template <bool thisTransposed, bool thisContiguous>
	inline vector<double> matrix<double, thisTransposed, thisContiguous>::max_colwise()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		vector<double> result(cols);

		double* dataResult = result._data;

		if constexpr (thisTransposed)
		{
			size_t matrix1ActualRows = this->actualRows;

			size_t finalPosCols = this->finalPosCols;
			size_t finalPosRows = this->finalPosRows;

			for (size_t j = 0; j < cols; j++)
			{
				__m256d _max = _mm256_set1_pd(DBL_MIN);

				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					_max = _mm256_max_pd(_max, _mm256_load_pd(&data1[j * matrix1ActualRows + i]));
				}
				__m256d tempmax = _mm256_permute2f128_pd(_max, _max, 0x01);
				_max = _mm256_max_pd(_max, tempmax);

				__m128d low = _mm256_castpd256_pd128(_max);
				__m128d high = _mm256_extractf128_pd(_max, 1);

				low = _mm_max_pd(low, high);
				long long maxLong = _mm_extract_epi64(low, 0);
				double max = reinterpret_cast<double&>(maxLong);

				for (size_t i = finalPosRows; i < rows; i++)
				{
					double data = data1[j * matrix1ActualRows + i];
					if (data > max) max = data;
				}
				dataResult[j] = max;
			}
		}
		else
		{
			size_t matrix1ActualCols = this->actualCols;

			size_t finalPosCols = this->finalPosCols;
			size_t finalPosRows = this->finalPosRows;

			for (size_t j = 0; j < finalPosCols; j += 4)
			{
				__m256d _max = _mm256_set1_pd(DBL_MIN);
				for (size_t i = 0; i < rows; i++)
				{
					_max = _mm256_max_pd(_max, _mm256_load_pd(&data1[i * matrix1ActualCols + j]));
				}
				_mm256_store_pd(&dataResult[j], _max);
			}
			for (size_t j = finalPosCols; j < cols; j++)
			{
				double max = DBL_MIN;

				for (size_t i = 0; i < rows; i++)
				{
					double data = data1[i * matrix1ActualCols + j];
					if (data > max) max = data;
				}
				dataResult[j] = max;
			}
		}
		return result;
	}

	template <bool thisTransposed, bool thisContiguous>
	inline double matrix<double, thisTransposed, thisContiguous>::max_all()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		__m256d _max = _mm256_set1_pd(DBL_MIN);
		double max = DBL_MIN;

		if constexpr (thisContiguous)
		{
			size_t size = this->_size;
			size_t finalPosSize = this->finalPosSize;

			for (size_t i = 0; i < finalPosSize; i += 4)
			{
				_max = _mm256_max_pd(_max, _mm256_load_pd(&data1[i]));
			}
			for (size_t i = finalPosSize; i < size; i++)
			{
				double data = data1[i];
				if (data > max) max = data;
			}
		}
		else if constexpr (thisTransposed)
		{
			size_t matrix1ActualRows = this->actualRows;

			size_t finalPosRows = this->finalPosRows;
			size_t finalPosCols = this->finalPosCols;

			for (size_t i = 0; i < finalPosRows; i += 4)
			{
				for (size_t j = 0; j < cols; j++)
				{
					_max = _mm256_max_pd(_max, _mm256_load_pd(&data1[j * matrix1ActualRows + i]));
				}
			}
			for (size_t i = finalPosRows; i < rows; i++)
			{
				for (size_t j = 0; j < cols; j++)
				{
					double data = data1[j * matrix1ActualRows + i];
					if (data > max) max = data;
				}
			}
		}
		else
		{
			size_t matrix1ActualCols = this->actualCols;

			size_t finalPosCols = this->finalPosCols;
			size_t finalPosRows = this->finalPosRows;

			for (size_t j = 0; j < finalPosCols; j += 4)
			{
				for (size_t i = 0; i < rows; i++)
				{
					_max = _mm256_max_pd(_max, _mm256_load_pd(&data1[i * matrix1ActualCols + j]));
				}
			}
			for (size_t j = finalPosCols; j < cols; j++)
			{
				for (size_t i = 0; i < rows; i++)
				{
					double data = data1[i * matrix1ActualCols + j];
					if (data > max) max = data;
				}
			}
		}

		__m256d tempmax = _mm256_permute2f128_pd(_max, _max, 0x01);
		_max = _mm256_max_pd(_max, tempmax);

		__m128d low = _mm256_castpd256_pd128(_max);
		__m128d high = _mm256_extractf128_pd(_max, 1);

		low = _mm_max_pd(low, high);
		long long temp_max_long_long = _mm_extract_epi64(low, 0);
		double temp_max_d = reinterpret_cast<double&>(temp_max_long_long);

		if (temp_max_d > max) max = temp_max_d;

		return max;
	}

	template <bool thisTransposed, bool thisContiguous>
	template<char axis>
	inline std::conditional<axis == 'a', double, vector<double>> matrix<double, thisTransposed, thisContiguous>::max()
	{
		if constexpr (axis != 'a' && axis != 'r' && axis != 'c')
		{
			std::cerr << "Valid parameters for the axis are 'a', 'r', 'c'" << std::endl;
			exit(1);
		}

		if constexpr (axis == 'a')
		{
			return this->max_all();
		}
		else if constexpr (axis == 'r')
		{
			return this->max_rowwsie();
		}
		else if constexpr (axis == 'c')
		{
			return this->max_colwise();
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline void matrix<double, thisTransposed, thisContiguous>::argmax_all(size_t* row, size_t* col)
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		__m256i four = _mm256_set1_epi64x(4);

		__m256d _max = _mm256_set1_pd(DBL_MIN);
		double max = DBL_MIN;

		if constexpr (thisContiguous)
		{
			size_t size = this->_size;

			size_t finalPosSize = this->finalPosSize;

			__m256i max_indices = _mm256_setr_epi64x(0, 1, 2, 4);
			size_t max_index = 0;

			__m256i indices = _mm256_setr_epi64x(0, 1, 2, 4);

			for (size_t i = 0; i < finalPosSize; i += 4)
			{
				__m256d a = _mm256_load_pd(&data1[i]);

				__m256d mask = _mm256_cmp_pd(a, _max, _CMP_GT_OQ);

				max_indices = _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(max_indices), _mm256_castsi256_pd(indices), mask));

				_max = _mm256_blendv_pd(_max, a, mask);

				indices = _mm256_add_epi64(indices, four);
			}
			for (size_t i = finalPosSize; i < size; i++)
			{
				double data = data1[i];
				if (data > max)
				{
					max = data;
					max_index = i;
				}
			}

			double maxs_arr[4];
			size_t indices_arr[4];

			_mm256_store_pd(maxs_arr, _max);
			_mm256_storeu_epi64(indices_arr, max_indices);

			for (size_t i = 0; i < 4; i++)
			{
				double element = maxs_arr[i];
				if (element > max)
				{
					max = element;
					max_index = indices_arr[i];
				}
			}
			if constexpr (thisTransposed)
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
		else if constexpr (thisTransposed)
		{
			size_t matrix1ActualRows = this->actualRows;

			size_t finalPosRows = this->finalPosRows;

			__m256i _i = _mm256_set1_epi64x(0, 1, 2, 3);
			__m256i _j = _mm256_setzero_si256();

			__m256i one = _mm256_set1_epi64x(1);

			__m256i _i_max = _mm256_setzero_si256();
			__m256i _j_max = _mm256_setzero_si256();

			size_t row_index;
			size_t col_index;

			for (size_t i = 0; i < finalPosRows; i += 4)
			{
				for (size_t j = 0; j < cols; j++)
				{
					__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

					int mask = _mm256_movemask_pd(_mm256_cmp_pd(a, _max, _CMP_GT_OQ));

					_i_max = _mm256_castpd_si256(_mm256_blend_pd(_mm256_castsi256_pd(_i_max), _mm256_castsi256_pd(_i), mask));

					_j_max = _mm256_castpd_si256(_mm256_blend_pd(_mm256_castsi256_pd(_j_max), _mm256_castsi256_pd(_j), mask));

					_max = _mm256_blend_pd(_max, a, mask);

					_j = _mm256_add_epi64(_j, one);
				}
				_i = _mm256_add_epi64(_i, four);
			}
			for (size_t i = finalPosRows; i < rows; i++)
			{
				for (size_t j = 0; j < cols; j++)
				{
					double data = data1[j * matrix1ActualRows + i];
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
				double element = maxs_arr[i];
				if (element > max)
				{
					max = element;
					row_index = i_arr[i];
					col_index = j_arr[i];
				}
			}
			*row = row_index;
			*col = col_index;
		}
		else
		{
			size_t matrix1ActualCols = this->actualCols;

			size_t finalPosCols = this->finalPosCols;

			__m256i _i = _mm256_setzero_si256();
			__m256i _j = _mm256_set1_epi64x(0, 1, 2, 3);

			__m256i one = _mm256_set1_epi64x(1);

			__m256i _i_max = _mm256_setzero_si256();
			__m256i _j_max = _mm256_setzero_si256();

			size_t row_index;
			size_t col_index;

			for (size_t j = 0; j < finalPosCols; j += 4)
			{
				for (size_t i = 0; i < rows; i++)
				{
					__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

					int mask = _mm256_movemask_pd(_mm256_cmp_pd(a, _max, _CMP_GT_OQ));

					_i_max = _mm256_castpd_si256(_mm256_blend_pd(_mm256_castsi256_pd(_i_max), _mm256_castsi256_pd(_i), mask));

					_j_max = _mm256_castpd_si256(_mm256_blend_pd(_mm256_castsi256_pd(_j_max), _mm256_castsi256_pd(_j), mask));

					_max = _mm256_blend_pd(_max, a, mask);

					_i = _mm256_add_epi64(_i, one);
				}
				_j = _mm256_add_epi64(_j_max, four);
			}
			for (size_t j = finalPosCols; j < cols; j++)
			{
				for (size_t i = 0; i < rows; i++)
				{
					double data = data1[i * matrix1ActualCols + j];
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
				double element = maxs_arr[i];
				if (element > max)
				{
					max = element;
					row_index = i_arr[i];
					col_index = j_arr[i];
				}
			}
			*row = row_index;
			*col = col_index;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline vector<uint64_t> matrix<double, thisTransposed, thisContiguous>::argmax_rowwise()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		vector<uint64_t> result(rows);

		uint64_t* dataResult = result._data;

		if constexpr (thisTransposed)
		{
			size_t matrix1ActualRows = this->actualRows;

			size_t finalPosRows = this->finalPosRows;

			__m256i one = _mm256_set1_epi64x(1);

			for (size_t i = 0; i < finalPosRows; i += 4)
			{
				__m256d _max = _mm256_set1_pd(DBL_MIN);
				__m256i indices = _mm256_setzero_si256();
				__m256i max_indices = _mm256_setzero_si256();
				for (size_t j = 0; j < cols; j++)
				{
					__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

					__m256d mask = _mm256_cmp_pd(a, _max, _CMP_GT_OQ);

					max_indices = _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(max_indices), _mm256_castsi256_pd(indices), mask));

					_max = _mm256_blendv_pd(_max, a, mask);

					indices = _mm256_add_epi64(indices, one);
				}
				_mm256_storeu_epi64(&dataResult[i], max_indices);
			}
			for (size_t i = finalPosRows; i < rows; i++)
			{
				double max = DBL_MIN;
				size_t index;
				for (size_t j = 0; j < cols; j++)
				{
					double data = data1[j * matrix1ActualRows + i];
					if (data > max)
					{
						max = data;
						index = j;
					}
				}
				dataResult[i] = index;
			}
		}
		else
		{
			size_t matrix1ActualCols = this->actualCols;

			size_t finalPosRows = this->finalPosRows;

			for (size_t i = 0; i < rows; i++)
			{
				double max = DBL_MIN;
				size_t index;
				for (size_t j = 0; j < cols; j++)
				{
					double data = data1[i * matrix1ActualCols + j];
					if (data > max)
					{
						max = data;
						index = j;
					}
				}
				dataResult[i] = index;
			}
		}
		return result;
	}

	template <bool thisTransposed, bool thisContiguous>
	inline vector<uint64_t> matrix<double, thisTransposed, thisContiguous>::argmax_colwise()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		vector<uint64_t> result(cols);

		uint64_t* dataResult = result._data;

		if constexpr (thisTransposed)
		{
			size_t matrix1ActualRows = this->actualRows;

			size_t finalPosCols = this->finalPosCols;

			for (size_t j = 0; j < cols; j++)
			{
				double max = DBL_MIN;
				size_t index;
				for (size_t i = 0; i < rows; i++)
				{
					double data = data1[j * matrix1ActualRows + i];
					if (data > max)
					{
						max = data;
						index = i;
					}
				}
				dataResult[j] = index;
			}
		}
		else
		{
			size_t matrix1ActualCols = this->actualCols;

			size_t finalPosCols = this->finalPosCols;

			__m256i one = _mm256_set1_epi64x(1);

			for (size_t j = 0; j < finalPosCols; j += 4)
			{
				__m256d _max = _mm256_set1_pd(DBL_MIN);
				__m256i indices = _mm256_setzero_si256();
				__m256i max_indices = _mm256_setzero_si256();
				for (size_t i = 0; i < cols; i++)
				{
					__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

					__m256d mask = _mm256_cmp_pd(a, _max, _CMP_GT_OQ);

					max_indices = _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(max_indices), _mm256_castsi256_pd(indices), mask));

					_max = _mm256_blendv_pd(_max, a, mask);

					indices = _mm256_add_epi64(indices, one);
				}
				_mm256_storeu_epi64(&dataResult[j], max_indices);
			}
			for (size_t j = finalPosCols; j < cols; j++)
			{
				double max = DBL_MIN;
				size_t index;
				for (size_t i = 0; i < rows; i++)
				{
					double data = data1[i * matrix1ActualCols + j];
					if (data > max)
					{
						max = data;
						index = i;
					}
				}
				dataResult[j] = index;
			}
		}
		return result;
	}

	template <bool thisTransposed, bool thisContiguous>
	template<char axis>
	inline std::conditional<axis == 'a', double, vector<double>> matrix<double, thisTransposed, thisContiguous>::argmax(size_t* row, size_t* col)
	{
		if constexpr (axis != 'a' && axis != 'r' && axis != 'c')
		{
			std::cerr << "Valid parameters for the axis are 'a', 'r', 'c'" << std::endl;
			exit(1);
		}

		if constexpr (axis == 'a')
		{
			return this->argmax_all(row, col);
		}
		else if constexpr (axis == 'r')
		{
			return this->argmax_rowwsie();
		}
		else if constexpr (axis == 'c')
		{
			return this->argmax_colwise();
		}
	}

	// Activation functions

	// ReLU

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed>
	inline matrix<double> matrix<double, thisTransposed, thisContiguous>::relu()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		__m256d zero = _mm256_setzero_pd();

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix1ActualCols = this->actualCols;

		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;
		size_t finalPosSize = this->finalPosSize;
		size_t size = this->_size;

		if constexpr (returnTransposed)
		{
			matrix<double> result(cols, rows);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&dataResult[i], _mm256_max_pd(zero, a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = std::max(0.0, data1[i]);
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

							_mm256_store_pd(&dataResult[j * rows + i], _mm256_max_pd(zero, a));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = std::max(0.0, data1[j * matrix1ActualRows + i]);
						}
					}
				}
			}
			else
			{
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						__m256d max = _mm256_max_pd(zero, a);

						__m128d val1 = _mm256_extractf128_pd(max, 1);
						__m128d val2 = _mm256_castpd256_pd128(max);

						_mm_store_sd(&dataResult[j * rows + i], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

						_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						dataResult[j * rows + i] = std::max(0.0, data1[i * matrix1ActualCols + j]);
					}
				}
			}
			return result;
		}
		else
		{
			matrix<double> result(rows, cols);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				for (size_t j = 0; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						__m256d max = _mm256_max_pd(zero, a);

						__m128d val1 = _mm256_extractf128_pd(max, 1);
						__m128d val2 = _mm256_castpd256_pd128(max);

						_mm_store_sd(&dataResult[i * cols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

						_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						dataResult[i * cols + j] = std::max(0.0, data1[j * matrix1ActualRows + i]);
					}
				}
			}
			else
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);
						_mm256_store_pd(&dataResult[i], _mm256_max_pd(zero, a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = std::max(0.0, data1[i]);
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

							_mm256_store_pd(&dataResult[i * cols + j], _mm256_max_pd(zero, a));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = std::max(0.0, data1[i * matrix1ActualCols + j]);
						}
					}
				}
			}
			return result;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline void matrix<double, thisTransposed, thisContiguous>::self_relu()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		__m256d zero = _mm256_setzero_pd();

		if constexpr (thisTransposed)
		{
			if constexpr (thisContiguous)
			{
				size_t finalPosSize = this->finalPosSize;
				size_t size = this->_size;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);

					_mm256_store_pd(&data1[i], _mm256_max_pd(zero, a));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] = std::max(0.0, data1[i]);
				}
			}
			else
			{
				size_t matrix1ActualRows = this->actualRows;

				size_t finalPosRows = this->finalPosRows;
				size_t finalPosCols = this->finalPosCols;

				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						_mm256_store_pd(&data1[j * matrix1ActualRows + i], _mm256_max_pd(zero, a));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						data1[j * matrix1ActualRows + i] = std::max(0.0, data1[j * matrix1ActualRows + i]);
					}
				}
			}
		}
		else
		{
			if constexpr (thisContiguous)
			{
				size_t size = this->_size;
				size_t finalPosSize = this->finalPosSize;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);
					_mm256_store_pd(&data1[i], _mm256_max_pd(zero, a));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] = std::max(0.0, data1[i]);
				}
			}
			else
			{
				size_t matrix1ActualCols = this->actualCols;

				size_t finalPosCols = this->finalPosCols;
				size_t finalPosRows = this->finalPosRows;

				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						_mm256_store_pd(&data1[i * matrix1ActualCols + j], _mm256_max_pd(zero, a));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < rows; i++)
					{
						data1[i * matrix1ActualCols + j] = std::max(0.0, data1[i * matrix1ActualCols + j]);
					}
				}
			}
		}
	}

	// LReLU

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed>
	inline matrix<double> matrix<double, thisTransposed, thisContiguous>::lrelu()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		__m256d _num = _mm256_set1_pd(0.01);
		__m256d _zero = _mm256_setzero_pd();

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix1ActualCols = this->actualRows;

		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;

		size_t finalPosSize = this->finalPosSize;
		size_t size = this->_size;

		if constexpr (returnTransposed)
		{
			matrix<double> result(cols, rows);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&dataResult[i], _mm256_blendv_pd(_mm256_mul_pd(a, _num), a, _mm256_cmp_pd(a, _zero, _CMP_GT_OQ)));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = data1[i] > 0.0 ? data1[i] : data1[i] * 0.01;
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

							_mm256_store_pd(&dataResult[j * rows + i], _mm256_blendv_pd(_mm256_mul_pd(a, _num), a, _mm256_cmp_pd(a, _zero, _CMP_GT_OQ)));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < cols; j++)
						{
							dataResult[j * rows + i] = data1[j * matrix1ActualRows + i] > 0.0 ? data1[j * matrix1ActualRows + i] : data1[j * matrix1ActualRows + i] * 0.01;
						}
					}
				}
			}
			else
			{
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						__m256d lrelu = _mm256_blendv_pd(_mm256_mul_pd(a, _num), a, _mm256_cmp_pd(a, _zero, _CMP_GT_OQ));

						__m128d val1 = _mm256_extractf128_pd(lrelu, 1);
						__m128d val2 = _mm256_castpd256_pd128(lrelu);

						_mm_store_sd(&dataResult[j * rows + i], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

						_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						dataResult[j * rows + i] = data1[i * matrix1ActualCols + j] > 0.0 ? data1[i * matrix1ActualCols + j] : data1[i * matrix1ActualCols + j] * 0.01;
					}
				}
			}
			return result;
		}
		else
		{
			matrix<double> result(rows, cols);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				for (size_t j = 0; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						__m256d lrelu = _mm256_blendv_pd(_mm256_mul_pd(a, _num), a, _mm256_cmp_pd(a, _zero, _CMP_GT_OQ));

						__m128d val1 = _mm256_extractf128_pd(lrelu, 1);
						__m128d val2 = _mm256_castpd256_pd128(lrelu);

						_mm_store_sd(&dataResult[i * cols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

						_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						dataResult[i * cols + j] = data1[j * matrix1ActualRows + i] > 0.0 ? data1[j * matrix1ActualRows + i] : data1[j * matrix1ActualRows + i] * 0.01;
					}
				}
			}
			else
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);
						_mm256_store_pd(&dataResult[i], _mm256_blendv_pd(_mm256_mul_pd(a, _num), a, _mm256_cmp_pd(a, _zero, _CMP_GT_OQ)));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = data1[i] > 0.0 ? data1[i] : data1[i] * 0.01;
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

							_mm256_store_pd(&dataResult[i * cols + j], _mm256_blendv_pd(_mm256_mul_pd(a, _num), a, _mm256_cmp_pd(a, _zero, _CMP_GT_OQ)));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < rows; i++)
						{
							dataResult[i * cols + j] = data1[i * matrix1ActualCols + j] > 0.0 ? data1[i * matrix1ActualCols + j] : data1[i * matrix1ActualCols + j] * 0.01;
						}
					}
				}
			}
			return result;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline void matrix<double, thisTransposed, thisContiguous>::self_lrelu()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		__m256d _num = _mm256_set1_pd(0.01);
		__m256d _zero = _mm256_setzero_pd();

		if constexpr (thisTransposed)
		{
			if constexpr (thisContiguous)
			{
				size_t finalPosSize = this->finalPosSize;
				size_t size = this->_size;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);

					_mm256_store_pd(&data1[i], _mm256_blendv_pd(_mm256_mul_pd(a, _num), a, _mm256_cmp_pd(a, _zero, _CMP_GT_OQ)));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] = data1[i] > 0.0 ? data1[i] : data1[i] * 0.01;
				}
			}
			else
			{
				size_t matrix1ActualRows = this->actualRows;

				size_t finalPosRows = this->finalPosRows;
				size_t finalPosCols = this->finalPosCols;

				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						_mm256_store_pd(&data1[j * matrix1ActualRows + i], _mm256_blendv_pd(_mm256_mul_pd(a, _num), a, _mm256_cmp_pd(a, _zero, _CMP_GT_OQ)));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						data1[j * matrix1ActualRows + i] = data1[j * matrix1ActualRows + i] > 0.0 ? data1[j * matrix1ActualRows + i] : data1[j * matrix1ActualRows + i] * 0.01;
					}
				}
			}
		}
		else
		{
			if constexpr (thisContiguous)
			{
				size_t size = this->_size;
				size_t finalPosSize = this->finalPosSize;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);
					_mm256_store_pd(&data1[i], _mm256_blendv_pd(_mm256_mul_pd(a, _num), a, _mm256_cmp_pd(a, _zero, _CMP_GT_OQ)));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] = data1[i] > 0.0 ? data1[i] : data1[i] * 0.01;
				}
			}
			else
			{
				size_t matrix1ActualCols = this->actualCols;

				size_t finalPosCols = this->finalPosCols;
				size_t finalPosRows = this->finalPosRows;

				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						_mm256_store_pd(&data1[i * matrix1ActualCols + j], _mm256_blendv_pd(_mm256_mul_pd(a, _num), a, _mm256_cmp_pd(a, _zero, _CMP_GT_OQ)));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < rows; i++)
					{
						data1[i * matrix1ActualCols + j] = data1[i * matrix1ActualCols + j] > 0.0 ? data1[i * matrix1ActualCols + j] : data1[i * matrix1ActualCols + j] * 0.01;
					}
				}
			}
		}
	}

	// Sigmoid

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed>
	inline matrix<double> matrix<double, thisTransposed, thisContiguous>::sigmoid()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		__m256d one = _mm256_set1_pd(1.0);

		__m256d mask = _mm256_set1_pd(-0.0);

		size_t finalPosSize = this->finalPosSize;
		size_t size = this->_size;

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix1ActualCols = this->actualCols;

		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;

		if constexpr (returnTransposed)
		{
			matrix<double> result(cols, rows);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&dataResult[i], _mm256_div_pd(one, _mm256_add_pd(_mm256_exp_pd(_mm256_xor_pd(a, mask)), one)));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = 1.0 / (1.0 + std::exp(-data1[i]));
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

							_mm256_store_pd(&dataResult[j * rows + i], _mm256_div_pd(one, _mm256_add_pd(_mm256_exp_pd(_mm256_xor_pd(a, mask)), one)));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < finalPosCols; j += 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
								data1[(j + 1) * matrix1ActualRows + i],
								data1[(j + 2) * matrix1ActualRows + i],
								data1[(j + 3) * matrix1ActualRows + i]);

							__m256d sigmoid = _mm256_div_pd(one, _mm256_add_pd(_mm256_exp_pd(_mm256_xor_pd(a, mask)), one));

							__m128d val1 = _mm256_extractf128_pd(sigmoid, 1);
							__m128d val2 = _mm256_castpd256_pd128(sigmoid);

							_mm_store_sd(&dataResult[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

							_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
						}
						for (size_t j = finalPosCols; j < cols; j++)
						{
							dataResult[j * rows + i] = 1.0 / (1.0 + std::exp(-data1[i]));
						}
					}
				}
			}
			else
			{
				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
							data1[(i + 1) * matrix1ActualCols + j],
							data1[(i + 2) * matrix1ActualCols + j],
							data1[(i + 3) * matrix1ActualCols + j]);

						_mm256_store_pd(&dataResult[j * rows + i], _mm256_div_pd(one, _mm256_add_pd(_mm256_exp_pd(_mm256_xor_pd(a, mask)), one)));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						__m256d sigmoid = _mm256_div_pd(one, _mm256_add_pd(_mm256_exp_pd(_mm256_xor_pd(a, mask)), one));

						__m128d val1 = _mm256_extractf128_pd(sigmoid, 1);
						__m128d val2 = _mm256_castpd256_pd128(sigmoid);

						_mm_store_sd(&dataResult[j * rows + i], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

						_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						dataResult[j * rows + i] = 1.0 / (1.0 + std::exp(-data1[i * matrix1ActualCols + j]));
					}
				}
			}
			return result;
		}
		else
		{
			matrix<double> result(rows, cols);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
							data1[(j + 1) * matrix1ActualRows + i],
							data1[(j + 2) * matrix1ActualRows + i],
							data1[(j + 3) * matrix1ActualRows + i]);
						_mm256_store_pd(&dataResult[i * cols + j], _mm256_div_pd(one, _mm256_add_pd(_mm256_exp_pd(_mm256_xor_pd(a, mask)), one)));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						__m256d sigmoid = _mm256_div_pd(one, _mm256_add_pd(_mm256_exp_pd(_mm256_xor_pd(a, mask)), one));

						__m128d val1 = _mm256_extractf128_pd(sigmoid, 1);
						__m128d val2 = _mm256_castpd256_pd128(sigmoid);

						_mm_store_sd(&dataResult[i * cols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

						_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						dataResult[i * cols + j] = 1.0 / (1.0 + std::exp(-data1[j * matrix1ActualRows + i]));
					}
				}
			}
			else
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);
						_mm256_store_pd(&dataResult[i], _mm256_div_pd(one, _mm256_add_pd(_mm256_exp_pd(_mm256_xor_pd(a, mask)), one)));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = 1.0 / (1.0 + std::exp(-data1[i]));
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

							_mm256_store_pd(&dataResult[i * cols + j], _mm256_div_pd(one, _mm256_add_pd(_mm256_exp_pd(_mm256_xor_pd(a, mask)), one)));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < finalPosRows; i += 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
								data1[(i + 1) * matrix1ActualCols + j],
								data1[(i + 2) * matrix1ActualCols + j],
								data1[(i + 3) * matrix1ActualCols + j]);

							__m256d sigmoid = _mm256_div_pd(one, _mm256_add_pd(_mm256_exp_pd(_mm256_xor_pd(a, mask)), one));

							__m128d val1 = _mm256_extractf128_pd(sigmoid, 1);
							__m128d val2 = _mm256_castpd256_pd128(sigmoid);

							_mm_store_sd(&dataResult[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

							_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
						}
						for (size_t i = finalPosRows; i < rows; i++)
						{
							dataResult[i * cols + j] = 1.0 / (1.0 + std::exp(-data1[i * matrix1ActualCols + j]));
						}
					}
				}
			}
			return result;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline void matrix<double, thisTransposed, thisContiguous>::self_sigmoid()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		__m256d one = _mm256_set1_pd(1.0);

		__m256d mask = _mm256_set1_pd(-0.0);

		if constexpr (thisTransposed)
		{
			if constexpr (thisContiguous)
			{
				size_t finalPosSize = this->finalPosSize;
				size_t size = this->_size;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);

					_mm256_store_pd(&data1[i], _mm256_div_pd(one, _mm256_add_pd(_mm256_exp_pd(_mm256_xor_pd(a, mask)), one)));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] = 1.0 / (1.0 + std::exp(-data1[i]));
				}
			}
			else
			{
				size_t matrix1ActualRows = this->actualRows;

				size_t finalPosRows = this->finalPosRows;
				size_t finalPosCols = this->finalPosCols;

				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						_mm256_store_pd(&data1[j * matrix1ActualRows + i], _mm256_div_pd(one, _mm256_add_pd(_mm256_exp_pd(_mm256_xor_pd(a, mask)), one)));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
							data1[(j + 1) * matrix1ActualRows + i],
							data1[(j + 2) * matrix1ActualRows + i],
							data1[(j + 3) * matrix1ActualRows + i]);

						__m256d sigmoid = _mm256_div_pd(one, _mm256_add_pd(_mm256_exp_pd(_mm256_xor_pd(a, mask)), one));

						__m128d val1 = _mm256_extractf128_pd(sigmoid, 1);
						__m128d val2 = _mm256_castpd256_pd128(sigmoid);

						_mm_store_sd(&data1[j * matrix1ActualRows + i], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&data1[(j + 1) * matrix1ActualRows + i], val2);

						_mm_store_sd(&data1[(j + 2) * matrix1ActualRows + i], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&data1[(j + 3) * matrix1ActualRows + i], val1);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						data1[j * matrix1ActualRows + i] = 1.0 / (1.0 + std::exp(-data1[j * matrix1ActualRows + i]));
					}
				}
			}
		}
		else
		{
			if constexpr (thisContiguous)
			{
				size_t size = this->_size;
				size_t finalPosSize = this->finalPosSize;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);
					_mm256_store_pd(&data1[i], _mm256_div_pd(one, _mm256_add_pd(_mm256_exp_pd(_mm256_xor_pd(a, mask)), one)));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] = 1.0 / (1.0 + std::exp(-data1[i]));
				}
			}
			else
			{
				size_t matrix1ActualCols = this->actualCols;

				size_t finalPosCols = this->finalPosCols;
				size_t finalPosRows = this->finalPosRows;

				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						_mm256_store_pd(&data1[i * matrix1ActualCols + j], _mm256_div_pd(one, _mm256_add_pd(_mm256_exp_pd(_mm256_xor_pd(a, mask)), one)));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
							data1[(i + 1) * matrix1ActualCols + j],
							data1[(i + 2) * matrix1ActualCols + j],
							data1[(i + 3) * matrix1ActualCols + j]);

						__m256d sigmoid = _mm256_div_pd(one, _mm256_add_pd(_mm256_exp_pd(_mm256_xor_pd(a, mask)), one));

						__m128d val1 = _mm256_extractf128_pd(sigmoid, 1);
						__m128d val2 = _mm256_castpd256_pd128(sigmoid);

						_mm_store_sd(&data1[i * matrix1ActualCols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&data1[(i + 1) * matrix1ActualCols + j], val2);

						_mm_store_sd(&data1[(i + 2) * matrix1ActualCols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&data1[(i + 3) * matrix1ActualCols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						data1[i * matrix1ActualCols + j] = 1.0 / (1.0 + std::exp(-data1[i * matrix1ActualCols + j]));
					}
				}
			}
		}
	}

	// Softplus

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed>
	inline matrix<double> matrix<double, thisTransposed, thisContiguous>::softplus()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		__m256d one = _mm256_set1_pd(1.0);

		size_t finalPosSize = this->finalPosSize;
		size_t size = this->_size;

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix1ActualCols = this->actualCols;

		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;

		if constexpr (returnTransposed)
		{
			matrix<double> result(cols, rows);

			double* dataResult = result._data;
			if constexpr (thisTransposed)
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&dataResult[i], _mm256_log_pd(_mm256_add_pd(one, _mm256_exp_pd(a))));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = std::log(1.0 + std::exp(data1[i]));
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

							_mm256_store_pd(&dataResult[j * rows + i], _mm256_log_pd(_mm256_add_pd(one, _mm256_exp_pd(a))));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < finalPosCols; j += 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
								data1[(j + 1) * matrix1ActualRows + i],
								data1[(j + 2) * matrix1ActualRows + i],
								data1[(j + 3) * matrix1ActualRows + i]);

							__m256d softplus = _mm256_log_pd(_mm256_add_pd(one, _mm256_exp_pd(a)));

							__m128d val1 = _mm256_extractf128_pd(softplus, 1);
							__m128d val2 = _mm256_castpd256_pd128(softplus);

							_mm_store_sd(&dataResult[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

							_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
						}
						for (size_t j = finalPosCols; j < cols; j++)
						{
							dataResult[j * rows + i] = std::log(1.0 + std::exp(data1[i]));
						}
					}
				}
			}
			else
			{
				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
							data1[(i + 1) * matrix1ActualCols + j],
							data1[(i + 2) * matrix1ActualCols + j],
							data1[(i + 3) * matrix1ActualCols + j]);

						_mm256_store_pd(&dataResult[j * rows + i], _mm256_log_pd(_mm256_add_pd(one, _mm256_exp_pd(a))));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						__m256d softplus = _mm256_log_pd(_mm256_add_pd(one, _mm256_exp_pd(a)));

						__m128d val1 = _mm256_extractf128_pd(softplus, 1);
						__m128d val2 = _mm256_castpd256_pd128(softplus);

						_mm_store_sd(&dataResult[j * rows + i], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

						_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						dataResult[j * rows + i] = std::log(1.0 + std::exp(data1[i * matrix1ActualCols + j]));
					}
				}
			}
			return result;
		}
		else
		{
			matrix<double> result(rows, cols);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
							data1[(j + 1) * matrix1ActualRows + i],
							data1[(j + 2) * matrix1ActualRows + i],
							data1[(j + 3) * matrix1ActualRows + i]);
						_mm256_store_pd(&dataResult[i * cols + j], _mm256_log_pd(_mm256_add_pd(one, _mm256_exp_pd(a))));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						__m256d sigmoid = _mm256_log_pd(_mm256_add_pd(one, _mm256_exp_pd(a)));

						__m128d val1 = _mm256_extractf128_pd(sigmoid, 1);
						__m128d val2 = _mm256_castpd256_pd128(sigmoid);

						_mm_store_sd(&dataResult[i * cols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

						_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						dataResult[i * cols + j] = std::log(1.0 + std::exp(data1[j * matrix1ActualRows + i]));
					}
				}
			}
			else
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);
						_mm256_store_pd(&dataResult[i], _mm256_log_pd(_mm256_add_pd(one, _mm256_exp_pd(a))));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = std::log(1.0 + std::exp(data1[i]));
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

							_mm256_store_pd(&dataResult[i * cols + j], _mm256_log_pd(_mm256_add_pd(one, _mm256_exp_pd(a))));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < finalPosRows; i += 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
								data1[(i + 1) * matrix1ActualCols + j],
								data1[(i + 2) * matrix1ActualCols + j],
								data1[(i + 3) * matrix1ActualCols + j]);

							__m256d softplus = _mm256_log_pd(_mm256_add_pd(one, _mm256_exp_pd(a)));

							__m128d val1 = _mm256_extractf128_pd(softplus, 1);
							__m128d val2 = _mm256_castpd256_pd128(softplus);

							_mm_store_sd(&dataResult[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

							_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
						}
						for (size_t i = finalPosRows; i < rows; i++)
						{
							dataResult[i * cols + j] = std::log(1.0 + std::exp(data1[i * matrix1ActualCols + j]));
						}
					}
				}
			}
			return result;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline void matrix<double, thisTransposed, thisContiguous>::self_softplus()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		__m256d one = _mm256_set1_pd(1.0);

		if constexpr (thisTransposed)
		{
			if constexpr (thisContiguous)
			{
				size_t finalPosSize = this->finalPosSize;
				size_t size = this->_size;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);

					_mm256_store_pd(&data1[i], _mm256_log_pd(_mm256_add_pd(one, _mm256_exp_pd(a))));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] = std::log(1.0 + std::exp(data1[i]));
				}
			}
			else
			{
				size_t matrix1ActualRows = this->actualRows;

				size_t finalPosRows = this->finalPosRows;
				size_t finalPosCols = this->finalPosCols;

				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						_mm256_store_pd(&data1[j * matrix1ActualRows + i], _mm256_log_pd(_mm256_add_pd(one, _mm256_exp_pd(a))));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
							data1[(j + 1) * matrix1ActualRows + i],
							data1[(j + 2) * matrix1ActualRows + i],
							data1[(j + 3) * matrix1ActualRows + i]);

						__m256d softplus = _mm256_log_pd(_mm256_add_pd(one, _mm256_exp_pd(a)));

						__m128d val1 = _mm256_extractf128_pd(softplus, 1);
						__m128d val2 = _mm256_castpd256_pd128(softplus);

						_mm_store_sd(&data1[j * matrix1ActualRows + i], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&data1[(j + 1) * matrix1ActualRows + i], val2);

						_mm_store_sd(&data1[(j + 2) * matrix1ActualRows + i], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&data1[(j + 3) * matrix1ActualRows + i], val1);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						data1[j * matrix1ActualRows + i] = std::log(1.0 + std::exp(data1[j * matrix1ActualRows + i]));
					}
				}
			}
		}
		else
		{
			if constexpr (thisContiguous)
			{
				size_t size = this->_size;
				size_t finalPosSize = this->finalPosSize;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);
					_mm256_store_pd(&data1[i], _mm256_log_pd(_mm256_add_pd(one, _mm256_exp_pd(a))));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] = std::log(1.0 + std::exp(data1[i]));
				}
			}
			else
			{
				size_t matrix1ActualCols = this->actualCols;

				size_t finalPosCols = this->finalPosCols;
				size_t finalPosRows = this->finalPosRows;

				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						_mm256_store_pd(&data1[i * matrix1ActualCols + j], _mm256_log_pd(_mm256_add_pd(one, _mm256_exp_pd(a))));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
							data1[(i + 1) * matrix1ActualCols + j],
							data1[(i + 2) * matrix1ActualCols + j],
							data1[(i + 3) * matrix1ActualCols + j]);

						__m256d softplus = _mm256_log_pd(_mm256_add_pd(one, _mm256_exp_pd(a)));

						__m128d val1 = _mm256_extractf128_pd(softplus, 1);
						__m128d val2 = _mm256_castpd256_pd128(softplus);

						_mm_store_sd(&data1[i * matrix1ActualCols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&data1[(i + 1) * matrix1ActualCols + j], val2);

						_mm_store_sd(&data1[(i + 2) * matrix1ActualCols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&data1[(i + 3) * matrix1ActualCols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						data1[i * matrix1ActualCols + j] = std::log(1.0 + std::exp(data1[i * matrix1ActualCols + j]));
					}
				}
			}
		}
	}

	// Tanh

	template <bool thisTransposed, bool thisContiguous>
	template<bool returnTransposed>
	inline matrix<double> matrix<double, thisTransposed, thisContiguous>::tanh()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		size_t finalPosSize = this->finalPosSize;
		size_t size = this->_size;

		size_t matrix1ActualRows = this->actualRows;
		size_t matrix1ActualCols = this->actualCols;

		size_t finalPosRows = this->finalPosRows;
		size_t finalPosCols = this->finalPosCols;

		if constexpr (returnTransposed)
		{
			matrix<double> result(cols, rows);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);

						_mm256_store_pd(&dataResult[i], _mm256_tanh_pd(a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = std::tanh(data1[i]);
					}
				}
				else
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						for (size_t j = 0; j < cols; j++)
						{
							__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

							_mm256_store_pd(&dataResult[j * rows + i], _mm256_tanh_pd(a));
						}
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						for (size_t j = 0; j < finalPosCols; j += 4)
						{
							__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
								data1[(j + 1) * matrix1ActualRows + i],
								data1[(j + 2) * matrix1ActualRows + i],
								data1[(j + 3) * matrix1ActualRows + i]);

							__m256d exp = _mm256_tanh_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&dataResult[j * rows + i], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

							_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
						}
						for (size_t j = finalPosCols; j < cols; j++)
						{
							dataResult[j * rows + i] = std::tanh(data1[j * matrix1ActualRows + i]);
						}
					}
				}
			}
			else
			{
				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
							data1[(i + 1) * matrix1ActualCols + j],
							data1[(i + 2) * matrix1ActualCols + j],
							data1[(i + 3) * matrix1ActualCols + j]);

						_mm256_store_pd(&dataResult[j * rows + i], _mm256_tanh_pd(a));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						__m256d exp = _mm256_tanh_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&dataResult[j * rows + i], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(j + 1) * rows + i], val2);

						_mm_store_sd(&dataResult[(j + 2) * rows + i], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(j + 3) * rows + i], val1);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						dataResult[j * rows + i] = std::tanh(data1[i * matrix1ActualCols + j]);
					}
				}
			}
			return result;
		}
		else
		{
			matrix<double> result(rows, cols);

			double* dataResult = result._data;

			if constexpr (thisTransposed)
			{
				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
							data1[(j + 1) * matrix1ActualRows + i],
							data1[(j + 2) * matrix1ActualRows + i],
							data1[(j + 3) * matrix1ActualRows + i]);
						_mm256_store_pd(&dataResult[i * cols + j], _mm256_tanh_pd(a));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						__m256d exp = _mm256_tanh_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&dataResult[i * cols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

						_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						dataResult[i * cols + j] = std::tanh(data1[j * matrix1ActualRows + i]);
					}
				}
			}
			else
			{
				if constexpr (thisContiguous)
				{
					for (size_t i = 0; i < finalPosSize; i += 4)
					{
						__m256d a = _mm256_load_pd(&data1[i]);
						_mm256_store_pd(&dataResult[i], _mm256_tanh_pd(a));
					}
					for (size_t i = finalPosSize; i < size; i++)
					{
						dataResult[i] = std::tanh(data1[i]);
					}
				}
				else
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						for (size_t i = 0; i < rows; i++)
						{
							__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

							_mm256_store_pd(&dataResult[i * cols + j], _mm256_tanh_pd(a));
						}
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						for (size_t i = 0; i < finalPosRows; i += 4)
						{
							__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
								data1[(i + 1) * matrix1ActualCols + j],
								data1[(i + 2) * matrix1ActualCols + j],
								data1[(i + 3) * matrix1ActualCols + j]);

							__m256d exp = _mm256_tanh_pd(a);

							__m128d val1 = _mm256_extractf128_pd(exp, 1);
							__m128d val2 = _mm256_castpd256_pd128(exp);

							_mm_store_sd(&dataResult[i * cols + j], val2);
							val2 = _mm_shuffle_pd(val2, val2, 1);
							_mm_store_sd(&dataResult[(i + 1) * cols + j], val2);

							_mm_store_sd(&dataResult[(i + 2) * cols + j], val1);
							val1 = _mm_shuffle_pd(val1, val1, 1);
							_mm_store_sd(&dataResult[(i + 3) * cols + j], val1);
						}
						for (size_t i = finalPosRows; i < rows; i++)
						{
							dataResult[i * cols + j] = std::tanh(data1[i * matrix1ActualCols + j]);
						}
					}
				}
			}
			return result;
		}
	}

	template <bool thisTransposed, bool thisContiguous>
	inline void matrix<double, thisTransposed, thisContiguous>::self_tanh()
	{
		size_t rows = this->_rows;
		size_t cols = this->_cols;

		double* data1 = this->_data;

		if constexpr (thisTransposed)
		{
			if constexpr (thisContiguous)
			{
				size_t finalPosSize = this->finalPosSize;
				size_t size = this->_size;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);

					_mm256_store_pd(&data1[i], _mm256_tanh_pd(a));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] = std::tanh(data1[i]);
				}
			}
			else
			{
				size_t matrix1ActualRows = this->actualRows;

				size_t finalPosRows = this->finalPosRows;
				size_t finalPosCols = this->finalPosCols;

				for (size_t i = 0; i < finalPosRows; i += 4)
				{
					for (size_t j = 0; j < cols; j++)
					{
						__m256d a = _mm256_load_pd(&data1[j * matrix1ActualRows + i]);

						_mm256_store_pd(&data1[j * matrix1ActualRows + i], _mm256_tanh_pd(a));
					}
				}
				for (size_t i = finalPosRows; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256d a = _mm256_setr_pd(data1[j * matrix1ActualRows + i],
							data1[(j + 1) * matrix1ActualRows + i],
							data1[(j + 2) * matrix1ActualRows + i],
							data1[(j + 3) * matrix1ActualRows + i]);

						__m256d exp = _mm256_tanh_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&data1[j * matrix1ActualRows + i], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&data1[(j + 1) * matrix1ActualRows + i], val2);

						_mm_store_sd(&data1[(j + 2) * matrix1ActualRows + i], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&data1[(j + 3) * matrix1ActualRows + i], val1);
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						data1[j * matrix1ActualRows + i] = std::tanh(data1[j * matrix1ActualRows + i]);
					}
				}
			}
		}
		else
		{
			if constexpr (thisContiguous)
			{
				size_t size = this->_size;
				size_t finalPosSize = this->finalPosSize;

				for (size_t i = 0; i < finalPosSize; i += 4)
				{
					__m256d a = _mm256_load_pd(&data1[i]);
					_mm256_store_pd(&data1[i], _mm256_tanh_pd(a));
				}
				for (size_t i = finalPosSize; i < size; i++)
				{
					data1[i] = std::tanh(data1[i]);
				}
			}
			else
			{
				size_t matrix1ActualCols = this->actualCols;

				size_t finalPosCols = this->finalPosCols;
				size_t finalPosRows = this->finalPosRows;

				for (size_t j = 0; j < finalPosCols; j += 4)
				{
					for (size_t i = 0; i < rows; i++)
					{
						__m256d a = _mm256_load_pd(&data1[i * matrix1ActualCols + j]);

						_mm256_store_pd(&data1[i * matrix1ActualCols + j], _mm256_tanh_pd(a));
					}
				}
				for (size_t j = finalPosCols; j < cols; j++)
				{
					for (size_t i = 0; i < finalPosRows; i += 4)
					{
						__m256d a = _mm256_setr_pd(data1[i * matrix1ActualCols + j],
							data1[(i + 1) * matrix1ActualCols + j],
							data1[(i + 2) * matrix1ActualCols + j],
							data1[(i + 3) * matrix1ActualCols + j]);

						__m256d exp = _mm256_tanh_pd(a);

						__m128d val1 = _mm256_extractf128_pd(exp, 1);
						__m128d val2 = _mm256_castpd256_pd128(exp);

						_mm_store_sd(&data1[i * matrix1ActualCols + j], val2);
						val2 = _mm_shuffle_pd(val2, val2, 1);
						_mm_store_sd(&data1[(i + 1) * matrix1ActualCols + j], val2);

						_mm_store_sd(&data1[(i + 2) * matrix1ActualCols + j], val1);
						val1 = _mm_shuffle_pd(val1, val1, 1);
						_mm_store_sd(&data1[(i + 3) * matrix1ActualCols + j], val1);
					}
					for (size_t i = finalPosRows; i < rows; i++)
					{
						data1[i * matrix1ActualCols + j] = std::tanh(data1[i * matrix1ActualCols + j]);
					}
				}
			}
		}
	}

	// Cast

	template <bool thisTransposed, bool thisContiguous>
	template <typename T>
	inline matrix<T> matrix<double, thisTransposed, thisContiguous>::cast()
	{
		size_t cols = this->_cols;
		size_t rows = this->_rows;

		matrix<T> result(rows, cols);

		double* data1 = this->_data;

		T* dataResult = result._data;

		size_t actualCols = this->actualCols;
		size_t actualRows = this->actualRows;

		if constexpr (std::is_same<T, uint8_t>::value)
		{
			if constexpr (thisTransposed)
			{
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						dataResult[i * cols + j] = data1[j * actualRows + j] != 0.0 ? True : False;
					}
				}
			}
			else
			{
				__m256d zero = _mm256_setzero_pd();
				size_t finalPosCols = this->finalPosCols;

				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						__m256i mask = _mm256_castpd_si256(_mm256_cmp_pd(_mm256_load_pd(&data1[i * actualCols + j]), zero, _CMP_NEQ_OQ));

						__m128i mask1 = _mm256_castsi256_si128(mask);
						__m128i mask2 = _mm256_extracti128_si256(mask, 1);

						mask1 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask1), 0b01111000)), 3);
						mask2 = _mm_srli_si128(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(mask2), 0b01111000)), 1);

						__m128i maskResult = _mm_blend_epi16(mask1, mask2, 0b10);

						_mm_store_ss(reinterpret_cast<float*>(&dataResult[i * cols + j]), _mm_castsi128_ps(maskResult));
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						dataResult[i * cols + j] = data1[i * actualCols + j] != 0.0 ? True : False;
					}
				}
			}
		}
		else if constexpr (std::is_same<T, float>::value)
		{
			if constexpr (thisTransposed)
			{
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						dataResult[i * cols + j] = static_cast<float>(data1[j * actualRows + j]);
					}
				}
			}
			else
			{
				size_t finalPosCols = this->finalPosCols;

				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						_mm_store_ps(&dataResult[i * cols + j], _mm256_cvtpd_ps(_mm256_load_pd(&data1[i * actualCols + j])));
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						dataResult[i * cols + j] = static_cast<float>(data1[i * actualCols + j]);
					}
				}
			}
		}
		else if constexpr (std::is_same<T, int>::value)
		{
			if constexpr (thisTransposed)
			{
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						dataResult[i * cols + j] = static_cast<int>(data1[j * actualRows + j]);
					}
				}
			}
			else
			{
				size_t finalPosCols = this->finalPosCols;

				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < finalPosCols; j += 4)
					{
						_mm_store_ps(&dataResult[i * cols + j], _mm256_cvtpd_epi32(_mm256_load_pd(&data1[i * actualCols + j])));
					}
					for (size_t j = finalPosCols; j < cols; j++)
					{
						double data = data1[i * actualCols + j];
						data += 6755399441055744.0;
						dataResult[i * cols + j] = data;
					}
				}
			}
		}
		else
		{
			if constexpr (thisTransposed)
			{
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						dataResult[i * cols + j] = static_cast<T>(data1[j * actualRows + j]);
					}
				}
			}
			else
			{
				size_t finalPosCols = this->finalPosCols;

				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = finalPosCols; j < cols; j++)
					{
						dataResult[i * cols + j] = static_cast<T>(data1[i * actualCols + j]);
					}
				}
			}
		}
		return result;
	}

}
