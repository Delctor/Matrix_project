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
	template<typename T>
	inline vector<T> concatenate(vector<T>& vector1, vector<T>& vector2)
	{
		size_t vector1Size = vector1._size;
		size_t vector2Size = vector2._size;

		T* data1 = vector1._data;
		T* data2 = vector2._data;

		size_t size = vector1Size + vector2Size;

		vector<T> result(size);

		T* dataResult = result._data;

		for (size_t i = 0; i < vector1Size; i++)
		{
			dataResult[i] = data1[i];
		}
		for (size_t iResult{ vector1Size }, iVector2{ 0 }; iResult < size; iResult++, iVector2++)
		{
			dataResult[iResult] = data2[iVector2];
		}
		return result;
	}

	template<bool returnTransposed = false, typename T, bool matrix1Transposed, bool matrix1Contiguous,
		bool matrix2Transposed, bool matrix2Contiguous>
	inline matrix<T> concatenate_rowwise(matrix<T, matrix1Transposed, matrix1Contiguous>& matrix1, matrix<T, matrix2Transposed, matrix2Contiguous>& matrix2)
	{
#ifdef _DEBUG
		if (matrix1._cols != matrix2._cols) throw std::invalid_argument("Wrong dimensions");
#else
#endif

		size_t matrix1Cols = matrix1._cols;
		size_t matrix1Rows = matrix1._rows;

		size_t matrix2Cols = matrix2._cols;
		size_t matrix2Rows = matrix2._rows;

		T* data1 = matrix1._data;
		T* data2 = matrix2._data;

		size_t rows = matrix1Rows + matrix2Rows;
		size_t cols = matrix1Cols;

		matrix<T> result(rows, cols);

		double* dataResult = result._data;

		if constexpr (matrix1Transposed)
		{
			size_t matrix1ActualRows = matrix1.actualRows;
			if constexpr (matrix2Transposed)
			{
				size_t matrix2ActualRows = matrix2.actualRows;
				if constexpr (returnTransposed)
				{
					for (size_t j = 0; j < cols; j++)
					{
						for (size_t i = 0; i < matrix1Rows; i++)
						{
							dataResult[j * rows + i] = data1[j * matrix1ActualRows + i];
						}
						for (size_t iResult{ matrix1Rows }, iMatrix2{ 0 }; iResult < rows; iResult++, iMatrix2++)
						{
							dataResult[j * rows + iResult] = data2[j * matrix2ActualRows + iMatrix2];
						}
					}
				}
				else
				{
					for (size_t j = 0; j < cols; j++)
					{
						for (size_t i = 0; i < matrix1Rows; i++)
						{
							dataResult[i * cols + j] = data1[j * matrix1ActualRows + i];
						}
						for (size_t iResult{ matrix1Rows }, iMatrix2{ 0 }; iResult < rows; iResult++, iMatrix2++)
						{
							dataResult[iResult * cols + j] = data2[j * matrix2ActualRows + iMatrix2];
						}
					}
				}
			}
			else
			{
				size_t matrix2ActualCols = matrix2.actualCols;
				if constexpr (returnTransposed)
				{
					for (size_t j = 0; j < cols; j++)
					{
						for (size_t i = 0; i < matrix1Rows; i++)
						{
							dataResult[j * rows + i] = data1[j * matrix1ActualRows + i];
						}
						for (size_t iResult{ matrix1Rows }, iMatrix2{ 0 }; iResult < rows; iResult++, iMatrix2++)
						{
							dataResult[j * rows + iResult] = data2[iMatrix2 * matrix2ActualCols + j];
						}
					}
				}
				else
				{
					for (size_t j = 0; j < cols; j++)
					{
						for (size_t i = 0; i < matrix1Rows; i++)
						{
							dataResult[i * cols + j] = data1[j * matrix1ActualRows + i];
						}
						for (size_t iResult{ matrix1Rows }, iMatrix2{ 0 }; iResult < rows; iResult++, iMatrix2++)
						{
							dataResult[iResult * cols + j] = data2[iMatrix2 * matrix2ActualCols + j];
						}
					}
				}
			}
		}
		else
		{
			size_t matrix1ActualCols = matrix1.actualCols;
			if constexpr (matrix2Transposed)
			{
				size_t matrix2ActualRows = matrix2.actualRows;
				if constexpr (returnTransposed)
				{
					for (size_t j = 0; j < cols; j++)
					{
						for (size_t i = 0; i < matrix1Rows; i++)
						{
							dataResult[j * rows + i] = data1[i * matrix1ActualCols + j];
						}
						for (size_t iResult{ matrix1Rows }, iMatrix2{ 0 }; iResult < rows; iResult++, iMatrix2++)
						{
							dataResult[j * rows + iResult] = data2[j * matrix2ActualRows + iMatrix2];
						}
					}
				}
				else
				{
					for (size_t j = 0; j < cols; j++)
					{
						for (size_t i = 0; i < matrix1Rows; i++)
						{
							dataResult[i * cols + j] = data1[i * matrix1ActualCols + j];
						}
						for (size_t iResult{ matrix1Rows }, iMatrix2{ 0 }; iResult < rows; iResult++, iMatrix2++)
						{
							dataResult[iResult * cols + j] = data2[j * matrix2ActualRows + iMatrix2];
						}
					}
				}
			}
			else
			{
				size_t matrix2ActualCols = matrix2.actualCols;
				if constexpr (returnTransposed)
				{
					for (size_t j = 0; j < cols; j++)
					{
						for (size_t i = 0; i < matrix1Rows; i++)
						{
							dataResult[j * rows + i] = data1[i * matrix1ActualCols + j];
						}
						for (size_t iResult{ matrix1Rows }, iMatrix2{ 0 }; iResult < rows; iResult++, iMatrix2++)
						{
							dataResult[j * rows + iResult] = data2[iMatrix2 * matrix2ActualCols + j];
						}
					}
				}
				else
				{
					for (size_t j = 0; j < cols; j++)
					{
						for (size_t i = 0; i < matrix1Rows; i++)
						{
							dataResult[i * cols + j] = data1[i * matrix1ActualCols + j];
						}
						for (size_t iResult{ matrix1Rows }, iMatrix2{ 0 }; iResult < rows; iResult++, iMatrix2++)
						{
							dataResult[iResult * cols + j] = data2[iMatrix2 * matrix2ActualCols + j];
						}
					}
				}
			}
		}
		return result;
	}

	template<bool returnTransposed = false, typename T, bool matrix1Transposed, bool matrix1Contiguous,
		bool matrix2Transposed, bool matrix2Contiguous>
	inline matrix<T> concatenate_colwise(matrix<T, matrix1Transposed, matrix1Contiguous>& matrix1, matrix<T, matrix2Transposed, matrix2Contiguous>& matrix2)
	{
#ifdef _DEBUG
		if (matrix1._rows != matrix2._rows) throw std::invalid_argument("Wrong dimensions");
#else
#endif

		size_t matrix1Cols = matrix1._cols;
		size_t matrix1Rows = matrix1._rows;

		size_t matrix2Cols = matrix2._cols;
		size_t matrix2Rows = matrix2._rows;

		T* data1 = matrix1._data;
		T* data2 = matrix2._data;

		size_t rows = matrix1Rows;
		size_t cols = matrix1Cols + matrix2Cols;

		matrix<T> result(rows, cols);

		double* dataResult = result._data;

		if constexpr (matrix1Transposed)
		{
			size_t matrix1ActualRows = matrix1.actualRows;
			if constexpr (matrix2Transposed)
			{
				size_t matrix2ActualRows = matrix2.actualRows;
				if constexpr (returnTransposed)
				{
					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < matrix1Cols; j++)
						{
							dataResult[j * rows + i] = data1[j * matrix1ActualRows + i];
						}
						for (size_t jResult{ matrix1Cols }, jMatrix2{ 0 }; jResult < cols; jResult++, jMatrix2++)
						{
							dataResult[jResult * rows + i] = data2[jMatrix2 * matrix2ActualRows + i];
						}
					}
				}
				else
				{
					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < matrix1Cols; j++)
						{
							dataResult[i * cols + j] = data1[j * matrix1ActualRows + i];
						}
						for (size_t jResult{ matrix1Cols }, jMatrix2{ 0 }; jResult < cols; jResult++, jMatrix2++)
						{
							dataResult[i * cols + jResult] = data2[jMatrix2 * matrix2ActualRows + i];
						}
					}
				}
			}
			else
			{
				size_t matrix2ActualCols = matrix2.actualCols;
				if constexpr (returnTransposed)
				{
					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < matrix1Cols; j++)
						{
							dataResult[j * rows + i] = data1[j * matrix1ActualRows + i];
						}
						for (size_t jResult{ matrix1Cols }, jMatrix2{ 0 }; jResult < cols; jResult++, jMatrix2++)
						{
							dataResult[jResult * rows + i] = data2[i * matrix2ActualCols + jMatrix2];
						}
					}
				}
				else
				{
					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < matrix1Cols; j++)
						{
							dataResult[i * cols + j] = data1[j * matrix1ActualRows + i];
						}
						for (size_t jResult{ matrix1Cols }, jMatrix2{ 0 }; jResult < cols; jResult++, jMatrix2++)
						{
							dataResult[i * cols + jResult] = data2[i * matrix2ActualCols + jMatrix2];
						}
					}
				}
			}
		}
		else
		{
			size_t matrix1ActualCols = matrix1.actualCols;
			if constexpr (matrix2Transposed)
			{
				size_t matrix2ActualRows = matrix2.actualRows;
				if constexpr (returnTransposed)
				{
					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < matrix1Cols; j++)
						{
							dataResult[j * rows + i] = data1[i * matrix1ActualCols + j];
						}
						for (size_t jResult{ matrix1Cols }, jMatrix2{ 0 }; jResult < cols; jResult++, jMatrix2++)
						{
							dataResult[jResult * rows + i] = data2[jMatrix2 * matrix2ActualRows + i];
						}
					}
				}
				else
				{
					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < matrix1Cols; j++)
						{
							dataResult[i * cols + j] = data1[i * matrix1ActualCols + j];
						}
						for (size_t jResult{ matrix1Cols }, jMatrix2{ 0 }; jResult < cols; jResult++, jMatrix2++)
						{
							dataResult[i * cols + jResult] = data2[jMatrix2 * matrix2ActualRows + i];
						}
					}
				}
			}
			else
			{
				size_t matrix2ActualCols = matrix2.actualCols;
				if constexpr (returnTransposed)
				{
					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < matrix1Cols; j++)
						{
							dataResult[j * rows + i] = data1[i * matrix1ActualCols + j];
						}
						for (size_t jResult{ matrix1Cols }, jMatrix2{ 0 }; jResult < cols; jResult++, jMatrix2++)
						{
							dataResult[jResult * rows + i] = data2[i * matrix2ActualCols + jMatrix2];
						}
					}
				}
				else
				{
					for (size_t i = 0; i < rows; i++)
					{
						for (size_t j = 0; j < matrix1Cols; j++)
						{
							dataResult[i * cols + j] = data1[i * matrix1ActualCols + j];
						}
						for (size_t jResult{ matrix1Cols }, jMatrix2{ 0 }; jResult < cols; jResult++, jMatrix2++)
						{
							dataResult[i * cols + jResult] = data2[i * matrix2ActualCols + jMatrix2];
						}
					}
				}
			}
		}
		return result;
	}

}