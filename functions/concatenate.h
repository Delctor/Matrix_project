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

	template<typename T>
	inline vector<T> concatenate(vector<T>** vectors, size_t nVectors, size_t totalSize)
	{
		vector<T> result(totalSize);
		T* dataResult = result._data;
		size_t initialPos = 0;
		for (size_t i = 0; i < nVectors; i++)
		{
			vector<T>* vec = vectors[i];
			
			size_t vecSize = (*vec)._size;

			T* dataVec = (*vec)._data;

			for (size_t j{ 0 }, j2{ initialPos }; j < vecSize; j++, j2++)
			{
				dataResult[j2] = dataVec[j];
			}
			initialPos += vecSize;
		}
		return result;
	}

	template<typename T>
	inline vector<T> concatenate(vector<T>** vectors, size_t nVectors)
	{
		size_t totalSize = 0;

		for (size_t i = 0; i < nVectors; i++) totalSize += (*vectors[i])._size;

		vector<T> result(totalSize);
		T* dataResult = result._data;
		size_t initialPos = 0;
		for (size_t i = 0; i < nVectors; i++)
		{
			vector<T>* vec = vectors[i];

			size_t vecSize = (*vec)._size;

			T* dataVec = (*vec)._data;

			for (size_t j{ 0 }, j2{ initialPos }; j < vecSize; j++, j2++)
			{
				dataResult[j2] = dataVec[j];
			}
			initialPos += vecSize;
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

		size_t matrix1ActualRows = matrix1.actualRows;
		size_t matrix1ActualCols = matrix1.actualRows;
		size_t matrix2ActualRows = matrix2.actualRows;
		size_t matrix2ActualCols = matrix2.actualRows;

		if constexpr (returnTransposed)
		{
			matrix<T> result(cols, rows);

			double* dataResult = result._data;

			if constexpr (matrix1Transposed)
			{
				if constexpr (matrix2Transposed)
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
							dataResult[j * rows + i] = data1[j * matrix1ActualRows + i];
						}
						for (size_t iResult{ matrix1Rows }, iMatrix2{ 0 }; iResult < rows; iResult++, iMatrix2++)
						{
							dataResult[j * rows + iResult] = data2[iMatrix2 * matrix2ActualCols + j];
						}
					}
				}
			}
			else
			{
				if constexpr (matrix2Transposed)
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
							dataResult[j * rows + i] = data1[i * matrix1ActualCols + j];
						}
						for (size_t iResult{ matrix1Rows }, iMatrix2{ 0 }; iResult < rows; iResult++, iMatrix2++)
						{
							dataResult[j * rows + iResult] = data2[iMatrix2 * matrix2ActualCols + j];
						}
					}
				}
			}
			return result;
		}
		else
		{
			matrix<T> result(rows, cols);

			double* dataResult = result._data;

			if constexpr (matrix1Transposed)
			{
				if constexpr (matrix2Transposed)
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
			else
			{
				if constexpr (matrix2Transposed)
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
			return result;
		}
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

		size_t matrix1ActualRows = matrix1.actualRows;
		size_t matrix1ActualCols = matrix1.actualRows;
		size_t matrix2ActualRows = matrix2.actualRows;
		size_t matrix2ActualCols = matrix2.actualRows;

		if constexpr (returnTransposed)
		{
			matrix<T> result(cols, rows);

			double* dataResult = result._data;

			if constexpr (matrix1Transposed)
			{
				if constexpr (matrix2Transposed)
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
							dataResult[j * rows + i] = data1[j * matrix1ActualRows + i];
						}
						for (size_t jResult{ matrix1Cols }, jMatrix2{ 0 }; jResult < cols; jResult++, jMatrix2++)
						{
							dataResult[jResult * rows + i] = data2[i * matrix2ActualCols + jMatrix2];
						}
					}
				}
			}
			else
			{
				if constexpr (matrix2Transposed)
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
							dataResult[j * rows + i] = data1[i * matrix1ActualCols + j];
						}
						for (size_t jResult{ matrix1Cols }, jMatrix2{ 0 }; jResult < cols; jResult++, jMatrix2++)
						{
							dataResult[jResult * rows + i] = data2[i * matrix2ActualCols + jMatrix2];
						}
					}
				}
			}
			return result;
		}
		else
		{
			matrix<T> result(rows, cols);

			double* dataResult = result._data;

			if constexpr (matrix1Transposed)
			{
				if constexpr (matrix2Transposed)
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
			else
			{
				if constexpr (matrix2Transposed)
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
			return result;
		}
	}

	template<typename T, bool returnTransposed = false>
	inline matrix<T> concatenate_rowwise(void** matrices, size_t nMatrices, size_t cols, size_t totalRows)
	{
		size_t initialRow = 0;

		if constexpr (returnTransposed)
		{
			matrix<T> result(cols, totalRows);

			T* dataResult = result._data;

			for (size_t i = 0; i < nMatrices; i++)
			{
				matrix<T, false, true>* matrixPtr = reinterpret_cast<alge::matrix<T, false, true>*>(matrices[i]);

				if ((*matrixPtr).transposed)
				{
					size_t matrixRows = (*matrixPtr)._rows;

					T* dataMatrix = (*matrixPtr)._data;

					size_t actualRows = (*matrixPtr).actualRows;

					for (size_t j{ 0 }, j2{ initialRow }; j < matrixRows; j++, j2++)
					{
						for (size_t k = 0; k < cols; k++)
						{
							dataResult[k * totalRows + j2] = dataMatrix[k * actualRows + j];
						}
					}
					initialRow += matrixRows;
				}
				else
				{
					size_t matrixRows = (*matrixPtr)._rows;

					T* dataMatrix = (*matrixPtr)._data;

					size_t actualCols = (*matrixPtr).actualCols;

					for (size_t j{ 0 }, j2{ initialRow }; j < matrixRows; j++, j2++)
					{
						for (size_t k = 0; k < cols; k++)
						{
							dataResult[k * totalRows + j2] = dataMatrix[j * actualCols + k];
						}
					}
					initialRow += matrixRows;
				}
			}
			return result;
		}
		else
		{
			matrix<T> result(totalRows, cols);

			T* dataResult = result._data;

			for (size_t i = 0; i < nMatrices; i++)
			{
				matrix<T, false, true>* matrixPtr = reinterpret_cast<alge::matrix<T, false, true>*>(matrices[i]);

				if ((*matrixPtr).transposed)
				{
					size_t matrixRows = (*matrixPtr)._rows;

					T* dataMatrix = (*matrixPtr)._data;

					size_t actualRows = (*matrixPtr).actualRows;

					for (size_t j{ 0 }, j2{ initialRow }; j < matrixRows; j++, j2++)
					{
						for (size_t k = 0; k < cols; k++)
						{
							dataResult[j2 * cols + k] = dataMatrix[k * actualRows + j];
						}
					}
					initialRow += matrixRows;
				}
				else
				{
					size_t matrixRows = (*matrixPtr)._rows;

					T* dataMatrix = (*matrixPtr)._data;

					size_t actualCols = (*matrixPtr).actualCols;

					for (size_t j{ 0 }, j2{ initialRow }; j < matrixRows; j++, j2++)
					{
						for (size_t k = 0; k < cols; k++)
						{
							dataResult[j2 * cols + k] = dataMatrix[j * actualCols + k];
						}
					}
					initialRow += matrixRows;
				}
			}
			return result;
		}
	}

	template<typename T, bool returnTransposed = false>
	inline matrix<T> concatenate_colwise(void** matrices, bool* transposed, size_t nMatrices, size_t rows, size_t totalCols)
	{
		size_t initialCol = 0;

		if constexpr (returnTransposed)
		{
			matrix<T> result(rows, totalCols);

			T* dataResult = result._data;

			for (size_t i = 0; i < nMatrices; i++)
			{
				matrix<T, false, true>* matrixPtr = reinterpret_cast<alge::matrix<T, false, true>*>(matrices[i]);

				if ((*matrixPtr).transposed)
				{
					size_t matrixCols = (*matrixPtr)._cols;

					T* dataMatrix = (*matrixPtr)._data;

					size_t actualRows = (*matrixPtr).actualRows;

					for (size_t j{ 0 }, j2{ initialCol }; j < matrixCols; j++, j2++)
					{
						for (size_t k = 0; k < rows; k++)
						{
							dataResult[j2 * rows + k] = dataMatrix[j * actualRows + k];
						}
					}
					initialCol += matrixCols;
				}
				else
				{
					size_t matrixCols = (*matrixPtr)._cols;

					T* dataMatrix = (*matrixPtr)._data;

					size_t actualCols = (*matrixPtr).actualCols;

					for (size_t j{ 0 }, j2{ initialCol }; j < matrixCols; j++, j2++)
					{
						for (size_t k = 0; k < rows; k++)
						{
							dataResult[j2 * rows + k] = dataMatrix[k * actualCols + j];
						}
					}
					initialCol += matrixCols;
				}
			}
			return result;
		}
		else
		{
			matrix<T> result(rows, totalCols);

			T* dataResult = result._data;

			for (size_t i = 0; i < nMatrices; i++)
			{
				matrix<T, false, true>* matrixPtr = reinterpret_cast<alge::matrix<T, false, true>*>(matrices[i]);

				if ((*matrixPtr).transposed)
				{
					size_t matrixCols = (*matrixPtr)._cols;

					T* dataMatrix = (*matrixPtr)._data;

					size_t actualRows = (*matrixPtr).actualRows;

					for (size_t j{ 0 }, j2{ initialCol }; j < matrixCols; j++, j2++)
					{
						for (size_t k = 0; k < rows; k++)
						{
							dataResult[k * totalCols + j2] = dataMatrix[j * actualRows + k];
						}
					}
					initialCol += matrixCols;
				}
				else
				{
					size_t matrixCols = (*matrixPtr)._cols;

					T* dataMatrix = (*matrixPtr)._data;

					size_t actualCols = (*matrixPtr).actualCols;

					for (size_t j{ 0 }, j2{ initialCol }; j < matrixCols; j++, j2++)
					{
						for (size_t k = 0; k < rows; k++)
						{
							dataResult[k * totalCols + j2] = dataMatrix[k * actualCols + j];
						}
					}
					initialCol += matrixCols;
				}
			}
			return result;
		}
	}

}
