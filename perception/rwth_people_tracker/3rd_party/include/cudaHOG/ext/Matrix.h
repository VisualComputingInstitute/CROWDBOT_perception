//    Copyright (c) 2009
//      Dennis Mitzel <mitzel@umic.rwth-aachen.de>
//      RWTH Aachen University, Germany
//
//    This file is part of groundHOG.
//
//    GroundHOG is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    GroundHOG is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with groundHOG.  If not, see <http://www.gnu.org/licenses/>.
//

#ifndef _DENNIS_MATRIX_H
#define	_DENNIS_MATRIX_H
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string.h>
#include <cassert>
#include "Vector.h"

namespace cudaHOG{
using namespace std;
//+++++++++++++++++++++++++++++++ Definition +++++++++++++++++++++++++++++++++
template <class T>
class Matrix
{
public:


    Matrix();
    Matrix(const Matrix<T>& copy);
    Matrix(const int xSizeO, const int ySizeO);
    Matrix(const int xSizeO, const int ySizeO, const T fillValO);
    Matrix(const int xSizeO, const int ySizeO, T* data);

    ~Matrix();

    inline T& operator()(const int xO, const int yO);
    inline T operator()(const int xO, const int yO) const;

    Matrix<T>& operator+=(const Matrix<T>& MatO);
    Matrix<T>& operator-=(const Matrix<T>& MatO);

    Matrix<T>& operator+=(const T scalar);
    Matrix<T>& operator*=(const Matrix<T>& MatO);
    Matrix<T>& operator*=(const Vector<T>& vecO);

    Matrix<T>& operator*=(const T scalar);
    Matrix<T>& operator=(const T aValue);
    Matrix<T>& operator=(const Matrix<T>& CopyO);

    void transposed();
    void show();


    int xSize();
    int ySize();
    int size();
    T* data();
    void inv();

    void fillColumn(const T fillValueO, const int colNumO);
    void fill(const T fillvalue);

    void cutPart(int startC, int endC, int startR, int endR, Matrix<T>& target);
    void getColumn(Vector<T>& target, const int columnNumber);
    void getRow(Vector<T>& target, const int rowNumber);
    void insertRow(Vector<T>& src, const int rowNumber);
    void sumAlongAxisX(Vector<T>& resultO);
    void sumAlongAxisY(Vector<T>& resultO);
    void setSize( const int xO,  const int yO);
    void setSize( const int xO,  const int yO, const T fillValO);
    void setSize(const int xSizeO, const int ySizeO, T* data);
    void inv22();
    void append(Matrix<T>& newPart);
    void swap();
    void transferStlVecVecToMat(vector<vector<T> >& objO);
    void transferMatToStlStlVec(vector<vector<T> >& res);
    void subVectorOfAllRows(Vector<T>& vecO);
    void diag(Vector<T> vecO);
    void eye(int val);

protected:

    int xSizeC;
    int ySizeC;
    T *dataC;

};
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//+++++++++++++++++++++++++++++++++ Implementation +++++++++++++++++++++++++++

template <class T> Matrix<T>::Matrix()
{
    xSizeC = 0;
    ySizeC = 0;
    dataC = 0;
}

template <class T> Matrix<T>::Matrix(const int xSizeO, const int ySizeO, T* data)
{
    xSizeC = xSizeO;
    ySizeC = ySizeO;
    dataC = new T[xSizeO*ySizeO];

    int whSize = xSizeC*ySizeC;
    memcpy(dataC, data, sizeof(T)*whSize);
//    for(int i = 0; i < whSize; i++)
//    {
//        dataC[i] = data[i];
//    }
}


template <class T> Matrix<T>::Matrix(const int xSizeO, const int ySizeO)
{
    xSizeC = xSizeO;
    ySizeC = ySizeO;
    dataC = new T[xSizeO*ySizeO];
}

template <class T> Matrix<T>::Matrix(const int xSizeO, const int ySizeO, const T fillVal0)
{
    xSizeC = xSizeO;
    ySizeC = ySizeO;
    int whSize = xSizeO*ySizeO;
    dataC = new T[whSize];
//
//
//    for (int i = 0; i < whSize; i++)
//    {
//        dataC[i] = fillVal0;
//    }

    std::fill(dataC, dataC+whSize, fillVal0);

}

template <class T> Matrix<T>::Matrix(const Matrix<T>& copy)
{
    xSizeC = copy.xSizeC;
    ySizeC = copy.ySizeC;
    int whSize = xSizeC*ySizeC;
    dataC = new T[whSize];

    memcpy(dataC, copy.dataC, sizeof(T)*whSize);
//    for ( int i = 0; i < whSize; i++)
//    {
//      dataC[i] = copy.dataC[i];
//    }
}

template <class T> Matrix<T>::~Matrix()
{
    if (dataC != 0)
        delete [] dataC;
    dataC = 0;
}

// operators /////
template <class T> inline T& Matrix<T>::operator()(int xO, int yO)
{
   assert(xO < xSizeC && yO < ySizeC && xO >= 0 && yO >= 0);

  return dataC[xSizeC*yO+xO];
}

template <class T> inline T Matrix<T>::operator()(int xO, int yO) const
{
   assert(xO < xSizeC && yO < ySizeC && xO >= 0 && yO >= 0);

  return dataC[xSizeC*yO+xO];
}

template <class T> Matrix<T>& Matrix<T>::operator=(const T fillVal0)
{
  int whSize = xSizeC*ySizeC;

//  for (int i = 0; i < whSize; i++)
//  {
//      dataC[i] = fillVal0;
//  }
  std::fill(dataC, dataC + whSize, fillVal0);
  return *this;
}

template <class T> Matrix<T>& Matrix<T>::operator=(const Matrix<T>& CopyO) {
  if (this != &CopyO)
  {
        delete[] dataC;

        xSizeC = CopyO.xSizeC;
        ySizeC = CopyO.ySizeC;
        int whSize = xSizeC*ySizeC;
        dataC = new T[whSize];

        memcpy(dataC, CopyO.dataC, sizeof (T) * whSize);
  }
  return *this;
}

template <class T> Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& MatO) {
  assert ((xSizeC == MatO.xSizeC) && (ySizeC == MatO.ySizeC));

  for (int i = 0; i < xSizeC*ySizeC; i++)
    dataC[i] += MatO.dataC[i];
  return *this;
}

template <class T> Matrix<T>& Matrix<T>::operator-=(const Matrix<T>& MatO) {
  assert ((xSizeC == MatO.xSizeC) && (ySizeC == MatO.ySizeC));

  for (int i = 0; i < xSizeC*ySizeC; i++)
    dataC[i] -= MatO.dataC[i];
  return *this;
}

template <class T> Matrix<T>& Matrix<T>::operator*=(const Matrix<T>& MatO) {
  assert ((xSizeC == MatO.ySizeC));
  assert (this != &MatO);
  T* copy = new T[xSizeC*ySizeC];
  for (int i = 0; i < xSizeC*ySizeC; i++) copy[i] = dataC[i];
  int copyXSize = xSizeC;
  xSizeC = MatO.xSizeC;

  delete [] dataC;

  dataC = new T[ySizeC * MatO.xSizeC];
  for (int i = 0; i < ySizeC; i++){
      for(int j = 0; j < MatO.xSizeC; j++){
          dataC[MatO.xSizeC*i + j] = 0;
          for (int l = 0; l < copyXSize; l++){
              dataC[MatO.xSizeC*i + j] += copy[copyXSize*i + l]*MatO(j,l);
          }
      }
  }
  delete [] copy;
  return *this;
}

template <class T> Matrix<T>& Matrix<T>::operator+=(const T scalar) {

  for (int i = 0; i < xSizeC*ySizeC; i++)
    dataC[i] += scalar;
  return *this;
}

template <class T> Matrix<T>& Matrix<T>::operator*=(const T scalar) {

  for (int i = 0; i < xSizeC*ySizeC; i++)
    dataC[i] *= scalar;
  return *this;
}


template <class T> int Matrix<T>::ySize(){
  return ySizeC;
}

template <class T> int Matrix<T>::xSize(){
  return xSizeC;
}

template <class T> int Matrix<T>::size(){
  return ySizeC * xSizeC;
}

template <class T> T* Matrix<T>::data(){
  return dataC;
}

template <class T> void Matrix<T>::inv22()
{
    float eps = 0.0000001;
    // check if it is a 2x2 matrix
    assert (xSizeC == 2 && ySizeC == 2);

    T* copy = new T[xSizeC*ySizeC];
    for (int i = 0; i < xSizeC*ySizeC; i++) copy[i] = dataC[i];

    T det = copy[0]*copy[3] - copy[1]*copy[2];
    if (fabs(det) < eps ){
        //cout << "Warning: low value of det, can lead to high nummerical inaccuracy" << endl;
    }

    dataC[0] = (1.0/det)*copy[3];
    dataC[1] = (-1.0/det)*copy[1];
    dataC[2] = (-1.0/det)*copy[2];
    dataC[3] = (1.0/det)*copy[0];

    delete [] copy;

}

template <class T> void Matrix<T>::transposed() {

  Matrix<T> copy(ySizeC, xSizeC);
  for (int i = 0; i < ySizeC; i++)
    for (int j = 0; j < xSizeC; j++)
      copy(i,j) = dataC[i*xSizeC + j];

//  for(int i = 0; i < xSizeC*ySizeC; i++) dataC[i] = copy.data()[i];

  memcpy(dataC, copy.dataC, sizeof(T)*xSizeC*ySizeC);
  ySizeC = copy.ySize();
  xSizeC = copy.xSize();
}


template <class T> void Matrix<T>::show()
{
    for (int i = 0; i < ySizeC; i++){
        printf("Row: %02d | ", i);
        for(int j = 0; j < xSizeC; j++){
            printf("%f ", dataC[i*xSizeC + j]);
        }
        printf("\n");
    }
    printf("\n");
}

template <class T> void Matrix<T>::setSize( const int xO,  const int yO) {
  if (dataC != 0) delete[] dataC;
  dataC = new T[xO*yO];
  xSizeC = xO;
  ySizeC = yO;
}

template <class T> void Matrix<T>::setSize(const int xSizeO, const int ySizeO, T* data)
{
    xSizeC = xSizeO;
    ySizeC = ySizeO;
    dataC = new T[xSizeO*ySizeO];

    memcpy(dataC, data, sizeof(T)*xSizeO*ySizeO);
}

template <class T> void Matrix<T>::setSize( const int xO,  const int yO, const T fillValO) {
  if (dataC != 0) delete[] dataC;
  dataC = new T[xO*yO];
  xSizeC = xO;
  ySizeC = yO;

 //for(int i = 0; i < xSizeC*ySizeC; i++) dataC[i] = fillValO;
  std::fill(dataC, dataC+xSizeC*ySizeC, fillValO);

}

template <class T> Matrix<T>& Matrix<T>::operator*=(const Vector<T>& vecO) {
  assert (xSizeC == vecO.size());

  T* copy = new T[xSizeC*ySizeC];
  for (int i = 0; i < xSizeC*ySizeC; i++) copy[i] = dataC[i];
  delete [] dataC;
  dataC = new T[ySizeC];
  xSizeC = 1;
  for (int i = 0; i < ySizeC; i++){
          dataC[i] = 0;
          for (int l = 0; l < vecO.size(); l++){
              dataC[i] += copy[vecO.size()*i + l]*vecO(l);
      }
  }
  delete [] copy;
  return *this;
}

template <class T> void Matrix<T>::cutPart(int startC, int endC, int startR, int endR, Matrix<T>& target)
{
    assert (startC <= endC && startR <= endR && startC  >= 0 && endC >= 0 && startR >= 0 && endR >= 0 && endC < xSizeC && endR < ySizeC);

    int x = endC-startC + 1;
    int y = endR-startR + 1;

    target.setSize(x, y);

    for (int i = startC; i < endC+1; i++)
    {
        for(int j = startR; j < endR+1; j++)
        {
            target(i-startC, j-startR) = dataC[j*xSizeC + i];
        }
    }
}

template <class T> void Matrix<T>::getColumn(Vector<T>& target, const int colNumber)
{
    assert(colNumber < xSizeC && colNumber >= 0);

    target.setSize(ySizeC);
    for(int i = 0; i < ySizeC; i++)
    {
        target(i) = operator ()(colNumber, i);
    }
}

template <class T> void Matrix<T>::getRow(Vector<T>& target, const int rowNumber)
{
    assert(rowNumber < ySizeC && rowNumber >= 0);

    target.setSize(xSizeC);
    for(int i = 0; i < xSizeC; i++)
    {
        target(i) = operator ()(i, rowNumber);
    }
}

template <class T> void Matrix<T>::insertRow(Vector<T>& src, const int rowNumber)
{
    assert(rowNumber < ySizeC && rowNumber >= 0 && src.size() == xSizeC);

    for(int i = 0; i < xSizeC; i++)
    {
        dataC[rowNumber*xSizeC + i] = src.data()[i];
    }
}

template <class T> void Matrix<T>::sumAlongAxisX(Vector<T>& resultO) {

    resultO.setSize(xSizeC, 0.0);

    for (int i = 0; i < xSizeC; i++) {
        for (int j = 0; j < ySizeC; j++) {
            resultO(i) += operator ()(i, j);
        }
    }
}

template <class T> void Matrix<T>::sumAlongAxisY(Vector<T>& resultO) {

    resultO.setSize(ySizeC, 0.0);

    for (int i = 0; i < xSizeC; i++) {
        for (int j = 0; j < ySizeC; j++) {
            resultO(j) += operator ()(i, j);
        }
    }
}



template <class T> void Matrix<T>::append(Matrix<T>& newPart) {
  assert (newPart.xSize() == xSizeC);

  T* newData = new T[xSizeC*(ySizeC + newPart.ySize())];
  int oldSize = xSizeC*ySizeC;
  for (int i = 0; i < oldSize; i++)
    newData[i] = dataC[i];
  int sizeOfPart = xSizeC*newPart.ySize();
  for (int i = 0; i < sizeOfPart; i++)
    newData[i+oldSize] = newPart.data()[i];
  delete[] dataC;
  dataC = newData;
  ySizeC += newPart.ySize();
}

template <class T> void Matrix<T>::swap()
{
    T* newData = new T[xSizeC*ySizeC];

    for(int i = 0; i < ySizeC; i++)
    {
        for(int j = 0; j < xSizeC; j++)
        {
            newData[i*xSizeC + j] = dataC[((ySizeC-1)-i)*xSizeC + j];
        }
    }

    delete [] dataC;
    dataC = newData;
}

template <class T> void Matrix<T>::fillColumn(const T fillValueO, const int colNumO)
{
    assert (colNumO < xSizeC );

    for(int i = 0; i < ySizeC; i++)
    {
        dataC[i*xSizeC + colNumO] = fillValueO;
    }
}

template <class T> void Matrix<T>::transferStlVecVecToMat(vector<vector<T> >& objO)
{
    int whSize = objO.size();
    ySizeC = objO.size();

    if(ySizeC == 0) return;
    xSizeC = objO.at(0).size();


    for(unsigned int i = 0; i < objO.size(); i++)
    {
        whSize +=objO.at(i).size();
    }

    T* newData = new T[whSize];

    for(int i = 0; i < ySizeC; i++)
    {
        for(int j = 0; j < xSizeC; j++)
        {
            newData[i*xSizeC + j] = objO.at(i).at(j);
        }
    }
    delete [] dataC;
    dataC = newData;
}

template <class T> void Matrix<T>::inv() {

    assert(ySizeC == xSizeC);
    int* p = new int[xSizeC];
    T* hv = new T[xSizeC];
    Matrix<T>& I(*this);
    int n = ySizeC;
    for (int j = 0; j < n; j++)
        p[j] = j;
    for (int j = 0; j < n; j++) {
        T max = fabs(I(j, j));
        int r = j;
        for (int i = j + 1; i < n; i++)
            if (fabs(I(j, i)) > max) {
                max = fabs(I(j, i));
                r = i;
            }

        if (max <= 0) return;

        if (r > j) {
            for (int k = 0; k < n; k++) {
                T hr = I(k, j);
                I(k, j) = I(k, r);
                I(k, r) = hr;
            }
            int hi = p[j];
            p[j] = p[r];
            p[r] = hi;
        }
        T hr = 1 / I(j, j);
        for (int i = 0; i < n; i++)
            I(j, i) *= hr;
        I(j, j) = hr;
        hr *= -1;
        for (int k = 0; k < n; k++)
            if (k != j) {
                for (int i = 0; i < n; i++)
                    if (i != j) I(k, i) -= I(j, i) * I(k, j);
                I(k, j) *= hr;
            }
    }
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++)
            hv[p[k]] = I(k, i);
        for (int k = 0; k < n; k++)
            I(k, i) = hv[k];
    }

    delete [] p;
    delete [] hv;
}

template <class T> void Matrix<T>::subVectorOfAllRows(Vector<T>& vecO)
{
    assert(xSizeC == vecO.size());

    for(int i = 0; i < ySizeC; i++)
    {
        for(int j = 0; j < xSizeC; j++)
        {
            dataC[i*xSizeC + j] -= vecO(j);
        }
    }
}


 template <class T> void Matrix<T>::diag(Vector<T> vecO)
 {
     xSizeC = vecO.size();
     ySizeC = xSizeC;

     int whSize = xSizeC*ySizeC;
     if (dataC != 0) delete[] dataC;
     dataC = new T[whSize];

     for(int i = 0; i < whSize; i++)
     {
         if(i % (xSizeC+1) == 0)
         {
             dataC[i] = vecO(int(floor(float(i)/xSizeC)));
         }else
         {
            dataC[i] = 0;
         }
     }
 }

template <class T> void Matrix<T>::fill(const T fillvalue)
{
//    for(int i = 0; i < xSizeC*ySizeC; i++)
//    {
//        dataC[i] = fillvalue;
//    }

    std::fill(dataC, dataC+(xSizeC*ySizeC), fillvalue);
}

 template <class T> void  Matrix<T>::eye(int val)
 {
     this->setSize(val, val, 0.0);

     for(int i = 0; i < val; i++)
     {
             this->operator ()(i,i) = 1;
     }
 }

 template <class T> void  Matrix<T>::transferMatToStlStlVec(vector<vector<T> >& res)
 {
     vector<T> help;
     res.clear();
     for(int i = 0; i < ySizeC; i++)
     {
         help.clear();
         for(int j = 0; j < xSizeC; j++)
         {
             help.push_back(this->operator()(j,i));
         }
         res.push_back(help);
     }
 }

}
#endif	/* _DENNIS_MATRIX_H */

