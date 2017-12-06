
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



#ifndef _DENNIS_VECTOR_H
#define	_DENNIS_VECTOR_H

#include <vector>
#include <string>
#include <cassert>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <algorithm>


namespace cudaHOG{
using namespace std;

template <class T> class Vector {
public:

    Vector();
    Vector(const Vector<T>& copy);
    Vector(const int sizeO);
    Vector(const int sizeO, const T FillValO);
    Vector(const Vector<T>& vec1, const Vector<T>& vec2);
    Vector(vector<float> vecO);


    ~Vector();

    inline T& operator()(const unsigned int aIndex);
    inline T operator()(const unsigned int aIndex) const;
    Vector<T>& operator=(const Vector<T>& vecO);
    Vector<T>& operator=(const T fillValO);

    Vector<T>& operator*=(const T scalar);
    // Pointwise mult
    Vector<T>& operator*=(const Vector<T>& vecO);
    Vector<T>& operator+=(const Vector<T>& vecO);
    Vector<T>& operator+=(const T value);

    Vector<T>& operator-=(const Vector<T>& vecO);
    Vector<T>& operator-=(const vector<T>& vecO);

    void setSize(int sizeO);
    void setSize(int sizeO, const T fillValO);
    void show();
    void fill(const T valO);
    void connect(Vector<T>& newPartO);
    void reallocate(int addSizeO);
    // change the order of vector values: v(0)->v(end), v(1)->v(end-1)
    void swap();
    // reduze vector size to sizeO and copy the content until the sizeO.
    void reduceCapacity(int sizeO);
    void intersection(Vector<T>& vec2, Vector<T>& intersect);
    void readTXT(char* FileName, int sizeO);
    void transferToSTL(vector<T>& vecO);
    void trasferToMyVec(vector<T>& vecO);

    void smooth(int smoothRange);


    bool compare(Vector<T>& vecO);
    int size() const;

    pair<T, int>  minim();
    pair<T, int>  maxim();
    T norm();
    T* data();
    T sum();

protected:
    unsigned int sizeC;
    T* dataC;

};

// cross product
template <class T> Vector<T> operator/( const Vector<T>& v1,  const Vector<T>& v2);

// dot product
template <class T> T operator*( const Vector<T>& v1, const Vector<T>& v2);

template <class T> void Vector<T>::readTXT(char* aFilename, int sizeO)
{
  ifstream aStream(aFilename);

  sizeC = sizeO;
  delete [] dataC;
  dataC = new T[sizeC];
  for (unsigned int i = 0; i < sizeC ; i++)
  {
    aStream >> dataC[i];
  }
}

template <class T> Vector<T>::Vector() {
    sizeC = 0;
    dataC = new T[1];
}

template <class T> Vector<T>::Vector(const Vector<T>& copy)
{
    sizeC = copy.sizeC;
    dataC = new T[sizeC];

    memcpy(dataC, copy.dataC, sizeof(T)*sizeC);
//    for (int i = 0; i < sizeC; i++)
//      dataC[i] = copy.dataC[i];
}

template <class T> int  Vector<T>::size() const {
    return sizeC;
}

template <class T> Vector<T>::Vector(int sizeO) {
    sizeC = sizeO;
    dataC = new T[sizeO];
}


template <class T> Vector<T>::Vector(const Vector<T>& vec1, const Vector<T>& vec2) {
    sizeC = vec1.sizeC + vec2.sizeC;
    dataC = new T[sizeC];

    for(int i = 0; i < vec1.sizeC; i++)
        dataC[i] = vec1.dataC[i];

    for(int i = vec1.sizeC; i < sizeC; i++)
        dataC[i] = vec2.dataC[i - vec1.sizeC];
}

template <class T> Vector<T>::Vector(vector<float> vecO) {
    sizeC = vecO.size();
    dataC = new T[sizeC];

    for(unsigned int i = 0; i < sizeC; i++) dataC[i] = vecO[i];
}


template <class T> Vector<T>::Vector(const int sizeO, const T FillValO) {
    sizeC = sizeO;
    dataC = new T[sizeO];
    //for(int i = 0; i < sizeC; i++) dataC[i] = FillValO;

    std::fill(dataC, dataC + sizeC, FillValO);
}

template <class T> inline T& Vector<T>::operator()(unsigned int indexO) {

   assert (indexO < sizeC);
   return dataC[indexO];
}

template <class T> inline T Vector<T>::operator()(unsigned int indexO) const {

    assert (indexO < sizeC);
    return dataC[indexO];
}

template <class T> Vector<T>& Vector<T>::operator+=(const Vector<T>& vecO) {

  assert (sizeC == vecO.sizeC);

  for (unsigned int i = 0; i < sizeC; i++)
    dataC[i] += vecO.dataC[i];
  return *this;
}


template <class T> Vector<T>& Vector<T>::operator+=(const T value) {

  for (unsigned int i = 0; i < sizeC; i++)
    dataC[i] += value;
  return *this;
}

template <class T> Vector<T>& Vector<T>::operator-=(const Vector<T>& vecO) {
    assert (sizeC == vecO.sizeC);

      for (unsigned int i = 0; i < sizeC; i++)
        dataC[i] -= vecO.dataC[i];
      return *this;
}

template <class T> Vector<T>& Vector<T>::operator-=(const vector<T>& vecO)
{
  assert (sizeC == vecO.size());

  for (unsigned int i = 0; i < sizeC; i++)
    dataC[i] -= vecO.at(i);
  return *this;
}


template <class T> Vector<T>& Vector<T>::operator*=(const T scalar) {
  for (unsigned int i = 0; i < sizeC; i++)
    dataC[i] *= scalar;
  return *this;
}

template <class T> Vector<T>& Vector<T>::operator*=(const Vector<T>& vecO)
{
    assert(vecO.sizeC == sizeC);
    for(int i = 0; i < sizeC; i++)
    {
        dataC[i] *= vecO(i);
    }
    return *this;
}




template <class T> void Vector<T>::setSize(int sizeO) {
  if (dataC != 0) delete[] dataC;
  dataC = new T[sizeO];
  sizeC = sizeO;
}

template <class T> void Vector<T>::setSize(int sizeO, const T fillValO) {
  if (dataC != 0) delete[] dataC;
  dataC = new T[sizeO];
  sizeC = sizeO;

 //for(int i = 0; i < sizeC; i++) dataC[i] = fillValO;
  std::fill(dataC, dataC + sizeC, fillValO);

}


template <class T> void Vector<T>::show()
{

  for(int i = 0; i < sizeC; i++)
  {
      cout << dataC[i] << endl;
  }

  cout << "\n"<< endl;
}

template <class T> void Vector<T>::fill(const T valO) {

// for(int i = 0; i < sizeC; i++)
//      dataC[i] = valO;

    std::fill(dataC, dataC + sizeC, valO);
}

template <class T> Vector<T>::~Vector() {

   if (dataC !=0)
       delete[] dataC;
   dataC = 0;
}

template <class T> Vector<T>& Vector<T>::operator=(const Vector<T>& vecO) {

    if(this == &vecO){return *this;}

    if (sizeC != vecO.sizeC) {
      delete[] dataC;
      sizeC = vecO.sizeC;
      dataC = new T[sizeC];
    }
    memcpy(dataC, vecO.dataC, sizeof(T)*sizeC);
//    for (int i = 0; i < sizeC; i++)
//      dataC[i] = vecO.dataC[i];
  return *this;
}

template <class T> Vector<T>& Vector<T>::operator=(const T fillValueO) {
//   for (int i = 0; i < sizeC; i++)
//      dataC[i] = fillValueO;
  std::fill(dataC, dataC + sizeC, fillValueO);
  return *this;
}

template <class T> pair<T, int> Vector<T>::maxim() {

  pair<T, int> maxValueWithPos(dataC[0], 0);
  for (unsigned int i = 1; i < sizeC; i++){
    if (dataC[i] > maxValueWithPos.first )  {
            maxValueWithPos.first = dataC[i];
            maxValueWithPos.second = i;
    }
  }
  return maxValueWithPos;
}

template <class T> T* Vector<T>::data() {
    return dataC;
}

template <class T> pair<T, int>  Vector<T>::minim() {
  pair<T, int> minValueWithPos(dataC[0], 0);
  for (unsigned int i = 1; i < sizeC; i++){
    if (dataC[i] < minValueWithPos.first){
        minValueWithPos.first=dataC[i];
        minValueWithPos.second = i;
    }
  }
  return minValueWithPos;
}

template <class T> T Vector<T>::sum() {
  T result = 0.0;
  for (unsigned int i = 0; i < sizeC; i++)
    result += dataC[i];
  return result;
}

template <class T> T Vector<T>::norm() {
  T sumVec = 0.0;
  for (unsigned int i = 0; i < sizeC; i++)
    sumVec += dataC[i]*dataC[i];
  return sqrtf(sumVec);
}


template <class T> Vector<T> operator/(const Vector<T>& v1, const Vector<T>& v2) {
   Vector<T> res(3);
   res(0)=v1(1)*v2(2) - v1(2)*v2(1);
   res(1)=v1(2)*v2(0) - v1(0)*v2(2);
   res(2)=v1(0)*v2(1) - v1(1)*v2(0);
   return res;
}

template <class T> T operator*( const Vector<T>& v1, const Vector<T>& v2) {
	return v1(0) * v2(0) + v1(1) * v2(1) + v1(2) * v2(2);
}

template <class T> void Vector<T>::connect(Vector<T>& newPartO)
{
    T* newData = new T[sizeC + newPartO.sizeC];

    for(int i = 0; i < sizeC; i++)
        newData[i] = dataC[i];

    for(int i  = sizeC; i < sizeC+newPartO.sizeC; i++)
        newData[i] = newPartO.dataC[i-sizeC];

    delete[] dataC;
    sizeC += newPartO.sizeC;
    dataC = newData;
}

template <class T> void Vector<T>::reallocate(int addSizeO)
{
    Vector<T> newVec(addSizeO, 0.0);
    this->connect(newVec);
}

template <class T> bool Vector<T>::compare(Vector<T>& vecO)
{
    assert (sizeC == vecO.sizeC);

    for(int i = 0; i < sizeC; i++)
    {
        if(dataC[i] != vecO.dataC[i])
            return false;
    }
    return true;
}

template <class T> void Vector<T>::swap()
{
    T* newData = new T[sizeC];

    for(unsigned int i = 0; i < sizeC; i++)
    {
        newData[i] = dataC[(sizeC-1)-i];
    }

    delete[] dataC;
    dataC = newData;
}

template <class T> void Vector<T>::reduceCapacity(int sizeO)
{
    if (sizeC != (unsigned) sizeO)
    {

        T* newData = new T[sizeO];

        for (int i = 0; i < sizeO; i++) {
            newData[i] = dataC[i];
        }

        delete[] dataC;
        sizeC = sizeO;
        dataC = newData;
    }
}

template <class T> void Vector<T>::intersection(Vector<T>& vec2, Vector<T>& intersect)
{
    if(sizeC < vec2.sizeC)
    {
        intersect.setSize(sizeC);
    }
    else
    {
        intersect.setSize(vec2.sizeC);
    }

    int counterElem = 0;

    int j = 0;
    for(int i = 0; i < sizeC; i++)
    {
        while (dataC[i] > vec2(j) && j < vec2.sizeC-1)
        {
            j++;
        }

        if(j == vec2.sizeC)
            break;

        if(dataC[i] == vec2(j))
        {
            intersect(counterElem) = dataC[i];
            counterElem += 1;
        }
    }

    intersect.reduceCapacity(counterElem);
}

template <class T> void Vector<T>::transferToSTL(vector<T>& vecO)
{
    if(vecO.size() > 0) vecO.clear();
    vecO.reserve(sizeC);
    for(unsigned int i = 0; i < sizeC; i++)
    {
        vecO.push_back(dataC[i]);
    }
}

template <class T> void Vector<T>::trasferToMyVec(vector<T>& vecO)
{
    if (dataC !=0)
       delete[] dataC;
    sizeC = vecO.size();
    dataC = new T[sizeC];

    for(unsigned int i = 0; i < sizeC; i++)
    {
        dataC[i] = vecO.at(i);
    }
}

template <class T> void Vector<T>::smooth(int smoothRange)
{
    int width = floor(smoothRange/2.0f);
    int nrX = sizeC;

    Vector<T> smoothed(sizeC);
    float mean;

    for(int i = 0; i < nrX; i++)
    {
        int cw = min(min(i, width),nrX-i-1);
        int upperCol = i - cw;
        int lowerCol = i + cw;

	mean = 0;
	for(int j = upperCol; j <= lowerCol; j++)
	{
		mean += this->dataC[j];
	}

        if((lowerCol - upperCol) > 0)
        {
            mean *=(1.0/((lowerCol - upperCol) + 1));
        }
        smoothed(i) = mean;
    }

    memcpy(dataC,smoothed.dataC,sizeof(T)*sizeC);
        //dataC[0]=smoothed.dataC[0];
}

}

#endif	/* _DENNIS_VECTOR_H */

