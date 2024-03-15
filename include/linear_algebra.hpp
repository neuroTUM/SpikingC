#ifndef LINEAR_ALGEBRA_HPP
#define LINEAR_ALGEBRA_HPP

#include<iostream>
#include <algorithm>

#include "network_config.hpp"

template <typename T>
T** transposeMatrix(T** m, unsigned int m_rows, unsigned int m_cols){
    T** m_new = new T*[m_cols];
    for(unsigned int i = 0; i < m_cols; i++){
        m_new[i] = new T [m_rows];
    }

    for(unsigned int i = 0; i < m_rows; i++){
        for(unsigned int j = 0; j < m_cols; j++){
            m_new[j][i] = m[i][j];
        }
    }
    
    for(unsigned int i = 0; i < m_rows; i++){
        delete[] m[i];
    }
    delete[] m;

    return m_new;
}

/* m1 is stored in row major order and m2 in column major */
template <typename T>
void matrixMAC(T** m1, unsigned int m1_rows, unsigned int m1_cols, 
               T** m2, unsigned int m2_rows, unsigned int m2_cols, 
               T** m3, unsigned int m3_rows, unsigned int m3_cols,
               T* v, unsigned int v_rows){

    if((m1_cols != m2_rows) || (m3_rows != m1_rows) || (m3_cols != m2_cols) || (v_rows != m3_rows)){
        std::cout << "matrixMul : Matrices don't have the appropriate dimensions" << std::endl; 
        exit(1);        
    }

    T r;
    for(unsigned int i = 0; i < m1_rows; i++){
        for(unsigned int j = 0; j < m2_cols; j++){
            r = 0;
            for(unsigned int k = 0; k < m1_cols; k++){
                r += m1[i][k] * m2[j][k];
             }
            m3[i][j] = r + v[i];
        }
    }
}

#endif