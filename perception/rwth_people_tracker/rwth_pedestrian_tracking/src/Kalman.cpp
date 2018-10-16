/*
 * Copyright 2012 Dennis Mitzel
 *
 * Authors: Dennis Mitzel
 * Computer Vision Group RWTH Aachen.
 */

#include "Kalman.h"

/***********************************************/
/* y   - the observation at time t             */
/* H - jacobian Observation Model              */
/* F - jacobian Dynamic Model                  */
/* Q - the system covariance                   */
/* R - the observation covariance              */
/* x - the state (column) Vector               */
/* P - the state covariance                    */
/***********************************************/

void Kalman::predict()
{
    //**************************************************
    // m_xprior = non_lin_state_equation(m_xpost, m_dt);
    // m_Prior = F*m_Post*F' + W*mQ*W'
    //***************************************************

    //std::cout << "---pred KF---" << std::endl;

    Matrix<double> mQ = makeQ(m_xpost, m_dt);

    Matrix<double> mF = makeF(m_xpost, m_dt);

    //Matrix<double> mW = makeW();

    //m_xprio = non_lin_state_equation(m_xpost, m_dt);
    m_xprio = mF*m_xpost;

    m_Pprio = mF*m_Ppost*Transpose(mF);

    //m_Pprio += mW*mQ*Transpose(mW);
    m_Pprio += mQ;

    //m_xprio.show();
    //m_Pprio.Show();
    //td::cout << "---" << std::endl;

}

Vector<double> Kalman::create3VecPos_Ori(Vector<double> vX)
{
    Vector<double> result(3);
    result(0) = (vX(0));
    result(1) = (vX(1));
    result(2) = (vX(2));

    return result;
}

Vector<double> Kalman::angleMinus(Vector<double> &a, Vector<double> &b)
{
    Vector<double> res = a;
    res -= b;

    if(res(2) < -M_PI) res(2) += 2*M_PI;
    else if(res(2) > M_PI) res(2) -= 2*M_PI;

    return res;
}

void Kalman::showDeg(double a)
{
    cout << (a / M_PI * 180) << endl;
}

void Kalman::update()
{

    //**************************************************
    // S = H*P_pred*H' + V*R*V';
    // Sinv = inv(S);
    // K = P_pred*H'* Sinv; % Kalman gain matrix
    // x_new = xpred + K*(y - H'xpred);
    // P_new = P_pred - K*H*P_pred;
    //***************************************************

    //std::cout << "---update KF---" << std::endl;

    if(m_measurement_found)
    {
        Matrix<double> mR = makeR();

        Matrix<double> mH = makeH();

        Matrix<double> mS_inv = mH*m_Pprio*Transpose(mH);
        mS_inv += mR;
        Matrix<double> mS = mS_inv;
        if(!mS_inv.inv())
           cerr << "Matrix mS is singular!" << endl;
        Matrix<double> K = m_Pprio*Transpose(mH)*mS_inv;

        //Vector<double> angdiff = angleMinus(m_measurement, m_xprio);
        Vector<double> residual = m_measurement - (mH*m_xprio);
        m_xpost = m_xprio + (K*residual);

        //if(m_xpost(2) < -M_PI) m_xpost(2) += 2*M_PI;
        //else if(m_xpost(2) > M_PI) m_xpost(2) -= 2*M_PI;

        m_Ppost = m_Pprio - (K*mS*Transpose(K));

        //m_measurement.show();

    }
    else
    {
        m_xpost = m_xprio;
        m_Ppost = m_Pprio;

        //std::cout << "no meas." << std::endl;
    }

    //m_xpost.show();
    //m_Ppost.Show();

    //std::cout << "---" << std::endl;

}

void Kalman::init(Vector<double> xInit, Matrix<double> Pinit, double dt)
{
    //std::cout << "---init KF---" << std::endl;
    m_xpost = xInit;
    m_Ppost = Pinit;
    m_dt = dt;
    //xInit.show();
    //Pinit.Show();
    //std::cout << "---" << std::endl;
}
