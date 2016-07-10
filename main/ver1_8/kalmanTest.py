
from numpy import diag, eye, zeros, pi, exp, dot, sum, tile, linalg, array
from numpy import log as logg
from numpy.linalg import inv, det
from numpy.random import randn
import numpy as np
import random
# Implementation of a Kalman filter for time series analysis.
# Ideas adopted from http://hal.archives-ouvertes.fr/docs/00/43/38/86/PDF/Laaraiedh_PythonPapers_Kalman.pdf

# Algorithm plots both the observed price and the signal produced
# by the Kalman filter for assets in the universe.

def initialize(context):
    # Set total number of securities
    context.total_securites = 2
    # Select securites
    context.securites = [sid(26578), sid(24)]
    context.X = np.zeros((context.total_securites*2,1))
    context.P = np.diag(np.zeros(context.total_securites*2))
    context.A = array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0,0]])    
    context.Q = eye(context.X.shape[0])
    context.B = eye(context.X.shape[0])
    context.U = zeros((context.X.shape[0],1))
    context.Y = array([[0], [0]])
    context.H = array([[1, 0, 0, 0], [0, 1, 0, 0]])
    context.R = eye(context.Y.shape[0])

def handle_data(context, data):
   # Apply the Kalman filter
    
   # Prediction step 
   (context.X, context.P) = kf_predict(context.X, context.P, context.A, context.Q, context.B, context.U)
   # Update step
   (context.X, context.P, K, IM, IS, LH) = kf_update(context.X, context.P, context.Y, context.H, context.R)  
   # Record observed price data
   context.Y = array([[data[sid(26578)].price],[data[sid(24)].price]])
      
   goog_price_observed = context.Y[0][0]    
   goog_price_filter = context.X[0][0]
   aapl_price_observed = context.Y[1][0]    
   aapl_price_filter = context.X[1][0]
   
   record(GOOG_actual = goog_price_observed, GOOG_filtered = goog_price_filter)
   record(AAPL_actual = aapl_price_observed, AAPL_filtered = aapl_price_filter) 
    
# PREDICTION STEP

# X is the mean state estimate of the previous step (k-1)
# P is the state covariance matrix of the previous step (k-1)
# A is the transition nxn matrix
# Q is the process noise covariance matrix
# B is the input effect matrix
# U is the control input

def kf_predict(X, P, A, Q, B, U):
    X = dot(A, X) + dot(B, U)
    P = dot(A, dot(P, A.T)) + Q
    return(X,P)

# UPDATE STEP

# At time step k, this update computes the posterior mean X and 
# covariance P of the system state given a new measurement Y. 

# Y is the measurement vector
# H is the measurement matrix
# R is the measurement covariance matrix
# K is the Kalman Gain matrix
# IM is the mean of the predictive distribution of Y
# IS is the covariance of the predictive mean of Y
# LH is the predictive probability (lieklihood) of measurement

def kf_update(X, P, Y, H, R):
    IM = dot(H, X)
    IS = R + dot(H, dot(P, H.T))
    K = dot(P, dot(H.T, inv(IS)))
    X = X + dot(K, (Y-IM))
    P = P - dot(K, dot(IS, K.T))
    LH = gauss_pdf(Y, IM, IS)
    return (X,P,K,IM,IS,LH)
    
def gauss_pdf(X, M, S):
    if M.shape[1] == 1:
        DX = X - tile(M, X.shape[1])
        E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0)
        E = E + 0.5 * M.shape[0] * logg(2 * pi) + 0.5 * logg(det(S))
        P = exp(-E)
    elif X.shape[1] == 1:
        DX = tile(X, M.shape[1])- M
        E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0)
        E = E + 0.5 * M.shape[0] * logg(2 * pi) + 0.5 * logg(det(S))
        P = exp(-E)
    else:
        DX = X-M
        E = 0.5 * dot(DX.T, dot(inv(S), DX))
        E = E + 0.5 * M.shape[0] * logg(2 * pi) + 0.5 * logg(det(S))
        P = exp(-E)
    return (P[0],E[0])
