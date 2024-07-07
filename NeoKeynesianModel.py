import numpy as np
import pandas as pd
import streamlit as st

with st.sidebar:
    st.title("Model Parameters")
    # Step 1: Define Model Parameters
    tau = st.slider(
    "Select tau",
    0.0, 5.0, 1.5)

    beta = st.slider(
    "Select beta",
    0.5, 1.0, .99)

    kappa = st.slider(
    "Select kappa",
    0.0, 1.0, .5)

    rho_R = st.slider(
    "Select rho_R",
    0.0, 1.0, .8)

    psi_1 = st.slider(
    "Select psi_1",
    0.0, 5.0, 1.5)
    psi_2 = st.slider(
    "Select psi_2",
    0.0, 1.0, 0.5)

    # Step 2: Set Initial Conditions
    y_0 = 0.01  # Initial output gap
    pi_0 = 0.02  # Initial inflation rate
    R_0 = 0.015  # Initial nominal interest rate

# Initial state vector
s_t = np.array([y_0, pi_0, R_0])

# Step 3: Define System Matrices
Gamma_0 = np.array([
    [1, 0, 1/tau],
    [-kappa, 1, 0],
    [-(1-rho_R)*psi_2, -(1-rho_R)*psi_1, 1]
])

Gamma_1 = np.array([
    [0, 0, 0],
    [0, beta, 0],
    [0, 0, rho_R]
])

Psi = np.array([
    [0, 0, -1/tau, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 1]
])

Pi = np.array([
    [0, -1/tau],
    [1, 0],
    [0, 0]
])

# Step 4: Define Structural Shocks (example values)
epsilon_t = np.array([0, 0, 0, 0.01])  # Example structural shock vector
eta_t = np.array([0, 0])  # Example expectational errors vector

# Time horizon for the simulation
T = 10  # Number of periods to simulate

# Step 5: Solve the System over Time
state_history = np.zeros((T, len(s_t)))
state_history[0, :] = s_t

for t in range(1, T):
    # Calculate the next state
    s_t = np.linalg.solve(Gamma_0, Gamma_1 @ state_history[t-1, :] + Psi @ epsilon_t + Pi @ eta_t)
    state_history[t, :] = s_t

# Step 6: Create a pandas DataFrame
columns = ['Output Gap', 'Inflation Rate', 'Nominal Interest Rate']
index = [f'Period {i}' for i in range(T)]
df = pd.DataFrame(state_history, columns=columns, index=index)


# Print the DataFrame
st.title("NeoKyenesian Model")
st.title("")
print(df)
st.line_chart(df)
