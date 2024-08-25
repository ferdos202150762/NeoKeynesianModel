import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

with st.sidebar:
    st.title("Model Parameters")
    # Step 1: Define Model Parameters
    tau = st.slider(
    "Select tau, reciprocal intertemporal elasticity of substitution",
    0.0, 5.0, 1.5)

    beta = st.slider(
    "Select beta, household discount factor ",
    0.5, 1.0, .99)

    kappa = st.slider(
    "Select kappa, ponderation of output gap in the Phillips curve",
    0.0, 1.0, .5)

    rho_R = st.slider(
    "Select rho_R, AR (1) for R (interest rate) in the monetary policy equation",
    0.0, 1.0, .8)

    rho_g = st.slider(
    "Select rho_g, AR (1) for g (growth rate)",
    0.0, 1.0, .8)

    rho_z = st.slider(
    "Select rho_z, AR (1) for z (technology rate)",
    0.0, 1.0, .8)

    psi_1 = st.slider(
    "Select psi_1, ponderation for inflation deviation from the target in the monetary policy equation",
    0.0, 5.0, 1.5)
    psi_2 = st.slider(
    "Select psi_2",
    0.0, 1.0, 0.5)

    # Step 2: Set Initial Conditions
    y_0 = 0.0  # Percentage deviation from the SS value for output
    pi_0 = 0.02  # Percentage deviation from the ss value for inflation
    R_0 = 0.015  # Percentage deviation fro the SS value for interest rate
    epsilon_R_0 = np.random.normal(0, .01, 1).item()   # Monetary policy shock
    g_0 = .02   # Percentage deviation from the SS value for output growth rate
    z_0 = 0.0   # Percentage deviation from the SS value for exogenous fluctation s of the technology growth rate
    E_y_1 = 0.0  # Expected Percentage deviation from the SS value for output next period
    E_pi_1 = 0.02  # Expected Percentage deviation from the ss value for inflation next period

# Initial state vector
s_t = np.array([y_0, pi_0, R_0, epsilon_R_0, g_0, z_0, E_y_1, E_pi_1])

# Step 3: Define System Matrices
Gamma_0 = np.array([
    [1, 0, 1/tau, 0, -rho_g, -rho_z*1/tau, -1, -1/tau],
    [-kappa, 1, 0, 0, kappa, 0, 0, -beta],
    [-(1-rho_R)*psi_2, -(1-rho_R)*psi_1, 1, -1, (1-rho_R)*psi_2, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0], 
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1]  

])

Gamma_1 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, rho_R, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, rho_g, 0, 0, 0],
    [0, 0, 0, 0, 0, rho_z, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1]  
])

Psi = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
])

Pi = np.array([
    [-1, -1/tau],
    [0, -beta],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0]
])



# Time horizon for the simulation
T = 100  # Number of periods to simulate

# Step 5: Solve the System over Time
state_history = np.zeros((T, len(s_t)))
state_history[0, :] = s_t

for t in range(1, T):

    # Step 4: Define Structural Shocks (example values)
    rand_z = np.random.normal(0, .01, 1).item()   
    rand_g = np.random.normal(0, .01, 1).item() 
    rand_R = np.random.normal(0, .01, 1).item() 
    rand_output = np.random.normal(0, .01, 1).item() 
    rand_inflation = np.random.normal(0, .01, 1).item()

    epsilon_t = np.array([rand_z, rand_g, rand_R, 0])  # Example structural shock vector
    eta_t = np.array([rand_output, rand_inflation])  # Example expectational errors vector
    # Calculate the next state
    s_t = np.linalg.solve(Gamma_0, Gamma_1 @ state_history[t-1, :] + Psi @ epsilon_t + Pi @ eta_t)
    state_history[t, :] = s_t

# Step 6: Create a pandas DataFrame
columns = ["% Deviation from SS output", "% Deviation from SS inflation", "% Deviation from SS interest rate", "Monetary policy shock", "% Deviation from SS output growth rate", "% Deviation from SS technology rate", " Expected deviation from SS output", " Expected deviation from SS inflation"]
index = [f'Period {i}' for i in range(T)]
df = pd.DataFrame(state_history, columns=columns, index=index)


# Print the DataFrame
st.title("NeoKyenesian Model")
st.title("")

# Prepare data for Altair (melt to long format)
df_long = df.reset_index().melt(id_vars='index', 
                                value_vars=["% Deviation from SS output", 
                                            "% Deviation from SS inflation", 
                                            "% Deviation from SS interest rate", 
                                            "Monetary policy shock"],
                                var_name='Series', value_name='Value')

# Create the Altair chart
chart = alt.Chart(df_long).mark_line().encode(
    x='index:O',  # Use the index (periods) as X-axis
    y='Value:Q',  # Quantitative Y-axis
    color='Series:N'  # Differentiate lines by series
).properties(
    width=1000,
    height=600
).configure_legend(
    orient='right'  # Position legend on the right-hand side
)

# Display the Altair chart in Streamlit
st.altair_chart(chart, use_container_width=True)

# Prepare data for Altair (melt to long format)
df_long2 = df.reset_index().melt(id_vars='index', 
                                value_vars=[ "% Deviation from SS output growth rate", "% Deviation from SS technology rate", " Expected deviation from SS output", " Expected deviation from SS inflation"],
                                var_name='Series', value_name='Value')

# Create the Altair chart
chart = alt.Chart(df_long2).mark_line().encode(
    x='index:O',  # Use the index (periods) as X-axis
    y='Value:Q',  # Quantitative Y-axis
    color='Series:N'  # Differentiate lines by series
).properties(
    width=1000,
    height=600
).configure_legend(
    orient='right'  # Position legend on the right-hand side
)

# Display the Altair chart in Streamlit
st.altair_chart(chart, use_container_width=True)