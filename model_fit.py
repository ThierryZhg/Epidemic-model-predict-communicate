import numpy as np
import numpy.linalg as linalg
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.optimize as optimize

df = pd.read_csv("C:/Users/ioost/OneDrive/Documents/1A Centrale/ST propagation virale/EI/data_all.txt", sep="\t")

beta_b = 0.1
gamma_b = 1/7
mu_b = 0.143

beta_m = 0.4
mu_m = 1/28

beta_h = 0.01
sigma_h = 1/4
gamma_h = 1/5

I_b0 = 8179
R_b0 = 2300
D_b0 = 1612
S_b0 = 34000 - I_b0 - D_b0 - R_b0

S_m01 = 7100
I_m01 = 1900
S_m02 = 2480
I_m02 = 350

E_h01 = 270
I_h01 = 118
R_h01= 368
S_h01= 4500-E_h01-I_h01-R_h01

E_h02 = 49
I_h02 = 51
R_h02 = 104
S_h02 = 3500 -E_h02-I_h02-R_h02

E_h03 = 33
I_h03 = 55
R_h03 = 144
S_h03 = 4500-E_h03-I_h03-R_h03
t_array = np.linspace(1, 150, 150)

I_oi_observed = df['I_b'].values
D_oi_observed = df['EstimatedD_b'].values
I_m1_observed = df['I_m1'].values
I_h1_observed = df['I_h1'].values
R_h1_observed = df['R_h1'].values
I_m2_observed = df['I_m2'].values
I_h2_observed = df['I_h2'].values
R_h2_observed = df['R_h2'].values
I_h3_observed = df['I_h3'].values
R_h3_observed = df['R_h3'].values


#S_b I_b R_b D_b S_m I_m S_h E_h I_h R_h
initial_conditions = [S_b0, I_b0, R_b0, D_b0, S_m01, I_m01, S_m02, I_m02, S_h01, E_h01, I_h01, R_h01, S_h02, E_h02, I_h02, R_h02, S_h03, E_h03, I_h03, R_h03]

# dY = model_class(Y,t,paramètres)
def model(Y, t, beta_b, gamma_b, mu_b, beta_m, mu_m, beta_h, sigma_h, gamma_h):
    S_b, I_b, R_b, D_b, S_m1, I_m1, S_m2, I_m2, S_h1, E_h1, I_h1, R_h1,S_h2, E_h2, I_h2, R_h2, S_h3, E_h3, I_h3, R_h3 = Y
        
    dS_b = -beta_b*S_b*(I_m1+I_m2)/(S_b+I_b+R_b)
    dI_b = beta_b*S_b*I_m1/(S_b+I_b+R_b) - mu_b*I_b - gamma_b*I_b
    dR_b = gamma_b*I_b
    dD_b = mu_b*I_b
    
    dS_m1 = -beta_m*S_m1*I_b/(S_m1+I_m1)
    dI_m1 = beta_m*S_m1*I_b/(S_m1+I_m1) - mu_m*I_m1
    dS_m2 = -beta_m*S_m2*I_b/(S_m2+I_m2)
    dI_m2 = beta_m*S_m2*I_b/(S_m2+I_m2) - mu_m*I_m1
    
    dS_h1 = -beta_h*S_h1*I_m1/(S_h1+I_h1+ E_h1 + R_h1)
    dE_h1 = beta_h*S_h1*I_m1/(S_h1+I_h1+ E_h1 + R_h1) - sigma_h*E_h1
    dI_h1 = sigma_h*E_h1 - gamma_h*I_h1
    dR_h1 = gamma_h*I_h1
    
    dS_h2 = -beta_h*S_h2*(I_m2+I_m1)/(S_h2+I_h2+ E_h2 + R_h2)
    dE_h2 = beta_h*S_h2*(I_m2+I_m1)/(S_h2+I_h2+ E_h2 + R_h2) - sigma_h*E_h2
    dI_h2 = sigma_h*E_h2 - gamma_h*I_h2
    dR_h2 = gamma_h*I_h2
    
        
    dS_h3 = -beta_h*S_h3*(I_m2+I_m1)/(S_h3+I_h3+ E_h3 + R_h3)
    dE_h3 = beta_h*S_h3*(I_m2+I_m1)/(S_h3+I_h3+ E_h3 + R_h3) - sigma_h*E_h3
    dI_h3 = sigma_h*E_h3 - gamma_h*I_h3
    dR_h3 = gamma_h*I_h3
    

    return [dS_b, dI_b, dR_b, dD_b, dS_m1, dI_m1, dS_m2, dI_m2, dS_h1, dE_h1, dI_h1, dR_h1, dS_h2, dE_h2, dI_h2, dR_h2, dS_h3, dE_h3, dI_h3, dR_h3]

def error_function(params):
    # Extraction des paramètres
    par = params

    # Simulation du modèle avec les paramètres actuels
    simulated_values = integrate.odeint(model, initial_conditions, t_array, args=(beta_b, gamma_b, mu_b, beta_m, mu_m, beta_h, sigma_h, gamma_h)).T

    # Calcul du coût (somme des carrés des différences)
    cost = np.sum((I_oi_observed - simulated_values[1][:len(I_oi_observed)])**2)
    cost += np.sum((D_oi_observed - simulated_values[2][:len(D_oi_observed)])**2)
    #cost += np.sum((S_m_observed - simulated_values[3][:len(S_m_observed)])**2)
    cost += np.sum((I_m1_observed - simulated_values[4][:len(I_m1_observed)])**2)
    cost += np.sum((I_m2_observed - simulated_values[4][:len(I_m2_observed)])**2)
    
    cost += np.sum((I_h1_observed - simulated_values[6][:len(I_h1_observed)])**2)
    cost += np.sum((R_h1_observed - simulated_values[7][:len(R_h1_observed)])**2)
    cost += np.sum((I_h2_observed - simulated_values[6][:len(I_h2_observed)])**2)
    cost += np.sum((R_h2_observed - simulated_values[7][:len(R_h2_observed)])**2)
    cost += np.sum((I_h2_observed - simulated_values[6][:len(I_h2_observed)])**2)
    cost += np.sum((R_h2_observed - simulated_values[7][:len(R_h2_observed)])**2)

    return cost

# Initialisation
t_array = np.linspace(1, 50, 50)
#model = integrate.odeint(model, initial_conditions, t_array, args=(beta_b, gamma_b, mu_b, beta_m, mu_m, beta_h, sigma_h, gamma_h))

result = optimize.minimize(error_function, [beta_b, gamma_b, mu_b, beta_m, mu_m, beta_h, sigma_h, gamma_h], method='L-BFGS-B')
optimal_params = result.x
print(optimal_params)
model = integrate.odeint(model, initial_conditions, t_array, args=tuple(optimal_params))



# Tracer des résultats dans des graphes séparés
fig, axes = plt.subplots(nrows=10, ncols=2, figsize=(12, 15))
axes = axes.flatten()

labels = ['S_b', 'I_b', 'R_b', 'D_b', 'S_m1', 'I_m1', 'S_m2', 'I_m2', 'S_h1', 'E_h1', 'I_h1', 'R_h1', 'S_h2', 'E_h2', 'I_h2', 'R_h2', 'S_h3', 'E_h3', 'I_h3', 'R_h3']

for i, ax in enumerate(axes):
    ax.plot(t_array, model[:, i], label=f'Simulated {labels[i]}', color='blue')
    
    # Plot the observed data when available
    if i == 1:
        ax.plot(t_array[:len(I_oi_observed)], I_oi_observed, label=f'Observed I_b', linestyle='dashed', color='red')
    elif i == 3:
        ax.plot(t_array[:len(D_oi_observed)], D_oi_observed, label=f'Observed D_b', linestyle='dashed', color='green')
    elif i == 5:
        ax.plot(t_array[:len(I_m1_observed)], I_m1_observed, label=f'Observed I_m1', linestyle='dashed', color='orange')
    elif i == 7:
        ax.plot(t_array[:len(I_m2_observed)], I_m2_observed, label=f'Observed I_m2', linestyle='dashed', color='orange')
    elif i == 10:
        ax.plot(t_array[:len(I_h1_observed)], I_h1_observed, label=f'Observed I_h1', linestyle='dashed', color='purple')
    elif i == 11:
        ax.plot(t_array[:len(R_h1_observed)], R_h1_observed, label=f'Observed R_h1', linestyle='dashed', color='cyan')
    elif i == 14:
        ax.plot(t_array[:len(I_h1_observed)], I_h2_observed, label=f'Observed I_h2', linestyle='dashed', color='purple')
    elif i == 15:
        ax.plot(t_array[:len(R_h1_observed)], R_h1_observed, label=f'Observed R_h2', linestyle='dashed', color='cyan')
    elif i == 18:
        ax.plot(t_array[:len(I_h1_observed)], I_h1_observed, label=f'Observed I_h3', linestyle='dashed', color='purple')
    elif i == 19:
        ax.plot(t_array[:len(R_h1_observed)], R_h1_observed, label=f'Observed R_h3', linestyle='dashed', color='cyan')

    ax.set_xlabel('Time')
    ax.set_ylabel('Population')
    ax.legend()

plt.tight_layout()
plt.show()
