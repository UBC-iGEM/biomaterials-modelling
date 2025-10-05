
from fenics import *
import matplotlib.pyplot as plt
import numpy as np
from config import paper_params, alginate_only_params, thirty_percent_sand_params, mgs_params
import crosslinking1D as cl


def calculate_fit_metrics(y_exp, y_model):
    #Calculate RMSE, RMSE% and R² between experimental and model values.
    y_exp = np.asarray(y_exp, dtype=float)
    y_model = np.asarray(y_model, dtype=float)

    if y_exp.shape != y_model.shape:
        raise ValueError("y_exp and y_model must have the same length")

    residuals = y_exp - y_model
    rmse = np.sqrt(np.mean(residuals**2))
    rmse_percent = (rmse / np.mean(y_exp)) * 100
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_exp - np.mean(y_exp))**2)
    r2 = 1 - (ss_res / ss_tot)

    return rmse, rmse_percent, r2

# # -----------------------
# # MODEL VALIDATION (PAPER PARAMETERS)
# # -----------------------
Vc_D0_paper = cl.constant_diffusion(paper_params)
Vc_D_alpha_paper = cl.diffusion_alpha(paper_params)
time_points_paper = np.linspace(0, paper_params.t_end, paper_params.num_steps + 1) / 60  # min

plt.figure()

# Plot constant diffusion (blue)
plt.plot(time_points_paper, Vc_D0_paper, label="D0", lw=2, color='blue')

# Plot D(alpha) diffusion (red, dashed)
plt.plot(time_points_paper, Vc_D_alpha_paper, label="D(α)", lw=2, color='red')

plt.xlabel("Time (minutes)")
plt.ylabel("Absorbed Volume of CaCl2 (µL)")
plt.title("Absorbed Volume of CaCl2 Over Time (Model Validation)")
plt.grid(True)
plt.legend()
plt.tight_layout()

# # -----------------------
# # MODEL VALIDATION (0% SAND)
# # -----------------------
Vc_D0 = cl.constant_diffusion(alginate_only_params)
Vc_D_alpha = cl.diffusion_alpha(alginate_only_params)
time_points = np.linspace(0, alginate_only_params.t_end, alginate_only_params.num_steps + 1) / 60  # min

plt.figure()

# Plot constant diffusion (blue)
plt.plot(time_points, Vc_D0, label="D0", lw=2, color='blue')

# Plot D(alpha) diffusion (red, dashed)
plt.plot(time_points, Vc_D_alpha, label="D(α)", lw=2, color='red')

#Plot experimental data (black dots)
overlay_x = [2,5,10,15,20,60]       
y_exp_alginate = [18.86792453, 22.7408143, 26.51439921, 55.21350546, 93.94240318, 133.9622642]

plt.scatter(overlay_x, y_exp_alginate, color='black', marker='o', label='Data Points', zorder=5)

plt.xlabel("Time (minutes)")
plt.ylabel("Absorbed Volume of CaCl2 (µL)")
plt.title("Absorbed Volume of CaCl2 Over Time (0% Sand)")
plt.grid(True)
plt.legend()
plt.tight_layout()

# # -----------------------
# # BIO-INK MODEL CALIBRATION (30% SAND)
# # -----------------------
plt.figure()

Vc_D0_thirty_sand = cl.constant_diffusion(thirty_percent_sand_params)
Vc_D_alpha_thirty_sand = cl.diffusion_alpha(thirty_percent_sand_params)
time_points_thirty_sand= np.linspace(0, thirty_percent_sand_params.t_end, thirty_percent_sand_params.num_steps + 1) / 60  # min

# Plot constant diffusion (blue)
plt.plot(time_points, Vc_D0_thirty_sand, label="D0", lw=2, color='blue')

# Plot D(alpha) diffusion (red, dashed)
plt.plot(time_points, Vc_D_alpha_thirty_sand, label="D(α)", lw=2, color='red')

#Plot experimental data (black dots)
overlay_x = [2,5,10,15,20,60]       
y_exp_thirty_sand = [34.75670308, 34.45878848, 48.36146971, 54.02184707, 63.75372393, 112.7110228]

plt.scatter(overlay_x, y_exp_thirty_sand, color='black', marker='o', label='Data Points', zorder=5)

plt.xlabel("Time (minutes)")
plt.ylabel("Absorbed Volume of CaCl2 (µL)")
plt.title("Absorbed Volume of CaCl2 Over Time (30% Sand)")
plt.grid(True)
plt.legend()
plt.tight_layout()


# -----------------------
# BIO-INK MODEL CALIBRATION (20% MGS + 3.5%CMC )
# -----------------------
plt.figure()

Vc_D0_mgs = cl.constant_diffusion(mgs_params)
Vc_D_alpha_mgs = cl.diffusion_alpha(mgs_params)
time_points_mgs= np.linspace(0, mgs_params.t_end, mgs_params.num_steps + 1) / 60  # min

# Plot constant diffusion (blue)
plt.plot(time_points_mgs, Vc_D0_mgs, label="D0", lw=2, color='blue')

# Plot D(alpha) diffusion (red, dashed)
plt.plot(time_points_mgs, Vc_D_alpha_mgs, label="D(α)", lw=2, color='red')

#Plot experimental data (black dots)
overlay_x = [2,5,10,15,20,60]       
y_exp_mgs = [26.51439921, 32.27408143, 39.32472691, 40.61569017, 50.74478649, 51.83714002]

plt.scatter(overlay_x, y_exp_mgs, color='black', marker='o', label='Data Points', zorder=5)

plt.xlabel("Time (minutes)")
plt.ylabel("Absorbed Volume of CaCl2 (µL)")
plt.title("Absorbed Volume of CaCl2 Over Time (20% MGS + 3.5% CMC)")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()


# -----------------------
# CALCULATE FIT METRICS
# -----------------------
#Calculate RMSE and R²
indices = [6, 15, 30, 45, 60, 180] # Corresponding to 2,5,10,15,20,60 minutes (in terms of time steps)
y_model_alginate = [Vc_D_alpha[i] for i in indices]
y_model_thirty_sand = [Vc_D_alpha_thirty_sand[i] for i in indices]     
y_model_mgs = [Vc_D_alpha_mgs[i] for i in indices]     

rmse_alginate, rmse_percent_alginate, r2_alginate = calculate_fit_metrics(y_exp_alginate, y_model_alginate)
rmse_thirty_sand, rmse_percent_thirty_sand, r2_thirty_sand = calculate_fit_metrics(y_exp_thirty_sand, y_model_thirty_sand)
rmse_mgs, rmse_percent_mgs, r2_mgs = calculate_fit_metrics(y_exp_mgs, y_model_mgs)

print("0% Sand - RMSE: {:.4f}, RMSE%: {:.2f}%, R²: {:.4f}".format(rmse_alginate, rmse_percent_alginate, r2_alginate))
print("30% Sand - RMSE: {:.4f}, RMSE%: {:.2f}%, R²: {:.4f}".format(rmse_thirty_sand, rmse_percent_thirty_sand, r2_thirty_sand))
print("20% MGS + 3.5% CMC - RMSE: {:.4f}, RMSE%: {:.2f}%, R²: {:.4f}".format(rmse_mgs, rmse_percent_mgs, r2_mgs))