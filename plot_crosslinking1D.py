
from fenics import *
import matplotlib.pyplot as plt
import numpy as np
from config import paper_params, alginate_only_params
import crosslinking1D as cl

# Plot paper_params results
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
overlay_y = [18.86792453, 22.7408143, 26.51439921, 55.21350546, 93.94240318, 133.9622642]

plt.scatter(overlay_x, overlay_y, color='black', marker='o', label='Data Points', zorder=5)

plt.xlabel("Time (minutes)")
plt.ylabel("V_c(t) (µL)")
plt.title("Accumulated Volume Over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
