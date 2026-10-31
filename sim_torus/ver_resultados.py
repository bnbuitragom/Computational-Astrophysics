import pyPLUTO as pp
import matplotlib.pyplot as plt

# Carga los datos
D = pp.Load()

# Extrae un plano 2D de la densidad (por ejemplo, el índice 0 de la segunda dimensión)
# O usa D.slices para recortar el volumen 3D a un plano 2D
rho_slice = D.rho[:, 0, :]  # Dependiendo de cómo quieras ver el plano (ej. plano meridional)

# Muestra la información de las dimensiones del corte
print("Forma del corte:", rho_slice.shape)

# Grafica usando matplotlib directamente o con el Image de PyPLUTO
plt.figure(figsize=(8, 6))
plt.imshow(rho_slice.T, origin='lower', aspect='auto', cmap='jet')
plt.colorbar(label='Densidad (rho)')
plt.title('Simulación Torus - Densidad')
plt.xlabel('Coordenada x1')
plt.ylabel('Coordenada x3')
plt.show()