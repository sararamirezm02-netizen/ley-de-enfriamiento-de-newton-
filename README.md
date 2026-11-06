# Estabilidad de clústeres autoadaptativos (auto-scaling)

Código del artículo del Proyecto Final de Cálculo Diferencial (Universidad de La Salle) sobre el modelo dN/dt = rN(1 − N/K) − cN.

## Requisitos
Python 3.14 y las librerías de requirements.txt:

    pip install -r requirements.txt

## Ejecución

    python calculo.py

El script compara las soluciones analíticas con la integración numérica RK45, calcula la tabla de propagación de incertidumbre (semilla 42, 20 000 muestras) y guarda las figuras 1 a 3 en la carpeta actual.
