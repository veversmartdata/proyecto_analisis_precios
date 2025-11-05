# Proyecto de análisis de precios mensuales

Este repositorio contiene un análisis exploratorio de datos (EDA) de los **precios
mensuales de alimentos, aseo e higiene por región**. Utilizamos un conjunto de
datos proporcionado por el usuario en formato CSV que incluye información de
distintas regiones de un país, productos, marcas, el precio promedio y el
mes de registro.

## Estructura del proyecto

```
proyecto_analisis_precios/
├── data/
│   └── precios_mensuales.csv          # Dataset original (provisto por el usuario)
├── images/
│   ├── mean_price_by_month.png        # Precio promedio global por mes
│   ├── mean_price_by_region_month.png # Precio promedio por región a través de los meses
│   ├── top_products_by_price.png      # Top 10 productos por precio promedio
│   └── product_clusters.png           # Resultado del análisis de clustering
├── notebooks/
│   └── analisis_precios.ipynb         # Cuaderno Jupyter con el EDA detallado
├── scripts/
│   └── analysis.py                    # Script para generar tablas y gráficos
├── descriptive_statistics.csv         # Estadísticas descriptivas exportadas por el script
└── README.md                          # Este archivo explicativo
```

## Descripción del dataset

El archivo `precios_mensuales.csv` contiene 23 901 registros y 5 columnas:

| Columna                    | Descripción                                           |
|----------------------------|--------------------------------------------------------|
| `REGION`                   | Región geográfica donde se recopiló el precio         |
| `producto`                 | Nombre del producto (alimentos, aseo, higiene)        |
| `marca`                    | Marca comercial del producto                          |
| `Promedio de Precio Nuevo` | Precio promedio del producto (formato «$X,XX»)        |
| `Mes`                      | Mes del año en español (enero, febrero, …)            |

Todos los registros están comprendidos entre los meses de **enero** y **septiembre**, aunque la
estructura admite los 12 meses del año.

## Análisis realizados

1. **Limpieza de datos:**
   - Conversión de la columna `Promedio de Precio Nuevo` a valores numéricos retirando el símbolo de dólar y reemplazando la coma por punto.
   - Revisión de filas y columnas para eliminar nulos (no se detectaron nulos en los campos principales).

2. **Estadísticas descriptivas:**
   - Se calcularon medidas como media, mediana, desviación estándar, valores mínimo y máximo para el precio.
   - Se identificó la cantidad de regiones, productos únicos y marcas presentes en el dataset.

3. **Tendencia de precios en el tiempo:**
   - **Precio promedio global por mes:** se observa la evolución del precio a lo largo de los meses de enero a septiembre.
   - **Precio promedio por región y mes:** compara cómo varían los precios en cada región a lo largo del tiempo.

4. **Comparación de productos:**
   - Se calcularon los productos con mayor precio promedio; el gráfico de barras muestra los 10 productos más caros del conjunto de datos.

5. **Análisis de clustering (opcional):**
   - Se agrupan productos según su patrón de precios mensuales utilizando **KMeans**. Para ello se calculó la media de cada producto por mes, se estandarizaron los valores y se aplicó un análisis de componentes principales (PCA) para visualizar los clusters en un plano bidimensional.

Los resultados se guardan en la carpeta `images/` y se describen con mayor detalle en el cuaderno `analisis_precios.ipynb`.

## Cómo reproducir el análisis

1. **Clonar el repositorio y crear un entorno virtual:**
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd proyecto_analisis_precios
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt  # (ver la sección de dependencias)
   ```

2. **Ejecutar el script:**
   ```bash
   python scripts/analysis.py
   ```
   Esto generará las estadísticas descriptivas y las figuras en la carpeta `images/`.

3. **Explorar el cuaderno Jupyter:**
   ```bash
   jupyter notebook notebooks/analisis_precios.ipynb
   ```
   El cuaderno incluye explicaciones paso a paso y permite modificar el análisis o explorar otras preguntas.

## Dependencias

Para ejecutar el proyecto se necesitan las siguientes bibliotecas de Python:

- pandas
- matplotlib
- seaborn
- scikit‑learn (opcional, para el análisis de clustering)

Puedes instalarlas mediante `pip install pandas matplotlib seaborn scikit-learn`.

## Ideas para trabajos futuros

- **Análisis de estacionalidad:** extender los datos para cubrir todo el año y aplicar modelos de descomposición de series temporales (por ejemplo, SARIMA).
- **Comparativa de marcas:** evaluar la diferencia de precios entre marcas dentro de un mismo producto y región.
- **Influencia regional:** incorporar información socioeconómica de las regiones para correlacionarla con los niveles de precios.
- **Detección de anomalías:** utilizar técnicas de aprendizaje automático para identificar meses o regiones con precios atípicos.

Esperamos que este proyecto sirva como base para entender la evolución de los precios de productos esenciales y facilite la generación de nuevas hipótesis y análisis más avanzados.