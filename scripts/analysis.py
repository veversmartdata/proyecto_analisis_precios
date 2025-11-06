#!/usr/bin/env python3
"""
Análisis descriptivo y exploratorio de los precios mensuales de alimentos,
aseo e higiene por región. Este script carga el conjunto de datos,
limpia las columnas relevantes, calcula estadísticas descriptivas y
genera gráficos de tendencia que se guardan en la carpeta ``images``.

Uso: ejecutar este script desde la raíz del proyecto con::

    python scripts/analysis.py

Requisitos: pandas, matplotlib, seaborn, scikit-learn (opcional para
el análisis de clustering).
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    # Optional: clustering analysis requires scikit‑learn
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None  # Will handle absence gracefully

def load_data(path: str) -> pd.DataFrame:
    """Cargar el conjunto de datos desde la ruta dada y limpiar los precios.

    La columna ``Promedio de Precio Nuevo`` contiene el precio como cadena
    con símbolo ``$`` y separador decimal ``,``. Esta función elimina el
    símbolo de moneda, reemplaza la coma por punto y convierte la columna
    a ``float``.

    Args:
        path: Ruta al archivo CSV.

    Returns:
        DataFrame limpio con columnas correctas y precios numéricos.
    """
    df = pd.read_csv(path)
    price_col = 'Promedio de Precio Nuevo'
    # Eliminar símbolo de peso y reemplazar coma decimal por punto
    df[price_col] = (
        df[price_col]
        .astype(str)
        .str.replace('$', '', regex=False)
        .str.replace(',', '.', regex=False)
        .astype(float)
    )
    return df


def describe_data(df: pd.DataFrame) -> pd.DataFrame:
    """Calcular estadísticas descriptivas básicas del precio.

    Args:
        df: DataFrame con la columna ``Promedio de Precio Nuevo`` convertida a ``float``.

    Returns:
        DataFrame con estadísticas descriptivas.
    """
    price_col = 'Promedio de Precio Nuevo'
    desc_stats = df[price_col].describe()
    return desc_stats.to_frame(name='statistics')


def plot_mean_price_by_month(df: pd.DataFrame, save_dir: Path) -> None:
    """Generar gráfico de tendencia del precio promedio por mes (global).

    Args:
        df: DataFrame limpio.
        save_dir: Carpeta donde se guardará la figura.
    """
    price_col = 'Promedio de Precio Nuevo'
    month_order = [
        'enero', 'febrero', 'marzo', 'abril', 'mayo',
        'junio', 'julio', 'agosto', 'septiembre',
        'octubre', 'noviembre', 'diciembre'
    ]
    # Agrupar y ordenar
    mean_month = df.groupby('Mes')[price_col].mean().reindex(month_order)
    mean_month = mean_month.dropna()

    # Dibujar usando matplotlib directamente para evitar dependencias internas de seaborn
    plt.figure(figsize=(8, 4))
    plt.plot(mean_month.index, mean_month.values, marker='o')
    plt.xticks(rotation=45)
    plt.title('Precio promedio mensual (todas las regiones y productos)')
    plt.ylabel('Precio promedio (USD)')
    plt.xlabel('Mes')
    plt.tight_layout()
    output_path = save_dir / 'mean_price_by_month.png'
    plt.savefig(output_path)
    plt.close()


def plot_mean_price_by_region_and_month(df: pd.DataFrame, save_dir: Path) -> None:
    """Generar gráfico de líneas con el precio promedio por región a través de los meses.

    Cada región se representa con una línea. Se utiliza el orden cronológico de los
    meses en español.

    Args:
        df: DataFrame limpio.
        save_dir: Carpeta donde se guardará la figura.
    """
    price_col = 'Promedio de Precio Nuevo'
    month_order = [
        'enero', 'febrero', 'marzo', 'abril', 'mayo',
        'junio', 'julio', 'agosto', 'septiembre',
        'octubre', 'noviembre', 'diciembre'
    ]
    pivot = df.pivot_table(index='Mes', columns='REGION', values=price_col, aggfunc='mean')
    pivot = pivot.reindex(month_order)
    pivot = pivot.dropna(how='all')  # Eliminar filas donde no hay datos

    plt.figure(figsize=(10, 6))
    for region in pivot.columns:
        plt.plot(pivot.index, pivot[region], marker='o', label=region)
    plt.xticks(rotation=45)
    plt.title('Precio promedio por región y mes')
    plt.ylabel('Precio promedio (USD)')
    plt.xlabel('Mes')
    plt.legend(title='Región', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    output_path = save_dir / 'mean_price_by_region_month.png'
    plt.savefig(output_path)
    plt.close()


def plot_top_products_by_average_price(df: pd.DataFrame, save_dir: Path, top_n: int = 10) -> None:
    """Crear gráfico de barras de los productos con mayor precio promedio.

    Args:
        df: DataFrame limpio.
        save_dir: carpeta donde guardar la figura.
        top_n: número de productos a mostrar.
    """
    price_col = 'Promedio de Precio Nuevo'
    product_means = df.groupby('producto')[price_col].mean()
    top_products = product_means.sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    # Usar matplotlib para crear barras horizontales
    plt.barh(y=top_products.index, width=top_products.values, color=plt.cm.viridis(np.linspace(0, 1, len(top_products))))
    plt.title(f'Top {top_n} productos por precio promedio')
    plt.xlabel('Precio promedio (USD)')
    plt.ylabel('Producto')
    plt.tight_layout()
    output_path = save_dir / 'top_products_by_price.png'
    plt.savefig(output_path)
    plt.close()


def cluster_products_by_monthly_pattern(df: pd.DataFrame, save_dir: Path, n_clusters: int = 4) -> None:
    """Aplicar clustering para identificar grupos de productos con patrones de precios similares.

    Se calcula la media mensual por producto, se estandariza y se aplica KMeans. El
    gráfico de dispersión resultante muestra los grupos. Si scikit‑learn no está
    disponible, la función se salta silenciosamente.

    Args:
        df: DataFrame limpio.
        save_dir: Carpeta donde se guardará la figura.
        n_clusters: número de clusters para KMeans.
    """
    if KMeans is None:
        print("Scikit-learn no está instalado; se omite el análisis de clustering.")
        return
    price_col = 'Promedio de Precio Nuevo'
    month_order = [
        'enero', 'febrero', 'marzo', 'abril', 'mayo',
        'junio', 'julio', 'agosto', 'septiembre',
        'octubre', 'noviembre', 'diciembre'
    ]
    # Crear tabla de productos x mes con precio medio
    pivot = df.pivot_table(index='producto', columns='Mes', values=price_col, aggfunc='mean')
    # Seleccionar solo los meses que están presentes en la tabla
    present_months = [m for m in month_order if m in pivot.columns]
    pivot = pivot.reindex(columns=present_months)
    # Imputar valores faltantes propagando hacia adelante y hacia atrás
    pivot = pivot.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)
    # Estándarizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(pivot)
    # Modelo KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    # Añadir columna de cluster al índice
    pivot['cluster'] = clusters
    # Reducir dimensionalidad para visualización con PCA (2 componentes)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    pivot['PC1'] = components[:, 0]
    pivot['PC2'] = components[:, 1]
    plt.figure(figsize=(8, 6))
    # Paleta de colores para clusters
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    for cluster_id in range(n_clusters):
        subset = pivot[pivot['cluster'] == cluster_id]
        plt.scatter(subset['PC1'], subset['PC2'], color=colors[cluster_id], label=f'Cluster {cluster_id}', alpha=0.7)
    plt.title(f'Clustering de productos (KMeans, {n_clusters} clusters)')
    plt.xlabel('Componente principal 1')
    plt.ylabel('Componente principal 2')
    plt.legend(title='Cluster')
    plt.tight_layout()
    output_path = save_dir / 'product_clusters.png'
    plt.savefig(output_path)
    plt.close()


def main():
    # Configurar carpetas
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / 'data' / 'precios_mensuales.csv'
    images_dir = project_root / 'images'
    images_dir.mkdir(exist_ok=True)

    # Cargar y describir datos
    df = load_data(str(data_path))
    desc_stats = describe_data(df)
    # Guardar estadísticas descriptivas como CSV
    desc_stats.to_csv(project_root / 'descriptive_statistics.csv')

    # Graficar precios promedio
    plot_mean_price_by_month(df, images_dir)
    plot_mean_price_by_region_and_month(df, images_dir)
    plot_top_products_by_average_price(df, images_dir, top_n=10)
    # Análisis de clustering (se omite si no se dispone de scikit‑learn)
    cluster_products_by_monthly_pattern(df, images_dir, n_clusters=4)

    print("Análisis completado. Las figuras se han guardado en la carpeta 'images'.")


if __name__ == '__main__':
    main()