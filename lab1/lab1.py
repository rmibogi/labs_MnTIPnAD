import os
import shutil
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import haversine_distances

# Настройки
DATA_URL = 'https://raw.githubusercontent.com/epogrebnyak/ru-cities/main/assets/towns.csv'
EARTH_RADIUS_KM = 6371.0
PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data')
DATA_FILE = os.path.join(DATA_FOLDER, 'towns.csv')
RESULTS_FOLDER = os.path.join(PROJECT_FOLDER, 'clusters')
MAX_K = 20
MIN_RADIUS = 50
MAX_RADIUS = 2000
RADIUS_STEP = 50


def save_csv(df, path):
    df.to_csv(path, index=False, encoding='utf-8-sig')


def finish_plot(path):
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.show()


def load_data():
    print('\n' + '=' * 70, 'ЗАГРУЗКА БАЗЫ ГОРОДОВ РОССИИ', '=' * 70, sep='\n')
    os.makedirs(DATA_FOLDER, exist_ok=True)

    if os.path.exists(DATA_FILE):
        print('\nИспользуется локальная база данных.', DATA_FILE, sep='\n')

        try:
            df = pd.read_csv(DATA_FILE)
        except Exception as error:
            print('\nОшибка чтения базы:', error, sep='\n')
            return None
    else:
        print(
            '\nЛокальная база данных не найдена.',
            'Выполняется первоначальная загрузка...',
            sep='\n',
        )

        try:
            df = pd.read_csv(DATA_URL)
        except Exception as error:
            print(
                '\nНе удалось скачать базу.',
                'Проверьте подключение к интернету.',
                error,
                sep='\n',
            )
            return None

        try:
            save_csv(df, DATA_FILE)

            print(
                '\nБаза успешно скачана.',
                f'Файл: {DATA_FILE}',
                'При следующих запусках скачивание выполняться не будет.',
                sep='\n',
            )
        except Exception as error:
            print('\nОшибка сохранения базы:', error, sep='\n')
            return None

    required_columns = ['city', 'lat', 'lon']

    for column in required_columns:
        if column not in df.columns:
            print(f"\nОшибка: отсутствует столбец '{column}'.")
            return None

    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')

    if 'population' in df.columns:
        df['population'] = pd.to_numeric(df['population'], errors='coerce')

    df = df.dropna(subset=['city', 'lat', 'lon'])
    df = df.reset_index(drop=True)

    print(f'\nБаза готова.', f'Количество городов: {len(df)}', sep='\n')

    return df


def show_data_info(df):
    print(
        '\n' + '=' * 70,
        'ИНФОРМАЦИЯ О ДАННЫХ',
        '=' * 70,
        f'\nКоличество городов: {len(df)}',
        '\nПоля базы:',
        sep='\n',
    )

    for column in df.columns:
        print(f' - {column}')

    possible_columns = ['city', 'region_name', 'federal_district', 'population', 'lat', 'lon']
    columns = [column for column in possible_columns if column in df.columns]

    print(
        '\nПервые 20 городов:\n',
        df[columns].head(20).to_string(index=False),
        sep='\n',
    )


def filter_data(original_df):
    df = original_df.copy()

    print(
        '\n' + '=' * 70,
        'ФИЛЬТРАЦИЯ',
        '=' * 70,
        '\n1. Все города',
        '2. Минимальное население',
        '3. Федеральный округ',
        '4. Население + федеральный округ',
        sep='\n',
    )

    choice = input('\nВыберите вариант [1]: ').strip()

    if choice == '':
        choice = '1'

    if choice in ['2', '4']:
        if 'population' not in df.columns:
            print('\nВ базе отсутствует информация о населении.')
        else:
            try:
                minimum_population = float(input('Введите минимальное население: '))
                df = df[df['population'] >= minimum_population]
            except ValueError:
                print('Некорректное значение населения.')

    if choice in ['3', '4']:
        if 'federal_district' not in df.columns:
            print('\nВ базе отсутствует информация о федеральных округах.')
        else:
            districts = sorted(df['federal_district'].dropna().unique().tolist())

            print('\nФедеральные округа:')

            for number, district in enumerate(districts, start=1):
                print(f'{number}. {district}')

            try:
                number = int(input('\nВведите номер округа: '))

                if not 1 <= number <= len(districts):
                    raise IndexError('Номер округа вне диапазона')

                selected_district = districts[number - 1]
                df = df[df['federal_district'] == selected_district]

                print(f'\nВыбран округ: {selected_district}')
            except (ValueError, IndexError):
                print('Некорректный номер.')

    df = df.reset_index(drop=True)

    print(f'\nПосле фильтрации: {len(df)} городов')

    return df


def prepare_folder(method_name):
    folder = os.path.join(RESULTS_FOLDER, method_name)

    if os.path.exists(folder):
        shutil.rmtree(folder)

    os.makedirs(folder, exist_ok=True)

    return folder


def save_clusters(df, folder):
    save_csv(df, os.path.join(folder, 'all_cities.csv'))

    for cluster, cluster_df in df.groupby('cluster', sort=True):
        save_csv(cluster_df, os.path.join(folder, f'cluster_{cluster}.csv'))

    print('\nРезультаты сохранены:', os.path.abspath(folder), sep='\n')


def calculate_distance_matrix(df):
    coordinates = df[['lat', 'lon']].values
    coordinates_rad = np.radians(coordinates)
    distance_matrix = haversine_distances(coordinates_rad) * EARTH_RADIUS_KM

    return distance_matrix


def find_elbow(x_values, y_values):
    """Индекс точки, наиболее удалённой от линии между концами графика."""

    x = np.array(x_values, dtype=float)
    y = np.array(y_values, dtype=float)

    # Нормализация осей перед поиском локтя.
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()

    if x_range == 0:
        return 0

    if y_range == 0:
        return 0

    x_normalized = (x - x.min()) / x_range
    y_normalized = (y - y.min()) / y_range
    points = np.column_stack((x_normalized, y_normalized))

    # Расстояния точек до линии между концами графика.
    first_point = points[0]
    last_point = points[-1]
    line_vector = last_point - first_point
    line_length = np.linalg.norm(line_vector)

    if line_length == 0:
        return 0

    line_vector = line_vector / line_length
    vectors = points - first_point
    projections = np.outer(np.dot(vectors, line_vector), line_vector)
    distances = np.linalg.norm(vectors - projections, axis=1)
    elbow_index = int(np.argmax(distances))

    return elbow_index


def run_kmeans(df):
    print('\n' + '=' * 80, 'МЕТОД №1 — K-MEANS + МЕТОД ЛОКТЯ', '=' * 80, sep='\n')

    if len(df) < 3:
        print('\nНедостаточно городов.')
        return None

    start_time = time.perf_counter()
    X = df[['lat', 'lon']].values
    max_k = min(MAX_K, len(df) - 1)
    k_values = list(range(1, max_k + 1))
    inertias = []
    models = []

    print(
        '\nПеребор количества кластеров:\n',
        f"{'K':<10}{'WCSS':>20}",
        '-' * 30,
        sep='\n',
    )

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X)
        models.append(model)
        inertia = model.inertia_
        inertias.append(inertia)
        print(f'{k:<10}{inertia:>20.2f}')

    # Выбор количества кластеров методом локтя.
    optimal_index = find_elbow(k_values, inertias)
    optimal_k = k_values[optimal_index]

    print(f'\nОптимальное количество кластеров K = {optimal_k}')

    # Выбранная модель уже обучена при переборе K.

    model = models[optimal_index]
    labels = model.labels_.copy()
    del models

    # Формирование таблицы результата.
    result = df.copy()
    result['cluster'] = labels + 1

    # Оценка качества по географическим расстояниям.
    silhouette = None

    if optimal_k > 1 and optimal_k < len(df):
        distance_matrix = calculate_distance_matrix(df)

        try:
            silhouette = silhouette_score(distance_matrix, labels, metric='precomputed')
        except ValueError:
            silhouette = None

    execution_time = time.perf_counter() - start_time

    print(
        '\n' + '=' * 80,
        'РЕЗУЛЬТАТ K-MEANS',
        '=' * 80,
        f'\nКоличество кластеров: {optimal_k}',
        sep='\n',
    )

    if silhouette is not None:
        print(f'Silhouette Score: {silhouette:.4f}')

    print(f'Время вычислений: {execution_time:.4f} с')

    for cluster in range(1, optimal_k + 1):
        cluster_df = result[result['cluster'] == cluster]

        print(
            f'\nКластер {cluster}: {len(cluster_df)} городов',
            ', '.join(cluster_df['city'].astype(str).tolist()),
            sep='\n',
        )

    # Сохранение результатов.
    folder = prepare_folder('kmeans')

    save_clusters(result, folder)

    k_search_df = pd.DataFrame({'k': k_values, 'wcss': inertias})

    save_csv(k_search_df, os.path.join(folder, 'k_search.csv'))

    plt.figure(figsize=(9, 6))
    plt.plot(k_values, inertias, marker='o')
    plt.scatter(
        optimal_k,
        inertias[optimal_index],
        s=180,
        label=f'Оптимальное K = {optimal_k}',
    )
    plt.xlabel('Количество кластеров K')
    plt.ylabel('WCSS')
    plt.title('K-means — определение K методом локтя')
    finish_plot(os.path.join(folder, 'elbow.png'))

    plt.figure(figsize=(14, 8))
    plt.scatter(result['lon'], result['lat'], c=result['cluster'], cmap='tab20', s=25)

    centers = model.cluster_centers_

    plt.scatter(
        centers[:, 1],
        centers[:, 0],
        marker='X',
        s=200,
        c='black',
        label='Центры кластеров',
    )
    plt.xlabel('Долгота')
    plt.ylabel('Широта')
    plt.title(f'K-means: K = {optimal_k}')
    finish_plot(os.path.join(folder, 'clusters.png'))

    return {
        'method': 'K-means',
        'clusters': optimal_k,
        'parameter': optimal_k,
        'parameter_name': 'K',
        'silhouette': silhouette,
        'execution_time': execution_time,
        'result': result,
    }


def custom_clustering(distance_matrix, radius):
    """Выбор самого плотного города с обновлением счётчиков соседей."""

    if not np.isfinite(radius) or radius < 0:
        raise ValueError('Радиус должен быть конечным и неотрицательным.')

    n = len(distance_matrix)
    labels = np.full(n, -1, dtype=int)
    unassigned = np.ones(n, dtype=bool)
    centers = []

    # Соседи определяются один раз для выбранного радиуса.
    neighbours = distance_matrix <= radius
    neighbour_counts = neighbours.sum(axis=1)

    while np.any(unassigned):
        center_index = int(np.argmax(neighbour_counts))
        cluster_members = np.flatnonzero(neighbours[center_index] & unassigned)

        if cluster_members.size == 0:
            raise ValueError('У города отсутствуют соседи, включая его самого.')

        labels[cluster_members] = len(centers)
        centers.append(center_index)
        unassigned[cluster_members] = False
        available = np.flatnonzero(unassigned)

        if available.size:
            removed_counts = neighbours[np.ix_(available, cluster_members)].sum(axis=1)

            # Исключаем уже распределённые города из счётчиков.
            neighbour_counts[available] -= removed_counts

        neighbour_counts[cluster_members] = -1

    return (labels, len(centers), centers)


def calculate_cluster_balance(labels):
    """Равномерность размеров кластеров от 0 до 1; не влияет на выбор R."""

    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_count = len(unique_labels)

    if cluster_count <= 1:
        return 0.0

    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log(probabilities))
    max_entropy = np.log(cluster_count)

    if max_entropy == 0:
        return 0.0

    return float(entropy / max_entropy)


def find_optimal_radius(
    distance_matrix,
    min_radius=MIN_RADIUS,
    max_radius=MAX_RADIUS,
    step=RADIUS_STEP,
):
    print(
        '\n' + '=' * 100,
        'СОБСТВЕННЫЙ МЕТОД — АВТОМАТИЧЕСКИЙ ПОИСК РАДИУСА',
        '=' * 100,
        f'\nДиапазон R: {min_radius}–{max_radius} км',
        f'Шаг: {step} км\n',
        f"{'R, км':<10}{'Кластеров':<13}{'Silhouette':<15}{'Balance':<12}{'Макс.%':<12}{'Одиночек':<12}",
        '-' * 100,
        sep='\n',
    )

    # Перебор радиусов.
    results = []
    silhouette_cache = {}
    n = len(distance_matrix)

    for radius in range(min_radius, max_radius + 1, step):
        labels, cluster_count, centers = custom_clustering(distance_matrix, radius)
        unique_labels, counts = np.unique(labels, return_counts=True)
        largest_cluster_share = counts.max() / n
        singleton_count = int(np.sum(counts == 1))
        balance = calculate_cluster_balance(labels)
        silhouette = None

        if cluster_count > 1 and cluster_count < n:
            try:
                labels_key = labels.tobytes()

                if labels_key not in silhouette_cache:
                    silhouette_cache[labels_key] = silhouette_score(
                        distance_matrix,
                        labels,
                        metric='precomputed',
                    )

                silhouette = silhouette_cache[labels_key]
            except ValueError:
                silhouette = None

        item = {
            'radius': radius,
            'clusters': cluster_count,
            'silhouette': silhouette,
            'balance': balance,
            'largest_share': largest_cluster_share,
            'singletons': singleton_count,
            'labels': labels,
            'centers': centers,
        }
        results.append(item)

        if silhouette is None:
            silhouette_text = '—'
        else:
            silhouette_text = f'{silhouette:.4f}'

        print(
            f'{radius:<10}{cluster_count:<13}{silhouette_text:<15}{balance:<12.4f}{largest_cluster_share * 100:<12.1f}{singleton_count:<12}',
        )

    print('-' * 100)

    if not results:
        print('\nНе удалось выполнить перебор радиусов.')
        return None

    radiuses = [item['radius'] for item in results]
    cluster_counts = [item['clusters'] for item in results]
    elbow_index = find_elbow(radiuses, cluster_counts)
    best_result = results[elbow_index]

    print(
        '\n' + '=' * 80,
        'ТОЧКА ЛОКТЯ ДЛЯ СОБСТВЕННОГО МЕТОДА',
        '=' * 80,
        f"\nОптимальный радиус R = {best_result['radius']} км",
        f"Полученное количество кластеров: {best_result['clusters']}",
        sep='\n',
    )

    if best_result['silhouette'] is not None:
        print(f"Silhouette Score: {best_result['silhouette']:.4f}")

    print(
        f"Balance: {best_result['balance']:.4f}",
        f"Самый большой кластер: {best_result['largest_share'] * 100:.1f}%",
        f"Кластеров-одиночек: {best_result['singletons']}",
        sep='\n',
    )

    return (best_result, results, elbow_index)


def run_custom(df):
    print(
        '\n' + '=' * 80,
        'МЕТОД №2 — АДАПТИВНАЯ РАДИУСНАЯ КЛАСТЕРИЗАЦИЯ',
        '=' * 80,
        sep='\n',
    )

    if len(df) < 3:
        print('\nНедостаточно городов.')
        return None

    start_time = time.perf_counter()

    print('\nРасчёт географических расстояний...')

    distance_matrix = calculate_distance_matrix(df)

    print('Матрица расстояний рассчитана.')

    search_result = find_optimal_radius(distance_matrix)

    if search_result is None:
        return None

    best_result, all_results, elbow_index = search_result
    optimal_radius = best_result['radius']
    labels = best_result['labels']
    centers = best_result['centers']
    cluster_count = best_result['clusters']

    # Формирование таблицы результата.
    result = df.copy()
    result['cluster'] = labels + 1
    result['is_cluster_center'] = False
    result.loc[centers, 'is_cluster_center'] = True

    # Расстояние каждого города до центра его кластера.
    center_indices = np.asarray(centers, dtype=int)[labels]
    result['distance_to_center_km'] = distance_matrix[center_indices, np.arange(len(labels))]
    execution_time = time.perf_counter() - start_time

    print(
        '\n' + '=' * 80,
        'РЕЗУЛЬТАТ СОБСТВЕННОГО МЕТОДА',
        '=' * 80,
        f'\nОптимальный радиус: {optimal_radius} км',
        f'Количество кластеров: {cluster_count}',
        sep='\n',
    )

    if best_result['silhouette'] is not None:
        print(f"Silhouette Score: {best_result['silhouette']:.4f}")

    print(
        f"Balance: {best_result['balance']:.4f}",
        f'Время вычислений: {execution_time:.4f} с',
        sep='\n',
    )

    for cluster in range(1, cluster_count + 1):
        cluster_df = result[result['cluster'] == cluster]
        center_df = cluster_df[cluster_df['is_cluster_center']]

        if not center_df.empty:
            center_name = center_df.iloc[0]['city']
        else:
            center_name = '—'

        max_distance = cluster_df['distance_to_center_km'].max()

        print(
            f'\nКластер {cluster}: {len(cluster_df)} городов',
            f'Центр: {center_name}',
            f'Максимальное расстояние от центра: {max_distance:.1f} км',
            ', '.join(cluster_df['city'].astype(str).tolist()),
            sep='\n',
        )

    # Сохранение результатов.
    folder = prepare_folder('custom')

    save_clusters(result, folder)

    # Таблица результатов для всех радиусов.
    radius_table = []

    for item in all_results:
        radius_table.append(
            {
                'radius_km': item['radius'],
                'clusters': item['clusters'],
                'silhouette': item['silhouette'],
                'balance': item['balance'],
                'largest_cluster_percent': item['largest_share'] * 100,
                'singletons': item['singletons'],
            },
        )

    radius_df = pd.DataFrame(radius_table)

    save_csv(radius_df, os.path.join(folder, 'radius_search.csv'))

    radiuses = [item['radius'] for item in all_results]
    cluster_counts = [item['clusters'] for item in all_results]

    plt.figure(figsize=(10, 6))
    plt.plot(radiuses, cluster_counts, marker='o')
    plt.scatter(
        optimal_radius,
        cluster_count,
        s=180,
        label=f'Оптимальный R = {optimal_radius} км',
    )
    plt.xlabel('Радиус R, км')
    plt.ylabel('Количество кластеров')
    plt.title('Собственный метод — определение радиуса методом локтя')
    finish_plot(os.path.join(folder, 'radius_elbow.png'))

    silhouette_radiuses = []
    silhouettes = []

    for item in all_results:
        if item['silhouette'] is not None:
            silhouette_radiuses.append(item['radius'])
            silhouettes.append(item['silhouette'])

    if silhouettes:

        plt.figure(figsize=(10, 6))
        plt.plot(silhouette_radiuses, silhouettes, marker='o')
        plt.axvline(
            x=optimal_radius,
            linestyle='--',
            label=f'Выбранный R = {optimal_radius} км',
        )
        plt.xlabel('Радиус R, км')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score при различных радиусах')
        finish_plot(os.path.join(folder, 'silhouette_by_radius.png'))

    balances = [item['balance'] for item in all_results]

    plt.figure(figsize=(10, 6))
    plt.plot(radiuses, balances, marker='o')
    plt.axvline(
        x=optimal_radius,
        linestyle='--',
        label=f'Выбранный R = {optimal_radius} км',
    )
    plt.xlabel('Радиус R, км')
    plt.ylabel('Balance')
    plt.title('Сбалансированность кластеров при различных радиусах')
    finish_plot(os.path.join(folder, 'balance_by_radius.png'))

    plt.figure(figsize=(14, 8))
    plt.scatter(result['lon'], result['lat'], c=result['cluster'], cmap='tab20', s=25)

    center_rows = result[result['is_cluster_center']]

    plt.scatter(
        center_rows['lon'],
        center_rows['lat'],
        marker='X',
        s=180,
        c='black',
        label='Центры кластеров',
    )
    plt.xlabel('Долгота')
    plt.ylabel('Широта')
    plt.title(f'Собственный метод: R = {optimal_radius} км, кластеров = {cluster_count}')
    finish_plot(os.path.join(folder, 'clusters.png'))

    return {
        'method': 'Адаптивная радиусная кластеризация',
        'clusters': cluster_count,
        'parameter': optimal_radius,
        'parameter_name': 'R, км',
        'silhouette': best_result['silhouette'],
        'balance': best_result['balance'],
        'execution_time': execution_time,
        'result': result,
    }


def compare_methods(kmeans_result, custom_result):
    if kmeans_result is None or custom_result is None:
        return

    print(
        '\n' + '=' * 85,
        'СРАВНЕНИЕ МЕТОДОВ',
        '=' * 85,
        f"\n{'Показатель':<30}{'K-means':<22}{'Собственный метод':<25}",
        '-' * 77,
        f"{'Количество кластеров':<30}{kmeans_result['clusters']:<22}{custom_result['clusters']:<25}",
        sep='\n',
    )

    if kmeans_result['silhouette'] is not None:
        k_silhouette = f"{kmeans_result['silhouette']:.4f}"
    else:
        k_silhouette = '—'

    if custom_result['silhouette'] is not None:
        custom_silhouette = f"{custom_result['silhouette']:.4f}"
    else:
        custom_silhouette = '—'

    print(
        f"{'Silhouette Score':<30}{k_silhouette:<22}{custom_silhouette:<25}",
        f"{'Оптимальное K':<30}{kmeans_result['parameter']:<22}{'-':<25}",
        f"{'Оптимальный R, км':<30}{'-':<22}{custom_result['parameter']:<25}",
        f"{'Balance':<30}{'-':<22}{custom_result['balance']:<25.4f}",
        f"{'Время вычислений, с':<30}{kmeans_result['execution_time']:<22.4f}{custom_result['execution_time']:<25.4f}",
        sep='\n',
    )

    comparison = pd.DataFrame(
        {
            'method': ['K-means', 'Adaptive radius clustering'],
            'clusters': [kmeans_result['clusters'], custom_result['clusters']],
            'parameter_name': ['K', 'R, km'],
            'parameter': [kmeans_result['parameter'], custom_result['parameter']],
            'silhouette': [kmeans_result['silhouette'], custom_result['silhouette']],
            'balance': [np.nan, custom_result['balance']],
            'execution_time_seconds': [kmeans_result['execution_time'], custom_result['execution_time']],
        },
    )
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    comparison_file = os.path.join(RESULTS_FOLDER, 'comparison.csv')

    save_csv(comparison, comparison_file)

    print('\nСравнение сохранено:', comparison_file, sep='\n')


def main():
    print(
        '\n' + '=' * 70,
        'КЛАСТЕРИЗАЦИЯ ГОРОДОВ РОССИИ',
        'Сравнение стандартного и собственного методов',
        '=' * 70,
        sep='\n',
    )

    original_df = load_data()

    if original_df is None:
        return

    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    current_df = original_df.copy()

    while True:
        print(
            '\n' + '=' * 70,
            'ГЛАВНОЕ МЕНЮ',
            '=' * 70,
            f'\nТекущий набор данных: {len(current_df)} городов\n',
            '1. Показать информацию о данных',
            '2. Настроить фильтрацию',
            '3. Сбросить фильтры',
            '4. Запустить K-means + метод локтя',
            '5. Запустить собственный метод',
            '6. Запустить оба метода и сравнить',
            '0. Выход',
            sep='\n',
        )

        choice = input('\nВыберите действие: ').strip()

        if choice == '1':
            show_data_info(current_df)
        elif choice == '2':
            current_df = filter_data(original_df)
        elif choice == '3':
            current_df = original_df.copy()

            print('\nФильтры сброшены.')
        elif choice == '4':
            run_kmeans(current_df)
        elif choice == '5':
            run_custom(current_df)
        elif choice == '6':
            print('\nЗапуск K-means...')

            kmeans_result = run_kmeans(current_df)

            print('\nЗапуск собственного метода...')

            custom_result = run_custom(current_df)
            compare_methods(kmeans_result, custom_result)
        elif choice == '0':
            print('\nРабота программы завершена.')
            break
        else:
            print('\nНеизвестная команда.')



if __name__ == '__main__':
    main()
