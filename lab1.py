import os
import shutil
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import haversine_distances


# ============================================================
# НАСТРОЙКИ
# ============================================================

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "epogrebnyak/ru-cities/main/assets/towns.csv"
)

EARTH_RADIUS_KM = 6371.0

# ------------------------------------------------------------
# Папка проекта
# ------------------------------------------------------------

PROJECT_FOLDER = os.path.dirname(
    os.path.abspath(__file__)
)

# ------------------------------------------------------------
# Локальная база данных
# ------------------------------------------------------------

DATA_FOLDER = os.path.join(
    PROJECT_FOLDER,
    "data"
)

DATA_FILE = os.path.join(
    DATA_FOLDER,
    "towns.csv"
)

# ------------------------------------------------------------
# Результаты
# ------------------------------------------------------------

RESULTS_FOLDER = os.path.join(
    PROJECT_FOLDER,
    "clusters"
)

# ------------------------------------------------------------
# K-means
# ------------------------------------------------------------

MAX_K = 20

# ------------------------------------------------------------
# Собственный метод
# ------------------------------------------------------------

MIN_RADIUS = 50
MAX_RADIUS = 2000
RADIUS_STEP = 50


# ============================================================
# ЗАГРУЗКА БАЗЫ
# ============================================================

def load_data():

    print("\n" + "=" * 70)
    print("ЗАГРУЗКА БАЗЫ ГОРОДОВ РОССИИ")
    print("=" * 70)

    os.makedirs(
        DATA_FOLDER,
        exist_ok=True
    )

    # ========================================================
    # БАЗА УЖЕ СКАЧАНА
    # ========================================================

    if os.path.exists(DATA_FILE):

        print("\nИспользуется локальная база данных.")
        print(DATA_FILE)

        try:

            df = pd.read_csv(
                DATA_FILE
            )

        except Exception as error:

            print("\nОшибка чтения базы:")
            print(error)

            return None

    # ========================================================
    # ПЕРВЫЙ ЗАПУСК
    # ========================================================

    else:

        print("\nЛокальная база данных не найдена.")
        print("Выполняется первоначальная загрузка...")

        try:

            df = pd.read_csv(
                DATA_URL
            )

        except Exception as error:

            print("\nНе удалось скачать базу.")
            print("Проверьте подключение к интернету.")
            print(error)

            return None

        try:

            df.to_csv(
                DATA_FILE,
                index=False,
                encoding="utf-8-sig"
            )

            print("\nБаза успешно скачана.")
            print(f"Файл: {DATA_FILE}")

            print(
                "При следующих запусках "
                "скачивание выполняться не будет."
            )

        except Exception as error:

            print("\nОшибка сохранения базы:")
            print(error)

            return None

    # ========================================================
    # ПРОВЕРКА
    # ========================================================

    required_columns = [
        "city",
        "lat",
        "lon"
    ]

    for column in required_columns:

        if column not in df.columns:

            print(
                f"\nОшибка: отсутствует "
                f"столбец '{column}'."
            )

            return None

    # --------------------------------------------------------
    # Координаты
    # --------------------------------------------------------

    df["lat"] = pd.to_numeric(
        df["lat"],
        errors="coerce"
    )

    df["lon"] = pd.to_numeric(
        df["lon"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Население
    # --------------------------------------------------------

    if "population" in df.columns:

        df["population"] = pd.to_numeric(
            df["population"],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Удаляем строки без координат
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "city",
            "lat",
            "lon"
        ]
    )

    df = df.reset_index(
        drop=True
    )

    print(
        f"\nБаза готова."
    )

    print(
        f"Количество городов: {len(df)}"
    )

    return df


# ============================================================
# ИНФОРМАЦИЯ О ДАННЫХ
# ============================================================

def show_data_info(df):

    print("\n" + "=" * 70)
    print("ИНФОРМАЦИЯ О ДАННЫХ")
    print("=" * 70)

    print(
        f"\nКоличество городов: "
        f"{len(df)}"
    )

    print("\nПоля базы:")

    for column in df.columns:

        print(
            f" - {column}"
        )

    possible_columns = [
        "city",
        "region_name",
        "federal_district",
        "population",
        "lat",
        "lon"
    ]

    columns = [
        column
        for column in possible_columns
        if column in df.columns
    ]

    print("\nПервые 20 городов:\n")

    print(
        df[columns]
        .head(20)
        .to_string(
            index=False
        )
    )


# ============================================================
# ФИЛЬТРАЦИЯ
# ============================================================

def filter_data(original_df):

    df = original_df.copy()

    print("\n" + "=" * 70)
    print("ФИЛЬТРАЦИЯ")
    print("=" * 70)

    print("\n1. Все города")
    print("2. Минимальное население")
    print("3. Федеральный округ")
    print("4. Население + федеральный округ")

    choice = input(
        "\nВыберите вариант [1]: "
    ).strip()

    if choice == "":
        choice = "1"

    # ========================================================
    # НАСЕЛЕНИЕ
    # ========================================================

    if choice in ["2", "4"]:

        if "population" not in df.columns:

            print(
                "\nВ базе отсутствует "
                "информация о населении."
            )

        else:

            try:

                minimum_population = float(
                    input(
                        "Введите минимальное население: "
                    )
                )

                df = df[
                    df["population"]
                    >= minimum_population
                ]

            except ValueError:

                print(
                    "Некорректное значение населения."
                )

    # ========================================================
    # ФЕДЕРАЛЬНЫЙ ОКРУГ
    # ========================================================

    if choice in ["3", "4"]:

        if "federal_district" not in df.columns:

            print(
                "\nВ базе отсутствует "
                "информация о федеральных округах."
            )

        else:

            districts = sorted(
                df[
                    "federal_district"
                ]
                .dropna()
                .unique()
                .tolist()
            )

            print(
                "\nФедеральные округа:"
            )

            for number, district in enumerate(
                districts,
                start=1
            ):

                print(
                    f"{number}. {district}"
                )

            try:

                number = int(
                    input(
                        "\nВведите номер округа: "
                    )
                )

                selected_district = districts[
                    number - 1
                ]

                df = df[
                    df["federal_district"]
                    == selected_district
                ]

                print(
                    f"\nВыбран округ: "
                    f"{selected_district}"
                )

            except (
                ValueError,
                IndexError
            ):

                print(
                    "Некорректный номер."
                )

    df = df.reset_index(
        drop=True
    )

    print(
        f"\nПосле фильтрации: "
        f"{len(df)} городов"
    )

    return df


# ============================================================
# ПАПКА РЕЗУЛЬТАТОВ
# ============================================================

def prepare_folder(method_name):

    folder = os.path.join(
        RESULTS_FOLDER,
        method_name
    )

    if os.path.exists(folder):

        shutil.rmtree(
            folder
        )

    os.makedirs(
        folder,
        exist_ok=True
    )

    return folder


# ============================================================
# СОХРАНЕНИЕ КЛАСТЕРОВ
# ============================================================

def save_clusters(df, folder):

    # --------------------------------------------------------
    # Все города
    # --------------------------------------------------------

    df.to_csv(
        os.path.join(
            folder,
            "all_cities.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Каждый кластер отдельно
    # --------------------------------------------------------

    clusters = sorted(
        df["cluster"].unique()
    )

    for cluster in clusters:

        cluster_df = df[
            df["cluster"]
            == cluster
        ].copy()

        cluster_df.to_csv(
            os.path.join(
                folder,
                f"cluster_{cluster}.csv"
            ),
            index=False,
            encoding="utf-8-sig"
        )

    print(
        "\nРезультаты сохранены:"
    )

    print(
        os.path.abspath(
            folder
        )
    )


# ============================================================
# МАТРИЦА ГЕОГРАФИЧЕСКИХ РАССТОЯНИЙ
# ============================================================

def calculate_distance_matrix(df):

    coordinates = df[
        [
            "lat",
            "lon"
        ]
    ].values

    # Haversine требует координаты в радианах
    coordinates_rad = np.radians(
        coordinates
    )

    distance_matrix = (
        haversine_distances(
            coordinates_rad
        )
        * EARTH_RADIUS_KM
    )

    return distance_matrix


# ============================================================
# УНИВЕРСАЛЬНЫЙ ПОИСК ЛОКТЯ
# ============================================================

def find_elbow(
        x_values,
        y_values
):

    """
    Поиск точки, максимально удалённой
    от прямой между первой и последней
    точками графика.

    Перед вычислением обе оси
    нормализуются в диапазон 0..1.
    """

    x = np.array(
        x_values,
        dtype=float
    )

    y = np.array(
        y_values,
        dtype=float
    )

    x_range = (
        x.max()
        - x.min()
    )

    y_range = (
        y.max()
        - y.min()
    )

    if x_range == 0:
        return 0

    if y_range == 0:
        return 0

    # --------------------------------------------------------
    # Нормализация
    # --------------------------------------------------------

    x_normalized = (
        (x - x.min())
        / x_range
    )

    y_normalized = (
        (y - y.min())
        / y_range
    )

    points = np.column_stack(
        (
            x_normalized,
            y_normalized
        )
    )

    first_point = points[0]
    last_point = points[-1]

    line_vector = (
        last_point
        - first_point
    )

    line_length = np.linalg.norm(
        line_vector
    )

    if line_length == 0:
        return 0

    line_vector = (
        line_vector
        / line_length
    )

    vectors = (
        points
        - first_point
    )

    projections = np.outer(
        np.dot(
            vectors,
            line_vector
        ),
        line_vector
    )

    distances = np.linalg.norm(
        vectors
        - projections,
        axis=1
    )

    elbow_index = int(
        np.argmax(
            distances
        )
    )

    return elbow_index


# ============================================================
# МЕТОД №1 — K-MEANS
# ============================================================

def run_kmeans(df):

    print("\n" + "=" * 80)
    print("МЕТОД №1 — K-MEANS + МЕТОД ЛОКТЯ")
    print("=" * 80)

    if len(df) < 3:

        print(
            "\nНедостаточно городов."
        )

        return None

    # ========================================================
    # ИЗМЕРЕНИЕ ВРЕМЕНИ
    # ========================================================

    # perf_counter() подходит для измерения длительности
    # выполнения вычислительных операций.
    start_time = time.perf_counter()

    # --------------------------------------------------------
    # Координаты
    # --------------------------------------------------------

    X = df[
        [
            "lat",
            "lon"
        ]
    ].values

    max_k = min(
        MAX_K,
        len(df) - 1
    )

    k_values = list(
        range(
            1,
            max_k + 1
        )
    )

    inertias = []

    # ========================================================
    # ПЕРЕБОР K
    # ========================================================

    print("\nПеребор количества кластеров:\n")

    print(
        f"{'K':<10}"
        f"{'WCSS':>20}"
    )

    print(
        "-" * 30
    )

    for k in k_values:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(
            X
        )

        inertia = (
            model.inertia_
        )

        inertias.append(
            inertia
        )

        print(
            f"{k:<10}"
            f"{inertia:>20.2f}"
        )

    # ========================================================
    # МЕТОД ЛОКТЯ
    # ========================================================

    optimal_index = find_elbow(
        k_values,
        inertias
    )

    optimal_k = k_values[
        optimal_index
    ]

    print(
        f"\nОптимальное количество "
        f"кластеров K = {optimal_k}"
    )

    # ========================================================
    # ФИНАЛЬНЫЙ K-MEANS
    # ========================================================

    model = KMeans(
        n_clusters=optimal_k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(
        X
    )

    result = df.copy()

    result[
        "cluster"
    ] = labels + 1

    # ========================================================
    # SILHOUETTE ПО ГЕОГРАФИЧЕСКИМ РАССТОЯНИЯМ
    # ========================================================

    silhouette = None

    if (
        optimal_k > 1
        and
        optimal_k < len(df)
    ):

        distance_matrix = (
            calculate_distance_matrix(
                df
            )
        )

        try:

            silhouette = silhouette_score(
                distance_matrix,
                labels,
                metric="precomputed"
            )

        except ValueError:

            silhouette = None

    # ========================================================
    # ВРЕМЯ ВЫПОЛНЕНИЯ
    # ========================================================

    execution_time = (
        time.perf_counter()
        - start_time
    )

    # ========================================================
    # РЕЗУЛЬТАТ
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "РЕЗУЛЬТАТ K-MEANS"
    )

    print(
        "=" * 80
    )

    print(
        f"\nКоличество кластеров: "
        f"{optimal_k}"
    )

    if silhouette is not None:

        print(
            f"Silhouette Score: "
            f"{silhouette:.4f}"
        )

    print(
        f"Время вычислений: "
        f"{execution_time:.4f} с"
    )

    # --------------------------------------------------------
    # Размеры кластеров
    # --------------------------------------------------------

    for cluster in range(
        1,
        optimal_k + 1
    ):

        cluster_df = result[
            result["cluster"]
            == cluster
        ]

        print(
            f"\nКластер {cluster}: "
            f"{len(cluster_df)} городов"
        )

        print(
            ", ".join(
                cluster_df[
                    "city"
                ]
                .astype(str)
                .tolist()
            )
        )

    # ========================================================
    # СОХРАНЕНИЕ
    # ========================================================

    folder = prepare_folder(
        "kmeans"
    )

    save_clusters(
        result,
        folder
    )

    # --------------------------------------------------------
    # Таблица перебора K
    # --------------------------------------------------------

    k_search_df = pd.DataFrame({

        "k":
            k_values,

        "wcss":
            inertias
    })

    k_search_df.to_csv(
        os.path.join(
            folder,
            "k_search.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # ГРАФИК ЛОКТЯ
    # ========================================================

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        k_values,
        inertias,
        marker="o"
    )

    plt.scatter(
        optimal_k,
        inertias[
            optimal_index
        ],
        s=180,
        label=(
            f"Оптимальное K = "
            f"{optimal_k}"
        )
    )

    plt.xlabel(
        "Количество кластеров K"
    )

    plt.ylabel(
        "WCSS"
    )

    plt.title(
        "K-means — определение K методом локтя"
    )

    plt.legend()
    plt.grid()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            folder,
            "elbow.png"
        ),
        dpi=200
    )

    plt.show()

    # ========================================================
    # КАРТА K-MEANS
    # ========================================================

    plt.figure(
        figsize=(14, 8)
    )

    plt.scatter(
        result["lon"],
        result["lat"],
        c=result["cluster"],
        cmap="tab20",
        s=25
    )

    centers = (
        model.cluster_centers_
    )

    plt.scatter(
        centers[:, 1],
        centers[:, 0],
        marker="X",
        s=200,
        c="black",
        label="Центры кластеров"
    )

    plt.xlabel(
        "Долгота"
    )

    plt.ylabel(
        "Широта"
    )

    plt.title(
        f"K-means: K = {optimal_k}"
    )

    plt.legend()
    plt.grid()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            folder,
            "clusters.png"
        ),
        dpi=200
    )

    plt.show()

    return {

        "method":
            "K-means",

        "clusters":
            optimal_k,

        "parameter":
            optimal_k,

        "parameter_name":
            "K",

        "silhouette":
            silhouette,

        "execution_time":
            execution_time,

        "result":
            result
    }


# ============================================================
# СОБСТВЕННЫЙ МЕТОД КЛАСТЕРИЗАЦИИ
# ============================================================

def custom_clustering(
        distance_matrix,
        radius
):

    """
    Адаптивная радиусная кластеризация.

    1. Рассматриваются ещё не распределённые города.

    2. Для каждого города определяется количество
       нераспределённых городов в радиусе R.

    3. Город с максимальным количеством соседей
       становится центром нового кластера.

    4. Все нераспределённые города на расстоянии
       не более R от этого центра включаются
       в кластер.

    5. Процесс повторяется до распределения
       всех городов.

    Количество кластеров заранее не задаётся.
    """

    n = len(
        distance_matrix
    )

    labels = np.full(
        n,
        -1,
        dtype=int
    )

    unassigned = np.ones(
        n,
        dtype=bool
    )

    centers = []

    cluster_number = 0

    # ========================================================
    # ФОРМИРОВАНИЕ КЛАСТЕРОВ
    # ========================================================

    while np.any(
        unassigned
    ):

        # ----------------------------------------------------
        # Нераспределённые города
        # ----------------------------------------------------

        available = np.where(
            unassigned
        )[0]

        # ----------------------------------------------------
        # Матрица расстояний между ними
        # ----------------------------------------------------

        submatrix = distance_matrix[
            np.ix_(
                available,
                available
            )
        ]

        # ----------------------------------------------------
        # Количество соседей в радиусе R
        # ----------------------------------------------------

        neighbour_counts = np.sum(
            submatrix <= radius,
            axis=1
        )

        # ----------------------------------------------------
        # Самый плотный город
        # ----------------------------------------------------

        best_local_index = int(
            np.argmax(
                neighbour_counts
            )
        )

        center_index = available[
            best_local_index
        ]

        centers.append(
            center_index
        )

        # ----------------------------------------------------
        # Все города в радиусе R
        # ----------------------------------------------------

        cluster_members = available[
            distance_matrix[
                center_index,
                available
            ] <= radius
        ]

        # ----------------------------------------------------
        # Присваиваем номер кластера
        # ----------------------------------------------------

        labels[
            cluster_members
        ] = cluster_number

        # ----------------------------------------------------
        # Удаляем распределённые города
        # ----------------------------------------------------

        unassigned[
            cluster_members
        ] = False

        cluster_number += 1

    return (
        labels,
        cluster_number,
        centers
    )


# ============================================================
# БАЛАНС КЛАСТЕРОВ
# ============================================================

def calculate_cluster_balance(
        labels
):

    """
    Дополнительная характеристика результата.

    1.0 — размеры кластеров относительно равномерны.

    Значение ближе к 0 означает сильное
    доминирование отдельных кластеров.

    Balance НЕ используется для выбора R.
    """

    unique_labels, counts = np.unique(
        labels,
        return_counts=True
    )

    cluster_count = len(
        unique_labels
    )

    if cluster_count <= 1:
        return 0.0

    probabilities = (
        counts
        / counts.sum()
    )

    entropy = -np.sum(
        probabilities
        * np.log(
            probabilities
        )
    )

    max_entropy = np.log(
        cluster_count
    )

    if max_entropy == 0:
        return 0.0

    return float(
        entropy
        / max_entropy
    )


# ============================================================
# ПОИСК ОПТИМАЛЬНОГО РАДИУСА
# ============================================================

def find_optimal_radius(
        distance_matrix,
        min_radius=MIN_RADIUS,
        max_radius=MAX_RADIUS,
        step=RADIUS_STEP
):

    print("\n" + "=" * 100)

    print(
        "СОБСТВЕННЫЙ МЕТОД — "
        "АВТОМАТИЧЕСКИЙ ПОИСК РАДИУСА"
    )

    print("=" * 100)

    print(
        f"\nДиапазон R: "
        f"{min_radius}–{max_radius} км"
    )

    print(
        f"Шаг: {step} км\n"
    )

    print(
        f"{'R, км':<10}"
        f"{'Кластеров':<13}"
        f"{'Silhouette':<15}"
        f"{'Balance':<12}"
        f"{'Макс.%':<12}"
        f"{'Одиночек':<12}"
    )

    print(
        "-" * 100
    )

    results = []

    n = len(
        distance_matrix
    )

    # ========================================================
    # ПЕРЕБОР РАДИУСА
    # ========================================================

    for radius in range(
        min_radius,
        max_radius + 1,
        step
    ):

        (
            labels,
            cluster_count,
            centers
        ) = custom_clustering(
            distance_matrix,
            radius
        )

        # ----------------------------------------------------
        # Размеры кластеров
        # ----------------------------------------------------

        unique_labels, counts = np.unique(
            labels,
            return_counts=True
        )

        # ----------------------------------------------------
        # Самый большой кластер
        # ----------------------------------------------------

        largest_cluster_share = (
            counts.max()
            / n
        )

        # ----------------------------------------------------
        # Количество одиночных кластеров
        # ----------------------------------------------------

        singleton_count = int(
            np.sum(
                counts == 1
            )
        )

        # ----------------------------------------------------
        # Balance
        # ----------------------------------------------------

        balance = (
            calculate_cluster_balance(
                labels
            )
        )

        # ----------------------------------------------------
        # Silhouette
        # ----------------------------------------------------

        silhouette = None

        if (
            cluster_count > 1
            and
            cluster_count < n
        ):

            try:

                silhouette = silhouette_score(
                    distance_matrix,
                    labels,
                    metric="precomputed"
                )

            except ValueError:

                silhouette = None

        # ----------------------------------------------------
        # Сохраняем результат
        # ----------------------------------------------------

        item = {

            "radius":
                radius,

            "clusters":
                cluster_count,

            "silhouette":
                silhouette,

            "balance":
                balance,

            "largest_share":
                largest_cluster_share,

            "singletons":
                singleton_count,

            "labels":
                labels,

            "centers":
                centers
        }

        results.append(
            item
        )

        # ----------------------------------------------------
        # Вывод строки
        # ----------------------------------------------------

        if silhouette is None:

            silhouette_text = "—"

        else:

            silhouette_text = (
                f"{silhouette:.4f}"
            )

        print(
            f"{radius:<10}"
            f"{cluster_count:<13}"
            f"{silhouette_text:<15}"
            f"{balance:<12.4f}"
            f"{largest_cluster_share * 100:<12.1f}"
            f"{singleton_count:<12}"
        )

    print(
        "-" * 100
    )

    if not results:

        print(
            "\nНе удалось выполнить "
            "перебор радиусов."
        )

        return None

    # ========================================================
    # МЕТОД ЛОКТЯ
    # ========================================================

    radiuses = [
        item["radius"]
        for item in results
    ]

    cluster_counts = [
        item["clusters"]
        for item in results
    ]

    elbow_index = find_elbow(
        radiuses,
        cluster_counts
    )

    best_result = results[
        elbow_index
    ]

    # ========================================================
    # РЕЗУЛЬТАТ
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "ТОЧКА ЛОКТЯ ДЛЯ СОБСТВЕННОГО МЕТОДА"
    )

    print(
        "=" * 80
    )

    print(
        f"\nОптимальный радиус R = "
        f"{best_result['radius']} км"
    )

    print(
        f"Полученное количество кластеров: "
        f"{best_result['clusters']}"
    )

    if (
        best_result[
            "silhouette"
        ]
        is not None
    ):

        print(
            f"Silhouette Score: "
            f"{best_result['silhouette']:.4f}"
        )

    print(
        f"Balance: "
        f"{best_result['balance']:.4f}"
    )

    print(
        f"Самый большой кластер: "
        f"{best_result['largest_share'] * 100:.1f}%"
    )

    print(
        f"Кластеров-одиночек: "
        f"{best_result['singletons']}"
    )

    return (
        best_result,
        results,
        elbow_index
    )


# ============================================================
# ЗАПУСК СОБСТВЕННОГО МЕТОДА
# ============================================================

def run_custom(df):

    print("\n" + "=" * 80)

    print(
        "МЕТОД №2 — "
        "АДАПТИВНАЯ РАДИУСНАЯ КЛАСТЕРИЗАЦИЯ"
    )

    print("=" * 80)

    if len(df) < 3:

        print(
            "\nНедостаточно городов."
        )

        return None

    # ========================================================
    # ИЗМЕРЕНИЕ ВРЕМЕНИ
    # ========================================================

    start_time = time.perf_counter()

    # ========================================================
    # МАТРИЦА РАССТОЯНИЙ
    # ========================================================

    print(
        "\nРасчёт географических расстояний..."
    )

    distance_matrix = (
        calculate_distance_matrix(
            df
        )
    )

    print(
        "Матрица расстояний рассчитана."
    )

    # ========================================================
    # ПОИСК R
    # ========================================================

    search_result = find_optimal_radius(
        distance_matrix
    )

    if search_result is None:

        return None

    (
        best_result,
        all_results,
        elbow_index
    ) = search_result

    optimal_radius = (
        best_result[
            "radius"
        ]
    )

    labels = (
        best_result[
            "labels"
        ]
    )

    centers = (
        best_result[
            "centers"
        ]
    )

    cluster_count = (
        best_result[
            "clusters"
        ]
    )

    # ========================================================
    # ТАБЛИЦА РЕЗУЛЬТАТА
    # ========================================================

    result = df.copy()

    result[
        "cluster"
    ] = labels + 1

    result[
        "is_cluster_center"
    ] = False

    result.loc[
        centers,
        "is_cluster_center"
    ] = True

    # --------------------------------------------------------
    # Расстояние города до центра
    # --------------------------------------------------------

    result[
        "distance_to_center_km"
    ] = 0.0

    for cluster_index, center_index in enumerate(
        centers
    ):

        members = np.where(
            labels == cluster_index
        )[0]

        result.loc[
            members,
            "distance_to_center_km"
        ] = distance_matrix[
            center_index,
            members
        ]

    # ========================================================
    # ВРЕМЯ ВЫПОЛНЕНИЯ
    # ========================================================

    execution_time = (
        time.perf_counter()
        - start_time
    )

    # ========================================================
    # ВЫВОД
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "РЕЗУЛЬТАТ СОБСТВЕННОГО МЕТОДА"
    )

    print(
        "=" * 80
    )

    print(
        f"\nОптимальный радиус: "
        f"{optimal_radius} км"
    )

    print(
        f"Количество кластеров: "
        f"{cluster_count}"
    )

    if (
        best_result[
            "silhouette"
        ]
        is not None
    ):

        print(
            f"Silhouette Score: "
            f"{best_result['silhouette']:.4f}"
        )

    print(
        f"Balance: "
        f"{best_result['balance']:.4f}"
    )

    print(
        f"Время вычислений: "
        f"{execution_time:.4f} с"
    )

    # ========================================================
    # КЛАСТЕРЫ
    # ========================================================

    for cluster in range(
        1,
        cluster_count + 1
    ):

        cluster_df = result[
            result["cluster"]
            == cluster
        ]

        center_df = cluster_df[
            cluster_df[
                "is_cluster_center"
            ]
        ]

        if not center_df.empty:

            center_name = (
                center_df.iloc[0][
                    "city"
                ]
            )

        else:

            center_name = "—"

        max_distance = (
            cluster_df[
                "distance_to_center_km"
            ].max()
        )

        print(
            f"\nКластер {cluster}: "
            f"{len(cluster_df)} городов"
        )

        print(
            f"Центр: "
            f"{center_name}"
        )

        print(
            f"Максимальное расстояние "
            f"от центра: "
            f"{max_distance:.1f} км"
        )

        print(
            ", ".join(
                cluster_df[
                    "city"
                ]
                .astype(str)
                .tolist()
            )
        )

    # ========================================================
    # СОХРАНЕНИЕ
    # ========================================================

    folder = prepare_folder(
        "custom"
    )

    save_clusters(
        result,
        folder
    )

    # ========================================================
    # ТАБЛИЦА ПЕРЕБОРА R
    # ========================================================

    radius_table = []

    for item in all_results:

        radius_table.append({

            "radius_km":
                item["radius"],

            "clusters":
                item["clusters"],

            "silhouette":
                item["silhouette"],

            "balance":
                item["balance"],

            "largest_cluster_percent":
                item["largest_share"]
                * 100,

            "singletons":
                item["singletons"]
        })

    radius_df = pd.DataFrame(
        radius_table
    )

    radius_df.to_csv(
        os.path.join(
            folder,
            "radius_search.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # ГРАФИК ЛОКТЯ ДЛЯ R
    # ========================================================

    radiuses = [
        item["radius"]
        for item in all_results
    ]

    cluster_counts = [
        item["clusters"]
        for item in all_results
    ]

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        radiuses,
        cluster_counts,
        marker="o"
    )

    plt.scatter(
        optimal_radius,
        cluster_count,
        s=180,
        label=(
            f"Оптимальный R = "
            f"{optimal_radius} км"
        )
    )

    plt.xlabel(
        "Радиус R, км"
    )

    plt.ylabel(
        "Количество кластеров"
    )

    plt.title(
        "Собственный метод — "
        "определение радиуса методом локтя"
    )

    plt.legend()
    plt.grid()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            folder,
            "radius_elbow.png"
        ),
        dpi=200
    )

    plt.show()

    # ========================================================
    # SILHOUETTE ОТ РАДИУСА
    # ========================================================

    silhouette_radiuses = []

    silhouettes = []

    for item in all_results:

        if item["silhouette"] is not None:

            silhouette_radiuses.append(
                item["radius"]
            )

            silhouettes.append(
                item["silhouette"]
            )

    if silhouettes:

        plt.figure(
            figsize=(10, 6)
        )

        plt.plot(
            silhouette_radiuses,
            silhouettes,
            marker="o"
        )

        plt.axvline(
            x=optimal_radius,
            linestyle="--",
            label=(
                f"Выбранный R = "
                f"{optimal_radius} км"
            )
        )

        plt.xlabel(
            "Радиус R, км"
        )

        plt.ylabel(
            "Silhouette Score"
        )

        plt.title(
            "Silhouette Score "
            "при различных радиусах"
        )

        plt.legend()
        plt.grid()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                folder,
                "silhouette_by_radius.png"
            ),
            dpi=200
        )

        plt.show()

    # ========================================================
    # BALANCE ОТ РАДИУСА
    # ========================================================

    balances = [
        item["balance"]
        for item in all_results
    ]

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        radiuses,
        balances,
        marker="o"
    )

    plt.axvline(
        x=optimal_radius,
        linestyle="--",
        label=(
            f"Выбранный R = "
            f"{optimal_radius} км"
        )
    )

    plt.xlabel(
        "Радиус R, км"
    )

    plt.ylabel(
        "Balance"
    )

    plt.title(
        "Сбалансированность кластеров "
        "при различных радиусах"
    )

    plt.legend()
    plt.grid()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            folder,
            "balance_by_radius.png"
        ),
        dpi=200
    )

    plt.show()

    # ========================================================
    # КАРТА КЛАСТЕРОВ
    # ========================================================

    plt.figure(
        figsize=(14, 8)
    )

    plt.scatter(
        result["lon"],
        result["lat"],
        c=result["cluster"],
        cmap="tab20",
        s=25
    )

    center_rows = result[
        result[
            "is_cluster_center"
        ]
    ]

    plt.scatter(
        center_rows["lon"],
        center_rows["lat"],
        marker="X",
        s=180,
        c="black",
        label="Центры кластеров"
    )

    plt.xlabel(
        "Долгота"
    )

    plt.ylabel(
        "Широта"
    )

    plt.title(
        f"Собственный метод: "
        f"R = {optimal_radius} км, "
        f"кластеров = {cluster_count}"
    )

    plt.legend()
    plt.grid()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            folder,
            "clusters.png"
        ),
        dpi=200
    )

    plt.show()

    return {

        "method":
            "Адаптивная радиусная кластеризация",

        "clusters":
            cluster_count,

        "parameter":
            optimal_radius,

        "parameter_name":
            "R, км",

        "silhouette":
            best_result[
                "silhouette"
            ],

        "balance":
            best_result[
                "balance"
            ],

        "execution_time":
            execution_time,

        "result":
            result
    }


# ============================================================
# СРАВНЕНИЕ МЕТОДОВ
# ============================================================

def compare_methods(
        kmeans_result,
        custom_result
):

    if (
        kmeans_result is None
        or
        custom_result is None
    ):

        return

    print(
        "\n" + "=" * 85
    )

    print(
        "СРАВНЕНИЕ МЕТОДОВ"
    )

    print(
        "=" * 85
    )

    print(
        f"\n{'Показатель':<30}"
        f"{'K-means':<22}"
        f"{'Собственный метод':<25}"
    )

    print(
        "-" * 77
    )

    print(
        f"{'Количество кластеров':<30}"
        f"{kmeans_result['clusters']:<22}"
        f"{custom_result['clusters']:<25}"
    )

    # --------------------------------------------------------
    # Silhouette
    # --------------------------------------------------------

    if (
        kmeans_result[
            "silhouette"
        ]
        is not None
    ):

        k_silhouette = (
            f"{kmeans_result['silhouette']:.4f}"
        )

    else:

        k_silhouette = "—"

    if (
        custom_result[
            "silhouette"
        ]
        is not None
    ):

        custom_silhouette = (
            f"{custom_result['silhouette']:.4f}"
        )

    else:

        custom_silhouette = "—"

    print(
        f"{'Silhouette Score':<30}"
        f"{k_silhouette:<22}"
        f"{custom_silhouette:<25}"
    )

    print(
        f"{'Оптимальное K':<30}"
        f"{kmeans_result['parameter']:<22}"
        f"{'-':<25}"
    )

    print(
        f"{'Оптимальный R, км':<30}"
        f"{'-':<22}"
        f"{custom_result['parameter']:<25}"
    )

    print(
        f"{'Balance':<30}"
        f"{'-':<22}"
        f"{custom_result['balance']:<25.4f}"
    )

    print(
        f"{'Время вычислений, с':<30}"
        f"{kmeans_result['execution_time']:<22.4f}"
        f"{custom_result['execution_time']:<25.4f}"
    )

    # ========================================================
    # СОХРАНЕНИЕ СРАВНЕНИЯ
    # ========================================================

    comparison = pd.DataFrame({

        "method": [
            "K-means",
            "Adaptive radius clustering"
        ],

        "clusters": [
            kmeans_result[
                "clusters"
            ],
            custom_result[
                "clusters"
            ]
        ],

        "parameter_name": [
            "K",
            "R, km"
        ],

        "parameter": [
            kmeans_result[
                "parameter"
            ],
            custom_result[
                "parameter"
            ]
        ],

        "silhouette": [
            kmeans_result[
                "silhouette"
            ],
            custom_result[
                "silhouette"
            ]
        ],

        "balance": [
            np.nan,
            custom_result[
                "balance"
            ]
        ],

        "execution_time_seconds": [
            kmeans_result[
                "execution_time"
            ],
            custom_result[
                "execution_time"
            ]
        ]
    })

    os.makedirs(
        RESULTS_FOLDER,
        exist_ok=True
    )

    comparison_file = os.path.join(
        RESULTS_FOLDER,
        "comparison.csv"
    )

    comparison.to_csv(
        comparison_file,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "\nСравнение сохранено:"
    )

    print(
        comparison_file
    )


# ============================================================
# ГЛАВНАЯ ПРОГРАММА
# ============================================================

def main():

    print(
        "\n" + "=" * 70
    )

    print(
        "КЛАСТЕРИЗАЦИЯ ГОРОДОВ РОССИИ"
    )

    print(
        "Сравнение стандартного "
        "и собственного методов"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # ЗАГРУЗКА
    # ========================================================

    original_df = load_data()

    if original_df is None:

        return

    os.makedirs(
        RESULTS_FOLDER,
        exist_ok=True
    )

    current_df = (
        original_df.copy()
    )

    # ========================================================
    # МЕНЮ
    # ========================================================

    while True:

        print(
            "\n" + "=" * 70
        )

        print(
            "ГЛАВНОЕ МЕНЮ"
        )

        print(
            "=" * 70
        )

        print(
            f"\nТекущий набор данных: "
            f"{len(current_df)} городов\n"
        )

        print(
            "1. Показать информацию о данных"
        )

        print(
            "2. Настроить фильтрацию"
        )

        print(
            "3. Сбросить фильтры"
        )

        print(
            "4. Запустить K-means + метод локтя"
        )

        print(
            "5. Запустить собственный метод"
        )

        print(
            "6. Запустить оба метода и сравнить"
        )

        print(
            "0. Выход"
        )

        choice = input(
            "\nВыберите действие: "
        ).strip()

        # ====================================================
        # ИНФОРМАЦИЯ
        # ====================================================

        if choice == "1":

            show_data_info(
                current_df
            )

        # ====================================================
        # ФИЛЬТРАЦИЯ
        # ====================================================

        elif choice == "2":

            current_df = filter_data(
                original_df
            )

        # ====================================================
        # СБРОС
        # ====================================================

        elif choice == "3":

            current_df = (
                original_df.copy()
            )

            print(
                "\nФильтры сброшены."
            )

        # ====================================================
        # K-MEANS
        # ====================================================

        elif choice == "4":

            run_kmeans(
                current_df
            )

        # ====================================================
        # СОБСТВЕННЫЙ МЕТОД
        # ====================================================

        elif choice == "5":

            run_custom(
                current_df
            )

        # ====================================================
        # СРАВНЕНИЕ
        # ====================================================

        elif choice == "6":

            print(
                "\nЗапуск K-means..."
            )

            kmeans_result = run_kmeans(
                current_df
            )

            print(
                "\nЗапуск собственного метода..."
            )

            custom_result = run_custom(
                current_df
            )

            compare_methods(
                kmeans_result,
                custom_result
            )

        # ====================================================
        # ВЫХОД
        # ====================================================

        elif choice == "0":

            print(
                "\nРабота программы завершена."
            )

            break

        else:

            print(
                "\nНеизвестная команда."
            )


# ============================================================
# ЗАПУСК
# ============================================================

if __name__ == "__main__":

    main()
