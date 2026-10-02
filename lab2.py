"""Лабораторная № 2: последовательный и генетический поиск максимума.
Запуск: python lab2.py --sizes 100 1000 10000 --trials 100
Зависимость: matplotlib (pip install matplotlib). Графики: lab2_results.
--no-show: без окон; --patience 0: без досрочной остановки.
Отклонение: |точный максимум − результат ГА| / |точный максимум| · 100%.
Для нулевого максимума процент не определён.
"""

import argparse
import random
from pathlib import Path
from statistics import mean, stdev
from math import sqrt
from time import perf_counter


def linear_search(array):
    if not array:
        raise ValueError("Массив не должен быть пустым")

    best_index = 0
    for i in range(1, len(array)):
        if array[i] > array[best_index]:
            best_index = i
    return best_index


def relative_error_pct(exact_value, ga_value):
    """Отклонение по значениям; при нулевом максимуме процент не определён."""
    if exact_value == 0:
        return None

    return 100 * abs(exact_value - ga_value) / abs(exact_value)


def format_error(value):
    return "не опр." if value is None else f"{value:.4f}"


def genetic_search(
    array,
    rng,
    population_size=50,
    generations=100,
    mutation_rate=0.2,
    tournament_size=3,
    patience=20,
    min_improvement=0.001,
):
    """Вернуть индекс, число оценок и число выполненных поколений.

    Значимым считается накопленное улучшение больше
    min_improvement * max(1, abs(reference_best)).
    reference_best обновляется только при значимом улучшении.
    Это критерий стабилизации, а не доказательство нахождения максимума.
    """
    if not array:
        raise ValueError("Массив не должен быть пустым")
    if population_size < 2 or generations < 0 or tournament_size < 1:
        raise ValueError("Недопустимые параметры генетического алгоритма")
    if not 0 <= mutation_rate <= 1:
        raise ValueError("Вероятность мутации должна быть от 0 до 1")
    if patience < 0 or min_improvement < 0:
        raise ValueError("Порог улучшения и терпение не должны быть отрицательными")

    n = len(array)
    bits = max(1, (n - 1).bit_length())
    population = [rng.randrange(n) for _ in range(population_size)]
    fitness = [array[index] for index in population]
    evaluations = population_size
    best_index = population[max(range(population_size), key=fitness.__getitem__)]

    reference_best = array[best_index]
    stagnant_generations = 0
    generations_done = 0

    def select_parent():
        candidates = [rng.randrange(population_size) for _ in range(tournament_size)]
        return population[max(candidates, key=fitness.__getitem__)]

    for generation in range(1, generations + 1):
        # Элитизм: переносим лучшее решение в следующее поколение.
        new_population = [best_index]
        while len(new_population) < population_size:
            parent1, parent2 = (select_parent(), select_parent())
            # Одноточечное скрещивание двоичных индексов.
            if bits > 1:
                mask = (1 << rng.randrange(1, bits)) - 1
                child = parent1 & ~mask | parent2 & mask
            else:
                child = parent1

            if child >= n:
                child = parent1

            if rng.random() < mutation_rate:
                child = rng.randrange(n)
            new_population.append(child)

        population = new_population
        fitness = [array[index] for index in population]
        evaluations += population_size
        best_index = population[max(range(population_size), key=fitness.__getitem__)]

        generations_done = generation
        improvement = array[best_index] - reference_best
        threshold = min_improvement * max(1, abs(reference_best))
        # Остановка по накопленному улучшению, без знания точного максимума.
        if improvement > threshold:
            reference_best = array[best_index]
            stagnant_generations = 0
        else:
            stagnant_generations += 1

        if patience > 0 and stagnant_generations >= patience:
            break

    return (best_index, evaluations, generations_done)


def compare(sizes, trials, population_size, generations, mutation_rate, seed, patience=20, min_improvement=0.001):
    data_rng = random.Random(seed)
    results = []

    print("Время — среднее по запускам, в миллисекундах.")
    print("Оба метода в каждом запуске получают один и тот же массив.")
    print("Генерация массива и проверка результата не входят в измерения.\n")
    print(
        f"{'N':>8} {'Поиск, мс':>12} {'ГА, мс':>12} "
        f"{'Сред. откл.%':>13} {'Макс. откл.%':>13} {'Точных ГА':>12} {'Успех, %':>10} {'Оценок ГА':>12} {'Поколений':>10}"
    )

    for n in sizes:
        linear_times, genetic_times, evaluation_counts, generation_counts, errors = ([], [], [], [], [])
        successes = undefined_errors = 0

        for trial in range(trials):
            array = list(range(n))
            data_rng.shuffle(array)
            ga_rng = random.Random(seed + n * 100003 + trial)

            for method in ("linear", "genetic") if trial % 2 == 0 else ("genetic", "linear"):
                start = perf_counter()
                if method == "linear":
                    exact_index = linear_search(array)
                    linear_times.append(perf_counter() - start)
                else:
                    ga_index, evaluations, generations_done = genetic_search(
                        array,
                        ga_rng,
                        population_size,
                        generations,
                        mutation_rate,
                        patience=patience,
                        min_improvement=min_improvement,
                    )
                    genetic_times.append(perf_counter() - start)

            error = relative_error_pct(array[exact_index], array[ga_index])
            if error is None:
                undefined_errors += 1
            else:
                errors.append(error)
            successes += array[ga_index] == array[exact_index]
            evaluation_counts.append(evaluations)
            generation_counts.append(generations_done)

        mean_error = mean(errors) if errors else None
        max_error = max(errors) if errors else None
        print(
            f"{n:8d} {mean(linear_times) * 1000:12.4f} "
            f"{mean(genetic_times) * 1000:12.4f} "
            f"{format_error(mean_error):>13} {format_error(max_error):>13} "
            f"{str(successes) + '/' + str(trials):>12} "
            f"{100 * successes / trials:10.1f} {mean(evaluation_counts):12.1f} "
            f"{mean(generation_counts):10.1f}"
        )

        results.append(
            {
                "size": n,
                "linear_ms": mean(linear_times) * 1000,
                "genetic_ms": mean(genetic_times) * 1000,
                "mean_error_pct": mean_error,
                "max_error_pct": max_error,
                "undefined_error_count": undefined_errors,
                "evaluations_mean": mean(evaluation_counts),
                "success_pct": 100 * successes / trials,
                "linear_sd": stdev(linear_times) * 1000 if trials > 1 else 0,
                "genetic_sd": stdev(genetic_times) * 1000 if trials > 1 else 0,
                "successes": successes,
                "generations_mean": mean(generation_counts),
            }
        )

    print("Отклонение вычисляется по значению элемента; среднее — по всем запускам, включая точные ответы.")
    print("При нулевом максимуме процент не определён и не включается в среднее.\n")

    return results


def plot_comparison(
    results,
    trials,
    population_size,
    generations,
    mutation_rate,
    output_dir,
    show=True,
    patience=20,
    min_improvement=0.001,
):
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = sorted(results, key=lambda row: row["size"])
    sizes = [row["size"] for row in results]
    colors = ("#2563EB", "#EA580C")
    subtitle = (
        f"{trials} запусков на размер; популяция {population_size}; поколений ≤ {generations}; мутация {mutation_rate:g}\n"
        + (
            f"Остановка: {patience} поколений без улучшения > {100 * min_improvement:g}% от max(1, |опорный результат|)"
            if patience
            else "Досрочная остановка отключена"
        )
    )

    def set_size_axis(ax):
        ax.set_xscale("log")
        ax.set_xticks(sizes, [f"{n:,}".replace(",", " ") for n in sizes], rotation=45, ha="right", fontsize=9)
        ax.set_xlabel("Количество элементов массива, N (логарифмическая шкала)")
        ax.grid(True, which="major", alpha=0.25)

    def save_plot(fig, ax, title, ylabel, filename, note=None):
        set_size_axis(ax)
        ax.set_ylabel(ylabel)
        ax.set_title(title + "\n" + subtitle, fontsize=12, pad=16)
        ax.legend(loc="lower left" if filename == "accuracy_comparison.png" else "best", fontsize=10)
        if note:
            fig.supxlabel(note, fontsize=9)
        path = output_dir / filename
        fig.savefig(path, dpi=200)
        print(f"График: {path.resolve()}")

    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    for key, color, label, marker in (
        ("linear", colors[0], "Последовательный поиск", "o"),
        ("genetic", colors[1], "Генетический алгоритм", "s"),
    ):
        values = [row[key + "_ms"] for row in results]
        deviations = [row[key + "_sd"] for row in results]
        ax.plot(sizes, values, marker=marker, color=color, label=label, linewidth=2)
        ax.fill_between(
            sizes,
            [max(v - d, v * 0.01) for v, d in zip(values, deviations)],
            [v + d for v, d in zip(values, deviations)],
            color=color,
            alpha=0.15,
        )
    ax.set_yscale("log")
    save_plot(
        fig,
        ax,
        "Сравнение времени выполнения",
        "Среднее время, мс (логарифмическая шкала)",
        "time_comparison.png",
        "Линия — среднее; область — ±1 стандартное отклонение (снизу ограничена для логарифмической шкалы)",
    )

    lower, upper = ([], [])
    z = 1.96
    for row in results:
        proportion = row["successes"] / trials
        denominator = 1 + z * z / trials
        center = (proportion + z * z / (2 * trials)) / denominator
        radius = z * sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)) / denominator
        lower.append(100 * max(0, center - radius))
        upper.append(100 * min(1, center + radius))

    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    ax.plot(sizes, [100] * len(sizes), "o-", color=colors[0], label="Последовательный поиск", linewidth=2)
    ax.plot(
        sizes,
        [row["success_pct"] for row in results],
        "s-",
        color=colors[1],
        label="Генетический алгоритм",
        linewidth=2,
    )
    ax.fill_between(sizes, lower, upper, color=colors[1], alpha=0.15, label="95% доверительный интервал ГА (Уилсон)")
    ax.set_ylim(-3, 110)
    ax.set_yticks(range(0, 101, 20))
    save_plot(
        fig,
        ax,
        "Сравнение точности поиска",
        "Запуски с точным максимумом, %",
        "accuracy_comparison.png",
        "Точки — измеренные результаты; отрезки соединяют соседние размеры массива",
    )

    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    ax.plot(
        sizes,
        [row["generations_mean"] for row in results],
        "o-",
        color=colors[1],
        linewidth=2,
        label="Среднее число поколений ГА",
    )
    ax.axhline(generations, color="#64748B", linestyle="--", label="Лимит поколений")
    ax.set_ylim(0, max(1, generations) * 1.1)
    save_plot(fig, ax, "Досрочная остановка генетического алгоритма", "Число поколений", "generations_comparison.png")

    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    for key, label, marker in (
        ("mean_error_pct", "Среднее отклонение ГА", "o"),
        ("max_error_pct", "Наибольшее отклонение ГА", "s"),
    ):
        values = [float("nan") if row[key] is None else row[key] for row in results]
        ax.plot(sizes, values, marker=marker, label=label, linewidth=2)
    ax.set_ylim(bottom=0)
    save_plot(
        fig,
        ax,
        "Отклонение результата генетического алгоритма",
        "Отклонение от точного максимума, %",
        "deviation_comparison.png",
        "Среднее по всем запускам; при нулевом максимуме процент не определён",
    )

    if show:
        plt.show()
    plt.close("all")


def plot_array(array, ga_index, exact_index, output_dir, filename):
    """Показать значения массива и связь индекса с решением ГА."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n = len(array)
    if n > 10:
        ga_start = max(0, min(ga_index - 2, n - 5))
        exact_start = max(0, min(exact_index - 2, n - 5))
        indices = list(range(ga_start, ga_start + 5)) + list(range(exact_start, exact_start + 5))
        positions = list(range(5)) + [5.7 + i for i in range(5)]
    else:
        indices = list(range(n))
        positions = list(range(n))

    fig = plt.figure(figsize=(12, 8), layout="constrained")
    grid = fig.add_gridspec(3, 1, height_ratios=[1.25, 0.8, 2.4])
    cells = fig.add_subplot(grid[0])
    cells.set_xlim(-0.7, positions[-1] + 0.7)
    cells.set_ylim(-0.35, 1.85 if n > 10 else 1.5)
    cells.axis("off")
    cells.set_title(
        (
            f"Массив из {n} элементов: индекс — номер ячейки, значение — число внутри"
            if n <= 10
            else f"Массив из {n} элементов: результат ГА и точный максимум"
        ),
        fontsize=14,
        pad=16,
    )
    if n > 10:
        cells.text(2, 1.55, "Фрагмент вокруг результата ГА", ha="center", color="#C2410C", fontsize=11, weight="bold")
        cells.text(
            7.7, 1.55, "Фрагмент вокруг точного максимума", ha="center", color="#15803D", fontsize=11, weight="bold"
        )
        cells.axvline(4.85, ymin=0.2, ymax=0.8, color="#CBD5E1", linestyle="--")

    for position, index in zip(positions, indices):
        selected = index == ga_index
        exact = index == exact_index
        cells.text(position, 1.12, f"i = {index}", ha="center", fontsize=11, color="#64748B")
        cells.add_patch(
            Rectangle(
                (position - 0.43, 0),
                0.86,
                0.85,
                facecolor="#FFEDD5" if selected else "#DCFCE7" if exact else "#F1F5F9",
                edgecolor="#16A34A" if exact else "#EA580C" if selected else "#CBD5E1",
                linewidth=3 if selected or exact else 1.5,
            )
        )
        cells.text(
            position,
            0.42,
            str(array[index]),
            ha="center",
            va="center",
            fontsize=17,
            weight="bold" if selected or exact else "normal",
        )

    cells.text(
        positions[-1] / 2,
        -0.25,
        "Оранжевая ячейка — результат ГА; зелёная рамка — точный максимум",
        ha="center",
        fontsize=10,
        color="#475569",
    )

    explanation = fig.add_subplot(grid[1])
    explanation.axis("off")
    explanation.text(
        0.5,
        0.65,
        f"Особь ГА: индекс {ga_index}  →  ячейка A[{ga_index}]  →  значение {array[ga_index]}",
        ha="center",
        va="center",
        fontsize=14,
        weight="bold",
        transform=explanation.transAxes,
    )
    error = relative_error_pct(array[exact_index], array[ga_index])
    explanation.text(
        0.5,
        0.18,
        f"Приспособленность этой особи: {array[ga_index]}.  "
        f"Точный максимум: A[{exact_index}] = {array[exact_index]}.\n"
        f"Отклонение ГА: {format_error(error)}" + (" %" if error is not None else ""),
        ha="center",
        va="center",
        fontsize=11,
        transform=explanation.transAxes,
    )

    values = fig.add_subplot(grid[2])
    if n <= 10:
        values.bar(range(n), array, color=["#EA580C" if i == ga_index else "#94A3B8" for i in range(n)], width=0.6)
        values.set_xticks(range(n))
    else:
        values.scatter(range(n), array, s=9, color="#94A3B8", alpha=0.5, label="Элементы массива", rasterized=True)

    values.scatter(
        [ga_index],
        [array[ga_index]],
        color="#EA580C",
        s=100,
        zorder=4,
        label=f"ГА: индекс {ga_index}, значение {array[ga_index]}",
    )
    values.scatter(
        [exact_index],
        [array[exact_index]],
        s=200,
        facecolors="none",
        edgecolors="#16A34A",
        linewidths=2,
        zorder=5,
        label=f"Точный максимум: индекс {exact_index}, значение {array[exact_index]}",
    )
    values.axhline(0, color="#64748B", linewidth=0.8)
    values.set_xlabel("Индекс элемента (нумерация с нуля)")
    values.set_ylabel("Значение элемента A[i]")
    values.set_title("Весь массив: положение и значение каждого элемента", fontsize=12)
    values.grid(axis="y", alpha=0.2)
    values.set_axisbelow(True)
    values.margins(y=0.22)
    values.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2, fontsize=9)

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    fig.savefig(path, dpi=200)
    print(f"Схема массива: {path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[100, 200, 300, 500, 700, 1000, 2000, 3000, 5000, 7000, 10000, 20000, 30000, 50000, 100000],
    )
    for name, kind, default in (
        ("trials", int, 100),
        ("population", int, 50),
        ("generations", int, 100),
        ("mutation", float, 0.2),
        ("seed", int, 42),
    ):
        parser.add_argument("--" + name, type=kind, default=default)
    parser.add_argument(
        "--patience", type=int, default=20, help="Поколений без заметного улучшения; 0 отключает остановку"
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.001,
        help="Относительный порог накопленного улучшения (0.001 = 0.1%%)",
    )
    parser.add_argument("--output-dir", default="lab2_results")
    parser.add_argument("--no-show", action="store_true", help="Сохранить графики без открытия окон")

    args = parser.parse_args()
    if any((n < 1 for n in args.sizes)) or args.trials < 1:
        parser.error("Размеры массивов и число запусков должны быть положительными")
    if args.population < 2 or args.generations < 0 or (not 0 <= args.mutation <= 1):
        parser.error("Проверьте размер популяции, число поколений и вероятность мутации")
    if args.patience < 0 or args.min_improvement < 0:
        parser.error("Порог улучшения и терпение не должны быть отрицательными")

    sample = [12, -5, 37, 8, 24, 3, 19, 41, 6, 15]
    exact = linear_search(sample)
    found, _, done = genetic_search(
        sample,
        random.Random(args.seed),
        args.population,
        args.generations,
        args.mutation,
        patience=args.patience,
        min_improvement=args.min_improvement,
    )

    print("Пример массива:", sample)
    print(f"Последовательный поиск: максимум {sample[exact]}, индекс {exact}")
    print(f"Генетический алгоритм:  найдено {sample[found]}, индекс {found}")
    print(f"Отклонение ГА, %: {format_error(relative_error_pct(sample[exact], sample[found]))}")
    print(f"ГА выполнил поколений: {done}")
    print("Индексы начинаются с нуля.\n")
    plot_array(sample, found, exact, args.output_dir, "array_example.png")

    demonstration = list(range(1000))
    random.Random(args.seed).shuffle(demonstration)
    demo_found, _, _ = genetic_search(
        demonstration,
        random.Random(args.seed + 1),
        args.population,
        args.generations,
        args.mutation,
        patience=args.patience,
        min_improvement=args.min_improvement,
    )
    demo_exact = linear_search(demonstration)
    print(
        f"Массив из 1000 элементов: ГА = {demonstration[demo_found]}, "
        f"точный максимум = {demonstration[demo_exact]}, "
        f"отклонение = {format_error(relative_error_pct(demonstration[demo_exact], demonstration[demo_found]))} %"
    )
    plot_array(demonstration, demo_found, demo_exact, args.output_dir, "array_overview.png")
    print(
        f"Популяция: {args.population}; поколений: {args.generations}; мутация: {args.mutation}; запусков на размер: {args.trials}\n"
    )

    results = compare(
        args.sizes,
        args.trials,
        args.population,
        args.generations,
        args.mutation,
        args.seed,
        args.patience,
        args.min_improvement,
    )

    plot_comparison(
        results,
        args.trials,
        args.population,
        args.generations,
        args.mutation,
        args.output_dir,
        show=not args.no_show,
        patience=args.patience,
        min_improvement=args.min_improvement,
    )


if __name__ == "__main__":
    main()
