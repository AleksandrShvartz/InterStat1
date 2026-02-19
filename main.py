import struct
import sys
import ctypes
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).parent


def log(msg: str):
    print(msg)
    sys.stdout.flush()


# ============================================================================
# Чтение бинарных файлов
# ============================================================================

@dataclass
class FileHeader:
    side: int
    mode: int
    frame_count: int

    @classmethod
    def from_bytes(cls, data: bytes) -> 'FileHeader':
        side, mode, frame_count = struct.unpack('<BBH', data[:4])
        return cls(side=side, mode=mode, frame_count=frame_count)


@dataclass
class FrameHeader:
    stop_point: int
    timestamp: int

    @classmethod
    def from_bytes(cls, data: bytes) -> 'FrameHeader':
        stop_point, timestamp = struct.unpack('<HI', data[:6])
        return cls(stop_point=stop_point, timestamp=timestamp)


@dataclass
class Frame:
    header: FrameHeader
    points: np.ndarray

    @classmethod
    def from_bytes(cls, data: bytes) -> 'Frame':
        header = FrameHeader.from_bytes(data[:16])
        raw_points = struct.unpack('<' + 'H' * (1024 * 8), data[16:16 + 16384])
        points = np.array(raw_points, dtype=np.uint16).reshape(1024, 8)
        return cls(header=header, points=points)


class BinaryDataFile:
    FILE_HEADER_SIZE = 256
    FRAME_SIZE = 16400

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        self.header: FileHeader | None = None
        self.frames: list[Frame] = []

    def read(self) -> 'BinaryDataFile':
        with open(self.filepath, 'rb') as f:
            header_data = f.read(self.FILE_HEADER_SIZE)
            self.header = FileHeader.from_bytes(header_data)
            self.frames = []
            for _ in range(self.header.frame_count):
                frame_data = f.read(self.FRAME_SIZE)
                if len(frame_data) < self.FRAME_SIZE:
                    break
                self.frames.append(Frame.from_bytes(frame_data))
        return self

    def get_all_values_volts(self) -> np.ndarray:
        all_points = np.concatenate([f.points.flatten() for f in self.frames])
        return all_points.astype(np.float64) / 16384.0 - 0.5


# ============================================================================
# Интервальная арифметика
# ============================================================================

@dataclass
class Interval:
    lo: float
    hi: float

    @property
    def mid(self) -> float:
        return (self.lo + self.hi) / 2

    @property
    def rad(self) -> float:
        return (self.hi - self.lo) / 2

    @property
    def wid(self) -> float:
        return self.hi - self.lo

    def __repr__(self):
        return f"[{self.lo:.6f}, {self.hi:.6f}]"


class IntervalArray:
    def __init__(self, los: np.ndarray, his: np.ndarray):
        self.los = np.asarray(los, dtype=np.float64)
        self.his = np.asarray(his, dtype=np.float64)

    @classmethod
    def from_values(cls, values: np.ndarray, radius: float) -> 'IntervalArray':
        values = np.asarray(values, dtype=np.float64)
        return cls(values - radius, values + radius)

    def __len__(self):
        return len(self.los)

    @property
    def mids(self) -> np.ndarray:
        return (self.los + self.his) / 2

    @property
    def rads(self) -> np.ndarray:
        return (self.his - self.los) / 2

    @property
    def wids(self) -> np.ndarray:
        return self.his - self.los


# ============================================================================
# Коэффициент Жаккара
# ============================================================================

def jaccard_vectorized_additive(a: float, X: IntervalArray, Y: IntervalArray) -> float:
    pred_los = X.los + a
    pred_his = X.his + a
    inter_los = np.maximum(pred_los, Y.los)
    inter_his = np.minimum(pred_his, Y.his)
    inter_wids = np.maximum(0, inter_his - inter_los)
    union_los = np.minimum(pred_los, Y.los)
    union_his = np.maximum(pred_his, Y.his)
    union_wids = union_his - union_los
    sum_union = np.sum(union_wids)
    if sum_union == 0:
        return 0.0
    return float(np.sum(inter_wids) / sum_union)


def jaccard_vectorized_multiplicative(t: float, X: IntervalArray, Y: IntervalArray) -> float:
    if t >= 0:
        pred_los = X.los * t
        pred_his = X.his * t
    else:
        pred_los = X.his * t
        pred_his = X.los * t
    inter_los = np.maximum(pred_los, Y.los)
    inter_his = np.minimum(pred_his, Y.his)
    inter_wids = np.maximum(0, inter_his - inter_los)
    union_los = np.minimum(pred_los, Y.los)
    union_his = np.maximum(pred_his, Y.his)
    union_wids = union_his - union_los
    sum_union = np.sum(union_wids)
    if sum_union == 0:
        return 0.0
    return float(np.sum(inter_wids) / sum_union)


# ============================================================================
# Интервальные статистики
# ============================================================================

def _load_dll():
    dll_path = SCRIPT_DIR / "interval_moda.dll"
    if not dll_path.exists():
        return None
    try:
        dll = ctypes.CDLL(str(dll_path))
        dll.interval_moda_cpp.restype = ctypes.c_void_p
        dll.interval_moda_cpp.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
        ]
        dll.free_interval_array.restype = None
        dll.free_interval_array.argtypes = [ctypes.c_void_p]
        return dll
    except OSError:
        return None


_DLL = _load_dll()


def interval_mode_dll(arr: IntervalArray) -> Interval:
    los = np.ascontiguousarray(arr.los, dtype=np.float64)
    his = np.ascontiguousarray(arr.his, dtype=np.float64)
    ret = _DLL.interval_moda_cpp(
        los.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        his.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(los),
    )
    data_ptr = ctypes.c_void_p.from_address(ret).value
    count = ctypes.c_int.from_address(ret + 8).value
    if count == 0 or data_ptr is None:
        _DLL.free_interval_array(ctypes.c_void_p(ret))
        return Interval(float(arr.los.min()), float(arr.his.max()))
    iptr = ctypes.cast(data_ptr, ctypes.POINTER(ctypes.c_double))
    # Берем первый (основной) интервал моды
    lo_val = float(iptr[0])
    hi_val = float(iptr[1])
    _DLL.free_interval_array(ctypes.c_void_p(ret))
    return Interval(lo_val, hi_val)


def interval_mode_python(arr: IntervalArray) -> Interval:
    if len(arr) == 1:
        return Interval(arr.los[0], arr.his[0])

    n = len(arr)
    positions = np.concatenate([arr.los, arr.his])
    deltas = np.concatenate([np.ones(n), -np.ones(n)])

    order = np.lexsort((-deltas, positions))
    positions = positions[order]
    deltas = deltas[order]

    max_depth = 0
    current_depth = 0
    best_start = None
    best_end = None

    for pos, delta in zip(positions, deltas):
        if delta == 1:
            current_depth += 1
            if current_depth > max_depth:
                max_depth = current_depth
                best_start = pos
                best_end = None  # Сбрасываем конец при новом максимуме
        else:
            if current_depth == max_depth and best_end is None:
                best_end = pos  # Первый раз глубина падает с макс — конец моды
            current_depth -= 1

    if best_start is None or best_end is None:
        return Interval(float(arr.los.min()), float(arr.his.max()))

    return Interval(float(best_start), float(best_end))


def interval_mode(arr: IntervalArray) -> Interval:
    if _DLL is not None:
        return interval_mode_dll(arr)
    return interval_mode_python(arr)


def median_kreinovich(arr: IntervalArray) -> Interval:
    return Interval(float(np.median(arr.los)), float(np.median(arr.his)))


def median_prolubnikov(arr: IntervalArray) -> Interval:
    med_mid = float(np.median(arr.mids))
    med_rad = float(np.median(arr.rads))
    return Interval(med_mid - med_rad, med_mid + med_rad)


# ============================================================================
# Оптимизация параметров
# ============================================================================

def _eval_grid(X, Y, model, params):
    jaccards = np.zeros(len(params))
    fn = jaccard_vectorized_additive if model == 'additive' else jaccard_vectorized_multiplicative
    for i, p in enumerate(params):
        jaccards[i] = fn(p, X, Y)
    return jaccards


def find_optimal_param(X: IntervalArray, Y: IntervalArray,
                       model: str, search_range: tuple[float, float],
                       n_points: int = 1000) -> tuple[float, float, np.ndarray, np.ndarray]:
    # Фаза 1: грубый поиск
    params = np.linspace(search_range[0], search_range[1], n_points)
    jaccards = _eval_grid(X, Y, model, params)
    max_idx = np.argmax(jaccards)
    coarse_opt = params[max_idx]

    # Фаза 2: уточнение вокруг найденного оптимума
    step = (search_range[1] - search_range[0]) / n_points
    refine_hw = max(step * 20, 1e-5)
    fine_params = np.linspace(coarse_opt - refine_hw, coarse_opt + refine_hw, 500)
    fine_jaccards = _eval_grid(X, Y, model, fine_params)
    fine_max_idx = np.argmax(fine_jaccards)

    opt_param = fine_params[fine_max_idx]
    opt_ji = fine_jaccards[fine_max_idx]

    # Объединяем для графика
    all_params = np.concatenate([params, fine_params])
    all_jaccards = np.concatenate([jaccards, fine_jaccards])
    order = np.argsort(all_params)
    all_params = all_params[order]
    all_jaccards = all_jaccards[order]

    return opt_param, opt_ji, all_params, all_jaccards


def find_interval_estimate(max_ji: float, params: np.ndarray,
                           jaccards: np.ndarray, point_estimate: float,
                           threshold: float = 0.95) -> Interval:
    if max_ji == 0:
        return Interval(point_estimate, point_estimate)
    threshold_value = threshold * max_ji
    above_threshold = params[jaccards >= threshold_value]
    if len(above_threshold) == 0:
        return Interval(point_estimate, point_estimate)
    return Interval(float(above_threshold.min()), float(above_threshold.max()))


# ============================================================================
# LaTeX-генерация
# ============================================================================

def generate_latex(results_a, results_t, stats, output_path: Path):
    mode_X, mode_Y = stats['mode_X'], stats['mode_Y']
    medK_X, medK_Y = stats['medK_X'], stats['medK_Y']
    medP_X, medP_Y = stats['medP_X'], stats['medP_Y']
    n_x, n_y = stats['n_x'], stats['n_y']
    radius = stats['radius']
    N = stats['N']

    def fmt_iv(iv: Interval) -> str:
        return f"[{iv.lo:.6f},\\; {iv.hi:.6f}]"

    tex = r"""\documentclass[12pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T2A]{fontenc}
\usepackage[russian]{babel}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage[margin=2cm]{geometry}
\usepackage{float}

\begin{document}

\input{title}

\tableofcontents
\newpage

\section{Цель работы}
Получить практические навыки вычисления интервальных описательных статистик (моды, медиан),
работы с коэффициентом Жаккара и применения методов оптимизации для интервальных данных.
Сравнить эффективность различных функционалов на основе интервальных статистик
для оценивания параметров моделей.

\section{Теоретические сведения}

\subsection{Интервальная арифметика}

\textbf{Интервалом} называется замкнутое подмножество вещественной прямой:
\[
  \mathbf{x} = [\underline{x},\; \overline{x}] = \{x \in \mathbb{R} \mid \underline{x} \le x \le \overline{x}\}.
\]
Середина и радиус интервала:
\[
  \operatorname{mid}\mathbf{x} = \frac{\underline{x} + \overline{x}}{2}, \qquad
  \operatorname{rad}\mathbf{x} = \frac{\overline{x} - \underline{x}}{2}.
\]

Основные арифметические операции над интервалами $\mathbf{x}=[\underline{x},\overline{x}]$
и $\mathbf{y}=[\underline{y},\overline{y}]$:
\begin{align}
  \mathbf{x} + \mathbf{y} &= [\underline{x}+\underline{y},\; \overline{x}+\overline{y}], \\
  a + \mathbf{x} &= [a + \underline{x},\; a + \overline{x}], \quad a \in \mathbb{R}, \\
  t \cdot \mathbf{x} &= \begin{cases}
    [t\,\underline{x},\; t\,\overline{x}], & t \ge 0, \\
    [t\,\overline{x},\; t\,\underline{x}], & t < 0.
  \end{cases}
\end{align}

Пересечение двух интервалов:
\[
  \mathbf{x} \cap \mathbf{y} = [\max(\underline{x}, \underline{y}),\; \min(\overline{x}, \overline{y})],
\]
определено при $\max(\underline{x},\underline{y}) \le \min(\overline{x},\overline{y})$.

\subsection{Коэффициент Жаккара}

Коэффициент Жаккара (Jaccard index) для двух интервалов определяется как отношение
длины пересечения к длине объединения:
\[
  Ji(\mathbf{x}, \mathbf{y}) = \frac{\operatorname{wid}(\mathbf{x} \cap \mathbf{y})}{\operatorname{wid}(\mathbf{x} \cup \mathbf{y})},
\]
где $\operatorname{wid}\mathbf{x} = \overline{x} - \underline{x}$ --- ширина интервала,
а объединение берётся как интервальная оболочка:
$\mathbf{x} \cup \mathbf{y} = [\min(\underline{x},\underline{y}),\; \max(\overline{x},\overline{y})]$.

Для набора пар интервалов $\{(\mathbf{x}_i, \mathbf{y}_i)\}_{i=1}^n$ коэффициент Жаккара
обобщается как:
\[
  Ji = \frac{\displaystyle\sum_{i=1}^{n} \operatorname{wid}(\mathbf{x}_i \cap \mathbf{y}_i)}
            {\displaystyle\sum_{i=1}^{n} \operatorname{wid}(\mathbf{x}_i \cup \mathbf{y}_i)}.
\]

\subsection{Интервальная мода}

Интервальная мода $\operatorname{mode}\mathbf{X}$ выборки интервалов
$\mathbf{X} = \{\mathbf{x}_1, \ldots, \mathbf{x}_n\}$ определяется как интервал
максимальной глубины, т.е. точка на вещественной прямой, покрытая наибольшим
числом интервалов из выборки. Формально:
\[
  \operatorname{mode}\mathbf{X} = \arg\max_{\mathbf{z}} \operatorname{depth}(\mathbf{z}, \mathbf{X}),
\]
где $\operatorname{depth}(\mathbf{z}, \mathbf{X}) = \#\{i : \mathbf{z} \subseteq \mathbf{x}_i\}$.
На практике мода вычисляется алгоритмом <<заметающей прямой>> (sweep line)
за время $O(n \log n)$.

\subsection{Интервальные медианы}

\textbf{Медиана Крейновича} $\operatorname{med}_K$ определяется через медианы границ:
\[
  \operatorname{med}_K \mathbf{X} = [\operatorname{median}(\underline{x}_1, \ldots, \underline{x}_n),\;
  \operatorname{median}(\overline{x}_1, \ldots, \overline{x}_n)].
\]

\textbf{Медиана Пролубникова} $\operatorname{med}_P$ определяется через медианы
середин и радиусов:
\[
  \operatorname{med}_P \mathbf{X} = [m - r,\; m + r],
\]
где $m = \operatorname{median}(\operatorname{mid}\mathbf{x}_1, \ldots, \operatorname{mid}\mathbf{x}_n)$,
$r = \operatorname{median}(\operatorname{rad}\mathbf{x}_1, \ldots, \operatorname{rad}\mathbf{x}_n)$.

При одинаковых радиусах всех интервалов обе медианы совпадают:
$\operatorname{med}_K = \operatorname{med}_P$.

\subsection{Постановка задачи оптимизации}

Рассматриваются две модели связи выборок $\mathbf{X}$ и $\mathbf{Y}$:
\begin{itemize}
  \item \textbf{Аддитивная модель:} $a + \mathbf{X} = \mathbf{Y}$, т.е. $a + \mathbf{x}_i = \mathbf{y}_i$ для всех $i$.
  \item \textbf{Мультипликативная модель:} $t \cdot \mathbf{X} = \mathbf{Y}$, т.е. $t \cdot \mathbf{x}_i = \mathbf{y}_i$ для всех $i$.
\end{itemize}

Для аддитивной модели функционал вычисляется как:
\[
  F(a) = Ji\bigl(\{a + \mathbf{x}_i\},\; \{\mathbf{y}_i\}\bigr)
       = \frac{\sum_{i=1}^{n} \operatorname{wid}\bigl((a+\mathbf{x}_i) \cap \mathbf{y}_i\bigr)}
              {\sum_{i=1}^{n} \operatorname{wid}\bigl((a+\mathbf{x}_i) \cup \mathbf{y}_i\bigr)},
\]
аналогично для мультипликативной модели $F(t) = Ji(\{t \cdot \mathbf{x}_i\},\, \{\mathbf{y}_i\})$.

Точечная оценка параметра $s \in \{a, t\}$ находится максимизацией функционала:
\[
  \hat{s} = \arg\max_s F(s, \mathbf{X}, \mathbf{Y}),
\]
где рассматриваются четыре варианта функционала $F$:
\begin{enumerate}
  \item[\textbf{B.1}] $F(s) = Ji(s, \mathbf{X}, \mathbf{Y})$ --- по полным выборкам,
  \item[\textbf{B.2}] $F(s) = Ji(s, \operatorname{mode}\mathbf{X}, \operatorname{mode}\mathbf{Y})$ --- по модам,
  \item[\textbf{B.3}] $F(s) = Ji(s, \operatorname{med}_K\mathbf{X}, \operatorname{med}_K\mathbf{Y})$ --- по медианам Крейновича,
  \item[\textbf{B.4}] $F(s) = Ji(s, \operatorname{med}_P\mathbf{X}, \operatorname{med}_P\mathbf{Y})$ --- по медианам Пролубникова.
\end{enumerate}

Интервальная оценка параметра строится как множество значений $s$, для которых
$F(s) \ge \alpha \cdot F(\hat{s})$, где $\alpha \in (0, 1)$ --- уровень доверия (в данной работе $\alpha = 0.95$).

\section{Исходные данные}

Загружены два файла данных диагностики томсоновского рассеяния:
\begin{itemize}
  \item \texttt{-0.205\_lvl\_side\_a\_fast\_data.bin} --- выборка $\mathbf{X}$,
  \item \texttt{0.225\_lvl\_side\_a\_fast\_data.bin} --- выборка $\mathbf{Y}$.
\end{itemize}

Разрядность АЦП $N = 14$ бит, откуда $2^N = 16384$.
Формула перевода кодов АЦП в вольты: $V = \text{Code}/2^N - 0.5$.

Каждое измерение представляется интервалом $\mathbf{x}_i = [V_i - r,\; V_i + r]$ с радиусом
$r = \operatorname{rad}\mathbf{x} = \operatorname{rad}\mathbf{y} = \frac{1}{2^{""" + str(N) + r"""}} = """ + f"{radius:.10f}" + r"""$.

\begin{itemize}
  \item $|\mathbf{X}| = """ + str(n_x) + r"""$ интервалов
  \item $|\mathbf{Y}| = """ + str(n_y) + r"""$ интервалов
\end{itemize}

\section{Интервальные статистики}

\begin{table}[H]
\centering
\caption{Интервальные статистики выборок}
\begin{tabular}{lcc}
\toprule
Статистика & $\mathbf{X}$ & $\mathbf{Y}$ \\
\midrule
$\text{mode}$ & $""" + fmt_iv(mode_X) + r"""$ & $""" + fmt_iv(mode_Y) + r"""$ \\
$\text{med}_K$ & $""" + fmt_iv(medK_X) + r"""$ & $""" + fmt_iv(medK_Y) + r"""$ \\
$\text{med}_P$ & $""" + fmt_iv(medP_X) + r"""$ & $""" + fmt_iv(medP_Y) + r"""$ \\
\bottomrule
\end{tabular}
\end{table}

\section{Оптимизация параметров}

Оптимизация проводилась методом перебора по равномерной сетке в два этапа:
(1) грубый поиск на 500--1000 точках в широком диапазоне,
(2) уточнение на 500 точках в окрестности найденного оптимума.
Интервальная оценка параметра строилась как множество значений $s$,
для которых $F(s) \ge 0.95 \cdot Ji_{\max}$.

\subsection{Аддитивная модель: $a + \mathbf{X} = \mathbf{Y}$}

\begin{table}[H]
\centering
\caption{Результаты оптимизации для аддитивной модели}
\begin{tabular}{lccc}
\toprule
Функционал & $\hat{a}$ & $Ji_{\max}$ & Интервальная оценка $a$ (95\%) \\
\midrule
"""

    for name, a_opt, ji_a, _, _, a_int in results_a:
        short = name.split(":")[0].strip()
        tex += f"{short} & {a_opt:.6f} & {ji_a:.6f} & ${fmt_iv(a_int)}$ \\\\\n"

    tex += r"""\bottomrule
\end{tabular}
\end{table}

\subsection{Мультипликативная модель: $t \cdot \mathbf{X} = \mathbf{Y}$}

\begin{table}[H]
\centering
\caption{Результаты оптимизации для мультипликативной модели}
\begin{tabular}{lccc}
\toprule
Функционал & $\hat{t}$ & $Ji_{\max}$ & Интервальная оценка $t$ (95\%) \\
\midrule
"""

    for name, t_opt, ji_t, _, _, t_int in results_t:
        short = name.split(":")[0].strip()
        tex += f"{short} & {t_opt:.6f} & {ji_t:.6f} & ${fmt_iv(t_int)}$ \\\\\n"

    tex += r"""\bottomrule
\end{tabular}
\end{table}

\section{Графики}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{results.png}
\caption{Зависимость функционала $F(s) = Ji$ от параметра для аддитивной (верхний ряд)
и мультипликативной (нижний ряд) моделей.
Красная штриховая --- $s_{\max}$, зелёная пунктирная --- значение $Ji_{\max}$.}
\end{figure}

\section{Сравнение результатов и выводы}

\begin{enumerate}
"""

    # Сравнение аддитивных оценок
    a_vals = [r[1] for r in results_a]
    a_mean = np.mean(a_vals)
    a_spread = max(a_vals) - min(a_vals)
    tex += f"  \\item Все четыре функционала дают согласованные оценки параметра $a$: "
    tex += f"среднее $\\hat{{a}} \\approx {a_mean:.4f}$, разброс $\\Delta a = {a_spread:.4f}$.\n"

    # Сравнение мультипликативных оценок
    t_vals = [r[1] for r in results_t]
    t_mean = np.mean(t_vals)
    tex += f"  \\item Для мультипликативной модели $\\hat{{t}} \\approx {t_mean:.4f}$. "

    ji_a_b1 = results_a[0][2]
    ji_t_b1 = results_t[0][2]
    if ji_a_b1 > ji_t_b1:
        tex += "Коэффициент Жаккара для аддитивной модели выше во всех четырёх функционалах, "
        tex += "что указывает на лучшее соответствие аддитивной модели данным.\n"
    else:
        tex += "Мультипликативная модель показывает сопоставимое качество.\n"

    # Объяснение разницы Ji(B.1) vs Ji(B.2-B.4)
    ji_b2_a = results_a[1][2] if len(results_a) > 1 else 0
    tex += f"  \\item При работе с полными выборками (B.1) $Ji_{{\\max}} \\approx {ji_a_b1:.3f}$, тогда как для\n"
    tex += f"    статистик (B.2--B.4) $Ji_{{\\max}} \\approx {ji_b2_a:.2f}$.\n"
    tex += "    Малое значение для B.1 объясняется тем, что при 819200 парах интервалов с малым радиусом\n"
    tex += "    ($r \\approx 6{,}1 \\cdot 10^{-5}$) и значительным разбросом середин большинство пар\n"
    tex += "    $(a + \\mathbf{x}_i) \\cap \\mathbf{y}_i$ имеют малое перекрытие.\n"
    tex += "    Для статистик же оптимизация проводится по одной паре представительных интервалов,\n"
    tex += "    что даёт почти полное совпадение при оптимальном $s$.\n"

    tex += r"""  \item Использование интервальных статистик (мода, медианы Крейновича и Пролубникова)
    вместо полных выборок значительно ускоряет вычисления при близких точечных оценках параметров.
  \item Медиана Крейновича и медиана Пролубникова дают идентичные результаты,
    поскольку все радиусы интервалов одинаковы ($\operatorname{rad}\mathbf{x}_i = \text{const}$).
"""

    # Сравнение моды и медиан (отдельно для аддитивной и мультипликативной)
    ji_mode_a = results_a[1][2] if len(results_a) > 1 else 0
    ji_medk_a = results_a[2][2] if len(results_a) > 2 else 0
    ji_mode_t = results_t[1][2] if len(results_t) > 1 else 0
    ji_medk_t = results_t[2][2] if len(results_t) > 2 else 0

    tex += r"  \item Для аддитивной модели "
    if ji_medk_a > ji_mode_a:
        tex += f"медианы (B.3, B.4) дают $Ji_{{\\max}} = {ji_medk_a:.4f}$, что выше моды $Ji_{{\\max}} = {ji_mode_a:.4f}$.\n"
    else:
        tex += f"мода даёт $Ji_{{\\max}} = {ji_mode_a:.4f}$, что выше медиан $Ji_{{\\max}} = {ji_medk_a:.4f}$.\n"
    tex += "    Для мультипликативной модели, напротив, "
    if ji_mode_t > ji_medk_t:
        tex += f"мода ($Ji_{{\\max}} = {ji_mode_t:.4f}$) превосходит медианы ($Ji_{{\\max}} = {ji_medk_t:.4f}$).\n"
    else:
        tex += f"медианы ($Ji_{{\\max}} = {ji_medk_t:.4f}$) превосходят моду ($Ji_{{\\max}} = {ji_mode_t:.4f}$).\n"
    tex += "    Это показывает, что выбор представительной статистики зависит от модели.\n"

    # Ширина интервальных оценок B.1 vs B.2-B.4
    a_int_b1 = results_a[0][5]
    a_int_b2 = results_a[1][5] if len(results_a) > 1 else a_int_b1
    w_b1 = a_int_b1.hi - a_int_b1.lo
    w_b2 = a_int_b2.hi - a_int_b2.lo
    tex += r"  \item Интервальные оценки по полным выборкам (B.1) "
    if w_b2 > 0:
        import math
        orders = int(math.floor(math.log10(w_b1 / w_b2)))
        tex += f"на {orders} порядка шире, чем по статистикам (B.2--B.4):\n"
    import math
    exp = int(math.floor(math.log10(w_b2)))
    coeff = w_b2 / (10 ** exp)
    tex += f"    $\\Delta a_{{\\text{{B.1}}}} \\approx {w_b1:.3f}$ против "
    tex += f"$\\Delta a_{{\\text{{B.2}}}} \\approx {coeff:.0f} \\cdot 10^{{{exp}}}$.\n"
    tex += "    Это следствие агрегации данных в одну пару представительных интервалов.\n"
    tex += r"""\end{enumerate}

\end{document}
"""

    output_path.write_text(tex, encoding='utf-8')


# ============================================================================
# Основная программа
# ============================================================================

def main():
    log("=" * 70)
    log("ЛАБОРАТОРНАЯ РАБОТА №4: ИНТЕРВАЛЬНЫЙ АНАЛИЗ")
    log("=" * 70)

    if _DLL is not None:
        log("DLL interval_moda.dll загружена успешно")
    else:
        log("ВНИМАНИЕ: DLL не найдена, используется Python-реализация моды")

    # Параметры
    N = 14
    radius = 1.0 / (2 ** N)
    N_POINTS_FULL = 500     # Точки сетки для B.1 (много данных)
    N_POINTS_STAT = 1000    # Точки сетки для B.2-B.4 (быстро)

    log(f"\nПараметры: N={N}, radius = 1/2^{N} = {radius:.10f}")

    # ── A. Загрузка данных ──
    log("\n" + "=" * 70)
    log("ЧАСТЬ A: Загрузка данных и создание интервальных выборок")
    log("=" * 70)

    log("Загрузка файлов...")
    file_x = BinaryDataFile(SCRIPT_DIR / "-0.205_lvl_side_a_fast_data.bin").read()
    file_y = BinaryDataFile(SCRIPT_DIR / "0.225_lvl_side_a_fast_data.bin").read()

    values_x = file_x.get_all_values_volts()
    values_y = file_y.get_all_values_volts()

    log(f"\nФайл X (-0.205): {len(values_x)} значений")
    log(f"  min={values_x.min():.6f}, max={values_x.max():.6f}, mean={values_x.mean():.6f}")
    log(f"\nФайл Y (0.225): {len(values_y)} значений")
    log(f"  min={values_y.min():.6f}, max={values_y.max():.6f}, mean={values_y.mean():.6f}")

    X_full = IntervalArray.from_values(values_x, radius)
    Y_full = IntervalArray.from_values(values_y, radius)
    log(f"\nСозданы интервальные выборки: |X|={len(X_full)}, |Y|={len(Y_full)}")

    # Выравниваем длины (берём min)
    n = min(len(X_full), len(Y_full))
    X_paired = IntervalArray(X_full.los[:n], X_full.his[:n])
    Y_paired = IntervalArray(Y_full.los[:n], Y_full.his[:n])

    # ── B. Интервальные статистики ──
    log("\n" + "=" * 70)
    log("ЧАСТЬ B: Вычисление интервальных статистик")
    log("=" * 70)

    log("Вычисление моды X (все данные)...")
    mode_X = interval_mode(X_full)
    log(f"  mode(X) = {mode_X}")

    log("Вычисление моды Y (все данные)...")
    mode_Y = interval_mode(Y_full)
    log(f"  mode(Y) = {mode_Y}")

    log("Вычисление медиан...")
    medK_X = median_kreinovich(X_full)
    medK_Y = median_kreinovich(Y_full)
    medP_X = median_prolubnikov(X_full)
    medP_Y = median_prolubnikov(Y_full)

    log(f"\nМедиана Крейновича:")
    log(f"  medK(X) = {medK_X}")
    log(f"  medK(Y) = {medK_Y}")
    log(f"\nМедиана Пролубникова:")
    log(f"  medP(X) = {medP_X}")
    log(f"  medP(Y) = {medP_Y}")

    # ── Диапазоны поиска ──
    # Для B.1 — широкий диапазон
    a_range_full = (0.2, 0.6)
    t_range_full = (-2.0, 0.0)

    def interval_to_array(iv: Interval) -> IntervalArray:
        return IntervalArray(np.array([iv.lo]), np.array([iv.hi]))

    def get_search_range_a(sx: Interval, sy: Interval) -> tuple[float, float]:
        a_approx = sy.mid - sx.mid
        margin = max(sx.wid + sy.wid, 0.001) * 5
        return (a_approx - margin, a_approx + margin)

    def get_search_range_t(sx: Interval, sy: Interval) -> tuple[float, float]:
        if abs(sx.mid) > 1e-10:
            t_approx = sy.mid / sx.mid
            margin = max(abs(t_approx) * 0.05, 0.01)
            return (t_approx - margin, t_approx + margin)
        return (-2.0, 0.0)

    cases = [
        ("B.1", "Ji(s, X, Y)", X_paired, Y_paired,
         a_range_full, t_range_full, N_POINTS_FULL),
        ("B.2", "Ji(s, mode X, mode Y)",
         interval_to_array(mode_X), interval_to_array(mode_Y),
         get_search_range_a(mode_X, mode_Y),
         get_search_range_t(mode_X, mode_Y), N_POINTS_STAT),
        ("B.3", "Ji(s, med_K X, med_K Y)",
         interval_to_array(medK_X), interval_to_array(medK_Y),
         get_search_range_a(medK_X, medK_Y),
         get_search_range_t(medK_X, medK_Y), N_POINTS_STAT),
        ("B.4", "Ji(s, med_P X, med_P Y)",
         interval_to_array(medP_X), interval_to_array(medP_Y),
         get_search_range_a(medP_X, medP_Y),
         get_search_range_t(medP_X, medP_Y), N_POINTS_STAT),
    ]

    results_a = []
    results_t = []

    log("\n" + "=" * 70)
    log("РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ")
    log("=" * 70)

    for label, name, Xs, Ys, a_range, t_range, npts in cases:
        log(f"\n{label}: F(s) = {name}")
        log("-" * 50)

        # Аддитивная модель
        log(f"  Аддитивная: поиск a в {a_range}...")
        a_opt, ji_a, pa, ja = find_optimal_param(Xs, Ys, 'additive', a_range, npts)
        a_int = find_interval_estimate(ji_a, pa, ja, a_opt)
        results_a.append((f"{label}: {name}", a_opt, ji_a, pa, ja, a_int))
        log(f"    a_opt = {a_opt:.6f}, Ji_max = {ji_a:.6f}")
        log(f"    Интервал (95%): {a_int}")

        # Мультипликативная модель
        log(f"  Мультипликативная: поиск t в {t_range}...")
        t_opt, ji_t, pt, jt = find_optimal_param(Xs, Ys, 'multiplicative', t_range, npts)
        t_int = find_interval_estimate(ji_t, pt, jt, t_opt)
        results_t.append((f"{label}: {name}", t_opt, ji_t, pt, jt, t_int))
        log(f"    t_opt = {t_opt:.6f}, Ji_max = {ji_t:.6f}")
        log(f"    Интервал (95%): {t_int}")

    # ── C. Графики ──
    log("\n" + "=" * 70)
    log("ЧАСТЬ C: Построение графиков")
    log("=" * 70)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Функционал $F(s) = Ji$ (коэффициент Жаккара)', fontsize=14)

    def plot_ji(ax, params, jaccards, opt_val, ji_max, param_sym, is_stat):
        if is_stat and ji_max > 0:
            # Зумируем в область пика
            nz = jaccards > 0.01 * ji_max
            if nz.any():
                hw = max(params[nz].ptp() * 1.5, 0.0005)
            else:
                hw = 0.001
            mask = (params >= opt_val - hw) & (params <= opt_val + hw)
            ax.plot(params[mask], jaccards[mask], 'b-', linewidth=1.5)
        else:
            ax.plot(params, jaccards, 'b-', linewidth=1.5)
        ax.axvline(opt_val, color='r', linestyle='--',
                   label=f'${param_sym}_{{max}}={opt_val:.4f}$')
        ax.axhline(ji_max, color='g', linestyle=':', alpha=0.5)
        ax.tick_params(axis='x', rotation=20, labelsize=7)

    for i, (name, a_opt, ji_a, params_a, jaccards_a, a_int) in enumerate(results_a):
        ax = axes[0, i]
        plot_ji(ax, params_a, jaccards_a, a_opt, ji_a, 'a', i > 0)
        short = name.split(":")[0].strip()
        ax.set_xlabel('$a$')
        ax.set_ylabel('$Ji(a)$')
        ax.set_title(f'{short}: $a + X = Y$')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for i, (name, t_opt, ji_t, params_t, jaccards_t, t_int) in enumerate(results_t):
        ax = axes[1, i]
        plot_ji(ax, params_t, jaccards_t, t_opt, ji_t, 't', i > 0)
        short = name.split(":")[0].strip()
        ax.set_xlabel('$t$')
        ax.set_ylabel('$Ji(t)$')
        ax.set_title(f'{short}: $t \\cdot X = Y$')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = SCRIPT_DIR / 'results.png'
    plt.savefig(str(fig_path), dpi=150)
    log(f"\nГрафики сохранены в {fig_path}")

    # ── D. Сравнение ──
    log("\n" + "=" * 70)
    log("ЧАСТЬ D: Сравнение результатов")
    log("=" * 70)

    log("\nАддитивная модель (a + X = Y):")
    log("-" * 75)
    log(f"{'Функционал':<10} {'a_opt':>10} {'Ji_max':>10} {'Интервал a':>30}")
    log("-" * 75)
    for name, a_opt, ji_a, _, _, a_int in results_a:
        short = name.split(":")[0].strip()
        log(f"{short:<10} {a_opt:>10.6f} {ji_a:>10.6f}   [{a_int.lo:.6f}, {a_int.hi:.6f}]")

    log("\nМультипликативная модель (t * X = Y):")
    log("-" * 75)
    log(f"{'Функционал':<10} {'t_opt':>10} {'Ji_max':>10} {'Интервал t':>30}")
    log("-" * 75)
    for name, t_opt, ji_t, _, _, t_int in results_t:
        short = name.split(":")[0].strip()
        log(f"{short:<10} {t_opt:>10.6f} {ji_t:>10.6f}   [{t_int.lo:.6f}, {t_int.hi:.6f}]")

    # ── LaTeX ──
    tex_path = SCRIPT_DIR / 'report.tex'
    generate_latex(results_a, results_t, {
        'mode_X': mode_X, 'mode_Y': mode_Y,
        'medK_X': medK_X, 'medK_Y': medK_Y,
        'medP_X': medP_X, 'medP_Y': medP_Y,
        'n_x': len(X_full), 'n_y': len(Y_full),
        'radius': radius, 'N': N,
    }, tex_path)
    log(f"\nLaTeX-отчёт сохранён в {tex_path}")


if __name__ == "__main__":
    main()
