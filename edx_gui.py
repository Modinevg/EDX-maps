"""EDX spectrum-image processing GUI for FEI TIA (.emi / _N.ser) and EDAX Genesis / TEAM (.spd / .lsd / .spc) data.

    python edx_gui.py ["file.emi" | "map.spd"]

Workflow: open file -> background -> element identification (auto / periodic table /
click on the spectrum) -> fit of the spectrum (peak deconvolution, refinement of
calibration and resolution) -> per-pixel maps (MLLS or windows) -> colour composite -> export.
Optionally the spectrum image is replaced by its PCA reconstruction (noise suppression): raw cube ->
preprocessing (dead channels, energy drift, bad pixels) -> decomposition (eigenvalues, loadings, scores).
"""
import csv
import functools
import os
import re
import sys
import traceback
from dataclasses import replace

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
import pyqtgraph as pg      # after PyQt5: pyqtgraph uses the already imported binding, not PySide6/PyQt6

import edx_core as core
from edx_xray_lines import ELEMENTS

pg.setConfigOptions(imageAxisOrder='row-major', antialias=True, background='#121519', foreground='#cfd3d8')

SYMBOLS = ('H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se '
           'Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy '
           'Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am').split()

PALETTE = ['#ff4b4b', '#3cff5a', '#4b8bff', '#ffd23c', '#ff4bff', '#3cf0ff', '#ff9b2d', '#b07bff',
           '#9bff9b', '#ff9bb4', '#2dd2aa', '#d2d25a', '#f0f0f0', '#8cb4ff', '#c86e3c', '#78c8ff']

C_DATA, C_BG, C_MODEL, C_RL = '#d8dbe0', '#5fbf5f', '#ff5050', '#c78bff'

BG_METHODS = ('snip', 'minima', 'poly', 'none')     # order of the items of the background method box

HELP = """<h3>Обработка EDX спектральных изображений (FEI TIA, EDAX Genesis / TEAM)</h3>
<ol>
<li><b>Открыть</b> файл FEI TIA .emi (или любой _N.ser той же серии) или EDAX Genesis / TEAM: карту .spd,
линейный скан .lsd или спектр .spc. Калибровка энергии, ускоряющее напряжение и разрешение детектора берутся
из .spc с тем же именем (.spc рядом с .spd — суммарный спектр карты, открывается сама карта); электронное
изображение (.bmp) и его калибровка (.ipr), если лежат рядом, показываются во вкладке «Обзор».
Слева — изображение области карты (список «Изображение») и обзорное изображение, справа — суммарный спектр.
Галочка <i>«Спектр из ROI»</i> берёт спектр из жёлтой рамки.<br>
<i>Загрузить…</i> (рядом со списком «Изображение», на панели инструментов и у «Подложки» на вкладке «Карты») —
электронное изображение SEM / STEM той же области (TIFF, BMP, PNG, JPEG) для наложения карт. Оно растягивается
на всю область карты; если пропорции другие, предлагается обрезать (информационная панель SEM снизу) или
растянуть. Калибровка из метаданных FEI / Thermo Fisher и Zeiss TIFF или .ipr EDAX: если у карты калибровки нет,
она берётся из изображения; если есть и поле зрения другое — предупреждение.</li>
<li><b>Фон</b>: диапазон фита, метод (SNIP, по минимумам или полином по окнам). Окна можно задать текстом
(«3.0-4.8, 12-15») или выделить на графике. Фон в фите = форма фона × полином Лежандра.
<i>По минимумам</i> — алгоритм скрипта DM «Autobackground SI from minima»: узлы в минимумах
сглаженного спектра, выступающие узлы срезаются за N циклов, фон — ломаная через узлы. Не
поднимается под группами перекрывающихся линий (Nb L + Pt M), но в чистом континууме лежит на
~1σ шума ниже (берутся минимумы шумящего спектра; полином в фите это частично компенсирует).</li>
<li><b>Элементы</b>: <i>Автоопределение</i> (итеративный фит остатка с проверкой всех серий элемента,
escape- и pile-up-пиков), либо вручную — таблица Менделеева (наведение показывает линии на спектре),
щелчок по спектру — список линий вблизи этой энергии (двойной щелчок добавляет элемент).
<i>Из файла</i> — элементы, определённые EDAX Genesis / TEAM при съёмке (список хранится в .spc).
Линия опознаётся, только если она возбуждается с перенапряжением E0/E<sub>края</sub> ≥ 1.5 (у E0 фон
неточен, а слабо возбуждаемая серия была бы много слабее другой серии того же элемента); элементу, который
нельзя подтвердить второй серией (Mo, Ag, Sn L при 15 кВ), нужен двойной порог. В SEM-спектрах с миллионами
отсчётов (больше 10⁵ в канале пика) мелкие неточности модели заметнее шума, поэтому там: пики суммирования
(pile-up: C+O 0.80, O+Na 1.57, Ti+C 4.79 кэВ…) входят в модель (амплитуда ограничена живым временем),
положения и ширины самых сильных линий уточняются (±20 эВ), лёгкие элементы (B…Ca) в конце ещё раз
проверяются на полной модели (N K под линией Ti Ll), элементу без видимой хорошо возбуждаемой второй серии
(Cu L без Cu K) или кандидату на плече сильной линии (< 2 % её отсчётов) нужен тройной порог. Сомнительные
элементы отмечены «возможно» — проверьте их по второй серии.<br>
Тяжёлые элементы: вторая серия кандидата должна нести не меньше 0.3 отсчётов найденной; если её отсчёты
забрал уже найденный сосед с перекрывающимися линиями (Pt M под Au M, а L-линии обоих разрешены), кандидат
проверяется фитом с этим соотношением — его отклоняют, только если данные ему противоречат. Пока L- и
M-серии тяжёлых элементов (выше 1.5 кэВ) несут ≥ 10 % отсчётов в линиях, калибровка и разрешение
уточняются только по K-линиям выше 0.6 кэВ: одна линия Au M, накрывающая дублет Pt + Au, иначе раздувала
ширину пиков. Пик на энергии суммы двух сильных линий (C K + Pt Lα = Au Lα 9.71 кэВ) отдаётся элементу,
только если его другие линии объясняют остаток и в других местах спектра.</li>
<li><b>Фит</b>: все серии (K, L, M) выбранных элементов моделируются гауссианами с разрешением
FWHM(E) детектора, хвостом неполного сбора заряда и escape-пиками Si. Перекрывающиеся линии
разделяются (деконволюция). M-серия Hf…U включает и слабые линии M2N4, M3N1, M2N1… (по 1–2 % Mα, данные
xraylib с поправками по спектру Pt + Au): без них остаток Pt M у 2.7 кэВ доставался Cl, и карта Cl
повторяла Pt. Можно уточнить калибровку энергии и разрешение; при уточнении калибровки
положение серий ниже 0.6 кэВ (Nb Mζ, C K, O K…) подгоняется отдельно — у SDD такие пики сдвинуты
на 10–30 эВ (столбец ΔE). Наклон шкалы уточняется, только если есть две линии выше 0.6 кэВ, разнесённые
больше чем на 1 кэВ (иначе — только сдвиг). В SEM-спектрах с миллионами отсчётов шкала, разрешение и хвост
подбираются по линиям выше 0.6 кэВ: огромные пики C, N, O с неполным сбором заряда иначе сдвигают шкалу
и пики Na, P, Si оказываются в модели на ~10 эВ не там.
Порог регистрации (~0.13 кэВ) срезает низкоэнергетический край пиков Nb M, C K:
он входит в модель (поля «Порог регистрации» на вкладке «Фит») и уточняется вместе с калибровкой; фит по
умолчанию начинается чуть ниже порога. Таблица — чистые
отсчёты серий ± σ. Деконволюция Ричардсона–Люси повышает разрешение спектра для просмотра.<br>
<i>Форма пиков → NLLS</i>: нелинейный МНК — у серий, найденных выше 10σ, подгоняются собственные положение,
ширина и отношения групп линий (Kβ/Kα, Lβ/Lα, Mβ/Mα…; столбцы «ΔE», «Ширина», «Отношения линий») с
априорными ограничениями около значений модели. Убирает систематические остатки (L-линии переходных
металлов, Nb L); найденная форма затем используется в картах MLLS. «По модели детектора» — сбросить её.</li>
<li><b>Карты</b>: MLLS — попиксельный фит теми же формами линий (разделение перекрытий); «окно − фон»
и «окно» — классические карты. Сглаживание/бининг, цвет каждого элемента, яркость, контраст, гамма,
режим смешивания. По умолчанию каждая карта растянута на весь диапазон цветов по своим процентилям —
яркость разных карт не сравнима, слабая линия (особенно после PCA, без шума) выглядит как сильная.
<i>«Общая шкала для всех карт»</i> показывает все карты в одном диапазоне 0 … максимум (чистые отсчёты на
пиксель): яркости можно сравнивать (отсчёты линий, не концентрации).
<i>Подложка</i> композита — электронное изображение (или сумма отсчётов) с заданной долей яркости; композит
строится в разрешении изображения (до 2048 пикс.), карты интерполируются на него.
<i>«Убрать горячие пиксели»</i> — по скрипту DM «Point Blemish»: пиксель плохой, если его лапласиан
(8·центр − 8 соседей) отклоняется от среднего больше чем на N σ («точки и области» — также яркость пикселя
относительно всей карты); плохие пиксели заполняются средним хороших соседей. Отличие от DM: σ лапласиана
оценивается локально (окно 15×15), а не по всей карте, и проход повторяется, пока находятся новые плохие
пиксели. Шум рентгеновских карт растёт с числом отсчётов: с одной σ на карту шум яркой фазы и края слоёв
принимаются за дефекты (в SEM-карте Al — 2000 пикс. вместо 77), а один очень яркий пиксель скрывает более
слабые. Применяется к каждой карте до сглаживания, исходные карты не меняются; число заменённых пикселей
каждой карты — в подписи под композитом.
Экспорт PNG (со шкалой и легендой), TIFF float32, NPZ (исходные карты), CSV, презентация PowerPoint (обзорное
изображение с областью карт, композит, все карты — отдельными рисунками). В MLLS серии ниже 1 кэВ
(Nb M, Fe L…) по умолчанию следуют за основной серией элемента — так они не обмениваются шумом с C K, O K.
Карта слабой линии (C K ~0.1 отсчёта на пиксель) по отдельным пикселям — в основном шум: используйте бининг.
<i>Фон по фазам</i> (по умолчанию): пиксели автоматически делятся на фазы, у каждой своя форма фона — иначе
слабые линии получают ложный контраст между фазами. «Окно − фон − перекрытия» вычитает также хвосты соседних
линий (Nb M в окне C K). Количественное сравнение слабого элемента между областями надёжнее делать фитом
спектров ROI (галочка «Спектр из ROI»).<br>
<i>Проверка карт</i> (MLLS): суммарный спектр каждой фазы фитуется тем же попиксельным фитом с моделью фона карт
и с другими разумными моделями — полином фона на 2 и 4 порядка выше, окно SNIP в 1.5 раза шире. Под слабой
линией (Fe K ~0.02 отсчёта на пиксель) фон данными не определён: его ошибка в несколько процентов уходит в
элемент и распределена как главные фазы. Отдельные пиксели этого не показывают (шум), а после PCA или бининга
такая карта выглядит как настоящее распределение. Карта помечается <i>⚠ фон</i>, если другая модель фона
меняет её контраст между фазами (с точностью до масштаба и сдвига, сверх 3σ) больше чем на 50 %, и
<i>⚠ шум</i>, если различие между фазами меньше 3σ и у карты (данных до PCA) нет пространственной структуры.
Такие карты не показывают распределение элемента и по умолчанию не входят в композит; средние по фазам при
каждой модели фона — в подсказке у элемента в таблице карт, список — под композитом и в презентации.</li>
<li><b>PCA</b> (подавление шума) по схеме: <i>исходный куб</i> (y × x × каналы, отсчёты) → <i>предобработка</i>
→ <i>разложение</i> (собственные значения, нагрузки, счета). «Вычислить PCA» выполняет оба шага.<br>
<i>Предобработка</i> (галочки «Предобработка», отчёт над графиками; данные копируются, только если что-то
изменено). <b>Мёртвые каналы</b>: канал суммарного спектра, отличающийся от среднего двух соседних больше,
чем допускает линия шириной FWHM детектора (застрявший или пропущенный код АЦП, всплеск в одном канале),
во всех пикселях заменяется интерполяцией по соседним каналам; каналы диапазона без единого отсчёта в PCA не
входят. <b>Дрейф энергии</b>: строки развёртки (время съёмки) делятся на 16 блоков, сдвиг спектра каждого блока
относительно суммарного измеряется по самым сильным пикам (у каждого пика своя амплитуда — состав блоков
разный). Спектры сдвигаются построчно, только если разброс сдвигов значим (p &lt; 0.001), больше 0.2 канала и
пики его подтверждают; сдвиги по блокам — во всплывающей подсказке отчёта. Попиксельный сдвиг не измерить
(~100 отсчётов на пиксель); у многокадровых карт EDAX дрейф внутри съёмки усредняется по кадрам.
<b>Плохие пиксели</b>: сумму отсчётов пикселя не объясняет даже второй по яркости (темноте) из 8 соседей
(точный тест равенства пуассоновских интенсивностей, порог в σ) — горячие пиксели в начале строки у TIA,
потерянные пиксели. Линии, края, частицы от 2×2 пикселей имеют похожего соседа и не трогаются. Плохой пиксель
заменяется спектром соседа с медианной суммой (данные остаются отсчётами); где они — «Изображение» →
«Плохие пиксели».<br>
<i>Разложение</i>: SVD без центрирования с пуассоновской нормировкой (компонента 1 — средний спектр).
<i>«Биннинг для разложения»</i> 2×2 или 4×4 (как «Spatial binning» в MSA для DigitalMicrograph, HREM Research):
собственные значения, спектры компонент и граница шума находятся по данным, просуммированным по блокам
пикселей, а карты — проекции спектров исходных пикселей на спектры компонент, в полном разрешении. Граница шума
падает с числом пикселей, а собственное значение протяжённой фазы — нет: слабые протяжённые компоненты выходят
над шумом (на тестовых картах — до двух сигнальных компонент больше). Детали меньше блока выделяются хуже;
собственные значения, доли и r относятся к данным с биннингом.
<i>«Локальный PCA, плитки N px»</i> (по мотивам «Spatial Local wPCA» в MSA): карта режется на плитки N×N с
перекрытием в полплитки, каждая раскладывается отдельно. Фаза, занимающая часть карты (сегрегация на границе,
отдельные частицы), в своей плитке выходит над шумом раньше, чем во всей карте. Модель плитки — выбранная
глобальная модель (k, «Состав») плюс те сигнальные компоненты плитки, которые больше чем наполовину лежат вне
неё; плитки сшиваются плавными весами. В MSA каждая плитка восстанавливается только по своим компонентам — так
теряется слабая фаза, протянутая по всей карте: её видит глобальный PCA, но не отдельная плитка (на 01593top
такая модель сохраняла лишь 32–81 % трёх слабых глобальных компонент). Здесь локальный PCA только добавляет;
где добавлять нечего, данные те же, что без него. Проверка на EDXmap1 с добавленным пятном Zr 16×16 пикселей
(плитки 64 px): при 2 отсчётах на пиксель глобальный PCA сохраняет в пятне 3 % Zr, с локальным — 56 %; при 4 Zr
уже входит в глобальную модель (85 %); при 1 — почти ничего (сигнал ниже шума и в плитке). Плитка должна
содержать хотя бы несколько тысяч пикселей; разложение ~0.3–1 с на плитку. Где добавлены локальные компоненты —
«Изображение» → «Локальный PCA: добавлено»; отдельная плитка может добавить и шумовое направление.
<i>Собственные значения</i> — график слева сверху, <i>нагрузки</i> (loadings, спектры компонент) — слева снизу,
<i>счета</i> (scores, карты компонент) — справа: все сохранённые компоненты (до 48), первые 12 видны сразу,
остальные — прокруткой колесом; Ctrl + колесо — увеличение карты. <i>Экспорт PCA…</i> — собственные значения и нагрузки (CSV),
счета (TIFF float32), всё вместе с отчётом предобработки (NPZ); данные ≈ Σ счета × нагрузки.
Сигнал — компоненты выше границы шума (Марченко–Пастур, оранжевая линия) с пространственно связной картой
(r — корреляция соседних пикселей; у шума r ≈ 0) или намного выше границы. k (последняя сигнальная компонента)
выбирается автоматически, его можно изменить (поле или жёлтая линия на графике). Шумовая компонента может
оказаться между сигнальными (× на графике): с галочкой <i>«без шумовых»</i> такие в модель не входят.
<i>Вручную</i>: двойной щелчок по карте компоненты или щелчок по точке на графике добавляет / убирает её,
номера можно ввести в поле «Состав» («1-3, 5»); компонента 1 (средний спектр) входит всегда; изменение k или
«Авто» возвращает автоматический выбор. Карты счетов: структура — сигнал, «соль с перцем» — шум.
Спектры компонент начиная со второй знакопеременны: это не спектры фаз, а отклонения от среднего спектра
(«больше Ti и меньше Na, O, чем в среднем»), знак произволен; спектр пикселя = сумма компонент с его счетами.
Галочка <i>«Использовать восстановленные данные»</i> подставляет данные, восстановленные по выбранным
компонентам (из предобработанных данных), во все спектры (ROI), фит и карты. Деталь, не попавшая в отдельную
компоненту (редкое включение, слабый элемент с собственным распределением), после PCA повторяет распределение
основных фаз. Поэтому при картах по данным PCA строятся и карты данных до PCA (после предобработки), и для
каждого элемента сравнивается пространственная структура (ковариация соседних пикселей остатка «до − после
PCA»; шум пикселей в неё не
входит): карты, где PCA теряет или добавляет больше 30 % структуры, помечены <i>⚠ PCA</i>, доли — под композитом.
EDAX Genesis / TEAM хранят отсчёты с поправкой на мёртвое время: часть событий записана дважды, дисперсия
≈ 1.4 × отсчёты, а не 1 × (Пуассон; строка «Count dispersion» в описании файла). Она учитывается в границе
шума PCA и в погрешностях фита (порог автоопределения — по-прежнему в пуассоновских σ).</li>
</ol>
<p>Наведение мыши на спектр показывает ближайшие линии; правая кнопка — меню pyqtgraph (масштаб, экспорт).</p>
"""


def table_position(z):
    """(row, col) of element Z in the periodic-table grid (lanthanides/actinides in rows 8-9)."""
    if z == 1:
        return 0, 0
    if z == 2:
        return 0, 17
    if z <= 10:
        return 1, (z - 3) if z <= 4 else (z - 5 + 12)
    if z <= 18:
        return 2, (z - 11) if z <= 12 else (z - 13 + 12)
    if z <= 36:
        return 3, z - 19
    if z <= 54:
        return 4, z - 37
    if z <= 56:
        return 5, z - 55
    if z <= 71:
        return 8, z - 57 + 2
    if z <= 86:
        return 5, z - 72 + 3
    if z <= 88:
        return 6, z - 87
    return 9, z - 89 + 2


def dspin(val, lo, hi, step, dec, suffix='', tip=''):
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setDecimals(dec)
    s.setValue(val)
    if suffix:
        s.setSuffix(suffix)
    if tip:
        s.setToolTip(tip)
    s.setKeyboardTracking(False)
    return s


def ispin(val, lo, hi, tip=''):
    s = QtWidgets.QSpinBox()
    s.setRange(lo, hi)
    s.setValue(val)
    if tip:
        s.setToolTip(tip)
    s.setKeyboardTracking(False)
    return s


def guarded(fn):
    """Slot wrapper: wait cursor + error dialog. Signal arguments are ignored."""
    @functools.wraps(fn)
    def wrapper(self, *args):
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            return fn(self)
        except Exception as e:
            traceback.print_exc()
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(self, 'Ошибка', f'{e}\n\n{traceback.format_exc(limit=4)}')
        finally:
            while QtWidgets.QApplication.overrideCursor() is not None:
                QtWidgets.QApplication.restoreOverrideCursor()
    return wrapper


def nice_length(width):
    """Round scale-bar length ~ width/5 to 1, 2 or 5 x 10^n."""
    target = width / 5.0
    p = 10 ** np.floor(np.log10(target))
    for m in (5, 2, 1):
        if m * p <= target:
            return m * p
    return p


def grid_columns(n, nx, ny, width, height, label_h=24, min_w=0):
    """Number of columns that gives the largest map tiles in a width x height area (columns >= min_w wide)."""
    width, height = max(width, 400), max(height, 400)
    best, best_scale = 1, 0.0
    for c in range(1, max(min(n, width // min_w if min_w else n), 1) + 1):
        rows = int(np.ceil(n / c))
        scale = min(width / c / nx, (height / rows - label_h) / ny)
        if scale > best_scale:
            best, best_scale = c, scale
    return best


class ScrollViewBox(pg.ViewBox):
    """ViewBox inside a scroll area: the wheel scrolls the area, Ctrl + wheel zooms the map."""

    def wheelEvent(self, ev, axis=None):
        if ev.modifiers() & Qt.ControlModifier:
            super().wheelEvent(ev, axis)
        else:
            ev.ignore()


def si_length(m):
    if m >= 1e-6:
        v, u = m * 1e6, 'µm'
    else:
        v, u = m * 1e9, 'nm'
    return f'{v:g} {u}'


def ranges_text(idx):
    """'1–3, 5' for the 0-based indices (0, 1, 2, 4)."""
    idx = sorted(idx)
    parts, start = [], None
    for a, b in zip(idx, idx[1:] + [None]):
        start = a if start is None else start
        if b != a + 1:
            parts.append(f'{start + 1}' if start == a else f'{start + 1}–{a + 1}')
            start = None
    return ', '.join(parts)


def parse_ranges(text):
    """0-based indices from '1-3, 5 7' (1-based numbers and ranges); ValueError(bad part) if not understood."""
    out = set()
    for part in re.split(r'[,;\s]+', re.sub(r'\s*[-–—]\s*', '-', text.strip())):
        if not part:
            continue
        m = re.fullmatch(r'(\d+)(?:-(\d+))?', part)
        a, b = (int(m.group(1)), int(m.group(2) or m.group(1))) if m else (0, 0)
        if a < 1 or b < a:
            raise ValueError(part)
        out.update(range(a - 1, b))
    return out


def ppt_text(slide, x, y, w, h, lines, size=14, center=False):
    """Text box at (x, y) of size (w, h), inches. lines = [[(text, '#rrggbb', bold[, size]), ...], ...]."""
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
    from pptx.util import Inches, Pt
    tf = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame
    tf.word_wrap = True
    for i, runs in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if center:
            p.alignment = PP_ALIGN.CENTER
        for run in runs:
            r = p.add_run()
            r.text = run[0]
            r.font.color.rgb = RGBColor.from_string(run[1].lstrip('#'))
            r.font.bold = run[2]
            r.font.size = Pt(run[3] if len(run) > 3 else size)


# ----------------------------------------------------------------------------------------------
# Small widgets
# ----------------------------------------------------------------------------------------------
class FlowLayout(QtWidgets.QLayout):
    """Widgets in a row that wraps to the next line when the width runs out, so a row of
    controls does not set a large minimum width of the window (splitters stay movable)."""

    def __init__(self, spacing=8):
        super().__init__()
        self._items = []
        self._spacing = spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientations(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._arrange(QtCore.QRect(0, 0, width, 0), move=False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._arrange(rect, move=True)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QtCore.QSize()
        for it in self._items:
            size = size.expandedTo(it.minimumSize())
        m = self.contentsMargins()
        return size + QtCore.QSize(m.left() + m.right(), m.top() + m.bottom())

    def _arrange(self, rect, move):
        m = self.contentsMargins()
        r = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, line_h = r.x(), r.y(), 0
        for it in self._items:
            hint = it.sizeHint()
            if line_h and x + hint.width() > r.right() + 1:
                x, y, line_h = r.x(), y + line_h + self._spacing, 0
            if move:
                it.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), hint))
            x += hint.width() + self._spacing
            line_h = max(line_h, hint.height())
        return y + line_h - rect.y() + m.bottom()


def ratios_text(det, key):
    """'Lβ×1.16  Lγ×0.90' - line-group intensity factors of a family refined by NLLS."""
    return '  '.join(f'{core.line_label(g)}×{v:.2f}' for g, v in sorted(det.ratios.get(key, {}).items()))


def hbox(*widgets):
    """Widgets kept together on one line (an item of a FlowLayout)."""
    w = QtWidgets.QWidget()
    lay = QtWidgets.QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    for x in widgets:
        lay.addWidget(x)
    return w


class ElementButton(QtWidgets.QToolButton):
    hovered = QtCore.pyqtSignal(str)
    left = QtCore.pyqtSignal(str)

    def __init__(self, sym, parent=None):
        super().__init__(parent)
        self.sym = sym
        self.setText(sym)
        self.setCheckable(True)
        self.setFixedSize(27, 22)
        self.set_color(None)

    def set_color(self, color):
        checked = f'background:{color}; color:#000; font-weight:bold;' if color else 'background:#888;'
        self.setStyleSheet('QToolButton{font-size:8pt; padding:0; border:1px solid #3a3f46; background:#22272e;}'
                           'QToolButton:hover{border:1px solid #ffd23c;}'
                           'QToolButton:disabled{color:#555; background:#1a1d22;}'
                           f'QToolButton:checked{{{checked}}}')

    def enterEvent(self, e):
        self.hovered.emit(self.sym)
        super().enterEvent(e)

    def leaveEvent(self, e):
        self.left.emit(self.sym)
        super().leaveEvent(e)


class PeriodicTable(QtWidgets.QWidget):
    toggled = QtCore.pyqtSignal(str, bool)
    hovered = QtCore.pyqtSignal(str)
    left = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        grid = QtWidgets.QGridLayout(self)
        grid.setSpacing(1)
        grid.setContentsMargins(0, 0, 0, 0)
        self.buttons = {}
        for z, sym in enumerate(SYMBOLS, start=1):
            r, c = table_position(z)
            b = ElementButton(sym)
            info = ELEMENTS.get(sym)
            if info is None or z < 4:
                b.setEnabled(False)
            else:
                lines = []
                for fam, lst in info[3].items():
                    if info[0] < core.FAMILY_MIN_Z[fam]:
                        continue
                    main = max(lst, key=lambda t: t[2])
                    lines.append(f'{fam}: {core.line_label(main[0])} {main[1]:.3f} кэВ')
                b.setToolTip(f'<b>{sym}</b> {info[1]}, Z={z}<br>' + '<br>'.join(lines))
            b.toggled.connect(lambda on, s=sym: self.toggled.emit(s, on))
            b.hovered.connect(self.hovered)
            b.left.connect(self.left)
            grid.addWidget(b, r if r < 8 else r, c)
            self.buttons[sym] = b
        grid.setRowMinimumHeight(7, 6)

    def set_checked(self, sym, on, color=None):
        b = self.buttons.get(sym)
        if b is None:
            return
        b.blockSignals(True)
        b.setChecked(on)
        b.set_color(color)
        b.blockSignals(False)


class ColorButton(QtWidgets.QPushButton):
    colorChanged = QtCore.pyqtSignal(object)

    def __init__(self, color, parent=None):
        super().__init__(parent)
        self.setFixedSize(40, 20)
        self.set_color(color)
        self.clicked.connect(self.pick)

    def set_color(self, color):
        self._c = QtGui.QColor(color)
        self.setStyleSheet(f'background:{self._c.name()}; border:1px solid #888;')

    def color(self):
        return QtGui.QColor(self._c)

    def pick(self):
        c = QtWidgets.QColorDialog.getColor(self._c, self, 'Цвет элемента')
        if c.isValid():
            self.set_color(c)
            self.colorChanged.emit(c)


# ----------------------------------------------------------------------------------------------
# Main window
# ----------------------------------------------------------------------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, path=None):
        super().__init__()
        self.setWindowTitle('EDX: обработка спектральных изображений (FEI TIA, EDAX)')
        self.resize(1700, 980)
        self.si = None
        self.si_raw = None            # loaded data; self.si is the PCA reconstruction when it is in use
        self.si_pre = None            # preprocessed data, the input of the PCA (core.preprocess)
        self.pre = None               # core.Preprocessing report of si_pre
        self.pca = None
        self.pca_local = None         # core.LocalPCA: the data in use are rebuilt tile by tile (None: by self.pca)
        self.pca_manual = None        # components of the PCA model chosen by hand (None: first k)
        self.images = []              # electron images of the mapped area: dict(name, image[, path])
        self._blem_cache = {}         # (maps id, element, family, settings) -> map without hot pixels, count
        self.maps_raw = None          # maps of the PCA input (si_pre) while the maps are of PCA data (check)
        self._raw_cache = None        # (key, maps of the PCA input)
        self._check_cache = {}        # (element, family) -> share of the structure lost / added by PCA
        self.map_checks = {}          # family key -> core.MapCheck: its distribution between the phases vs the background
        self._struct_cache = {}       # (element, family) -> spatial structure of the map before PCA (core.map_structure)
        self._phases = None           # (key, phase labels) cache for the phase backgrounds of the maps
        self.map_phases = None
        self.help_dlg = None          # help window, made on the first «Справка»
        self.det = core.Detector()
        self.elements = []            # selected element symbols
        self.colors = {}              # element -> QColor
        self.fam_enabled = {}         # (element, family) -> bool
        self.id_info = {}             # element -> identification details
        self.unidentified = []
        self.fit = None
        self.maps = None
        self._common = None           # (maps, key, upper limit) cache of the display range shared by all maps
        self.map_rows = {}            # element -> dict(widgets)
        self.spec = None
        self.bg = None
        self.bg_sum = None
        self.noise = None
        self.rl = None
        self.marker_items = []
        self.preview_items = []
        self._log_mode = False
        self.comp_items = []
        self.map_views = []
        self._build_ui()
        if path:
            QtCore.QTimer.singleShot(0, lambda: self.open_file(path))

    # ------------------------------------------------------------------ UI construction
    def _build_ui(self):
        tb = self.addToolBar('Файл')
        tb.setMovable(False)
        a = tb.addAction('Открыть…', lambda: self.open_file())
        a.setShortcut('Ctrl+O')
        a = tb.addAction('Изображение SEM / STEM…', self.load_image_dialog)
        a.setToolTip('Электронное изображение области карты (TIFF, BMP, PNG, JPEG) — для наложения карт')
        tb.addSeparator()
        tb.addAction('Экспорт спектра (CSV)', self.export_spectrum)
        tb.addAction('Экспорт таблицы фита (CSV)', self.export_fit_table)
        tb.addAction('Сохранить композит (PNG)', self.save_composite)
        tb.addAction('Сохранить все карты…', self.save_all_maps)
        tb.addAction('Экспорт в PowerPoint…', self.export_pptx)
        tb.addSeparator()
        tb.addAction('Справка', self.show_help)

        self.main_tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.main_tabs)
        self.main_tabs.addTab(self._build_spectrum_page(), 'Спектр')
        self.main_tabs.addTab(self._build_maps_page(), 'Карты')
        self.main_tabs.addTab(self._build_pca_page(), 'PCA')

        dock = QtWidgets.QDockWidget('Обработка', self)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable)
        self.ctrl_tabs = QtWidgets.QTabWidget()
        for title, w in (('Фон', self._build_bg_tab()), ('Элементы', self._build_elements_tab()),
                         ('Фит', self._build_fit_tab()), ('Карты', self._build_maps_tab())):
            sa = QtWidgets.QScrollArea()
            sa.setWidgetResizable(True)
            sa.setWidget(w)
            self.ctrl_tabs.addTab(sa, title)
        dock.setWidget(self.ctrl_tabs)
        dock.setMinimumWidth(540)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        self.status_info = QtWidgets.QLabel('')
        self.statusBar().addPermanentWidget(self.status_info)
        self.statusBar().showMessage('Откройте файл .emi / .ser (FEI TIA) или .spd / .spc (EDAX) — Ctrl+O')

    def _build_spectrum_page(self):
        split = QtWidgets.QSplitter(Qt.Horizontal)

        # --- left: SI image with ROI, survey image
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(2, 2, 2, 2)
        self.img_tabs = QtWidgets.QTabWidget()
        si_page = QtWidgets.QWidget()
        sv = QtWidgets.QVBoxLayout(si_page)
        sv.setContentsMargins(0, 0, 0, 0)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel('Изображение:'))
        self.cb_img = QtWidgets.QComboBox()
        self.cb_img.currentIndexChanged.connect(self.update_si_image)
        row.addWidget(self.cb_img, 1)
        b = QtWidgets.QPushButton('Загрузить…')
        b.setToolTip('Электронное изображение SEM / STEM области карты (TIFF, BMP, PNG, JPEG): для наложения карт '
                     '(подложка композита на вкладке «Карты»)')
        b.clicked.connect(self.load_image_dialog)
        row.addWidget(b)
        sv.addLayout(row)
        self.si_glw = pg.GraphicsLayoutWidget()
        self.si_vb = self.si_glw.addViewBox(lockAspect=True, invertY=True)
        self.si_img = pg.ImageItem()
        self.si_vb.addItem(self.si_img)
        # pixels replaced by the preprocessing of the PCA (image «Плохие пиксели»)
        self.bad_marks = pg.ScatterPlotItem(symbol='s', size=11, pen=pg.mkPen('#ff4b4b', width=2),
                                            brush=pg.mkBrush(0, 0, 0, 0))
        self.bad_marks.setVisible(False)
        self.si_vb.addItem(self.bad_marks)
        self.roi = pg.RectROI([0, 0], [10, 10], pen=pg.mkPen('#ffd23c', width=2), hoverPen=pg.mkPen('#fff', width=2))
        self.roi.translateSnap = True
        self.roi.scaleSnap = True
        self.roi.sigRegionChangeFinished.connect(self.on_roi_changed)
        self.roi.setVisible(False)
        self.si_vb.addItem(self.roi)
        sv.addWidget(self.si_glw, 1)
        row = QtWidgets.QHBoxLayout()
        self.cb_roi = QtWidgets.QCheckBox('Спектр из ROI')
        self.cb_roi.toggled.connect(self.on_roi_toggled)
        row.addWidget(self.cb_roi)
        b = QtWidgets.QPushButton('ROI = вся карта')
        b.clicked.connect(self.reset_roi)
        row.addWidget(b)
        row.addStretch(1)
        sv.addLayout(row)
        self.lbl_roi = QtWidgets.QLabel('')
        sv.addWidget(self.lbl_roi)
        self.img_tabs.addTab(si_page, 'Карта SI')

        self.sv_glw = pg.GraphicsLayoutWidget()
        self.sv_vb = self.sv_glw.addViewBox(lockAspect=True, invertY=True)
        self.sv_img = pg.ImageItem()
        self.sv_vb.addItem(self.sv_img)
        self.sv_rect = QtWidgets.QGraphicsRectItem()
        self.sv_rect.setPen(pg.mkPen('#ffd23c', width=2, cosmetic=True))
        self.sv_vb.addItem(self.sv_rect)
        self.img_tabs.addTab(self.sv_glw, 'Обзор')
        lv.addWidget(self.img_tabs, 1)
        self.lbl_meta = QtWidgets.QLabel('')
        self.lbl_meta.setWordWrap(True)
        self.lbl_meta.setStyleSheet('color:#9aa3ad; font-size:8pt;')
        lv.addWidget(self.lbl_meta)
        split.addWidget(left)

        # --- right: spectrum + residual
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        rv.setContentsMargins(2, 2, 2, 2)
        row = FlowLayout()
        self.cb_log = QtWidgets.QCheckBox('Лог. шкала')
        self.cb_net = QtWidgets.QCheckBox('Вычесть фон')
        self.cb_comp = QtWidgets.QCheckBox('Компоненты')
        self.cb_comp.setChecked(True)
        self.cb_markers = QtWidgets.QCheckBox('Маркеры линий')
        self.cb_markers.setChecked(True)
        self.cb_show_rl = QtWidgets.QCheckBox('RL-деконволюция')
        for cb in (self.cb_log, self.cb_net, self.cb_comp, self.cb_markers, self.cb_show_rl):
            cb.toggled.connect(self.redraw_spectrum)
            row.addWidget(cb)
        b = QtWidgets.QPushButton('Весь спектр')
        b.clicked.connect(lambda: self.pi.autoRange())
        row.addWidget(b)
        rv.addLayout(row)

        self.pw = pg.PlotWidget()
        self.pi = self.pw.getPlotItem()
        self.pi.setLabel('bottom', 'Энергия (кэВ)')
        self.pi.setLabel('left', 'Отсчёты')
        self.pi.showGrid(x=True, y=True, alpha=0.15)
        self.pi.vb.setDefaultPadding(0.06)   # room for line labels above the peaks
        self.legend = self.pi.addLegend(offset=(-10, 10))
        self.c_data = self.pi.plot(pen=pg.mkPen(C_DATA, width=1), name='спектр')
        self.c_bg = self.pi.plot(pen=pg.mkPen(C_BG, width=1.2, style=Qt.DashLine), name='фон')
        self.c_model = self.pi.plot(pen=pg.mkPen(C_MODEL, width=1.5), name='модель')
        self.c_rl = self.pi.plot(pen=pg.mkPen(C_RL, width=1.2), name='RL')
        self.cursor_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#ffd23c', width=1, style=Qt.DotLine))
        self.cursor_line.setVisible(False)
        self.pi.addItem(self.cursor_line, ignoreBounds=True)
        self.range_lo = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#4b8bff', width=1, style=Qt.DashDotLine))
        self.range_hi = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#4b8bff', width=1, style=Qt.DashDotLine))
        self.pi.addItem(self.range_lo, ignoreBounds=True)
        self.pi.addItem(self.range_hi, ignoreBounds=True)
        self.bg_region = pg.LinearRegionItem(values=(3.0, 4.5), brush=pg.mkBrush(95, 191, 95, 40))
        self.bg_region.setVisible(False)
        self.pi.addItem(self.bg_region, ignoreBounds=True)
        self.win_items = []
        rv.addWidget(self.pw, 4)

        self.rw = pg.PlotWidget()
        self.rw.setMaximumHeight(170)
        self.rw.setXLink(self.pw)
        self.rw.getPlotItem().setLabel('left', 'Остаток (σ)')
        self.rw.getPlotItem().showGrid(x=True, y=True, alpha=0.15)
        self.c_res = self.rw.plot(pen=pg.mkPen('#8cb4ff', width=1))
        for v in (-3, 3):
            self.rw.addItem(pg.InfiniteLine(pos=v, angle=0, pen=pg.mkPen('#666', style=Qt.DashLine)))
        rv.addWidget(self.rw, 1)
        self.lbl_cursor = QtWidgets.QLabel(' ')
        self.lbl_cursor.setStyleSheet('font-family:Consolas,monospace;')
        # a long read-out is cut, it must not widen the window while the mouse moves
        self.lbl_cursor.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        rv.addWidget(self.lbl_cursor)
        split.addWidget(right)
        split.setSizes([420, 1000])

        self._mouse_proxy = pg.SignalProxy(self.pw.scene().sigMouseMoved, rateLimit=30, slot=self.on_mouse_moved)
        self._marker_timer = QtCore.QTimer(self)
        self._marker_timer.setSingleShot(True)
        self._marker_timer.timeout.connect(self.draw_markers)
        self.pi.vb.sigRangeChanged.connect(self._schedule_markers)
        self.pw.scene().sigMouseClicked.connect(self.on_plot_clicked)
        return split

    def _build_maps_page(self):
        split = QtWidgets.QSplitter(Qt.Horizontal)
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(2, 2, 2, 2)
        self.comp_glw = pg.GraphicsLayoutWidget()
        self.comp_vb = self.comp_glw.addViewBox(lockAspect=True, invertY=True)
        self.comp_img = pg.ImageItem()
        self.comp_vb.addItem(self.comp_img)
        self.comp_scale = None
        lv.addWidget(self.comp_glw, 1)
        self.comp_legend = QtWidgets.QLabel('')
        self.comp_legend.setTextFormat(Qt.RichText)
        self.comp_legend.setWordWrap(True)      # otherwise the text width blocks the splitter
        self.comp_legend.setStyleSheet('font-size:11pt; padding:4px;')
        lv.addWidget(self.comp_legend)
        split.addWidget(left)
        self.grid_glw = pg.GraphicsLayoutWidget()
        self._grid_ncols = 0
        self._grid_timer = QtCore.QTimer(self)
        self._grid_timer.setSingleShot(True)
        self._grid_timer.timeout.connect(self.render_maps)
        self.grid_glw.installEventFilter(self)
        self._grid_proxy = pg.SignalProxy(self.grid_glw.scene().sigMouseMoved, rateLimit=30, slot=self.on_map_mouse)
        self._comp_proxy = pg.SignalProxy(self.comp_glw.scene().sigMouseMoved, rateLimit=30, slot=self.on_map_mouse)
        split.addWidget(self.grid_glw)
        split.setSizes([800, 800])
        return split

    def _build_pca_page(self):
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        v.setContentsMargins(2, 2, 2, 2)
        row = FlowLayout(spacing=14)
        b = QtWidgets.QPushButton('Вычислить PCA')
        b.setStyleSheet('font-weight:bold; padding:4px 10px;')
        b.setToolTip('Разложение спектрального изображения в диапазоне энергий фита (вкладка «Фон»)')
        b.clicked.connect(self.run_pca)
        row.addWidget(b)
        self.cb_pca_poisson = QtWidgets.QCheckBox('пуассоновская нормировка')
        self.cb_pca_poisson.setChecked(True)
        self.cb_pca_poisson.setToolTip('Деление на √(ожидаемых отсчётов) (Keenan & Kotula): шум становится белым, '
                                       'компоненты выше границы шума — сигнал')
        row.addWidget(self.cb_pca_poisson)
        self.cb_pca_bin = QtWidgets.QComboBox()
        self.cb_pca_bin.addItems(['нет', '2×2', '4×4'])
        self.cb_pca_bin.setToolTip(
            'Разложение (собственные значения, спектры компонент, граница шума) — по данным, просуммированным '
            'по блокам пикселей, как «Spatial binning» в MSA (HREM Research). Шум на блок растёт медленнее '
            'сигнала протяжённой фазы: слабые протяжённые компоненты выходят над шумом. '
            'Карты (счета) — проекции спектров исходных пикселей: полное разрешение сохраняется. '
            'Детали меньше блока выделяются хуже. Применяется кнопкой «Вычислить PCA»')
        row.addWidget(hbox(QtWidgets.QLabel('Биннинг для разложения:'), self.cb_pca_bin))
        self.cb_pca_local = QtWidgets.QCheckBox('локальный PCA, плитки')
        self.cb_pca_local.setToolTip(
            'Как «Spatial Local wPCA» в MSA (HREM Research): карта режется на плитки с перекрытием в полплитки, '
            'каждая раскладывается отдельно. Фаза, занимающая часть карты (сегрегация на границе, отдельные '
            'частицы), в своей плитке выходит над шумом раньше, чем во всей карте. Модель плитки — выбранные '
            'глобальные компоненты плюс те её сигнальные компоненты, которых в глобальной модели нет; плитки '
            'сшиваются плавно. Локальный PCA только добавляет: слабая фаза по всей карте, которую видит лишь '
            'глобальный PCA, не теряется. Плитка должна содержать хотя бы несколько тысяч пикселей. '
            'Применяется кнопкой «Вычислить PCA»')
        self.sp_pca_tile = ispin(core.LOCAL_PCA_TILE, 16, 1024, 'Сторона плитки локального PCA')
        self.sp_pca_tile.setSuffix(' px')
        row.addWidget(hbox(self.cb_pca_local, self.sp_pca_tile))
        self.sp_pca_k = ispin(4, 1, 48, 'Число компонент, по которым восстанавливаются данные')
        self.sp_pca_k.setEnabled(False)
        self.sp_pca_k.valueChanged.connect(self.on_pca_k_changed)
        b = QtWidgets.QPushButton('Авто')
        b.setToolTip('k = последняя сигнальная компонента: собственное значение выше границы шума '
                     '(Марченко–Пастур) и пространственно связная карта, или намного выше границы; '
                     'ручной выбор компонент сбрасывается')
        b.clicked.connect(self.pca_auto)
        self.cb_pca_skip = QtWidgets.QCheckBox('без шумовых')
        self.cb_pca_skip.setChecked(True)
        self.cb_pca_skip.setToolTip('Шумовые компоненты среди первых k (карта без пространственной связности, '
                                    'r ≈ 0; например, одиночные отсчёты в почти пустых пикселях) в модель не '
                                    'входят, даже если стоят между сигнальными')
        self.cb_pca_skip.toggled.connect(self.on_pca_k_changed)
        row.addWidget(hbox(QtWidgets.QLabel('Компонент в модели:'), self.sp_pca_k, b, self.cb_pca_skip))
        self.ed_pca_comps = QtWidgets.QLineEdit()
        self.ed_pca_comps.setFixedWidth(120)
        self.ed_pca_comps.setEnabled(False)
        self.ed_pca_comps.setToolTip('Номера компонент модели, например «1-3, 5» (Enter — применить). Также: двойной '
                                     'щелчок по карте компоненты или щелчок по точке на графике собственных значений '
                                     'добавляет / убирает её. Компонента 1 (средний спектр) входит всегда. '
                                     'Изменение k, «без шумовых» или «Авто» возвращает автоматический выбор')
        self.ed_pca_comps.editingFinished.connect(self.on_pca_comps_edited)
        row.addWidget(hbox(QtWidgets.QLabel('Состав:'), self.ed_pca_comps))
        self.cb_pca_use = QtWidgets.QCheckBox('Использовать восстановленные данные')
        self.cb_pca_use.setToolTip('Данные, восстановленные по k компонентам, подставляются во все спектры (ROI), '
                                   'фит и карты')
        self.cb_pca_use.setEnabled(False)
        self.cb_pca_use.toggled.connect(self.apply_pca)
        row.addWidget(self.cb_pca_use)
        b = QtWidgets.QPushButton('Экспорт PCA…')
        b.setToolTip('Собственные значения и нагрузки (CSV), счета (TIFF float32), всё вместе с отчётом '
                     'предобработки (NPZ)')
        b.clicked.connect(self.export_pca)
        row.addWidget(b)
        v.addLayout(row)
        # preprocessing of the raw cube before the decomposition (applied by «Вычислить PCA»)
        row = FlowLayout(spacing=14)
        self.cb_pre_ch = QtWidgets.QCheckBox('мёртвые каналы')
        self.cb_pre_ch.setToolTip('Канал суммарного спектра, отличающийся от среднего двух соседних больше, чем '
                                  'допускает линия шириной FWHM детектора (застрявший или пропущенный код АЦП, '
                                  'всплеск в одном канале), заменяется во всех пикселях интерполяцией по соседним')
        self.cb_pre_drift = QtWidgets.QCheckBox('дрейф энергии')
        self.cb_pre_drift.setToolTip('Сдвиг шкалы энергии по ходу съёмки: по блокам строк развёртки, по самым '
                                     'сильным пикам. Исправляется построчно, только если он значим')
        self.cb_pre_px = QtWidgets.QCheckBox('плохие пиксели')
        self.cb_pre_px.setToolTip('Горячие и пустые пиксели: сумму отсчётов не объясняет даже второй по яркости '
                                  '(темноте) из 8 соседей. Заменяются спектром соседнего пикселя')
        self.sp_pre_sigma = dspin(core.PRE_PIXEL_SIGMA, 3.0, 50.0, 0.5, 1, ' σ', 'Порог плохого пикселя')
        for cb in (self.cb_pre_ch, self.cb_pre_drift, self.cb_pre_px):
            cb.setChecked(True)
        row.addWidget(hbox(QtWidgets.QLabel('Предобработка:'), self.cb_pre_ch, self.cb_pre_drift, self.cb_pre_px,
                           self.sp_pre_sigma))
        v.addLayout(row)
        self.lbl_pre = QtWidgets.QLabel('')
        self.lbl_pre.setWordWrap(True)
        v.addWidget(self.lbl_pre)
        self.lbl_pca = QtWidgets.QLabel('PCA не вычислен.')
        self.lbl_pca.setWordWrap(True)
        v.addWidget(self.lbl_pca)

        split = QtWidgets.QSplitter(Qt.Horizontal)
        left = QtWidgets.QSplitter(Qt.Vertical)
        self.scree_pw = pg.PlotWidget()
        sp = self.scree_pw.getPlotItem()
        sp.setLogMode(y=True)
        sp.setLabel('bottom', 'Компонента')
        sp.setLabel('left', 'λ / λ шума')
        sp.showGrid(x=True, y=True, alpha=0.15)
        sp.setTitle('Собственные значения (eigenvalues, scree plot)')
        self.scree_noise = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('#ff9b2d', width=1, style=Qt.DashLine),
                                           label='граница шума', labelOpts=dict(position=0.75, color='#ff9b2d'))
        sp.addItem(self.scree_noise, ignoreBounds=True)
        self.scree_cut = pg.InfiniteLine(pos=4.5, angle=90, movable=True, pen=pg.mkPen('#ffd23c', width=2),
                                         hoverPen=pg.mkPen('#fff', width=2))
        self.scree_cut.setToolTip('Перетащите, чтобы выбрать число компонент')
        self.scree_cut.sigPositionChangeFinished.connect(
            lambda line: self.sp_pca_k.setValue(max(1, int(round(line.value() - 0.5)))))
        sp.addItem(self.scree_cut, ignoreBounds=True)
        # one point per component: colour = in the model or not, x = classified as noise; a click toggles it
        self.scree_pts = sp.plot(pen=None, symbol='o')
        self.scree_pts.sigPointsClicked.connect(self.on_scree_clicked)
        self.scree_pw.setToolTip('Щелчок по точке — добавить компоненту в модель или убрать её')
        left.addWidget(self.scree_pw)
        self.load_pw = pg.PlotWidget()
        lp = self.load_pw.getPlotItem()
        lp.setLabel('bottom', 'Энергия (кэВ)')
        lp.showGrid(x=True, y=True, alpha=0.15)
        lp.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('#555')), ignoreBounds=True)
        self.load_curve = lp.plot(pen=pg.mkPen('#8cb4ff', width=1.2))
        left.addWidget(self.load_pw)
        left.setSizes([400, 400])
        split.addWidget(left)
        self.pca_glw = pg.GraphicsLayoutWidget()
        self.pca_glw.scene().sigMouseClicked.connect(self.on_pca_map_clicked)
        # maps of all components: the first 12 fill the visible area, the others are below (scroll)
        self.pca_scroll = QtWidgets.QScrollArea()
        self.pca_scroll.setWidgetResizable(True)
        self.pca_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.pca_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.pca_scroll.setWidget(self.pca_glw)
        self.pca_scroll.installEventFilter(self)
        split.addWidget(self.pca_scroll)
        split.setSizes([600, 1000])
        v.addWidget(split, 1)
        self.pca_views = []
        self.pca_sel = 0
        self._pca_ncols = 0
        self._pca_grid_timer = QtCore.QTimer(self)
        self._pca_grid_timer.setSingleShot(True)
        self._pca_grid_timer.timeout.connect(self.render_pca)
        self._pca_apply_timer = QtCore.QTimer(self)
        self._pca_apply_timer.setSingleShot(True)
        self._pca_apply_timer.timeout.connect(self.apply_pca)
        return page

    def _build_bg_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        g = QtWidgets.QGroupBox('Диапазон энергий для фита, идентификации и карт')
        f = QtWidgets.QHBoxLayout(g)
        self.sp_elo = dspin(0.2, 0.0, 40, 0.01, 3, ' кэВ')
        self.sp_ehi = dspin(20.0, 0.5, 80, 0.5, 2, ' кэВ')
        f.addWidget(QtWidgets.QLabel('от'))
        f.addWidget(self.sp_elo)
        f.addWidget(QtWidgets.QLabel('до'))
        f.addWidget(self.sp_ehi)
        f.addStretch(1)
        for s in (self.sp_elo, self.sp_ehi):
            s.valueChanged.connect(self.on_range_changed)
        v.addWidget(g)

        g = QtWidgets.QGroupBox('Шумовой пик детектора')
        f = QtWidgets.QVBoxLayout(g)
        self.cb_noise = QtWidgets.QCheckBox('Учитывать хвост шумового пика в фоне')
        self.cb_noise.setChecked(True)
        self.cb_noise.toggled.connect(self.invalidate_fit)
        self.lbl_noise = QtWidgets.QLabel('—')
        self.lbl_noise.setWordWrap(True)
        f.addWidget(self.cb_noise)
        f.addWidget(self.lbl_noise)
        v.addWidget(g)

        g = QtWidgets.QGroupBox('Оценка фона (форма фона)')
        f = QtWidgets.QFormLayout(g)
        self.cb_bg_method = QtWidgets.QComboBox()
        self.cb_bg_method.addItems(['SNIP (автоматически)', 'По минимумам (Autobackground, DM)', 'Полином по окнам',
                                    'Нет (только полином в фите)'])
        self.cb_bg_method.setItemData(1, 'Узлы — локальные минимумы сглаженного спектра (значение — исходные '
                                         'отсчёты), за каждый цикл узел выше обоих соседних узлов заменяется их '
                                         'средним; фон — ломаная через узлы', Qt.ToolTipRole)
        self.cb_bg_method.currentIndexChanged.connect(self.recompute_background)
        f.addRow('Метод:', self.cb_bg_method)
        self.sp_snip = dspin(2.0, 0.5, 10, 0.25, 2, ' × FWHM', 'Ширина окна отсечения SNIP')
        self.sp_snip.valueChanged.connect(self.recompute_background)
        f.addRow('SNIP, ширина:', self.sp_snip)
        self.sp_min_smooth = ispin(5, 1, 51, 'Длина гауссова ядра сглаживания перед поиском минимумов, каналов '
                                             '(σ = длина / 5; smoothrange в скрипте DM)')
        self.sp_min_smooth.setSuffix(' кан.')
        self.sp_min_smooth.valueChanged.connect(self.recompute_background)
        self.sp_min_cycles = ispin(20, 0, 500, 'Число циклов срезания узлов, выступающих над соседними '
                                               '(ncycles в скрипте DM)')
        self.sp_min_cycles.setSuffix(' цикл.')
        self.sp_min_cycles.valueChanged.connect(self.recompute_background)
        f.addRow('Минимумы:', hbox(self.sp_min_smooth, self.sp_min_cycles))
        self.sp_poly = ispin(3, 0, 12, 'Порядок полинома по окнам')
        self.sp_poly.valueChanged.connect(self.recompute_background)
        f.addRow('Порядок полинома:', self.sp_poly)
        self.ed_windows = QtWidgets.QLineEdit('0.9-1.0, 3.0-4.8, 10.0-10.8, 12.0-16.0')
        self.ed_windows.setToolTip('Окна без пиков, кэВ: "a-b, c-d, ..."')
        self.ed_windows.editingFinished.connect(self.recompute_background)
        f.addRow('Окна фона (кэВ):', self.ed_windows)
        row = QtWidgets.QHBoxLayout()
        self.bt_region = QtWidgets.QPushButton('Показать окно на графике')
        self.bt_region.setCheckable(True)
        self.bt_region.toggled.connect(lambda on: self.bg_region.setVisible(on))
        b = QtWidgets.QPushButton('Добавить окно')
        b.clicked.connect(self.add_bg_window)
        row.addWidget(self.bt_region)
        row.addWidget(b)
        f.addRow(row)
        self.sp_nbg = ispin(3, 1, 10, 'Число членов полинома Лежандра, умножающего форму фона, в фите и в картах')
        self.sp_nbg.valueChanged.connect(self.invalidate_fit)
        f.addRow('Полином фона в фите:', self.sp_nbg)
        v.addWidget(g)
        b = QtWidgets.QPushButton('Пересчитать фон')
        b.clicked.connect(self.recompute_background)
        v.addWidget(b)
        v.addStretch(1)
        return w

    def _build_elements_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        self.ptable = PeriodicTable()
        self.ptable.toggled.connect(self.on_element_toggled)
        self.ptable.hovered.connect(self.show_preview)
        self.ptable.left.connect(lambda s: self.show_preview(None))
        v.addWidget(self.ptable)

        g = QtWidgets.QGroupBox('Автоматическое определение элементов')
        gv = QtWidgets.QVBoxLayout(g)
        row = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton('Автоопределение')
        b.setStyleSheet('font-weight:bold; padding:4px 10px;')
        b.clicked.connect(self.run_auto_id)
        row.addWidget(b)
        self.b_file_els = QtWidgets.QPushButton('Из файла')
        self.b_file_els.setToolTip('Элементы, определённые программой съёмки (EDAX Genesis / TEAM, список в .spc)')
        self.b_file_els.setEnabled(False)
        self.b_file_els.clicked.connect(self.elements_from_file)
        row.addWidget(self.b_file_els)
        row.addWidget(QtWidgets.QLabel('порог'))
        self.sp_thr = dspin(5.0, 2.0, 50.0, 0.5, 1, ' σ', 'Минимальная значимость пика и прироста χ²')
        row.addWidget(self.sp_thr)
        self.cb_id_keep = QtWidgets.QCheckBox('дополнить текущие')
        row.addWidget(self.cb_id_keep)
        row.addStretch(1)
        gv.addLayout(row)
        self.id_tree = QtWidgets.QTreeWidget()
        self.id_tree.setHeaderLabels(['Элемент', 'Серии', 'Найден по', 'σ', 'Оценка'])
        self.id_tree.setRootIsDecorated(False)
        self.id_tree.setMinimumHeight(170)
        self.id_tree.itemDoubleClicked.connect(self.on_id_item_clicked)
        gv.addWidget(self.id_tree)
        self.lbl_unid = QtWidgets.QLabel('')
        self.lbl_unid.setWordWrap(True)
        gv.addWidget(self.lbl_unid)
        row = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton('Удалить выбранный')
        b.clicked.connect(self.remove_selected_element)
        row.addWidget(b)
        b = QtWidgets.QPushButton('Очистить все')
        b.clicked.connect(self.clear_elements)
        row.addWidget(b)
        row.addStretch(1)
        gv.addLayout(row)
        v.addWidget(g)

        g = QtWidgets.QGroupBox('Линии вблизи (щёлкните по спектру; двойной щелчок по строке — добавить)')
        gv = QtWidgets.QVBoxLayout(g)
        row = QtWidgets.QHBoxLayout()
        self.lbl_near = QtWidgets.QLabel('E = —')
        row.addWidget(self.lbl_near)
        row.addStretch(1)
        row.addWidget(QtWidgets.QLabel('допуск'))
        self.sp_tol = dspin(1.0, 0.2, 5, 0.1, 1, ' × FWHM')
        self.sp_tol.valueChanged.connect(self.update_near_lines)
        row.addWidget(self.sp_tol)
        gv.addLayout(row)
        self.near_tree = QtWidgets.QTreeWidget()
        self.near_tree.setHeaderLabels(['Линия', 'E, кэВ', 'ΔE, эВ', 'Вес', 'Элемент'])
        self.near_tree.setRootIsDecorated(False)
        self.near_tree.setMinimumHeight(160)
        self.near_tree.itemDoubleClicked.connect(lambda it, c: self.add_element(it.text(4)))
        gv.addWidget(self.near_tree)
        v.addWidget(g)
        self._near_e = None
        return w

    def _build_fit_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        g = QtWidgets.QGroupBox('Детектор и калибровка')
        f = QtWidgets.QGridLayout(g)
        self.sp_fwhm = dspin(130.0, 40, 400, 1, 1, ' эВ', 'FWHM на линии Mn Kα (5.899 кэВ)')
        self.sp_fano = dspin(0.114, 0.05, 0.4, 0.002, 3, '', 'Фактор Фано (Si ≈ 0.114)')
        self.sp_spread = dspin(0.0, 0.0, 2.0, 0.02, 3, ' %', 'Относительный разброс усиления (суммирование нескольких детекторов)')
        self.sp_off = dspin(0.0, -200, 200, 1, 1, ' эВ', 'Поправка смещения шкалы: E = E·(1+g) + сдвиг')
        self.sp_gain = dspin(0.0, -5, 5, 0.01, 3, ' %', 'Поправка наклона шкалы энергии')
        self.sp_tail = dspin(0.0, 0, 30, 0.5, 2, ' %', 'Доля отсчётов линии в низкоэнергетическом хвосте')
        self.sp_tailw = dspin(1.0, 0.3, 6, 0.1, 2, ' × FWHM', 'Ширина хвоста')
        self.sp_cut = dspin(0.0, 0, 500, 1, 1, ' эВ', 'Нижний порог регистрации: процессор отбрасывает импульсы ниже '
                            'него, низкоэнергетический край пиков (Nb M, C K) срезан. 0 — не учитывать. '
                            'Уточняется вместе с калибровкой, если диапазон фита начинается у порога')
        self.sp_cutw = dspin(10.0, 2, 50, 0.5, 1, ' эВ', 'Ширина (размытие) порога регистрации')
        self.cb_escape = QtWidgets.QCheckBox('Escape-пики Si')
        self.cb_escape.setChecked(True)
        items = [('FWHM (Mn Kα):', self.sp_fwhm), ('Фано:', self.sp_fano), ('Разброс усиления:', self.sp_spread),
                 ('Сдвиг энергии:', self.sp_off), ('Наклон:', self.sp_gain), ('Хвост:', self.sp_tail),
                 ('Ширина хвоста:', self.sp_tailw), ('Порог регистрации:', self.sp_cut),
                 ('Ширина порога:', self.sp_cutw)]
        for i, (lab, sp) in enumerate(items):
            f.addWidget(QtWidgets.QLabel(lab), i // 2, (i % 2) * 2)
            f.addWidget(sp, i // 2, (i % 2) * 2 + 1)
            sp.valueChanged.connect(self.on_det_ui_changed)
        f.addWidget(self.cb_escape, 4, 2, 1, 2)
        self.cb_escape.toggled.connect(self.on_det_ui_changed)
        v.addWidget(g)

        g = QtWidgets.QGroupBox('Фит спектра (деконволюция перекрывающихся линий)')
        gv = QtWidgets.QVBoxLayout(g)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel('Уточнять:'))
        self.cb_ref_cal = QtWidgets.QCheckBox('калибровку')
        self.cb_ref_res = QtWidgets.QCheckBox('разрешение')
        self.cb_ref_tail = QtWidgets.QCheckBox('хвост')
        for cb in (self.cb_ref_cal, self.cb_ref_res, self.cb_ref_tail):
            cb.setChecked(True)
            row.addWidget(cb)
        row.addStretch(1)
        gv.addLayout(row)
        row = FlowLayout()
        self.cb_shape = QtWidgets.QComboBox()
        self.cb_shape.addItems(['по модели детектора', 'NLLS — подгонка формы серий'])
        self.cb_shape.setToolTip(
            'По модели детектора: форма каждой серии задана калибровкой, разрешением FWHM(E) и табличными '
            'отношениями линий, подбираются только интенсивности (линейный МНК).\n'
            'NLLS: у каждой серии, найденной выше 10σ, дополнительно подгоняются (нелинейный МНК) положение, '
            'ширина и отношения групп линий (Kβ/Kα, Lβ/Lα, Lγ, Lℓ, Mβ/Mα…) — с априорными ограничениями, '
            'удерживающими плохо определённые параметры у значений модели. Отношения не освобождаются, если '
            'группа лежит под сильной линией другого элемента. Подогнанная форма используется и в картах MLLS.')
        self.cb_nlls_shift = QtWidgets.QCheckBox('положение')
        self.cb_nlls_width = QtWidgets.QCheckBox('ширина')
        self.cb_nlls_ratio = QtWidgets.QCheckBox('отношения линий')
        row.addWidget(hbox(QtWidgets.QLabel('Форма пиков:'), self.cb_shape))
        for cb in (self.cb_nlls_shift, self.cb_nlls_width, self.cb_nlls_ratio):
            cb.setChecked(True)
            cb.setEnabled(False)
            row.addWidget(cb)
        self.cb_shape.currentIndexChanged.connect(
            lambda i: [cb.setEnabled(i == 1) for cb in (self.cb_nlls_shift, self.cb_nlls_width, self.cb_nlls_ratio)])
        gv.addLayout(row)
        b = QtWidgets.QPushButton('Фитировать спектр')
        b.setStyleSheet('font-weight:bold; padding:5px;')
        b.clicked.connect(self.run_fit)
        gv.addWidget(b)
        self.lbl_chi2 = QtWidgets.QLabel('')
        self.lbl_chi2.setWordWrap(True)
        gv.addWidget(self.lbl_chi2)
        self.fit_table = QtWidgets.QTableWidget(0, 11)
        self.fit_table.setHorizontalHeaderLabels(['Вкл', 'Элемент', 'Серия', 'Линия', 'E, кэВ', 'ΔE, эВ', 'Ширина',
                                                  'Отсчёты', '± σ', 'σ', 'Отношения линий'])
        self.fit_table.verticalHeader().setVisible(False)
        self.fit_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.fit_table.setMinimumHeight(300)
        self.fit_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        gv.addWidget(self.fit_table)
        row = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton('Таблица → CSV')
        b.clicked.connect(self.export_fit_table)
        row.addWidget(b)
        b = QtWidgets.QPushButton('Спектр и модель → CSV')
        b.clicked.connect(self.export_spectrum)
        row.addWidget(b)
        gv.addLayout(row)
        v.addWidget(g)

        g = QtWidgets.QGroupBox('Деконволюция спектра Ричардсона–Люси (повышение разрешения)')
        row = QtWidgets.QHBoxLayout(g)
        row.addWidget(QtWidgets.QLabel('итераций'))
        self.sp_rl = ispin(30, 1, 2000)
        row.addWidget(self.sp_rl)
        b = QtWidgets.QPushButton('Вычислить')
        b.clicked.connect(self.run_rl)
        row.addWidget(b)
        row.addStretch(1)
        v.addWidget(g)
        v.addStretch(1)
        return w

    def _build_maps_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        g = QtWidgets.QGroupBox('Расчёт карт')
        f = QtWidgets.QFormLayout(g)
        self.cb_map_method = QtWidgets.QComboBox()
        self.cb_map_method.addItems(['MLLS-фит (деконволюция перекрытий)', 'Окно − фон − перекрытия', 'Окно (брутто)'])
        for i, tip in enumerate((
                'Попиксельный фит формами линий и фоном: перекрытия разделяются, фон вычитается. '
                'Рекомендуется, особенно для слабых элементов.',
                'Отсчёты в окне ±0.6 FWHM вокруг главной линии минус фон и хвосты соседних линий '
                '(например, Nb M в окне C K) из попиксельного фита. Менее чувствителен к неточной форме '
                'собственного пика, чем MLLS.',
                'Все отсчёты в окне, без вычитания фона: в карту входят тормозной фон и хвосты соседних '
                'линий, поэтому участки с большим общим числом отсчётов (тяжёлые, плотные слои) светятся '
                'во всех картах.')):
            self.cb_map_method.setItemData(i, tip, Qt.ToolTipRole)
        f.addRow('Метод:', self.cb_map_method)
        self.cb_tie_low = QtWidgets.QCheckBox('Серии < 1 кэВ следуют за основной серией элемента')
        self.cb_tie_low.setChecked(True)
        self.cb_tie_low.setToolTip(
            'MLLS: интенсивность Nb M, Fe L, Cr L и т. п. в каждом пикселе = основная серия (Nb L, Fe K…) × '
            'отношение из суммарного спектра. Слабые низкоэнергетические серии тогда не обмениваются шумом '
            'с перекрывающимися линиями C K, O K, N K (меньше шума и ложной структуры в их картах); '
            'изменение отношения M/L от пикселя к пикселю (поглощение) при этом не видно')
        f.addRow(self.cb_tie_low)
        self.cb_phase_bg = QtWidgets.QCheckBox('Фон по фазам (автоматические кластеры)')
        self.cb_phase_bg.setChecked(True)
        self.cb_phase_bg.setToolTip(
            'Пиксели делятся на фазы (k-means по счетам PCA, число фаз = число значимых компонент); '
            'у каждой фазы своя форма фона (SNIP или минимумы её суммарного спектра), фон пикселя — их смесь. '
            'Форма фона при низких энергиях у фаз разная (края поглощения), с общей формой фона '
            'слабые линии (C K, O K) получают ложный контраст между фазами. Только для фона SNIP и '
            'по минимумам. '
            'Фазы: список «Изображение» на вкладке «Спектр»')
        f.addRow(self.cb_phase_bg)
        b = QtWidgets.QPushButton('Построить карты')
        b.setStyleSheet('font-weight:bold; padding:5px;')
        b.clicked.connect(self.run_maps)
        f.addRow(b)
        v.addWidget(g)

        g = QtWidgets.QGroupBox('Отображение')
        f = QtWidgets.QFormLayout(g)
        row = QtWidgets.QHBoxLayout()
        self.cb_smooth = QtWidgets.QComboBox()
        self.cb_smooth.addItems(['нет', 'Гаусс σ (пикс.)', 'Бининг k×k'])
        self.sp_smooth = dspin(1.0, 0.0, 20, 0.5, 1)
        row.addWidget(self.cb_smooth)
        row.addWidget(self.sp_smooth)
        f.addRow('Сглаживание:', row)
        row = QtWidgets.QHBoxLayout()
        self.b_blemish = QtWidgets.QPushButton('Убрать горячие пиксели')
        self.b_blemish.setCheckable(True)
        self.b_blemish.setToolTip(
            'По скрипту DM «Point Blemish»: пиксель плохой, если его лапласиан (8·центр − 8 соседей) '
            'отклоняется от среднего больше чем на N σ; плохие пиксели заполняются средним хороших соседей, от '
            'краёв к центру. σ — локальная (окно 15×15; в DM — по всей карте): шум карты растёт с числом '
            'отсчётов, с общей σ шум яркой фазы и края слоёв считались бы дефектами. Применяется к каждой карте '
            'до сглаживания, исходные карты не меняются (отжать кнопку — вернуть). Число заменённых пикселей — '
            'в подписи под композитом')
        self.cb_blemish = QtWidgets.QComboBox()
        self.cb_blemish.addItems(['точки', 'точки и области'])
        self.cb_blemish.setItemData(0, 'Только лапласиан: одиночные горячие и холодные пиксели и пятна '
                                       'из нескольких пикселей', Qt.ToolTipRole)
        self.cb_blemish.setItemData(1, 'Также пиксели, яркость которых отличается от среднего по карте больше чем '
                                       'на N σ: убирает и пятна из нескольких пикселей, но может стереть '
                                       'настоящие яркие частицы малой площади', Qt.ToolTipRole)
        self.sp_blemish = dspin(5.0, 2.0, 20.0, 0.5, 1, ' σ', 'Порог (в скрипте DM — 5σ)')
        row.addWidget(self.b_blemish)
        row.addWidget(self.cb_blemish)
        row.addWidget(self.sp_blemish)
        f.addRow(row)
        row = QtWidgets.QHBoxLayout()
        self.sp_plo = dspin(1.0, 0, 50, 0.5, 1, ' %')
        self.sp_phi = dspin(99.5, 50, 100, 0.1, 2, ' %')
        row.addWidget(self.sp_plo)
        row.addWidget(QtWidgets.QLabel('–'))
        row.addWidget(self.sp_phi)
        f.addRow('Контраст (процентили):', row)
        self.cb_common = QtWidgets.QCheckBox('Общая шкала для всех карт')
        self.cb_common.setToolTip(
            'Все карты показываются в одном диапазоне: от 0 до наибольшего верхнего процентиля среди карт '
            '(в чистых отсчётах на пиксель). Яркость тогда можно сравнивать между элементами: слабая линия '
            'выглядит тёмной, а не растянутой на весь диапазон. Сравниваются отсчёты линий, а не концентрации '
            '(сечения ионизации и эффективность детектора у линий разные). Нижний процентиль не используется')
        self.cb_common.toggled.connect(lambda on: self.sp_plo.setEnabled(not on))
        self.cb_common.toggled.connect(self.render_maps)
        f.addRow(self.cb_common)
        self.sp_gamma = dspin(1.0, 0.2, 3, 0.05, 2)
        f.addRow('Гамма:', self.sp_gamma)
        self.cb_blend = QtWidgets.QComboBox()
        self.cb_blend.addItems(['Сложение', 'Экран (screen)', 'Максимум'])
        f.addRow('Смешивание цветов:', self.cb_blend)
        row = QtWidgets.QHBoxLayout()
        self.cb_base = QtWidgets.QComboBox()
        self.cb_base.setToolTip('Изображение под композитом в оттенках серого. Электронное изображение берётся в '
                                'своём разрешении (до 2048 пикс.), карты интерполируются на него')
        self.cb_base.addItem('Сумма отсчётов (X-ray)')
        self.sp_base = dspin(0.0, 0, 100, 5, 0, ' %', 'Яркость подложки (0 — без подложки)')
        b = QtWidgets.QPushButton('Загрузить…')
        b.setToolTip('Электронное изображение SEM / STEM области карты (TIFF, BMP, PNG, JPEG)')
        b.clicked.connect(self.load_image_dialog)
        row.addWidget(self.cb_base, 1)
        row.addWidget(self.sp_base)
        row.addWidget(b)
        f.addRow('Подложка:', row)
        self.cb_cmap = QtWidgets.QComboBox()
        self.cb_cmap.addItems(['цвет элемента', 'viridis', 'inferno', 'magma', 'gray'])
        f.addRow('Палитра отдельных карт:', self.cb_cmap)
        for wdg in (self.cb_smooth, self.cb_blend, self.cb_cmap, self.cb_base, self.cb_blemish):
            wdg.currentIndexChanged.connect(self.render_maps)
        for wdg in (self.sp_smooth, self.sp_plo, self.sp_phi, self.sp_gamma, self.sp_base, self.sp_blemish):
            wdg.valueChanged.connect(self.render_maps)
        self.b_blemish.toggled.connect(self.render_maps)
        v.addWidget(g)

        g = QtWidgets.QGroupBox('Элементы на картах (галочка — в композите)')
        gv = QtWidgets.QVBoxLayout(g)
        self.map_table = QtWidgets.QTableWidget(0, 5)
        self.map_table.setHorizontalHeaderLabels(['', 'Элемент', 'Серия', 'Цвет', 'Яркость'])
        self.map_table.verticalHeader().setVisible(False)
        self.map_table.setMinimumHeight(260)
        self.map_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        gv.addWidget(self.map_table)
        row = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton('Сохранить композит (PNG)…')
        b.clicked.connect(self.save_composite)
        row.addWidget(b)
        b = QtWidgets.QPushButton('Сохранить все карты…')
        b.clicked.connect(self.save_all_maps)
        row.addWidget(b)
        gv.addLayout(row)
        b = QtWidgets.QPushButton('Экспорт в PowerPoint…')
        b.setToolTip('Презентация .pptx: обзорное изображение с областью карт, композит и все карты элементов')
        b.clicked.connect(self.export_pptx)
        gv.addWidget(b)
        v.addWidget(g)
        v.addStretch(1)
        return w

    def show_help(self):
        """Help in its own window with a scroll bar (the text is taller than a message box allows); not modal,
        so it can stay open while working."""
        if self.help_dlg is None:
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle('Справка')
            v = QtWidgets.QVBoxLayout(dlg)
            text = QtWidgets.QTextBrowser()
            text.setHtml(HELP)
            v.addWidget(text)
            b = QtWidgets.QPushButton('Закрыть')
            b.clicked.connect(dlg.close)
            v.addWidget(b, 0, Qt.AlignRight)
            scr = self.screen().availableGeometry()
            dlg.resize(min(900, int(0.9 * scr.width())), int(0.8 * scr.height()))
            self.help_dlg = dlg
        self.help_dlg.show()
        self.help_dlg.raise_()
        self.help_dlg.activateWindow()

    def eventFilter(self, obj, ev):
        if obj is getattr(self, 'grid_glw', None) and ev.type() == QtCore.QEvent.Resize and self.maps is not None:
            self._grid_timer.start(200)
        if obj is getattr(self, 'pca_scroll', None) and ev.type() == QtCore.QEvent.Resize and self.pca is not None:
            self._pca_grid_timer.start(200)
        return super().eventFilter(obj, ev)

    # ------------------------------------------------------------------ helpers
    def erange(self):
        lo, hi = self.sp_elo.value(), self.sp_ehi.value()
        if self.si is not None:
            hi = min(hi, float(self.si.energy[-1]))
        return lo, max(hi, lo + 0.1)

    def e0(self):
        return self.si.voltage if self.si is not None else 300.0

    def noise_peak(self):
        return self.noise if self.cb_noise.isChecked() else None

    def families(self):
        lo, hi = self.erange()
        out = []
        for el in self.elements:
            for f in core.element_families(el, lo, hi, self.e0()):
                if self.fam_enabled.get((el, f.family), True):
                    out.append(f)
        return out

    def color(self, el):
        if el not in self.colors:
            used = {c.name() for c in self.colors.values()}
            free = [c for c in PALETTE if c not in used]
            self.colors[el] = QtGui.QColor(free[0] if free else PALETTE[len(self.colors) % len(PALETTE)])
        return self.colors[el]

    def ui_to_det(self):
        self.det = core.Detector(fwhm_mn=self.sp_fwhm.value() / 1000, fano=self.sp_fano.value(),
                                 spread=self.sp_spread.value() / 100, offset=self.sp_off.value() / 1000,
                                 gain=self.sp_gain.value() / 100, tail=self.sp_tail.value() / 100,
                                 tail_width=self.sp_tailw.value(), escape=self.cb_escape.isChecked(),
                                 shifts=dict(self.det.shifts), thr=self.sp_cut.value() / 1000,
                                 thr_width=self.sp_cutw.value() / 1000, widths=dict(self.det.widths),
                                 ratios={k: dict(v) for k, v in self.det.ratios.items()})

    def det_to_ui(self):
        d = self.det
        vals = [(self.sp_fwhm, d.fwhm_mn * 1000), (self.sp_fano, d.fano), (self.sp_spread, d.spread * 100),
                (self.sp_off, d.offset * 1000), (self.sp_gain, d.gain * 100), (self.sp_tail, d.tail * 100),
                (self.sp_tailw, d.tail_width), (self.sp_cut, d.thr * 1000), (self.sp_cutw, d.thr_width * 1000)]
        for sp, val in vals:
            sp.blockSignals(True)
            sp.setValue(float(val))
            sp.blockSignals(False)
        self.cb_escape.blockSignals(True)
        self.cb_escape.setChecked(d.escape)
        self.cb_escape.blockSignals(False)

    def roi_bounds(self):
        ny, nx = self.si.shape
        p, s = self.roi.pos(), self.roi.size()
        x0 = int(np.clip(round(p.x()), 0, nx - 1))
        y0 = int(np.clip(round(p.y()), 0, ny - 1))
        x1 = int(np.clip(round(p.x() + s.x()), x0 + 1, nx))
        y1 = int(np.clip(round(p.y() + s.y()), y0 + 1, ny))
        return y0, y1, x0, x1

    def current_spectrum(self):
        if self.cb_roi.isChecked():
            y0, y1, x0, x1 = self.roi_bounds()
            return self.si.roi_spectrum(y0, y1, x0, x1)
        return self.si.sum_spectrum()

    def bg_method(self):
        return BG_METHODS[self.cb_bg_method.currentIndex()]

    def background_of(self, y):
        method = self.bg_method()
        E = self.si.energy
        if method == 'snip':
            return core.snip_background(E, y, self.det, self.sp_snip.value())
        if method == 'minima':
            return core.minima_background(E, y, self.sp_min_smooth.value(), self.sp_min_cycles.value())
        if method == 'poly':
            wins = core.parse_windows(self.ed_windows.text())
            return core.poly_background(E, y, wins, self.sp_poly.value())
        return None

    # ------------------------------------------------------------------ file
    def open_file(self, path=None):
        if not path:
            start = os.path.dirname(self.si.path) if self.si else os.path.dirname(os.path.abspath(__file__))
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Открыть спектральное изображение', start,
                                                            'Все поддерживаемые (*.emi *.ser *.spd *.lsd *.spc);;'
                                                            'FEI TIA (*.emi *.ser);;'
                                                            'EDAX Genesis / TEAM (*.spd *.lsd *.spc);;Все файлы (*)')
            if not path:
                return
        self._load(path)

    def _load(self, path):
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            si = core.load_si(path)
        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(self, 'Ошибка чтения', f'{path}\n\n{e}')
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.si = self.si_raw = si
        self._phases = self.map_phases = None
        self.reset_pca()
        self.fit = self.maps = self.rl = None
        self.id_info, self.unidentified = {}, []
        for el in list(self.elements):
            self.ptable.set_checked(el, False)
        self.elements = []
        self.fam_enabled = {}
        self.b_file_els.setEnabled(bool(self.file_elements()))
        self.b_file_els.setToolTip('Элементы, определённые программой съёмки (EDAX Genesis / TEAM, список в .spc)'
                                   + (': ' + ', '.join(self.file_elements()) if self.file_elements() else ''))
        E = si.energy
        y = si.sum_spectrum()
        self.noise = core.detect_noise_peak(E, y)
        thr = core.estimate_threshold(E, y)
        fwhm = {'fwhm_mn': si.fwhm_mn} if si.fwhm_mn else {}    # resolution stored in the file (EDAX)
        self.det = core.Detector(**fwhm)
        lo = 0.2
        if self.noise:
            lo = float(np.clip(round(self.noise[0] + 2.5 * self.noise[1], 2), 0.1, 0.5))
            self.lbl_noise.setText(f'найден: {self.noise[0] * 1000:.0f} эВ, σ = {self.noise[1] * 1000:.0f} эВ')
        else:
            if thr:     # the spectrum starts at the pulse-processor threshold: model it and fit from there
                self.det = core.Detector(thr=thr[0], thr_width=thr[1], **fwhm)
                lo = float(np.clip(round(thr[0] - 2 * thr[1], 2), 0.05, 0.5))
            self.lbl_noise.setText('не найден' + (f'; порог регистрации ≈ {thr[0] * 1000:.0f} эВ учитывается в модели '
                                                  f'(вкладка «Фит»)' if thr else ''))
        for sp, val in ((self.sp_elo, lo), (self.sp_ehi, min(20.0, float(E[-1]) - 0.05))):
            sp.blockSignals(True)
            sp.setValue(val)
            sp.blockSignals(False)
        self.det_to_ui()

        self.reset_images()
        ny, nx = si.shape
        self.roi.setVisible(True)
        self.roi.maxBounds = QtCore.QRectF(0, 0, nx, ny)
        self.roi.setPos([nx // 4, ny // 4], finish=False)
        self.roi.setSize([max(nx // 2, 1), max(ny // 2, 1)], finish=False)
        self.update_si_image()
        self.si_vb.autoRange()
        self.update_survey()

        self._counts_total = float(y.sum())
        self.update_meta_label()
        self.setWindowTitle(f'EDX: {si.name}')
        self.status_info.setText(f'{nx}×{ny} пикс., {len(E)} каналов, E0 = {si.voltage:g} кэВ')
        self.update_spectrum()
        self.pi.setXRange(0, min(20.0, float(E[-1])), padding=0.01)
        self.update_fit_table()
        self.update_id_tree()
        self.clear_maps()
        if 'missing_image' in si.meta:
            self.statusBar().showMessage(f'Файл загружен. Электронного изображения {si.meta["missing_image"]} '
                                         f'(.bmp / .tif) нет в папке; изображение SEM для наложения карт — кнопка '
                                         f'«Загрузить…».')
        else:
            self.statusBar().showMessage('Файл загружен. Далее: «Элементы» → «Автоопределение» или выбор в таблице.')

    # ------------------------------------------------------------------ images
    def update_si_image(self, *args):
        if self.si is None:
            return
        txt = self.cb_img.currentText()
        ny, nx = self.si.shape
        bad = self.pre.bad_pixels if txt == 'Плохие пиксели' and self.pre is not None else None
        self.bad_marks.setVisible(bad is not None)
        if bad is not None:      # loaded data, the replaced pixels framed
            by, bx = np.nonzero(bad)
            self.bad_marks.setData(bx + 0.5, by + 0.5)
            img = self.si_raw.total_image(e_min=self.sp_elo.value())
            lo, hi = np.percentile(img, [0.5, 99.5])
            self.si_img.setImage(img, levels=(lo, hi if hi > lo else lo + 1), lut=None)
            self.si_img.setRect(QtCore.QRectF(0, 0, nx, ny))
            return
        if txt == 'Локальный PCA: добавлено' and self.pca_local is not None:
            # local directions added to the global model, blended over the overlapping tiles as the reconstruction
            img = self.pca_local.tile_map(self.local_added())
            hi = float(img.max())
            self.si_img.setImage(img, levels=(0.0, hi if hi > 0 else 1.0),
                                 lut=pg.colormap.get('viridis').getLookupTable(0.0, 1.0, 256))
            self.si_img.setRect(QtCore.QRectF(0, 0, nx, ny))
            return
        if txt == 'Фазы фона' and self.map_phases is not None:
            n = int(self.map_phases.max()) + 1
            lut = np.array([QtGui.QColor(PALETTE[i % len(PALETTE)]).getRgb()[:3] for i in range(n)], dtype=np.uint8)
            self.si_img.setImage(self.map_phases.astype(float), levels=(0, n - 1 if n > 1 else 1), lut=lut)
            self.si_img.setRect(QtCore.QRectF(0, 0, nx, ny))
            return
        i = self.cb_img.currentIndex()
        if 0 <= i < len(self.images):
            img = self.images[i]['image']
        elif txt.startswith('Карта: ') and self.maps is not None:
            img = self.element_map(txt[len('Карта: '):])
        else:
            img = self.si.total_image(e_min=self.sp_elo.value())
        lo, hi = np.percentile(img, [0.5, 99.5])
        self.si_img.setImage(img, levels=(lo, hi if hi > lo else lo + 1), lut=None)
        # binned maps and electron images of any resolution span the SI pixel grid (ROI)
        self.si_img.setRect(QtCore.QRectF(0, 0, nx, ny))

    # ------------------------------------------------------------------ electron images for the overlay
    def reset_images(self):
        """Electron images of a newly loaded file: the one stored with the spectrum image (HAADF, EDAX .bmp)."""
        si = self.si_raw
        self.images = []
        if si.haadf is not None:        # name_en: in the presentation export
            self.images.append(dict(name=si.image_name.replace('Electron image', 'Электронное изображение'),
                                    name_en=si.image_name,
                                    image=si.image_full if si.image_full is not None else si.haadf))
        self._fill_image_boxes(select=0)

    def _fill_image_boxes(self, select=None):
        """Lists «Изображение» (electron images, X-ray counts, phases, maps) and «Подложка» (electron images,
        X-ray counts); select = index of the item to show in both, None: keep the current ones."""
        names = [im['name'] for im in self.images] + ['Сумма отсчётов (X-ray)']
        extra = (['Фазы фона'] if self.map_phases is not None else []) + [f'Карта: {el}' for el in self.map_elements()]
        if self.pre is not None and self.pre.bad_pixels.any():
            extra.append('Плохие пиксели')
        if self.pca_local is not None:
            extra.append('Локальный PCA: добавлено')
        for box, items in ((self.cb_img, names + extra), (self.cb_base, names)):
            cur = box.currentText()
            box.blockSignals(True)
            box.clear()
            box.addItems(items)
            box.setCurrentIndex(select if select is not None else max(box.findText(cur), 0))
            box.blockSignals(False)

    @guarded
    def load_image_dialog(self):
        if self.si is None:
            QtWidgets.QMessageBox.information(self, 'Изображение', 'Сначала откройте спектральное изображение.')
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Электронное изображение области карты',
                                                        os.path.dirname(self.si.path),
                                                        core.IMAGE_FILTER + ';;Все файлы (*)')
        if path:
            self.add_image(path)

    def add_image(self, path):
        """Electron image from a file, fitted to the field of view of the map, added to the image lists
        and selected (as the base of the composite too). False if cancelled."""
        name = os.path.basename(path)
        fitted = self.fit_image(core.read_image(path), name)
        if fitted is None:
            return False
        img, notes = fitted
        self.images = [im for im in self.images if im['name'] != name] + [dict(name=name, image=img, path=path)]
        self._fill_image_boxes(select=len(self.images) - 1)
        if self.sp_base.value() == 0:
            self.sp_base.blockSignals(True)
            self.sp_base.setValue(50)
            self.sp_base.blockSignals(False)
            notes.append('подложка композита 50 %')
        self.update_si_image()
        self.render_maps()
        self.statusBar().showMessage(f'Изображение {name} ({img.shape[1]}×{img.shape[0]} пикс.)'
                                     + (': ' + '; '.join(notes) if notes else ''))
        return True

    def ask_image_fit(self, name, w, h, nx, ny, taller):
        """'crop', 'stretch' or None (cancel) for an image whose proportions differ from those of the map."""
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Question)
        box.setWindowTitle('Изображение')
        box.setText(f'Пропорции изображения {name} ({w}×{h}) не совпадают с картой ({nx}×{ny}).')
        box.setInformativeText(
            'Изображение накладывается на всю область карты. '
            + ('Если внизу информационная панель SEM — обрежьте снизу. ' if taller
               else 'Обрезать — оставить центральную часть по ширине. ')
            + '«Растянуть» — только если изображение точно покрывает область карты.')
        crop = box.addButton('Обрезать снизу' if taller else 'Обрезать по бокам', QtWidgets.QMessageBox.AcceptRole)
        stretch = box.addButton('Растянуть', QtWidgets.QMessageBox.AcceptRole)
        box.addButton('Отмена', QtWidgets.QMessageBox.RejectRole)
        box.exec_()
        clicked = box.clickedButton()
        return 'crop' if clicked is crop else ('stretch' if clicked is stretch else None)

    def fit_image(self, r, name):
        """Image of core.read_image() cut to the field of view of the map: (image, notes) or None (cancelled).
        The SEM data bar below the scanned area (FEI metadata) is cut off; other proportions: ask. The map
        takes the calibration of the image if it has none; a different field of view is reported."""
        img, notes = r['image'], []
        ny, nx = self.si.shape
        sx, sy = self.si.step
        aspect = nx / ny * (abs(sx / sy) if sx and sy else 1.0)     # width / height of the mapped field
        scan = r.get('scan')
        if scan and scan[0] == img.shape[1] and scan[1] < img.shape[0]:
            notes.append(f'информационная панель SEM ({img.shape[0] - scan[1]} строк) обрезана')
            img = img[:scan[1]]
        h, w = img.shape
        ra = w / h / aspect
        stretched = False
        if abs(ra - 1) > core.IMAGE_ASPECT_TOL:
            choice = self.ask_image_fit(name, w, h, nx, ny, ra < 1)
            if choice is None:
                return None
            if choice == 'stretch':
                stretched = True
                notes.append('растянуто до пропорций карты')
            elif ra < 1:
                img = img[:max(1, int(round(w / aspect)))]
                notes.append(f'обрезано снизу до {w}×{img.shape[0]}')
            else:
                ww = max(1, int(round(h * aspect)))
                img = img[:, (w - ww) // 2:(w - ww) // 2 + ww]
                notes.append(f'обрезано по бокам до {ww}×{h}')
        px = r.get('px')
        if px:
            fov = px * img.shape[1]
            if not self.si.pixel_size:
                if not stretched:
                    self.set_pixel_size(fov / nx)
                    notes.append(f'калибровка карты из изображения ({r.get("calib", "")}): пиксель карты '
                                 f'{si_length(fov / nx)}')
            elif abs(fov / (self.si.pixel_size * nx) - 1) > 0.05:
                fov_map = self.si.pixel_size * nx
                QtWidgets.QMessageBox.warning(
                    self, 'Изображение', f'Ширина поля изображения {name} — {si_length(fov)} ({r.get("calib", "")}), '
                    f'карты — {si_length(fov_map)}. Вероятно, изображение снято с другим увеличением или с другой '
                    f'областью: наложение будет неверным.')
                notes.append(f'поле изображения {si_length(fov)} ≠ поле карты {si_length(fov_map)}')
        return np.ascontiguousarray(img), notes

    def set_pixel_size(self, p):
        """Calibration of a map that had none (taken from an electron image); scale bars are rebuilt."""
        for si in {id(self.si_raw): self.si_raw, id(self.si): self.si}.values():
            si.pixel_size = p
            si.step = (p, p)
        self.update_meta_label()
        if self.maps is not None:
            self._drop_scalebars()

    def base_image(self):
        """(image, is electron image) chosen as the base of the composite: an electron image at its own
        resolution or the X-ray counts on the map grid."""
        i = self.cb_base.currentIndex()
        if 0 <= i < len(self.images):
            return self.images[i]['image'], True
        return self.si.total_image(self.sp_elo.value()), False

    def spectrum_live_time(self):
        """Live time (s) of the spectrum shown (whole map or ROI); None if the file does not give it."""
        t = self.si_raw.meta.get('live_time_s') if self.si_raw is not None else None
        if not t:
            return None
        if self.cb_roi.isChecked():
            y0, y1, x0, x1 = self.roi_bounds()
            t *= (y1 - y0) * (x1 - x0) / float(np.prod(self.si.shape))
        return t

    def update_meta_label(self):
        si = self.si_raw
        ny, nx = si.shape
        px = si.pixel_size
        n = getattr(self, '_counts_total', 0.0)
        self.lbl_meta.setText(f'<b>{si.name}</b><br>{self.meta_text()}<br>Пиксель: {si_length(px) if px else "—"}; '
                              f'всего отсчётов: {n:.3g} ({n / (nx * ny):.0f} на пиксель)')

    def update_survey(self, *args):
        sv = self.si.survey if self.si else None
        if sv is None:
            self.sv_img.clear()
            self.sv_rect.setRect(QtCore.QRectF())
            return
        img = sv['image']
        lo, hi = np.percentile(img, [0.5, 99.8])
        self.sv_img.setImage(img, levels=(lo, hi))
        frame = self.survey_frame()
        self.sv_rect.setRect(QtCore.QRectF(*frame) if frame else QtCore.QRectF())
        self.sv_rect.setToolTip('Положение области SI (по калибровке файла)')
        self.sv_vb.autoRange()

    def survey_frame(self):
        """SI field of view on the survey image: (col, row, width, height), survey pixel edges; None if uncalibrated."""
        sv = self.si.survey
        if not sv.get('dx'):
            return None
        x0, y0, x1, y1 = self.si.si_rect_m()
        dx = sv['dx']
        dy = sv.get('dy') or dx
        sy0 = sv.get('y0') if sv.get('y0') is not None else sv['x0']
        c0, c1 = (x0 - sv['x0']) / dx, (x1 - sv['x0']) / dx + 1
        r0, r1 = (y0 - sy0) / dy, (y1 - sy0) / dy + 1
        return min(c0, c1), min(r0, r1), abs(c1 - c0), abs(r1 - r0)

    def meta_text(self, en=False):
        """Description of the file (metadata) in Russian or English (en: presentation export)."""
        meta, si = self.si.meta, self.si
        parts = []
        if 'no_spc' in meta:
            parts.append(f'Warning: no {meta["no_spc"]}, calibration 10 eV/channel from 0 assumed' if en
                         else f'Внимание: нет {meta["no_spc"]}, калибровка 10 эВ/канал от 0')
        keys = ['Microscope', 'Software', 'High tension', 'Magnification', 'AcquireDate', 'DwellTimePath',
                'Live time', 'Frames', 'EDS detector', 'Take-off angle', 'Pixels', 'Channels', 'Elements (EDAX)']
        parts += [f'{k}: {meta[k]}' for k in keys if k in meta]
        if 'missing_image' in meta:
            parts.append(f'electron image {meta["missing_image"]} (.bmp / .tif) not found' if en
                         else f'электронное изображение {meta["missing_image"]} (.bmp / .tif) — нет в папке')
        if si.dispersion > 1.0:
            parts.append(f'Count dispersion: {si.dispersion:.2f} ' + (
                '(variance / mean, 1 for Poisson counts: part of the events recorded twice - dead-time correction)'
                if en else '(дисперсия / среднее; 1 у пуассоновских отсчётов: часть событий записана дважды — '
                           'поправка на мёртвое время)'))
        return ', '.join(parts)

    def on_roi_toggled(self, on):
        self.update_spectrum()

    def on_roi_changed(self, *args):
        if self.si is None:
            return
        y0, y1, x0, x1 = self.roi_bounds()
        self.lbl_roi.setText(f'ROI: x {x0}…{x1 - 1}, y {y0}…{y1 - 1} ({(x1 - x0) * (y1 - y0)} пикс.)')
        if self.cb_roi.isChecked():
            self.update_spectrum()

    def reset_roi(self, *args):
        if self.si is None:
            return
        ny, nx = self.si.shape
        self.roi.setPos([0, 0], finish=False)
        self.roi.setSize([nx, ny])

    # ------------------------------------------------------------------ spectrum
    def update_spectrum(self, *args):
        if self.si is None:
            return
        self.spec = self.current_spectrum()
        self.recompute_background(quiet=True)
        if self.fit is not None and self.families():
            self._fit(refine=False)
        else:
            self.fit = None
        self.rl = None
        self.redraw_spectrum()
        self.update_fit_table()

    def recompute_background(self, *args, quiet=False):
        if self.si is None or self.spec is None:
            return
        try:
            self.bg = self.background_of(self.spec)
            if self.cb_roi.isChecked():
                self.bg_sum = self.background_of(self.si.sum_spectrum())
            else:
                self.bg_sum = self.bg
        except Exception as e:
            self.statusBar().showMessage(f'Фон: {e}')
            return
        self.draw_windows()
        if not quiet:
            if self.fit is not None and self.families():
                self._fit(refine=False)
                self.update_fit_table()
            self.redraw_spectrum()

    def draw_windows(self):
        for it in self.win_items:
            self.pi.removeItem(it)
        self.win_items = []
        if self.bg_method() != 'poly':
            return
        for a, b in core.parse_windows(self.ed_windows.text()):
            r = pg.LinearRegionItem(values=(a, b), movable=False, brush=pg.mkBrush(95, 191, 95, 25),
                                    pen=pg.mkPen(None))
            self.pi.addItem(r, ignoreBounds=True)
            self.win_items.append(r)

    def add_bg_window(self, *args):
        if not self.bg_region.isVisible():
            self.bt_region.setChecked(True)
            return
        a, b = self.bg_region.getRegion()
        txt = self.ed_windows.text().strip()
        self.ed_windows.setText((txt + ', ' if txt else '') + f'{a:.2f}-{b:.2f}')
        if self.bg_method() != 'poly':
            self.cb_bg_method.setCurrentIndex(BG_METHODS.index('poly'))
        else:
            self.recompute_background()

    def on_range_changed(self, *args):
        lo, hi = self.erange()
        self.range_lo.setValue(lo)
        self.range_hi.setValue(hi)
        self.invalidate_fit()

    def invalidate_fit(self, *args):
        if self.fit is not None and self.si is not None and self.families():
            try:
                self._fit(refine=False)
            except Exception as e:
                self.statusBar().showMessage(f'Фит: {e}')
                self.fit = None
            self.update_fit_table()
        self.redraw_spectrum()

    def on_det_ui_changed(self, *args):
        self.ui_to_det()
        self.invalidate_fit()

    def redraw_spectrum(self, *args):
        if self.si is None or self.spec is None:
            return
        E = self.si.energy
        y = self.spec
        log = self.cb_log.isChecked()
        net = self.cb_net.isChecked()
        if log != self._log_mode:
            self._log_mode = log
            xr = self.pi.vb.viewRange()[0]
            self.pi.setLogMode(y=log)
            self.pi.setXRange(*xr, padding=0)
            self.pi.enableAutoRange(axis='y')
        fit = self.fit
        bgline = fit.background if fit is not None else self.bg
        if bgline is None:
            bgline = np.zeros_like(y)
        sub = bgline if net else 0.0
        floor = 0.5 if log else -np.inf

        def clip(a):
            return np.maximum(a, floor) if log else a

        self.c_data.setData(E, clip(y - sub))
        if net:
            self.c_bg.setData([], [])
        else:
            m = (bgline > 0)
            self.c_bg.setData(E[m], clip(bgline[m]))
        for it in self.comp_items:
            self.pi.removeItem(it)
        self.comp_items = []
        if fit is not None:
            sel = fit.sel
            self.c_model.setData(E[sel], clip((fit.model - sub)[sel]))
            if self.cb_comp.isChecked():
                for f, comp in zip(fit.families, fit.components):
                    m = sel & (comp > 1e-3 * max(comp.max(), 1e-9))
                    if not m.any() or comp.max() <= 0:
                        continue
                    base = 0 if net else bgline
                    yy = (comp + base)[m] if not net else comp[m]
                    it = pg.PlotDataItem(E[m], clip(yy), pen=pg.mkPen(self.color(f.element), width=1))
                    self.pi.addItem(it)
                    self.comp_items.append(it)
                if fit.pileup is not None and fit.pileup.max() > 0:     # sum peaks
                    pu = fit.pileup
                    yy = pu[sel] if net else (pu + bgline)[sel]
                    it = pg.PlotDataItem(E[sel], clip(yy), pen=pg.mkPen('#b0b0b0', width=1, style=Qt.DotLine))
                    self.pi.addItem(it)
                    self.comp_items.append(it)
            r = (y - fit.model) / np.sqrt(np.maximum(fit.model, 1.0))
            self.c_res.setData(E[sel], r[sel])
        else:
            self.c_model.setData([], [])
            self.c_res.setData([], [])
        if self.cb_show_rl.isChecked() and self.rl is not None:
            self.c_rl.setData(E, clip(self.rl + (0 if net else bgline)))
        else:
            self.c_rl.setData([], [])
        lo, hi = self.erange()
        self.range_lo.setValue(lo)
        self.range_hi.setValue(hi)
        self.draw_markers()

    def _local_height(self, e):
        E = self.si.energy
        y = self.spec - (self.fit.background if (self.cb_net.isChecked() and self.fit is not None) else 0)
        h = 1.2 * float(self.det.fwhm(max(e, 0.1)))
        m = (E > e - h) & (E < e + h)
        return float(np.max(y[m])) if m.any() else 1.0

    def _marker_items(self, el, color, style=Qt.SolidLine, labels=None):
        """Line sticks (height ~ relative intensity) of all families of `el`; label specs
        (x, y_view, text, color) are appended to `labels`."""
        items = []
        if self.si is None:
            return items
        E = self.si.energy
        log = self.cb_log.isChecked()
        net = self.cb_net.isChecked()
        bgline = self.fit.background if self.fit is not None else self.bg
        if net or bgline is None:
            bgline = np.zeros_like(E)
        floor = 1.0 if log else 0.0
        for f in core.element_families(el, float(E[0]) + 0.05, float(E[-1]), self.e0()):
            # positions on the recorded axis: calibration and the fitted shift of the family
            main = f.main
            main = (main[0], float(self.det.measured_energy(main[1], f.key)), main[2])
            H = max(self._local_height(main[1]), 2.0)
            hb = max(float(np.interp(main[1], E, bgline)), floor)
            # a family the fit did not detect: short dotted sticks, no label (the local maximum
            # may belong to a neighbour, e.g. Mo M under Nb M)
            detected = self._family_detected(f)
            if not detected:
                H = hb + 0.3 * (H - hb)
            xs, ys = [], []
            for sym, le, w in f.lines:
                le = float(self.det.measured_energy(le, f.key))
                if w < 0.02 or le < E[0] or le > E[-1]:
                    continue
                # stick from the background up to the expected relative height of the line
                b = max(float(np.interp(le, E, bgline)), floor)
                top = b + max((H - hb) * w, 0.02 * max(H - hb, 1.0))
                if log:
                    top = max(top, b * 1.3)
                xs += [le, le]
                ys += [b, top]
            if not xs:
                continue
            it = pg.PlotDataItem(np.array(xs), np.array(ys), connect='pairs',
                                 pen=pg.mkPen(color, width=1.6, style=style if detected else Qt.DotLine))
            items.append(it)
            if labels is not None and detected:
                labels.append((main[1], np.log10(H) if log else H, f'{el} {core.line_label(main[0])}', color))
        return items

    def _family_detected(self, fam):
        """True unless the current fit says the family is below 3 sigma."""
        if self.fit is None:
            return True
        for f, c, e in zip(self.fit.families, self.fit.coef, self.fit.err):
            if f == fam:
                return e <= 0 or c >= 3 * e
        return True

    def _place_labels(self, labels):
        """TextItems for label specs, shifted upwards (in text heights) so that they do not overlap
        on screen; the tallest peaks keep their labels in place."""
        vb = self.pi.vb
        xr = vb.viewRange()[0]
        placed, items = [], []
        fm = QtGui.QFontMetrics(QtGui.QFont())
        h = fm.height() + 1
        for x, yv, text, color in sorted(labels, key=lambda t: -t[1]):
            if not (xr[0] <= x <= xr[1]):
                continue
            p = vb.mapViewToScene(QtCore.QPointF(x, yv))
            w = fm.horizontalAdvance(text) + 6
            k = 0
            while k < 15:
                x0, x1 = p.x() - w / 2, p.x() + w / 2
                y1 = p.y() - 0.15 * h - k * h
                y0 = y1 - h
                if not any(x0 < b[1] and x1 > b[0] and y0 < b[3] and y1 > b[2] for b in placed):
                    break
                k += 1
            placed.append((x0, x1, y0, y1))
            t = pg.TextItem(text, color=color, anchor=(0.5, 1.15 + k))
            t.setPos(x, yv)
            items.append(t)
        return items

    def _schedule_markers(self, *args):
        self._marker_timer.start(120)

    def draw_markers(self):
        for it in self.marker_items:
            self.pi.removeItem(it)
        self.marker_items = []
        if self.si is None or not self.cb_markers.isChecked():
            return
        labels = []
        for el in self.elements:
            for it in self._marker_items(el, self.color(el), labels=labels):
                self.pi.addItem(it, ignoreBounds=True)
                self.marker_items.append(it)
        for it in self._place_labels(labels):
            self.pi.addItem(it, ignoreBounds=True)
            self.marker_items.append(it)

    def show_preview(self, sym):
        for it in self.preview_items:
            self.pi.removeItem(it)
        self.preview_items = []
        if sym is None or self.si is None:
            return
        labels = []
        items = self._marker_items(sym, '#ffffff', Qt.DashLine, labels=labels)
        for it in items + self._place_labels(labels):
            self.pi.addItem(it, ignoreBounds=True)
            self.preview_items.append(it)

    def on_mouse_moved(self, evt):
        if self.si is None:
            return
        pos = evt[0]
        vb = self.pi.vb
        if not vb.sceneBoundingRect().contains(pos):
            return
        e = vb.mapSceneToView(pos).x()
        E = self.si.energy
        i = int(np.clip(np.searchsorted(E, e), 0, len(E) - 1))
        tol = max(float(self.det.fwhm(max(e, 0.1))) * 0.6, 0.02)
        near = core.lines_near(e, tol, min_weight=0.1, e0_kev=self.e0())
        sel = [t for t in near if t[1] in self.elements]
        rest = [t for t in near if t[1] not in self.elements]
        txt = ', '.join(f'{t[1]} {core.line_label(t[3])} {t[4]:.3f}' for t in (sel + rest)[:6])
        self.lbl_cursor.setText(f'E = {e:7.3f} кэВ   отсчёты = {self.spec[i]:9.0f}   FWHM = {tol / 0.6 * 1000:4.0f} эВ   │ {txt}')

    def on_plot_clicked(self, ev):
        if self.si is None or ev.button() != Qt.LeftButton:
            return
        vb = self.pi.vb
        pos = ev.scenePos()
        if not vb.sceneBoundingRect().contains(pos):
            return
        e = vb.mapSceneToView(pos).x()
        self._near_e = e
        self.cursor_line.setValue(e)
        self.cursor_line.setVisible(True)
        self.update_near_lines()

    def update_near_lines(self, *args):
        self.near_tree.clear()
        e = self._near_e
        if e is None:
            return
        tol = self.sp_tol.value() * float(self.det.fwhm(max(e, 0.1)))
        self.lbl_near.setText(f'E = {e:.3f} кэВ, ±{tol * 1000:.0f} эВ')
        for de, el, fam, sym, le, w in core.lines_near(e, tol, min_weight=0.02, e0_kev=self.e0())[:60]:
            it = QtWidgets.QTreeWidgetItem([f'{el} {core.line_label(sym)}', f'{le:.4f}', f'{de * 1000:+.0f}',
                                            f'{w:.2f}', el])
            if el in self.elements:
                it.setForeground(0, QtGui.QBrush(self.color(el)))
            if ELEMENTS[el][0] < core.FAMILY_MIN_Z[fam]:
                it.setForeground(3, QtGui.QBrush(QtGui.QColor('#777')))
                it.setToolTip(0, 'Серия обычно не используется для этого элемента')
            self.near_tree.addTopLevelItem(it)
        for c in range(5):
            self.near_tree.resizeColumnToContents(c)

    # ------------------------------------------------------------------ elements
    def on_element_toggled(self, sym, on):
        if on:
            self.add_element(sym)
        else:
            self.remove_element(sym)

    def add_element(self, sym):
        if sym not in ELEMENTS or sym in self.elements:
            return
        lo, hi = self.erange()
        if self.si is not None and not core.element_families(sym, lo, hi, self.e0()):
            self.statusBar().showMessage(f'{sym}: нет линий в диапазоне {lo:.2f}–{hi:.2f} кэВ')
        self.elements.append(sym)
        self.ptable.set_checked(sym, True, self.color(sym).name())
        self.update_id_tree()
        self.invalidate_fit()

    def remove_element(self, sym):
        if sym in self.elements:
            self.elements.remove(sym)
        self.ptable.set_checked(sym, False)
        self.update_id_tree()
        if self.fit is not None and not self.families():
            self.fit = None
            self.update_fit_table()
        self.invalidate_fit()

    def remove_selected_element(self, *args):
        it = self.id_tree.currentItem()
        if it is not None:
            self.remove_element(it.text(0))

    def file_elements(self):
        """Elements identified by the acquisition software (EDAX .spc), [] if none."""
        txt = self.si.meta.get('Elements (EDAX)', '') if self.si is not None else ''
        return [s.strip() for s in txt.split(',') if s.strip() in ELEMENTS]

    def elements_from_file(self, *args):
        els = self.file_elements()
        if not els:
            return
        if not self.cb_id_keep.isChecked():
            self.clear_elements()
        for el in els:
            self.add_element(el)
        self.run_fit()

    def clear_elements(self, *args):
        for el in list(self.elements):
            self.ptable.set_checked(el, False)
        self.elements = []
        self.id_info, self.unidentified = {}, []
        self.fit = None
        self.update_id_tree()
        self.update_fit_table()
        self.redraw_spectrum()

    def on_id_item_clicked(self, item, col):
        info = self.id_info.get(item.text(0))
        if info:
            self._near_e = info['energy']
            self.cursor_line.setValue(info['energy'])
            self.cursor_line.setVisible(True)
            self.update_near_lines()

    def update_id_tree(self):
        self.id_tree.clear()
        lo, hi = self.erange()
        for el in self.elements:
            fams = ', '.join(f.family for f in core.element_families(el, lo, hi, self.e0()))
            info = self.id_info.get(el)
            if info:
                row = [el, fams, f"{info['line']} ({info['energy']:.2f})", f"{info['sigma']:.0f}",
                       info.get('confidence', '')]
            else:
                row = [el, fams, 'вручную', '', '']
            it = QtWidgets.QTreeWidgetItem(row)
            it.setForeground(0, QtGui.QBrush(self.color(el)))
            if info and info.get('confidence') == 'возможно':
                it.setForeground(4, QtGui.QBrush(QtGui.QColor('#ffb43c')))
                if info.get('missing'):
                    it.setToolTip(4, 'Не видна серия: ' + ', '.join(info['missing']))
            self.id_tree.addTopLevelItem(it)
        for c in range(5):
            self.id_tree.resizeColumnToContents(c)
        if self.unidentified:
            self.lbl_unid.setText('Неопознанные пики (кэВ): ' + ', '.join(f'{e:.2f}' for e in sorted(self.unidentified)))
        else:
            self.lbl_unid.setText('')

    @guarded
    def run_auto_id(self):
        if self.si is None:
            return
        self.ui_to_det()
        y = self.spec
        bg = self.bg if self.bg is not None else core.snip_background(self.si.energy, y, self.det, 2.0)
        start = list(self.elements) if self.cb_id_keep.isChecked() else []
        log = []
        res = core.auto_identify(self.si.energy, y, self.det, bg, erange=self.erange(), e0_kev=self.e0(),
                                 threshold=self.sp_thr.value(), start=start, noise=self.noise_peak(),
                                 log=log.append, live_time=self.spectrum_live_time(), pileup=True)
        if not self.cb_id_keep.isChecked():
            for el in list(self.elements):
                self.ptable.set_checked(el, False)
            self.elements = []
            self.id_info = {}
        for d in res.details:
            self.id_info[d['element']] = d
        self.unidentified = res.unidentified
        for el in res.elements:
            if el not in self.elements:
                self.elements.append(el)
            self.ptable.set_checked(el, True, self.color(el).name())
        self.det = res.det
        self.det_to_ui()
        self.update_id_tree()
        self._fit(refine=True)
        self.update_fit_table()
        self.redraw_spectrum()
        self.statusBar().showMessage(f'Найдено: {", ".join(res.elements)}  (журнал: {len(log)} шагов)')
        print('\n'.join(log))

    # ------------------------------------------------------------------ fitting
    def _fit(self, refine):
        fams = self.families()
        if not fams:
            self.fit = None
            return
        self.ui_to_det()
        nlls = ()
        if refine:
            if self.cb_shape.currentIndex() == 1:
                nlls = tuple(k for k, cb in (('shift', self.cb_nlls_shift), ('width', self.cb_nlls_width),
                                             ('ratio', self.cb_nlls_ratio)) if cb.isChecked())
            else:       # shapes from the detector model only
                self.det = core.strip_nlls(self.det)
        bgshape = self.bg if self.bg is not None else None
        # SEM spectra with millions of counts: sum peaks (C+O 0.80 keV, O+Na 1.57, Ti+C 4.79...) in the model
        pile = (core.pileup_shape(self.si.energy, self.spec, self.erange()[0])
                if core.high_statistics(self.spec) else None)
        self.fit = core.fit_spectrum(self.si.energy, self.spec, fams, self.det, bgshape, self.sp_nbg.value(),
                                     self.erange(), refine_calib=refine and self.cb_ref_cal.isChecked(),
                                     refine_res=refine and self.cb_ref_res.isChecked(),
                                     refine_tail=refine and self.cb_ref_tail.isChecked(),
                                     noise=self.noise_peak(), nlls=nlls, dispersion=self.si_raw.dispersion,
                                     pileup=pile)
        if refine:
            self.det = self.fit.det
            self.det_to_ui()

    @guarded
    def run_fit(self):
        if self.si is None:
            return
        if not self.families():
            QtWidgets.QMessageBox.information(self, 'Фит', 'Выберите элементы (вкладка «Элементы»).')
            return
        self._fit(refine=True)
        self.update_fit_table()
        self.redraw_spectrum()
        self.statusBar().showMessage(f'Фит выполнен: χ²/ν = {self.fit.chi2_red:.2f}')

    def update_fit_table(self):
        t = self.fit_table
        t.setRowCount(0)
        if self.si is None:
            self.lbl_chi2.setText('')
            return
        lo, hi = self.erange()
        rows = []
        fit_map = {}
        if self.fit is not None:
            for f, c, e in zip(self.fit.families, self.fit.coef, self.fit.err):
                fit_map[(f.element, f.family)] = (c, e)
        for el in self.elements:
            for f in core.element_families(el, lo, hi, self.e0()):
                rows.append((f, fit_map.get((el, f.family))))
        t.setRowCount(len(rows))
        for i, (f, ce) in enumerate(rows):
            cb = QtWidgets.QCheckBox()
            cb.setChecked(self.fam_enabled.get((f.element, f.family), True))
            cb.toggled.connect(lambda on, k=(f.element, f.family): self.on_family_toggled(k, on))
            cell = QtWidgets.QWidget()
            lay = QtWidgets.QHBoxLayout(cell)
            lay.addWidget(cb)
            lay.setAlignment(Qt.AlignCenter)
            lay.setContentsMargins(0, 0, 0, 0)
            t.setCellWidget(i, 0, cell)
            main = f.main
            sh = self.det.shifts.get(f.key)
            kw = self.det.widths.get(f.key)
            vals = [f.element, f.family, core.line_label(main[0]), f'{main[1]:.3f}',
                    f'{round(sh * 1000):+d}' if sh is not None else '', f'×{kw:.2f}' if kw is not None else '']
            if ce is not None:
                c, e = ce
                vals += [f'{c:.0f}', f'{e:.0f}', f'{c / e:.1f}' if e > 0 else '']
            else:
                vals += ['', '', '']
            vals.append(ratios_text(self.det, f.key))
            for j, v in enumerate(vals, start=1):
                it = QtWidgets.QTableWidgetItem(v)
                if 4 <= j <= 9:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if j == 1:
                    it.setForeground(QtGui.QBrush(self.color(f.element)))
                if ce is not None and ce[1] > 0 and ce[0] < 3 * ce[1]:
                    it.setForeground(QtGui.QBrush(QtGui.QColor('#777')))
                    it.setToolTip('Ниже предела обнаружения (< 3σ)')
                t.setItem(i, j, it)
        if self.fit is not None:
            d = self.fit.det
            note = ('<br><span style="color:#ff9b2d;">Данные PCA: шум удалён, поэтому χ² и ± σ занижены '
                    '(для погрешностей фитируйте исходные данные).</span>' if self.pca_k() else '')
            thr = f', порог = {d.thr * 1000:.0f} эВ' if d.thr > 0 else ''
            shape = (f'; форма пиков: NLLS ({len(d.widths) or len(d.ratios)} серий)' if d.widths or d.ratios
                     else '; форма пиков: модель детектора')
            pu = self.fit.pileup
            if pu is not None:
                n_pu = float(pu[self.fit.sel].sum())
                shape += (f'; пики суммирования (pile-up): {n_pu:.3g} отсч. '
                          f'({n_pu / max(float(self.spec[self.fit.sel].sum()), 1.0) * 100:.2f} %)')
            self.lbl_chi2.setText(f'χ²/ν = {self.fit.chi2_red:.2f};  FWHM(Mn Kα) = {d.fwhm_mn * 1000:.1f} эВ, '
                                  f'сдвиг = {d.offset * 1000:+.1f} эВ, наклон = {d.gain * 100:+.3f} %{thr}{shape}' + note)
        else:
            self.lbl_chi2.setText('Фит не выполнен')

    def on_family_toggled(self, key, on):
        self.fam_enabled[key] = on
        self.invalidate_fit()
        self.update_fit_table()

    @guarded
    def run_rl(self):
        if self.si is None:
            return
        self.ui_to_det()
        bg = self.fit.background if self.fit is not None else self.bg
        self.rl = core.richardson_lucy(self.si.energy, self.spec, self.det, bg, self.sp_rl.value())
        self.cb_show_rl.setChecked(True)
        self.redraw_spectrum()

    # ------------------------------------------------------------------ maps
    @guarded
    def run_maps(self):
        if self.si is None:
            return
        if not self.families():
            QtWidgets.QMessageBox.information(self, 'Карты', 'Выберите элементы (вкладка «Элементы»).')
            return
        self.main_tabs.setCurrentIndex(1)
        QtWidgets.QApplication.processEvents()
        self._compute_maps()

    def _maps_of(self, si, fams, method, weight, key=None):
        """compute_maps of the spectrum image `si` with the settings of the GUI: (maps, phase labels or
        None, (background shape, n_bg) of the fit). key: cache the result under this key (maps of the PCA
        input, see _compute_maps)."""
        bg, n_bg, phases = self.bg_sum, self.sp_nbg.value(), None
        # phases need a map (not a single spectrum / a few pixels)
        if self.cb_phase_bg.isChecked() and self.bg_method() in ('snip', 'minima') and np.prod(si.shape) >= 100:
            labels = self.phase_labels()
            if labels.max() > 0:
                phases = labels
                bg = core.phase_backgrounds(si, labels, self.det, estimator=lambda e, y: self.background_of(y))
                n_bg = 1
        if key is not None:
            key = key + (hash(np.asarray(bg).tobytes()) if bg is not None else None, n_bg)
            if self._raw_cache is not None and self._raw_cache[0] == key:
                return self._raw_cache[1], phases, (bg, n_bg)
        maps = core.compute_maps(si, fams, self.det, bg, n_bg, self.erange(), method=method, weight_spectrum=weight,
                                 noise=self.noise_peak(), tie_low=self.cb_tie_low.isChecked())
        if key is not None:
            self._raw_cache = (key, maps)
        return maps, phases, (bg, n_bg)

    def _compute_maps(self):
        fams = self.families()
        self.ui_to_det()
        method = ['mlls', 'window_net', 'window'][self.cb_map_method.currentIndex()]
        weight = None   # weights of the per-pixel least squares: sum-spectrum model if available
        if self.fit is not None and not self.cb_roi.isChecked():
            weight = np.maximum(self.fit.model, 1.0)
        self.maps, self.map_phases, bg_model = self._maps_of(self.si, fams, method, weight)
        # PCA data: the maps of the PCA input (preprocessed data) as well (cached: k changes do not recompute
        # them), to check how much of their spatial structure the PCA maps keep
        self.maps_raw = None
        check_si = self.si
        if self.pca_k():
            key = (id(self.si_pre), tuple((f.element, f.family) for f in fams), method, repr(self.det),
                   self.erange(), self.cb_tie_low.isChecked(), repr(self.noise_peak()),
                   hash(weight.tobytes()) if weight is not None else None)
            self.maps_raw, _, bg_model = self._maps_of(self.si_pre, fams, method, weight, key=key)
            check_si = self.si_pre
        self._check_cache = {}
        self._struct_cache = {}
        self.map_checks = self._check_maps(check_si, fams, weight, bg_model) if method == 'mlls' else {}
        self.map_views = []
        self._blem_cache = {}
        self.build_map_table()
        self.render_maps()
        self._fill_image_boxes()
        self.update_si_image()
        pca =f', данные PCA ({self.pca_desc()})' if self.pca_k() else ''
        self.statusBar().showMessage(f'Карты построены ({self.cb_map_method.currentText()}{pca}): {len(fams)} серий')

    def _check_maps(self, si, fams, weight, bg_model):
        """core.map_background_check of the maps: {family key: MapCheck}. si: the measured data - the PCA
        input when the maps are of PCA data (the background under a line is a property of the measured
        spectra, their counts give the statistical errors; the verdict does not change with the PCA).
        bg_model: (shape, n_bg) of the maps of si. Background models: that of the maps, the same x a
        polynomial 2 and 4 orders higher and, for SNIP, a 1.5x wider SNIP window. Phases: those of the
        phase backgrounds (computed for the check if those are off)."""
        if np.prod(si.shape) < 100:
            return {}
        labels = self.phase_labels()
        if labels.max() == 0:
            return {}
        spectra = core.phase_spectra(si, labels)
        bg, n_bg = bg_model
        models = [('модель карт', bg, n_bg), ('полином фона +2', bg, n_bg + 2), ('полином фона +4', bg, n_bg + 4)]
        if self.bg_method() == 'snip':
            width = 1.5 * self.sp_snip.value()
            est = lambda e, y: core.snip_background(e, y, self.det, width)
            wide = (core.phase_backgrounds(si, labels, self.det, estimator=est, spectra=spectra)
                    if self.map_phases is not None else est(si.energy, si.sum_spectrum()))
            models.append(('SNIP ×1.5', wide, n_bg))
        checks = core.map_background_check(si, fams, self.det, labels, models, self.erange(),
                                           weight if weight is not None else si.sum_spectrum(), self.noise_peak(),
                                           self.cb_tie_low.isChecked(), spectra=spectra)
        return {f.key: c for f, c in zip(fams, checks) if c is not None}

    def phase_labels(self):
        """Phase map of the loaded data for the phase backgrounds (cached; the PCA of the PCA tab
        is reused when it covers the same energy range)."""
        key = (id(self.si_raw), self.erange())
        if self._phases is None or self._phases[0] != key:
            p = self.pca
            if p is None or not p.poisson or getattr(self, '_pca_range', None) != self.erange():
                p = core.pca_decompose(self.si_raw, self.erange())
            self._phases = (key, core.phase_labels(p))
        return self._phases[1]

    # ------------------------------------------------------------------ PCA
    def pca_model(self):
        """Components of the model set in the PCA tab: chosen by hand, or the first k without the noise ones
        if «без шумовых»."""
        if self.pca_manual is not None:
            return self.pca_manual
        return core.pca_components(self.pca, self.sp_pca_k.value(), self.cb_pca_skip.isChecked())

    def set_pca_model(self, comps):
        """Model of the components `comps` chosen by hand (component 1, the mean spectrum, is always kept);
        k is set to the last of them, a set equal to the automatic choice for that k is not 'manual'."""
        comps = tuple(sorted({0} | {int(i) for i in comps if 0 <= i < len(self.pca.factors)}))
        self.sp_pca_k.blockSignals(True)
        self.sp_pca_k.setValue(comps[-1] + 1)
        self.sp_pca_k.blockSignals(False)
        auto = core.pca_components(self.pca, self.sp_pca_k.value(), self.cb_pca_skip.isChecked())
        self.pca_manual = None if comps == auto else comps
        self.render_pca()
        if self.cb_pca_use.isChecked():
            self._pca_apply_timer.start(300)
        self.statusBar().showMessage(f'Модель PCA: компоненты {ranges_text(comps)}'
                                     + (' (выбраны вручную)' if self.pca_manual else ''))

    def toggle_pca_component(self, i):
        if self.pca is None or not 0 <= i < len(self.pca.factors):
            return
        if i == 0:
            self.statusBar().showMessage('Компонента 1 (средний спектр) входит в модель всегда')
            return
        self.set_pca_model(set(self.pca_model()) ^ {i})

    def on_pca_comps_edited(self):
        if self.pca is None:
            return
        try:
            comps = parse_ranges(self.ed_pca_comps.text())
        except ValueError as e:
            self.statusBar().showMessage(f'Номера компонент не распознаны («{e}»); пример: 1-3, 5')
            comps = None
        n = len(self.pca.factors)
        if not comps or ({0} | {i for i in comps if i < n}) == set(self.pca_model()):
            self.ed_pca_comps.setText(ranges_text(self.pca_model()))     # nothing changed / not understood
            return
        self.set_pca_model(comps)
        if max(comps) >= n:
            self.statusBar().showMessage(self.statusBar().currentMessage() + f'; компонент всего {n}, номера '
                                         f'больше отброшены')

    def on_scree_clicked(self, item, points, *args):
        if len(points):
            self.toggle_pca_component(int(round(points[0].pos().x())) - 1)

    def pca_auto(self):
        """«Авто»: the automatic choice, k = the last signal component."""
        if self.pca is None:
            return
        self.sp_pca_k.blockSignals(True)
        self.sp_pca_k.setValue(self.pca.suggested)
        self.sp_pca_k.blockSignals(False)
        self.on_pca_k_changed()

    def pca_local_used(self):
        """The data in use are rebuilt by the global model complemented by the local PCA."""
        return self.pca_local is not None and self.cb_pca_use.isChecked()

    def local_added(self):
        """Local directions added to the global model in every tile (core.local_pca_added), cached."""
        lp = self.pca_local
        key = (id(lp), lp.skip_noise, self.pca_model())
        if getattr(self, '_local_added', (None,))[0] != key:
            self._local_added = (key, core.local_pca_added(lp, self.pca, self.pca_model()))
        return self._local_added[1]

    def pca_comps(self):
        """Components of the global PCA the data in use are rebuilt from (() = the loaded data); with the
        local PCA the tiles add their own directions to them."""
        return self.pca_model() if self.pca is not None and self.cb_pca_use.isChecked() else ()

    def pca_k(self):
        """Number of global PCA components of the data in use (0: the loaded data)."""
        return len(self.pca_comps())

    def pca_desc(self, en=False):
        """'4 комп.' or '4 комп.: 1–3, 5' for the data in use ('' = the loaded data); en: '4 comp.';
        with the local PCA: '4 комп. + локальный PCA (плитки 64 px, добавлено в 2 из 18)'."""
        c = self.pca_comps()
        if not c:
            return ''
        unit = 'comp.' if en else 'комп.'
        out = f'{len(c)} {unit}' if c == tuple(range(len(c))) else f'{len(c)} {unit}: {ranges_text(c)}'
        if self.pca_local_used():
            a = self.local_added()
            out += (f' + local PCA ({self.pca_local.tile}-px tiles, added in {int((a > 0).sum())} of {len(a)})' if en
                    else f' + локальный PCA (плитки {self.pca_local.tile} px, добавлено в {int((a > 0).sum())} '
                         f'из {len(a)})')
        return out

    def reset_pca(self):
        self.pca = None
        # images «Плохие пиксели» / «Локальный PCA: компонент» leave the list
        listed = self.pre is not None and self.pre.bad_pixels.any() or self.pca_local is not None
        self.pca_local = None
        self.si_pre = self.pre = None
        self.lbl_pre.setText('')
        self.lbl_pre.setToolTip('')
        if listed:
            self._fill_image_boxes()
            self.update_si_image()
        self.pca_manual = None
        self._pca_k_applied = ()
        self.cb_pca_use.blockSignals(True)
        self.cb_pca_use.setChecked(False)
        self.cb_pca_use.blockSignals(False)
        self.cb_pca_use.setEnabled(False)
        self.sp_pca_k.setEnabled(False)
        self.ed_pca_comps.setEnabled(False)
        self.ed_pca_comps.setText('')
        self.scree_pts.setData([], [])
        self.load_curve.setData([], [])
        for it in getattr(self, '_pca_marks', []):
            self.load_pw.removeItem(it)
        self._pca_marks = []
        self.pca_glw.clear()
        self.pca_glw.setMinimumHeight(0)
        self.pca_views = []
        self._pca_ncols = 0
        self.lbl_pca.setText('PCA не вычислен.')

    def show_preprocessing(self):
        """Report of the preprocessing of the PCA input: label above the plots (shifts of the row blocks in its
        tooltip), replaced pixels as the image «Плохие пиксели»."""
        r, E = self.pre, self.si_raw.energy
        if not self.cb_pre_ch.isChecked():
            ch = 'не проверялись'
        elif len(r.dead_channels):
            ch = (f'<b>{len(r.dead_channels)}</b> ({", ".join(f"{E[j]:.3f}" for j in r.dead_channels[:8])}'
                  f'{" …" if len(r.dead_channels) > 8 else ""} кэВ) — заменены интерполяцией по соседним каналам')
        else:
            ch = 'нет'
        if r.empty_channels:
            ch += f', пустых каналов в диапазоне {r.empty_channels} (в PCA не входят)'
        d, tip = r.drift, ''
        if not self.cb_pre_drift.isChecked():
            dr = 'не проверялся'
        elif d is None:
            dr = 'не измерен (меньше 6 строк или нет сильных пиков)'
        else:
            ok = np.isfinite(d['shift'])
            s, err = d['shift'][ok] * 1000, float(np.median(d['err'][ok])) * 1000
            where = f'{int(ok.sum())} блоков строк, пики {", ".join(f"{e:.2f}" for e in d["peaks"])} кэВ'
            if r.drift_applied:
                dr = f'<b>исправлен</b>: от {s.min():+.1f} до {s.max():+.1f} эВ (σ ≈ {err:.1f} эВ; {where})'
            elif d['p'] >= core.PRE_DRIFT_P:
                dr = f'не обнаружен: сдвиги до {np.abs(s).max():.1f} эВ при σ ≈ {err:.1f} эВ (p = {d["p"]:.2g}; {where})'
            elif d['agree'] >= core.PRE_DRIFT_AGREE:
                dr = (f'сдвиги до {np.abs(s).max():.1f} эВ значимы, но пики не согласуются (χ²/ν = {d["agree"]:.1f}): '
                      f'скорее разный состав, чем дрейф — не исправлялся ({where})')
            else:
                dr = (f'значим, но меньше {core.PRE_DRIFT_MIN:g} канала ({np.abs(s).max():.1f} эВ) — не исправлялся '
                      f'({where})')
            tip = 'Сдвиг шкалы энергии по блокам строк развёртки (относительно среднего):\n' + '\n'.join(
                f'строка ≈ {row:.0f}: {v * 1000:+.1f} ± {e * 1000:.1f} эВ'
                for row, v, e in zip(d['rows'], d['shift'], d['err']) if np.isfinite(v))
        if not self.cb_pre_px.isChecked():
            px = 'не проверялись'
        else:
            by, bx = np.nonzero(r.bad_pixels)
            if len(by):
                lst = ', '.join(f'({x}, {y}) {"горячий" if r.pixel_z[y, x] > 0 else "тёмный"}'
                                for y, x in list(zip(by, bx))[:8])
                px = (f'<b>{len(by)}</b> (x, y): {lst}{" …" if len(by) > 8 else ""} — заменены спектром соседа '
                      f'(вкладка «Спектр», «Изображение» → «Плохие пиксели»)')
            else:
                px = f'нет (порог {self.sp_pre_sigma.value():g}σ)'
        self.lbl_pre.setText(f'<b>Предобработка</b> (вход PCA): мёртвые каналы — {ch}; дрейф энергии — {dr}; '
                             f'плохие пиксели — {px}.' + ('' if r.changed else ' Данные не изменены.'))
        self.lbl_pre.setToolTip(tip)
        self._fill_image_boxes()
        self.update_si_image()

    @guarded
    def export_pca(self):
        if self.pca is None:
            QtWidgets.QMessageBox.information(self, 'Экспорт PCA', 'Сначала вычислите PCA.')
            return
        d = QtWidgets.QFileDialog.getExistingDirectory(self, 'Папка для результатов PCA',
                                                       os.path.dirname(self.si_raw.path))
        if not d:
            return
        out = os.path.join(d, f'{self.si_raw.name}_pca')
        os.makedirs(out, exist_ok=True)
        self.export_pca_to(out)
        self.statusBar().showMessage(f'PCA сохранён в {out}')

    def export_pca_to(self, out):
        """eigenvalues.csv, loadings.csv (spectra, count scale), scores.tif (maps, float32 stack) - the data
        ~ sum_i scores_i(y, x) loadings_i(E) - and pca.npz: all of it with the weights and the preprocessing."""
        import tifffile
        p, r, E = self.pca, self.pre, self.si_raw.energy
        n, used = len(p.factors), set(self.pca_model())
        cum = np.cumsum(p.eigvals) / p.total if p.total > 0 else np.zeros(len(p.eigvals))
        with open(os.path.join(out, 'eigenvalues.csv'), 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['component', 'eigenvalue', 'eigenvalue_over_noise_edge', 'explained_fraction',
                        'cumulative_fraction', 'map_autocorrelation', 'signal', 'in_model'])
            for i, lam in enumerate(p.eigvals):
                kept = i < n
                w.writerow([i + 1, f'{lam:.6g}', f'{lam / p.noise_edge:.6g}',
                            f'{lam / p.total:.6g}' if p.total > 0 else '', f'{cum[i]:.6g}',
                            f'{p.autocorr[i]:.4f}' if kept else '', int(p.signal[i]) if kept else '', int(i in used)])
        loadings = np.stack([p.spectrum(i) for i in range(n)])
        Es = E[p.sel]
        with open(os.path.join(out, 'loadings.csv'), 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['energy_keV'] + [f'loading_{i + 1}' for i in range(n)])
            for j in range(len(Es)):
                w.writerow([f'{Es[j]:.5f}'] + [f'{v:.6g}' for v in loadings[:, j]])
        scores = np.stack([p.score_map(i) for i in range(n)]).astype(np.float32)
        tifffile.imwrite(os.path.join(out, 'scores.tif'), scores)
        dr = r.drift or {}
        lp = self.pca_local
        local = {} if lp is None else dict(     # tiles of the local PCA: (y0, x0, h, w), signal / added components
            local_tile=lp.tile, local_tiles=np.array([t[:4] for t in lp.tiles], dtype=int),
            local_signal_components=lp.n_comps(), local_skip_noise=lp.skip_noise,
            local_added=self.local_added(), local_added_map=lp.tile_map(self.local_added()).astype(np.float32))
        np.savez_compressed(
            os.path.join(out, 'pca.npz'), source=self.si_raw.path, erange_keV=np.array(self._pca_range),
            poisson=p.poisson, binning=p.binning, eigenvalues=p.eigvals, noise_edge=p.noise_edge, total=p.total, energy_keV=Es,
            loadings=loadings, scores=scores, loadings_weighted=p.factors, channel_weights=p.col_w,
            pixel_weights=p.row_w.reshape(p.shape), map_autocorrelation=p.autocorr, signal=p.signal,
            suggested=p.suggested, model_components=np.array(sorted(used)) + 1,
            dead_channels_keV=E[r.dead_channels], empty_channels=r.empty_channels,
            drift_rows=np.asarray(dr.get('rows', [])), drift_keV=np.asarray(dr.get('shift', [])),
            drift_err_keV=np.asarray(dr.get('err', [])), drift_peaks_keV=np.asarray(dr.get('peaks', ())),
            drift_p=dr.get('p', np.nan), drift_applied=r.drift_applied, bad_pixels=r.bad_pixels,
            bad_pixel_sigma=r.pixel_z, **local)

    @guarded
    def run_pca(self):
        if self.si_raw is None:
            return
        # raw cube -> preprocessing -> decomposition (eigenvalues, loadings, scores)
        self._pca_range = self.erange()
        self.ui_to_det()
        self.si_pre, self.pre = core.preprocess(
            self.si_raw, self.det, self._pca_range, channels=self.cb_pre_ch.isChecked(),
            drift=self.cb_pre_drift.isChecked(), pixels=self.cb_pre_px.isChecked(),
            pixel_sigma=self.sp_pre_sigma.value())
        self.show_preprocessing()
        binning = (1, 2, 4)[self.cb_pca_bin.currentIndex()]
        self.pca = core.pca_decompose(self.si_pre, self._pca_range, poisson=self.cb_pca_poisson.isChecked(),
                                      binning=binning)
        p = self.pca
        self.pca_local = None
        if self.cb_pca_local.isChecked():
            def progress(f):
                self.statusBar().showMessage(f'Локальный PCA: {f * 100:.0f} % плиток')
                QtWidgets.QApplication.processEvents()
            self.pca_local = core.local_pca_decompose(
                self.si_pre, self._pca_range, self.sp_pca_tile.value(), poisson=self.cb_pca_poisson.isChecked(),
                binning=binning, skip_noise=self.cb_pca_skip.isChecked(), progress=progress)
        self._fill_image_boxes()
        self.pca_manual = None
        self.sp_pca_k.blockSignals(True)
        self.sp_pca_k.setMaximum(len(p.factors))
        self.sp_pca_k.setValue(p.suggested)
        self.sp_pca_k.blockSignals(False)
        self.sp_pca_k.setEnabled(True)
        self.ed_pca_comps.setEnabled(True)
        self.cb_pca_use.setEnabled(True)
        self.pca_sel = 0
        self.main_tabs.setCurrentIndex(2)
        QtWidgets.QApplication.processEvents()
        self.pca_views = []
        self.render_pca()
        if self.cb_pca_use.isChecked():     # new decomposition: rebuild the data in use
            self._pca_k_applied = None
            self.apply_pca()
        msg = f'PCA: {int(p.signal.sum())} сигнальных компонент'
        if self.pca_local is not None:
            a = self.local_added()
            msg += f'; локальный PCA: {len(a)} плиток, добавлено в {int((a > 0).sum())}'
        self.statusBar().showMessage(msg)

    def on_pca_k_changed(self, *args):
        """k or «без шумовых» changed: back to the automatic choice of the components."""
        if self.pca is None:
            return
        if self.pca_local is not None:
            self.pca_local.skip_noise = self.cb_pca_skip.isChecked()
        self.pca_manual = None
        self.render_pca()
        if self.cb_pca_use.isChecked():
            self._pca_apply_timer.start(300)

    @guarded
    def apply_pca(self):
        """Switch the data in use between the loaded spectrum image and the PCA reconstruction of the
        preprocessed one; spectra, fit and maps are recomputed."""
        if self.si_raw is None:
            return
        comps = self.pca_comps()
        lp = self.pca_local if self.pca_local_used() else None
        key = ('local', id(lp), lp.skip_noise, comps) if lp is not None else comps
        if key == self._pca_k_applied:
            return
        if lp is not None:
            self.si = replace(self.si_pre, data=core.local_pca_reconstruct(self.si_pre, lp, self.pca, comps))
        else:
            self.si = (replace(self.si_pre, data=core.pca_reconstruct(self.si_pre, self.pca, comps)) if comps
                       else self.si_raw)
        self._pca_k_applied = key
        self.update_si_image()
        self.update_spectrum()
        if self.maps is not None and self.families():
            self._compute_maps()
        base = self.status_info.text().split('  |  ')[0]
        self.status_info.setText(base + (f'  |  PCA: {self.pca_desc()}' if comps else ''))
        self.statusBar().showMessage(f'Данные: восстановлены по компонентам PCA ({self.pca_desc()})' if comps
                                     else 'Данные: исходные')

    def render_pca(self, *args):
        p = self.pca
        if p is None:
            return
        k = self.sp_pca_k.value()
        used = set(self.pca_model())
        rel = p.eigvals / p.noise_edge
        n_all, nf = len(rel), len(p.factors)
        x = np.arange(1, n_all + 1)
        sig = np.zeros(n_all, bool)       # eigenvalues beyond the kept factors: noise
        sig[:nf] = p.signal[:nf] if p.signal is not None else True
        # green: signal in the model, orange x: noise in the model, hollow green: signal left out, grey x: noise
        sym, brush, pen, size = [], [], [], []
        for i in range(n_all):
            inm = i in used
            sym.append('o' if sig[i] else 'x')
            col = '#3cff5a' if sig[i] else ('#ff9b2d' if inm else '#777')
            brush.append(pg.mkBrush(col if inm or not sig[i] else (0, 0, 0, 0)))
            pen.append(pg.mkPen('#3cff5a', width=1.5) if sig[i] and not inm else pg.mkPen(None))
            size.append(9 if inm or sig[i] else 7)
        self.scree_pts.setData(x, rel, symbol=sym, symbolBrush=brush, symbolPen=pen, symbolSize=size)
        self.scree_cut.setValue(k + 0.5)
        self.scree_pw.getPlotItem().setXRange(0.5, min(len(rel), 60) + 0.5, padding=0)

        # maps of all kept components; tiles are sized so that the first 12 fill the visible area
        n = len(p.factors)
        n_fit = min(n, 12)
        ny, nx = p.shape
        sa = self.pca_scroll
        width = sa.width() - (sa.verticalScrollBar().sizeHint().width() if n > n_fit else 0)
        label_h = 36                      # two-line labels, ~130 px wide: columns are not narrower
        ncols = grid_columns(n_fit, nx, ny, width, sa.height(), label_h, min_w=140)
        rows, rows_fit = -(-n // ncols), -(-n_fit // ncols)
        self.pca_glw.setMinimumHeight(int(rows * sa.height() / rows_fit) if rows > rows_fit else 0)
        if len(self.pca_views) != n or ncols != self._pca_ncols:
            self.pca_glw.clear()
            self.pca_views = []
            self._pca_ncols = ncols
            for i in range(n):
                r, c = (i // ncols) * 2, i % ncols
                lab = self.pca_glw.addLabel('', row=r, col=c)
                lab.setMaximumHeight(label_h)
                self.pca_glw.ci.layout.setRowFixedHeight(r, label_h)
                vb = ScrollViewBox(lockAspect=True, invertY=True)
                self.pca_glw.addItem(vb, row=r + 1, col=c)
                img = pg.ImageItem()
                vb.addItem(img)
                self.pca_views.append(dict(vb=vb, img=img, label=lab, fresh=True))
        # scores: blue - black - red, symmetric around zero
        lut = pg.ColorMap([0, 0.5, 1], [(70, 130, 255), (0, 0, 0), (255, 85, 55)]).getLookupTable(0.0, 1.0, 256)
        for i, v in enumerate(self.pca_views):
            m = p.score_map(i)
            a = float(np.percentile(np.abs(m), 99.5)) or 1.0
            v['img'].setImage(m, levels=(-a, a), lut=lut)
            if v.pop('fresh', False):
                v['vb'].autoRange(padding=0.01)
            inm = i in used
            col = ('#3cff5a' if sig[i] else '#ff9b2d') if inm else '#888'
            mark = (' ✓' if inm else '') + ('' if sig[i] else ' шум') + (' ◀' if i == self.pca_sel else '')
            v['label'].setText(f'<span style="color:{col}; font-weight:bold; font-size:10pt;">#{i + 1}{mark}</span>'
                               f'<br><span style="color:#999; font-size:9pt;">λ/λш = {rel[i]:.3g}, '
                               f'r = {p.autocorr[i]:.2f}</span>')
        self.show_pca_component(self.pca_sel)
        if not self.ed_pca_comps.hasFocus():
            self.ed_pca_comps.setText(ranges_text(used))
        lo, hi = self._pca_range
        how = ('выбраны вручную' if self.pca_manual is not None
               else f'первые k = {k}' + (', без шумовых' if self.cb_pca_skip.isChecked() else ''))
        self.lbl_pca.setText(
            f'{lo:.2f}–{hi:.2f} кэВ, {len(p.sel)} каналов × {nx * ny} пикселей, '
            + (f'разложение по блокам {p.binning}×{p.binning} ({ny // p.binning}×{nx // p.binning}), карты — в '
               f'полном разрешении, ' if p.binning > 1 else '')
            + (f'пуассоновская нормировка. Сигнальных компонент (над границей шума Марченко–Пастур, со связной '
               f'картой): <b>{int(p.signal.sum())}</b>'
               if p.poisson else f'без нормировки: шум неоднороден, граница шума оценена по картам; '
                                 f'связных карт: <b>{int(p.signal.sum())}</b>')
            + f'. В модели: <b>{ranges_text(used)}</b> ({how}). '
            f'Точки: зелёные — в модели, полые — сигнальная не в модели; × — шум (оранжевый — шум в модели). '
            f'Карты — счета (scores) всех {n} компонент (прокрутка колесом, Ctrl + колесо — увеличение), '
            f'r — корреляция соседних пикселей карты{" (по блокам разложения)" if p.binning > 1 else ""} '
            f'(≈0 у шума). Щелчок по карте — нагрузка (loading) компоненты на графике слева; '
            f'<b>двойной щелчок по карте или щелчок по точке графика — добавить / убрать компоненту</b>.'
            + self.local_pca_text())

    def local_pca_text(self):
        """Summary of the local PCA for the label of the PCA tab ('' without it)."""
        lp = self.pca_local
        if lp is None:
            return ''
        n, a = lp.n_comps(), self.local_added()
        where = (f'в <b>{int((a > 0).sum())}</b> плитках из {len(a)} добавлено до {int(a.max())} направлений — там '
                 f'есть то, чего нет в глобальной модели (где — вкладка «Спектр», «Изображение» → «Локальный PCA: '
                 f'добавлено»)' if a.any() else 'ни в одной плитке нет сигнала вне глобальной модели — данные '
                                                 'восстанавливаются как без локального PCA')
        return (f'<br><b>Локальный PCA</b>: {len(n)} плиток {lp.tile}×{lp.tile} px с перекрытием в полплитки, '
                f'сигнальных компонент в плитке {n.min()}–{n.max()}. Модель плитки — глобальная модель выше плюс '
                f'направления её компонент вне глобальной модели: {where}.')

    def show_pca_component(self, i):
        p = self.pca
        E = self.si_raw.energy[p.sel]
        y = p.spectrum(i)
        self.load_curve.setData(E, y)
        self.load_pw.getPlotItem().setTitle(f'Компонента #{i + 1}: нагрузка (loading) — спектр, знак произволен')
        for it in self._pca_marks:
            self.load_pw.removeItem(it)
        self._pca_marks = []
        # main lines of the selected elements where the component has a peak (positive or negative)
        ymax = float(np.max(np.abs(y))) or 1.0
        marks = []
        for el in self.elements:
            for f in core.element_families(el, float(E[0]), float(E[-1]), self.e0()):
                x = float(self.det.measured_energy(f.main[1], f.key))
                near = np.abs(E - x) <= 0.5 * float(self.det.fwhm(max(x, 0.1)))
                if near.any() and np.abs(y[near]).max() >= 0.15 * ymax:
                    marks.append((x, el, f.family))
        for n, (x, el, fam) in enumerate(sorted(marks)):
            line = pg.InfiniteLine(pos=x, angle=90, pen=pg.mkPen(self.color(el), width=1, style=Qt.DotLine),
                                   label=f'{el} {fam}',
                                   labelOpts=dict(position=0.95 - 0.08 * (n % 4), color=self.color(el)))
            self.load_pw.addItem(line, ignoreBounds=True)
            self._pca_marks.append(line)

    def on_pca_map_clicked(self, ev):
        pos = ev.scenePos()
        for i, v in enumerate(self.pca_views):
            if v['vb'].sceneBoundingRect().contains(pos):
                self.pca_sel = i
                if ev.double():
                    self.toggle_pca_component(i)    # renders too
                else:
                    self.render_pca()
                break

    def pca_map_check(self, el, fam):
        """Share of the spatial structure of the map of the PCA input (preprocessed data) lost / added in the PCA map of el / fam
        (core.pca_map_check); None: maps not of PCA data or nothing to compare."""
        if self.maps_raw is None:
            return None
        if (el, fam) not in self._check_cache:
            raw = [m for f, m in zip(self.maps_raw.families, self.maps_raw.maps)
                   if f.element == el and (fam == 'Σ' or f.family == fam)]
            self._check_cache[(el, fam)] = (core.pca_map_check(np.sum(raw, axis=0), self.raw_map(el, fam))
                                            if raw else None)
        return self._check_cache[(el, fam)]

    def pca_flagged(self):
        """{element: share} of the maps whose PCA version does not agree with the data before the PCA."""
        out = {}
        for el in self.map_elements():
            f = self.pca_map_check(el, self.map_family(el))
            if f is not None and f > core.PCA_MAP_LOST:
                out[el] = f
        return out

    def map_check(self, el, fam):
        """core.MapCheck of the map of el / fam (Σ: of its strongest family); None: not checked."""
        chks = [self.map_checks[f.key] for f in self.maps.families
                if f.element == el and (fam == 'Σ' or f.family == fam) and f.key in self.map_checks]
        return max(chks, key=lambda c: abs(float(np.average(c.values[0], weights=c.npix))), default=None)

    def map_structure(self, el, fam):
        """core.map_structure of the map of el / fam of the data before PCA (of the maps themselves without PCA)."""
        if (el, fam) not in self._struct_cache:
            src = self.maps_raw if self.maps_raw is not None else self.maps
            ms = [m for f, m in zip(src.families, src.maps) if f.element == el and (fam == 'Σ' or f.family == fam)]
            self._struct_cache[(el, fam)] = core.map_structure(np.sum(ms, axis=0)) if ms else 0.0
        return self._struct_cache[(el, fam)]

    def map_warning(self, el, fam):
        """'шум': no significant difference between the phases and no spatial structure in the map of the data
        before PCA; 'фон': the distribution between the phases depends on the background model (the map
        shows the error of the background under a weak line); None: fine or not checked."""
        c = self.map_check(el, fam)
        if c is None:
            return None
        if c.signif < core.MAP_PHASE_SIGMA and self.map_structure(el, fam) < core.MAP_STRUCTURE:
            return 'шум'
        if c.share >= core.MAP_BG_SHARE:
            return 'фон'
        return None

    def map_warnings(self):
        """{element: 'фон' / 'шум'} of the maps (selected families) that do not show the distribution of the element."""
        out = {}
        for el in self.map_elements():
            k = self.map_warning(el, self.map_family(el))
            if k:
                out[el] = k
        return out

    def map_check_text(self, el, fam):
        """Tooltip: verdict of the check and the phase means of the map under every background model."""
        c = self.map_check(el, fam)
        if c is None:
            return ''
        share = 'обращает или стирает' if not np.isfinite(c.share) else f'меняет на {min(c.share, 9.99) * 100:.0f} %'
        pca = self.pca_k() > 0
        head = {'шум': f'⚠ Различие между фазами не значимо ({c.signif:.1f}σ), и у карты '
                       + ('данных до PCA ' if pca else '') + 'нет пространственной структуры: карта — шум'
                       + ('; с PCA — комбинация главных фаз без опоры в данных' if pca else '') + '.',
                'фон': f'⚠ Распределение по фазам зависит от модели фона: «{c.worst}» {share} контраст между фазами. '
                       'Фон под слабой линией данными не определён, карта показывает ошибку модели фона '
                       '(распределённую как главные фазы), а не элемент.',
                None: f'Распределение по фазам определено: контраст {c.signif:.0f}σ; '
                      + (f'другая модель фона («{c.worst}») {share} контраст.' if c.worst else
                         'другие модели фона его значимо не меняют.')}[self.map_warning(el, fam)]
        w = max(len(n) for n in c.models) + 2
        rows = ['фаза'.ljust(w) + ''.join(f'{i + 1:>10d}' for i in range(len(c.npix))),
                'пикселей'.ljust(w) + ''.join(f'{int(n):>10d}' for n in c.npix)]
        rows += [n.ljust(w) + ''.join(f'{v:>10.3g}' for v in vals) for n, vals in zip(c.models, c.values)]
        rows.append('±1σ'.ljust(w) + ''.join(f'{v:>10.2g}' for v in c.err))
        return (f'<b>{el} {fam}</b>: {head}<br>Среднее, отсч./пикс., по фазам при разных моделях фона:'
                f'<pre>{chr(10).join(rows)}</pre>')

    def clear_maps(self):
        self.maps = None
        self.maps_raw = None
        self.map_checks = {}
        self.map_table.setRowCount(0)
        self.map_rows = {}
        self.grid_glw.clear()
        self.map_views = []
        self.comp_img.clear()
        self.comp_legend.setText('')
        self._drop_scalebars()

    def _drop_scalebars(self):
        """Scale bars removed (they are made again by render_maps: new calibration / new maps)."""
        if self.comp_scale is not None:
            # (ScaleBar.setParentItem(None) raises in pyqtgraph; removal from the scene detaches it)
            scene = self.comp_scale.scene()
            if scene is not None:
                scene.removeItem(self.comp_scale)
            self.comp_scale = None
        self._comp_rect = None
        self.map_views = []         # the grid of maps is rebuilt, with its scale bars

    def map_elements(self):
        if self.maps is None:
            return []
        out = []
        for f in self.maps.families:
            if f.element not in out:
                out.append(f.element)
        return out

    def build_map_table(self):
        old = {el: dict(on=r['on'].isChecked(), fam=r['fam'].currentText(), gain=r['gain'].value())
               for el, r in self.map_rows.items()}
        els = self.map_elements()
        self.map_table.setRowCount(len(els))
        self.map_rows = {}
        sig = {}
        if self.fit is not None:
            for f, c, e in zip(self.fit.families, self.fit.coef, self.fit.err):
                sig[(f.element, f.family)] = c / e if e > 0 else 0

        def score(el, fm):
            return sig.get((el, fm), self._map_snr(el, fm))

        def best_family(el, fams):
            good = [fm for fm in fams if core.Family(el, fm).main[1] >= 1.0 and score(el, fm) >= 3]
            return max(good or fams, key=lambda fm: score(el, fm))

        best = {el: best_family(el, [f.family for f in self.maps.families if f.element == el]) for el in els}
        # composite by default: the 4 strongest maps that show the distribution of their element (map_warning)
        top = sorted([el for el in els if not self.map_warning(el, best[el])], key=lambda el: -score(el, best[el]))[:4]
        for i, el in enumerate(els):
            fams = [f.family for f in self.maps.families if f.element == el]
            cb = QtWidgets.QCheckBox()
            cb.setChecked(old.get(el, {}).get('on', el in top))
            cell = QtWidgets.QWidget()
            lay = QtWidgets.QHBoxLayout(cell)
            lay.addWidget(cb)
            lay.setAlignment(Qt.AlignCenter)
            lay.setContentsMargins(0, 0, 0, 0)
            self.map_table.setCellWidget(i, 0, cell)
            it = QtWidgets.QTableWidgetItem(el)
            it.setForeground(QtGui.QBrush(self.color(el)))
            it.setFlags(Qt.ItemIsEnabled)
            self.map_table.setItem(i, 1, it)
            combo = QtWidgets.QComboBox()
            combo.addItems(fams + (['Σ'] if len(fams) > 1 else []))
            prev = old.get(el, {}).get('fam')
            combo.setCurrentText(prev if prev and prev in fams + ['Σ'] else best[el])
            self.map_table.setCellWidget(i, 2, combo)
            cbtn = ColorButton(self.color(el))
            cbtn.colorChanged.connect(lambda c, e=el: self.set_element_color(e, c))
            self.map_table.setCellWidget(i, 3, cbtn)
            gain = dspin(old.get(el, {}).get('gain', 1.0), 0.1, 10, 0.1, 2)
            self.map_table.setCellWidget(i, 4, gain)
            cb.toggled.connect(self.render_maps)
            combo.currentIndexChanged.connect(self.render_maps)
            gain.valueChanged.connect(self.render_maps)
            self.map_rows[el] = dict(on=cb, fam=combo, color=cbtn, gain=gain, item=it)

    def _map_snr(self, el, fam):
        for f, m in zip(self.maps.families, self.maps.maps):
            if f.element == el and f.family == fam:
                return float(m.sum() / np.sqrt(max(np.abs(m).sum(), 1.0)))
        return 0.0

    def set_element_color(self, el, c):
        self.colors[el] = QtGui.QColor(c)
        self.ptable.set_checked(el, el in self.elements, self.colors[el].name())
        self.update_id_tree()
        self.update_fit_table()
        self.redraw_spectrum()
        self.render_maps()

    def raw_map(self, el, fam):
        maps = [m for f, m in zip(self.maps.families, self.maps.maps)
                if f.element == el and (fam == 'Σ' or f.family == fam)]
        return np.sum(maps, axis=0) if maps else None

    def smoothing(self):
        mode = ['none', 'gauss', 'bin'][self.cb_smooth.currentIndex()]
        return mode, self.sp_smooth.value()

    def blemish(self):
        """(points and areas, threshold σ) of the hot-pixel removal; None if it is off."""
        if not self.b_blemish.isChecked():
            return None
        return self.cb_blemish.currentIndex() == 1, self.sp_blemish.value()

    def blemish_desc(self, en=False):
        b = self.blemish()
        if not b:
            return ''
        if en:
            return f'hot pixels removed ({"points and areas" if b[0] else "points"}, {b[1]:g}σ)'
        return f'горячие пиксели убраны ({"точки и области" if b[0] else "точки"}, {b[1]:g}σ)'

    def map_family(self, el):
        row = self.map_rows.get(el)
        return row['fam'].currentText() if row else [f.family for f in self.maps.families if f.element == el][0]

    def clean_map(self, el, fam):
        """(raw map of el / fam with the hot pixels replaced if the removal is on, number of pixels replaced)."""
        m = self.raw_map(el, fam)
        b = self.blemish()
        if b is None or m is None:
            return m, 0
        key = (id(self.maps), el, fam, b)
        if key not in self._blem_cache:
            self._blem_cache[key] = core.remove_blemishes(m, b[1], area=b[0])
        return self._blem_cache[key]

    def element_map(self, el):
        m, _ = self.clean_map(el, self.map_family(el))
        mode, val = self.smoothing()
        return core.smooth_map(m, mode, val)

    def normalize(self, m, levels=None):
        lo, hi = levels if levels is not None else np.percentile(m, [self.sp_plo.value(), self.sp_phi.value()])
        if hi <= lo:
            hi = lo + 1e-9
        return np.clip((m - lo) / (hi - lo), 0, 1) ** self.sp_gamma.value(), lo, hi

    def common_range(self):
        """Display range shared by all element maps: 0 ... the largest upper percentile among them."""
        els = self.map_elements()
        key = (tuple(self.map_rows[el]['fam'].currentText() if el in self.map_rows else '' for el in els),
               self.smoothing(), self.sp_phi.value(), self.blemish())
        if self._common is None or self._common[0] is not self.maps or self._common[1] != key:
            hi = max(float(np.percentile(self.element_map(el), self.sp_phi.value())) for el in els)
            self._common = (self.maps, key, hi)
        return 0.0, max(self._common[2], 1e-9)

    def normalize_map(self, m):
        """normalize() of an element map: in the range shared by all maps or in its own percentiles."""
        return self.normalize(m, self.common_range() if self.cb_common.isChecked() else None)

    def composite(self, max_side=2048):
        """Colour composite (rgb 0..1, elements). With an electron image as the base it is built at the
        resolution of that image (long side up to max_side), the maps are interpolated onto it."""
        els = [el for el in self.map_elements() if self.map_rows.get(el) and self.map_rows[el]['on'].isChecked()]
        mode, val = self.smoothing()
        shape = core.smooth_map(self.maps.maps[0], mode, val).shape
        base = None
        if self.sp_base.value() > 0:
            img, electron = self.base_image()
            if electron:
                s = min(1.0, max_side / max(img.shape))
                fine = (max(int(round(img.shape[0] * s)), 1), max(int(round(img.shape[1] * s)), 1))
                if fine[0] > shape[0] and fine[1] > shape[1]:
                    shape = fine
                img = core.resize_image(img, shape)
                lo, hi = np.percentile(img, [0.5, 99.5])
                base = np.clip((img - lo) / max(hi - lo, 1e-9), 0, 1)
            else:
                base, _, _ = self.normalize(core.smooth_map(img, mode, val))
        rgb = np.zeros(shape + (3,), np.float32)
        if base is not None:
            rgb += (self.sp_base.value() / 100.0) * base[..., None]
        blend = self.cb_blend.currentIndex()
        for el in els:
            v, _, _ = self.normalize_map(self.element_map(el))
            v = np.clip(v * self.map_rows[el]['gain'].value(), 0, 1)
            if v.shape != shape:
                v = np.clip(core.resize_image(v, shape), 0, 1)
            c = np.array(self.color(el).getRgbF()[:3], np.float32)
            layer = v[..., None] * c
            if blend == 0:
                rgb = rgb + layer
            elif blend == 1:
                rgb = 1 - (1 - np.clip(rgb, 0, 1)) * (1 - layer)
            else:
                rgb = np.maximum(rgb, layer)
        return np.clip(rgb, 0, 1), els

    def lut_for(self, el):
        name = self.cb_cmap.currentText()
        if name == 'цвет элемента':
            c = self.color(el)
            cmap = pg.ColorMap([0, 1], [(0, 0, 0), (c.red(), c.green(), c.blue())])
        else:
            cmap = pg.colormap.get(name) if name != 'gray' else pg.ColorMap([0, 1], [(0, 0, 0), (255, 255, 255)])
        return cmap.getLookupTable(0.0, 1.0, 256)

    def _add_scalebar(self, vb, width_m):
        if not self.si.pixel_size:
            return None
        L = nice_length(width_m)
        sb = pg.ScaleBar(size=L, suffix='m', width=4, brush=pg.mkBrush('w'), pen=pg.mkPen('w'))
        sb.setParentItem(vb)
        sb.anchor((1, 1), (1, 1), offset=(-12, -12))
        return sb

    def render_maps(self, *args):
        if self.maps is None or self.si is None:
            return
        ny, nx = self.si.shape
        px = self.si.pixel_size or 1.0
        W, H = nx * px, ny * px
        rgb, els = self.composite()
        self.comp_img.setImage((rgb * 255).astype(np.uint8), levels=(0, 255))
        self.comp_img.setRect(QtCore.QRectF(0, 0, W, H))
        if getattr(self, '_comp_rect', None) != (W, H):     # new maps / calibration (not on every redraw: zoom)
            self._comp_rect = (W, H)
            self.comp_vb.autoRange(padding=0.01)
        if self.comp_scale is None:
            self.comp_scale = self._add_scalebar(self.comp_vb, W)
        leg = '&nbsp;&nbsp;'.join(f'<span style="color:{self.color(el).name()}; font-weight:bold;">■ {el} '
                                  f'{self.map_rows[el]["fam"].currentText()}</span>' for el in els)
        info = self.maps_info()
        if self.pca_k():
            info += f'; данные PCA ({self.pca_desc()})'
            bad = self.pca_flagged()
            if bad:
                info += ('<br><span style="color:#ff9b2d;">⚠ Карты с PCA не согласуются с исходными данными '
                         '(доля пространственной структуры, потерянной или добавленной PCA): '
                         + ', '.join(f'{el} {min(f, 1.0) * 100:.0f} %' for el, f in bad.items())
                         + '. Собственное распределение такого элемента не выделено в компоненту, его карта — '
                           'комбинация карт основных фаз. Смотрите их по исходным данным (снять «Использовать '
                           'восстановленные данные», бининг) или фитом спектров ROI.</span>')
            elif self.maps_raw is not None:
                info += ' — карты согласуются с картами исходных данных (проверка по остатку)'
        if self.maps.method == 'window':
            info += (' — <span style="color:#ff9b2d;">карты содержат фон: участки с большим числом отсчётов '
                     'светятся во всех картах; для слабых элементов используйте MLLS</span>')
        if self.cb_common.isChecked():
            info += f'; общая шкала карт 0 … {self.common_range()[1]:.3g} отсч./пикс.'
        if self.blemish():
            counts = ', '.join(f'{el} {self.clean_map(el, self.map_family(el))[1]}' for el in self.map_elements())
            info += f'; {self.blemish_desc()}, заменено пикселей: {counts}'
        warned = self.map_warnings()
        if warned:
            bg_w = [el for el, k in warned.items() if k == 'фон']
            noise_w = [el for el, k in warned.items() if k == 'шум']
            parts = ([f'распределение по фазам зависит от модели фона — {", ".join(bg_w)} (фон под слабой линией '
                      f'данными не определён, карта показывает его ошибку, распределённую как главные фазы)']
                     if bg_w else []) + (
                [f'различие между фазами не значимо — {", ".join(noise_w)} (карта — шум'
                 + ('; с PCA — комбинация главных фаз' if self.pca_k() else '') + ')'] if noise_w else [])
            info += ('<br><span style="color:#ff9b2d;">⚠ Карты не показывают распределение элемента: '
                     + '; '.join(parts) + '. В композит по умолчанию не включаются; значения по фазам — '
                                          'в подсказке у элемента в таблице карт.</span>')
        self.comp_legend.setText(leg + f'<br><span style="color:#999; font-size:9pt;">{info}</span>')

        allels = self.map_elements()
        label_h = 36                      # two-line labels (element, warnings / display range), ~130 px wide
        ncols = grid_columns(len(allels), nx, ny, self.grid_glw.width(), self.grid_glw.height(), label_h, min_w=140)
        rebuild = ([v['el'] for v in self.map_views] != allels or ncols != self._grid_ncols)
        if rebuild:
            self.grid_glw.clear()
            self.map_views = []
            self._grid_ncols = ncols
            for k, el in enumerate(allels):
                r, c = (k // ncols) * 2, k % ncols
                lab = self.grid_glw.addLabel('', row=r, col=c)
                lab.setMaximumHeight(label_h)
                self.grid_glw.ci.layout.setRowFixedHeight(r, label_h)
                vb = self.grid_glw.addViewBox(row=r + 1, col=c, lockAspect=True, invertY=True)
                img = pg.ImageItem()
                vb.addItem(img)
                self.map_views.append(dict(el=el, label=lab, vb=vb, img=img, sb=None))
        for v in self.map_views:
            el = v['el']
            m = self.element_map(el)
            nv, lo, hi = self.normalize_map(m)
            v['img'].setImage(nv, levels=(0, 1), lut=self.lut_for(el))
            v['img'].setRect(QtCore.QRectF(0, 0, W, H))
            v['data'] = m
            fam = self.map_rows[el]['fam'].currentText()
            rng = f'среднее {m.mean():.3g}' if self.cb_common.isChecked() else f'{lo:.3g} … {hi:.3g}'
            chk = self.pca_map_check(el, fam)
            warn = ' <span style="color:#ff9b2d;">⚠ PCA</span>' if chk is not None and chk > core.PCA_MAP_LOST else ''
            kind = self.map_warning(el, fam)
            if kind:
                warn += f' <span style="color:#ff9b2d;">⚠ {kind}</span>'
            row = self.map_rows[el]
            row['item'].setText(f'{el} ⚠' if kind else el)
            row['item'].setToolTip(self.map_check_text(el, fam))
            v['label'].setText(f'<span style="color:{self.color(el).name()}; font-size:11pt; font-weight:bold;">'
                               f'{el} {fam}</span>{warn}<br><span style="color:#999; font-size:9pt;">{rng}</span>')
            if v['sb'] is None:
                v['sb'] = self._add_scalebar(v['vb'], W)
            if rebuild:
                v['vb'].autoRange(padding=0.01)
        if self.cb_img.currentText().startswith('Карта: '):
            self.update_si_image()

    def maps_info(self, en=False):
        """'Метод: MLLS-фит; фон по фазам (3); форма пиков из NLLS' - how the maps were computed
        (en: in English, for the presentation export)."""
        method = {'mlls': ('MLLS-фит', 'MLLS fit'), 'window_net': ('окно − фон − перекрытия',
                                                                   'window − background − overlaps'),
                  'window': ('окно (брутто)', 'window (gross counts)')}[self.maps.method][en]
        info = f'{"Method" if en else "Метод"}: {method}'
        if self.maps.method != 'window':
            if self.map_phases is not None:
                n = int(self.map_phases.max()) + 1
                info += f'; background per phase ({n})' if en else f'; фон по фазам ({n})'
            else:
                info += '; common background shape' if en else '; общая форма фона'
            if self.det.widths or self.det.ratios:
                info += '; peak shapes from NLLS' if en else '; форма пиков из NLLS'
        return info

    def smoothing_desc(self, en=False):
        """'Гаусс σ (пикс.) = 1' / 'Gaussian, σ = 1 px'; '' without smoothing."""
        mode, val = self.smoothing()
        if mode == 'none':
            return ''
        if en:
            return f'Smoothing: Gaussian, σ = {val:g} px' if mode == 'gauss' else f'Binning: {int(val)}×{int(val)}'
        return f'Сглаживание: {self.cb_smooth.currentText()} = {val:g}'

    def on_map_mouse(self, evt):
        if self.maps is None:
            return
        pos = evt[0]
        ny, nx = self.si.shape
        px = self.si.pixel_size or 1.0
        views = [dict(el=None, vb=self.comp_vb)] + self.map_views
        for v in views:
            if v['vb'].sceneBoundingRect().contains(pos):
                p = v['vb'].mapSceneToView(pos)
                ix, iy = int(p.x() / px), int(p.y() / px)
                if 0 <= ix < nx and 0 <= iy < ny:
                    parts = []
                    for w in self.map_views:
                        d = w.get('data')
                        if d is None:
                            continue
                        k = d.shape[1] / nx
                        jx, jy = min(int(ix * k), d.shape[1] - 1), min(int(iy * k), d.shape[0] - 1)
                        parts.append(f'{w["el"]}={d[jy, jx]:.1f}')
                    self.statusBar().showMessage(f'x={ix}, y={iy}:  ' + '  '.join(parts))
                return

    # ------------------------------------------------------------------ export
    def _ask_save(self, title, name, filt):
        start = os.path.join(os.path.dirname(self.si.path), name) if self.si else name
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, title, start, filt)
        return path

    @guarded
    def export_spectrum(self):
        if self.si is None or self.spec is None:
            return
        path = self._ask_save('Экспорт спектра', f'{self.si.name}_spectrum.csv', 'CSV (*.csv)')
        if not path:
            return
        E = self.si.energy
        cols = {'energy_keV': E, 'counts': self.spec}
        if self.bg is not None:
            cols['background_estimate'] = self.bg
        if self.fit is not None:
            cols['fit_background'] = self.fit.background
            cols['fit_model'] = self.fit.model
            for f, comp in zip(self.fit.families, self.fit.components):
                cols[f'{f.element}_{f.family}'] = comp
        if self.rl is not None:
            cols['richardson_lucy'] = self.rl
        keys = list(cols)
        with open(path, 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(keys)
            for i in range(len(E)):
                w.writerow([f'{cols[k][i]:.6g}' for k in keys])
        self.statusBar().showMessage(f'Сохранено: {path}')

    @guarded
    def export_fit_table(self):
        if self.fit is None:
            QtWidgets.QMessageBox.information(self, 'Экспорт', 'Сначала выполните фит.')
            return
        path = self._ask_save('Экспорт результатов фита', f'{self.si.name}_fit.csv', 'CSV (*.csv)')
        if not path:
            return
        self._write_fit_csv(path)
        self.statusBar().showMessage(f'Сохранено: {path}')

    def _write_fit_csv(self, path):
        d = self.fit.det
        with open(path, 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['element', 'family', 'main_line', 'energy_keV', 'shift_eV', 'width_factor', 'net_counts',
                        'sigma', 'significance', 'line_group_factors'])
            for r in self.fit.table():
                key = f"{r['element']} {r['family']}"
                sh, kw = d.shifts.get(key), d.widths.get(key)
                w.writerow([r['element'], r['family'], r['line'], f"{r['energy']:.4f}",
                            '' if sh is None else f'{sh * 1000:.1f}', '' if kw is None else f'{kw:.4f}',
                            f"{r['counts']:.1f}", f"{r['err']:.1f}", f"{r['sigma']:.2f}",
                            ' '.join(f'{g}={v:.3f}' for g, v in sorted(d.ratios.get(key, {}).items()))])
            w.writerow([])
            w.writerow(['source', 'ROI' if self.cb_roi.isChecked() else 'sum spectrum'])
            w.writerow(['pca_components', self.pca_desc() or 'none (raw data)'])
            w.writerow(['range_keV', *self.fit.erange])
            w.writerow(['chi2_red', f'{self.fit.chi2_red:.3f}'])
            for k in ('fwhm_mn', 'fano', 'spread', 'offset', 'gain', 'tail', 'tail_width', 'escape', 'thr', 'thr_width'):
                w.writerow([k, getattr(d, k)])

    def _mpl_scalebar(self, ax, nx, ny, px, color='white', fontsize=11):
        if not px:
            return
        L = nice_length(nx * px)
        lpx = L / px
        from matplotlib.patches import Rectangle
        from matplotlib import patheffects
        x1 = nx - 0.04 * nx
        y1 = ny - 0.06 * ny
        ax.add_patch(Rectangle((x1 - lpx, y1 - max(ny * 0.015, 0.6)), lpx, max(ny * 0.015, 0.6), facecolor=color,
                               edgecolor='black', linewidth=0.8))
        ax.text(x1 - lpx / 2, y1 - max(ny * 0.03, 1.2), si_length(L), color=color, ha='center', va='bottom',
                fontsize=fontsize, fontweight='bold', path_effects=[patheffects.withStroke(linewidth=3, foreground='black')])

    def _figure(self, w_in, h_in):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig = Figure(figsize=(w_in, h_in), facecolor='black')
        FigureCanvasAgg(fig)
        return fig

    def _save_composite_png(self, path, dpi=200):
        rgb, els = self.composite()
        ny, nx = rgb.shape[:2]
        px = (self.si.pixel_size or 0) * self.si.shape[1] / nx
        fig = self._figure(9, 9 * ny / nx + 0.6)
        ax = fig.add_axes([0, 0, 1, 1 - 0.6 / (9 * ny / nx + 0.6)])
        ax.imshow(rgb, interpolation='nearest')
        ax.set_axis_off()
        self._mpl_scalebar(ax, nx, ny, px)
        x = 0.01
        for el in els:
            fam = self.map_rows[el]['fam'].currentText()
            t = fig.text(x, 1 - 0.3 / (9 * ny / nx + 0.6), f'{el} {fam}', color=self.color(el).name(),
                         fontsize=14, fontweight='bold', va='center')
            x += 0.03 + 0.017 * len(f'{el} {fam}')
        fig.savefig(path, dpi=max(dpi, int(np.ceil(nx / 9))), facecolor='black')     # an electron base: its pixels

    @guarded
    def save_composite(self):
        if self.maps is None:
            QtWidgets.QMessageBox.information(self, 'Экспорт', 'Сначала постройте карты.')
            return
        path = self._ask_save('Сохранить композит', f'{self.si.name}_composite.png', 'PNG (*.png)')
        if path:
            self._save_composite_png(path)
            self.statusBar().showMessage(f'Сохранено: {path}')

    @guarded
    def save_all_maps(self):
        if self.maps is None:
            QtWidgets.QMessageBox.information(self, 'Экспорт', 'Сначала постройте карты.')
            return
        start = os.path.dirname(self.si.path)
        d = QtWidgets.QFileDialog.getExistingDirectory(self, 'Папка для карт', start)
        if not d:
            return
        out = os.path.join(d, f'{self.si.name}_maps')
        os.makedirs(out, exist_ok=True)
        self.export_maps_to(out)
        self.statusBar().showMessage(f'Карты сохранены в {out}')

    def _mpl_cmap_norm(self, el, m):
        """Matplotlib colormap and normalisation of an element map as displayed (palette, contrast, gamma)."""
        from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm
        import matplotlib
        _, lo, hi = self.normalize_map(m)
        name = self.cb_cmap.currentText()
        cmap = (LinearSegmentedColormap.from_list(el, ['black', self.color(el).name()])
                if name == 'цвет элемента' else matplotlib.colormaps[name])
        g = self.sp_gamma.value()
        norm = (Normalize(vmin=lo, vmax=hi, clip=True) if abs(g - 1) < 1e-3
                else PowerNorm(g, vmin=lo, vmax=hi, clip=True))
        return cmap, norm, lo, hi

    def export_maps_to(self, out):
        import tifffile
        ny, nx = self.si.shape
        els = self.map_elements()
        mode, val = self.smoothing()
        self._save_composite_png(os.path.join(out, 'composite.png'))
        n = len(els)
        ncols = grid_columns(n, nx, ny, 1600, 1200)
        nrows = int(np.ceil(n / ncols))
        mont = self._figure(4.5 * ncols, 4.5 * ncols * ny / nx * nrows / ncols + 0.5 * nrows)
        for k, el in enumerate(els):
            fam = self.map_rows[el]['fam'].currentText()
            m = self.element_map(el)
            cmap, norm, lo, hi = self._mpl_cmap_norm(el, m)
            tag = f'{el}_{fam}'.replace('Σ', 'sum')
            tifffile.imwrite(os.path.join(out, f'{tag}.tif'), m.astype(np.float32))
            fig = self._figure(8, 8 * m.shape[0] / m.shape[1] + 0.8)
            ax = fig.add_subplot(111)
            im = ax.imshow(m, cmap=cmap, norm=norm, interpolation='nearest')
            ax.set_axis_off()
            ax.set_title(f'{el} {fam}', color=self.color(el).name(), fontsize=16, fontweight='bold')
            px = (self.si.pixel_size or 0) * nx / m.shape[1]
            self._mpl_scalebar(ax, m.shape[1], m.shape[0], px)
            cb = fig.colorbar(im, ax=ax, fraction=0.046 * m.shape[0] / m.shape[1], pad=0.02)
            cb.set_label('чистые отсчёты', color='w')
            cb.ax.yaxis.set_tick_params(color='w', labelcolor='w')
            fig.savefig(os.path.join(out, f'{tag}.png'), dpi=150, facecolor='black', bbox_inches='tight')
            ax = mont.add_subplot(nrows, ncols, k + 1)
            ax.imshow(m, cmap=cmap, norm=norm, interpolation='nearest')
            ax.set_axis_off()
            ax.set_title(f'{el} {fam}', color=self.color(el).name(), fontsize=14, fontweight='bold')
            self._mpl_scalebar(ax, m.shape[1], m.shape[0], px)
        mont.tight_layout()
        mont.savefig(os.path.join(out, 'maps_montage.png'), dpi=150, facecolor='black')
        raw = {f'{f.element}_{f.family}': m for f, m in zip(self.maps.families, self.maps.maps)}
        np.savez_compressed(os.path.join(out, 'maps_raw.npz'), pixel_size_m=self.si.pixel_size,
                            method=self.maps.method, smoothing=f'{mode}:{val}',
                            blemish_in_tif_png=self.blemish_desc() or 'off',
                            pca_map_check=', '.join(f'{el}:{self.pca_map_check(el, self.map_family(el)):.2f}'
                                                    for el in els
                                                    if self.pca_map_check(el, self.map_family(el)) is not None),
                            map_background_check='; '.join(f'{k}: contrast {c.signif:.1f} sigma, background share '
                                                           f'{c.share:.2f}' for k, c in self.map_checks.items()),
                            map_warnings=', '.join(f'{el}:{"background" if k == "фон" else "noise"}'
                                                   for el, k in self.map_warnings().items()),
                            pca_components=np.array([i + 1 for i in self.pca_comps()], dtype=int),
                            pca_data=self.pca_desc(en=True), **raw)
        if self.fit is not None:
            self._write_fit_csv(os.path.join(out, 'fit_results.csv'))

    def _image_png(self, img, px, w_in, frame=None, **kw):
        """In-memory PNG of an image to be shown w_in inches wide: no margins, scale bar, optional yellow
        frame (col, row, width, height in pixel edges). Pixels stay sharp: integer upsampling to >= 1000 px."""
        import io
        from matplotlib.patches import Rectangle
        ny, nx = img.shape[:2]
        fig = self._figure(w_in, w_in * ny / nx)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img, interpolation='nearest', **kw)
        if frame is not None:
            ax.add_patch(Rectangle((frame[0] - 0.5, frame[1] - 0.5), frame[2], frame[3], fill=False,
                                   edgecolor='#ffd23c', linewidth=2))
        ax.set_xlim(-0.5, nx - 0.5)
        ax.set_ylim(ny - 0.5, -0.5)
        ax.set_axis_off()
        self._mpl_scalebar(ax, nx, ny, px, fontsize=min(11.0, 2.5 * w_in))    # small tiles: label fits the bar
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=int(np.ceil(1000 / nx)) * nx / w_in, facecolor='black')
        buf.seek(0)
        return buf

    @guarded
    def export_pptx(self):
        if self.maps is None:
            QtWidgets.QMessageBox.information(self, 'Экспорт', 'Сначала постройте карты.')
            return
        try:
            import pptx  # noqa: F401
        except ImportError:
            QtWidgets.QMessageBox.warning(self, 'Экспорт', 'Нужен пакет python-pptx:\n\npip install python-pptx')
            return
        path = self._ask_save('Экспорт в PowerPoint', f'{self.si.name}_EDX.pptx', 'PowerPoint (*.pptx)')
        if path:
            self.write_pptx(path)
            self.statusBar().showMessage(f'Сохранено: {path}')

    def write_pptx(self, path):
        """16:9 presentation: survey image with the SI frame, colour composite, all element maps
        (each map a separate picture, labels are editable text). All text in English."""
        from pptx import Presentation
        from pptx.dml.color import RGBColor
        from pptx.util import Inches
        si = self.si
        nx = si.shape[1]
        W, H, M, TOP = 13.333, 7.5, 0.4, 1.0      # slide size, side margin, top of the content (inches)
        prs = Presentation()
        prs.slide_width, prs.slide_height = Inches(W), Inches(H)

        def new_slide(title):
            s = prs.slides.add_slide(prs.slide_layouts[6])     # blank
            s.background.fill.solid()
            s.background.fill.fore_color.rgb = RGBColor(0, 0, 0)
            ppt_text(s, M, 0.25, W - 2 * M, 0.6, [[(title, '#f0f0f0', True)]], size=24)
            return s

        def fit(shape, bw, bh):
            r = shape[0] / shape[1]
            w = min(bw, bh / r)
            return w, w * r

        # --- overview: survey image with the SI field of view, HAADF / X-ray counts of the mapped area
        s = new_slide(f'{si.name} — overview')
        panels = []
        if si.survey is not None:
            img = si.survey['image']
            lo, hi = np.percentile(img, [0.5, 99.8])
            panels.append((img, abs(si.survey.get('dx') or 0), self.survey_frame(), dict(cmap='gray', vmin=lo, vmax=hi),
                           'Survey image (frame: mapped area)'))
        i = self.cb_base.currentIndex()     # electron image of the base of the composite, else the first one
        im = self.images[i] if 0 <= i < len(self.images) else (self.images[0] if self.images else None)
        base = im['image'] if im else si.total_image(e_min=self.sp_elo.value())
        lo, hi = np.percentile(base, [0.5, 99.5])
        panels.append((base, (si.pixel_size or 0) * nx / base.shape[1], None,
                       dict(cmap='gray', vmin=lo, vmax=hi if hi > lo else lo + 1),
                       (im.get('name_en', im['name']) if im else 'X-ray counts (sum)') + ' — mapped area'))
        bw, bh = (W - 2 * M - 0.3 * (len(panels) - 1)) / len(panels), H - TOP - 1.3
        for i, (img, px, frame, kw, caption) in enumerate(panels):
            w, h = fit(img.shape, bw, bh)
            x, y = M + i * (bw + 0.3) + (bw - w) / 2, TOP + (bh - h) / 2
            s.shapes.add_picture(self._image_png(img, px, w, frame, **kw), Inches(x), Inches(y), width=Inches(w))
            ppt_text(s, x, y + h + 0.05, w, 0.4, [[(caption, '#b0b6bd', False)]], size=12, center=True)
        px = si.pixel_size
        ppt_text(s, M, H - 0.8, W - 2 * M, 0.6,
                 [[(f'{self.meta_text(en=True)}; pixel: {si_length(px) if px else "—"}', '#9aa3ad', False)]],
                 size=11)

        # --- colour composite with the legend and the map settings
        rgb, els = self.composite()
        s = new_slide(f'{si.name} — composite map')
        legend_w = 3.4
        w, h = fit(rgb.shape, W - 2 * M - legend_w - 0.3, H - TOP - 0.3)
        s.shapes.add_picture(self._image_png(rgb, (px or 0) * nx / rgb.shape[1], w), Inches(M), Inches(TOP),
                             width=Inches(w))
        x = M + w + 0.3
        ppt_text(s, x, TOP, legend_w, 0.45 * max(len(els), 1),
                 [[(f'■ {el} {self.map_rows[el]["fam"].currentText()}', self.color(el).name(), True)] for el in els],
                 size=20)
        info = self.maps_info(en=True).split('; ')
        if self.pca_k():
            info.append(f'PCA-reconstructed data ({self.pca_desc(en=True)})')
            r = self.pre
            done = ([f'{len(r.dead_channels)} dead channels interpolated'] if len(r.dead_channels) else []) + (
                ['energy drift corrected'] if r.drift_applied else []) + (
                [f'{int(r.bad_pixels.sum())} bad pixels replaced'] if r.bad_pixels.any() else [])
            info.append('Preprocessing before PCA: ' + (', '.join(done) if done else 'no changes needed'))
            bad = self.pca_flagged()
            if bad:
                info.append('⚠ PCA maps disagree with the raw data (share of the spatial structure lost or added): '
                            + ', '.join(f'{el} {min(f, 1.0) * 100:.0f} %' for el, f in bad.items()))
        warned = self.map_warnings()
        if warned:
            bg_w = [el for el, k in warned.items() if k == 'фон']
            noise_w = [el for el, k in warned.items() if k == 'шум']
            info.append('⚠ Maps not showing the distribution of the element: '
                        + '; '.join(([f'background-dependent (the distribution between the phases changes with '
                                      f'the background model): {", ".join(bg_w)}'] if bg_w else [])
                                    + ([f'no significant difference between the phases: {", ".join(noise_w)}']
                                       if noise_w else [])))
        if self.blemish():
            info.append(self.blemish_desc(en=True).capitalize())
        if self.smoothing_desc(en=True):
            info.append(self.smoothing_desc(en=True))
        if self.cb_common.isChecked():
            info.append(f'Common scale of all maps: 0 … {self.common_range()[1]:.3g} net counts/pixel, '
                        f'gamma {self.sp_gamma.value():g}')
        else:
            info.append(f'Contrast: {self.sp_plo.value():g}–{self.sp_phi.value():g} percentile, '
                        f'gamma {self.sp_gamma.value():g}')
        ppt_text(s, x, TOP + 0.45 * max(len(els), 1) + 0.3, legend_w, 2.5,
                 [[(t, '#9aa3ad', False)] for t in info], size=12)

        # --- all element maps, up to 12 per slide; labels: element, series, display range
        allels = self.map_elements()
        nslides = int(np.ceil(len(allels) / 12))
        per = max(1, int(np.ceil(len(allels) / max(nslides, 1))))
        gap, lab_h = 0.15, 0.42
        for k0 in range(0, len(allels), per):
            chunk = allels[k0:k0 + per]
            s = new_slide(f'{si.name} — element maps' + (f' ({k0 // per + 1}/{nslides})' if nslides > 1 else ''))
            maps = [self.element_map(el) for el in chunk]
            mh, mw = maps[0].shape
            ncols = grid_columns(len(chunk), mw, mh, 100 * (W - 2 * M), 100 * (H - TOP - 0.2), label_h=100 * lab_h)
            nrows = int(np.ceil(len(chunk) / ncols))
            cw = (W - 2 * M - gap * (ncols - 1)) / ncols
            ch = (H - TOP - 0.2 - gap * (nrows - 1)) / nrows
            for i, (el, m) in enumerate(zip(chunk, maps)):
                cmap, norm, lo, hi = self._mpl_cmap_norm(el, m)
                w, h = fit(m.shape, cw, ch - lab_h)
                cx, y = M + (i % ncols) * (cw + gap), TOP + (i // ncols) * (ch + gap)
                fam = self.map_rows[el]['fam'].currentText()
                kind = self.map_warning(el, fam)
                warn = [(' ⚠ ' + {'фон': 'background', 'шум': 'noise'}[kind], '#ff9b2d', False, 10)] if kind else []
                ppt_text(s, cx, y, cw, lab_h, [[(f'{el} {fam}', self.color(el).name(), True, 16)] + warn
                                               + [(f'   {lo:.3g} … {hi:.3g}', '#999999', False, 10)]], center=True)
                s.shapes.add_picture(self._image_png(m, (px or 0) * nx / m.shape[1], w, cmap=cmap, norm=norm),
                                     Inches(cx + (cw - w) / 2), Inches(y + lab_h), width=Inches(w))
        prs.save(path)


def setup_style(app):
    app.setStyle('Fusion')
    pal = QtGui.QPalette()
    for role, col in ((QtGui.QPalette.Window, '#1e2227'), (QtGui.QPalette.WindowText, '#d8dbe0'),
                      (QtGui.QPalette.Base, '#16191d'), (QtGui.QPalette.AlternateBase, '#1e2227'),
                      (QtGui.QPalette.ToolTipBase, '#2b3036'), (QtGui.QPalette.ToolTipText, '#e8e8e8'),
                      (QtGui.QPalette.Text, '#d8dbe0'), (QtGui.QPalette.Button, '#2a2f35'),
                      (QtGui.QPalette.ButtonText, '#d8dbe0'), (QtGui.QPalette.Highlight, '#3d6fb6'),
                      (QtGui.QPalette.HighlightedText, '#ffffff'), (QtGui.QPalette.Link, '#8cb4ff')):
        pal.setColor(role, QtGui.QColor(col))
    pal.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, QtGui.QColor('#666'))
    pal.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, QtGui.QColor('#666'))
    app.setPalette(pal)


def main():
    app = QtWidgets.QApplication(sys.argv)
    setup_style(app)
    w = MainWindow(sys.argv[1] if len(sys.argv) > 1 else None)
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
