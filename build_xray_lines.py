"""Build edx_xray_lines.py from the xraydb sqlite database (Elam, Ravel & Sieber).

    pip download xraydb --no-deps      (unzip the wheel, take xraydb/xraydb.sqlite)
    python build_xray_lines.py xraydb.sqlite edx_xray_lines.py

Line weights inside a family (K / L / M) are estimated for electron excitation:
primary vacancy population ~ n_electrons / E_edge^2 (Bethe, high overvoltage),
Coster-Kronig redistribution, then fluorescence yield x radiative branching.
"""
import sqlite3
import sys

db, out = sys.argv[1], sys.argv[2]
c = sqlite3.connect(db)

N_EL = {'K': 2, 'L1': 2, 'L2': 2, 'L3': 4, 'M1': 2, 'M2': 2, 'M3': 4, 'M4': 4, 'M5': 6}
FAMILY_LEVELS = {'K': ['K'], 'L': ['L1', 'L2', 'L3'], 'M': ['M1', 'M2', 'M3', 'M4', 'M5']}

names = {r[1]: (r[0], r[2]) for r in c.execute('select atomic_number, element, name from elements')}


def populations(el, fam, edges, yields):
    levels = [lv for lv in FAMILY_LEVELS[fam] if lv in edges and edges[lv] > 0]
    if not levels:
        return {}
    p = {lv: N_EL[lv] / edges[lv] ** 2 for lv in levels}
    ck = {(a, b): v for a, b, v in c.execute(
        'select initial_level, final_level, transition_probability from Coster_Kronig where element=?', (el,))}
    n = {}
    for lv in levels:  # ordered from deepest to shallowest
        n[lv] = p[lv] + sum(ck.get((src, lv), 0.0) * n[src] for src in n)
    return {lv: n[lv] * yields.get(lv, 0.0) for lv in levels}


lines_out = {}
for el, (z, name) in sorted(names.items(), key=lambda kv: kv[1][0]):
    if z < 4 or z > 95:
        continue
    edges, yields = {}, {}
    for lv, e, fy in c.execute(
            'select iupac_symbol, absorption_edge, fluorescence_yield from xray_levels where element=?', (el,)):
        edges[lv], yields[lv] = e, fy
    trans = c.execute('select siegbahn_symbol, initial_level, emission_energy, intensity '
                      'from xray_transitions where element=?', (el,)).fetchall()
    # Elam omits L3-N4,5 (Lb2,15) and L2-N4 (Lg1) for the partially filled 4d metals (Y..Ru);
    # add them scaled by the 4d occupancy (per-electron rates of Rh/Pd), E = edge - 2 eV
    n4d = {'Y': 1, 'Zr': 2, 'Nb': 4, 'Mo': 5, 'Tc': 5, 'Ru': 7}.get(el)
    if n4d:
        add = {'L3': ('Lb2,15', 0.0103 * n4d), 'L2': ('Lg1', 0.0093 * n4d)}
        new = []
        for sym, lvl, e, inten in trans:
            new.append((sym, lvl, e, inten * (1 - add[lvl][1]) if lvl in add else inten))
        for lvl, (sym, inten) in add.items():
            new.append((sym, lvl, edges[lvl] - 2.0, inten))
        trans = new
    fam_lines = {}
    for fam in ('K', 'L', 'M'):
        pop = populations(el, fam, edges, yields)
        if fam == 'K':
            pop = {'K': 1.0}
        lst = []
        for sym, lvl, e, inten in trans:
            lvl0 = 'M4' if lvl == 'M4,5' else lvl
            if lvl0 not in FAMILY_LEVELS[fam] or e < 50:
                continue
            w = inten * pop.get(lvl0, 0.0)
            if w > 0:
                lst.append([sym, e / 1000.0, w])
        if not lst:
            continue
        wmax = max(x[2] for x in lst)
        if fam == 'M':
            # Elam gives M-zeta (M4,5-N2,3) ~0.3 % of M-alpha; EDS spectra show ~3-4 %
            # (e.g. Pt M-zeta at 1.59 keV), use the empirical value
            for x in lst:
                if x[0] == 'Mz':
                    x[2] = max(x[2], 0.035 * wmax)
        lst = [x for x in lst if x[2] >= 0.003 * wmax]
        # merge lines closer than 1 eV
        lst.sort(key=lambda x: x[1])
        merged = []
        for x in lst:
            if merged and abs(x[1] - merged[-1][1]) < 0.001:
                if x[2] > merged[-1][2]:
                    merged[-1][0] = x[0]
                merged[-1][2] += x[2]
            else:
                merged.append(x)
        wmax = max(x[2] for x in merged)
        fam_lines[fam] = [(s, round(e, 5), round(w / wmax, 4)) for s, e, w in merged]
    if not fam_lines:
        continue
    edge_k = {'K': edges.get('K', 0) / 1000, 'L': edges.get('L3', 0) / 1000, 'M': edges.get('M5', edges.get('M4', 0)) / 1000}
    lines_out[el] = (z, name.capitalize(), {f: round(edge_k[f], 5) for f in fam_lines}, fam_lines)

with open(out, 'w', encoding='utf-8') as f:
    f.write('"""X-ray emission lines for EDX (generated, do not edit by hand).\n\n'
            'Source: xraydb (M. Newville, MIT licence) - Elam, Ravel & Sieber,\n'
            'Radiat. Phys. Chem. 63 (2002) 121. Energies in keV.\n'
            'Weights are relative intensities inside each family (K, L, M), strongest line = 1,\n'
            'estimated for electron excitation (Bethe sub-shell populations, Coster-Kronig,\n'
            'fluorescence yields). Edge = ionisation energy needed to excite the family.\n'
            'Corrections to the database: L3-N4,5 (Lb2,15) and L2-N4 (Lg1) added for the 4d metals\n'
            'Y..Ru (scaled by 4d occupancy); M-zeta set to >= 3.5 % of M-alpha (empirical).\n'
            'Regenerate with build_xray_lines.py.\n\n'
            'ELEMENTS[symbol] = (Z, name, {family: edge_keV}, {family: [(line, E_keV, weight), ...]})\n'
            '"""\n\n')
    f.write('ELEMENTS = {\n')
    for el, (z, name, edges, fams) in lines_out.items():
        f.write(f'    {el!r}: ({z}, {name!r}, {edges!r}, {{\n')
        for fam, lst in fams.items():
            f.write(f'        {fam!r}: {lst!r},\n')
        f.write('    }),\n')
    f.write('}\n')
print(len(lines_out), 'elements written')

# --- Si(Li)/SDD escape-peak fraction (Reed & Ware 1972), normal incidence ---
import json
import numpy as np
le, lmu = c.execute('select log_energy, log_photoabsorption from photoabsorption where element="Si"').fetchone()
le, lmu = np.array(json.loads(le)), np.array(json.loads(lmu))
_, e_k, w_k, r_k = c.execute('select iupac_symbol, absorption_edge, fluorescence_yield, jump_ratio '
                             'from xray_levels where element="Si" and iupac_symbol="K"').fetchone()
si_ka = 1739.8


def mu(e_ev):
    return np.exp(np.interp(np.log(e_ev), le, lmu))


grid = np.geomspace(e_k + 5, 40000, 90)
mu_k = mu(si_ka)
x = mu(grid) / mu_k
frac = 0.5 * w_k * (1 - 1 / r_k) * (1 - np.log1p(x) / x)
with open(out, 'a', encoding='utf-8') as f:
    f.write('\n# Si detector escape peak: E_escape = E - SI_KA; fraction of counts moved to the escape peak\n'
            '# (Reed & Ware 1972 formula with Si absorption from the same database).\n')
    f.write(f'SI_KA = {si_ka / 1000:.5f}\nSI_K_EDGE = {e_k / 1000:.5f}\n')
    f.write('ESCAPE_E = [' + ', '.join(f'{v / 1000:.4f}' for v in grid) + ']\n')
    f.write('ESCAPE_FRAC = [' + ', '.join(f'{v:.3e}' for v in frac) + ']\n')
print('escape at 2.17 keV:', np.interp(2170, grid, frac), ' at 6.4 keV:', np.interp(6400, grid, frac))
