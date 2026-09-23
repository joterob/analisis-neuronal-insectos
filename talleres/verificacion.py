# -*- coding: utf-8 -*-
"""
Módulo de verificación — Taller 2
Curso: De la neurona a la figura · Análisis neuronal en insectos asistido por IA

Compara sus resultados contra la verdad oculta del archivo de datos.

    import verificacion as v
    v.usar('taller2_ruidosa.h5')
    v.verificar_deteccion(tiempos_s)
    v.verificar_canal_pico(tiempos_s, canales)
    v.verificar_unidades(tiempos_s, grupos)
    v.verificar_curacion(tiempos_s, grupos, etiquetas)
"""
import numpy as np
import h5py

_ARCHIVO = None
TOLERANCIA_S = 0.0015          # 1,5 ms

# umbrales del curso, los mismos de la sección 7
MAX_RPV = 0.015
MAX_FALTANTES = 20.0
MIN_DISPAROS = 300


def usar(ruta):
    global _ARCHIVO
    with h5py.File(ruta, 'r') as f:
        if 'verdad' not in f:
            raise ValueError('Ese archivo no tiene el grupo verdad: no es el archivo del taller.')
    _ARCHIVO = ruta
    print(f'Verificación configurada con: {ruta}')


def _verdad():
    if _ARCHIVO is None:
        raise ValueError('Llame primero a v.usar("taller2_ruidosa.h5")')
    with h5py.File(_ARCHIVO, 'r') as f:
        return (f['verdad/tiempos_s'][:], f['verdad/unidad'][:],
                f['verdad/canal_pico'][:],
                f['verdad/unidades_buenas'][:],
                f['verdad/clase_objetivo'][:].astype('U12'))


def _barra(x, n=28):
    x = 0.0 if x != x else min(max(x, 0.0), 1.0)
    return '█' * int(round(x * n)) + '·' * (n - int(round(x * n)))


def _emparejar(det, real, tol=TOLERANCIA_S):
    """Empareja cada detección con un disparo real, sin reutilizar ninguno."""
    det = np.asarray(det, dtype=float)
    res = np.full(len(det), -1, dtype=int)
    usado = np.zeros(len(real), dtype=bool)
    pos = np.searchsorted(real, det)
    for i, (d, p) in enumerate(zip(det, pos)):
        mejor, mejor_d = -1, tol
        for j in (p - 2, p - 1, p, p + 1):
            if 0 <= j < len(real) and not usado[j]:
                dist = abs(real[j] - d)
                if dist <= mejor_d:
                    mejor, mejor_d = j, dist
        if mejor >= 0:
            usado[mejor] = True
            res[i] = mejor
    return res


def verificar_deteccion(tiempos_s, silencio=False):
    """Precisión, exhaustividad y F1 de la detección.

    La última línea, "de una unidad identificable", es la que importa para la
    sección 10: qué fracción de lo que usted conservó viene de una neurona que
    se puede identificar una por una. Vuelva a llamar esta función pasándole
    sólo los disparos de los grupos que NO marcó como noise ni como mua, y
    compare ese número antes y después.
    """
    t, u, _, _, _ = _verdad()
    d = np.sort(np.asarray(tiempos_s, dtype=float))
    m = _emparejar(d, t)
    vp = int((m >= 0).sum())
    fp = len(d) - vp
    prec = vp / len(d) if len(d) else 0.0

    # La exhaustividad se mide sobre las unidades bien aisladas. Los disparos de
    # las neuronas lejanas (la actividad multiunitaria) llegan casi todos por
    # debajo del umbral: no son detectables y contarlos como fallas no diría nada.
    aislados = u > 0
    enc = np.zeros(len(t), dtype=bool)
    enc[m[m >= 0]] = True
    exh = enc[aislados].sum() / aislados.sum() if aislados.sum() else 0.0
    f1 = 2 * prec * exh / (prec + exh) if (prec + exh) else 0.0
    # fracción de lo que usted conservó que viene de una neurona identificable
    de_aislada = int((m >= 0).sum() and (u[m[m >= 0]] > 0).sum())
    util = de_aislada / len(d) if len(d) else 0.0
    if not silencio:
        print(f'  detectados {len(d)}')
        print(f'  verdaderos positivos {vp}   falsos positivos {fp}')
        print(f'  precisión     {prec:5.3f}  {_barra(prec)}')
        print(f'  exhaustividad {exh:5.3f}  {_barra(exh)}   '
              f'(sobre los {int(aislados.sum())} disparos de unidades bien aisladas)')
        print(f'  F1            {f1:5.3f}  {_barra(f1)}')
        print(f'  faltan {int((~enc[aislados]).sum())} disparos de unidades bien aisladas')
        print()
        print(f'  de una unidad identificable  {util:5.3f}  {_barra(util)}')
        print('  ^ esta es la que sube al curar: qué fracción de lo que usted conservó')
        print('    viene de una neurona que se puede identificar una por una, y no de')
        print('    actividad multiunitaria ni de artefactos.')
    return dict(precision=prec, exhaustividad=exh, f1=f1, util=util)


def verificar_canal_pico(tiempos_s, canales):
    """¿Acertaron en qué sitio de la sonda es máxima cada forma de onda?"""
    t, u, cp, _, _ = _verdad()
    d = np.asarray(tiempos_s, dtype=float)
    c = np.asarray(canales, dtype=int)
    orden = np.argsort(d)
    d, c = d[orden], c[orden]
    m = _emparejar(d, t)
    ok = m >= 0
    if ok.sum() == 0:
        print('  ninguna detección se pudo emparejar'); return
    real = cp[m[ok]]
    exacto = np.mean(c[ok] == real)
    cerca = np.mean(np.abs(c[ok] - real) <= 1)
    print(f'  sobre {int(ok.sum())} detecciones emparejadas:')
    print(f'  canal exacto        {exacto:5.3f}  {_barra(exacto)}')
    print(f'  a un sitio o menos  {cerca:5.3f}  {_barra(cerca)}')
    print('  nota: en la actividad multiunitaria el canal pico es inestable por')
    print('  naturaleza, así que no espere acertarle a todo.')
    return dict(exacto=exacto, cerca=cerca)


def verificar_unidades(tiempos_s, grupos):
    """Matriz de confusión entre sus grupos y las unidades verdaderas."""
    t, u, _, buenas, _ = _verdad()
    d = np.asarray(tiempos_s, dtype=float)
    g = np.asarray(grupos)
    orden = np.argsort(d)
    d, g = d[orden], g[orden]
    m = _emparejar(d, t)
    clase = np.where(m >= 0, u[np.clip(m, 0, None)], 99)
    etiquetas = list(buenas) + [0, -1, 99]
    nombres = [f'U{b}' for b in buenas] + ['MUA', 'ruido', 'nada']
    gs = sorted(set(g.tolist()))
    print('  filas = sus grupos, columnas = la verdad\n')
    print('  ' + 'grupo'.ljust(8) + ''.join(n.rjust(7) for n in nombres) + '    n')
    for gi in gs:
        s = g == gi
        fila = [int(((clase == e) & s).sum()) for e in etiquetas]
        print('  ' + str(gi).ljust(8) + ''.join(str(x).rjust(7) for x in fila)
              + str(int(s.sum())).rjust(7))
    print('\n  Para cada unidad verdadera, cuánto de ella quedó en un solo grupo:')
    for e, n in zip(etiquetas[:len(buenas)], nombres[:len(buenas)]):
        s = clase == e
        if s.sum() == 0:
            continue
        cuentas = np.array([int(((g == gi) & s).sum()) for gi in gs])
        mejor = int(np.argmax(cuentas))
        recu = cuentas[mejor] / s.sum()
        pur = cuentas[mejor] / max(int((g == gs[mejor]).sum()), 1)
        print(f'   {n:5s} grupo {str(gs[mejor]):>4s}  recuperada {recu:5.3f} {_barra(recu, 14)}'
              f'   pura {pur:5.3f} {_barra(pur, 14)}')


def verificar_curacion(tiempos_s, grupos, etiquetas):
    """¿Acertaron las cuatro etiquetas?

    etiquetas: diccionario {grupo: 'good' | 'mua' | 'nonsomatic' | 'noise'}

    Para cada grupo suyo, el módulo mira de qué está hecho realmente y decide la
    etiqueta correcta con las mismas reglas de la sección 9: si lo que domina el
    grupo son artefactos es noise; si es la unidad no somática es nonsomatic; si
    es actividad multiunitaria, o una unidad buena contaminada o incompleta, es
    mua; si es una unidad buena limpia es good.
    """
    t, u, _, buenas, obj = _verdad()
    d = np.asarray(tiempos_s, dtype=float)
    g = np.asarray(grupos)
    orden = np.argsort(d)
    d, g = d[orden], g[orden]
    m = _emparejar(d, t)
    clase = np.where(m >= 0, u[np.clip(m, 0, None)], 99)
    objetivo = {int(b): o for b, o in zip(buenas, obj)}

    aciertos, total = 0, 0
    print('  grupo   su etiqueta     correcta      de qué está hecho')
    for gi in sorted(set(g.tolist())):
        s = g == gi
        n = int(s.sum())
        vals, cnt = np.unique(clase[s], return_counts=True)
        dom = int(vals[np.argmax(cnt)])
        frac = cnt.max() / n
        if dom == -1 or (dom == 99 and frac > 0.5):
            correcta = 'noise'
        elif dom == 0:
            correcta = 'mua'
        else:
            correcta = objetivo.get(dom, 'good')
            # una unidad buena repartida o contaminada ya no es good
            if correcta == 'good' and (frac < 0.75 or n < MIN_DISPAROS):
                correcta = 'mua'
        suya = str(etiquetas.get(gi, '?'))
        marca = 'ok ' if suya == correcta else '<-- '
        aciertos += suya == correcta
        total += 1
        comp = f'{frac:.0%} de ' + ('artefacto' if dom == -1 else
                                    'MUA' if dom == 0 else
                                    'nada (ruido térmico)' if dom == 99 else f'U{dom}')
        print(f'  {str(gi):5s}   {suya:12s} {marca}{correcta:12s} {comp}   n={n}')
    print(f'\n  etiquetas correctas: {aciertos} de {total}   '
          f'{_barra(aciertos / total if total else 0)}')
    return aciertos / total if total else 0.0
