# factor-lab

**Framework modular de investigación y producción cuantitativa cross-sectional para la
captura de la anomalía de calidad en renta variable (S&P 500).**

Implementa la hipótesis económica *Quality-Minus-Junk* (QMJ) de Asness, Frazzini y Pedersen:
empresas rentables, en crecimiento y financieramente seguras no están sistemáticamente
sobrevaloradas en proporción a esa calidad, y una cartera larga en calidad / corta en "junk"
captura un premio ajustado por riesgo persistente. `factor-lab` cubre el ciclo completo —
ingesta de datos contables y de mercado, construcción del score, neutralización de riesgo y
optimización convexa de portafolio — y entrega pesos objetivo diseñados para alimentar
directamente el motor de gestión de riesgo del mismo autor, [`var-engine`](https://github.com/amateogr/var-engine)
(C99, VaR paramétrico/histórico/Monte Carlo con backtesting Kupiec/Christoffersen). La
generación de alfa (`factor-lab`) y la gestión defensiva del riesgo (`var-engine`) están
diseñadas como dos mitades del mismo sistema, no como proyectos aislados.

---

## 1. Arquitectura del pipeline

**⚠️ Nota de Ejecución:** La carpeta `reports/` está excluida del control de versiones. Antes de ejecutar el Jupyter Notebook (`qmj_tearsheet.ipynb`), es obligatorio correr `python src/run_pipeline.py` en la terminal para generar el archivo `optimal_weights.csv`.

```
SEC EDGAR (XBRL companyfacts/frames)     yfinance (Adj Close)
        │                                        │
        ▼                                        ▼
xbrl_tag_mapper.py                      returns_ingestor.py
(fallback multi-tag, filed-date)        (precios → retornos diarios + SPY)
        │                                        │
        └──────────────┬─────────────────────────┘
                        ▼
              universe_selector.py
        (S&P 500 constituents + CIK + GICS sector)
                        │
                        ▼
              factor_builder.py
     ROE, Leverage → Winsorización → Z-score cross-sectional
                        │
                        ▼
               neutralizer.py
   Sector-neutral (de-mean GICS) → Beta-neutral (Rolling OLS + residualización)
                        │
                        ▼
            portfolio_optimizer.py
     Ledoit-Wolf shrinkage → QP dollar-neutral (cvxpy)
                        │
                        ▼
              run_pipeline.py  →  reports/optimal_weights.csv
                        │
                        ▼
         notebooks/qmj_tearsheet.ipynb  (research narrative)
```

### Estructura del repositorio

```
factor-lab/
├── src/
│   ├── data_ingest/
│   │   ├── universe_selector.py       # S&P 500 (Wikipedia) + mapeo CIK (SEC)
│   │   ├── xbrl_tag_mapper.py         # fundamentals SEC EDGAR, fallback + point-in-time
│   │   ├── returns_ingestor.py        # precios yfinance, retornos, panel largo
│   │   ├── test_universe_selector.py
│   │   ├── test_returns_ingestor.py
│   │   └── test_xbrl_tag_mapper.py
│   ├── factor_construct/
│   │   ├── factor_builder.py          # ROE, leverage, winsorización, z-score, qmj_score
│   │   └── test_factor_builder.py
│   ├── neutralize/
│   │   ├── neutralizer.py             # sector-neutral, rolling beta, beta-neutral
│   │   └── test_neutralizer.py
│   ├── optimizer/
│   │   ├── portfolio_optimizer.py     # Ledoit-Wolf + QP dollar-neutral (cvxpy)
│   │   └── test_portfolio_optimizer.py
│   ├── run_pipeline.py                # orquestador end-to-end
│   └── test_run_pipeline.py           # test de integración (mocks en fronteras de red)
├── notebooks/
│   └── qmj_tearsheet.ipynb            # research tearsheet
├── reports/
│   └── optimal_weights.csv            # output del último rebalanceo
└── cache/                             # companyfacts/, frames/, universe/, fundamentals/
```

> **Nota de estructura:** los tests están colocados junto a cada módulo (`test_x.py` junto a
> `x.py` dentro de su subpaquete), no en un directorio `tests/` centralizado — decisión
> deliberada para que cada subpaquete sea autocontenible y ejecutable de forma aislada
> (`pytest src/neutralize/`) sin depender de rutas relativas a un directorio hermano.

---

## 2. Ingesta de datos y mitigación del look-ahead bias

`xbrl_tag_mapper.py` consume dos endpoints de la API pública de SEC EDGAR:

- **`companyfacts`** (por CIK): historial completo de un concepto XBRL, con `filed` (fecha
  de disclosure) además de `end` (fin de periodo fiscal).
- **`frames`** (por concepto+periodo, todas las empresas): más eficiente para construir el
  panel cross-sectional, pero **no incluye `filed`** — solo `end`. Usarlo directamente para
  point-in-time introduce exactamente el sesgo que se busca evitar.

La función `as_of(df, fecha)` resuelve el valor conocido de un concepto en una fecha dada
usando `filed <= fecha`, no `end <= fecha`: un balance con cierre de ejercicio el 31 de
diciembre no está disponible para un modelo hasta que se publica el 10-K semanas o meses
después. Cuando un mismo periodo tiene múltiples versiones (10-K original + 10-K/A por
restatement), se conserva el historial completo de vintages y `as_of` selecciona la versión
con `filed` más reciente que aún sea anterior a la fecha de consulta — el valor "conocido"
cambia correctamente antes y después de una restatement.

**Fallback multi-tag:** distintos filers etiquetan el mismo concepto económico con tags
XBRL distintos (`NetIncomeLoss` vs. `ProfitLoss` vs.
`NetIncomeLossAvailableToCommonStockholdersBasic`). `CONCEPT_MAP` define listas de prioridad
por concepto; `resolve_concept_timeseries` prueba cada tag en orden y usa el primero con
cobertura para un periodo dado, sin descartar el resto del historial. Para `Liabilities`,
cuando el tag directo no existe, se deriva analíticamente vía la identidad contable:

```
Liabilities = LiabilitiesAndStockholdersEquity − StockholdersEquity
```

Toda respuesta de EDGAR se cachea en disco (JSON crudo por CIK/tag/periodo) para que
reruns no vuelvan a pegarle a la red — necesario dado el rate limit efectivo de EDGAR
(~10 req/s, con bloqueo de IP si no se declara un `User-Agent` con contacto real).

---

## 3. Construcción del factor QMJ y robustez estadística

`factor_builder.py` calcula dos componentes cross-sectional por periodo:

- **Profitability:** `ROE = NetIncomeLoss / StockholdersEquity`
- **Safety:** `Leverage = Liabilities / TotalAssets` (signo invertido antes de combinar —
  mayor apalancamiento implica menor calidad)

**Manejo de valores faltantes:** cuando `StockholdersEquity ≤ 0`, el ROE se enmascara a
`NaN` en lugar de calcularse — con equity negativo, el signo del ratio se invierte de forma
económicamente engañosa (una empresa en distress con pérdidas puede mostrar un ROE
positivo). Nunca se rellena con cero ni con la media del universo; una empresa con un solo
componente válido contribuye al score compuesto solo con lo que tiene, vía media con
`skipna=True`, no vía imputación.

**Winsorización:** antes del z-score, cada componente se trunca cross-sectionalmente por
periodo — percentiles `[1, 99]` por defecto, o alternativamente `mediana ± n·MAD` (Median
Absolute Deviation escalada por 1.4826 para consistencia con la desviación estándar bajo
normalidad), más robusto que percentiles en universos pequeños o muy sesgados.

**Z-score cross-sectional:** estandarización dentro de cada periodo (o sector-relativa, si
se provee `gics_sector`), con umbral mínimo de observaciones por grupo para evitar
z-scores sin significancia estadística en grupos chicos. El score compuesto final:

```
qmj_score = mean(roe_zscore, safety_zscore)     [skipna=True]
```

---

## 4. Neutralización multifactorial de riesgo

`neutralizer.py` remueve dos fuentes de riesgo no intencionado del score antes de pasarlo
al optimizador.

**Neutralización sectorial.** El sector GICS de cada constituyente se obtiene vía scraping
de la tabla de componentes del S&P 500 en Wikipedia (`universe_selector.py`), cruzado con
el CIK oficial de la SEC (`company_tickers.json`) para poder unir sector con fundamentals.
`sector_neutralize` calcula, para cada `(fecha, sector)`:

```
score_sector_neutral_i = (score_i − mean(score | sector)) / std(score | sector)
```

Esto asegura que una posición larga en una tecnológica refleje calidad relativa a otras
tecnológicas, no una apuesta direccional al sector completo.

**Neutralización de beta de mercado**, en dos etapas:

1. **Beta histórico por activo** — Rolling OLS (`statsmodels.regression.rolling.RollingOLS`)
   de los retornos del activo contra el proxy de mercado (SPY) sobre una ventana móvil
   (60 periodos por defecto):

   ```
   r_i,t = α_i + β_i,t · r_mkt,t + ε_i,t
   ```

2. **Residualización cross-sectional en la fecha de rebalanceo** — regresión transversal
   del score contra el beta estimado de cada activo:

   ```
   score_i = a + b · β_i + u_i
   ```

   El vector final de alfa es **u**, el residuo — por construcción de OLS, ortogonal a β
   (covarianza cero exacta contra la exposición a mercado estimada, no una aproximación).
   Fechas o ventanas con muestra insuficiente (`min_group_size`, `min_nobs`) devuelven
   `NaN` en vez de una estimación de beta o un residuo sin base estadística suficiente.

---

## 5. Optimización convexa de portafolios

`portfolio_optimizer.py` resuelve un programa cuadrático (QP) convexo vía `cvxpy`.

**Covarianza shrinkage.** La covarianza muestral simple es mal condicionada (o singular)
cuando el número de activos se aproxima o supera el número de observaciones — exactamente
el régimen de un universo tipo S&P 500 con un par de años de historia diaria. Se usa el
estimador de contracción de Ledoit-Wolf (`sklearn.covariance.LedoitWolf`), una combinación
convexa entre la covarianza muestral y un target estructurado, que garantiza una matriz
simétrica definida positiva (PSD) por construcción — condición necesaria para que
`cp.quad_form` sea un problema convexo bien planteado.

**Formulación del problema:**

```
maximizar      μᵀw − λ · wᵀΣw
sujeto a       ‖w‖₁ ≤ max_leverage        (apalancamiento bruto)
               −max_weight ≤ wᵢ ≤ max_weight    ∀i    (concentración)
               1ᵀw = 0                     (dollar-neutral estricto)
```

donde `μ` es el vector de scores QMJ ya neutralizado (sector + beta), `Σ` la covarianza
shrinkage, y `λ` (`risk_aversion`) el coeficiente que pondera penalización de riesgo contra
exposición al factor. Antes de optimizar, `expected_scores` y `cov_matrix` se alinean por
la intersección de sus índices — activos presentes en uno pero no en el otro se excluyen
explícitamente (logueado), nunca se rellenan con score o covarianza ficticios.

---

## 6. Guía de ejecución

### Requisitos

Python **3.10+** (el código usa genéricos nativos en anotaciones de dataclass:
`tuple[str, ...]`, `dict[str, int]`).

```bash
pip install -r requirements.txt
```

`requirements.txt` fija versiones exactas contra las que este proyecto fue probado
— en particular `yfinance`, `pandas` y `seaborn` tienen comentarios inline explicando
por qué la versión importa (cambios de API entre versiones que rompen silenciosamente
partes específicas del pipeline si no se fijan).

### Pipeline completo

Desde la raíz del proyecto:

```bash
python src/run_pipeline.py
```

El flag `use_real_sec` (en la llamada a `main()`, default `False`) controla la fuente de
fundamentals:

- **`use_real_sec=False`** (validación estructural): genera un panel contable **simulado**
  (`source_tag="SIMULATED"` en cada columna), para validar que el pipeline completo corre
  sin esperar cientos de llamadas rate-limited a EDGAR. **Nunca presentar estos resultados
  como investigación real** — verificar `source_tag` antes de interpretar cualquier output.
- **`use_real_sec=True`** (producción): descarga fundamentals reales vía
  `xbrl_tag_mapper.resolve_concept_panel` para los ~500 CIKs del universo. Antes de activarlo:
  1. Reemplazar el `User-Agent` placeholder en `HEADERS` (`src/data_ingest/xbrl_tag_mapper.py`
     y `src/data_ingest/universe_selector.py`) por un contacto real — EDGAR bloquea IPs sin
     User-Agent identificable.
  2. Esperar varios minutos en la primera corrida (rate limit ~10 req/s); las corridas
     posteriores usan la caché en disco.

El output se guarda en `reports/optimal_weights.csv` (`ticker`, `cik`, `weight`,
`gics_sector`). El tearsheet (`notebooks/qmj_tearsheet.ipynb`) lee ese archivo directamente
— re-ejecutar sus celdas después de cada corrida del pipeline para refrescar los gráficos.

### Suite de tests

```bash
pytest -v src/
```

Cada módulo se testea de forma aislada, sin dependencia de red — `requests.get`,
`pd.read_html` y `yfinance.download` se mockean vía `monkeypatch` contra fixtures que
replican la forma exacta de las respuestas reales de EDGAR/Wikipedia/Yahoo Finance. Los
tests cubren tanto comportamiento esperado como invariantes matemáticas verificables por
construcción — por ejemplo, `test_beta_neutralize_residuals_orthogonal_to_beta` no solo
verifica que el resultado esté en un rango razonable, sino que la covarianza entre el
residuo y el beta sea ≈0, una propiedad exacta de la regresión OLS, no una aproximación
esperada.

---

## Roadmap

- [ ] Descomposición de riesgo vía PCA sobre retornos residuales, exposición a Fama-French
      5 + momentum, para distinguir alpha genuino de beta disfrazado.
- [ ] Análisis de decay del alpha (half-life de la señal, turnover implicado).
- [ ] Reporte de capacidad (AUM máximo antes de que el market impact destruya el edge).
- [ ] Integración con `var-engine` para backtesting formal (Kupiec, Christoffersen) de la
      cartera resultante.

---

## Licencia

MIT — ver [`LICENSE.txt`](./LICENSE.txt). Misma licencia que [`var-engine`](https://github.com/amateogr/var-engine).