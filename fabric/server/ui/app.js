/* Experiment Fabric interface.
 *
 * Vanilla JavaScript, no build step, no dependencies -- the same constraint
 * the rest of the project works under, for the same reason: a tool that
 * claims to make results reproducible should not need a package manager to
 * display one.
 *
 * Every view answers a scientific question:
 *
 *   dashboard    what has been run, and did any of it fail?
 *   experiment   what was the claim, and what evidence backs it?
 *   analysis     how big is the effect, how uncertain, and how was it tested?
 *   graph        what produced this number?
 *   runs         what happened in each individual execution?
 *   certificate  has any of the evidence changed since it was recorded?
 *
 * Two rules are enforced throughout: never show an estimate without its
 * interval, and never signal a verdict with colour alone.
 */
'use strict';

const $ = (selector, root = document) => root.querySelector(selector);
const view = $('#view');

/* ------------------------------------------------------------------ util */

async function api(path, options) {
  const response = await fetch(path, options);
  const type = response.headers.get('content-type') || '';
  const body = type.includes('json') ? await response.json() : await response.text();
  if (!response.ok) {
    const detail = (body && body.error) || response.statusText;
    const error = new Error(detail);
    error.status = response.status;
    error.remedy = (body && body.remedy) || '';
    throw error;
  }
  return body;
}

/* Escape before inserting into markup. Condition names, dataset names and
 * failure messages all come from user-supplied files, and one of them
 * containing a closing tag must not be able to close ours. */
function esc(value) {
  return String(value === null || value === undefined ? '' : value)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

const num = (value, digits = 4) =>
  (value === null || value === undefined || Number.isNaN(value))
    ? '-' : Number(value).toFixed(digits);

const pval = (value) => {
  if (value === null || value === undefined || Number.isNaN(value)) return '-';
  return value < 0.0001 ? '<0.0001' : Number(value).toFixed(4);
};

const shortId = (id) => String(id || '').slice(0, 16);

const when = (seconds) =>
  seconds ? new Date(seconds * 1000).toLocaleString() : '-';

/* Chart gutters are fixed, and condition names are not. Truncate for the
 * axis and keep the full name in a <title> so hovering still tells the
 * truth. */
const clip = (text, width = 26) => {
  const value = String(text || '');
  return value.length <= width ? value : value.slice(0, width - 1) + '…';
};

/* A verdict's tone. The word is always shown too: colour is never the only
 * carrier of the distinction. */
function tone(verdict) {
  const v = String(verdict || '').toLowerCase();
  if (['improvement', 'completed', 'succeeded', 'pass', 'consistent', 'robust',
       'good', 'deterministic', 'load_bearing', 'ok'].includes(v)) return 'good';
  if (['regression', 'failed', 'fail', 'fragile', 'bad', 'diverges',
       'non-deterministic'].includes(v)) return 'bad';
  if (['partial', 'inconclusive', 'sensitive', 'undetermined', 'running',
       'queued', 'warn', 'fails_tolerance', 'pass with deviation'].includes(v)) return 'warn';
  return 'info';
}

const tag = (text, override) =>
  `<span class="tag ${override || tone(text)}">${esc(text)}</span>`;

function table(columns, rows, caption) {
  const head = columns.map((c) =>
    `<th${c.num ? ' class="num"' : ''}>${esc(c.label)}</th>`).join('');
  const body = rows.map((row) => '<tr>' + row.map((cell, index) => {
    const column = columns[index] || {};
    if (cell && cell.html) return `<td class="${column.num ? 'num' : ''} ${column.wrap ? 'wrap' : ''}">${cell.html}</td>`;
    return `<td class="${column.num ? 'num' : ''} ${column.wrap ? 'wrap' : ''}">${esc(cell)}</td>`;
  }).join('') + '</tr>').join('');
  return `<div class="scroll"><table>${caption ? `<caption>${esc(caption)}</caption>` : ''}` +
         `<thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function panel(title, inner) {
  return `<section class="panel">${title ? `<h3>${esc(title)}</h3>` : ''}${inner}</section>`;
}

function callout(level, title, text) {
  return `<div class="callout ${esc(level)}"><strong>${esc(title)}</strong>${esc(text)}</div>`;
}

function empty(heading, detail) {
  return `<div class="empty"><h2>${esc(heading)}</h2><p class="muted">${esc(detail)}</p></div>`;
}

/* --------------------------------------------------------------- charts */

/* A forest plot: the estimate with its interval, which is the only honest
 * way to show a difference. An interval crossing zero is drawn crossing
 * zero, and the reader can see for themselves. */
function forest(rows, label) {
  if (!rows.length) return '';
  const width = 700, rowHeight = 30, left = 200, right = 75;
  const height = rows.length * rowHeight + 46;
  const lows = rows.map((r) => r.low), highs = rows.map((r) => r.high);
  let lo = Math.min(0, ...lows), hi = Math.max(0, ...highs);
  if (hi - lo < 1e-12) { lo -= 0.5; hi += 0.5; }
  const pad = (hi - lo) * 0.12;
  lo -= pad; hi += pad;
  const x = (value) => left + ((value - lo) / (hi - lo)) * (width - left - right);

  const marks = rows.map((row, index) => {
    const y = 30 + index * rowHeight;
    const cls = tone(row.verdict) === 'good' ? '' :
                tone(row.verdict) === 'bad' ? ' bad' : ' warn';
    return `
      <text x="${left - 10}" y="${y + 4}" text-anchor="end"
        ><title>${esc(row.label)}</title>${esc(clip(row.label))}</text>
      <line class="bar${cls}" x1="${x(row.low).toFixed(1)}" y1="${y}"
            x2="${x(row.high).toFixed(1)}" y2="${y}"/>
      <circle class="point${cls}" cx="${x(row.estimate).toFixed(1)}" cy="${y}" r="4"/>
      <text class="num" x="${width - right + 8}" y="${y + 4}">${esc(row.note || '')}</text>`;
  }).join('');

  return `<svg class="forest" viewBox="0 0 ${width} ${height}" role="img"
      aria-label="${esc(label || 'Estimated differences with confidence intervals')}">
    <line class="zero" x1="${x(0).toFixed(1)}" y1="14" x2="${x(0).toFixed(1)}" y2="${height - 26}"/>
    <line class="rule" x1="${left}" y1="${height - 26}" x2="${width - right}" y2="${height - 26}"/>
    <text class="num" x="${left}" y="${height - 10}">${num(lo, 3)}</text>
    <text class="num" x="${x(0).toFixed(1)}" y="${height - 10}" text-anchor="middle">0</text>
    <text class="num" x="${width - right}" y="${height - 10}" text-anchor="end">${num(hi, 3)}</text>
    ${marks}
  </svg>`;
}

/* Per-seed values, so the reader sees the spread rather than a mean that
 * hides it. */
function strip(series, label) {
  const names = Object.keys(series);
  if (!names.length) return '';
  const width = 700, rowHeight = 34, left = 200;
  const height = names.length * rowHeight + 30;
  const all = names.flatMap((name) => series[name]);
  let lo = Math.min(...all), hi = Math.max(...all);
  if (hi - lo < 1e-12) { lo -= 0.01; hi += 0.01; }
  const pad = (hi - lo) * 0.1;
  lo -= pad; hi += pad;
  const x = (v) => left + ((v - lo) / (hi - lo)) * (width - left - 30);

  const marks = names.map((name, index) => {
    const y = 22 + index * rowHeight;
    const values = series[name];
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const dots = values.map((v) =>
      `<circle class="dot" cx="${x(v).toFixed(1)}" cy="${y}" r="3.5"/>`).join('');
    return `<text x="${left - 10}" y="${y + 4}" text-anchor="end"
        ><title>${esc(name)}</title>${esc(clip(name))}</text>
      ${dots}
      <line class="mean" x1="${x(mean).toFixed(1)}" y1="${y - 9}"
            x2="${x(mean).toFixed(1)}" y2="${y + 9}"/>`;
  }).join('');

  return `<svg class="strip forest" viewBox="0 0 ${width} ${height}" role="img"
      aria-label="${esc(label || 'Per-seed values by condition')}">
    ${marks}
    <text class="num" x="${left}" y="${height - 6}">${num(lo, 3)}</text>
    <text class="num" x="${width - 30}" y="${height - 6}" text-anchor="end">${num(hi, 3)}</text>
  </svg>`;
}

/* ------------------------------------------------------------- rendering */

function crumbs(parts) {
  $('#crumbs').innerHTML = parts.map((p) =>
    p.href ? `<span><a href="${esc(p.href)}">${esc(p.text)}</a></span>`
           : `<span>${esc(p.text)}</span>`).join('');
}

async function health() {
  const element = $('#health');
  try {
    const body = await api('/health');
    const jobs = body.jobs || {};
    element.className = 'health ok';
    element.textContent =
      `v${body.fabric_version} · ${body.execution_enabled ? 'execute on' : 'execute off'}` +
      ` · ${jobs.running || 0} running · ${jobs.queued || 0} queued`;
    $('#footnote').textContent =
      `Workspace ${body.workspace}. ` +
      (body.execution_enabled
        ? 'This server executes submitted code as its own user, with no sandbox.'
        : 'This server validates and plans submissions but will not execute them.');
    return body;
  } catch (error) {
    element.className = 'health down';
    element.textContent = 'server unreachable';
    return null;
  }
}

/* ------------------------------------------------------------ dashboard */

async function dashboard() {
  crumbs([]);
  const [listing, jobs] = await Promise.all([
    api('/experiments?limit=100'),
    api('/jobs?limit=20').catch(() => ({ jobs: [], counts: {} })),
  ]);
  const experiments = listing.experiments || [];

  if (!experiments.length) {
    view.innerHTML = empty(
      'No experiments yet',
      'Submit one with: curl --data-binary @experiment.yaml ' +
      location.origin + '/experiments');
    return;
  }

  const counts = experiments.reduce((acc, e) => {
    acc[e.status] = (acc[e.status] || 0) + 1;
    acc.runs += e.runs.completed;
    acc.failed += e.runs.failed;
    return acc;
  }, { runs: 0, failed: 0 });

  const stats = `
    <div class="grid">
      <div class="stat"><div class="label">Experiments</div>
        <div class="value">${experiments.length}</div>
        <div class="note">${counts.completed || 0} complete, ${counts.partial || 0} partial, ${counts.planned || 0} planned</div></div>
      <div class="stat"><div class="label">Runs completed</div>
        <div class="value">${counts.runs}</div>
        <div class="note">across every experiment in this workspace</div></div>
      <div class="stat"><div class="label">Runs failed</div>
        <div class="value">${counts.failed}</div>
        <div class="note">${counts.failed ? 'a failed run is still evidence; see the failure analysis' : 'none'}</div></div>
      <div class="stat"><div class="label">Queue</div>
        <div class="value">${(jobs.counts && jobs.counts.queued) || 0}</div>
        <div class="note">${(jobs.counts && jobs.counts.running) || 0} running now</div></div>
    </div>`;

  const rows = experiments.map((e) => [
    { html: `<a href="#/e/${esc(e.experiment_id)}">${esc(e.experiment_id)}</a>` },
    e.name,
    { html: tag(e.status) },
    `${e.runs.completed}/${e.runs.planned}`,
    e.runs.failed || '',
    when(e.created_at),
  ]);

  view.innerHTML = `
    <h1>Experiments</h1>
    <p class="lede">Every experiment in this workspace. A status of
      <em>partial</em> means some runs are missing or failed, and its
      conclusions rest on the ones that finished.</p>
    ${stats}
    ${panel('', table([
      { label: 'id' }, { label: 'name' }, { label: 'status' },
      { label: 'runs', num: true }, { label: 'failed', num: true },
      { label: 'created' },
    ], rows))}
    ${jobs.jobs && jobs.jobs.length ? panel('Recent jobs', table([
      { label: 'job' }, { label: 'kind' }, { label: 'experiment' },
      { label: 'state' }, { label: 'progress', num: true }, { label: 'message', wrap: true },
    ], jobs.jobs.map((j) => [
      shortId(j.id), j.kind,
      { html: `<a href="#/e/${esc(j.experiment_id)}">${esc(j.experiment_id)}</a>` },
      { html: tag(j.state) },
      j.total ? `${j.progress}/${j.total}` : '-',
      j.message || '',
    ]))) : ''}`;
}

/* ----------------------------------------------------------- experiment */

function tabs(id, active) {
  const items = [
    ['', 'Overview'], ['analysis', 'Analysis'], ['runs', 'Runs'],
    ['graph', 'Provenance'], ['certificate', 'Certificate'],
  ];
  return `<nav class="tabs">${items.map(([slug, label]) =>
    `<a class="${active === slug ? 'on' : ''}" href="#/e/${esc(id)}${slug ? '/' + slug : ''}">${esc(label)}</a>`
  ).join('')}</nav>`;
}

async function experimentPage(id) {
  const body = await api(`/experiments/${encodeURIComponent(id)}`);
  crumbs([{ text: 'experiments', href: '#/' }, { text: body.experiment_id }]);

  const b = body.benchmark;
  const unmet = b ? (b.constraints || []).filter((c) => !c.satisfied) : [];

  const conditions = table([
    { label: 'condition' }, { label: 'role' }, { label: 'removed' },
    { label: 'perturbs' }, { label: 'configuration', wrap: true },
  ], body.conditions.map((c) => [
    c.name, { html: tag(c.role) }, (c.removed || []).join(', ') || '-',
    (c.perturbation && c.perturbation.axis) || '-',
    Object.entries(c.config || {}).map(([k, v]) => `${k}=${v}`).join(', ') || '-',
  ]), 'Every experimental arm the specification expanded to');

  const inputs = ['dataset', 'model', 'algorithm'].map((kind) => {
    const entity = body.inputs[kind];
    if (!entity) return '';
    const fields = {
      dataset: ['version', 'digest', 'n_examples', 'license', 'parent'],
      model: ['version', 'weights_digest', 'architecture', 'precision',
              'quantization', 'tokenizer', 'checkpoint', 'repository'],
      algorithm: ['entrypoint', 'commit', 'source_digest', 'dirty'],
    }[kind];
    const rows = fields
      .filter((f) => entity[f] !== undefined && entity[f] !== '' && entity[f] !== null)
      .map((f) => [f, { html: `<code>${esc(entity[f])}</code>` }]);
    rows.unshift(['name', entity.name]);
    rows.push(['identifier', { html: `<code>${esc(entity.id)}</code>` }]);
    return panel(kind, table([{ label: 'field' }, { label: 'value', wrap: true }], rows));
  }).join('');

  view.innerHTML = `
    <h1>${esc(body.name)}</h1>
    <p class="lede">${esc(body.research_question)}</p>
    ${tabs(body.experiment_id, '')}
    ${body.hypothesis ? callout('info', 'Hypothesis, stated before the runs', body.hypothesis) : ''}
    <div class="grid">
      <div class="stat"><div class="label">Status</div>
        <div class="value">${esc(body.status)}</div>
        <div class="note">${body.runs.completed} of ${body.runs.planned} runs, ${body.runs.failed} failed</div></div>
      <div class="stat"><div class="label">Primary metric</div>
        <div class="value">${esc(body.primary_metric || '-')}</div>
        <div class="note">${body.metrics.length} metric(s) declared</div></div>
      <div class="stat"><div class="label">Seeds</div>
        <div class="value">${body.seeds.length}</div>
        <div class="note">${esc(body.seeds.slice(0, 12).join(', '))}</div></div>
      <div class="stat"><div class="label">Specification</div>
        <div class="value mono" style="font-size:13px">${esc(shortId(body.spec_digest.replace('sha256:', '')))}</div>
        <div class="note"><a href="/experiments/${esc(body.experiment_id)}/spec">view the YAML</a></div></div>
    </div>
    ${b ? panel('Benchmark', `
      <p class="muted">${esc(b.description)}</p>
      <p>Claims <code>${esc(b.ref)}</code> at digest <code>${esc(shortId(b.digest))}</code>.</p>
      ${unmet.length ? callout('warn', `${unmet.length} constraint(s) unmet`,
          unmet.map((c) => c.detail).join('; ') +
          '. The results stand on their own, but calling them results on this ' +
          'benchmark claims a comparability this design does not have.')
        : callout('good', 'Conforms', 'Every constraint this benchmark imposes is satisfied.')}
      ${table([{ label: 'constraint' }, { label: 'met' }, { label: 'detail', wrap: true }],
        (b.constraints || []).map((c) => [c.kind,
          { html: tag(c.satisfied ? 'ok' : 'unmet', c.satisfied ? 'good' : 'warn') },
          c.detail]))}`) : ''}
    ${body.advisories.length ? panel('Advisories',
      body.advisories.map((a) => callout('warn', 'Before you spend the compute', a)).join('')) : ''}
    ${panel('Conditions', conditions)}
    ${inputs}`;
}

/* ------------------------------------------------------------- analysis */

async function analysisPage(id) {
  crumbs([{ text: 'experiments', href: '#/' },
          { text: id, href: `#/e/${id}` }, { text: 'analysis' }]);
  view.innerHTML = `<h1>Analysis</h1>${tabs(id, 'analysis')}<p class="muted">Analysing&hellip;</p>`;

  let body;
  try {
    body = await api(`/experiments/${encodeURIComponent(id)}/analysis`);
  } catch (error) {
    view.innerHTML = `<h1>Analysis</h1>${tabs(id, 'analysis')}` +
      empty('Nothing to analyse yet', error.remedy || error.message);
    return;
  }

  const metric = body.primary_metric;
  const level = `${(body.confidence_level * 100).toFixed(10).replace(/\.?0+$/, '')}%`;
  const comparisons = (body.baselines.comparisons || []).filter((c) => c.metric === metric);

  const series = {};
  (body.conditions || []).forEach((c) => {
    const values = ((c.metrics || {})[metric] || {}).values;
    // An empty object is truthy in JavaScript, so `!c.perturbation` would
    // exclude every condition rather than only the perturbed ones.
    const perturbed = Object.keys(c.perturbation || {}).length > 0;
    if (values && values.length && !perturbed) series[c.name] = values;
  });

  const comparisonRows = comparisons.map((c) => [
    c.baseline,
    num(c.treatment_summary.mean), num(c.baseline_summary.mean),
    num(c.effects.find((e) => e.name === 'absolute_difference')?.value),
    `[${num(c.difference_interval.low, 3)}, ${num(c.difference_interval.high, 3)}]`,
    c.test.test, pval(c.test.p_value), pval(c.adjusted_p),
    num((c.effects.find((e) => e.name === 'hedges_g') || {}).value, 2),
    { html: tag(c.verdict) },
  ]);

  const forestRows = comparisons
    .filter((c) => Number.isFinite(c.difference_interval.low))
    .map((c) => ({
      label: c.baseline,
      estimate: (c.effects.find((e) => e.name === 'absolute_difference') || {}).value,
      low: c.difference_interval.low,
      high: c.difference_interval.high,
      verdict: c.verdict,
      note: 'p=' + pval(c.adjusted_p !== null ? c.adjusted_p : c.test.p_value),
    }));

  const references = (body.references && body.references.comparisons) || [];
  const axes = (body.reproducibility && body.reproducibility.axes) || [];
  const ablations = (body.ablations && body.ablations.effects) || [];

  view.innerHTML = `
    <h1>Analysis</h1>
    <p class="lede">${esc(body.headline)}</p>
    ${tabs(id, 'analysis')}

    ${panel('How this was read', `
      <div class="grid">
        <div class="stat"><div class="label">Significance level</div>
          <div class="value">${body.alpha}</div>
          <div class="note">with ${esc(body.correction)} correction across the family</div></div>
        <div class="stat"><div class="label">Confidence level</div>
          <div class="value">${esc(level)}</div>
          <div class="note">applied to every interval on this page</div></div>
        <div class="stat"><div class="label">Primary metric</div>
          <div class="value">${esc(metric)}</div>
          <div class="note">declared before the runs</div></div>
      </div>`)}

    ${forestRows.length ? panel(`Difference in ${metric} against each baseline`,
      forest(forestRows, `Differences in ${metric} with ${level} intervals`) +
      `<p class="muted">An interval crossing zero means the difference is
        unresolved at this sample size &mdash; not that it is absent.</p>`) : ''}

    ${comparisonRows.length ? panel('Baseline comparisons', table([
      { label: 'baseline' }, { label: 'treatment', num: true },
      { label: 'baseline mean', num: true }, { label: 'difference', num: true },
      { label: `${level} CI`, num: true }, { label: 'test' },
      { label: 'p', num: true }, { label: 'adjusted p', num: true },
      { label: "hedges' g", num: true }, { label: 'verdict' },
    ], comparisonRows, 'Difference is signed so that positive always means better'))
      : panel('Baselines', '<p class="muted">No baselines were declared, so nothing ' +
        'here establishes that the method is better than an alternative.</p>')}

    ${Object.keys(series).length ? panel(`Every run, by condition`,
      strip(series, `Per-seed ${metric} by condition`) +
      '<p class="muted">One dot per seed; the bar is the mean. A wide spread ' +
      'here is the noise floor every claim above has to clear.</p>') : ''}

    ${references.length ? panel('Published reference values', table([
      { label: 'arm' }, { label: 'metric' }, { label: 'published', num: true },
      { label: 'observed', num: true }, { label: 'gap', num: true },
      { label: 'in seed noise', num: true }, { label: 'verdict' }, { label: 'source', wrap: true },
    ], references.map((r) => [
      r.condition, r.metric, num(r.published), num(r.observed.mean),
      num(r.difference), Number.isFinite(r.noise_multiple) ? num(r.noise_multiple, 2) + 'x' : '-',
      { html: tag(r.verdict) }, r.source || 'not cited',
    ]), 'Cited numbers, not runs: described, never tested')) : ''}

    ${ablations.length ? panel('Ablations', table([
      { label: 'removed' }, { label: 'change', num: true },
      { label: 'interval', num: true }, { label: 'detectable' },
    ], ablations.map((e) => {
      const c = (e.comparisons || {})[metric];
      if (!c) return [e.ablation, '-', '-', '-'];
      const d = (c.effects.find((x) => x.name === 'absolute_difference') || {}).value;
      return [e.ablation, num(-d),
        `[${num(-c.difference_interval.high, 3)}, ${num(-c.difference_interval.low, 3)}]`,
        { html: tag(c.verdict === 'improvement' || c.verdict === 'regression'
            ? 'load-bearing' : 'not detected',
          c.verdict === 'improvement' || c.verdict === 'regression' ? 'good' : 'info') }];
    }), 'How much each component contributes, with the uncertainty on that estimate')) : ''}

    ${axes.length ? panel('Reproducibility under controlled perturbation', table([
      { label: 'axis' }, { label: 'perturbation' }, { label: 'stability', num: true },
      { label: 'shift', num: true }, { label: 'in seed noise', num: true },
      { label: 'tolerance' },
    ], axes.map((a) => [
      a.axis, a.label, num(a.stability, 3), num(a.delta_mean),
      num(a.noise_multiples, 1) + 'x', { html: tag(a.tolerance_verdict) },
    ]), 'One thing changed per row; the seed axis is the yardstick, not a score')) : ''}

    ${(body.limitations || []).length ? panel('Limitations',
      body.limitations.map((l) => callout('warn', 'Limitation', l)).join('')) : ''}`;
}

/* ----------------------------------------------------------------- runs */

async function runsPage(id) {
  crumbs([{ text: 'experiments', href: '#/' },
          { text: id, href: `#/e/${id}` }, { text: 'runs' }]);
  const body = await api(`/experiments/${encodeURIComponent(id)}/runs`);
  const runs = body.runs || [];
  const metrics = [...new Set(runs.flatMap((r) => Object.keys(r.metrics || {})))];

  const columns = [
    { label: 'run' }, { label: 'condition' }, { label: 'seed', num: true },
    { label: 'rep', num: true }, { label: 'status' },
    ...metrics.map((m) => ({ label: m, num: true })),
    { label: 'wall s', num: true }, { label: 'artifacts', num: true },
    { label: 'failure', wrap: true },
  ];
  const rows = runs.map((r) => [
    { html: `<a href="#/e/${esc(id)}/run/${esc(r.id)}">${esc(shortId(r.id))}</a>` },
    r.condition, r.seed, r.replicate || '',
    { html: tag(r.status) },
    ...metrics.map((m) => num((r.metrics || {})[m])),
    num(r.wall_seconds, 3), r.artifacts || '',
    r.failure ? `${r.failure.classification}: ${r.failure.message || ''}` : '',
  ]);

  view.innerHTML = `<h1>Runs</h1>
    <p class="lede">Every execution, with the seed it used and what it produced.
      A failed run is kept, classified and counted &mdash; it is evidence about
      the method, not an inconvenience.</p>
    ${tabs(id, 'runs')}
    ${panel('', table(columns, rows, `${runs.length} runs`))}`;
}

async function runPage(id, runId) {
  crumbs([{ text: 'experiments', href: '#/' }, { text: id, href: `#/e/${id}` },
          { text: 'runs', href: `#/e/${id}/runs` }, { text: shortId(runId) }]);
  const r = await api(`/experiments/${encodeURIComponent(id)}/runs/${encodeURIComponent(runId)}`);

  const facts = [
    ['run', r.id], ['condition', r.condition_name], ['seed', r.seed],
    ['replicate', r.replicate], ['status', r.status],
    ['environment', r.environment_id], ['hardware', r.hardware_id],
    ['algorithm', r.algorithm_id], ['dataset', r.dataset_id], ['model', r.model_id],
  ].filter(([, v]) => v !== '' && v !== null && v !== undefined);

  const resources = r.resources || {};
  const energy = resources.energy_joules === null || resources.energy_joules === undefined
    ? 'not measured &mdash; no energy_watts was declared'
    : `${num(resources.energy_joules / 3600, 3)} Wh &mdash; ESTIMATED from a declared ` +
      `${resources.energy_watts} W, not measured`;

  view.innerHTML = `<h1>Run ${esc(shortId(r.id))}</h1>
    <p class="lede">One execution of <code>${esc(r.condition_name)}</code> at seed
      ${esc(r.seed)}. Everything below was recorded at the moment it ran.</p>
    ${tabs(id, 'runs')}
    ${r.failure ? callout('bad', `Failed: ${r.failure.classification}`,
        r.failure.message || 'no message recorded') : ''}
    ${panel('Identity', table([{ label: 'field' }, { label: 'value', wrap: true }],
      facts.map(([k, v]) => [k, { html: `<code>${esc(v)}</code>` }])))}
    ${panel('Metrics', table([{ label: 'metric' }, { label: 'value', num: true }],
      Object.entries(r.metrics || {}).map(([k, v]) => [k, num(v, 6)])))}
    ${panel('Configuration', table([{ label: 'parameter' }, { label: 'value' }],
      Object.entries(r.config || {}).map(([k, v]) => [k, String(v)])))}
    ${panel('Resources', table([{ label: 'measure' }, { label: 'value', wrap: true }], [
      ['wall seconds', num(resources.wall_seconds, 4) + ' (measured)'],
      ['cpu seconds', num(resources.cpu_seconds, 4) + ' (measured)'],
      ['peak memory', resources.peak_rss_bytes
        ? (resources.peak_rss_bytes / 1048576).toFixed(1) + ' MiB (measured)' : 'not measured'],
      ['energy', { html: energy }],
      ['cost', resources.cost_usd === null || resources.cost_usd === undefined
        ? 'not priced' : '$' + num(resources.cost_usd, 4) + ' (ESTIMATED)'],
    ]))}
    ${(r.artifacts || []).length ? panel('Artifacts', table([
      { label: 'path' }, { label: 'bytes', num: true }, { label: 'digest', wrap: true },
    ], r.artifacts.map((a) => [a.path, a.size_bytes, { html: `<code>${esc(a.digest)}</code>` }]),
      'Each one hashed when it was written, and re-checked by fabric verify')) : ''}`;
}

/* ------------------------------------------------------------ provenance */

async function graphPage(id) {
  crumbs([{ text: 'experiments', href: '#/' },
          { text: id, href: `#/e/${id}` }, { text: 'provenance' }]);
  view.innerHTML = `<h1>Provenance</h1>${tabs(id, 'graph')}<p class="muted">Building the graph&hellip;</p>`;

  const [graph, why] = await Promise.all([
    api(`/experiments/${encodeURIComponent(id)}/graph`),
    api(`/experiments/${encodeURIComponent(id)}/why`).catch(() => null),
  ]);

  const byKind = {};
  (graph.nodes || []).forEach((n) => { byKind[n.kind] = (byKind[n.kind] || 0) + 1; });
  const byRelation = {};
  (graph.edges || []).forEach((e) => { byRelation[e.relation] = (byRelation[e.relation] || 0) + 1; });

  const trace = why ? `<pre class="block tree">${esc(renderWhy(why))}</pre>` :
    '<p class="muted">No metric has been recorded yet, so there is nothing to trace.</p>';

  view.innerHTML = `<h1>Provenance</h1>
    <p class="lede">Every node has a stable identifier and every edge has a
      meaning. This is what makes &ldquo;why does this number exist&rdquo; a
      query rather than an archaeology exercise.</p>
    ${tabs(id, 'graph')}
    ${panel('Tracing the primary metric back to what produced it', trace)}
    <div class="grid">
      ${panel('Nodes', table([{ label: 'kind' }, { label: 'count', num: true }],
        Object.entries(byKind).sort((a, b) => b[1] - a[1])))}
      ${panel('Relations', table([{ label: 'relation' }, { label: 'count', num: true }],
        Object.entries(byRelation).sort((a, b) => b[1] - a[1])))}
    </div>
    <p class="muted">The full interactive graph, with click-to-highlight lineage,
      is embedded in the generated report:
      <a href="/experiments/${esc(id)}/report?format=html">open the HTML report</a>.</p>`;
}

/* The same tree `fabric why` prints in a terminal, from the same payload. */
function renderWhy(why) {
  const lines = [`${why.metric} = ${why.value}`, ''];
  (why.chain || []).forEach((step, index) => {
    const indent = '  '.repeat(index);
    lines.push(`${indent}${step.kind}`);
    lines.push(`${indent} └─ ${step.label || step.id}`);
    Object.entries(step.detail || {}).forEach(([key, value]) => {
      if (value === '' || value === null || value === undefined) return;
      const text = typeof value === 'object'
        ? Object.entries(value).map(([k, v]) => `${k}=${v}`).join(', ')
        : String(value);
      if (text) lines.push(`${indent}    ${key}: ${text}`);
    });
    lines.push(`${indent}    id: ${step.id}`);
  });
  (why.artifacts || []).forEach((artifact) => {
    lines.push('');
    lines.push(`artifact  ${artifact.path || artifact.label || ''}`);
    if (artifact.digest) lines.push(`    ${artifact.digest}`);
  });
  if ((why.missing || []).length) {
    lines.push('');
    lines.push(`not recorded: ${why.missing.join(', ')}`);
    lines.push('(Fabric reports the gaps in a provenance chain rather than');
    lines.push(' letting an absent link look like a present one.)');
  }
  return lines.join('\n');
}

/* ----------------------------------------------------------- certificate */

async function certificatePage(id) {
  crumbs([{ text: 'experiments', href: '#/' },
          { text: id, href: `#/e/${id}` }, { text: 'certificate' }]);
  view.innerHTML = `<h1>Certificate</h1>${tabs(id, 'certificate')}<p class="muted">Issuing&hellip;</p>`;

  let cert;
  try {
    cert = await api(`/experiments/${encodeURIComponent(id)}/certificate`);
  } catch (error) {
    view.innerHTML = `<h1>Certificate</h1>${tabs(id, 'certificate')}` +
      empty('No certificate yet', error.remedy || error.message);
    return;
  }

  const checks = cert.checks || [];
  const failed = checks.filter((c) => c.status === 'fail');
  const level = failed.length ? 'bad' : 'good';

  view.innerHTML = `<h1>Certificate</h1>
    <p class="lede">A certificate records what was hashed and how, so that a
      later change to any of it is detectable. It is evidence-integrity
      verification &mdash; not proof that the science is correct.</p>
    ${tabs(id, 'certificate')}
    ${callout(level, failed.length ? `${failed.length} check(s) failed` : 'All checks passed',
      cert.summary || '')}
    ${panel('Checks', table([
      { label: 'check' }, { label: 'status' }, { label: 'detail', wrap: true },
    ], checks.map((c) => [c.name, { html: tag(c.status) }, c.detail || ''])))}
    ${panel('Identity', table([{ label: 'field' }, { label: 'value', wrap: true }], [
      ['certificate', { html: `<code>${esc(cert.id || '')}</code>` }],
      ['experiment', { html: `<code>${esc(cert.experiment_id || '')}</code>` }],
      ['specification', { html: `<code>${esc(cert.spec_digest || '')}</code>` }],
      ['fabric version', cert.fabric_version || ''],
      ['issued', when(cert.issued_at)],
    ]))}
    <p class="muted">Verify an exported bundle offline with
      <code>fabric verify &lt;bundle&gt;</code>. Modify any file in it and the
      verification fails.</p>`;
}

/* -------------------------------------------------------------- routing */

const routes = [
  [/^\/?$/, dashboard],
  [/^\/e\/([^/]+)$/, experimentPage],
  [/^\/e\/([^/]+)\/analysis$/, analysisPage],
  [/^\/e\/([^/]+)\/runs$/, runsPage],
  [/^\/e\/([^/]+)\/run\/(.+)$/, runPage],
  [/^\/e\/([^/]+)\/graph$/, graphPage],
  [/^\/e\/([^/]+)\/certificate$/, certificatePage],
];

async function route() {
  const path = decodeURIComponent(location.hash.replace(/^#/, '')) || '/';
  for (const [pattern, handler] of routes) {
    const match = pattern.exec(path);
    if (match) {
      try {
        await handler(...match.slice(1));
      } catch (error) {
        view.innerHTML = empty(
          error.status === 404 ? 'Not found' : 'Something went wrong',
          (error.message || '') + (error.remedy ? ' — ' + error.remedy : ''));
      }
      window.scrollTo(0, 0);
      return;
    }
  }
  view.innerHTML = empty('No such page', path);
}

window.addEventListener('hashchange', route);
route();
health();
setInterval(health, 5000);
