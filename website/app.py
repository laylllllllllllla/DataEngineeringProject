"""
NYC Homes & Streets: housing maintenance problems × serious traffic crashes, 2022–2025.
Dash website for DEV Milestone 1.

Run locally:   python website/app.py        → http://127.0.0.1:8050
Deployed with: gunicorn --chdir website app:server --timeout 180
"""
import os
import re
from functools import lru_cache

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from dash import Dash, html, dcc, Input, Output, State

import charts

# ----------------------------------------------------------------------
# Data (small pre-aggregated files exported in notebook Section 6)
# ----------------------------------------------------------------------
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web_data')
df_integrated = pd.read_csv(os.path.join(DATA, 'integrated_cd_year.csv'))
df_housing_monthly = pd.read_parquet(os.path.join(DATA, 'housing_monthly.parquet'))
df_housing_nta = pd.read_parquet(os.path.join(DATA, 'housing_nta.parquet'))
df_centroids = pd.read_csv(os.path.join(DATA, 'cd_centroids.csv'))
df_crashes = pd.read_parquet(os.path.join(DATA, 'crashes.parquet'))

# Consistent types, whatever pandas version wrote the files
for _df in (df_integrated, df_housing_monthly, df_housing_nta, df_crashes):
    _df['year'] = pd.to_numeric(_df['year'], errors='coerce').astype('Int64')
    _df['borough'] = _df['borough'].astype(str).str.upper()
df_centroids['borough'] = df_centroids['borough'].astype(str).str.upper()
df_integrated['year'] = df_integrated['year'].astype(int)
df_housing_monthly['month'] = df_housing_monthly['month'].astype(int)
for c in ('major_category', 'type'):
    df_housing_monthly[c] = df_housing_monthly[c].astype(str)
df_crashes['crash_hour'] = pd.to_numeric(df_crashes['crash_hour'], errors='coerce')

BOROUGHS = sorted(df_integrated['borough'].unique())
YEARS = sorted(int(y) for y in df_integrated['year'].unique())
CATEGORIES = sorted(df_housing_monthly['major_category'].unique())
ALL = 'ALL'

# ----------------------------------------------------------------------
# One visual theme for every chart
# ----------------------------------------------------------------------
pio.templates['nyc'] = go.layout.Template(layout=dict(
    font=dict(family='Inter, Segoe UI, system-ui, sans-serif', size=12, color='#334155'),
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    colorway=['#2563eb', '#f97316', '#10b981', '#8b5cf6', '#eab308', '#ef4444', '#06b6d4'],
    xaxis=dict(gridcolor='#eef2f7', zerolinecolor='#cbd5e1', linecolor='#e2e8f0'),
    yaxis=dict(gridcolor='#eef2f7', zerolinecolor='#cbd5e1', linecolor='#e2e8f0'),
    hoverlabel=dict(bgcolor='#0f172a', font=dict(color='white', family='Inter, sans-serif')),
    legend=dict(bgcolor='rgba(0,0,0,0)'),
    margin=dict(l=60, r=30, t=40, b=50),
))

# ----------------------------------------------------------------------
# Search mode: free text → dropdown values
# ----------------------------------------------------------------------
BOROUGH_WORDS = {'staten island': 'STATEN ISLAND', 'staten': 'STATEN ISLAND', 'manhattan': 'MANHATTAN',
                 'bronx': 'BRONX', 'brooklyn': 'BROOKLYN', 'queens': 'QUEENS',
                 r'\bsi\b': 'STATEN ISLAND', r'\bmn\b': 'MANHATTAN', r'\bbx\b': 'BRONX',
                 r'\bbk\b': 'BROOKLYN', r'\bqn\b': 'QUEENS'}
METRIC_WORDS = {r'fatal|death|dead|kill': 'total_fatalities', r'severe|injur': 'total_severe_injuries',
                r'victim': 'total_victims', r'crash|collision|accident': 'crashes'}
CATEGORY_WORDS = {'heat': 'HEAT/HOT WATER', 'hot water': 'HEAT/HOT WATER', 'leak': 'WATER LEAK',
                  'mold': 'UNSANITARY CONDITION', 'pest': 'UNSANITARY CONDITION', 'roach': 'UNSANITARY CONDITION',
                  'mice': 'UNSANITARY CONDITION', 'unsanitary': 'UNSANITARY CONDITION', 'paint': 'PAINT/PLASTER',
                  'plaster': 'PAINT/PLASTER', 'door': 'DOOR/WINDOW', 'window': 'DOOR/WINDOW',
                  'stair': 'FLOORING/STAIRS', 'floor': 'FLOORING/STAIRS', 'electric': 'ELECTRIC',
                  'elevator': 'ELEVATOR', 'plumb': 'PLUMBING', 'appliance': 'APPLIANCE', 'safety': 'SAFETY'}


def parse_search(text):
    """Return (borough, year, category, metric, list of what was understood) from a free-text query."""
    q = (text or '').lower()
    found, borough, year, category, metric = [], None, None, None, None
    for pattern, value in BOROUGH_WORDS.items():
        if re.search(pattern, q):
            borough = value
            found.append(value.title())
            break
    m = re.search(r'\b(20\d\d)\b', q)
    if m and int(m.group(1)) in YEARS:
        year = int(m.group(1))
        found.append(str(year))
    for word, value in CATEGORY_WORDS.items():
        if word in q and value in CATEGORIES:
            category = value
            break
    if category is None:
        category = next((v for v in CATEGORIES if v.lower() in q), None)
    if category:
        found.append(category.title())
    for pattern, value in METRIC_WORDS.items():
        if re.search(pattern, q):
            metric = value
            found.append(charts.CRASH_METRICS[value])
            break
    return borough, year, category, metric, found


# ----------------------------------------------------------------------
# Building the report (cached: each filter combination is computed once)
# ----------------------------------------------------------------------
def filter_data(borough, year):
    def f(df):
        if borough != ALL:
            df = df[df['borough'] == borough]
        if year != ALL:
            df = df[df['year'] == int(year)]
        return df
    centroids = df_centroids if borough == ALL else df_centroids[df_centroids['borough'] == borough]
    return f(df_integrated), f(df_housing_monthly), f(df_housing_nta), centroids, f(df_crashes)


def styled(build, *args, keep_title=False, height=None):
    """Build a chart with the shared theme; never let one chart break the whole report."""
    try:
        fig = build(*args)
    except Exception as err:   # noqa: BLE001
        fig = charts.message_figure(f'Not enough data for this selection ({type(err).__name__}).')
    fig.update_layout(template='nyc')
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    if not keep_title:
        fig.update_layout(title_text='', margin=dict(t=30))
    if height:
        fig.update_layout(height=height)
    return fig.to_plotly_json()


@lru_cache(maxsize=64)
def build_report(borough, year, category, metric):
    integ, hm, nta, cents, crashes = filter_data(borough, year)
    hm_cat = hm if category == ALL else hm[hm['major_category'] == category]
    cat_label = 'all categories' if category == ALL else category.title()
    integ_all_boroughs = df_integrated if year == ALL else df_integrated[df_integrated['year'] == int(year)]

    n_crashes = len(crashes)
    n_deaths = int(crashes['total_fatalities'].sum())
    kpis = {
        'housing': f"{int(hm_cat['problems'].sum()):,}",
        'crashes': f'{n_crashes:,}',
        'deaths': f'{n_deaths:,}',
        'rate': f'{n_deaths / n_crashes * 100:.1f}' if n_crashes else '–',
    }
    figs = {
        'overview': styled(charts.build_overview, hm_cat, crashes, cat_label, metric, keep_title=True),
        'q1': styled(charts.build_q1, integ),
        'q2': styled(charts.build_q2, integ),
        'q3': styled(charts.build_q3, integ),
        'q4': styled(charts.build_q4, hm, crashes),
        'q5': styled(charts.build_q5, integ, cents, height=560),
        'q6': styled(charts.build_q6, integ_all_boroughs),
        'q7': styled(charts.build_q7, hm),
        'q8': styled(charts.build_q8, hm),
        'q8b': styled(charts.build_q8_borough, hm),
        'q9': styled(charts.build_q9, nta),
        'q10': styled(charts.build_q10, crashes),
        'q10b': styled(charts.build_q10_counts, crashes),
        'q11': styled(charts.build_q11, crashes),
        'q12': styled(charts.build_q12, crashes),
    }
    chips = ['All boroughs' if borough == ALL else borough.title(),
             '2022–2025' if year == ALL else str(year),
             'All housing categories' if category == ALL else category.title(),
             charts.CRASH_METRICS[metric]]
    return {'kpis': kpis, 'figs': figs, 'chips': chips}


build_report(ALL, ALL, ALL, 'crashes')   # warm the default report at start-up

# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------
QUESTIONS = {
    'q1': ('Q1', 'Which districts are "double-burdened"?',
           'Each dot is a community district. The shaded box holds districts in the top 25% for both measures.',
           'Member 1', 'wide'),
    'q2': ('Q2', 'Is the type of housing problem linked to crashes more than the volume?',
           'Correlation of each category\'s share with serious crashes. Bars inside the dotted lines are not significant.',
           'Member 2', ''),
    'q3': ('Q3', 'Did housing problems and crashes grow in the same districts?',
           'Change from the first to the last selected year. Top-right = both grew.', 'Member 3', ''),
    'q4': ('Q4', 'Do housing problems and crashes peak in the same months?',
           'Average per calendar month: housing peaks in winter, serious crashes in summer.', 'Member 4', 'wide'),
    'q6': ('Q6', 'Does each borough carry more housing or more crash burden?',
           'Left: share of NYC serious crashes minus share of housing problems. Right: both shares side by side.',
           'Member 6', 'wide'),
    'q5': ('Q5', 'Where in NYC do both burdens concentrate?',
           'Bubble size = housing problems; colour = serious crashes. Scroll to zoom, hover for details.',
           'Member 5', 'wide'),
    'q7': ('Q7', 'How does the heating season differ by borough?',
           'Each row sums to 100% of that borough\'s heat/hot-water problems.', 'Member 1', 'wide'),
    'q8': ('Q8', 'Which problem categories are most often emergencies?',
           'Share of each category by urgency type.', 'Member 4', ''),
    'q8b': ('Q8b', 'Does the emergency share differ by borough?',
            'Emergency % for each category in each borough.', 'Member 4', ''),
    'q9': ('Q9', 'Which neighbourhoods have the most multi-problem complaints?',
           'Average problems per complaint, a sign of deteriorating buildings.', 'Member 6', 'wide'),
    'q10': ('Q10', 'At what hours and days are serious crashes most deadly?',
            'Fatalities per 100 serious crashes. Blank cells have fewer than 30 crashes.', 'Member 2', ''),
    'q10b': ('Q10b', 'When do most serious crashes happen?',
             'The busiest times are not the deadliest.', 'Member 2', ''),
    'q11': ('Q11', 'Which streets are the most dangerous corridors?',
            'Top 10 streets by serious crashes, split by year.', 'Member 3', ''),
    'q12': ('Q12', 'Are night crashes more severe than day crashes?',
            'Night = 20:00–05:59. Hover shows the number of crashes behind each bar.', 'Member 5', ''),
}
TABS = {
    'overview': ('Overview', []),
    'cross': ('Housing × Crashes', ['q1', 'q2', 'q3', 'q4', 'q6', 'q5']),
    'housing': ('Housing', ['q7', 'q8', 'q8b', 'q9']),
    'streets': ('Street safety', ['q10', 'q10b', 'q11', 'q12']),
    'story': ('The story', []),
}
GRAPH_CONFIG = {'displaylogo': False, 'responsive': True,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'autoScale2d']}

FINDINGS = [
    ('r = 0.15', 'Weak link', 'Housing problems and serious crashes mostly fall on different districts.'),
    ('5 of 59', 'Double-burdened', 'Brooklyn CD 5 & 14, Bronx CD 4 & 12 and Manhattan CD 12 rank high on both.'),
    ('Dec vs Jul', 'Opposite seasons', 'Housing problems peak in winter (heat); serious crashes peak in summer.'),
    ('2×', 'Deadly nights', 'Crashes between 00:00 and 06:00 kill 14.3 per 100, vs 6.9 the rest of the day.'),
    ('6 of 10', 'Highway corridors', 'Most of the deadliest corridors are expressways and parkways.'),
    ('51 of 59', 'Housing growth', 'Housing problems grew almost everywhere, but crash growth did not follow.'),
]


def dropdown(label, dd_id, options, value):
    return html.Div(className='field', children=[
        html.Label(label, htmlFor=dd_id),
        dcc.Dropdown(id=dd_id, options=options, value=value, clearable=False, searchable=False),
    ])


def chart_card(key, figure):
    tag, question, note, owner, size = QUESTIONS[key]
    return html.Div(className=f'card {size}', children=[
        html.Div(className='card-head', children=[
            html.Span(tag, className='tag'),
            html.H3(question),
            html.Span(owner, className='owner'),
        ]),
        html.P(note, className='note'),
        dcc.Graph(figure=figure, config=GRAPH_CONFIG),
    ])


def story_section():
    return html.Div(className='story', children=[
        html.Div(className='story-block', children=[
            html.H3('The question'),
            html.P('Housing maintenance problems and serious traffic crashes are both everyday risks for New Yorkers. '
                   'We asked whether they fall on the same neighbourhoods and at the same times, so that the housing '
                   '(HPD) and transport (DOT) agencies could target them together.'),
        ]),
        html.Div(className='story-block', children=[
            html.H3('What we found'),
            html.Ul([
                html.Li('The two burdens mostly fall on different places: across district-years they are only weakly '
                        'correlated (r = 0.15).'),
                html.Li('Five districts are double-burdened, in the top quarter for both: Brooklyn CD 5, Bronx CD 12, '
                        'Brooklyn CD 14, Bronx CD 4 and Manhattan CD 12.'),
                html.Li('They peak in opposite seasons: housing problems in December (driven by heat), serious crashes '
                        'in July (monthly profiles r = −0.55).'),
                html.Li('Serious crashes between midnight and 06:00 are about twice as deadly (14.3 deaths per 100 '
                        'crashes) as the rest of the day (6.9).'),
                html.Li('Six of the ten most dangerous corridors are highways; the Long Island Expressway rose every '
                        'year from 2022 to 2025.'),
                html.Li('Housing problems grew in 51 of 59 districts (median +16.5%), but crash growth did not follow '
                        '(r = −0.09).'),
            ]),
        ]),
        html.Div(className='story-block', children=[
            html.H3('Limitations'),
            html.Ul([
                html.Li('Counts are not adjusted for population, area or road length.'),
                html.Li('Housing problems are tenant-reported complaints, which reflect reporting behaviour as well as '
                        'building conditions.'),
                html.Li('Serious crashes are rare (about 3,000 a year), so small groups give noisy rates.'),
                html.Li('9.3% of crashes could not be placed in a community district; they appear in the time and '
                        'borough charts only.'),
                html.Li('All findings are associations, not causes.'),
            ]),
        ]),
        html.Div(className='story-block', children=[
            html.H3('How NYC benefits'),
            html.Ul([
                html.Li('Joint HPD and DOT action can start in the five double-burdened districts.'),
                html.Li('Because the peaks do not overlap, winter housing inspections and summer street-safety campaigns '
                        'can be staffed without competing for the same months.'),
                html.Li('Vision Zero enforcement should include late nights and expressways, not only rush hours and '
                        'local streets.'),
            ]),
        ]),
        html.P(className='sources', children=[
            'Data: ',
            html.A('Housing Maintenance Code Complaints and Problems', href='https://data.cityofnewyork.us/d/ygpa-z7cr',
                   target='_blank'),
            ' and ', html.A('SIRTA Crash Data', href='https://data.cityofnewyork.us/d/xeqp-qz8h', target='_blank'),
            ' (NYC Open Data), 2022–2025, cleaned and integrated at community district × year.',
        ]),
    ])


app = Dash(__name__, title='NYC Homes & Streets',
           external_stylesheets=['https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'])
server = app.server   # used by gunicorn on Render

app.layout = html.Div(className='app', children=[
    dcc.Store(id='report-store'),

    html.Header(className='hero', children=[
        html.Div(className='hero-inner', children=[
            html.Div(className='brand', children=[html.Span(className='brand-dot'), 'NYC Homes & Streets']),
            html.H1(['Unsafe homes, ', html.Span('dangerous streets', className='hl'), '?']),
            html.P('2.66 million housing maintenance problems and 11,953 serious traffic crashes across NYC\'s '
                   '59 community districts, 2022–2025. Do they hit the same neighbourhoods?', className='lede'),
            html.Div(className='hero-stats', children=[
                html.Div([html.B('2.66M'), html.Span('housing problems')]),
                html.Div([html.B('11,953'), html.Span('serious crashes')]),
                html.Div([html.B('59'), html.Span('community districts')]),
                html.Div([html.B('12'), html.Span('research questions')]),
            ]),
        ]),
    ]),

    html.Div(className='layout', children=[
        # ---------------- Sidebar: filters ----------------
        html.Aside(className='sidebar', children=[
            html.Div(className='panel', children=[
                html.H2('Build your report'),
                html.Div(className='search', children=[
                    dcc.Input(id='search-box', type='text', debounce=False, n_submit=0,
                              placeholder='Try "Bronx 2024 heat deaths"'),
                    html.Button('Search', id='search-btn', n_clicks=0, className='btn-ghost'),
                ]),
                html.Div(id='search-feedback', className='feedback'),
                dropdown('Borough', 'borough-dd',
                         [{'label': 'All boroughs', 'value': ALL}] +
                         [{'label': b.title(), 'value': b} for b in BOROUGHS], ALL),
                dropdown('Year', 'year-dd',
                         [{'label': 'All years (2022–2025)', 'value': ALL}] +
                         [{'label': str(y), 'value': y} for y in YEARS], ALL),
                dropdown('Housing category', 'category-dd',
                         [{'label': 'All categories', 'value': ALL}] +
                         [{'label': c.title(), 'value': c} for c in CATEGORIES], ALL),
                dropdown('Crash metric', 'metric-dd',
                         [{'label': v, 'value': k} for k, v in charts.CRASH_METRICS.items()], 'crashes'),
                html.Button('Generate Report', id='generate-btn', n_clicks=0, className='btn-primary'),
                html.P('Charts change only when you click Generate Report. Borough and Year filter every chart; '
                       'Housing category and Crash metric drive the summary cards and the overview chart.',
                       className='hint'),
            ]),
        ]),

        # ---------------- Main: report ----------------
        html.Main(className='main', children=[
            html.Div(id='chips', className='chips'),
            html.Div(className='kpis', children=[
                html.Div(className='kpi', children=[html.Span('Housing problems', className='kpi-label'),
                                                    html.Div(id='kpi-housing', className='kpi-value')]),
                html.Div(className='kpi', children=[html.Span('Serious crashes', className='kpi-label'),
                                                    html.Div(id='kpi-crashes', className='kpi-value')]),
                html.Div(className='kpi', children=[html.Span('Fatalities', className='kpi-label'),
                                                    html.Div(id='kpi-deaths', className='kpi-value')]),
                html.Div(className='kpi accent', children=[html.Span('Deaths per 100 crashes', className='kpi-label'),
                                                           html.Div(id='kpi-rate', className='kpi-value')]),
            ]),
            dcc.Tabs(id='tabs', value='overview', className='tabs', parent_className='tabs-wrap', children=[
                dcc.Tab(label=label, value=key, className='tab', selected_className='tab--selected')
                for key, (label, _) in TABS.items()
            ]),
            dcc.Loading(type='dot', color='#2563eb', children=html.Div(id='tab-content', className='report-content')),
        ]),
    ]),

    html.Footer(className='footer', children=[
        'DEV Milestone 1 · German International University · Data: NYC Open Data (HPD, SIRTA)',
    ]),
])


# ----------------------------------------------------------------------
# Callback 1: search → dropdown values (does NOT redraw charts)
# ----------------------------------------------------------------------
@app.callback(
    Output('borough-dd', 'value'), Output('year-dd', 'value'),
    Output('category-dd', 'value'), Output('metric-dd', 'value'),
    Output('search-feedback', 'children'),
    Input('search-btn', 'n_clicks'), Input('search-box', 'n_submit'),
    State('search-box', 'value'),
    State('borough-dd', 'value'), State('year-dd', 'value'),
    State('category-dd', 'value'), State('metric-dd', 'value'),
    prevent_initial_call=True,
)
def apply_search(_clicks, _submit, text, borough, year, category, metric):
    b, y, c, m, found = parse_search(text)
    if not found:
        return (borough, year, category, metric,
                'Nothing recognised. Try a borough, a year (2022–2025), a problem (heat, leak, mold) or "deaths".')
    return (b or ALL, y or ALL, c or ALL, m or metric,
            ['Set to ', html.B(' · '.join(found)), '. Now click Generate Report.'])


# ----------------------------------------------------------------------
# Callback 2: Generate Report → compute everything (the ONLY trigger for new charts)
# ----------------------------------------------------------------------
@app.callback(
    Output('report-store', 'data'),
    Output('kpi-housing', 'children'), Output('kpi-crashes', 'children'),
    Output('kpi-deaths', 'children'), Output('kpi-rate', 'children'),
    Output('chips', 'children'),
    Input('generate-btn', 'n_clicks'),
    State('borough-dd', 'value'), State('year-dd', 'value'),
    State('category-dd', 'value'), State('metric-dd', 'value'),
)
def generate_report(_clicks, borough, year, category, metric):
    report = build_report(borough, year, category, metric)
    k = report['kpis']
    chips = [html.Span('Showing', className='chips-label')] + [html.Span(c, className='chip') for c in report['chips']]
    return report['figs'], k['housing'], k['crashes'], k['deaths'], k['rate'], chips


# ----------------------------------------------------------------------
# Callback 3: show the selected tab (uses the stored report; nothing is recomputed)
# ----------------------------------------------------------------------
@app.callback(Output('tab-content', 'children'), Input('tabs', 'value'), Input('report-store', 'data'))
def render_tab(tab, figs):
    if not figs:
        return html.Div('Click Generate Report to build the report.', className='empty')
    if tab == 'story':
        return story_section()
    if tab == 'overview':
        return html.Div(children=[
            html.Div(className='card wide', children=[dcc.Graph(figure=figs['overview'], config=GRAPH_CONFIG)]),
            html.H2('Key findings (2022–2025, all of NYC)', className='section-title'),
            html.Div(className='findings', children=[
                html.Div(className='finding', children=[html.B(big), html.H4(title), html.P(text)])
                for big, title, text in FINDINGS
            ]),
            html.Div(className='grid', children=[chart_card('q1', figs['q1']), chart_card('q5', figs['q5'])]),
        ])
    return html.Div(className='grid', children=[chart_card(k, figs[k]) for k in TABS[tab][1]])


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8050)))
