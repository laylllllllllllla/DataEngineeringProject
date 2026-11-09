"""
NYC Homes & Streets: housing maintenance problems × serious traffic crashes, 2022–2025.
Dash website for DEV Milestone 1.

Run locally:   python website/app.py        → http://127.0.0.1:8050
Deployed with: gunicorn --chdir website app:server
"""
import os
import re

import pandas as pd
from dash import Dash, html, dcc, Input, Output, State, ctx

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

BOROUGHS = sorted(df_integrated['borough'].unique())
YEARS = sorted(int(y) for y in df_integrated['year'].unique())
CATEGORIES = sorted(df_housing_monthly['major_category'].unique())
ALL = 'ALL'

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
            found.append(f'borough = {value.title()}')
            break
    m = re.search(r'\b(20\d\d)\b', q)
    if m and int(m.group(1)) in YEARS:
        year = int(m.group(1))
        found.append(f'year = {year}')
    for word, value in CATEGORY_WORDS.items():
        if word in q and value in CATEGORIES:
            category = value
            found.append(f'housing category = {value.title()}')
            break
    if category is None:   # also accept an exact category name typed by the user
        for value in CATEGORIES:
            if value.lower() in q:
                category = value
                found.append(f'housing category = {value.title()}')
                break
    for pattern, value in METRIC_WORDS.items():
        if re.search(pattern, q):
            metric = value
            found.append(f'crash metric = {charts.CRASH_METRICS[value]}')
            break
    return borough, year, category, metric, found


# ----------------------------------------------------------------------
# Filtering
# ----------------------------------------------------------------------
def filter_data(borough, year):
    """Apply the Borough and Year filters to every table."""
    def f(df):
        if borough != ALL:
            df = df[df['borough'] == borough]
        if year != ALL:
            df = df[df['year'] == int(year)]
        return df
    centroids = df_centroids if borough == ALL else df_centroids[df_centroids['borough'] == borough]
    return f(df_integrated), f(df_housing_monthly), f(df_housing_nta), centroids, f(df_crashes)


def safe(build, *args):
    """Never let one chart break the whole report."""
    try:
        fig = build(*args)
        if build is not charts.build_overview:
            # The card heading already shows the question, so drop the chart title (kept in the notebook)
            fig.update_layout(title_text='', margin=dict(t=30))
        return fig
    except Exception as err:   # noqa: BLE001
        return charts.message_figure(f'Not enough data for this selection ({type(err).__name__}).')


# ----------------------------------------------------------------------
# Layout helpers
# ----------------------------------------------------------------------
def chart_card(chart_id, question, note, owner):
    return html.Div(className='card', children=[
        html.Div(className='card-head', children=[
            html.H3(question),
            html.Span(owner, className='owner'),
        ]),
        html.P(note, className='note'),
        dcc.Graph(id=chart_id, config={'displaylogo': False}),
    ])


def kpi(label, value_id):
    return html.Div(className='kpi', children=[html.Div(id=value_id, className='kpi-value'),
                                               html.Div(label, className='kpi-label')])


def dropdown(label, dd_id, options, value):
    return html.Div(className='control', children=[
        html.Label(label, htmlFor=dd_id),
        dcc.Dropdown(id=dd_id, options=options, value=value, clearable=False),
    ])


app = Dash(__name__, title='NYC Homes & Streets')
server = app.server   # used by gunicorn on Render

app.layout = html.Div(className='page', children=[
    html.Header(className='hero', children=[
        html.P('DEV Milestone 1 · NYC Open Data · 2022–2025', className='eyebrow'),
        html.H1('NYC Homes & Streets'),
        html.P('Do the neighbourhoods with the worst housing conditions also have the most dangerous streets? '
               'We combine 2.66 million housing maintenance problems (HPD) with 11,953 serious traffic crashes '
               '(SIRTA) across NYC\'s 59 community districts.', className='lede'),
    ]),

    # ---------------- Controls ----------------
    html.Section(className='panel', children=[
        html.Div(className='search-row', children=[
            dcc.Input(id='search-box', type='text', debounce=False, className='search',
                      placeholder='Search, e.g. "Bronx 2024 heat fatalities" or "Brooklyn crashes"'),
            html.Button('Apply search', id='search-btn', className='btn-secondary'),
        ]),
        html.Div(id='search-feedback', className='feedback'),
        html.Div(className='controls', children=[
            dropdown('Borough', 'borough-dd',
                     [{'label': 'All boroughs', 'value': ALL}] + [{'label': b.title(), 'value': b} for b in BOROUGHS],
                     ALL),
            dropdown('Year', 'year-dd',
                     [{'label': 'All years (2022–2025)', 'value': ALL}] + [{'label': str(y), 'value': y} for y in YEARS],
                     ALL),
            dropdown('Housing category', 'category-dd',
                     [{'label': 'All categories', 'value': ALL}] + [{'label': c.title(), 'value': c} for c in CATEGORIES],
                     ALL),
            dropdown('Crash metric', 'metric-dd',
                     [{'label': v, 'value': k} for k, v in charts.CRASH_METRICS.items()], 'crashes'),
        ]),
        html.Div(className='generate-row', children=[
            html.Button('Generate Report', id='generate-btn', className='btn-primary'),
            html.Span('Charts update only when you click Generate Report. Borough and Year filter every chart; '
                      'Housing category and Crash metric control the summary cards and the overview chart.',
                      className='hint'),
        ]),
    ]),

    # ---------------- Report ----------------
    dcc.Loading(type='circle', color='#2557a7', children=html.Div(children=[
        html.Div(id='report-caption', className='caption'),
        html.Div(className='kpis', children=[
            kpi('Housing problems', 'kpi-housing'),
            kpi('Serious crashes', 'kpi-crashes'),
            kpi('Fatalities', 'kpi-fatalities'),
            kpi('Deaths per 100 serious crashes', 'kpi-rate'),
        ]),
        html.Div(className='card wide', children=[dcc.Graph(id='overview', config={'displaylogo': False})]),

        html.H2('Housing × crashes together', className='section-title'),
        html.Div(className='grid', children=[
            chart_card('q1', 'Q1. Which districts are "double-burdened"?',
                       'Each dot is a community district. The shaded box holds districts in the top 25% for both '
                       'housing problems and serious crashes.', 'Member 1'),
            chart_card('q2', 'Q2. Is the type of housing problem linked to crashes more than the volume?',
                       'Correlation of each category\'s share of housing problems with serious crashes. Bars '
                       'inside the dotted lines are not statistically significant.', 'Member 2'),
            chart_card('q3', 'Q3. Did housing problems and crashes grow in the same districts?',
                       'Percentage change from the first to the last selected year. Top-right = both grew.',
                       'Member 3'),
            chart_card('q4', 'Q4. Do housing problems and crashes peak in the same months?',
                       'Average per calendar month. Housing peaks in winter, crashes in summer.', 'Member 4'),
            chart_card('q5', 'Q5. Where in NYC do both burdens concentrate?',
                       'Bubble size = housing problems; colour = serious crashes. Hover for details.', 'Member 5'),
            chart_card('q6', 'Q6. Does each borough carry more housing or more crash burden?',
                       'Share of NYC serious crashes minus share of NYC housing problems.', 'Member 6'),
        ]),

        html.H2('Housing conditions', className='section-title'),
        html.Div(className='grid', children=[
            chart_card('q7', 'Q7. How does the heating season differ by borough?',
                       'Each row sums to 100% of that borough\'s heat/hot-water problems.', 'Member 1'),
            chart_card('q8', 'Q8. Which problem categories are most often emergencies?',
                       'Share of each category by urgency type.', 'Member 4'),
            chart_card('q8b', 'Q8b. Does the emergency share differ by borough?',
                       'Emergency % for each category in each borough.', 'Member 4'),
            chart_card('q9', 'Q9. Which neighbourhoods have the most multi-problem complaints?',
                       'Average number of problems per complaint, a sign of deteriorating buildings.', 'Member 6'),
        ]),

        html.H2('Street safety', className='section-title'),
        html.Div(className='grid', children=[
            chart_card('q10', 'Q10. At what hours and days are serious crashes most deadly?',
                       'Fatalities per 100 serious crashes. Cells with fewer than 30 crashes are left blank.',
                       'Member 2'),
            chart_card('q10b', 'Q10b. When do most serious crashes happen?',
                       'Number of serious crashes: the busiest times are not the deadliest.', 'Member 2'),
            chart_card('q11', 'Q11. Which streets are the most dangerous corridors?',
                       'Top 10 streets by serious crashes, split by year.', 'Member 3'),
            chart_card('q12', 'Q12. Are night crashes more severe than day crashes?',
                       'Night = 20:00–05:59. Hover shows the number of crashes.', 'Member 5'),
        ]),
    ])),

    # ---------------- Story ----------------
    html.Section(className='story', children=[
        html.H2('The story'),
        html.Div(className='story-grid', children=[
            html.Div(className='story-block', children=[
                html.H3('The question'),
                html.P('Housing maintenance problems and serious traffic crashes are both everyday risks for New '
                       'Yorkers. We asked whether they fall on the same neighbourhoods, at the same times, so that '
                       'the city\'s housing (HPD) and transport (DOT) agencies could target them together.'),
            ]),
            html.Div(className='story-block', children=[
                html.H3('What we found'),
                html.Ul([
                    html.Li('The two burdens mostly fall on different places: across district-years they are '
                            'only weakly correlated (r = 0.15).'),
                    html.Li('Five districts are double-burdened, in the top quarter for both: Brooklyn CD 5, '
                            'Bronx CD 12, Brooklyn CD 14, Bronx CD 4 and Manhattan CD 12.'),
                    html.Li('They peak in opposite seasons: housing problems in December (driven by heat), serious '
                            'crashes in July (monthly profiles r = −0.55).'),
                    html.Li('Serious crashes between midnight and 06:00 are about twice as deadly (14.3 deaths per '
                            '100 crashes) as the rest of the day (6.9).'),
                    html.Li('Six of the ten most dangerous corridors are highways; the Long Island Expressway rose '
                            'every year from 2022 to 2025.'),
                    html.Li('Housing problems grew in 51 of 59 districts (median +16.5%), but growth in crashes '
                            'did not follow (r = −0.09).'),
                ]),
            ]),
            html.Div(className='story-block', children=[
                html.H3('Limitations'),
                html.Ul([
                    html.Li('Counts are not adjusted for population, area or road length.'),
                    html.Li('Housing problems are tenant-reported complaints, which reflect reporting behaviour '
                            'as well as building conditions.'),
                    html.Li('Serious crashes are rare (about 3,000 a year), so small groups give noisy rates.'),
                    html.Li('9.3% of crashes could not be placed in a community district; they appear in the '
                            'time and borough charts only.'),
                    html.Li('All findings are associations, not causes.'),
                ]),
            ]),
            html.Div(className='story-block', children=[
                html.H3('How NYC benefits'),
                html.Ul([
                    html.Li('Joint HPD and DOT action can start in the five double-burdened districts.'),
                    html.Li('Because peaks do not overlap, winter housing inspections and summer street-safety '
                            'campaigns can be staffed without competing for the same months.'),
                    html.Li('Vision Zero enforcement should include late nights and expressways, not only rush '
                            'hours and local streets.'),
                ]),
            ]),
        ]),
        html.P(['Data: ',
                html.A('Housing Maintenance Code Complaints and Problems',
                       href='https://data.cityofnewyork.us/d/ygpa-z7cr', target='_blank'),
                ' and ',
                html.A('SIRTA Crash Data', href='https://data.cityofnewyork.us/d/xeqp-qz8h', target='_blank'),
                ' (NYC Open Data), 2022–2025. Cleaned and integrated at community district × year; '
                'see the project notebook for every step.'], className='sources'),
    ]),
])


# ----------------------------------------------------------------------
# Callback 1: search box → dropdown values (does NOT redraw charts)
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
        return borough, year, category, metric, 'No filters recognised. Try a borough, a year (2022–2025), ' \
                                                'a housing problem (heat, leak, mold…) or deaths / injuries.'
    msg = 'Understood: ' + ', '.join(found) + '. Click Generate Report to update the charts.'
    return (b or ALL, y or ALL, c or ALL, m or metric, msg)


# ----------------------------------------------------------------------
# Callback 2: Generate Report → every chart (the ONLY trigger for the charts)
# ----------------------------------------------------------------------
CHART_IDS = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q8b', 'q9', 'q10', 'q10b', 'q11', 'q12']


@app.callback(
    Output('report-caption', 'children'),
    Output('kpi-housing', 'children'), Output('kpi-crashes', 'children'),
    Output('kpi-fatalities', 'children'), Output('kpi-rate', 'children'),
    Output('overview', 'figure'),
    *[Output(c, 'figure') for c in CHART_IDS],
    Input('generate-btn', 'n_clicks'),
    State('borough-dd', 'value'), State('year-dd', 'value'),
    State('category-dd', 'value'), State('metric-dd', 'value'),
)
def generate_report(_clicks, borough, year, category, metric):
    integ, hm, nta, cents, crashes = filter_data(borough, year)
    hm_cat = hm if category == ALL else hm[hm['major_category'] == category]
    cat_label = 'all categories' if category == ALL else category.title()

    n_crashes = len(crashes)
    n_deaths = int(crashes['total_fatalities'].sum())
    caption = (f"Showing: {'All boroughs' if borough == ALL else borough.title()} · "
               f"{'2022–2025' if year == ALL else year} · housing: {cat_label} · "
               f"metric: {charts.CRASH_METRICS[metric].lower()}")
    kpis = (f"{int(hm_cat['problems'].sum()):,}", f'{n_crashes:,}', f'{n_deaths:,}',
            f'{n_deaths / n_crashes * 100:.1f}' if n_crashes else '–')

    # Q6 compares boroughs, so it always uses every borough (Year filter only)
    integ_all_boroughs = df_integrated if year == ALL else df_integrated[df_integrated['year'] == int(year)]

    figures = [
        safe(charts.build_q1, integ),
        safe(charts.build_q2, integ),
        safe(charts.build_q3, integ),
        safe(charts.build_q4, hm, crashes),
        safe(charts.build_q5, integ, cents),
        safe(charts.build_q6, integ_all_boroughs),
        safe(charts.build_q7, hm),
        safe(charts.build_q8, hm),
        safe(charts.build_q8_borough, hm),
        safe(charts.build_q9, nta),
        safe(charts.build_q10, crashes),
        safe(charts.build_q10_counts, crashes),
        safe(charts.build_q11, crashes),
        safe(charts.build_q12, crashes),
    ]
    overview = safe(charts.build_overview, hm_cat, crashes, cat_label, metric)
    return (caption, *kpis, overview, *figures)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8050)))
