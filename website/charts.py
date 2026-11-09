"""
Chart functions for the 12 research questions.

Every function takes already-filtered data frames (the website applies the Borough / Year filters
before calling it) and RETURNS a Plotly figure. The same functions are used in the notebook.

Q1, Q7        Member 1        Q4, Q8   Member 4
Q2, Q10       Member 2        Q5, Q12  Member 5
Q3, Q11       Member 3        Q6, Q9   Member 6
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BOROUGH_COLORS = {'Manhattan': '#2557a7', 'Bronx': '#b5562f', 'Brooklyn': '#2f7d5b',
                  'Queens': '#8a4fa8', 'Staten Island': '#c7951c'}
BOROUGH_COLORS_UPPER = {k.upper(): v for k, v in BOROUGH_COLORS.items()}
BOROUGH_SHORT = {'MANHATTAN': 'MN', 'BRONX': 'BX', 'BROOKLYN': 'BK', 'QUEENS': 'QN', 'STATEN ISLAND': 'SI'}
MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def message_figure(text):
    """Empty chart with a message (used when a filter leaves too little data)."""
    fig = go.Figure()
    fig.add_annotation(text=text, showarrow=False, x=0.5, y=0.5, xref='paper', yref='paper',
                       font=dict(size=14, color='#5f6b7a'))
    fig.update_layout(template='plotly_white', height=380,
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


# ======================================================================
# Overview (website only): selected housing category vs selected crash metric over time
# ======================================================================
CRASH_METRICS = {'crashes': 'Serious crashes', 'total_fatalities': 'Fatalities',
                 'total_severe_injuries': 'Severe injuries', 'total_victims': 'Total victims'}


def build_overview(df_housing_monthly, df_crashes, category_label, metric):
    """Dual-axis trend: by year, or by month when only one year is selected."""
    by = 'year' if df_housing_monthly['year'].nunique() > 1 else 'month'
    housing = df_housing_monthly.groupby(by)['problems'].sum()
    if metric == 'crashes':
        crashes = df_crashes.groupby(by).size()
    else:
        crashes = df_crashes.groupby(by)[metric].sum()
    idx = sorted(set(housing.index) | set(crashes.index))
    x = [MONTH_NAMES[int(i) - 1] for i in idx] if by == 'month' else [str(int(i)) for i in idx]

    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(go.Bar(x=x, y=housing.reindex(idx, fill_value=0).values, name=f'Housing problems: {category_label}',
                         marker_color='#e8b9a3', hovertemplate='%{x}: %{y:,.0f} problems<extra></extra>'),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=crashes.reindex(idx, fill_value=0).values, name=CRASH_METRICS[metric],
                             mode='lines+markers', line=dict(color='#2557a7', width=3),
                             hovertemplate='%{x}: %{y:,.0f}<extra>' + CRASH_METRICS[metric] + '</extra>'),
                  secondary_y=True)
    fig.update_layout(template='plotly_white', height=380, hovermode='x unified',
                      title=f'Housing problems ({category_label}) vs {CRASH_METRICS[metric].lower()}, '
                            f'by {"year" if by == "year" else "month"}',
                      legend=dict(orientation='h', y=-0.2))
    fig.update_yaxes(title_text='Housing problems', secondary_y=False, rangemode='tozero')
    fig.update_yaxes(title_text=CRASH_METRICS[metric], secondary_y=True, rangemode='tozero')
    fig.update_xaxes(type='category')   # years as labels, not 2022.5
    fig.update_layout(margin=dict(l=70, r=70, t=50, b=40))
    return fig


# ======================================================================
# Q1 (Member 1): Which community districts are "double-burdened"?
# ======================================================================
def double_burdened_table(df_integrated):
    """District totals over the selected years, with a flag for the top quarter on both measures."""
    d = (df_integrated.groupby(['cd_id', 'cd_label', 'borough'], as_index=False)
                      [['housing_problems', 'crashes']].sum())
    x_cut = d['housing_problems'].quantile(0.75)
    y_cut = d['crashes'].quantile(0.75)
    d['double_burdened'] = (d['housing_problems'] >= x_cut) & (d['crashes'] >= y_cut)
    d['crashes_per_10k_problems'] = (d['crashes'] / d['housing_problems'].replace(0, np.nan) * 10_000).round(1)
    return d, x_cut, y_cut


def build_q1(df_integrated):
    """Quadrant scatter: housing problems vs serious crashes per community district."""
    d, x_cut, y_cut = double_burdened_table(df_integrated)
    short = d['borough'].map(BOROUGH_SHORT) + ' ' + (d['cd_id'] % 100).astype(str)
    d['label'] = np.where(d['double_burdened'], short, '')
    d['Borough'] = d['borough'].str.title()

    fig = px.scatter(
        d, x='housing_problems', y='crashes', color='Borough', text='label',
        color_discrete_map=BOROUGH_COLORS, hover_name='cd_label',
        hover_data={'label': False, 'Borough': False, 'housing_problems': ':,',
                    'crashes': ':,', 'crashes_per_10k_problems': True},
        labels={'housing_problems': 'Housing problems (count)', 'crashes': 'Serious crashes (count)',
                'crashes_per_10k_problems': 'Crashes per 10k problems'},
        title='Q1. Double-burdened districts: housing problems vs serious crashes')
    fig.add_shape(type='rect', x0=x_cut, x1=d['housing_problems'].max() * 1.05,
                  y0=y_cut, y1=d['crashes'].max() * 1.08,
                  fillcolor='rgba(200, 40, 40, 0.07)', line_width=0, layer='below')
    fig.add_vline(x=x_cut, line_dash='dash', line_color='gray',
                  annotation_text='Top 25% housing problems', annotation_position='bottom right')
    fig.add_hline(y=y_cut, line_dash='dash', line_color='gray',
                  annotation_text='Top 25% serious crashes', annotation_position='top left')
    fig.update_traces(marker=dict(size=10, line=dict(width=0.5, color='white')),
                      textposition='top center', textfont_size=10)
    fig.update_layout(template='plotly_white', height=520, legend_title_text='Borough')
    return fig


# ======================================================================
# Q2 (Member 2): Is the type of housing problem more closely linked to serious crashes than the volume?
# ======================================================================
def category_label(col):
    """'cat_unsanitary_condition' → 'Unsanitary condition'; 'heat_problems' → 'Heat/hot water'."""
    if col == 'heat_problems':
        return 'Heat/hot water'
    return (col.replace('cat_', '').replace('_', ' ').capitalize().replace('Door window', 'Door/window')
            .replace('Paint plaster', 'Paint/plaster').replace('Flooring stairs', 'Flooring/stairs'))


def problem_type_corr_table(df_integrated):
    """Correlation of each category's share of housing problems with serious crashes, across district-years."""
    d = df_integrated
    cols = [c for c in d.columns if c.startswith('cat_')] + ['heat_problems']
    total = d['housing_problems'].replace(0, np.nan)
    rows = []
    for c in cols:
        share = d[c] / total * 100
        if not share.std() > 0:
            continue
        r = share.corr(d['crashes'])
        if pd.notna(r):
            rows.append({'measure': category_label(c), 'r': r, 'problems': int(d[c].sum()),
                         'avg_share_%': share.mean(), 'kind': 'Positive' if r >= 0 else 'Negative'})
    rows.append({'measure': 'TOTAL VOLUME (reference)', 'r': d['housing_problems'].corr(d['crashes']),
                 'problems': int(d['housing_problems'].sum()), 'avg_share_%': 100.0,
                 'kind': 'Total volume (reference)'})
    table = pd.DataFrame(rows).sort_values('r').reset_index(drop=True)
    n = len(d)
    r_crit = 1.97 / np.sqrt(n - 2 + 1.97 ** 2) if n > 3 else np.nan
    return table, n, r_crit


def build_q2(df_integrated):
    """Sorted horizontal bar: correlation of each problem-type share with serious crashes."""
    if len(df_integrated) < 5:
        return message_figure('Q2 needs more district-years: select "All years" or a whole borough.')
    table, n, r_crit = problem_type_corr_table(df_integrated)
    fig = px.bar(
        table, x='r', y='measure', orientation='h', color='kind',
        color_discrete_map={'Positive': '#b5562f', 'Negative': '#2557a7',
                            'Total volume (reference)': '#8c8c8c'},
        custom_data=['problems', 'avg_share_%'],
        labels={'r': 'Correlation with serious crashes (Pearson r)',
                'measure': 'Housing problem type (share of all problems)', 'kind': ''},
        title=f'Q2. Problem type vs serious crashes ({n} district-years)')
    fig.update_traces(hovertemplate='<b>%{y}</b><br>r = %{x:.3f}<br>Problems: %{customdata[0]:,}'
                                    '<br>Average share: %{customdata[1]:.1f}%<extra></extra>')
    fig.add_vline(x=0, line_color='black', line_width=1)
    if pd.notna(r_crit):
        for x in (-r_crit, r_crit):
            fig.add_vline(x=x, line_dash='dot', line_color='gray')
        fig.add_annotation(x=r_crit, y=1.07, yref='paper', showarrow=False, xanchor='left',
                           text=f'|r| < {r_crit:.2f}: not significant', font=dict(size=10, color='gray'))
    lim = max(0.3, table['r'].abs().max() * 1.2)
    fig.update_layout(template='plotly_white', height=max(480, 30 * len(table)), xaxis_range=[-lim, lim],
                      yaxis=dict(categoryorder='array', categoryarray=table['measure'].tolist()),
                      legend=dict(orientation='h', y=-0.15))
    return fig


# ======================================================================
# Q3 (Member 3): Did districts where housing problems grew fastest also see serious crashes grow?
# ======================================================================
def build_q3(df_integrated):
    """Q3: Did districts where housing problems grew fastest also see serious crashes grow?"""
    first_year, last_year = df_integrated['year'].min(), df_integrated['year'].max()
    if first_year == last_year:
        return message_figure('Q3 compares two years: select "All years" to see growth.')
    d = df_integrated[df_integrated['year'].isin([first_year, last_year])]
    p = d.pivot_table(index=['cd_label', 'borough'], columns='year',
                      values=['housing_problems', 'crashes'], aggfunc='sum').reset_index()
    p.columns = [f'{a}_{b}' if b != '' else a for a, b in p.columns]
    p = p[(p[f'housing_problems_{first_year}'] > 0) & (p[f'crashes_{first_year}'] > 0)]
    p['housing_change_%'] = ((p[f'housing_problems_{last_year}'] - p[f'housing_problems_{first_year}'])
                             / p[f'housing_problems_{first_year}'] * 100)
    p['crash_change_%'] = (p[f'crashes_{last_year}'] - p[f'crashes_{first_year}']) / p[f'crashes_{first_year}'] * 100
    fig = px.scatter(p, x='housing_change_%', y='crash_change_%', color='borough',
                     color_discrete_map=BOROUGH_COLORS_UPPER, hover_name='cd_label',
                     hover_data={f'housing_problems_{first_year}': True, f'housing_problems_{last_year}': True,
                                 f'crashes_{first_year}': True, f'crashes_{last_year}': True},
                     title=f'Q3. Growth vs growth: housing problems vs serious crashes, {first_year}–{last_year}',
                     labels={'housing_change_%': 'Change in housing problems (%)',
                             'crash_change_%': 'Change in serious crashes (%)', 'borough': 'Borough'})
    fig.add_vline(x=0, line_dash='dash', line_color='grey')
    fig.add_hline(y=0, line_dash='dash', line_color='grey')
    fig.update_layout(template='plotly_white', height=480)
    return fig


# ======================================================================
# Q4 (Member 4): Do housing problems and serious crashes peak in the same months?
# ======================================================================
def build_q4(df_housing_monthly, df_crashes):
    """Q4: Do housing problems and serious crashes peak in the same months?"""
    months = range(1, 13)
    h_years = max(df_housing_monthly['year'].nunique(), 1)
    c_years = max(df_crashes['year'].nunique(), 1)
    housing = (df_housing_monthly.groupby('month')['problems'].sum()
               .reindex(months, fill_value=0) / h_years)
    heat = (df_housing_monthly[df_housing_monthly['major_category'] == 'HEAT/HOT WATER']
            .groupby('month')['problems'].sum()
            .reindex(months, fill_value=0) / h_years)
    crashes = (df_crashes.groupby('month').size()
               .reindex(months, fill_value=0) / c_years)

    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(go.Scatter(x=MONTH_NAMES, y=housing.values, name='All housing problems (left axis)',
                             mode='lines+markers', line=dict(width=3),
                             hovertemplate='%{x}: %{y:,.0f} problems/yr<extra>All housing</extra>'),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=MONTH_NAMES, y=heat.values, name='Heat/hot-water problems (left axis)',
                             mode='lines', line=dict(dash='dot'),
                             hovertemplate='%{x}: %{y:,.0f} problems/yr<extra>Heat/hot water</extra>'),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=MONTH_NAMES, y=crashes.values, name='Serious crashes (right axis)',
                             mode='lines+markers', line=dict(width=3),
                             hovertemplate='%{x}: %{y:,.0f} crashes/yr<extra>Serious crashes</extra>'),
                  secondary_y=True)
    fig.update_layout(title='Q4. Do housing problems and serious crashes peak in the same months?',
                      xaxis_title='Month', hovermode='x unified', template='plotly_white', height=460,
                      legend=dict(orientation='h', y=-0.25))
    fig.update_yaxes(title_text='Housing problems (avg per year)', secondary_y=False, rangemode='tozero')
    fig.update_yaxes(title_text='Serious crashes (avg per year)', secondary_y=True, rangemode='tozero')
    return fig


# ======================================================================
# Q5 (Member 5): Where in NYC do housing problems and serious crashes concentrate?
# ======================================================================
MAP_LAYERS = [{   # free Esri light-grey street map (OpenStreetMap and Carto tiles are blocked / need a key)
    'below': 'traces', 'sourcetype': 'raster',
    'sourceattribution': 'Esri, HERE, Garmin, © OpenStreetMap contributors',
    'source': ['https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/'
               'World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}'],
}]


def build_q5(df_integrated, df_centroids):
    """Q5: Map of housing problems (bubble size) and serious crashes (colour) per community district."""
    d = (df_integrated.groupby('cd_id', as_index=False)
         [['housing_problems', 'crashes']].sum())
    d = d.merge(df_centroids[['cd_id', 'cd_label', 'borough', 'latitude', 'longitude']],
                on='cd_id', how='inner')
    if d.empty:
        return message_figure('No districts for this selection.')
    d['crashes_per_10k_problems'] = (
        d['crashes'] / d['housing_problems'].replace(0, np.nan) * 10_000
    ).round(1)
    fig = px.scatter_mapbox(
        d, lat='latitude', lon='longitude',
        size='housing_problems', color='crashes',
        hover_name='cd_label',
        hover_data={'borough': True, 'housing_problems': ':,', 'crashes': ':,',
                    'crashes_per_10k_problems': True, 'latitude': False, 'longitude': False},
        color_continuous_scale='YlOrRd', size_max=35,
        zoom=9.8, center={'lat': 40.71, 'lon': -73.94},
        title='Where housing problems and serious crashes concentrate (by community district)',
        labels={'housing_problems': 'Housing problems', 'crashes': 'Serious crashes',
                'crashes_per_10k_problems': 'Crashes per 10k problems', 'borough': 'Borough'},
    )
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=50, b=0),
                      mapbox_style='white-bg', mapbox_layers=MAP_LAYERS)
    return fig


# ============================================================
# Q6 (Member 6): How does each borough's share of housing problems compare with its share of serious crashes?
# ============================================================
def borough_share_table(df_integrated):
    """Borough totals, each borough's share of the NYC total, and the gap in percentage points."""
    d = df_integrated.groupby('borough', as_index=False)[['housing_problems', 'crashes']].sum()
    d['housing_share'] = d['housing_problems'] / d['housing_problems'].sum() * 100
    d['crash_share']   = d['crashes'] / d['crashes'].sum() * 100
    d['gap_pp']        = d['crash_share'] - d['housing_share']
    d['Borough']       = d['borough'].str.title()
    return d.sort_values('gap_pp').reset_index(drop=True)


def build_q6(df_integrated):
    """Diverging bar of the share gap (left) and the two shares side by side (right)."""
    d = borough_share_table(df_integrated)
    if d.empty:
        return px.bar(title='Q6. No data for the selected filters')

    fig = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5], horizontal_spacing=0.18,
                        subplot_titles=('Gap: crash share minus housing share',
                                        'Share of NYC total, side by side'))
    custom = d[['housing_problems', 'crashes', 'housing_share', 'crash_share']].to_numpy()

    # Left: diverging gap (red = more crash burden than housing burden, blue = the opposite)
    fig.add_trace(go.Bar(
        y=d['Borough'], x=d['gap_pp'], orientation='h', showlegend=False,
        marker_color=np.where(d['gap_pp'] >= 0, '#c0392b', '#2557a7'),
        text=[f'{v:+.1f} pp' for v in d['gap_pp']], textposition='outside', cliponaxis=False,
        customdata=custom,
        hovertemplate=('<b>%{y}</b><br>Gap: %{x:+.1f} pp<br>'
                       'Housing share: %{customdata[2]:.1f}% (%{customdata[0]:,.0f} problems)<br>'
                       'Crash share: %{customdata[3]:.1f}% (%{customdata[1]:,.0f} crashes)<extra></extra>')),
        row=1, col=1)
    fig.add_vline(x=0, line_color='black', line_width=1, row=1, col=1)

    # Right: the two shares as grouped bars
    for col, name, colour in [('housing_share', 'Share of housing problems', '#7f8c8d'),
                              ('crash_share', 'Share of serious crashes', '#c0392b')]:
        fig.add_trace(go.Bar(
            y=d['Borough'], x=d[col], orientation='h', name=name, marker_color=colour,
            text=[f'{v:.1f}%' for v in d[col]], textposition='outside', cliponaxis=False,
            customdata=custom,
            hovertemplate=('<b>%{y}</b><br>' + name + ': %{x:.1f}%<br>'
                           'Housing problems: %{customdata[0]:,.0f}<br>'
                           'Serious crashes: %{customdata[1]:,.0f}<extra></extra>')),
            row=1, col=2)

    gmax = max(abs(d['gap_pp']).max(), 1) * 1.5
    fig.update_xaxes(title_text='Gap (percentage points)', range=[-gmax, gmax], row=1, col=1)
    fig.update_xaxes(title_text='% of NYC total', range=[0, d[['housing_share', 'crash_share']].max().max() * 1.2],
                     row=1, col=2)
    fig.update_yaxes(title_text='Borough', row=1, col=1)
    fig.update_layout(barmode='group', height=450,
                      title='Q6. Borough share of housing problems vs share of serious crashes',
                      legend=dict(orientation='h', y=-0.2, x=0.55))
    return fig



# ======================================================================
# Q7 (Member 1): How does the heating season differ across boroughs?
# ======================================================================
BOROUGH_ORDER = ['BRONX', 'BROOKLYN', 'MANHATTAN', 'QUEENS', 'STATEN ISLAND']


def heat_season_table(df_housing_monthly):
    """% of each borough's heat/hot-water problems that fall in each calendar month."""
    heat = df_housing_monthly[df_housing_monthly['major_category'] == 'HEAT/HOT WATER']
    counts = (heat.groupby(['borough', 'month'])['problems'].sum()
                  .unstack(fill_value=0)
                  .reindex(columns=range(1, 13), fill_value=0))
    counts = counts.reindex([b for b in BOROUGH_ORDER if b in counts.index])
    share = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) * 100
    return share, counts


def build_q7(df_housing_monthly):
    """Heatmap: share of each borough's heat problems per month (each row sums to 100%)."""
    share, counts = heat_season_table(df_housing_monthly)
    if share.empty:
        return message_figure('No heat/hot-water problems for this selection.')
    fig = px.imshow(
        share.round(1), x=MONTH_NAMES, y=[b.title() for b in share.index],
        color_continuous_scale='OrRd', aspect='auto', text_auto='.0f',
        labels={'x': 'Month', 'y': 'Borough', 'color': "% of borough's heat problems"},
        title='Q7. Heating season by borough: share of heat/hot-water problems per month')
    fig.update_traces(
        customdata=counts.values,
        hovertemplate='%{y}, %{x}<br>Share: %{z:.1f}%<br>Problems: %{customdata:,}<extra></extra>')
    fig.update_layout(template='plotly_white', height=400,
                      coloraxis_colorbar=dict(title='% of year', ticksuffix='%'))
    return fig


# ======================================================================
# Q8 (Member 4): Which housing problem categories are most often emergencies?
# ======================================================================
def build_q8(df_housing_monthly):
    """Q8: Which housing problem categories are most often emergencies?"""
    d = df_housing_monthly.groupby(['major_category', 'type'], as_index=False)['problems'].sum()
    d['total'] = d.groupby('major_category')['problems'].transform('sum')
    d['pct'] = d['problems'] / d['total'] * 100
    is_emerg = d['type'].str.upper().str.contains('EMERGENCY') & ~d['type'].str.upper().str.contains('NON')
    d['is_emergency'] = is_emerg
    order = (d[d['is_emergency']].groupby('major_category')['pct'].sum()
             .reindex(d['major_category'].unique(), fill_value=0)
             .sort_values())
    cats = order.index.tolist()
    types = (d[['type', 'is_emergency']].drop_duplicates()
             .sort_values(['is_emergency', 'type'], ascending=[False, True])['type'].tolist())
    palette = px.colors.qualitative.Set2

    fig = go.Figure()
    for i, t in enumerate(types):
        s = d[d['type'] == t].set_index('major_category').reindex(cats)
        fig.add_trace(go.Bar(
            y=cats, x=s['pct'].fillna(0), name=t, orientation='h',
            marker_color=palette[i % len(palette)],
            customdata=np.c_[s['problems'].fillna(0), s['total'].fillna(0)],
            hovertemplate=('%{y}<br>' + t + ': %{x:.1f}%'
                           '<br>%{customdata[0]:,.0f} of %{customdata[1]:,.0f} problems<extra></extra>')))
    fig.update_layout(barmode='stack', template='plotly_white',
                      title='Q8. Which housing problem categories are most often emergencies?',
                      xaxis_title='Share of problems in category (%)', yaxis_title='Housing problem category',
                      xaxis_range=[0, 100], legend_title_text='Urgency type',
                      height=max(450, 28 * len(cats) + 200))
    return fig


def emergency_share_table(df_housing_monthly):
    """% of problems classified as emergency, per category and borough."""
    d = df_housing_monthly.copy()
    t = d['type'].astype(str).str.upper()
    d['emergency_problems'] = np.where(t.str.contains('EMERGENCY') & ~t.str.contains('NON'), d['problems'], 0)
    g = d.groupby(['major_category', 'borough'])[['emergency_problems', 'problems']].sum()
    share = (g['emergency_problems'] / g['problems'].replace(0, np.nan) * 100).unstack('borough')
    share = share.loc[share.mean(axis=1).sort_values(ascending=False).index]
    return share, g['problems'].unstack('borough').reindex(share.index)


def build_q8_borough(df_housing_monthly):
    """Heatmap: emergency share of each housing category, by borough."""
    share, counts = emergency_share_table(df_housing_monthly)
    fig = px.imshow(share.round(1), x=[b.title() for b in share.columns], y=share.index,
                    color_continuous_scale='Reds', aspect='auto', text_auto='.0f', zmin=0, zmax=100,
                    labels={'x': 'Borough', 'y': 'Housing problem category', 'color': 'Emergency %'},
                    title='Q8b. Emergency share of each housing category, by borough')
    fig.update_traces(customdata=counts.values,
                      hovertemplate='%{y}, %{x}<br>Emergency: %{z:.1f}%<br>Problems: %{customdata:,.0f}<extra></extra>')
    fig.update_layout(template='plotly_white', height=max(450, 30 * len(share) + 150),
                      coloraxis_colorbar=dict(ticksuffix='%'))
    return fig


# ============================================================
# Q9 (Member 6): Which neighbourhoods have the most multi-problem complaints?
# ============================================================
MIN_COMPLAINTS = 500

def multi_problem_table(df_housing_nta, min_complaints=MIN_COMPLAINTS):
    """Problems, complaints and average problems per complaint per neighbourhood (NTA)."""
    d = (df_housing_nta.groupby(['nta', 'borough'], as_index=False)[['problems', 'complaints']].sum())
    d = d[d['complaints'] >= min_complaints].copy()
    d['avg_problems_per_complaint'] = d['problems'] / d['complaints']
    d['Borough'] = d['borough'].str.title()
    return d.sort_values('avg_problems_per_complaint', ascending=False).reset_index(drop=True)


def build_q9(df_housing_nta):
    """Horizontal bar of the 15 neighbourhoods with the most problems per complaint."""
    d = multi_problem_table(df_housing_nta).head(15)
    if d.empty:
        return px.bar(title=f'Q9. No neighbourhood has at least {MIN_COMPLAINTS} complaints for the selected filters')

    fig = px.bar(
        d.iloc[::-1], x='avg_problems_per_complaint', y='nta', color='Borough', orientation='h',
        color_discrete_map=BOROUGH_COLORS,
        custom_data=['problems', 'complaints'],
        text=d.iloc[::-1]['avg_problems_per_complaint'].round(2),
        labels={'avg_problems_per_complaint': 'Average problems per complaint',
                'nta': 'Neighbourhood'},
        title=f'Q9. Neighbourhoods with the most problems per complaint (min. {MIN_COMPLAINTS} complaints)')
    fig.update_traces(textposition='outside', cliponaxis=False,
                      hovertemplate=('<b>%{y}</b><br>Avg problems per complaint: %{x:.2f}<br>'
                                     'Problems: %{customdata[0]:,.0f}<br>'
                                     'Complaints: %{customdata[1]:,.0f}<extra></extra>'))
    fig.update_xaxes(range=[0, d['avg_problems_per_complaint'].max() * 1.12])
    fig.update_layout(height=520, yaxis=dict(dtick=1), legend_title_text='Borough')
    return fig



# ======================================================================
# Q10 (Member 2): At what hours and days are serious crashes most deadly?
# ======================================================================
WEEKDAY_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
HOUR_BLOCKS = [f'{h:02d}–{h + 2:02d}h' for h in range(0, 24, 3)]
MIN_CRASHES = 30


def deadly_hours_table(df_crashes):
    """Crashes, fatalities and fatality rate per weekday × 3-hour block."""
    d = df_crashes.dropna(subset=['crash_hour']).copy()
    d['day'] = d['weekday'].astype(str)
    d['block'] = (d['crash_hour'].astype(int) // 3 * 3).map(lambda h: f'{h:02d}–{h + 2:02d}h')
    d['is_fatal'] = d['total_fatalities'] > 0
    t = (d.groupby(['day', 'block'])
          .agg(crashes=('total_fatalities', 'size'), fatalities=('total_fatalities', 'sum'),
               fatal_crashes=('is_fatal', 'sum'))
          .reset_index())
    t['fatality_rate'] = t['fatalities'] / t['crashes'] * 100
    t['fatal_crash_%'] = t['fatal_crashes'] / t['crashes'] * 100
    return t


def _grid(t, value):
    """Pivot to a weekday (rows, Mon→Sun) × hour-block (columns) grid."""
    days = [w for w in WEEKDAY_ORDER if w in set(t['day'])]
    return t.pivot(index='day', columns='block', values=value).reindex(index=days, columns=HOUR_BLOCKS)


def build_q10(df_crashes):
    """Heatmap: fatalities per 100 serious crashes, by weekday and 3-hour block (crash count in hover)."""
    t = deadly_hours_table(df_crashes)
    if t.empty:
        return message_figure('No crashes for this selection.')
    rate = _grid(t, 'fatality_rate')
    crashes = _grid(t, 'crashes').fillna(0)
    rate = rate.where(crashes >= MIN_CRASHES)
    fig = px.imshow(
        rate.round(1), x=HOUR_BLOCKS, y=rate.index.tolist(),
        color_continuous_scale='Reds', aspect='auto', text_auto='.1f',
        labels={'x': 'Time of day (3-hour block)', 'y': 'Day of week', 'color': 'Fatalities per 100 crashes'},
        title='Q10. Deadliest hours and days: fatalities per 100 serious crashes')
    fig.update_traces(
        customdata=np.dstack([crashes.values, _grid(t, 'fatalities').fillna(0).values,
                              _grid(t, 'fatal_crash_%').values]),
        hovertemplate='%{y}, %{x}<br>Fatalities per 100 crashes: %{z:.1f}'
                      '<br>Serious crashes: %{customdata[0]:,.0f}<br>Fatalities: %{customdata[1]:,.0f}'
                      '<br>Crashes with a death: %{customdata[2]:.1f}%<extra></extra>')
    fig.update_layout(template='plotly_white', height=420,
                      coloraxis_colorbar=dict(title='Deaths per<br>100 crashes'))
    if (crashes < MIN_CRASHES).any().any():
        fig.add_annotation(text=f'Blank cells: fewer than {MIN_CRASHES} crashes', showarrow=False,
                           x=0, y=-0.18, xref='paper', yref='paper', xanchor='left',
                           font=dict(size=10, color='gray'))
    return fig


def build_q10_counts(df_crashes):
    """Companion heatmap: number of serious crashes (the busiest times are not the deadliest)."""
    t = deadly_hours_table(df_crashes)
    if t.empty:
        return message_figure('No crashes for this selection.')
    crashes = _grid(t, 'crashes').fillna(0)
    fig = px.imshow(
        crashes, x=HOUR_BLOCKS, y=crashes.index.tolist(),
        color_continuous_scale='Blues', aspect='auto', text_auto=',.0f',
        labels={'x': 'Time of day (3-hour block)', 'y': 'Day of week', 'color': 'Serious crashes'},
        title='Q10b. Busiest hours and days: number of serious crashes')
    fig.update_traces(hovertemplate='%{y}, %{x}<br>Serious crashes: %{z:,.0f}<extra></extra>')
    fig.update_layout(template='plotly_white', height=420, coloraxis_colorbar=dict(title='Crashes'))
    return fig


# ======================================================================
# Q11 (Member 3): Which streets are NYC's most dangerous corridors, and are they getting worse?
# ======================================================================
def build_q11(df_crashes):
    """Q11: Which streets are NYC's most dangerous corridors, and are they getting worse?"""
    d = df_crashes.dropna(subset=['on_street']).copy()
    if d.empty:
        return message_figure('No crashes with a street name for this selection.')
    s = d['on_street'].str.upper().str.strip().str.replace(r'\s+', ' ', regex=True)
    s = s.str.replace(r'\bAVE\b', 'AVENUE', regex=True)
    s = s.str.replace(r'\bBLVD\b', 'BOULEVARD', regex=True)
    s = s.str.replace(r'\bST$', 'STREET', regex=True)
    s = s.str.replace(r'\bPKWY\b', 'PARKWAY', regex=True)
    s = s.str.replace(r'\bEXPY\b', 'EXPRESSWAY', regex=True)
    d['street'] = s
    top10 = d['street'].value_counts().head(10).index
    counts = (d[d['street'].isin(top10)]
              .groupby(['street', 'year']).size().reset_index(name='crashes'))
    counts['year'] = counts['year'].astype(str)
    fig = px.bar(counts, x='crashes', y='street', color='year', orientation='h',
                 title='Q11. Most dangerous corridors: serious crashes on the top 10 streets, by year',
                 labels={'crashes': 'Serious crashes', 'street': 'Street', 'year': 'Year'},
                 category_orders={'year': sorted(counts['year'].unique())})
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, template='plotly_white', height=480)
    return fig


# ======================================================================
# Q12 (Member 5): Are night crashes more severe than day crashes, and does that differ by borough?
# ======================================================================
def build_q12(df_crashes):
    """Q12: Day vs night severity of serious crashes per borough (victims/crash and % fatal)."""
    d = df_crashes.dropna(subset=['crash_hour', 'borough']).copy()
    if d.empty:
        return message_figure('No crashes for this selection.')
    is_night = (d['crash_hour'] >= 20) | (d['crash_hour'] < 6)
    d['period'] = np.where(is_night, 'Night', 'Day')
    d['has_fatality'] = d['total_fatalities'] > 0
    g = (d.groupby(['borough', 'period'], as_index=False)
           .agg(crashes=('has_fatality', 'size'),
                victims=('total_victims', 'sum'),
                fatal_crashes=('has_fatality', 'sum')))
    g['victims_per_crash'] = (g['victims'] / g['crashes']).round(2)
    g['fatal_pct'] = (g['fatal_crashes'] / g['crashes'] * 100).round(1)
    boroughs = sorted(g['borough'].unique())
    colors = {'Day': '#F2A541', 'Night': '#2E3A87'}
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Victims per crash', '% of crashes with a fatality'))
    for period in ['Day', 'Night']:
        p = g[g['period'] == period].set_index('borough').reindex(boroughs).reset_index()
        hover = ('<b>%{x}</b> – ' + period +
                 '<br>Crashes: %{customdata[0]:,}'
                 '<br>Victims per crash: %{customdata[1]}'
                 '<br>Fatal crashes: %{customdata[2]:,} (%{customdata[3]}%)<extra></extra>')
        custom = p[['crashes', 'victims_per_crash', 'fatal_crashes', 'fatal_pct']].values
        fig.add_trace(go.Bar(x=p['borough'], y=p['victims_per_crash'], name=period,
                             marker_color=colors[period], customdata=custom,
                             hovertemplate=hover, legendgroup=period), row=1, col=1)
        fig.add_trace(go.Bar(x=p['borough'], y=p['fatal_pct'], name=period,
                             marker_color=colors[period], customdata=custom,
                             hovertemplate=hover, legendgroup=period, showlegend=False), row=1, col=2)
    fig.update_layout(barmode='group', height=500,
                      title='Night vs day severity of serious crashes by borough',
                      legend_title_text='Time of day')
    fig.update_xaxes(title_text='Borough')
    fig.update_yaxes(title_text='Victims per crash', row=1, col=1)
    fig.update_yaxes(title_text='Crashes with a fatality (%)', row=1, col=2)
    return fig
