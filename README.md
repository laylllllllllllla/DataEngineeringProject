# NYC Homes & Streets

**Data Engineering and Visualization, Milestone 1 (GIU, Winter 2026)**

Do the neighbourhoods with the worst housing conditions also have the most dangerous streets?
We integrate NYC housing maintenance problems with serious traffic crashes across NYC's 59 community
districts (2022–2025) and present the results in an interactive website.

**Live website:** _add your Render link here_

## Datasets

| Domain | Dataset | Link | Rows used |
|---|---|---|---|
| Housing (H6) | Housing Maintenance Code Complaints and Problems | https://data.cityofnewyork.us/d/ygpa-z7cr | 3,160,386 problems loaded → 2,657,915 after cleaning |
| Transportation (T4) | SIRTA Crash Data | https://data.cityofnewyork.us/d/xeqp-qz8h | 11,958 serious crashes loaded → 11,953 after cleaning |

Both were loaded with the NYC Open Data SODA API, filtered to 2022–2025.

## Repository structure

```
notebook/DataEngineering.ipynb   Full pipeline: loading, EDA, cleaning, integration, post-integration
                                 cleaning, website export and the 12 research questions
website/app.py                   Dash app: filters, search, Generate Report button, story section
website/charts.py                One function per research question (shared with the notebook)
website/assets/style.css         Website styling
website/web_data/                Small pre-aggregated files exported by the notebook (Section 6)
requirements.txt                 Python packages for the website
```

The raw and cleaned full datasets (hundreds of MB) are not in the repository; they are re-created by
running the notebook in Google Colab.

## Pipeline summary

1. **Loading:** SODA API (JSON, paginated, verified against `count(*)`), 2022–2025.
2. **EDA:** structure, missing values, duplicates, distributions, temporal and geographic checks.
3. **Cleaning:** formats and types, 502,470 HPD-flagged duplicate problems removed, invalid ZIPs and
   district codes corrected, crash locations recovered from coordinates (nearest housing location,
   validated: 93.3% ZIP agreement, 91.4% district agreement), outliers handled with domain rules.
4. **Integration:** both datasets aggregated to **community district × year** (59 × 4 = 236 rows) and merged.
5. **Post-integration cleaning:** merge NaNs, data types, redundant columns, totals verified.
6. **Export:** small tables for the website (the server has 512 MB RAM).
7. **Research questions:** 12 questions, each with an interactive chart and a written answer.

## Run the website locally

```bash
pip install -r requirements.txt
python website/app.py
```

Then open http://127.0.0.1:8050.

## Deployment (Render)

- New → Web Service → connect this GitHub repository
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn --chdir website app:server`
- Environment variable: `PYTHON_VERSION` = `3.11.9`
- Instance type: Free

## Website features

- Dropdown filters: Borough, Year, Housing category, Crash metric
- Search mode: free text such as "Bronx 2024 heat fatalities" sets the dropdowns
- **Generate Report** button: the only trigger that updates the charts
- 14 interactive Plotly charts (12 research questions + 2 companion views) and an overview chart,
  including cross-domain charts (Q1–Q6)
- Story section: question, findings, limitations and benefit to NYC

## Team contributions

| Member | Name / ID | Contribution |
|---|---|---|
| 1 | _name, ID_ | Data loading, EDA, cleaning, integration, post-integration cleaning, website export; Q1, Q7; website |
| 2 | _name, ID_ | Q2 (problem type vs crashes), Q10 (deadliest hours and days) |
| 3 | _name, ID_ | Q3 (growth vs growth), Q11 (most dangerous corridors) |
| 4 | _name, ID_ | Q4 (seasonal peaks), Q8 (emergency share by category) |
| 5 | _name, ID_ | Q5 (map of both burdens), Q12 (night vs day severity) |
| 6 | _name, ID_ | Q6 (borough share gap), Q9 (multi-problem neighbourhoods) |

## AI usage

AI assistance (Claude) was used for code and documentation. All prompts are listed in the final
section of the notebook, as required by the project description.
