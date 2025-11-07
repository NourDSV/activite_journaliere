import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
import calendar
from calendar import monthrange
import io
from PIL import Image
import plotly.io as pio





st.set_page_config(page_title="Activité journalière", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Pages",
    ["Upload", "Weekly Analysis", "Monthly Analysis", "Yearly analysis"],
    index=0
)
###################################################################################################
if 'fig_store' not in st.session_state:
    st.session_state['fig_store'] = []  

def _add_or_replace_fig(title, fig):
 
    for i, item in enumerate(st.session_state['fig_store']):
        if item['title'] == title:
            st.session_state['fig_store'][i] = {'title': title, 'fig': fig}
            break
    else:
        st.session_state['fig_store'].append({'title': title, 'fig': fig})

if page == "Upload":
    st.title("Upload & Preview")

    uploaded = st.file_uploader("Put the excel file 'Activité journalière' ", type=["xlsx","xls"])


    if uploaded:
        df = pd.read_excel(uploaded, header=[1, 2])  

        st.success("File uploaded !")

        df = df.iloc[:, 1:]
        def normalize(s):
            return str(s).strip().lower().replace(' ', '_')

        flat_cols = []
        for top, sub in df.columns:
            if pd.isna(sub):
                flat_cols.append(normalize(top))
            elif pd.isna(top):
                flat_cols.append(normalize(sub))
            else:
                flat_cols.append(f"{normalize(top)}__{normalize(sub)}")

        df.columns = flat_cols

        df = df.rename(columns={df.columns[0]: "jour", df.columns[1]: "date"})

        num_cols = [c for c in df.columns if '__' in c]
        df[num_cols] = (
            df[num_cols]
            .replace({',': '.'}, regex=True)
            .apply(pd.to_numeric, errors='coerce')
        )

        df = df.dropna(subset=num_cols, how='all')
        categories = ['system_domestique', 'system_export', 'system_import',
                    'direct_domestique', 'direct_inter']

        frames = []

        for cat in categories:
            sub = df[['jour', 'date', 
                    f'{cat}__nb_dossiers', 
                    f'{cat}__poids']].copy()
            sub['categorie'] = cat
            sub = sub.rename(columns={
                f'{cat}__nb_dossiers': 'nb_dossiers',
                f'{cat}__poids': 'poids'})
            frames.append(sub)
        df = pd.concat(frames, ignore_index=True)
        df['famille'] = df['categorie'].apply(lambda x: 'system' if x.startswith('system') else 'direct')


        df['date'] = pd.to_datetime(df['date'], errors='coerce')



        st.session_state["df"] = df

###################################################################

    if 'df' not in st.session_state:
        st.error("⚠️ Please upload the table first.")
        st.stop()

    df = st.session_state['df'].copy()

    if 'date' not in df.columns:
        st.error("Le DataFrame doit contenir une colonne 'date'.")
        st.stop()

    # Conversion et vérification
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    if df.empty:
        st.info("Aucune donnée valide détectée.")
        st.stop()

    # Dernière date disponible
    last_date = df['date'].max()

    # Formater la date joliment (ex: 5 novembre 2025)
    last_date_str = last_date.strftime("%d %B %Y")

    # Affichage
    st.markdown( f""" ### 🗓️ Activité journalière à la date du **{last_date_str}** """)

    st.subheader("Global overview" )

    ##############"



    def prev_month(year: int, month: int) -> tuple[int, int]:
        return (year - 1, 12) if month == 1 else (year, month - 1)

    def date_range(start, end):
        return (df['date'] >= start) & (df['date'] <= end)

    # --- WTD (Mon..last_date) vs previous week same span
    dow = last_date.weekday()  # Monday=0
    wtd_start = last_date - pd.Timedelta(days=dow)
    wtd_end   = last_date
    prev_wtd_end   = wtd_end - pd.Timedelta(days=7)
    prev_wtd_start = wtd_start - pd.Timedelta(days=7)

    # --- MTD (1st..last_date) vs previous month same number of days
    cur_y, cur_m, cur_d = last_date.year, last_date.month, last_date.day
    mtd_start = pd.Timestamp(year=cur_y, month=cur_m, day=1)
    mtd_end   = last_date
    py, pm = prev_month(cur_y, cur_m)
    pm_last_day = monthrange(py, pm)[1]
    pm_end_day = min(cur_d, pm_last_day)
    prev_mtd_start = pd.Timestamp(year=py, month=pm, day=1)
    prev_mtd_end   = pd.Timestamp(year=py, month=pm, day=pm_end_day)

    # --- YTD (Jan 1..last_date) vs previous year same elapsed days
    ytd_start = pd.Timestamp(year=cur_y, month=1, day=1)
    ytd_end   = last_date
    prev_ytd_start = pd.Timestamp(year=cur_y-1, month=1, day=1)
    days_elapsed = (ytd_end - ytd_start).days
    prev_ytd_end = prev_ytd_start + pd.Timedelta(days=days_elapsed)

    # =========================
    # Aggregation + KPI helpers
    # =========================
    def sum_metric(famille_value: str, start: pd.Timestamp, end: pd.Timestamp, metric: str) -> float:
        d = df[(df['famille'] == famille_value) & date_range(start, end)]
        return float(d[metric].sum()) if not d.empty else 0.0

    def fmt_int(x: float) -> str:
        return f"{x:,.0f}".replace(",", " ")

    def mk_delta_str(cur_val: float, prev_val: float) -> tuple[str, str]:
        if prev_val == 0:
            return ("—", "off")
        abs_delta = cur_val - prev_val
        pct = (abs_delta / prev_val) * 100.0
        txt = f"{abs_delta:+,.0f} ({pct:+.1f}%)".replace(",", " ")
        return (txt, "normal")

    def kpi_block(title: str, fam: str, cur_range: tuple, prev_range: tuple):
        cur_start, cur_end = cur_range
        prev_start, prev_end = prev_range

        cur_nb   = sum_metric(fam, cur_start, cur_end, 'nb_dossiers')
        prev_nb  = sum_metric(fam, prev_start, prev_end, 'nb_dossiers')
        cur_pds  = sum_metric(fam, cur_start, cur_end, 'poids')
        prev_pds = sum_metric(fam, prev_start, prev_end, 'poids')

        delta_nb_txt,  delta_nb_color  = mk_delta_str(cur_nb,  prev_nb)
        delta_pds_txt, delta_pds_color = mk_delta_str(cur_pds, prev_pds)

        st.markdown(f"#### {title}")
        c1, c2, c3, c4 = st.columns(4)

        # nb_dossiers: show current value + absolute & % delta (with color)
        with c1:
            st.metric("nb_dossiers (actuel)", fmt_int(cur_nb))
        with c2:
            st.metric("Variation", " ", delta=delta_nb_txt, delta_color=delta_nb_color)

        # poids: show current value + absolute & % delta (with color)
        with c3:
            st.metric("poids (actuel)", fmt_int(cur_pds))
        with c4:
            st.metric("Variation", " ", delta=delta_pds_txt, delta_color=delta_pds_color)

    # =========================
    # Tabs per famille (Direct / System)
    # =========================
    available = [f for f in ['direct', 'system'] if f in df['famille'].unique()]
    if not available:
        st.info("⚠️ Aucune famille 'Direct' ou 'System' trouvée dans les données.")
    else:
        tabs = st.tabs(available)
        for fam, tab in zip(available, tabs):
            with tab:
                st.markdown(f"###  Famille **{fam}**")

                # --- WTD
                kpi_block(
                    title=f"Week To Date (du {wtd_start:%d %b} au {wtd_end:%d %b}) vs WTD préc. ({prev_wtd_start:%d %b} → {prev_wtd_end:%d %b})",
                    fam=fam,
                    cur_range=(wtd_start, wtd_end),
                    prev_range=(prev_wtd_start, prev_wtd_end),
                )

                # --- MTD
                kpi_block(
                    title=f"Month To Date ({mtd_start:%b %Y} — jour {cur_d}) vs MTD préc. ({prev_mtd_start:%b %Y} — jour {min(cur_d, pm_last_day)})",
                    fam=fam,
                    cur_range=(mtd_start, mtd_end),
                    prev_range=(prev_mtd_start, prev_mtd_end),
                )

                # --- YTD
                kpi_block(
                    title=f"Year To Date ({ytd_start:%d %b %Y} → {ytd_end:%d %b %Y}) vs YTD {cur_y-1} ({prev_ytd_start:%d %b %Y} → {prev_ytd_end:%d %b %Y})",
                    fam=fam,
                    cur_range=(ytd_start, ytd_end),
                    prev_range=(prev_ytd_start, prev_ytd_end),
                )

    # with st.expander("voire tableau"):
    #     st.dataframe(df)



##########################################################################################################################################


if page == "Weekly Analysis":
    st.title("Weekly Comparison")   
    if 'df' not in st.session_state:
        st.error("⚠️ Please upload the table first.")
        st.stop()

    df = st.session_state['df'] 

    df = st.session_state['df'].copy()


    if 'df' not in st.session_state:
        st.error("Please upload the table first'.")
        st.stop()

    df = st.session_state['df'].copy()

    # -----------------------
    # 1) Ensure date + ISO week/year
    # -----------------------
    if 'date' not in df.columns:
        st.error("Le DataFrame doit contenir une colonne 'date'.")
        st.stop()

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if df['date'].isna().all():
        st.error("Aucune date valide détectée après conversion. Vérifiez la colonne 'date'.")
        st.stop()

    iso = df['date'].dt.isocalendar()
    df['year'] = iso.year.astype(int)
    df['week'] = iso.week.astype(int)

    # -----------------------
    # 2) Required columns + weekly aggregate
    # -----------------------
    required_cols = {'famille', 'categorie', 'nb_dossiers', 'poids'}
    missing = required_cols - set(df.columns)
    if missing:
        st.error(f"Colonnes manquantes dans df: {missing}")
        st.stop()

    weekly = (
        df.groupby(['year', 'week', 'famille', 'categorie'], as_index=False)
        .agg(nb_dossiers=('nb_dossiers', 'sum'),
            poids=('poids', 'sum'))
    )

    if weekly.empty:
        st.info("Aucune donnée hebdomadaire après agrégation.")
        st.stop()

    # -----------------------
    # 3) Build labels "YYYY-Www (Mon)" and sort DESC (last week first)
    # -----------------------
    weekly['week_start'] = weekly.apply(
        lambda r: date.fromisocalendar(int(r['year']), int(r['week']), 1),
        axis=1
    )
    weekly['month_name'] = pd.to_datetime(weekly['week_start']).dt.month_name().str[:3]
    weekly['year_week_label'] = weekly.apply(
        lambda r: f"{int(r['year'])}-W{int(r['week']):02d} ({r['month_name']})",
        axis=1
    )

    labels_sorted = (
        weekly[['year', 'week', 'year_week_label']]
        .drop_duplicates()
        .sort_values(['year', 'week'], ascending=[False, False])   # newest first
        ['year_week_label']
        .tolist()
    )
    default_labels = labels_sorted[:2] if len(labels_sorted) >= 2 else labels_sorted

   
    if not labels_sorted:
        st.info("Aucune semaine disponible.")
        st.stop()
    elif len(labels_sorted) == 1:
        default_left  = labels_sorted[0]
        default_right = labels_sorted[0]
    else:
        default_left  = labels_sorted[0]  
        default_right = labels_sorted[1]  

    # 3) UI : "Comparer semaine [input] à semaine [input]"
    st.markdown("**Comparer semaine**")
    col1, col2 = st.columns(2)
    with col1:
        week_left = st.selectbox(
            "Semaine (récente)",
            options=labels_sorted,
            index=labels_sorted.index(default_left),
            key="cmp_week_left"
        )
    with col2:
        week_right = st.selectbox(
            "à semaine (ancienne)",
            options=labels_sorted,
            index=labels_sorted.index(default_right),
            key="cmp_week_right"
        )

    # 4) Parsing util
    def parse_label(label: str) -> tuple[int, int]:
        core = label.split(' ')[0]  # "YYYY-Www"
        y, w = core.split('-W')
        return int(y), int(w)

    # 5) Résultat
    selected_labels = [week_left, week_right]
    selected_pairs = [parse_label(lab) for lab in selected_labels]

    if week_left == week_right:
        st.warning("Veuillez choisir deux semaines différentes pour la comparaison.")

    

    # -----------------------
    # 5) Subset to the two weeks (no other filters)
    # -----------------------
    two_weeks = weekly[weekly.apply(lambda r: (int(r['year']), int(r['week'])) in selected_pairs, axis=1)].copy()
    if two_weeks.empty:
        st.info("No data found.")
        

    # Keep exact labels present
    # (already in two_weeks via year/week -> year_week_label)
    available_labels = (
        two_weeks['year_week_label']
        .drop_duplicates()
        .tolist()
    )
    # Keep order as selected, but only those that exist
    selected_labels = [lab for lab in selected_labels if lab in available_labels]
    if len(selected_labels) != 2:
        st.error("Les labels sélectionnés ne sont pas présents dans les données agrégées.")
        st.stop()
    w1, w2 = selected_labels

    st.header(f"{w1}  vs  {w2}")

    tab1,tab2=st.tabs(["Par catégories","Par famille"])
    with tab1:
   
        # 6) Build pivots for both metrics

        def build_pivot(metric: str) -> pd.DataFrame:
            piv = two_weeks.pivot_table(
                index=['famille', 'categorie'],
                columns='year_week_label',
                values=metric,
                aggfunc='sum',
                fill_value=0
            )
            # Ensure columns order = selected_labels
            return piv.reindex(columns=selected_labels)
        
        def prepend_total_row(pivot: pd.DataFrame) -> pd.DataFrame:
            # Sum across all categories for the selected weeks
            cols = [c for c in pivot.columns if c in selected_labels]
            totals = pivot[cols].sum(axis=0)

            # Create a MultiIndex row so downstream reset_index() still has famille/categorie
            total_row = pd.DataFrame(
                [totals],
                index=pd.MultiIndex.from_tuples([('TOTAL', 'Total')], names=['famille', 'categorie'])
            )

            # Put Total first so it shows at the left in charts
            return pd.concat([total_row, pivot], axis=0, sort=False)


        pivot_dossiers = build_pivot('nb_dossiers')
        pivot_poids = build_pivot('poids')

        # -----------------------
        # 7) Plots side by side: left nb_dossiers, right poids
        
        


        def add_percentage_change(pivot: pd.DataFrame) -> pd.DataFrame:
            w1, w2 = selected_labels
            pivot['pct_change'] = ((pivot[w2] - pivot[w1]) / pivot[w1].replace({0: np.nan})) * 100
            return pivot

                ###############
        pivot_dossiers = prepend_total_row(build_pivot('nb_dossiers'))
        pivot_poids    = prepend_total_row(build_pivot('poids'))
                 ###########
        pivot_dossiers = add_percentage_change(pivot_dossiers)
        pivot_poids = add_percentage_change(pivot_poids)

        # -----------------------
        # 8) Prepare data for Plotly (melt + join % change)
        # -----------------------


        def prepare_long_df(pivot: pd.DataFrame, metric_name: str):
            w1, w2 = selected_labels
            long_df = (
                pivot.reset_index()
                    .assign(cat_key=lambda d: d['categorie'])
                    .drop(columns=['famille', 'categorie'])
                    .melt(id_vars=['cat_key', 'pct_change'], var_name='Semaine', value_name=metric_name)
            )
            # Remove pct_change row for the melted "weeks" columns
            long_df = long_df[long_df['Semaine'].isin(selected_labels)]
            # Keep order
            order = (pivot.reset_index()
                        .assign(cat_key=lambda d:   d['categorie'])
                        ['cat_key'].tolist())
            return long_df, order

        dossiers_long, order_cat = prepare_long_df(pivot_dossiers, 'nb_dossiers')
        poids_long, _ = prepare_long_df(pivot_poids, 'poids')

        # -----------------------
        # 9) Plot side-by-side interactive charts with % above bars
        # -----------------------
        col1, col2 = st.columns(2, gap="large")

        def plot_plotly_with_pct(df_long, y_col, title, ylabel):
        
            # Latest week is the FIRST selection
            w_latest, w_prev = selected_labels[0], selected_labels[1]

            # Base grouped bar (values only in hover)
            fig = px.bar(
                df_long,
                x='cat_key',
                y=y_col,
                color='Semaine',
                barmode='group',
                category_orders={'cat_key': order_cat, 'Semaine': [w_latest, w_prev]},
                title=title
            )

            # Clean hover template
            fig.update_traces(
                hovertemplate=f"{ylabel}=%{{y:.0f}}<extra></extra>")
            

            # Build wide table to get bar heights per category for both weeks
            wide = (
                df_long[df_long['Semaine'].isin([w_latest, w_prev])]
                .pivot_table(index='cat_key', columns='Semaine', values=y_col, aggfunc='sum')
            ).fillna(0)

            max_y = wide.max().max() if not wide.empty else 0
            min_offset = max_y * 0.02

            # Add % change labels above the LATEST week bar (first bar)
            for cat in wide.index:
                latest_val = wide.at[cat, w_latest] if w_latest in wide.columns else 0
                prev_val   = wide.at[cat, w_prev]   if w_prev   in wide.columns else np.nan

                # Compute % change vs previous week
                if prev_val and not np.isnan(prev_val) and prev_val != 0:
                    pct = (latest_val - prev_val) / prev_val * 100.0
                    text = f"{pct:+.1f}%"
                    color = "green" if pct > 0 else "red"
                else:
                    # No previous value (0 or NaN) -> show em dash
                    text = "—"
                    color = "gray"
                y_offset = latest_val + max(latest_val * 0.05, min_offset)
                fig.add_annotation(
                    x=cat,
                    y=y_offset,   
                    text=text,
                    showarrow=False,
                    font=dict(size=12, color=color, family="Arial Black"),
                    xanchor="right"
                )

            fig.update_layout(
                xaxis_title="Catégorie ",
                yaxis_title=ylabel,
                bargap=0.25,
                legend_title="week",
                height=520,
                margin=dict(t=60, b=60, l=10, r=10)
            )
            return fig
            

        with col1:
        
            fig_d = plot_plotly_with_pct(
                dossiers_long,
                y_col='nb_dossiers',
                title="Evolution nb_dossiers",
                ylabel="Nombre de dossiers"
            )
            st.plotly_chart(fig_d, use_container_width=True)
            _add_or_replace_fig(f"Weekly comparaison nb_dossiers", fig_d)

        with col2:
        
            fig_p = plot_plotly_with_pct(
                poids_long,
                y_col='poids',
                title="Evolution du poids",
                ylabel="Poids total"
            )
            st.plotly_chart(fig_p, use_container_width=True)
            _add_or_replace_fig(f"Weekly comparaison poids", fig_p)   
            



    with tab2:
        st.subheader("Par famille")

        def build_family_pivot(metric: str) -> pd.DataFrame:
            piv = (
                two_weeks.groupby(['famille', 'year_week_label'], as_index=False)[metric]
                .sum()
                .pivot(index='famille', columns='year_week_label', values=metric)
                .fillna(0)
            )
            return piv.reindex(columns=selected_labels)

        pivot_dossiers_fam = build_family_pivot('nb_dossiers')
        pivot_poids_fam    = build_family_pivot('poids')

        # 2) % change (same convention: selected_labels[0] = latest, [1] = previous)
        def add_percentage_change_family(pivot: pd.DataFrame) -> pd.DataFrame:
            w_latest, w_prev = selected_labels[0], selected_labels[1]
            pivot['pct_change'] = ((pivot[w_latest] - pivot[w_prev]) / pivot[w_prev].replace({0: np.nan})) * 100
            return pivot

        pivot_dossiers_fam = add_percentage_change_family(pivot_dossiers_fam)
        pivot_poids_fam    = add_percentage_change_family(pivot_poids_fam)

        # 3) Long format for Plotly (x = famille)
        def prepare_long_df_family(pivot: pd.DataFrame, metric_name: str):
            w_latest, w_prev = selected_labels[0], selected_labels[1]
            long_df = (
                pivot.reset_index()
                    .rename(columns={'famille': 'fam_key'})
                    .melt(id_vars=['fam_key', 'pct_change'], var_name='Semaine', value_name=metric_name)
            )
            long_df = long_df[long_df['Semaine'].isin([w_latest, w_prev])]
            order = pivot.reset_index()['famille'].tolist()
            return long_df, order

        dossiers_fam_long, order_fam = prepare_long_df_family(pivot_dossiers_fam, 'nb_dossiers')
        poids_fam_long,   _          = prepare_long_df_family(pivot_poids_fam,    'poids')

        # 4) Plot helper: % above the latest (left) bar with adaptive offset + color
        def plot_plotly_with_pct_family(df_long, y_col, title, ylabel):
            w_latest, w_prev = selected_labels[0], selected_labels[1]

            fig = px.bar(
                df_long,
                x='fam_key',
                y=y_col,
                color='Semaine',
                barmode='group',
                category_orders={'fam_key': order_fam, 'Semaine': [w_latest, w_prev]},
                title=title
            )

            # Clean hover (value only)
            fig.update_traces(
                hovertemplate=f"{ylabel}=%{{y:.0f}}<extra></extra>"
            )

            # Wide table to read heights of bars
            wide = (
                df_long[df_long['Semaine'].isin([w_latest, w_prev])]
                .pivot_table(index='fam_key', columns='Semaine', values=y_col, aggfunc='sum')
                .fillna(0)
            )

            # Adaptive offset so labels don't sit on the bar
            max_y = wide.max().max() if not wide.empty else 0
            min_offset = max_y * 0.02  # at least 2% of chart height

            # Get % change per family from the pivot (already computed)
            pct_map = (
                df_long[['fam_key', 'pct_change']]
                .drop_duplicates()
                .set_index('fam_key')['pct_change']
                .to_dict()
            )

            for fam in wide.index:
                latest_val = wide.at[fam, w_latest] if w_latest in wide.columns else 0
                prev_val   = wide.at[fam, w_prev]   if w_prev   in wide.columns else np.nan

                pct = pct_map.get(fam, np.nan)
                if np.isnan(pct):
                    # fallback if needed
                    if prev_val and not np.isnan(prev_val) and prev_val != 0:
                        pct = (latest_val - prev_val) / prev_val * 100.0

                if pct is None or np.isnan(pct):
                    text, color = "—", "gray"
                else:
                    text  = f"{pct:+.1f}%"
                    color = "green" if pct > 0 else "red"

                # Position label above the latest (left) bar:
                y_offset = latest_val + max(latest_val * 0.05, min_offset)

                fig.add_annotation(
                    x=fam,           # group center on x
                    y=y_offset,      # adaptive vertical offset
                    xref="x",
                    yref="y",
                    text=text,
                    showarrow=False,
                    font=dict(size=12, color=color, family="Arial Black"),
                    xanchor="right",
                    xshift=-20       # shift left so it sits above the left bar (tune if needed)
                )

            fig.update_layout(
                xaxis_title="Famille",
                yaxis_title=ylabel,
                bargap=0.25,
                legend_title="week",
                height=520,
                margin=dict(t=60, b=60, l=10, r=10)
            )
            return fig

        # 5) Render two charts side by side
        col3, col4 = st.columns(2, gap="large")

        with col3:
            fig_fam_d = plot_plotly_with_pct_family(
                dossiers_fam_long,
                y_col='nb_dossiers',
                title="Evolution nb_dossiers",
                ylabel="Nombre de dossiers"
            )
            st.plotly_chart(fig_fam_d, use_container_width=True)
            _add_or_replace_fig(f"Weekly comparaison nb_dossiers par famille", fig_fam_d)

        with col4:
            fig_fam_p = plot_plotly_with_pct_family(
                poids_fam_long,
                y_col='poids',
                title="Evolution du poids",
                ylabel="Poids total"
            )
            st.plotly_chart(fig_fam_p, use_container_width=True)
            _add_or_replace_fig(f"weekly comparaison poids par famille", fig_fam_p)
            



                     ###############################################



    
    # =========================
# Weekly evolution: Bar + Line (user-chosen period, 2 tabs)
# =========================
    st.subheader("Weekly evolution")

    # Build week labels ASC for clean time axis
    labels_sorted_asc = (
        weekly[['year', 'week', 'year_week_label']]
        .drop_duplicates()
        .sort_values(['year', 'week'], ascending=[False, False])
        ['year_week_label']
        .tolist()
    )

    # Default period = last 12 weeks (fallbacks if less)
    default_start = labels_sorted_asc[11] if len(labels_sorted_asc) > 11 else labels_sorted_asc[-1]
    default_end   = labels_sorted_asc[0]


############
    col1, col2 = st.columns(2)

    with col1:
        end_week = st.selectbox(
            "A partir de semaine",
            options=labels_sorted_asc,
            index=labels_sorted_asc.index(default_end)
        )

    with col2:
        start_week = st.selectbox(
            "jusqu'à",
            options=labels_sorted_asc,
            index=labels_sorted_asc.index(default_start)
        )

    period = (start_week, end_week)
#############""
    # Helpers to parse label -> (year, week) and to build key
    def parse_week_label(label: str):
        core = label.split(' ')[0]  # "YYYY-Www"
        y, w = core.split('-W')
        return int(y), int(w)

    (y_start, w_start) = parse_week_label(period[0])
    (y_end,   w_end)   = parse_week_label(period[1])

    weekly = weekly.copy()
    weekly['_key'] = weekly['year'] * 100 + weekly['week']
    k_start = y_start * 100 + w_start
    k_end   = y_end   * 100 + w_end

    # Aggregate per week across all familles/catégories for the evolution charts
    evol = (
        weekly[(weekly['_key'] >= k_start) & (weekly['_key'] <= k_end)]
        .groupby(['year', 'week', 'year_week_label'], as_index=False)
        .agg(nb_dossiers=('nb_dossiers', 'sum'),
            poids=('poids', 'sum'))
        .sort_values(['year', 'week'])
    )

    if evol.empty:
        st.info("No data found.")
    else:
      

        tab_a, tab_b = st.tabs(["Total", "Par famille"])
        with tab_a:
            for bar_metric, y_label in [
                ('nb_dossiers', 'Nombre de dossiers'),
                ('poids',       'Poids total'),
            ]:
                evol_metric = evol.copy()
                # 4-week moving average
                evol_metric['ema4'] = evol_metric[bar_metric].ewm(span=4, adjust=False).mean()


                fig = go.Figure()

                # Bars
                fig.add_bar(
                    x=evol_metric['year_week_label'],
                    y=evol_metric[bar_metric],
                    name=bar_metric,
                    hovertemplate=f"Week=%{{x}}<br>{bar_metric}=%{{y:.0f}}<extra></extra>"
                )

                # Line (MA4)
                fig.add_trace(go.Scatter(
                    x=evol_metric['year_week_label'],
                    y=evol_metric['ema4'],
                    mode='lines+markers',
                    name=f"EMA 4 W. ({bar_metric})",
                    hovertemplate=f"Week=%{{x}}<br>MA4=%{{y:.0f}}<extra></extra>"
                ))

                fig.update_layout(
                    title=f"Evolution {bar_metric} — {period[1]} → {period[0]}",
                    xaxis_title="Week",
                    yaxis_title=y_label,
                    bargap=0.25,
                    height=520,
                    margin=dict(t=60, b=60, l=10, r=10),
                    legend_title="label"
                )
                fig.update_xaxes(tickangle=-45)

        

                st.plotly_chart(fig, use_container_width=True)
                _add_or_replace_fig(f"Weekly évolution nb_dossiers ", fig)

            with st.expander("EMA?"):
                st.write("""
                La **moyenne mobile pondérée**, ou **EMA**, est une courbe qui montre la tendance
                des dernières semaines tout en réagissant plus vite aux changements récents.

                Elle donne **plus d’importance aux dernières données** en les prenant davantage en compte
                dans le calcul de la moyenne.  
                Concrètement, cela permet de mieux voir quand la tendance commence à monter ou à baisser.
                
                🔹 **formule:**  
                Chaque nouvelle valeur compte davantage dans le calcul de la moyenne :
                
                EMA_t = α × Valeur_t + (1 - α) × EMA_{t-1}
                Pour une période de 4 semaines, le facteur de pondération est :
                α = 2 / (4 + 1) = 0.4
                
                👉 Cela signifie que **la dernière semaine pèse 40 %** dans le calcul, tandis que les semaines
                précédentes représentent les **60 % restants**, rendant la courbe plus réactive aux changements récents.
                """)
            


        with tab_b:
            # Aggregate over the selected period by famille
            fam_week = (
            weekly[(weekly['_key'] >= k_start) & (weekly['_key'] <= k_end)]
            .groupby(['year', 'week', 'year_week_label', 'famille'], as_index=False)
            .agg(nb_dossiers=('nb_dossiers', 'sum'),
                poids=('poids', 'sum'))
            .sort_values(['year', 'week'])
        )

            # 2) Filter only desired familles (Direct, System) if they exist
            target_familles = ['Direct', 'System']
            fam_week = fam_week[fam_week['famille'].isin(target_familles)] if any(
                f in fam_week['famille'].unique() for f in target_familles
            ) else fam_week

            # 3) Pivot by week vs famille
            def make_week_pivot(metric):
                return (fam_week
                    .pivot_table(index='year_week_label', columns='famille', values=metric, aggfunc='sum')
                    .fillna(0)
                    .reset_index())

            # 4) Custom color palette for familles
            fam_colors = {
                'direct': '#1f77b4',  # blue
                'system': '#2ca02c',  # 
                'Autre': '#ff7f0e'    #  fallback if exists
            }
            ema_colors = {
                'direct': '#155485',
                'system': '#207820'
            }


            for bar_metric, y_label in [
                ('nb_dossiers', 'Nombre de dossiers'),
                ('poids',       'Poids total'),
            ]:
                pvt = make_week_pivot(bar_metric)

                fam_cols = [c for c in ['Direct', 'System'] if c in pvt.columns]
                fam_cols += [c for c in pvt.columns if c not in fam_cols + ['year_week_label']]

                fig_fam_week = go.Figure()

            # 5) Add bars per famille with defined colors
                for fam in fam_cols:
                    if fam == 'year_week_label':
                        continue
                    fig_fam_week.add_bar(
                        x=pvt['year_week_label'],
                        y=pvt[fam],
                        name=fam,
                        marker_color=fam_colors.get(fam, '#cccccc'),
                        hovertemplate=f"Semaine=%{{x}}<br>Famille={fam}<br>{bar_metric}=%{{y:.0f}}<extra></extra>"
                    )
                for fam in fam_cols:
                    if fam == 'year_week_label':
                        continue
                    ema4 = pvt[fam].ewm(span=4, adjust=False, min_periods=1).mean()
                    fig_fam_week.add_trace(
                        go.Scatter(
                            x=pvt['year_week_label'],
                            y=ema4,
                            mode='lines',
                            name=f"EMA 4 sem — {fam}",
                            line=dict(width=2, color=ema_colors.get(fam, 'black')),
                            hovertemplate=f"Semaine=%{{x}}<br>Famille={fam}<br>EMA 4 sem=%{{y:.0f}}<extra></extra>",
                            legendgroup=fam
                        ))
                    

                fig_fam_week.update_layout(
                    title=f"{bar_metric}, semaine par semaine — {period[1]} → {period[0]}",
                    xaxis_title="Semaine",
                    yaxis_title=y_label,
                    barmode='group',
                    bargap=0.25,
                    height=520,
                    margin=dict(t=60, b=60, l=10, r=10),
                    legend_title="Famille"
                )
                fig_fam_week.update_xaxes(tickangle=-45)

                st.plotly_chart(fig_fam_week, use_container_width=True)

########################################################################################################################################################################







#######################################################################################################################################################################
# =========================
# Monthly Analysis
# =========================

if page == "Monthly Analysis":
    st.title("Monthly Comparison")
    if 'df' not in st.session_state:
        st.error("⚠️ Please upload the table first.")
        st.stop()

    df = st.session_state['df'].copy()

    # -----------------------
    # 1) Ensure date + year/month
    # -----------------------
    if 'date' not in df.columns:
        st.error("Le DataFrame doit contenir une colonne 'date'.")
        st.stop()

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if df['date'].isna().all():
        st.error("Aucune date valide détectée après conversion. Vérifiez la colonne 'date'.")
        st.stop()

    df['year'] = df['date'].dt.year.astype(int)
    df['month'] = df['date'].dt.month.astype(int)
    df['month_name'] = df['date'].dt.month_name().str[:3]

    # -----------------------
    # 2) Required columns + monthly aggregate
    # -----------------------
    required_cols = {'famille', 'categorie', 'nb_dossiers', 'poids'}
    missing = required_cols - set(df.columns)
    if missing:
        st.error(f"Colonnes manquantes dans df: {missing}")
        st.stop()

    monthly = (
        df.groupby(['year', 'month', 'famille', 'categorie'], as_index=False)
          .agg(nb_dossiers=('nb_dossiers', 'sum'),
               poids=('poids', 'sum'))
    )
    if monthly.empty:
        st.info("Aucune donnée mensuelle après agrégation.")
        st.stop()

    # -----------------------
    # 3) Build labels "YYYY-MMM" and sort DESC (last month first)
    # -----------------------
    monthly['year_month_label'] = monthly.apply(
        lambda r: f"{int(r['year'])}-{int(r['month']):02d} ({calendar.month_abbr[int(r['month'])]})", axis=1
    )

    labels_sorted = (
        monthly[['year', 'month', 'year_month_label']]
        .drop_duplicates()
        .sort_values(['year', 'month'], ascending=[False, False])   # newest first
        ['year_month_label']
        .tolist()
    )

    if not labels_sorted:
        st.info("Aucun mois disponible.")
        st.stop()

    # Helper: parse "YYYY-MM (Mon)"
    def parse_month_label(label: str) -> tuple[int, int]:
        core = label.split(' ')[0]  # "YYYY-MM"
        y, m = core.split('-')
        return int(y), int(m)

    # Map (y,m) -> label to safely find "same month last year"
    ym_to_label = {
        (int(r['year']), int(r['month'])): r['year_month_label']
        for _, r in monthly[['year', 'month', 'year_month_label']].drop_duplicates().iterrows()
    }

    # Defaults: last month (latest found) vs same month last year if exists, else previous available label
    default_left = labels_sorted[0]  # most recent month present in data
    y_latest, m_latest = parse_month_label(default_left)
    default_right = ym_to_label.get((y_latest - 1, m_latest), (labels_sorted[1] if len(labels_sorted) > 1 else labels_sorted[0]))

    # -----------------------
    # 4) UI: comparer mois A à mois B
    # -----------------------
    st.markdown("**Comparer mois**")
    col1, col2 = st.columns(2)
    with col1:
        month_left = st.selectbox(
            "Mois (récent)",
            options=labels_sorted,
            index=labels_sorted.index(default_left),
            key="cmp_month_left"
        )
    with col2:
        month_right = st.selectbox(
            "à mois (référence)",
            options=labels_sorted,
            index=labels_sorted.index(default_right),
            key="cmp_month_right"
        )

    selected_labels = [month_left, month_right]
    selected_pairs = [parse_month_label(lab) for lab in selected_labels]

    if month_left == month_right:
        st.warning("Veuillez choisir deux mois différents pour la comparaison.")

    # -----------------------
    # 5) Subset to the two months (no other filters)
    # -----------------------
    two_months = monthly[monthly.apply(lambda r: (int(r['year']), int(r['month'])) in selected_pairs, axis=1)].copy()
    if two_months.empty:
        st.info("No data found.")

    # Keep only labels present & preserve order
    available_labels = two_months['year_month_label'].drop_duplicates().tolist()
    selected_labels = [lab for lab in selected_labels if lab in available_labels]
    if len(selected_labels) != 2:
        st.error("Les labels sélectionnés ne sont pas présents dans les données agrégées.")
        st.stop()
    m1, m2 = selected_labels

    st.header(f"{m1}  vs  {m2}")

    # -----------------------
    # Tabs: Par catégories / Par famille
    # -----------------------
    tab1, tab2 = st.tabs(["Par catégories", "Par famille"])

    # ---------- Common helpers ----------
    def prepend_total_row(pivot: pd.DataFrame, selected_labels_order: list[str]) -> pd.DataFrame:
        cols = [c for c in pivot.columns if c in selected_labels_order]
        totals = pivot[cols].sum(axis=0)
        total_row = pd.DataFrame(
            [totals],
            index=pd.MultiIndex.from_tuples([('TOTAL', 'Total')], names=['famille', 'categorie'])
        )
        return pd.concat([total_row, pivot], axis=0, sort=False)

    def add_percentage_change(pivot: pd.DataFrame, sel_labels: list[str]) -> pd.DataFrame:
        w1, w2 = sel_labels  # w1 = latest, w2 = reference
        pivot['pct_change'] = ((pivot[w1] - pivot[w2]) / pivot[w2].replace({0: np.nan})) * 100
        return pivot

    with tab1:
        # ---------- Catégories ----------
        def build_pivot(metric: str) -> pd.DataFrame:
            piv = two_months.pivot_table(
                index=['famille', 'categorie'],
                columns='year_month_label',
                values=metric,
                aggfunc='sum',
                fill_value=0
            )
            return piv.reindex(columns=selected_labels)

        pivot_dossiers = prepend_total_row(build_pivot('nb_dossiers'), selected_labels)
        pivot_poids    = prepend_total_row(build_pivot('poids'),       selected_labels)

        pivot_dossiers = add_percentage_change(pivot_dossiers, selected_labels)
        pivot_poids    = add_percentage_change(pivot_poids,    selected_labels)

        def prepare_long_df(pivot: pd.DataFrame, metric_name: str):
            w1, w2 = selected_labels
            long_df = (
                pivot.reset_index()
                     .assign(cat_key=lambda d: d['categorie'])
                     .drop(columns=['famille', 'categorie'])
                     .melt(id_vars=['cat_key', 'pct_change'], var_name='Mois', value_name=metric_name)
            )
            long_df = long_df[long_df['Mois'].isin(selected_labels)]
            order = (pivot.reset_index()
                        .assign(cat_key=lambda d: d['categorie'])
                        ['cat_key'].tolist())
            return long_df, order

        dossiers_long, order_cat = prepare_long_df(pivot_dossiers, 'nb_dossiers')
        poids_long, _            = prepare_long_df(pivot_poids,    'poids')

        col1, col2 = st.columns(2, gap="large")

        def plot_plotly_with_pct(df_long, y_col, title, ylabel):
            w_latest, w_ref = selected_labels[0], selected_labels[1]
            fig = px.bar(
                df_long,
                x='cat_key',
                y=y_col,
                color='Mois',
                barmode='group',
                category_orders={'cat_key': order_cat, 'Mois': [w_latest, w_ref]},
                title=title
            )
            fig.update_traces(hovertemplate=f"{ylabel}=%{{y:.0f}}<extra></extra>")

            wide = (
                df_long[df_long['Mois'].isin([w_latest, w_ref])]
                .pivot_table(index='cat_key', columns='Mois', values=y_col, aggfunc='sum')
            ).fillna(0)

            max_y = wide.max().max() if not wide.empty else 0
            min_offset = max_y * 0.02

            for cat in wide.index:
                latest_val = wide.at[cat, w_latest] if w_latest in wide.columns else 0
                prev_val   = wide.at[cat, w_ref]    if w_ref    in wide.columns else np.nan

                if prev_val and not np.isnan(prev_val) and prev_val != 0:
                    pct = (latest_val - prev_val) / prev_val * 100.0
                    text = f"{pct:+.1f}%"
                    color = "green" if pct > 0 else "red"
                else:
                    text = "—"
                    color = "gray"

                y_offset = latest_val + max(latest_val * 0.05, min_offset)
                fig.add_annotation(
                    x=cat,
                    y=y_offset,
                    text=text,
                    showarrow=False,
                    font=dict(size=12, color=color, family="Arial Black"),
                    xanchor="right"
                )

            fig.update_layout(
                xaxis_title="Catégorie",
                yaxis_title=ylabel,
                bargap=0.25,
                legend_title="Mois",
                height=520,
                margin=dict(t=60, b=60, l=10, r=10)
            )
            return fig

        with col1:
            st.plotly_chart(
                plot_plotly_with_pct(dossiers_long, 'nb_dossiers', "Évolution nb_dossiers", "Nombre de dossiers"),
                use_container_width=True)
            _add_or_replace_fig(f"monthly comparaison nb_dossiers", "Évolution nb_dossiers")
            
        with col2:
            st.plotly_chart(
                plot_plotly_with_pct(poids_long, 'poids', "Évolution du poids", "Poids total"),
                use_container_width=True
            )
            _add_or_replace_fig(f"monthly comparaison poids", "Évolution du poids")

    with tab2:
        # ---------- Famille ----------
        def build_family_pivot(metric: str) -> pd.DataFrame:
            piv = (
                two_months.groupby(['famille', 'year_month_label'], as_index=False)[metric]
                         .sum()
                         .pivot(index='famille', columns='year_month_label', values=metric)
                         .fillna(0)
            )
            return piv.reindex(columns=selected_labels)

        pivot_dossiers_fam = build_family_pivot('nb_dossiers')
        pivot_poids_fam    = build_family_pivot('poids')

        def add_percentage_change_family(pivot: pd.DataFrame) -> pd.DataFrame:
            w_latest, w_prev = selected_labels[0], selected_labels[1]
            pivot['pct_change'] = ((pivot[w_latest] - pivot[w_prev]) / pivot[w_prev].replace({0: np.nan})) * 100
            return pivot

        pivot_dossiers_fam = add_percentage_change_family(pivot_dossiers_fam)
        pivot_poids_fam    = add_percentage_change_family(pivot_poids_fam)

        def prepare_long_df_family(pivot: pd.DataFrame, metric_name: str):
            w_latest, w_prev = selected_labels[0], selected_labels[1]
            long_df = (
                pivot.reset_index()
                     .rename(columns={'famille': 'fam_key'})
                     .melt(id_vars=['fam_key', 'pct_change'], var_name='Mois', value_name=metric_name)
            )
            long_df = long_df[long_df['Mois'].isin([w_latest, w_prev])]
            order = pivot.reset_index()['famille'].tolist()
            return long_df, order

        dossiers_fam_long, order_fam = prepare_long_df_family(pivot_dossiers_fam, 'nb_dossiers')
        poids_fam_long,   _          = prepare_long_df_family(pivot_poids_fam,    'poids')

        def plot_plotly_with_pct_family(df_long, y_col, title, ylabel):
            w_latest, w_prev = selected_labels[0], selected_labels[1]
            fig = px.bar(
                df_long,
                x='fam_key',
                y=y_col,
                color='Mois',
                barmode='group',
                category_orders={'fam_key': order_fam, 'Mois': [w_latest, w_prev]},
                title=title
            )
            fig.update_traces(hovertemplate=f"{ylabel}=%{{y:.0f}}<extra></extra>")

            wide = (
                df_long[df_long['Mois'].isin([w_latest, w_prev])]
                .pivot_table(index='fam_key', columns='Mois', values=y_col, aggfunc='sum')
                .fillna(0)
            )

            max_y = wide.max().max() if not wide.empty else 0
            min_offset = max_y * 0.02

            pct_map = (
                df_long[['fam_key', 'pct_change']]
                .drop_duplicates()
                .set_index('fam_key')['pct_change']
                .to_dict()
            )

            for fam in wide.index:
                latest_val = wide.at[fam, w_latest] if w_latest in wide.columns else 0
                prev_val   = wide.at[fam, w_prev]   if w_prev   in wide.columns else np.nan

                pct = pct_map.get(fam, np.nan)
                if np.isnan(pct):
                    if prev_val and not np.isnan(prev_val) and prev_val != 0:
                        pct = (latest_val - prev_val) / prev_val * 100.0

                if pct is None or np.isnan(pct):
                    text, color = "—", "gray"
                else:
                    text, color = f"{pct:+.1f}%", ("green" if pct > 0 else "red")

                y_offset = latest_val + max(latest_val * 0.05, min_offset)
                fig.add_annotation(
                    x=fam,
                    y=y_offset,
                    xref="x", yref="y",
                    text=text, showarrow=False,
                    font=dict(size=12, color=color, family="Arial Black"),
                    xanchor="right",
                    xshift=-20
                )

            fig.update_layout(
                xaxis_title="Famille",
                yaxis_title=ylabel,
                bargap=0.25,
                legend_title="Mois",
                height=520,
                margin=dict(t=60, b=60, l=10, r=10)
            )
            return fig

        col3, col4 = st.columns(2, gap="large")
        with col3:
            st.plotly_chart(
                plot_plotly_with_pct_family(dossiers_fam_long, 'nb_dossiers', "Évolution nb_dossiers", "Nombre de dossiers"),
                use_container_width=True
            )
            _add_or_replace_fig(f"monthly comparaison nb_dossiers par famille", "Évolution nb_dossiers")
        with col4:
            st.plotly_chart(
                plot_plotly_with_pct_family(poids_fam_long, 'poids', "Évolution du poids", "Poids total"),
                use_container_width=True
            )
            _add_or_replace_fig(f"monthly comparaison poids par famille", "Évolution du poids")

    # =====================================================
    # Monthly evolution: Bar + Line (user-chosen period, 2 tabs)
    # =====================================================
    st.subheader("Monthly evolution")

    labels_sorted_desc = (
        monthly[['year', 'month', 'year_month_label']]
        .drop_duplicates()
        .sort_values(['year', 'month'], ascending=[False, False])  # newest first
        ['year_month_label']
        .tolist()
    )

    # Default period = last 12 months window
    default_end   = labels_sorted_desc[0]
    default_start = labels_sorted_desc[12] if len(labels_sorted_desc) > 12 else labels_sorted_desc[-1]

    col1, col2 = st.columns(2)
    with col1:
        end_month = st.selectbox(
            "À partir de mois",
            options=labels_sorted_desc,
            index=labels_sorted_desc.index(default_end)
        )
    with col2:
        start_month = st.selectbox(
            "Jusqu'à",
            options=labels_sorted_desc,
            index=labels_sorted_desc.index(default_start)
        )
    period = (start_month, end_month)

    def parse_month_key(label: str) -> tuple[int, int]:
        y, m = parse_month_label(label)
        return y, m

    (y_start, m_start) = parse_month_key(period[0])
    (y_end,   m_end)   = parse_month_key(period[1])

    monthly = monthly.copy()
    monthly['_key'] = monthly['year'] * 100 + monthly['month']
    k_start = y_start * 100 + m_start
    k_end   = y_end   * 100 + m_end

    evol = (
        monthly[(monthly['_key'] >= k_start) & (monthly['_key'] <= k_end)]
        .groupby(['year', 'month', 'year_month_label'], as_index=False)
        .agg(nb_dossiers=('nb_dossiers', 'sum'),
             poids=('poids',       'sum'))
        .sort_values(['year', 'month'])
    )

    if evol.empty:
        st.info("No data found.")
    else:
        tab_a, tab_b = st.tabs(["Total", "Par famille"])

        with tab_a:
            for bar_metric, y_label in [
                ('nb_dossiers', 'Nombre de dossiers'),
                ('poids',       'Poids total'),
            ]:
                evol_metric = evol.copy()
                # 3-month EMA for months
                evol_metric['ema3'] = evol_metric[bar_metric].ewm(span=3, adjust=False).mean()

                fig = go.Figure()
                fig.add_bar(
                    x=evol_metric['year_month_label'],
                    y=evol_metric[bar_metric],
                    name=bar_metric,
                    hovertemplate=f"Mois=%{{x}}<br>{bar_metric}=%{{y:.0f}}<extra></extra>"
                )
                fig.add_trace(go.Scatter(
                    x=evol_metric['year_month_label'],
                    y=evol_metric['ema3'],
                    mode='lines+markers',
                    name=f"EMA 3 mois ({bar_metric})",
                    hovertemplate=f"Mois=%{{x}}<br>EMA 3 mois=%{{y:.0f}}<extra></extra>"
                ))
                fig.update_layout(
                    title=f"Évolution {bar_metric} — {period[1]} → {period[0]}",
                    xaxis_title="Mois",
                    yaxis_title=y_label,
                    bargap=0.25,
                    height=520,
                    margin=dict(t=60, b=60, l=10, r=10),
                    legend_title="Label"
                )
                fig.update_xaxes(tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                _add_or_replace_fig(f"monthly evolution {bar_metric}", fig)

            with st.expander("EMA ?"):
                st.write("""
                La **moyenne mobile exponentielle (EMA)** lisse l’évolution mensuelle tout en
                réagissant davantage aux changements récents.  
                Chaque nouveau mois compte plus que les anciens dans le calcul de la moyenne.

                Formule (rappel) :  
                EMA_t = α × Valeur_t + (1 − α) × EMA_{t−1}

                Pour **3 mois**, le facteur de pondération est **α = 2 / (3 + 1) = 0,5**.  
                👉 Le dernier mois pèse donc **50 %** dans la moyenne, ce qui rend la courbe plus réactive.
                """)

        with tab_b:
            fam_month = (
                monthly[(monthly['_key'] >= k_start) & (monthly['_key'] <= k_end)]
                .groupby(['year', 'month', 'year_month_label', 'famille'], as_index=False)
                .agg(nb_dossiers=('nb_dossiers', 'sum'),
                     poids=('poids',       'sum'))
                .sort_values(['year', 'month'])
            )

            target_familles = ['Direct', 'System']
            fam_month = fam_month[fam_month['famille'].isin(target_familles)] if any(
                f in fam_month['famille'].unique() for f in target_familles
            ) else fam_month

            def make_month_pivot(metric):
                return (fam_month
                    .pivot_table(index='year_month_label', columns='famille', values=metric, aggfunc='sum')
                    .fillna(0)
                    .reset_index())

            fam_colors = {
                'direct': '#1f77b4',
                'system': '#2ca02c',
                'Autre':  '#ff7f0e'
            }
            ema_colors = {
                'direct': '#155485',
                'system': '#207820'
            }

            for bar_metric, y_label in [
                ('nb_dossiers', 'Nombre de dossiers'),
                ('poids',       'Poids total'),
            ]:
                pvt = make_month_pivot(bar_metric)
                fam_cols = [c for c in ['Direct', 'System'] if c in pvt.columns]
                fam_cols += [c for c in pvt.columns if c not in fam_cols + ['year_month_label']]

                fig_fam_month = go.Figure()

                for fam in fam_cols:
                    if fam == 'year_month_label':
                        continue
                    fig_fam_month.add_bar(
                        x=pvt['year_month_label'],
                        y=pvt[fam],
                        name=fam,
                        marker_color=fam_colors.get(fam, '#cccccc'),
                        hovertemplate=f"Mois=%{{x}}<br>Famille={fam}<br>{bar_metric}=%{{y:.0f}}<extra></extra>"
                    )

                for fam in fam_cols:
                    if fam == 'year_month_label':
                        continue
                    ema3 = pvt[fam].ewm(span=3, adjust=False, min_periods=1).mean()
                    fig_fam_month.add_trace(
                        go.Scatter(
                            x=pvt['year_month_label'],
                            y=ema3,
                            mode='lines',
                            name=f"EMA 3 mois — {fam}",
                            line=dict(width=2, color=ema_colors.get(fam, 'black')),
                            hovertemplate=f"Mois=%{{x}}<br>Famille={fam}<br>EMA 3 mois=%{{y:.0f}}<extra></extra>",
                            legendgroup=fam
                        )
                    )

                fig_fam_month.update_layout(
                    title=f"{bar_metric}, mois par mois — {period[1]} → {period[0]}",
                    xaxis_title="Mois",
                    yaxis_title=y_label,
                    barmode='group',
                    bargap=0.25,
                    height=520,
                    margin=dict(t=60, b=60, l=10, r=10),
                    legend_title="Famille"
                )
                fig_fam_month.update_xaxes(tickangle=-45)
                st.plotly_chart(fig_fam_month, use_container_width=True)
                _add_or_replace_fig(f"monthly evolution {bar_metric} par famille", fig)



##################################################################################################""





#############################################################################################
# =========================
# Yearly Analysis (same UX as Weekly/Monthly)
# =========================
# =========================
# Yearly Analysis
# =========================
# =========================
# Yearly Analysis
# =========================
if page == "Yearly analysis":
    st.title("Yearly Comparison")

    if 'df' not in st.session_state:
        st.error("⚠️ Please upload the table first.")
        st.stop()

    df = st.session_state['df'].copy()

    # -----------------------
    # 1) Ensure date + year
    # -----------------------
    if 'date' not in df.columns:
        st.error("Le DataFrame doit contenir une colonne 'date'.")
        st.stop()

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if df['date'].isna().all():
        st.error("Aucune date valide détectée après conversion. Vérifiez la colonne 'date'.")
        st.stop()

    df['year'] = df['date'].dt.year.astype(int)

    # -----------------------
    # 2) Required columns + yearly aggregate
    # -----------------------
    required_cols = {'famille', 'categorie', 'nb_dossiers', 'poids'}
    missing = required_cols - set(df.columns)
    if missing:
        st.error(f"Colonnes manquantes dans df: {missing}")
        st.stop()

    yearly = (
        df.groupby(['year', 'famille', 'categorie'], as_index=False)
          .agg(nb_dossiers=('nb_dossiers', 'sum'),
               poids=('poids', 'sum'))
    )
    if yearly.empty:
        st.info("Aucune donnée annuelle après agrégation.")
        st.stop()

    # -----------------------
    # 3) Build labels and sort DESC
    # -----------------------
    yearly['year_label'] = yearly['year'].astype(str)

    labels_sorted = (
        yearly[['year', 'year_label']]
        .drop_duplicates()
        .sort_values(['year'], ascending=[False])
        ['year_label']
        .tolist()
    )

    if not labels_sorted:
        st.info("Aucune année disponible.")
        st.stop()

    # Defaults: most recent year vs previous year (if exists)
    default_left  = labels_sorted[0]
    default_right = labels_sorted[1] if len(labels_sorted) > 1 else labels_sorted[0]

    # -----------------------
    # 4) UI: comparer année A à année B
    # -----------------------
    st.markdown("**Comparer année**")
    col1, col2 = st.columns(2)
    with col1:
        year_left = st.selectbox(
            "Année (récente)",
            options=labels_sorted,
            index=labels_sorted.index(default_left),
            key="cmp_year_left"
        )
    with col2:
        year_right = st.selectbox(
            "à année (référence)",
            options=labels_sorted,
            index=labels_sorted.index(default_right),
            key="cmp_year_right"
        )

    selected_labels = [year_left, year_right]
    selected_years = [int(l) for l in selected_labels]

    if year_left == year_right:
        st.warning("Veuillez choisir deux années différentes pour la comparaison.")

    # -----------------------
    # 5) Subset to the two years
    # -----------------------
    two_years = yearly[yearly['year'].isin(selected_years)].copy()
    if two_years.empty:
        st.info("No data found.")

    available_labels = two_years['year_label'].drop_duplicates().tolist()
    selected_labels = [lab for lab in selected_labels if lab in available_labels]
    if len(selected_labels) != 2:
        st.error("Les années sélectionnées ne sont pas présentes dans les données.")
        st.stop()

    y1, y2 = selected_labels
    st.header(f"{y1}  vs  {y2}")

    # -----------------------
    # Tabs: Par catégories / Par famille
    # -----------------------
    tab1, tab2 = st.tabs(["Par catégories", "Par famille"])

    def prepend_total_row(pivot: pd.DataFrame, selected_labels_order: list[str]) -> pd.DataFrame:
        cols = [c for c in pivot.columns if c in selected_labels_order]
        totals = pivot[cols].sum(axis=0)
        total_row = pd.DataFrame(
            [totals],
            index=pd.MultiIndex.from_tuples([('TOTAL', 'Total')], names=['famille', 'categorie'])
        )
        return pd.concat([total_row, pivot], axis=0, sort=False)

    def add_percentage_change(pivot: pd.DataFrame, sel_labels: list[str]) -> pd.DataFrame:
        latest, prev = sel_labels[0], sel_labels[1]
        pivot['pct_change'] = ((pivot[latest] - pivot[prev]) / pivot[prev].replace({0: np.nan})) * 100
        return pivot

    with tab1:
        # ---------- Catégories ----------
        def build_pivot(metric: str) -> pd.DataFrame:
            piv = two_years.pivot_table(
                index=['famille', 'categorie'],
                columns='year_label',
                values=metric,
                aggfunc='sum',
                fill_value=0
            )
            return piv.reindex(columns=selected_labels)

        pivot_dossiers = prepend_total_row(build_pivot('nb_dossiers'), selected_labels)
        pivot_poids    = prepend_total_row(build_pivot('poids'),       selected_labels)

        pivot_dossiers = add_percentage_change(pivot_dossiers, selected_labels)
        pivot_poids    = add_percentage_change(pivot_poids,    selected_labels)

        def prepare_long_df(pivot: pd.DataFrame, metric_name: str):
            long_df = (
                pivot.reset_index()
                     .assign(cat_key=lambda d: d['categorie'])
                     .drop(columns=['famille', 'categorie'])
                     .melt(id_vars=['cat_key', 'pct_change'], var_name='Année', value_name=metric_name)
            )
            long_df = long_df[long_df['Année'].isin(selected_labels)]
            order = pivot.reset_index()['categorie'].tolist()
            return long_df, order

        dossiers_long, order_cat = prepare_long_df(pivot_dossiers, 'nb_dossiers')
        poids_long, _            = prepare_long_df(pivot_poids,    'poids')

        col1, col2 = st.columns(2, gap="large")

        def plot_with_pct(df_long, y_col, title, ylabel):
            latest, prev = selected_labels
            fig = px.bar(
                df_long,
                x='cat_key',
                y=y_col,
                color='Année',
                barmode='group',
                category_orders={'cat_key': order_cat, 'Année': [latest, prev]},
                title=title
            )
            fig.update_traces(hovertemplate=f"{ylabel}=%{{y:.0f}}<extra></extra>")

            wide = (
                df_long[df_long['Année'].isin([latest, prev])]
                .pivot_table(index='cat_key', columns='Année', values=y_col, aggfunc='sum')
                .fillna(0)
            )
            max_y = wide.max().max() if not wide.empty else 0
            min_offset = max_y * 0.02

            for cat in wide.index:
                val_latest = wide.at[cat, latest] if latest in wide.columns else 0
                val_prev   = wide.at[cat, prev]   if prev   in wide.columns else np.nan
                if val_prev and not np.isnan(val_prev) and val_prev != 0:
                    pct = (val_latest - val_prev) / val_prev * 100.0
                    text = f"{pct:+.1f}%"
                    color = "green" if pct > 0 else "red"
                else:
                    text = "—"
                    color = "gray"

                y_offset = val_latest + max(val_latest * 0.05, min_offset)
                fig.add_annotation(
                    x=cat,
                    y=y_offset,
                    text=text,
                    showarrow=False,
                    font=dict(size=12, color=color, family="Arial Black"),
                    xanchor="right"
                )

            fig.update_layout(
                xaxis_title="Catégorie",
                yaxis_title=ylabel,
                bargap=0.25,
                legend_title="Année",
                height=520,
                margin=dict(t=60, b=60, l=10, r=10)
            )
            return fig

        with col1:
            st.plotly_chart(plot_with_pct(dossiers_long, 'nb_dossiers', "Évolution nb_dossiers", "Nombre de dossiers"), use_container_width=True)
            _add_or_replace_fig(f"Yearly comparaison nb_dossier", "Évolution nb_dossiers")
        with col2:
            st.plotly_chart(plot_with_pct(poids_long, 'poids', "Évolution du poids", "Poids total"), use_container_width=True)
            _add_or_replace_fig(f"Yearly comparaison poids", "Évolution du poids")
    with tab2:
        # ---------- Famille ----------
        def build_family_pivot(metric: str) -> pd.DataFrame:
            piv = (
                two_years.groupby(['famille', 'year_label'], as_index=False)[metric]
                         .sum()
                         .pivot(index='famille', columns='year_label', values=metric)
                         .fillna(0)
            )
            return piv.reindex(columns=selected_labels)

        pivot_dossiers_fam = build_family_pivot('nb_dossiers')
        pivot_poids_fam    = build_family_pivot('poids')

        def add_pct_family(pivot: pd.DataFrame) -> pd.DataFrame:
            latest, prev = selected_labels
            pivot['pct_change'] = ((pivot[latest] - pivot[prev]) / pivot[prev].replace({0: np.nan})) * 100
            return pivot

        pivot_dossiers_fam = add_pct_family(pivot_dossiers_fam)
        pivot_poids_fam    = add_pct_family(pivot_poids_fam)

        def prepare_long_df_family(pivot: pd.DataFrame, metric_name: str):
            latest, prev = selected_labels

            # Work on a reset copy once
            df_wide = pivot.reset_index()

            # Order should come from the original 'famille' column
            order = df_wide['famille'].tolist()

            # Melt on a version where 'famille' is renamed to 'fam_key'
            long_df = (
                df_wide.rename(columns={'famille': 'fam_key'})
                    .melt(id_vars=['fam_key', 'pct_change'],
                            var_name='Année',
                            value_name=metric_name)
            )

            # Keep only the two selected years
            long_df = long_df[long_df['Année'].isin([latest, prev])]

            return long_df, order


        dossiers_fam_long, order_fam = prepare_long_df_family(pivot_dossiers_fam, 'nb_dossiers')
        poids_fam_long,   _          = prepare_long_df_family(pivot_poids_fam,    'poids')

        def plot_family(df_long, y_col, title, ylabel):
            latest, prev = selected_labels
            fig = px.bar(
                df_long,
                x='fam_key',
                y=y_col,
                color='Année',
                barmode='group',
                category_orders={'fam_key': order_fam, 'Année': [latest, prev]},
                title=title
            )
            fig.update_traces(hovertemplate=f"{ylabel}=%{{y:.0f}}<extra></extra>")

            wide = (
                df_long[df_long['Année'].isin([latest, prev])]
                .pivot_table(index='fam_key', columns='Année', values=y_col, aggfunc='sum')
                .fillna(0)
            )

            max_y = wide.max().max() if not wide.empty else 0
            min_offset = max_y * 0.02

            pct_map = (
                df_long[['fam_key', 'pct_change']]
                .drop_duplicates()
                .set_index('fam_key')['pct_change']
                .to_dict()
            )

            for fam in wide.index:
                val_latest = wide.at[fam, latest] if latest in wide.columns else 0
                pct = pct_map.get(fam, np.nan)
                if np.isnan(pct):
                    text, color = "—", "gray"
                else:
                    text, color = f"{pct:+.1f}%", ("green" if pct > 0 else "red")

                y_offset = val_latest + max(val_latest * 0.05, min_offset)
                fig.add_annotation(
                    x=fam,
                    y=y_offset,
                    text=text,
                    showarrow=False,
                    font=dict(size=12, color=color, family="Arial Black"),
                    xanchor="right",
                    xshift=-20
                )

            fig.update_layout(
                xaxis_title="Famille",
                yaxis_title=ylabel,
                bargap=0.25,
                legend_title="Année",
                height=520,
                margin=dict(t=60, b=60, l=10, r=10)
            )
            return fig

        col3, col4 = st.columns(2, gap="large")
        with col3:
            st.plotly_chart(plot_family(dossiers_fam_long, 'nb_dossiers', "Évolution nb_dossiers", "Nombre de dossiers"), use_container_width=True)
            _add_or_replace_fig(f"Yearly comparaison nb_dossier par famille", "Évolution nb_dossiers")
        with col4:

            st.plotly_chart(plot_family(poids_fam_long, 'poids', "Évolution du poids", "Poids total"), use_container_width=True)
            _add_or_replace_fig(f"Yearly comparaison poids par famille", "Évolution du poids")
    # =====================================================
    # Yearly evolution: Bar + Line (EMA 2 years)
    # =====================================================
    st.subheader("Yearly evolution")

    labels_sorted_desc = (
        yearly[['year', 'year_label']]
        .drop_duplicates()
        .sort_values(['year'], ascending=[False])
        ['year_label']
        .tolist()
    )

    # Default window: all years
    default_end   = labels_sorted_desc[0]
    default_start = labels_sorted_desc[ len(labels_sorted_desc) - 1]

    col1, col2 = st.columns(2)
    with col1:
        end_year = st.selectbox("À partir de l'année", options=labels_sorted_desc, index=labels_sorted_desc.index(default_end))
    with col2:
        start_year = st.selectbox("Jusqu'à l'année", options=labels_sorted_desc, index=labels_sorted_desc.index(default_start))

    period = (start_year, end_year)
    y_start, y_end = int(period[0]), int(period[1])

    evol = (
        yearly[(yearly['year'] >= y_start) & (yearly['year'] <= y_end)]
        .groupby(['year', 'year_label'], as_index=False)
        .agg(nb_dossiers=('nb_dossiers', 'sum'),
             poids=('poids', 'sum'))
        .sort_values(['year'])
    )

    if evol.empty:
        st.info("No data found.")
    else:
        tab_a, tab_b = st.tabs(["Total", "Par famille"])

        with tab_a:
            for bar_metric, y_label in [
                ('nb_dossiers', 'Nombre de dossiers'),
                ('poids',       'Poids total'),
            ]:
                evol_metric = evol.copy()
                # EMA 2 ans (α = 2 / (2+1) ≈ 0.67)
                evol_metric['ema2'] = evol_metric[bar_metric].ewm(span=2, adjust=False).mean()

                fig = go.Figure()
                fig.add_bar(
                    x=evol_metric['year_label'],
                    y=evol_metric[bar_metric],
                    name=bar_metric,
                    hovertemplate=f"Année=%{{x}}<br>{bar_metric}=%{{y:.0f}}<extra></extra>"
                )
                fig.add_trace(go.Scatter(
                    x=evol_metric['year_label'],
                    y=evol_metric['ema2'],
                    mode='lines+markers',
                    name=f"EMA 2 ans ({bar_metric})",
                    hovertemplate=f"Année=%{{x}}<br>EMA 2 ans=%{{y:.0f}}<extra></extra>"
                ))

                fig.update_layout(
                    title=f"Évolution {bar_metric} — {period[1]} → {period[0]}",
                    xaxis_title="Année",
                    yaxis_title=y_label,
                    bargap=0.25,
                    height=520,
                    margin=dict(t=60, b=60, l=10, r=10),
                    legend_title="Label"
                )
                fig.update_xaxes(tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                _add_or_replace_fig(f"Yearly evolution {bar_metric} ", fig)
            with st.expander("EMA ?"):
                st.write("""
                La **moyenne mobile exponentielle (EMA)** sur 2 ans lisse la tendance des données
                annuelles en donnant davantage de poids à la dernière année.
                
                Formule : EMA_t = α × Valeur_t + (1 − α) × EMA_{t−1}  
                Pour **2 ans**, α = 2 / (2 + 1) ≈ **0,67** → **la dernière année pèse ~67 %**.
                """)

        with tab_b:
            fam_year = (
                yearly[(yearly['year'] >= y_start) & (yearly['year'] <= y_end)]
                .groupby(['year', 'year_label', 'famille'], as_index=False)
                .agg(nb_dossiers=('nb_dossiers', 'sum'),
                     poids=('poids', 'sum'))
                .sort_values(['year'])
            )

            fam_colors = {'direct': '#1f77b4', 'system': '#2ca02c', 'Autre': '#ff7f0e'}
            ema_colors = {'direct': '#155485', 'system': '#207820'}

            def make_year_pivot(metric):
                return (fam_year
                        .pivot_table(index='year_label', columns='famille', values=metric, aggfunc='sum')
                        .fillna(0)
                        .reset_index())

            for bar_metric, y_label in [
                ('nb_dossiers', 'Nombre de dossiers'),
                ('poids',       'Poids total'),
            ]:
                pvt = make_year_pivot(bar_metric)
                fam_cols = [c for c in ['Direct', 'System'] if c in pvt.columns]
                fam_cols += [c for c in pvt.columns if c not in fam_cols + ['year_label']]

                fig_fam_year = go.Figure()

                # Bars
                for fam in fam_cols:
                    if fam == 'year_label':
                        continue
                    fig_fam_year.add_bar(
                        x=pvt['year_label'],
                        y=pvt[fam],
                        name=fam,
                        marker_color=fam_colors.get(fam, '#cccccc'),
                        hovertemplate=f"Année=%{{x}}<br>Famille={fam}<br>{bar_metric}=%{{y:.0f}}<extra></extra>"
                    )

                # EMA 2 ans per famille
                for fam in fam_cols:
                    if fam == 'year_label':
                        continue
                    ema2 = pvt[fam].ewm(span=2, adjust=False, min_periods=1).mean()
                    fig_fam_year.add_trace(
                        go.Scatter(
                            x=pvt['year_label'],
                            y=ema2,
                            mode='lines',
                            name=f"EMA 2 ans — {fam}",
                            line=dict(width=2, color=ema_colors.get(fam, 'black')),
                            hovertemplate=f"Année=%{{x}}<br>Famille={fam}<br>EMA 2 ans=%{{y:.0f}}<extra></extra>",
                            legendgroup=fam
                        )
                    )

                fig_fam_year.update_layout(
                    title=f"{bar_metric}, année par année — {period[1]} → {period[0]}",
                    xaxis_title="Année",
                    yaxis_title=y_label,
                    barmode='group',
                    bargap=0.25,
                    height=520,
                    margin=dict(t=60, b=60, l=10, r=10),
                    legend_title="Famille"
                )
                fig_fam_year.update_xaxes(tickangle=-45)
                st.plotly_chart(fig_fam_year, use_container_width=True)
                _add_or_replace_fig(f"Yearly evolution {bar_metric} par famille ", fig_fam_year)

                        



if 'fig_store' not in st.session_state:
    st.session_state['fig_store'] = []  

with st.sidebar:
    st.markdown("### 📄 Export charts to PDF")

    titles = [item['title'] for item in st.session_state['fig_store']]

    if not titles:
        st.info("No charts registered yet. Open pages to render charts first.")
    else:
        selected_titles = st.multiselect(
            "Choose charts to include",
            options=titles,
            default=[],                     # <-- empty by default
            key="pdf_select_titles_sidebar"
        )

        btn_disabled = (len(selected_titles) == 0)

        if st.button("📥 Build PDF from selection", disabled=btn_disabled, key="pdf_build_sidebar"):
            try:
                images = []
                for t in selected_titles:
                    # find the figure by title
                    fig = next(item['fig'] for item in st.session_state['fig_store'] if item['title'] == t)

                    

                    png_bytes = pio.to_image(fig, format="png", scale=2, engine="kaleido")  # needs kaleido
                    images.append(Image.open(io.BytesIO(png_bytes)).convert("RGB"))

                if not images:
                    st.warning("Nothing selected.")
                else:
                    pdf_buf = io.BytesIO()
                    images[0].save(pdf_buf, format="PDF", save_all=True, append_images=images[1:])
                    pdf_buf.seek(0)

                    st.download_button(
                        "⬇️ Save PDF",
                        data=pdf_buf,
                        file_name=f"charts_{pd.Timestamp.now():%Y%m%d_%H%M}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key="pdf_download_sidebar"
                    )
            except Exception as e:
                st.error(f"Export failed: {e}\nCheck requirements: kaleido, Pillow.")


