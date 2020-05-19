from flask import Flask
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import requests, json
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots

app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

covid_api = "https://data.nepalcorona.info/api/v1/covid"
covid_response = requests.get(covid_api)
covid = json.loads(covid_response.text)

dis_df = pd.DataFrame(columns=["dis_code", "dis_en"])
url_dis = "https://data.nepalcorona.info/api/v1/districts"
response_dis = requests.get(url_dis)
total_dis = json.loads(response_dis.text)

for i in range(len(total_dis)):
    dis_df = dis_df.append(
        pd.Series([total_dis[i]["id"], total_dis[i]["title_en"]], index=dis_df.columns),
        ignore_index=True,
    )

mun_df = pd.DataFrame(columns=["mun_code", "type", "mun_en", "dis_code", "dis_en"])
url_mun = "https://data.nepalcorona.info/api/v1/municipals/"
response_mun = requests.get(url_mun)
total_mun = json.loads(response_mun.text)

for i in range(len(total_mun)):
    mun_df = mun_df.append(
        pd.Series(
            [
                total_mun[i]["id"],
                total_mun[i]["type"],
                total_mun[i]["title_en"],
                total_mun[i]["district"],
                "District",
            ],
            index=mun_df.columns,
        ),
        ignore_index=True,
    )

# find type and municipality name
def get_name_type(code):
    for j in range(len(total_mun)):
        if total_mun[j]["id"] == code:
            return total_mun[j]


def create_covid_df():
    covid_df = pd.DataFrame(
        columns=[
            "provience",
            "district",
            "type",
            "municipality",
            "currentstate",
            "cases",
            "gender",
            "age",
        ]
    )

    for i in range(len(covid)):
        mun_type = get_name_type(covid[i]["municipality"])
        district = dis_df[dis_df["dis_code"] == covid[i]["district"]]
        #     print(district.iloc[0]['dis_en'])
        district_name = district.iloc[0]["dis_en"]
        covid_df = covid_df.append(
            {
                "provience": covid[i]["province"],
                "district": district_name,
                "type": mun_type["type"],
                "municipality": mun_type["title_en"],
                "currentstate": covid[i]["currentState"],
                "cases": 1,
                "gender": covid[i]["gender"],
                "age": covid[i]["age"],
            },
            ignore_index=True,
        )
    return covid_df


def get_nepal_cumulative(country):
    death_df = pd.read_csv(
        "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
    )

    confirmed_df = pd.read_csv(
        "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    )
    recovered_df = pd.read_csv(
        "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"
    )
    country_df = pd.read_csv(
        "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv"
    )

    nepal_cases = confirmed_df.loc[confirmed_df["Country/Region"] == country]
    nepal_recovered = recovered_df.loc[recovered_df["Country/Region"] == country]
    nepal_death = death_df.loc[death_df["Country/Region"] == country]

    nc_data = nepal_cases.iloc[:, 4:]
    nr_data = nepal_recovered.iloc[:, 4:]
    nd_data = nepal_death.iloc[:, 4:]

    df3 = pd.concat([nc_data, nr_data, nd_data], ignore_index=True)

    return df3


def get_countires():
    confirmed_df = pd.read_csv(
        "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    )
    pdToList = list(confirmed_df["Country/Region"])
    new_line = sorted(set(pdToList), key=pdToList.index)

    return new_line


def get_summary(country):
    country_df = pd.read_csv(
        "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv"
    )
    cou = country_df.loc[country_df["Country_Region"] == country]
    summary = cou.iloc[0].tolist()
    return summary


def country_wise_summary(country):
    country_df = pd.read_csv(
        "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv"
    )
    detail = country_df.loc[country_df["Country_Region"] == country]

    return detail


def nepal_stats():
    url = "https://nepalcorona.info/api/v1/data/nepal"
    response = requests.get(url)
    todos = json.loads(response.text)

    return todos


#  provience wise data onf test quaratine , rdt , appendnd isolatio
def provience_test_all(provience):
    rdt = pd.read_csv(
        "https://raw.githubusercontent.com/mepesh/python-dashboard-covid19/master/site-report-mohp%20-%20rdttest.csv"
    )
    isolation = pd.read_csv(
        "https://raw.githubusercontent.com/mepesh/python-dashboard-covid19/master/site-report-mohp%20-%20isolation.csv"
    )
    quaratine = pd.read_csv(
        "https://raw.githubusercontent.com/mepesh/python-dashboard-covid19/master/site-report-mohp%20-%20quarantine.csv"
    )

    r = rdt.iloc[:, -1]
    i = isolation.iloc[:, -1]
    q = quaratine.iloc[:, -1]
    if provience == 8:
        ret = [r.sum(), q.sum(), i.sum(), isolation.columns[-1]]
    else:
        ret = [r[provience], q[provience], i[provience], isolation.columns[-1]]

    return ret


# provience wise summaray of test, cases, deaths and recovered.
def nepal_allprovience_stats(code):
    rdt = pd.read_csv(
        "https://raw.githubusercontent.com/mepesh/python-dashboard-covid19/master/site-report-mohp%20-%20rdttest.csv"
    )
    isolation = pd.read_csv(
        "https://raw.githubusercontent.com/mepesh/python-dashboard-covid19/master/site-report-mohp%20-%20isolation.csv"
    )
    quaratine = pd.read_csv(
        "https://raw.githubusercontent.com/mepesh/python-dashboard-covid19/master/site-report-mohp%20-%20quarantine.csv"
    )

    r = rdt.iloc[:, -1]
    i = isolation.iloc[:, -1]
    q = quaratine.iloc[:, -1]

    df = pd.concat([r, i, q], axis=1, sort=False)

    return df


def get_ac_re_type(code):
    covid_df = create_covid_df()
    covid_df_active = pd.DataFrame(columns=covid_df.columns)
    covid_df_recovered = pd.DataFrame(columns=covid_df.columns)
    for i in range(len(covid_df)):
        data = pd.Series(covid_df.iloc[i, :])
        if covid_df.iloc[i]["currentstate"] == "active":
            covid_df_active = covid_df_active.append(data, ignore_index=True)
        else:
            covid_df_recovered = covid_df_recovered.append(data, ignore_index=True)

    covid_df_active["Active Cases"] = covid_df_active.groupby([code])[code].transform(
        "count"
    )
    covid_df_active = covid_df_active.drop_duplicates(code)
    #     print(covid_df_active[[code,'Active Cases']])

    covid_df_recovered["Recovered Cases"] = covid_df_recovered.groupby([code])[
        code
    ].transform("count")
    covid_df_recovered = covid_df_recovered.drop_duplicates(code)
    #     print(covid_df_recovered[[code, "Recovered Cases"]])

    covid_ac_re = pd.DataFrame()
    #     print(covid_df_active)
    covid_df_active_mod = covid_df_active[[code, "Active Cases"]]
    covid_df_recovered_mod = covid_df_recovered[[code, "Recovered Cases"]]

    covid_ac_re = covid_df_active_mod.merge(
        covid_df_recovered_mod, on=[code], how="outer"
    )

    covid_ac_re["Active Cases"] = covid_ac_re["Active Cases"].fillna(0)
    covid_ac_re["Recovered Cases"] = covid_ac_re["Recovered Cases"].fillna(0)

    return covid_ac_re.sort_values(by=["Active Cases"])


def provience_ac_re(provience):
    df = create_covid_df()
    abr = df[(df["provience"] == provience) & (df["currentstate"] == "recovered")]
    aba = df[(df["provience"] == provience) & (df["currentstate"] == "active")]
    acre = [len(aba), len(abr)]
    return acre


def provience_ma_fe(provience):
    df = create_covid_df()
    abm = df[(df["provience"] == provience) & (df["gender"] == "male")]
    abf = df[(df["provience"] == provience) & (df["gender"] == "female")]
    mafe = [len(abm), len(abf)]
    return mafe


def agedistribution_list():
    x = create_covid_df()
    x = x[~x.astype(str).eq("None").any(1)]

    x1 = x[(x["age"] > 0) & (x["age"] <= 10)]
    x2 = x[(x["age"] > 10) & (x["age"] <= 20)]
    x3 = x[(x["age"] > 20) & (x["age"] <= 30)]
    x4 = x[(x["age"] > 30) & (x["age"] <= 40)]
    x5 = x[(x["age"] > 40) & (x["age"] <= 50)]
    x6 = x[(x["age"] > 50) & (x["age"] <= 60)]
    x7 = x[(x["age"] > 60) & (x["age"] <= 70)]
    x8 = x[(x["age"] > 70) & (x["age"] <= 80)]
    x9 = x[(x["age"] > 80) & (x["age"] <= 90)]
    x10 = x[(x["age"] > 90) & (x["age"] <= 100)]
    ydata = [
        len(x1),
        len(x2),
        len(x3),
        len(x4),
        len(x5),
        len(x6),
        len(x7),
        len(x8),
        len(x9),
        len(x10),
    ]

    # def agegroup_barchart():

    age = [
        "0-10",
        "11-20",
        "21-30",
        "31-40",
        "41-50",
        "51-60",
        "61-70",
        "71-80",
        "81-90",
        "91-100",
    ]
    #     fig = go.Figure(data=[go.Bar(name="Total", x=age, y=ydata,labels={'age':'AGE', 'ydata':'Total Cases'}),])
    fig = px.bar(x=age, y=ydata)

    fig.update_layout(
        showlegend=False,
        paper_bgcolor="rgb(255,255,255)",
        plot_bgcolor="rgb(255,255,255)",
        margin=dict(l=0, r=0, b=30, t=20, pad=0.8),
        barmode="stack",
        yaxis=dict(title="Total(Infections)", titlefont_size=14, tickfont_size=12,),
        xaxis=dict(title="Age Group", titlefont_size=14, tickfont_size=12,),
    )
    return fig


# In[5]:


def piechart_test(provience):
    labels = ["1st", "2nd", "3rd", "4th", "5th"]
    night_colors = [
        "rgb(56, 75, 126)",
        "rgb(18, 36, 37)",
        "rgb(34, 53, 101)",
        "rgb(36, 55, 57)",
        "rgb(6, 4, 4)",
    ]
    sunflowers_colors = [
        "rgb(177, 127, 38)",
        "rgb(205, 152, 36)",
        "rgb(99, 79, 37)",
        "rgb(129, 180, 179)",
        "rgb(124, 103, 37)",
    ]
    irises_colors = [
        "rgb(33, 75, 99)",
        "rgb(79, 129, 102)",
        "rgb(151, 179, 100)",
        "rgb(175, 49, 35)",
        "rgb(36, 73, 147)",
    ]
    cafe_colors = [
        "rgb(146, 123, 21)",
        "rgb(177, 180, 34)",
        "rgb(206, 206, 40)",
        "rgb(175, 51, 21)",
        "rgb(35, 36, 21)",
    ]
    specs = [
        [{"type": "domain"}, {"type": "domain"}],
        [{"type": "domain"}, {"type": "domain"}],
    ]
    fig = make_subplots(rows=2, cols=2, specs=specs,)

    # Define pie charts
    fig.add_trace(
        go.Pie(
            labels=["RDT", "Quarantine", "ISolation"],
            values=provience_test_all(provience),
            name="Total Test",
            marker_colors=night_colors,
        ),
        1,
        1,
    )
    fig.add_trace(
        go.Pie(
            labels=["Cases", "Recovered"],
            values=provience_ac_re(provience + 1),
            name="Infections",
            marker_colors=cafe_colors,
        ),
        1,
        2,
    )
    fig.add_trace(
        go.Pie(
            labels=["Male", "Female"],
            values=provience_ma_fe(provience + 1),
            name="Gender",
            marker_colors=irises_colors,
        ),
        2,
        1,
    )
    fig.add_trace(
        go.Pie(
            labels=labels,
            values=provience_ac_re(provience + 1),
            name="Other Graph",
            marker_colors=cafe_colors,
        ),
        2,
        2,
    )

    fig.update_traces(hoverinfo="label+percent+value+name", textinfo="label+percent")
    fig.update(layout_showlegend=False,),
    fig.update_layout(
        autosize=True, margin=dict(l=0, r=0, b=0, t=35, pad=0),
    )

    fig = go.Figure(fig)
    return fig


# In[6]:


dasd = get_nepal_cumulative("Nepal")

with open("nepal.geojson") as f:
    nepal_districts_geo = json.load(f)

with open("state-1-districts.geojson") as f:
    state1 = json.load(f)

with open("nepal-municipalities.geojson") as f:
    geomuni = json.load(f)


def card_content(code, value):
    card_content = [
        dbc.CardHeader(code),
        dbc.CardBody([html.H4(value, className="card-title"),]),
    ]
    return card_content


def district_map():
    df1 = get_ac_re_type("district")
    a = dis_df.iloc[:, 1].tolist()
    district_list = a
    zeros = pd.DataFrame(
        0,
        index=np.arange(len(district_list)),
        columns=["Active Cases", "Recovered Cases"],
    )

    zeros["district"] = district_list
    y = (
        pd.concat([df1, zeros], sort=False)
        .drop_duplicates(["district"])
        .reset_index(drop=True)
    )
    y = y.rename(
        columns={"Active Cases": "activecases", "Recovered Cases": "recoveredcases"}
    )
    y["district"] = y["district"].str.upper()
    figg2 = go.Figure(
        go.Choroplethmapbox(
            featureidkey="properties.DISTRICT",
            geojson=nepal_districts_geo,
            locations=y.district,
            z=(y.activecases + y.recoveredcases) * 10,
            zauto=True,
            zmin=5,
            zmax=16,
            colorscale=[
                "#f7fbff",
                "#d2e3f3",
                "#c6dbef",
                "#b3d2e9",
                "#85bcdb",
                "#6baed6",
                "#4292c6",
                "#1361a9",
                "#08519c",
                "#08306b",
            ],
            reversescale=False,
            marker_opacity=0.8,
            marker_line_width=1,
            customdata=np.vstack((y.district, y.activecases, y.recoveredcases)).T,
            hovertemplate="<b>%{customdata[0]}</b><br><br>"
            + "Total Cases：%{customdata[1]}<br>"
            + "Recovered：%{customdata[2]}<br>"
            + "<extra></extra>",
            showscale=True,
        )
    )
    figg2.update_layout(
        margin=dict(l=0, r=0, b=0, t=0, pad=0.8),
        mapbox_style="carto-positron",
        mapbox_zoom=5.5,
        mapbox_center={"lat": 28.021106723604202, "lon": 83.80321443774628},
    )

    return figg2


# figg2.show()
def municipality_map():
    # zeros municiple
    df1 = get_ac_re_type("municipality")
    print(df1)
    a = mun_df.iloc[:, 2].tolist()
    municipality_list = a
    print(municipality_list)
    zeros = pd.DataFrame(
        0,
        index=np.arange(len(municipality_list)),
        columns=["Active Cases", "Recovered Cases"],
    )

    zeros["municipality"] = municipality_list
    y = (
        pd.concat([df1, zeros], sort=False)
        .drop_duplicates(["municipality"])
        .reset_index(drop=True)
    )
    y = y.rename(
        columns={"Active Cases": "activecases", "Recovered Cases": "recoveredcases"}
    )
    print(y)
    municipals = go.Figure(
        go.Choroplethmapbox(
            featureidkey="properties.NAME",
            geojson=geomuni,
            locations=y.municipality,
            z=y.activecases + y.recoveredcases,
            zauto=True,
            zmin=6,
            zmax=16,
            colorscale=[
                "#f7fbff",
                "#d2e3f3",
                "#c6dbef",
                "#b3d2e9",
                "#85bcdb",
                "#6baed6",
                "#4292c6",
                "#1361a9",
                "#08519c",
                "#08306b",
            ],
            reversescale=False,
            marker_opacity=0.8,
            marker_line_width=1,
            customdata=np.vstack((y.municipality, y.activecases, y.recoveredcases)).T,
            hovertemplate="<b>%{customdata[0]}</b><br><br>"
            + "Total Cases：%{customdata[1]}<br>"
            + "Recovered：%{customdata[2]}<br>"
            + "<extra></extra>",
            showscale=True,
        )
    )
    municipals.update_layout(
        margin=dict(l=0, r=0, b=0, t=0, pad=0.8),
        mapbox_style="carto-positron",
        mapbox_zoom=6,
        mapbox_center={"lat": 28.277433, "lon": 83.581612},
    )

    return municipals

    # def get_testfigure():

@app.route("/")
def hello():
	nav_item = dbc.NavItem(dbc.NavLink("CoronaChatbot", href="http://m.me/coronanepal2020", target="_blank"))
	nav_item1 = dbc.NavItem(dbc.NavLink("Clap me in Medium", href="https://medium.com/@dipesh.pandey42", target="_blank"))

	default = dbc.NavbarSimple(
	    children=[nav_item, nav_item1],
	    brand="Default",
	    brand_href="#",
	    sticky="top",
	    className="mb-5",
	)

	logo = dbc.Navbar(
	    dbc.Container(
	        [
	            html.A(
	                # Use row and col to control vertical alignment of logo / brand
	                dbc.Row(
	                    [
	                        dbc.Col(html.Img(src='http://exceltech.com.np/wp-content/uploads/2020/05/Chat-Bot-Messenger-e1588504947299.png')),
	                        dbc.Col(dbc.NavbarBrand("ChatBot | Covid 19 Dashboard Tracking Nepal", className="ml-3")),
	                    ],
	                    align="left",
	                    no_gutters=True,
	                ),href="https://m.me/coronanepal2020",target="_blank", 
	            ),
	            dbc.NavbarToggler(id="navbar-toggler2"),
	            dbc.Collapse(
	                dbc.Nav(
	                    [nav_item, nav_item1], className="ml-auto", navbar=True
	                ),
	                id="navbar-collapse2",
	                navbar=True,
	            ),
	        ]
	    ),
	    color="light",
	    dark=False,
	    className="mb-5",
	)


# In[8]:


	co = nepal_stats()
	app.layout = dbc.Container(
	    [
	        html.Div([logo]),
	        dbc.Row(
	            [
	                dbc.Col(
	                    [
	                        dcc.Dropdown(
	                            id="in-11-dropdown",
	                            options=[
	                                {"label": name, "value": name}
	                                for name in get_countires()
	                            ],
	                            value="China",
	                        ),
	                        html.Hr(),
	                        dbc.Card(
	                            [
	                                dbc.CardBody(
	                                    [
	                                        html.H2(
	                                            id="out-11-1-dropdown",
	                                            className="card-title",
	                                        ),
	                                        html.P("Total Cases", className="card-text",),
	                                    ]
	                                ),
	                            ],
	                            color="primary",
	                            outline=True,
	                        ),
	                        html.Hr(),
	                        dbc.Card(
	                            [
	                                dbc.CardBody(
	                                    [
	                                        html.H2(
	                                            id="out-11-2-dropdown",
	                                            className="card-title",
	                                        ),
	                                        html.P("Recovered ", className="card-text",),
	                                    ]
	                                ),
	                            ],
	                            color="success",
	                            outline=True,
	                        ),
	                        html.Hr(),
	                        dbc.Card(
	                            [
	                                dbc.CardBody(
	                                    [
	                                        html.H2(
	                                            id="out-11-3-dropdown",
	                                            className="card-title",
	                                        ),
	                                        html.P("Total Dead ", className="card-text",),
	                                    ]
	                                ),
	                            ],
	                            color="danger",
	                            outline=True,
	                        ),
	                    ],
	                    md=2,
	                ),
	                dbc.Col([dcc.Graph(figure=district_map())], md=8,),
	                dbc.Col(
	                    [
	                        dcc.Dropdown(
	                            id="in-13-dropdown",
	                            options=[
	                                {"label": "Provience ", "value": 8},
	                                {"label": "Provience 1", "value": 0},
	                                {"label": "Provience 2", "value": 1},
	                                {"label": "Provience 3", "value": 2},
	                                {"label": "Provience 4", "value": 3},
	                                {"label": "Provience 5", "value": 4},
	                                {"label": "Provience 6", "value": 5},
	                                {"label": "Provience 7", "value": 6},
	                            ],
	                            value=8,
	                        ),
	                        html.Hr(),
	                        html.Div(
	                            [
	                                html.H5("RDT Tested"),
	                                dbc.Progress(
	                                    html.H2(id="out-13-1-dropdown"),
	                                    value=100,
	                                    style={"height": "65px"},
	                                    className="mb-3",
	                                    striped=True,
	                                    color="success",
	                                ),
	                                html.H5("Quarantine"),
	                                dbc.Progress(
	                                    html.H2(id="out-13-2-dropdown"),
	                                    value=100,
	                                    style={"height": "65px"},
	                                    striped=True,
	                                    className="mb-3",
	                                    color="warning",
	                                ),
	                                html.H5("In Isolation"),
	                                dbc.Progress(
	                                    html.H2(id="out-13-3-dropdown"),
	                                    striped=True,
	                                    value=100,
	                                    style={"height": "65px"},
	                                    className="mb-3",
	                                    color="info",
	                                ),
	                                html.I(id="out-13-0-dropdown"),
	                            ],
	                        ),
	                    ],
	                    md=2,
	                ),
	            ],
	            align="center",
	        ),
	        # row-2-data-display-card-viewL
	        dbc.Row(
	            [
	                dbc.Col(md=2),
	                dbc.Col(
	                    dbc.Card(
	                        card_content("Positive", co["tested_positive"]),
	                        color="info",
	                        outline=True,
	                    ),
	                    md=2,
	                ),
	                dbc.Col(
	                    dbc.Card(
	                        card_content("Tested Total", co["tested_total"]),
	                        color="secondary",
	                        outline=True,
	                    ),
	                    md=2,
	                ),
	                dbc.Col(
	                    dbc.Card(
	                        card_content("Recovered", co["recovered"]),
	                        color="success",
	                        outline=True,
	                    ),
	                    md=2,
	                ),
	                dbc.Col(
	                    dbc.Card(
	                        card_content("Deaths", co["deaths"]),
	                        color="danger",
	                        outline=True,
	                    ),
	                    md=2,
	                ),
	                dbc.Col(md=2),
	            ],
	            align="center",
	        ),
	        # row 2 -controls
	        dbc.Row(
	            [
	                dbc.Col(md=3),
	                dbc.Col(
	                    [
	                        html.Hr(),
	                        dcc.Dropdown(
	                            id="in-32-dropdown",
	                            options=[
	                                {"label": name, "value": name}
	                                for name in get_countires()
	                            ],
	                            value="Nepal",
	                        ),
	                    ],
	                    md=3,
	                ),
	                dbc.Col(
	                    [
	                        html.Hr(),
	                        dcc.Dropdown(
	                            id="in-33-dropdown",
	                            options=[
	                                {"label": "Daily", "value": "daily"},
	                                {"label": "Total", "value": "total"},
	                            ],
	                            value="total",
	                        ),
	                    ],
	                    md=3,
	                ),
	                dbc.Col(md=3),
	            ],
	            align="bottom",
	        ),
	        # row 3 - histtogram
	        dbc.Row(
	            [dbc.Col(dcc.Graph(id="out-33-dropdown", figure={}), md=12),],
	            align="center",
	        ),
	        # row 4 -controls - only
	        dbc.Row(
	            [
	                dbc.Col(md=4),
	                dbc.Col(
	                    [
	                        dcc.Dropdown(
	                            id="in-42-dropdown",
	                            options=[
	                                {"label": "RDT Test", "value": "rdt"},
	                                {"label": "ISolation", "value": "isolation"},
	                                {"label": "Quarantine", "value": "quarantine"},
	                            ],
	                            value="rdt",
	                        ),
	                    ],
	                    md=2,
	                ),
	                dbc.Col(md=2),
	                dbc.Col(
	                    [
	                        dcc.Dropdown(
	                            id="in-41-dropdown",
	                            options=[
	                                {"label": "District Bar Chart", "value": "district"},
	                                {"label": "Provience Bar Chart", "value": "provience"},
	                                {
	                                    "label": "Municipality Bar Chart",
	                                    "value": "municipality",
	                                },
	                            ],
	                            value="district",
	                        ),
	                    ],
	                    md=3,
	                ),
	                dbc.Col(md=1),
	            ],
	            align="center",
	        ),
	        # row -5 display bar charts and test pie chart
	        dbc.Row(
	            [
	                dbc.Col(
	                    [
	                        html.H5("Infectious Distribution By Age Group"),
	                        dcc.Graph(figure=agedistribution_list()),
	                    ],
	                    md=5,
	                ),
	                dbc.Col([dcc.Graph(id="out-42-dropdown", figure={}),], md=3,),
	                dbc.Col(
	                    [
	                        dcc.Graph(
	                            id="out-41-dropdown",
	                            config={"modeBarButtonsToRemove": ["pan2d", "lasso2d"]},
	                            figure={},
	                        )
	                    ],
	                    md=4,
	                ),
	            ],
	            align="center",
	        ),
	        # row 6 -controls - provience map
	        dbc.Row(
	            [
	                dbc.Col(md=2),
	                dbc.Col(
	                    [
	                        dcc.Dropdown(
	                            id="in-61-dropdown",
	                            options=[
	                                {"label": "Provience 1", "value": 0},
	                                {"label": "Provience 2", "value": 1},
	                                {"label": "Provience 3", "value": 2},
	                                {"label": "Provience 4", "value": 3},
	                                {"label": "Provience 5", "value": 4},
	                                {"label": "Provience 6", "value": 5},
	                                {"label": "Provience 7", "value": 6},
	                            ],
	                            value=1,
	                        ),
	                        html.Hr(),
	                    ],
	                    md=2,
	                ),
	                dbc.Col(md=3),
	                dbc.Col([html.H5(id="out-60-dropdown"), html.Hr(),], md=3),
	                dbc.Col(md=2),
	            ],
	            align="center",
	        ),
	        # row 7- provience map - pie charts
	        dbc.Row(
	            [
	                dbc.Col([dcc.Graph(id="out-61-dropdown", figure={})], md=6),
	                dbc.Col([dcc.Graph(id="out-62-dropdown", figure={})], md=6),
	            ],
	            align="center",
	        ),
	        dbc.Row(
	            [
	                dbc.Col(md=4),
	                dbc.Col(
	                    [
	                        html.A(
	                            [
	                                html.Img(
	                                    src="http://exceltech.com.np/wp-content/uploads/2020/03/message-us1.png",
	                                    style={
	                                        "float": "right",
	                                        "position": "relative",
	                                        "padding-top": 50,
	                                    },
	                                )
	                            ],
	                            href="https://m.me/coronanepal2020",
	                            target="_blank",
	                        ),
	                        html.Hr(),
	                        html.H5(
	                            "Nepali News, Stats, Facts and Quizzles !"
	                        ),
	                        html.I("Everything you need to know as it happens right into your messenger"),
	                    ],
	                    md=4,
	                ),
	                dbc.Col(md=4),
	            ],
	        ),
	    ],
	    fluid=True,
	)


# In[9]:


# functions

# 11-dropdown-world-data-function
	@app.callback(
	    [
	        dash.dependencies.Output("out-11-1-dropdown", "children"),
	        dash.dependencies.Output("out-11-2-dropdown", "children"),
	        dash.dependencies.Output("out-11-3-dropdown", "children"),
	    ],
	    [dash.dependencies.Input("in-11-dropdown", "value")],
	)
	def update_output(value):
	    co = country_wise_summary(value)

	    return (co.iloc[0]["Confirmed"], co.iloc[0]["Recovered"], co.iloc[0]["Deaths"])


	# row 13- provience nepal functions
	@app.callback(
	    [
	        dash.dependencies.Output("out-13-1-dropdown", "children"),
	        dash.dependencies.Output("out-13-2-dropdown", "children"),
	        dash.dependencies.Output("out-13-3-dropdown", "children"),
	        dash.dependencies.Output("out-13-0-dropdown", "children"),
	    ],
	    [dash.dependencies.Input("in-13-dropdown", "value")],
	)
	def update_output(value):
	    a = provience_test_all(value)
	    date = "Updated on: " + a[3] + " ."
	    return (a[0], a[1], a[2], date)


	# row- barchart -nepal
	@app.callback(
	    dash.dependencies.Output("out-33-dropdown", "figure"),
	    [dash.dependencies.Input("in-32-dropdown", "value")],
	)
	def update_output(value):
	    df = get_nepal_cumulative(value)
	    fig = go.Figure()
	    fig.add_trace(
	        go.Bar(
	            x=df.columns.values.tolist(),
	            hovertemplate="<b>"
	            + value.upper()
	            + "</b></br></br>"
	            + "DATE: %{x}<br>"
	            + "<i><b>%{y}</b> Total Cases</i><br>"
	            + "Source: CSSEGIS</br>",
	            y=df.iloc[0].values.tolist(),
	            marker_color="#330C73",
	            opacity=0.75,
	        )
	    )
	    fig.add_trace(
	        go.Bar(
	            x=df.columns.values.tolist(),
	            hovertemplate="<b>"
	            + value.upper()
	            + "</b></br></br>"
	            + "DATE: %{x}</br>"
	            + "<i><b> %{y}</b> Recovered</i></br>"
	            + "Source: CSSEGIS</br>",
	            y=df.iloc[1].values.tolist(),
	            marker_color="#EB89B5",
	            opacity=0.75,
	        )
	    )
	    fig.add_trace(
	        go.Bar(
	            x=df.columns.values.tolist(),
	            hovertemplate="<b>"
	            + value.upper()
	            + "</b></br></br>"
	            + "DATE: %{x}</br>"
	            + "<i><b> %{y}</b> Deaths</i><br>"
	            + "Source: CSSEGIS</br>",
	            y=df.iloc[2].values.tolist(),
	            marker_color="#ff0d00",
	            opacity=0.75,
	        )
	    )

	    fig.update_layout(
	        title_text="Cumulative Bar Covid 19 Victims " + value.upper() + " .",
	        showlegend=False,
	        barmode="stack",
	        bargap=0.1,
	        paper_bgcolor="rgb(255,255,255)",
	        plot_bgcolor="rgb(255,255,255)",
	        margin=dict(l=20, r=20, b=20, t=40, pad=0.8),
	        yaxis=dict(
	            ticklen=20,
	            zeroline=False,
	            showticklabels=True,
	            showgrid=False,
	            showline=False,
	            titlefont=dict(family="Gilroy", size=20),
	        ),
	    )

	    return fig


# row-51- display- district--provience municipals
	@app.callback(
	    dash.dependencies.Output("out-41-dropdown", "figure"),
	    [dash.dependencies.Input("in-41-dropdown", "value")],
	)
	def update_output(value):
	    ap = get_ac_re_type(value)
	    months = ap.iloc[:, 0].tolist()
	    fig = go.Figure()
	    fig.add_trace(
	        go.Bar(
	            y=months,
	            x=ap.iloc[:, 1].tolist(),
	            name="Active Cases",
	            orientation="h",
	            marker_color="#330C73",
	            opacity=0.75,
	        )
	    )
	    fig.add_trace(
	        go.Bar(
	            y=months,
	            x=ap.iloc[:, 2].tolist(),
	            name="Recovered",
	            orientation="h",
	            marker_color="#EB89B5",
	            opacity=0.75,
	        )
	    )
	    fig.update_layout(
	        title_text="Corona Virus Victims Distribution by " + value.upper() + " .",
	        showlegend=False,
	        barmode="group",
	        paper_bgcolor="rgb(255,255,255)",
	        plot_bgcolor="rgb(255,255,255)",
	        margin=dict(l=20, r=20, b=30, t=55, pad=0.8),
	        yaxis=dict(
	            ticklen=20,
	            zeroline=False,
	            showticklabels=True,
	            showgrid=False,
	            showline=False,
	            titlefont=dict(family="Gilroy", size=20),
	        ),
	    )

	    return fig


	# row -52- pie charts
	@app.callback(Output("out-42-dropdown", "figure"), [Input("in-42-dropdown", "value")])
	def update_layout(value):
	    a = nepal_allprovience_stats(value)
	    if value == "rdt":
	        values = a.iloc[:, 0].tolist()
	    elif value == "isolation":
	        values = a.iloc[:, 1].tolist()
	    elif value == "quarantine":
	        values = a.iloc[:, 2].tolist()

	    labels = [
	        "Provience 1",
	        "Provience 2",
	        "Provience 3",
	        "Provience 4",
	        "Provience 5",
	        "Provience 6",
	        "Provience 7",
	    ]
	    fig = go.Figure(
	        data=[go.Pie(labels=labels, textinfo="label+percent", values=values, hole=0.4)]
	    )
	    fig.update_layout(
	        margin=dict(l=0, r=0, b=30, t=55, pad=0.8),
	        showlegend=False,
	        title_text="Provience  Data For " + value + " Test .",
	        annotations=[dict(text=value, font_size=24, showarrow=False)],
	    )
	    return fig


	# row -71- provience map
	@app.callback(
	    [
	        Output("out-61-dropdown", "figure"),
	        Output("out-62-dropdown", "figure"),
	        Output("out-60-dropdown", "children"),
	    ],
	    [Input("in-61-dropdown", "value")],
	)
	def update_layout(value):
	    midpointlatlong = {
	        0: [87.06789158534278, 27.181814756262924, 7],
	        1: [85.56531220869543, 26.986159229293165, 7],
	        2: [85.44078719752078, 27.67769038360649, 7],
	        3: [83.9973481375455, 28.35494649649246, 7],
	        4: [82.41038219013222, 27.9641441834616, 6.5],
	        5: [83.05217011480939, 29.170655636885265, 6.5],
	        6: [81.29711239175032, 29.11256817441899, 7],
	    }
	    state = [
	        "state-1-districts.geojson",
	        "state-2-districts.geojson",
	        "state-3-districts.geojson",
	        "state-4-districts.geojson",
	        "state-5-districts.geojson",
	        "state-6-districts.geojson",
	        "state-7-districts.geojson",
	    ]
	    with open(state[value]) as f:
	        state = json.load(f)

	    df1 = get_ac_re_type("district")
	    a = dis_df.iloc[:, 1].tolist()
	    district_list = a
	    zeros = pd.DataFrame(
	        0,
	        index=np.arange(len(district_list)),
	        columns=["Active Cases", "Recovered Cases"],
	    )

	    zeros["district"] = district_list
	    y = (
	        pd.concat([df1, zeros], sort=False)
	        .drop_duplicates(["district"])
	        .reset_index(drop=True)
	    )
	    y = y.rename(
	        columns={"Active Cases": "activecases", "Recovered Cases": "recoveredcases"}
	    )
	    y["district"] = y["district"].str.upper()

	    figg2 = go.Figure(
	        go.Choroplethmapbox(
	            featureidkey="properties.DISTRICT",
	            geojson=state,
	            locations=y.district,
	            z=(y.activecases + y.recoveredcases) * 10,
	            zmin=6,
	            zmax=16,
	            colorscale=[
	                "#f7fbff",
	                "#d2e3f3",
	                "#c6dbef",
	                "#b3d2e9",
	                "#85bcdb",
	                "#6baed6",
	                "#4292c6",
	                "#1361a9",
	                "#08519c",
	                "#08306b",
	            ],
	            reversescale=False,
	            marker_opacity=0.8,
	            marker_line_width=0.8,
	            customdata=np.vstack((y.district, y.activecases, y.recoveredcases)).T,
	            hovertemplate="<b>%{customdata[0]}</b><br><br>"
	            + "Total Cases：%{customdata[1]}<br>"
	            + "Recovered：%{customdata[2]}<br>"
	            + "<extra></extra>",
	            showscale=False,
	        )
	    )
	    figg2.update_layout(
	        margin=dict(l=0, r=0, b=0, t=0, pad=6.5),
	        mapbox_style="carto-positron",
	        mapbox_zoom=midpointlatlong[value][2],
	        mapbox_center={
	            "lat": midpointlatlong[value][1],
	            "lon": midpointlatlong[value][0],
	        },
	    )

	    piechart = piechart_test(value)
	    #     piechart.show()
	    st = "Pie Charts for Provience " + str(value + 1) + "."

	    return (figg2, piechart, st)


if __name__ == "__main__":
    app.run_server(host='0.0.0.0')