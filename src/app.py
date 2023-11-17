"""
Author: Tim Hermans (tim-hermans@hotmail.com).
"""
from collections import defaultdict

import numpy as np
from dash import Dash, html, dcc, Output, Input, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash_bootstrap_templates import load_figure_template
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import get_demand, get_solar_power, segment_1d, EnergySimulator, convert_time_unit, convert_power_unit, \
    SIGNAL_LABELS, convert_bat_charge

# Figure template.
load_figure_template('simplex')

app_title = 'Energie dashboard'

# Default (starting) values.
DEFAULT_VALUES = defaultdict(lambda: None)
DEFAULT_VALUES.update(
    {
        'battery_capacity': 48,  # [kWh]
        'battery_charge_power': 20,  # [kW]
        'battery_discharge_power': 10,  # [kW]
        'battery_efficiency': 95,  # [%]
        'battery_initial_load': 0,  # [%]
        'day_ratio_demand': 4,  # [-]
        'day_start_hour': 6,
        'day_end_hour': 18,
        'dt': 1,  # [h]
        'max_net_power': 31.7,  # [kW]
        'season_ratio_demand': 2,  # [-]
        'season_ratio_solar': 4,  # [-]
        'solar_panels_amount': 240,  # [#]
        'solar_wp': 365,  # [W]
        'total_energy_demand': 250e3,  # [kWh]
    }
)

app = Dash(__name__)
server = app.server

app.title = app_title

options_col = dbc.Stack(
    [
        # Demand.
        dbc.Row([
            dbc.Card(html.Div(
                [
                    dbc.Label('Verbruik', style={'font-weight': 'bold'}),
                    dbc.Row([
                        dbc.Col(dbc.Label('Totaal verbruik [kWh]:')),
                        dbc.Col(dcc.Input(
                            id="input-total-energy-demand",
                            type='number',
                            placeholder="[kWh]",
                            value=DEFAULT_VALUES['total_energy_demand'],
                            min=1,
                            # step=1,
                            debounce=True,
                            style={'width': '80px'},
                        ), width='auto'),
                    ], justify='between', align='center'),
                    dbc.Row([
                        dbc.Col(html.Div('Seizoen ratio [-]:')),
                        dbc.Col(dcc.Input(
                            id="input-season-ratio-demand",
                            type='number',
                            placeholder="[-]",
                            value=DEFAULT_VALUES['season_ratio_demand'],
                            min=1,
                            # step=0.5,
                            debounce=True,
                            style={'width': '80px'},
                        ), width='auto'),
                    ], justify='between', align='center'),
                    dbc.Row([
                        dbc.Col(html.Div('Dag ratio [-]:')),
                        dbc.Col(dcc.Input(
                            id="input-day-ratio-demand",
                            type='number',
                            placeholder="[-]",
                            value=DEFAULT_VALUES['day_ratio_demand'],
                            min=1,
                            # step=0.5,
                            debounce=True,
                            style={'width': '80px'},
                        ), width='auto'),
                    ], justify='between', align='center'),
                ]),
            ),
        ]),
        # Solar.
        dbc.Row([
            dbc.Card(html.Div(
                [
                    dbc.Label('Zon', style={'font-weight': 'bold'}),
                    dbc.Row([
                        dbc.Col(dbc.Label('Piekvermogen [Wp]:')),
                        dbc.Col(dcc.Input(
                            id="input-solar-wp",
                            type='number',
                            placeholder="[W]",
                            value=DEFAULT_VALUES['solar_wp'],
                            min=1,
                            # step=1,
                            debounce=True,
                            style={'width': '80px'},
                        ), width='auto'),
                    ], justify='between', align='center'),
                    dbc.Row([
                        dbc.Col(dbc.Label('Aanal panelen [#]:')),
                        dbc.Col(dcc.Input(
                            id="input-solar-panels-amount",
                            type='number',
                            placeholder="[#]",
                            value=DEFAULT_VALUES['solar_panels_amount'],
                            min=1,
                            # step=1,
                            debounce=True,
                            style={'width': '80px'},
                        ), width='auto'),
                    ], justify='between', align='center'),
                    dbc.Row([
                        dbc.Col(html.Div('Seizoen ratio [-]:')),
                        dbc.Col(dcc.Input(
                            id="input-season-ratio-solar",
                            type='number',
                            placeholder="[-]",
                            value=DEFAULT_VALUES['season_ratio_solar'],
                            min=1,
                            # step=0.5,
                            debounce=True,
                            style={'width': '80px'},
                        ), width='auto'),
                    ], justify='between', align='center'),
                ]),
            ),
        ]),
        # Battery.
        dbc.Row([
            dbc.Card(html.Div(
                [
                    dbc.Label('Batterij', style={'font-weight': 'bold'}),
                    dbc.Row([
                        dbc.Col(dbc.Label('Capaciteit [kWh]:')),
                        dbc.Col(dcc.Input(
                            id="input-battery-capacity",
                            type='number',
                            placeholder="[kWh]",
                            value=DEFAULT_VALUES['battery_capacity'],
                            min=1,
                            # step=1,
                            debounce=True,
                            style={'width': '80px'},
                        ), width='auto'),
                    ], justify='between', align='center'),
                    dbc.Row([
                        dbc.Col(dbc.Label('Maximaal laadvermogen [kW]:')),
                        dbc.Col(dcc.Input(
                            id="input-battery-charge-power",
                            type='number',
                            placeholder="[kW]",
                            value=DEFAULT_VALUES['battery_charge_power'],
                            min=1,
                            # step=1,
                            debounce=True,
                            style={'width': '80px'},
                        ), width='auto'),
                    ], justify='between', align='center'),
                    dbc.Row([
                        dbc.Col(dbc.Label('Maximaal afgiftevermogen [kW]:')),
                        dbc.Col(dcc.Input(
                            id="input-battery-discharge-power",
                            type='number',
                            placeholder='[kW]',
                            value=DEFAULT_VALUES['battery_discharge_power'],
                            min=1,
                            # step=1,
                            debounce=True,
                            style={'width': '80px'},
                        ), width='auto'),
                    ], justify='between', align='center'),
                    dbc.Row([
                        dbc.Col(dbc.Label('Rendament [%]:')),
                        dbc.Col(dcc.Input(
                            id="input-battery-efficiency",
                            type='number',
                            placeholder="[%]",
                            value=DEFAULT_VALUES['battery_efficiency'],
                            min=1,
                            # step=1,
                            debounce=True,
                            style={'width': '80px'},
                        ), width='auto'),
                    ], justify='between', align='center'),
                    html.Div(
                        dbc.Row([
                            dbc.Col(dbc.Label('Initiële lading [%]:')),
                            dbc.Col(dcc.Input(
                                id="input-battery-initial-load",
                                type='number',
                                placeholder="[%]",
                                value=DEFAULT_VALUES['battery_initial_load'],
                                min=0,
                                max=100,
                                step=1,
                                debounce=True,
                                style={'width': '80px'},
                            ), width='auto'),
                        ], justify='between', align='center'),
                        hidden=False),
                ]),
            ),
        ]),
        # Other.
        dbc.Row([
            dbc.Card(html.Div(
                [
                    dbc.Label('Overig', style={'font-weight': 'bold'}),
                    dbc.Row([
                        dbc.Col(dbc.Label('Maximaal vermogen net [kW]:')),
                        dbc.Col(dcc.Input(
                            id="input-max-net-power",
                            type='number',
                            placeholder="[kW]",
                            value=DEFAULT_VALUES['max_net_power'],
                            min=1,
                            # step=1,
                            debounce=True,
                            style={'width': '80px'},
                        ), width='auto'),
                    ], justify='between', align='center'),
                    dbc.Row([
                        dbc.Col(dbc.Label('Start dag [uur]:')),
                        dbc.Col(dcc.Input(
                            id="input-day-start-hour",
                            type='number',
                            placeholder="[uur]",
                            value=DEFAULT_VALUES['day_start_hour'],
                            min=0,
                            max=23,
                            # step=1,
                            debounce=True,
                            style={'width': '80px'},
                        ), width='auto'),
                    ], justify='between', align='center'),
                    dbc.Row([
                        dbc.Col(dbc.Label('Eind dag [uur]:')),
                        dbc.Col(dcc.Input(
                            id="input-day-end-hour",
                            type='number',
                            placeholder="[uur]",
                            value=DEFAULT_VALUES['day_end_hour'],
                            min=0,
                            max=23,
                            # step=1,
                            debounce=True,
                            style={'width': '80px'},
                        ), width='auto'),
                    ], justify='between', align='center'),
                    html.Div(
                        dbc.Row(
                            [
                                dbc.Col(html.Div('Tijdsstap [h]:')),
                                dbc.Col(
                                    dcc.Input(
                                        id="input-dt",
                                        type='number',
                                        placeholder="Tijdsstap",
                                        value=DEFAULT_VALUES['dt'],
                                        min=1,
                                        debounce=True,
                                        style={'width': '80px'},
                                    ), width='auto'),
                            ], justify='between', align='center'),
                        hidden=True,
                    ),
                ]),
            ),
        ]),
    ]
)

graphs_col = dbc.Stack([
    # Input graph.
    dbc.Card([
        dbc.Row([
            dbc.Col(
                dbc.Label('Input', style={'font-weight': 'bold'}),
                width=1,
            ),
        ]),
        dbc.Row([
            dbc.Col(
                dbc.Label('Eenheid:', style={'font-weight': 'bold'}),
                width=1,
            ),
            dbc.Col(
                dcc.Dropdown([
                    'Vermogen [kW]',
                    'Energie/dag [kWh]',
                    'Energie/week [kWh]',
                    'Energie/maand [kWh]',
                ], value='Vermogen [kW]', id='dropdown-input-unit'),
            ),
        ]),
        dcc.Loading(
            id="loading-input",
            type="default",
            children=dcc.Graph(id='graph-input'),
        ),
    ]),

    # Output graph.
    dbc.Card([
        dbc.Row([
            dbc.Col(
                dbc.Label('Output:', style={'font-weight': 'bold'}),
                width=1,
            ),
            dbc.Col(
                dcc.Dropdown([
                    'Verbruik',
                    'Zon',
                    'Batterij',
                    'Alle stromen',
                ], value='Verbruik', id='dropdown-output-which'),
            ),
        ]),
        dbc.Row([
            dbc.Col(
                dbc.Label('Eenheid:', style={'font-weight': 'bold'}),
                width=1,
            ),
            dbc.Col(
                dcc.Dropdown([
                    'Vermogen [kW]',
                    'Energie/dag [kWh]',
                    'Energie/week [kWh]',
                    'Energie/maand [kWh]',
                ], value='Vermogen [kW]', id='dropdown-output-unit'),
            ),
        ]),

        dcc.Loading(
            id="loading-output",
            type="default",
            children=dcc.Graph(id='graph-output'),
        ),
    ]),
])

# Layout.
layout = html.Div(
    id='layout', children=
    [
        # Store.
        dcc.Store(id='store-input'),
        dbc.Row(
            [
                # Title.
                dbc.Col([
                    html.H1(app_title)
                ]),
            ]
        ),
        dbc.Row([
            dbc.Col(
                options_col,
                width=2,
            ),
            dbc.Col(
                graphs_col,
                width=10,
            ),
        ], align='center',),
    ]
)

layout = dbc.Container(layout, fluid=True)

app.layout = layout


# Callbacks.
@app.callback(
    Output('store-input', 'data'),
    Input('input-total-energy-demand', 'value'),
    Input('input-season-ratio-demand', 'value'),
    Input('input-day-ratio-demand', 'value'),
    Input('input-solar-wp', 'value'),
    Input('input-solar-panels-amount', 'value'),
    Input('input-season-ratio-solar', 'value'),
    Input('input-dt', 'value'),
)
def compute_input(
        total_energy_demand, season_ratio_demand, day_ratio_demand,
        solar_wp, solar_panels_amount, season_ratio_solar, dt):
    if any((inp is None
            for inp in (total_energy_demand, season_ratio_demand, day_ratio_demand,
                        solar_wp, solar_panels_amount, season_ratio_solar, dt))):
        raise PreventUpdate

    # Total solar peak power.
    total_wp = solar_wp * solar_panels_amount

    # Get energy demand.
    t_demand, P_demand = get_demand(
        total_energy=total_energy_demand,
        season_ratio=season_ratio_demand,
        day_ratio=day_ratio_demand,
        dt=dt)

    # Get solar power.
    t_solar, P_solar = get_solar_power(
        Wp=total_wp,
        season_ratio=season_ratio_solar,
        dt=dt)

    assert len(t_demand) == len(t_solar)

    # Collect in dict.
    input_data = {
        'time': t_demand,
        'P_demand': P_demand,
        'P_solar': P_solar,
    }

    return input_data


# Callbacks.
@app.callback(
    Output('graph-input', 'figure'),
    Input('store-input', 'data'),
    Input('dropdown-input-unit', 'value'),
    State('input-dt', 'value'),
)
def plot_input(input_data, unit, dt):
    if any((inp is None
            for inp in (input_data, unit))):
        raise PreventUpdate

    # Extract data.
    time = np.asarray(input_data['time'])

    # Create a figure.
    fig = go.Figure()

    # Convert time.
    t, xlabel = convert_time_unit(t=time, unit=unit)

    for lab in ['P_demand', 'P_solar']:
        pi = np.asarray(input_data[lab])

        # Convert power.
        yi = convert_power_unit(p=pi, unit=unit, dt=dt)

        # Plot.
        fig.add_trace(
            go.Scatter(
                x=t, y=yi,
                mode='lines',
                line=dict(
                    shape='linear' if 'Vermogen' in unit else 'hvh',
                    # color='#89e053',
                    # dash='dash',
                ),
                showlegend=True,
                name=SIGNAL_LABELS.get(lab, lab),
            )
        )

    fig.update_layout(
        xaxis=dict(
            title=xlabel,
        ),
        yaxis=dict(
            title=unit,
        ),
        # hovermode="x unified",

        # Maintain zoom level unless this 'uirevision' property changes.
        # https://community.plotly.com/t/preserving-ui-state-like-zoom-in-dcc-graph-with-uirevision-with-dash/15793
        uirevision=f'{unit}',
    )

    if 'maand' in unit.lower():
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=np.arange(1, 13),
                ticktext=['Jan', 'Feb', 'Mrt', 'Apr', 'Mei', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec'],
            ),
        )

    return fig


# Callbacks.
@app.callback(
    Output('graph-output', 'figure'),
    Input('store-input', 'data'),
    Input('input-max-net-power', 'value'),
    Input('input-battery-capacity', 'value'),
    Input('input-battery-charge-power', 'value'),
    Input('input-battery-discharge-power', 'value'),
    Input('input-battery-efficiency', 'value'),
    Input('input-battery-initial-load', 'value'),
    Input('input-day-start-hour', 'value'),
    Input('input-day-end-hour', 'value'),
    State('input-dt', 'value'),
    Input('dropdown-output-unit', 'value'),
    Input('dropdown-output-which', 'value'),
)
def plot_output(input_data, P_net_max, Q_bat_max,
                P_bat_charge_max, P_bat_discharge_max, efficiency_bat, pQ_bat_init,
                day_start_hour, day_end_hour, dt, unit, which):
    if input_data is None:
        raise PreventUpdate

    P_demand = input_data['P_demand']
    P_solar = input_data['P_solar']
    sim = EnergySimulator(
        P_demand=P_demand, P_solar=P_solar,
        P_net_max=P_net_max*1000, Q_bat_max=Q_bat_max*1000,
        P_bat_charge_max=P_bat_charge_max*1000,
        P_bat_discharge_max=P_bat_discharge_max*1000, efficiency_bat=efficiency_bat,
        pQ_bat_init=pQ_bat_init, day_start_hour=day_start_hour, day_end_hour=day_end_hour)
    output = sim.run()

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    time = output['time']
    output['P_demand'] = input_data['P_demand']
    output['P_solar'] = input_data['P_solar']

    # Convert time.
    t, xlabel = convert_time_unit(t=time, unit=unit)

    # Plot battery charge.
    lab = 'pQ_bat'
    pi = np.asarray(output[lab])
    yi = convert_bat_charge(pi, unit=unit, dt=dt)
    fig.add_trace(
        go.Scatter(
            x=t, y=yi,
            mode='lines',
            line=dict(
                shape='linear' if 'Vermogen' in unit else 'hvh',
                # color='#89e053',
                dash='dot',
            ),
            showlegend=True,
            name=SIGNAL_LABELS.get(lab, lab),
        ),
        secondary_y=True,
    )

    # Plot energy flows.
    if which == 'Verbruik':
        signals_to_plot = ['P_sol_dem', 'P_bat_dem', 'P_net_dem', 'P_demand']
    elif which == 'Zon':
        signals_to_plot = ['P_sol_dem', 'P_sol_bat', 'P_sol_net', 'P_solar']
    elif which == 'Batterij':
        signals_to_plot = ['P_bat_dem', 'P_sol_bat', 'P_net_bat']
    elif which == 'Alle stromen':
        signals_to_plot = ['P_sol_dem', 'P_bat_dem', 'P_net_dem', 'P_sol_bat', 'P_sol_net',
                           'P_net_bat']
    else:
        raise NotImplementedError(f'Figure not implemented for "{which}".')

    for lab in signals_to_plot:
        pi = np.asarray(output[lab])

        # Convert power.
        yi = convert_power_unit(p=pi, unit=unit, dt=dt)

        stackgroup = 'Cumulative' if ((lab not in ['P_demand', 'P_solar'])
                                      and (which in ['Verbruik', 'Zon'])) else None

        # Plot.
        fig.add_trace(
            go.Scatter(
                x=t, y=yi,
                mode='lines',
                line=dict(
                    shape='hv' if 'Vermogen' in unit else 'hvh',
                    # color='#89e053',
                    dash='dash' if lab in ['P_demand', 'P_solar'] else None,
                ),
                showlegend=True,
                name=SIGNAL_LABELS.get(lab, lab),
                stackgroup=stackgroup,
            )
        )

    if np.max(output['P_net_dem']) > P_net_max*1000:
        title = '!Zekering door!'
    else:
        title = 'Zekering ok'

    fig.update_layout(
        title=title,
        xaxis=dict(
            title=xlabel,
        ),
        yaxis=dict(
            title=unit,
            fixedrange=True,
        ),
        # hovermode="x unified",

        # Maintain zoom level unless this 'uirevision' property changes.
        # https://community.plotly.com/t/preserving-ui-state-like-zoom-in-dcc-graph-with-uirevision-with-dash/15793
        uirevision=f'{unit}',
    )

    if 'maand' in unit.lower():
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=np.arange(1, 13),
                ticktext=['Jan', 'Feb', 'Mrt', 'Apr', 'Mei', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec'],
            ),
        )

    # Set y-axes titles
    fig.update_yaxes(
        range=[-0.5, 100.5],
        fixedrange=True,
        title_text="Batterijlading [%]",
        secondary_y=True)

    return fig


if __name__ == '__main__':
    app.run(debug=True)
