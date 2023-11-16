# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 20:24:51 2023
Parrell and series capacitor under the same value 
@author: nicoy
"""

from dash import Dash, dcc, html, Input, Output, callback
import control
import control.matlab
import numpy as np
import math
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt


app = Dash(__name__)
#configure the layout of the html
#%%about its layout
app.layout = html.Div([
    html.H1("Film capacitor under test"),
    
    html.H2("--------------------------------------Input--------------------------------------"),
    html.Div([
        html.Div(children=[
            html.H3("Capacitor combination"),
            "Parallel number ",
            dcc.Input(id='Parallel_number', value=1, type ='number'),
            html.Br(),
            "Series number: ",
            dcc.Input(id='Series_number', value=1, type ='number'),
            html.Br(),
            
            html.H3("Each Capacitor information"),
            "Rp: ",
            dcc.Input(id='Rp', value=1e5, type ='number'),"Ohm",
            html.Br(),
                        
            "Cd: ",
            dcc.Input(id='Cd', value=0.1, type ='number'),"uF",
            html.Br(),
            "Rd: ",
            dcc.Input(id='Rd', value=1e5, type ='number'),"Ohm",
            
            html.Br(),
            "Ls: ",
            dcc.Input(id='Ls', value=1e-9, type ='number'),"H",
            html.Br(),
            "Rs: ",
            dcc.Input(id='Rs', value=0.01, type ='number'),"Ohm",
                        
            html.Br(),
            "C  : ",
            dcc.Input(id='C', value=1, type ='number'),"uF",
            
            html.Br(),
            html.H3("Input voltage"),
            "Amp of DC component : ",
            dcc.Input(id='DC_bias', value=1000, type ='number'),"V",
            html.Br(),
            "Amp of AC component: ",
            dcc.Input(id='AC_amp', value=100, type ='number'),"V",
            html.Br(),
            "AC frequency: ",
            dcc.Input(id='AC_fre', value=60, type ='number'),"Hz",
            
            html.Br(),
            html.H3("System settings"),
            " t_AD: ",
            dcc.Input(id='Tsa', value=1e-4, type ='number'),"s",
            html.Br(),
            "T_total: ",
            dcc.Input(id='Trun', value=10.0/60, type ='number'),"s",
            
        ], style={'padding': 10, 'flex': 1}),
        
        html.Div(children=[
            html.Img(src="https://uark-my.sharepoint.com/:i:/r/personal/yushiy_uark_edu/Documents/eaton/input%20and%20output%20structures%20of%20the%20capacitor%20circuit%20model.jpg?csf=1&web=1&e=lbndLc",
                     width = 800),
            
            
            ], style={'padding': 10, 'flex': 1}),
        ], style={'display': 'flex', 'flex-direction': 'row'}),
    
    html.Br(),
    
    html.H2("--------------------------------------Output--------------------------------------"),
    
    html.H3("u information"),
    html.Div(id='uac_ripple'),
    html.Div(id='fac_ripple'),
    
    html.H3("i information"),
    html.Div(id='i_rms'),
    html.Div(id='I_dc'),
    html.Div(id='i_ac_ripple'),
    

    html.H3("Power loss"),
    html.Div(id='P_loss'),
    html.Div(id='each_P_loss'),
    
    html.H3("Voltage_current_figure"),
    dcc.Graph(id='voltage_current_figure'),
    html.H3("Capacitor_impedance"),
    dcc.Graph(id='capacitor_bode'),
    html.H3("FFT_result"),
    dcc.Graph(id='fig_fft'),

])
#%%definition of the input and output
#configure the relationship beneath the layout
@callback(
    Output(component_id='i_rms', component_property='children'),
    Output(component_id='P_loss', component_property='children'),
    Output(component_id='each_P_loss', component_property='children'),
    Output(component_id='I_dc', component_property='children'),
    Output(component_id='i_ac_ripple', component_property='children'),
    Output(component_id='uac_ripple', component_property='children'),
    Output(component_id='fac_ripple', component_property='children'),
    
    Output('voltage_current_figure', 'figure'),
    Output('capacitor_bode', 'figure'),
    Output('fig_fft', 'figure'),
    
    Input(component_id='Parallel_number', component_property='value'),
    Input(component_id='Series_number', component_property='value'),
    
    Input(component_id='Rp', component_property='value'),
    Input(component_id='Rd', component_property='value'),
    Input(component_id='Cd', component_property='value'),
    Input(component_id='Rs', component_property='value'),
    Input(component_id='Ls', component_property='value'),
    Input(component_id='C', component_property='value'),
    
    Input(component_id='DC_bias', component_property='value'),
    Input(component_id='AC_amp', component_property='value'),
    Input(component_id='AC_fre', component_property='value'),
    
    Input(component_id='Tsa', component_property='value'),
    Input(component_id='Trun', component_property='value'),
)
#%%set the transfer function of the capacitor
def update_output_div(Parallel_number,Series_number,Rp,Rd,Cd,Rs,Ls,C,DC_bias,AC_amp,AC_fre,Tsa,Trun):
    Cd= 1e-6*Cd#uF when input
    C = 1e-6*C#uF when input
    
    s = control.TransferFunction.s
    H = Parallel_number/Series_number/( s*Ls+Rs + 1/(1/Rp+C*s+1/(Rd+1/(Cd*s))));#transfer function for the capactor model I(s)/U(s)
    
    
    mag, phase, omega = control.matlab.bode(H)#,plot=1
    fig_bode = make_subplots(   rows=2, cols=1    )
    # Add traces
    
    fig_bode.add_trace(go.Scatter(x=omega,y=np.log10(mag)*20), row=1, col=1)
    fig_bode.add_trace(go.Scatter(x=omega,y=phase*180/math.pi), row=2, col=1)
    
    # fig_bode.add_trace(px.Scatter(d, x='omega',y='mag', log_x=True), row=1, col=1)
    
    # fig_bode.add_trace(px.Scatter(omega,y=phase*180/math.pi, log_x=True), row=2, col=1)

    # Update xaxis properties
    
    fig_bode.update_xaxes(title_text="Frequency(rad/sec)", row=1, col=1, type="log")
    fig_bode.update_xaxes(title_text="Frequency(rad/sec)", row=2, col=1, type="log")
    # Update yaxis properties
    fig_bode.update_yaxes(title_text="Magnitude(dB)", row=1, col=1)
    fig_bode.update_yaxes(title_text="Phase(deg)", row=2, col=1)

    # Update title and height
    fig_bode.update_layout(showlegend=False,width = 500, height = 500,
                            margin=dict(l=20, r=20, t=20, b=20),
                            paper_bgcolor="LightSteelBlue")    
    
    # print('omega=',omega)
    
    print('H(s)=',H)
    #%%current output of the capacitor under the votlage input 
    w = 2*AC_fre*math.pi;
    denominator = np.array([1,0,w*w])
    numerator = np.array([w,0])

    ac_part = control.tf(numerator,denominator)*AC_amp+DC_bias#I
    # print('I(s)=',ac_part)

    total_system = control.series(H, ac_part)

    t ,voltage = control.step_response(ac_part,Trun,0,None,None,round(Trun/Tsa)) #np.array
    t ,current = control.step_response(total_system,Trun,0,None,None,round(Trun/Tsa)) #np.array



    fig = make_subplots(
        rows=2, cols=1
    )

    # Add traces
    fig.add_trace(go.Scatter(x=t,y=voltage), row=1, col=1)
    fig.add_trace(go.Scatter(x=t,y=current), row=2, col=1)

    # Update xaxis properties
    fig.update_xaxes(title_text="t/s", row=1, col=1)
    fig.update_xaxes(title_text="t/s", row=2, col=1)

    # Update yaxis properties
    fig.update_yaxes(title_text="capacitor voltage/V", row=1, col=1)
    fig.update_yaxes(title_text="capacitor current/A", row=2, col=1)

    # Update title and height
    fig.update_layout(showlegend=False,width = 500, height = 500, 
                      margin=dict(l=20, r=20, t=20, b=20),
                      paper_bgcolor="LightSteelBlue",)
    
    #%%calculate the Irms and Ploss
    # print(current[-1-round(1/AC_fre/Tsa):-1:1])
    #only final cycle is calculated
    current_one_cycle = current[-1-round(1/AC_fre/Tsa):-1:1];
    voltage_one_cycle = voltage[-1-round(1/AC_fre/Tsa):-1:1];
    
    i_rms = np.sqrt(np.mean(current_one_cycle**2))
    P_loss= np.dot(current_one_cycle, voltage_one_cycle) / (1/AC_fre/Tsa)
    each_P_loss = P_loss/Parallel_number/Series_number
    # P_loss= np.dot(current, voltage) / (Trun/Tsa)
    #%%fft analysis for the current and voltage

    
    
    Fs = 1/Tsa#sampling frequency
    tstep = 1/Fs#
    f0 = 60#signal frequency
    fstep = 1/f0
    N = int(Fs/f0)

    t_fft = t[-1-N:-1:1]
    fstep = Fs/N
    f = np.linspace(0, (N-1)*fstep,N)

    y = current[-1-N:-1:1]


    X = np.fft.fft(y)
    X_mag = np.abs(X)/N

    f_plot = f[0:int(N/2+1)]
    X_mag_plot = 2 * X_mag[0:int(N/2+1)]
    X_mag_plot[0] = X_mag_plot[0]/2#dc componenet should not be doubled 

    
    fig_fft = make_subplots(
        rows=2, cols=2, subplot_titles=("voltage_part_been_fft", "current_part_been_fft")
    )

    # Add traces
    fig_fft.add_trace(go.Scatter(x=t_fft,y=y), row=1, col=2)
    fig_fft.add_trace(go.Scatter(x=f_plot,y=X_mag_plot), row=2, col=2)

    # Update xaxis properties
    fig_fft.update_xaxes(title_text="t/s", row=1, col=2)
    fig_fft.update_xaxes(title_text="frequency/Hz", row=2, col=2)

    # Update yaxis properties
    fig_fft.update_yaxes(title_text="curremt/A", row=1, col=2)
    fig_fft.update_yaxes(title_text="frequency domain/A", row=2, col=2)

    # Update title and height
    fig_fft.update_layout(showlegend=False,width = 1000, height = 500, 
                      margin=dict(l=20, r=20, t=20, b=20),
                      paper_bgcolor="LightSteelBlue",)

    Idc=X_mag_plot[0]
    iac_ripple=X_mag_plot[1]
    #%% voltage fft
    y = voltage[-1-N:-1:1]


    X = np.fft.fft(y)
    X_mag = np.abs(X)/N

    f_plot = f[0:int(N/2+1)]
    X_mag_plot = 2 * X_mag[0:int(N/2+1)]
    X_mag_plot[0] = X_mag_plot[0]/2#dc componenet should not be doubled 
    # Add traces
    fig_fft.add_trace(go.Scatter(x=t_fft,y=y), row=1, col=1)
    fig_fft.add_trace(go.Scatter(x=f_plot,y=X_mag_plot), row=2, col=1)

    # Update xaxis properties
    fig_fft.update_xaxes(title_text="t/s", row=1, col=1)
    fig_fft.update_xaxes(title_text="frequency/Hz", row=2, col=1)

    # Update yaxis properties
    fig_fft.update_yaxes(title_text="voltage/V", row=1, col=1)
    fig_fft.update_yaxes(title_text="frequency domain/A", row=2, col=1)

    # Update title and height
    fig_fft.update_layout(showlegend=False,width = 1000, height = 500, 
                      margin=dict(l=20, r=20, t=20, b=20),
                      paper_bgcolor="LightSteelBlue",)
    uac_ripple=X_mag[0]
    fac_ripple=f0
    
    return f'I_rms: {i_rms}A', f'total_P_loss: {P_loss}W',f'each_P_loss: {each_P_loss}W',f'I_dc: {Idc}A',f'i_ac_ripple_Amp: {iac_ripple}A',f'u_ac_ripple_Amp: {uac_ripple}V',f'f_ac_ripple: {fac_ripple}Hz',fig,fig_bode,fig_fft


if __name__ == '__main__':
    app.run(debug=True)
