# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 20:24:51 2023
Parrell and series capacitor under the same value 
@author: nicoy
"""

from dash import Dash, dcc, html, Input, Output, callback, dash_table
import control
import control.matlab
import numpy as np
import math
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd

capacitor_df = pd.read_csv('https://raw.githubusercontent.com/YushiYang2023/Eaton-Project-capacitor-aging/main/capacitor%20parameter.csv')
capacitor_df_after_shift=pd.read_csv('https://raw.githubusercontent.com/YushiYang2023/Eaton-Project-capacitor-aging/main/capacitor%20parameter.csv')


app = Dash(__name__)
#configure the layout of the html
#%%about its layout
app.layout = html.Div([
    html.H1("Film capacitor under test"),
    html.H3("Capacitor combination"),
    "Parallel number(M): ",
    dcc.Input(id='Parallel_number', value=2, type ='number'),
    html.Br(),
    "Series number(N): ",
    dcc.Input(id='Series_number', value=3, type ='number'),
    html.Br(),
    html.Img(src="https://raw.githubusercontent.com/YushiYang2023/Eaton-Project-capacitor-aging/main/capacitor%20in%20group1.jpg",
             width = 500),
    html.H2("Original data"),
    dash_table.DataTable(data=capacitor_df.to_dict('records')),
    html.H2("Data after shifting"),
    
    
    "whether shift or not ",
    dcc.RadioItems(options=['shift', 'not_shift'], value='not_shift', id='capacitor_shift_signal'),
    html.Div(id='shift_signal'),
    html.Div(id='table_after_shifting'),
    
    
    
    html.H2("--------------------------------------Input--------------------------------------"),
    html.Div([
        
        html.Div(children=[
            
            
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
            
            # dcc.Store stores the tf parameter
            dcc.Store(id='tf_num'),
            dcc.Store(id='tf_den'),
            
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
        # dcc.Store stores the intermediate value
        dcc.Store(id='time_dut'),
        dcc.Store(id='voltage_dut'),
        dcc.Store(id='current_dut'),
        
        html.Div(children=[
            html.Img(src="https://raw.githubusercontent.com/YushiYang2023/Eaton-Project-capacitor-aging/main/input%20and%20output%20structures%20of%20the%20capacitor%20circuit%20model.jpg",
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

    # html.Div(id='tf_den'),
    # html.Div(id='tf_num'),
])
#%%definition of the input and output
# Add controls to build the interaction
@callback(
    Output(component_id='shift_signal', component_property='children'),
    Output(component_id='table_after_shifting', component_property='children'),
    Input(component_id='capacitor_shift_signal', component_property='value')
)
def update_capacitor_pra(capacitor_shift_signal):
    # df1 = df+1
    shift_signal = capacitor_shift_signal
    

    dfshape=capacitor_df_after_shift.iloc[:, capacitor_df_after_shift.columns != 'capacitor parameter'].shape
    capacitor_df_after_shift.iloc[:, capacitor_df_after_shift.columns != 'capacitor parameter'] = capacitor_df.iloc[:, capacitor_df_after_shift.columns != 'capacitor parameter'].multiply(np.random.rand(dfshape[0],dfshape[1])*0.3*(shift_signal=='shift')+1)
    # print(dfshape[0])
    return shift_signal,dash_table.DataTable(data=capacitor_df_after_shift.to_dict('records'))


#%%calculation of the total transfer function
@callback(
    Output(component_id='tf_den', component_property='data'),
    Output(component_id='tf_num', component_property='data'),
    
    Input(component_id='Parallel_number', component_property='value'),
    Input(component_id='Series_number', component_property='value'),
    
    Input(component_id='Rp', component_property='value'),
    Input(component_id='Rd', component_property='value'),
    Input(component_id='Cd', component_property='value'),
    Input(component_id='Rs', component_property='value'),
    Input(component_id='Ls', component_property='value'),
    Input(component_id='C', component_property='value'),
    

)
def update_total_tf(Parallel_number,Series_number,Rp,Rd,Cd,Rs,Ls,C):
    

    
    s = control.TransferFunction.s
    H = s-s
    for sn in range(1,Series_number+1):
        I=s-s
        for pn in range(1,Parallel_number+1):
            # print(pn,sn)
            # C = 1e-6 * capacitor_df_after_shift.iloc['C/uF', 3*pn+sn ]
            # print(capacitor_df_after_shift.iloc[:, (pn-1)*Series_number+sn])
            # [C_value,Rp_value,Cd_value,Rd_value,Ls_value,Rs_value] = capacitor_df_after_shift.iloc[:, (pn-1)*Series_number+sn]
            [C_value,Rp_value,Cd_value,Rd_value,Ls_value,Rs_value] = [C,Rp,Cd,Rd,Ls,Rs]
            Cd_value= 1e-6*Cd_value#uF when input
            C_value = 1e-6*C_value#uF when input
            print('para=',1/(s*Ls_value+Rs_value + 1/(1/Rp_value+C_value*s+1/(Rd_value+1/(Cd_value*s)))))
            I_v = I
            add = 1/(s*Ls_value+Rs_value + 1/(1/Rp_value+C_value*s+1/(Rd_value+1/(Cd_value*s))))
            I=control.parallel(I_v, add)
            print('I(s)=',I)
        H=H+1/I
    H=1/H
    print('H(s)=',H)
    Cd= 1e-6*Cd#uF when input
    C = 1e-6*C#uF when input
    H = Parallel_number/Series_number/( s*Ls+Rs + 1/(1/Rp+C*s+1/(Rd+1/(Cd*s))));#transfer function for the capactor model I(s)/U(s)
    # print('H(s)=',H)
    # print('2H(s)=',control.parallel(H, 1))
    # print(control.tfdata(H))
    [den,num]=control.tfdata(H)
    return den,num
#%%produce the bode figure of the capacitor group
@callback(
    Output('capacitor_bode', 'figure'),

    
    Input(component_id='tf_den', component_property='data'),
    Input(component_id='tf_num', component_property='data'),

)
def update_tf_bode(den,num):
    
    
    H = control.TransferFunction(den,num)
    # print(control.TransferFunction(den,num))
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
    return fig_bode
#%%produce the performance device under test (DUT)
@callback(
    Output(component_id='time_dut', component_property='data'),
    Output(component_id='voltage_dut', component_property='data'),
    Output(component_id='current_dut', component_property='data'),



    Input(component_id='tf_den', component_property='data'),
    Input(component_id='tf_num', component_property='data'),
    
    Input(component_id='DC_bias', component_property='value'),
    Input(component_id='AC_amp', component_property='value'),
    Input(component_id='AC_fre', component_property='value'),
    
    Input(component_id='Tsa', component_property='value'),
    Input(component_id='Trun', component_property='value'),

)
def update_votlage_current_fig(den,num,DC_bias,AC_amp,AC_fre,Tsa,Trun):
    
    
    H = control.TransferFunction(den,num)
    
    w = 2*AC_fre*math.pi;
    denominator = np.array([1,0,w*w])
    numerator = np.array([w,0])

    ac_part = control.tf(numerator,denominator)*AC_amp+DC_bias#I
    # print('I(s)=',ac_part)

    total_system = control.series(H, ac_part)

    t ,voltage = control.step_response(ac_part,Trun,0,None,None,round(Trun/Tsa)) #np.array
    t ,current = control.step_response(total_system,Trun,0,None,None,round(Trun/Tsa)) #np.array
    
    return t,voltage,current
#%%plot the voltage current figure of DUT
@callback(
    Output('voltage_current_figure', 'figure'),
    
    Input(component_id='time_dut', component_property='data'),
    Input(component_id='voltage_dut', component_property='data'),
    Input(component_id='current_dut', component_property='data'),
)
def update_voltage_current_figure(t,voltage,current):
    
    
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
    
    return fig
#%%plot the power loss and each power loss
@callback(

    Output(component_id='i_rms', component_property='children'),
    Output(component_id='P_loss', component_property='children'),
    Output(component_id='each_P_loss', component_property='children'),
    Output(component_id='I_dc', component_property='children'),
    Output(component_id='i_ac_ripple', component_property='children'),
    Output(component_id='uac_ripple', component_property='children'),
    Output(component_id='fac_ripple', component_property='children'),
    
    Output('fig_fft', 'figure'),
    
    Input(component_id='Parallel_number', component_property='value'),
    Input(component_id='Series_number', component_property='value'),
    
    Input(component_id='time_dut', component_property='data'),
    Input(component_id='voltage_dut', component_property='data'),
    Input(component_id='current_dut', component_property='data'),
    
    Input(component_id='AC_fre', component_property='value'),
    Input(component_id='Tsa', component_property='value'),
)
def update_Ploss(Parallel_number,Series_number,t,voltage,current,AC_fre,Tsa):
    #calculate the Irms and Ploss
    # print(current[-1-round(1/AC_fre/Tsa):-1:1])
    #only final cycle is calculated
    current_one_cycle = current[-1-round(1/AC_fre/Tsa):-1:1];
    voltage_one_cycle = voltage[-1-round(1/AC_fre/Tsa):-1:1];
    
    i_rms = np.sqrt(np.dot(current_one_cycle, current_one_cycle) / (1/AC_fre/Tsa) )
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
    uac_ripple=X_mag_plot[1]
    fac_ripple=f0
    return f'I_rms: {i_rms}A', f'total_P_loss: {P_loss}W',f'each_P_loss: {each_P_loss}W',f'I_dc: {Idc}A',f'i_ac_ripple_Amp: {iac_ripple}A',f'u_ac_ripple_Amp: {uac_ripple}V',f'f_ac_ripple: {fac_ripple}Hz',fig_fft

if __name__ == '__main__':
    app.run(debug=True)
