# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 01:46:18 2021

@author: crist
"""

from scipy.integrate import odeint
import numpy as np 
from scipy.optimize import fsolve


#%%     Secado incicial del orujo

def Bloque_Secado(Masa_0,Hum_0,T_Sec,PTOS,PPFS):
    
    #INPUTS
    
    #Masa_0: Masa de orujo humedo a procesar (kg)
    #Hum_0 : Humedad inicial (%) AJUSTAR
    #T_sec : Temperatura de secado (K°)
    #PTOS  : Polifenoles totales orujo seco (%) AJUSTAR 
    #PPFS  : Porcentaje peso PT en semillas (%)
    
    #OUTPUTS

    #R     : % de Polifenoles recuperados en solidos posterior al secado (%)
    #ts    : Tiempo total necesario para realizar el tiempo de secado (min)

    
    #Datos balance de masa
    Peso_solidos    = Masa_0*(1-Hum_0);             #(kg)
    Peso_liquido    = Masa_0*Hum_0;                 #(kg)   
    Peso_total_5p   = Peso_solidos/0.95;            #(kg)
    Peso_H2O_obj    = Peso_total_5p - Peso_solidos; #(kg)
    X0              = Peso_liquido/Peso_solidos;    #(base seca)
    S0              = Peso_solidos;                 #(kg)
    Peso_a_perder   = Peso_liquido-Peso_H2O_obj;    #(kg) (Entrada a modelo)
    
    #Parametros cinetica de secado
    if (T_Sec<60+273):
        i = 0;
    elif (60+273<=T_Sec and T_Sec<65+272):
        i=  1;
    elif (65+273<=T_Sec and T_Sec<70+273):
        i=  2;
    elif (70+273<=T_Sec and T_Sec<75+273):
        i=  3;
    elif (75+273<=T_Sec and T_Sec<80+273):
        i=  4;
    else:
        i=  5;
    
    MS = np.array([[ 1.00615e+00,  2.75300e-03,  9.64220e-01, -7.34000e-04],
                   [ 1.00102e+00,  1.50400e-03,  1.18771e+00, -3.13000e-04],
                   [ 9.98810e-01,  1.66300e-03,  1.17540e+00, -3.52000e-04],
                   [ 1.00342e+00,  3.56900e-03,  1.06121e+00, -4.47000e-04],
                   [ 9.99540e-01,  2.36900e-03,  1.17125e+00, -3.74000e-04],
                   [ 9.89650e-01,  1.54600e-03,  1.30818e+00, -2.45000e-04]])
    
    a = MS[i,0]; k = MS[i,1]; n = MS[i,2]; b = MS[i,3];
    
    #Integracion del modelo: Determinacion del tiempo de secado al 5% humedad
    MR_objetivo     = Peso_total_5p/Masa_0;
    
    def MR(t):
        return a*np.exp(-k*t**n)+b*t - MR_objetivo;
    ts              = fsolve(MR,100)/0.5;
   
    #Datos cinetica de degradacion por tratamiento termico
    PT0_semillas  	= Peso_solidos*PTOS*PPFS*1e6;           #(mg_PT)
    C0              = PT0_semillas/(Peso_solidos*1000);     #(mg_PT/g_ss)(Entrada a modelo)
    x0              = C0;
    tspan           = np.linspace(0,float(ts),100) ;

    def Modelo_Secado(x, t, T, X0, S0, a, k, n, b):
        # variables de estado
        C = x[0];   # mgPF/gSS
    
        # Proceso de secado
        MR = a*np.exp(-k*t**n)+b*t;
        
        # Parametros perdida
        k0 = 483.77e-5;
        Ea = 19.39;
        A1 = 1127.5;
        A2 = 730.77;
        A3 = 60.38;
        R  = 8.31e-3; 
        
        # Ecuaciones constitutivas
        M0 = X0*S0+S0;
        M  = MR*M0;  
        X  = (M-S0)/M;
        kt = k0*np.exp(-Ea/(R*T));
        kx = A1+A2*X+A3*X**2;
        k  = kt*kx;  
    
        # Sistema ODE
        dCdt   = -k*C;                     # mg_polifenoles totales/g
        dzdt = [dCdt];
        
        return dzdt
    
    # Obtener carga polifenolica final
    x = odeint(Modelo_Secado, x0, tspan, args=(T_Sec, X0, S0, a, k, n, b));
    R = x[-1]/C0;
    return [R,ts,Peso_a_perder]

#%% Extraccion de polifenoles del orujo
    
#INPUTS

#Masa_OS: Masa de semilla molida seca a procesar (kg)
#ppPF   : Porcentaje de peso con polifenoles activos en semillas (mgPF/gSS)
#te     : Tiempo designado para extracción (min)
#Temp_e : Temperatura de extraccion (C°) (Opciones: 40, 60, 100 C°)
#ID_sol : Tipo de solvente utilizado    (1) Etanol-Agua pH 2.0, 50% w/w
#                                       (2) Etanol-Agua 50% w/w

#ID_Proceso : Tipo de proceso realizado (1) Pressurized Liquid Extracion (PLE)
#                                       (2) PLE + Ultrasonido (Incluir Input Nivel Amplitud (A) y relacion Soluto/Solvente (LP)) (IGNORAR ESTE)


#OUTPUTS

#R     : % de Polifenoles recuperados en extracto crudo posterior a la extraccion
#ts    : Tiempo total necesario para realizar el tiempo de secado

def Bloque_Extraccion(ID_Proceso,Masa_OS,te,Temp_e,ppPF,ID_sol):
    
    # Modelo de extraccion PLE: Parametros de la extraccion

    y_inf   = ppPF; #(mgPF/gSS)
    
    if ID_Proceso == 1:
                if ID_sol == 1:
                    
                    if Temp_e >= 100:
                        f = 0.50; k1 = 0.41; k2 = 0.02;
                    else:
                        f = 0.39; k1 = 0.36; k2 = 0.01;
                    
                elif ID_sol == 2:
                        f = 0.52; k1 = 0.02; k2 = 0.19;
            
    
    #Modelo de extraccion
    def y(t):
        return y_inf*(1-f*np.exp(-k1*t)-(1-f)*np.exp(-k2*t)); #(g_PFTotales/g_solidoseco)
            
    R       = y(te)/y_inf*100;
    Masa_PF = (Masa_OS*1000)*ppPF*R/100; 
    
    return [R, Masa_PF]
               
def Proceso_Global(Masa_0,Hum_0,T_Sec,PTOS,PPFS,te,Temp_e,ID_Sol,ID_Proceso,PP_Tan,PP_AF,PP_FLA):
    
    #Proceso de secado

    #Masa_0: Masa de orujo humedo a procesar (kg)                                   (USUARIO) 		[0-1000]
    #Hum_0 : Humedad inicial (%)			                                (MACHINE LEARNING)	[0.5-0.9]
    #T_sec : Temperatura de secado (K°)                                             (USUARIO)		[60-85] (Pasos de a 5)
    #PTOS  : Polifenoles totales orujo seco (%)		 			(MACHINE LEARNING)	[0.001-0.02]
    #PPFS  : Porcentaje peso PT en semillas (%)                                     (MACHINE LEARNING)      [0.6-0.9]   
    
    
    #Proceso de extraccion (ID_proceso solo funciona 1 por el momento)
    	
    #te     : Tiempo designado para extracción (min)                                (USUARIO) 		[10-240]
    #Temp_e : Temperatura de extraccion (C°) (Opciones: 40, 60, 100 C°)             (USUARIO) 		[40-100] (Pasos de a 20)
    #ID_sol : Tipo de solvente utilizado    (1) Etanol-Agua pH 2.0, 50% w/w         (USUARIO)		[1 o 2]
    #                                       (2) Etanol-Agua 50% w/w
    
    #ID_Proceso : Tipo de proceso realizado (1) Pressurized Liquid Extracion (PLE)  (USUARIO)		[Solo 1 por el momento]
    #                                       (2) PLE + Ultrasonido (Incluir Input Nivel Amplitud (A) y relacion Soluto/Solvente (LP))
    
    
    #Otros
    
    #PP_Tan : Porcentaje de los polifenoles totales asociados a taninos (%)          (MACHINE LEARNING)	[0.35-0.6]
    #PP_AF  : Porcentaje de los polifenoles totales asociados a acidos-fenolicos (%) (MACHINE LEARNING)	[0.1-0.25]
    #PP_FLA : Porcentaje de los polifenoles totales asociados a flavanoides (%)      (MACHINE LEARNING)	[0.05-0.2]
    
    [Rs,ts,Peso_a_perder] = Bloque_Secado(Masa_0,Hum_0,T_Sec,PTOS,PPFS)
    Masa_OS = Masa_0 - Peso_a_perder;
        #Salidas
            #Rs: % del polifenol retenido posterior al secado (entre 0 y 1)
            #ts: tiempo total para llegar al 5% de humedad
    
    ppPF          = Rs*PPFS;
    [Re, Masa_PF] = Bloque_Extraccion(ID_Proceso,Masa_OS,te,Temp_e,ppPF,ID_Sol)
        #Salidas
            #Rs: % del polifenol retenido posterior al secado (entre 0 y 1)
            #ts: tiempo total para llegar al 5% de humedad 
            
    return [Rs,ts,Re,Masa_PF]

#Asignacion de valores (Ejemplificado)

#MACHINE LEARNING 
Hum_0   = 0.6       #Vector_ML[0];
PTOS    = 0.004     #Vector_ML[1]; 
PPFS    = 0.7       #Vector_ML[2];
PP_Tan  = 0.5       #Vector_ML[3];
PP_AF   = 0.3       #Vector_ML[4];
PP_FLA  = 0.2       #Vector_ML[5];

#USUARIO
Masa_0  = 100       #Vector_INPUT[0];
T_Sec   = 70+273    #Vector_INPUT[1];
te      = 80        #Vector_INPUT[2];
Temp_e  = 80        #Vector_INPUT[3];
ID_Sol  = 1         #Vector_INPUT[4];
ID_Proceso = 1;

[Rs,ts,Re,Masa_PF] = Proceso_Global(Masa_0,Hum_0,T_Sec,PTOS,PPFS,te,Temp_e,ID_Sol,ID_Proceso,PP_Tan,PP_AF,PP_FLA)