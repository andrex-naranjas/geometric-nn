import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pythia8
import math


pythia=pythia8.Pythia()
pythia.readString("Higgs:useBSM = on")
pythia.readString("HiggsBSM:ffbar2A3 = on")
pythia.readString("36:onMode = off")
pythia.readString("36:onIfMatch = 25 23")
#pythia.readString("Beams:eCM = 100000")

#decaimientos Z
pythia.readString("23:onMode = off")
pythia.readString("23:onIfMatch = 11 -11") #electron
pythia.readString("23:onIfMatch = 13 -13") #muon
#pythia.readString("PartonLevel:MPI = off")
#pythia.readString("23:onIfMatch = 15 -15") #tau

#decaimientos Higgs
pythia.readString("25:onMode = off")
pythia.readString("25:onIfMatch= 5 -5") # quark b

pythia.init()

#definir listas donde guardar los datos
electron_px = []
electron_py = []
electron_pz = [] 
electron_e = []
positron_px  = []
positron_py = []
positron_pz = []
positron_e = []
muon_px = []
muon_py = []
muon_pz = []
muon_e = []
antimuon_px = []
antimuon_py = []
antimuon_pz = []
antimuon_e = []

#listas para b quarks asociados a electrones
bquark_px_e = []
bquark_py_e = []
bquark_pz_e = []
bquark_e_e = []
anti_bquark_px_e = []
anti_bquark_py_e = []
anti_bquark_pz_e = []
anti_bquark_e_e = []

#listas para b quarks asociados a muones
bquark_px_m = []
bquark_py_m = []
bquark_pz_m = []
bquark_e_m = []
anti_bquark_px_m = []
anti_bquark_py_m = []
anti_bquark_pz_m = []
anti_bquark_e_m = []

Z_px = []
Z_px_m = []
cos_theta_hist =[]
eta=[]
contador=0

num_events=10000
for j in range(num_events):
    
    if not pythia.next():
        continue
    
    evento = pythia.event
    for particle in evento:
        if particle.id() == 111:
          #  cos_theta = (particle.pz())/np.sqrt((pow(particle.px(),2) + pow(particle.py(),2) + pow(particle.pz(),2)))
            cos_theta_hist.append(particle.phi())
            
        mother=particle.mother1()
        madre_id=evento[mother].id()
        mother2=particle.mother2()
        madre2_id=evento[mother2].id()
        
        if particle.id() == 36: #A0
            
            daughter1_A0 = evento[particle.daughter1()]
            daughter2_A0 = evento[particle.daughter2()]
            
            if (abs(daughter1_A0.id()) == 23 or abs(daughter1_A0.id()) == 25) and (abs(daughter2_A0.id()) == 23 or abs(daughter2_A0.id()) == 25) :
                #acomodar H y Z
                
                if abs(daughter1_A0.id())==23:
                    Zboson_A0 = daughter1_A0
                elif abs(daughter2_A0.id())==23:
                    Zboson_A0 = daughter2_A0
                    
                if abs(daughter1_A0.id())==25:
                    Hboson_A0 = daughter1_A0
                elif abs(daughter2_A0.id())==25:
                    Hboson_A0 = daughter2_A0
                
                #Hijas de Higgs
                daughter1_H = evento[Hboson_A0.daughter1()]
                daughter2_H = evento[Hboson_A0.daughter2()]
                Hdecay_ok = False
                if abs(daughter1_H.id()) == 5 and abs(daughter2_H.id()) == 5:
                    Hdecay_ok = True
                #Hijas de Z
                daughter1_Z = evento[Zboson_A0.daughter1()]
                daughter2_Z = evento[Zboson_A0.daughter2()]
                
                if abs(daughter1_Z.id()) == 11 and abs(daughter2_Z.id()) == 11 and Hdecay_ok:
                   #acomodar electrones
                    if daughter1_Z.id()==11:
                        electron = daughter1_Z
                    elif daughter2_Z.id()==11:
                        electron = daughter2_Z
                    
                    if daughter1_Z.id()==-11:
                        positron = daughter1_Z
                    elif daughter2_Z.id()==-11:
                        positron = daughter2_Z
                    
                    #bottoms para electrones
                    if daughter1_H.id()==5:
                        quark_b = daughter1_H
                    elif daughter2_H.id()==5:
                        quark_b= daughter2_H
                    
                    if daughter1_H.id()==-5:
                        antiquark_b = daughter1_H
                    elif daughter2_H.id()==-5:
                        antiquark_b = daughter2_H
                       
                    # variables cinemáticas electrón
                    electron_px.append(electron.px())
                    electron_py.append(electron.py())
                    electron_pz.append(electron.pz())
                    electron_e.append(electron.e())
                    
                    # variables cinemáticas positrón  
                    positron_px.append(positron.px())
                    positron_py.append(positron.py())
                    positron_pz.append(positron.pz())
                    positron_e.append(positron.e())
                    
                    # variables cinemáticas de los b quarks asociados a electrones
                    bquark_px_e.append(quark_b.px())
                    bquark_py_e.append(quark_b.py()) 
                    bquark_pz_e.append(quark_b.pz()) 
                    bquark_e_e.append(quark_b.e())
                    
                    # variables cinemáticas de los anti b quarks asociados a electrones
                    anti_bquark_px_e.append(antiquark_b.px()) 
                    anti_bquark_py_e.append(antiquark_b.py()) 
                    anti_bquark_pz_e.append(antiquark_b.pz())
                    anti_bquark_e_e.append(antiquark_b.e())
                    
                    #cos_theta = (Zboson_A0.pz())/(Zboson_A0.px() + Zboson_A0.py() + Zboson_A0.pz())
                    #cos_theta = (Zboson_A0.pz())/np.sqrt((pow(Zboson_A0.px(),2) + pow(Zboson_A0.py(),2) + pow(Zboson_A0.pz(),2)))
                    #cos_theta_hist.append(particle.eta())
                    #cos_theta_hist.append(cos_theta)
                
                    Z_px.append(Zboson_A0.px())
                    
                 
                    
                    #Hmass_squared = pow(quark_b.e() + antiquark_b.e(), 2) - pow(quark_b.px() + antiquark_b.px(), 2) - pow(quark_b.py() + antiquark_b.py(), 2) - pow(quark_b.pz() + antiquark_b.pz(), 2)
                    #Hmass = math.sqrt(Hmass_squared)
                    
                if abs(daughter1_Z.id()) == 13 and abs(daughter2_Z.id()) == 13 and Hdecay_ok:
                   #acomodar muones
                    if daughter1_Z.id()==13:
                        muon = daughter1_Z
                    elif daughter2_Z.id()==13:
                        muon= daughter2_Z
                    
                    if daughter1_Z.id()==-13:
                        antimuon = daughter1_Z
                    elif daughter2_Z.id()==-13:
                        antimuon = daughter2_Z
                    
                    #bottoms para muones
                    if daughter1_H.id()==5:
                        quark_b = daughter1_H
                    elif daughter2_H.id()==5:
                        quark_b= daughter2_H
                    
                    if daughter1_H.id()==-5:
                        antiquark_b = daughter1_H
                    elif daughter2_H.id()==-5:
                        antiquark_b = daughter2_H
                        
                    #variables cinemáticas muon  
                    muon_px.append(muon.px())
                    muon_py.append(muon.py())
                    muon_pz.append(muon.pz())
                    muon_e.append(muon.e())
                    
                    #variables cinemáticas antimuon
                    antimuon_px.append(antimuon.px())
                    antimuon_py.append(antimuon.py())
                    antimuon_pz.append(antimuon.pz())
                    antimuon_e.append(antimuon.e())
                    
                    #variables cinemáticas de los b quarks asociados a muones
                    bquark_px_m.append(quark_b.px())
                    bquark_py_m.append(quark_b.py()) 
                    bquark_pz_m.append(quark_b.pz()) 
                    bquark_e_m.append(quark_b.e())
                    
                     #variables cinemáticas de los anti b quarks asociados a muones
                    anti_bquark_px_m.append(antiquark_b.px()) 
                    anti_bquark_py_m.append(antiquark_b.py()) 
                    anti_bquark_pz_m.append(antiquark_b.pz())
                    anti_bquark_e_m.append(antiquark_b.e())
                    contador+=1
                    Z_px_m.append(Zboson_A0.px())

print(len(anti_bquark_px_e), len(anti_bquark_px_m))
# Crear DataFrame para electrones
df_electrones = pd.DataFrame({
    'electron_px': electron_px,
    'electron_py': electron_py,
    'electron_pz': electron_pz,
    'electron_e': electron_e,
    'positron_px': positron_px,
    'positron_py': positron_py,
    'positron_pz': positron_pz,
    'positron_e': positron_e,
    'bquark_px_e': bquark_px_e,
    'bquark_py_e': bquark_py_e,
    'bquark_pz_e': bquark_pz_e,
    'bquark_e_e': bquark_e_e,
    'anti_bquark_px_e': anti_bquark_px_e,
    'anti_bquark_py_e': anti_bquark_py_e,
    'anti_bquark_pz_e': anti_bquark_pz_e,
    'anti_bquark_e_e': anti_bquark_e_e,
    'Z_boson_px_e': Z_px
})
# Guardar como CSV
df_electrones.to_csv('dataframe_electrones.csv', index=False) 
        

# Crear DataFrame para muones
df_muones = pd.DataFrame({
    'muon_px': muon_px,
    'muon_py': muon_py,
    'muon_pz': muon_pz,
    'muon_e': muon_e,
    'antimuon_px': antimuon_px,
    'antimuon_py': antimuon_py,
    'antimuon_pz': antimuon_pz,
    'antimuon_e': antimuon_e,
    'bquark_px_m': bquark_px_m,
    'bquark_py_m': bquark_py_m,
    'bquark_pz_m': bquark_pz_m,
    'bquark_e_m': bquark_e_m,
    'anti_bquark_px_m': anti_bquark_px_m,
    'anti_bquark_py_m': anti_bquark_py_m,
    'anti_bquark_pz_m': anti_bquark_pz_m,
    'anti_bquark_e_m': anti_bquark_e_m,
    'Z_boson_px_m': Z_px_m
})

# Guardar como CSV
df_muones.to_csv('dataframe_muones.csv', index=False)

plt.hist(cos_theta_hist, bins=50, range=(-1, 1), alpha=0.75, color='green', label='pi0 Cos theta')
plt.xlabel("pi0 Cos theta")
plt.ylabel("Events")
plt.title("Cosine of theta of pi0")
plt.legend()
plt.show()
