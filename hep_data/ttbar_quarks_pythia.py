import pandas as pd
import pythia8

pythia_ttbar=pythia8.Pythia()
pythia_ttbar.readString("Top:gg2ttbar= on")
pythia_ttbar.readString("6:onMode = off")
pythia_ttbar.readString("6:onIfMatch = 24 5")
pythia_ttbar.readString("-6:onMode = off")
pythia_ttbar.readString("-6:onIfMatch = -24 -5")
pythia_ttbar.readString("24:onMode = off")
pythia_ttbar.readString("24:onIfMatch = -11 12")
pythia_ttbar.readString("24:onIfMatch = -13 14")
pythia_ttbar.readString("-24:onMode = off")
pythia_ttbar.readString("-24:onIfMatch = 11 -12")
pythia_ttbar.readString("-24:onIfMatch = 13 -14")
pythia_ttbar.init()

#**************LISTAS PARA BACKGROUND****************
electron_px_bg = []
electron_py_bg = []
electron_pz_bg = [] 
electron_e_bg = []
positron_px_bg  = []
positron_py_bg = []
positron_pz_bg = []
positron_e_bg = []
muon_px_bg = []
muon_py_bg = []
muon_pz_bg = []
muon_e_bg = []
antimuon_px_bg = []
antimuon_py_bg = []
antimuon_pz_bg = []
antimuon_e_bg = []

#listas para b quarks asociados a electrones
bquark_px_e_bg = []
bquark_py_e_bg = []
bquark_pz_e_bg = []
bquark_e_e_bg = []
anti_bquark_px_e_bg = []
anti_bquark_py_e_bg = []
anti_bquark_pz_e_bg = []
anti_bquark_e_e_bg = []

#listas para b quarks asociados a muones
bquark_px_m_bg = []
bquark_py_m_bg = []
bquark_pz_m_bg = []
bquark_e_m_bg = []
anti_bquark_px_m_bg = []
anti_bquark_py_m_bg = []
anti_bquark_pz_m_bg = []
anti_bquark_e_m_bg = []

#Listas para histogramas
masasZ_background_hist = []
masasH_background_hist = []
masasA0_background_hist = []

contador=0
contador_electrones=0
contador_positrones=0
num_events=5000

def find_ancestor(event, particle_idx, particle_id,target_id):
    current_idx = particle_idx
    while True:  # Continúa retrocediendo mientras haya madres
        mother1 = event[current_idx].mother1()  # posicion de la madre principal
        mother_particle = event[mother1]
        
        if mother_particle.id() == target_id:  # Comparamos el ID
            return True, mother1  # Encontramos el ancestro buscado
        
        if event[mother1].id() != particle_id:
            #print(event[mother1].id())
            return False
        
        # Seguimos retrocediendo en la madre
        current_idx = mother1 


for i in range(num_events):
    if not pythia_ttbar.next():
        continue
    
    evento = pythia_ttbar.event
    
    leptones_minus_W_encontrados = []
    leptones_plus_W_encontrados = []
    
    posicion_ancestro_lm = []
    posicion_ancestro_lp= []
    
    bquark_encontrados = []
    antibquark_encontrados = []
 
    
    for particle in evento:
        madre1_posicion = particle.mother1() #POSICION EN VECTOR DE EVENTOS
        madre1_particula = evento[madre1_posicion] #TIPO PARTÍCULA
        madre1_id = madre1_particula.id() #ID
        
        madre2_posicion = particle.mother2() #POSICION EN VECTOR DE EVENTOS
        madre2_particula = evento[madre2_posicion] #TIPO PARTÍCULA
        madre2_id = madre2_particula.id() #ID
        
        abuela_id = evento[madre1_particula.mother1()].id()
        
        if particle.id() == 11 and madre1_id==-24:
            hijo_top, madre = find_ancestor(evento, madre1_posicion, -24,-6)
            if hijo_top: 
                leptones_minus_W_encontrados.append(particle)
                posicion_ancestro_lm.append(madre)
        
        if particle.id() == -11 and madre1_id == 24:
            hijo_top, madre = find_ancestor(evento, madre1_posicion, 24,6)
            if hijo_top:
                leptones_plus_W_encontrados.append(particle)
                posicion_ancestro_lp.append(madre)
        
        if particle.id() == 13 and madre1_id==-24:
            hijo_top, madre = find_ancestor(evento, madre1_posicion, -24,-6)
            if hijo_top:
                leptones_minus_W_encontrados.append(particle)
                posicion_ancestro_lm.append(madre)
        
        if particle.id() == -13 and madre1_id==24:            
            hijo_top, madre = find_ancestor(evento, madre1_posicion, 24,6)
            if hijo_top:
                leptones_plus_W_encontrados.append(particle)
                posicion_ancestro_lp.append(madre)
                
        if particle.id() == 5 and madre1_id==6:
            bquark_encontrados.append(particle)
        
        if particle.id() == -5 and madre1_id == -6:
            antibquark_encontrados.append(particle)
        
    if len(leptones_minus_W_encontrados)>1 or len(leptones_plus_W_encontrados) >1 or len(bquark_encontrados) >1 or len(antibquark_encontrados)>1:
        print('EVENTO EXTRAÑO ENCONTRADO')
    elif len(leptones_minus_W_encontrados)==1 and len(leptones_plus_W_encontrados)==1 and len(bquark_encontrados)==1 and len(antibquark_encontrados) == 1:
        lepton_plus = leptones_plus_W_encontrados[0]
        lepton_minus = leptones_minus_W_encontrados[0] #TIPO PARTÍCULAS, NO POSICIÓN
        bquark = bquark_encontrados[0]
        antibquark = antibquark_encontrados[0]
        
        posi_abuela_lp = posicion_ancestro_lp[0]
        posi_abuela_lm = posicion_ancestro_lm[0]
        
        if posi_abuela_lm == antibquark.mother1() and posi_abuela_lp == bquark.mother1():
            contador+=1
            if lepton_plus.id()==-11 and lepton_minus.id()==11:
            # variables cinemáticas electrón
                electron_px_bg.append(lepton_minus.px())
                electron_py_bg.append(lepton_minus.py())
                electron_pz_bg.append(lepton_minus.pz())
                electron_e_bg.append(lepton_minus.e())
                
                #variables cinemáticas positrón   
                positron_px_bg.append(lepton_plus.px())
                positron_py_bg.append(lepton_plus.py())
                positron_pz_bg.append(lepton_plus.pz())
                positron_e_bg.append(lepton_plus.e())
                
                #variables cinemáticas de los b quarks asociados a electrones
                bquark_px_e_bg.append(bquark.px())
                bquark_py_e_bg.append(bquark.py()) 
                bquark_pz_e_bg.append(bquark.pz()) 
                bquark_e_e_bg.append(bquark.e())
                
                    #variables cinemáticas de los anti b quarks asociados a electrones
                anti_bquark_px_e_bg.append(antibquark.px()) 
                anti_bquark_py_e_bg.append(antibquark.py()) 
                anti_bquark_pz_e_bg.append(antibquark.pz())
                anti_bquark_e_e_bg.append(antibquark.e())
                
                masasH_background_hist.append((bquark.p() + antibquark.p()).mCalc())
                masasZ_background_hist.append((lepton_plus.p() + lepton_minus.p()).mCalc())
                masasA0_background_hist.append((bquark.p() + antibquark.p() + lepton_plus.p() + lepton_minus.p()).mCalc())
            
            if lepton_plus.id()==-13 and lepton_minus.id()==13:
            #variables cinemáticas muon  
                muon_px_bg.append(lepton_minus.px())
                muon_py_bg.append(lepton_minus.py())
                muon_pz_bg.append(lepton_minus.pz())
                muon_e_bg.append(lepton_minus.e())
                
                #variables cinemáticas antimuon
                antimuon_px_bg.append(lepton_plus.px())
                antimuon_py_bg.append(lepton_plus.py())
                antimuon_pz_bg.append(lepton_plus.pz())
                antimuon_e_bg.append(lepton_plus.e())
                
                #variables cinemáticas de los b quarks asociados a muones
                bquark_px_m_bg.append(bquark.px())
                bquark_py_m_bg.append(bquark.py()) 
                bquark_pz_m_bg.append(bquark.pz()) 
                bquark_e_m_bg.append(bquark.e())
                
                    #variables cinemáticas de los anti b quarks asociados a muones
                anti_bquark_px_m_bg.append(antibquark.px()) 
                anti_bquark_py_m_bg.append(antibquark.py()) 
                anti_bquark_pz_m_bg.append(antibquark.pz())
                anti_bquark_e_m_bg.append(antibquark.e())
                
                masasH_background_hist.append((bquark.p() + antibquark.p()).mCalc())
                masasZ_background_hist.append((lepton_plus.p() + lepton_minus.p()).mCalc())
                masasA0_background_hist.append((bquark.p() + antibquark.p() + lepton_plus.p() + lepton_minus.p()).mCalc())
            
    else:
        print('ALGUNA LISTA VACÍA')
        
print("Grupos de partículas bbll por evento:", contador)
print("Grupos que además cumplen bbee, o bbmm", len(masasH_background_hist))


df_electrones_bg = pd.DataFrame({
    'electron_px_bg': electron_px_bg,
    'electron_py_bg': electron_py_bg,
    'electron_pz_bg': electron_pz_bg,
    'electron_e_bg': electron_e_bg,
    'positron_px_bg': positron_px_bg,
    'positron_py_bg': positron_py_bg,
    'positron_pz_bg': positron_pz_bg,
    'positron_e_bg': positron_e_bg,
    'bquark_px_e_bg': bquark_px_e_bg,
    'bquark_py_e_bg': bquark_py_e_bg,
    'bquark_pz_e_bg': bquark_pz_e_bg,
    'bquark_e_e_bg': bquark_e_e_bg,
    'anti_bquark_px_e_bg': anti_bquark_px_e_bg,
    'anti_bquark_py_e_bg': anti_bquark_py_e_bg,
    'anti_bquark_pz_e_bg': anti_bquark_pz_e_bg,
    'anti_bquark_e_e_bg': anti_bquark_e_e_bg,
})
# Guardar como CSV
df_electrones_bg.to_csv('data_electrons_bg_ttbar.csv', index=False) 

df_muones_bg = pd.DataFrame({
    'muon_px_bg': muon_px_bg,
    'muon_py_bg': muon_py_bg,
    'muon_pz_bg': muon_pz_bg,
    'muon_e_bg': muon_e_bg,
    'antimuon_px_bg': antimuon_px_bg,
    'antimuon_py_bg': antimuon_py_bg,
    'antimuon_pz_bg': antimuon_pz_bg,
    'antimuon_e_bg': antimuon_e_bg,
    'bquark_px_m_bg': bquark_px_m_bg,
    'bquark_py_m_bg': bquark_py_m_bg,
    'bquark_pz_m_bg': bquark_pz_m_bg,
    'bquark_e_m_bg': bquark_e_m_bg,
    'anti_bquark_px_m_bg': anti_bquark_px_m_bg,
    'anti_bquark_py_m_bg': anti_bquark_py_m_bg,
    'anti_bquark_pz_m_bg': anti_bquark_pz_m_bg,
    'anti_bquark_e_m_bg': anti_bquark_e_m_bg,
})

df_muones_bg.to_csv('data_muons_bg_ttbar.csv', index=False)