import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy
from derivative import dxdt
class instronReader():
    def __init__(self):
        self.strain_colname = 'Strain DIC (%)'
        self.stress_colname = 'Axial Stress (MPa)'
        self.loop_name = 'Loop 1' #name for the variable counting cycles
        self.stepUp = 5 # number of the step fo loading
        self.stepDown = 6 #number of the step for unloading
        self.dSdELim = 50 #[MPa/%] relative allowed margin of the stress plateau increase per % of deformation, absolute alowed increase of the stress plateau per 1% strain is min(dSdE[dSdE>0])+ self.dSdELim 
        self.dSdEAbsLim=100 #[MPa/%] aboslute allowed stress plateau increase per 1 %
        self.minstress = None #if set to a value, unrecovered strain upon unloading is taken at this stress 
        self.dE = 0.2 #plastic strain increment for yield stress identification
        #parameters of the savitzky_golay filter used for the calculation of the derivative 
        self.kind="savitzky_golay"
        self.left=0.2
        self.right=0.2
        self.order=1
        self.first=1
        self.iwindow = False


    def setAttributes(self,**kwargs):    
        self.__dict__.update(kwargs)
    def read(self,filename):
        self.df = pd.read_csv(filename)
        self.filename = filename
        try:
            self.cycles= self.df.groupby(self.loop_name)
            self.ncycles = self.cycles.ngroups
        except:
            pass
    def readMercury(self,filename):
        self.filename = filename
        self.strain_colname = 'A1 - Length: Extension [%]'
        self.stress_colname = 'A1 - Stress [MPa]'
        self.df = pd.read_csv(filename,sep=";")
    def getRupture(self):
        strain = self.df[self.strain_colname].to_numpy()
        stress = self.df[self.stress_colname].to_numpy()
        idx = np.nanargmax(np.abs(strain))
        strain=strain[:idx]
        stress=stress[:idx]

        var = stress
        idxs = np.isnan(var)==False
        
        strain = strain[idxs]
        stress = stress[idxs]

        var = strain
        idxs = np.isnan(var)==False
        
        strain = strain[idxs]
        stress = stress[idxs]
        strength = stress.max()
        strengthidx = stress.argmax()
        ductility = strain.max()
        ductilityidx = strain.argmax()
        
        true_strain = np.log(1+strain/100)*100
        true_stress = stress*(1+strain/100)

        true_strength = true_stress.max()
        true_strengthidx = true_stress.argmax()
        true_ductility = true_strain.max()
        true_ductilityidx = true_strain.argmax()
        
        dSdE = dxdt(stress,strain , kind=self.kind, left=self.left, right=self.right, order=self.order,iwindow=self.iwindow)
        dSdE2 = dxdt(dSdE,strain , kind=self.kind, left=self.left, right=self.right, order=self.order,iwindow=self.iwindow)
        

        true_dSdE = dxdt(true_stress,true_strain , kind=self.kind, left=self.left, right=self.right, order=self.order,iwindow=self.iwindow)
        true_dSdE2 = dxdt(true_dSdE,true_strain , kind=self.kind, left=self.left, right=self.right, order=self.order,iwindow=self.iwindow)
        
        idx = dSdE2.argmax()
        idx = idx + dSdE[idx:-1].argmax()

        
        inflectionidx = idx
        stressLin = dSdE[idx]*(strain-strain[idx])+stress[idx]
        strainLin=strain+self.dE
        idxs = np.where((stressLin>stress.min())*(stressLin<stress.max()))
        stressLin=stressLin[idxs]
        strainLin=strainLin[idxs]
        
        stress2=np.interp(strainLin,strain,stress)
        Diff4lim = stressLin-stress2
        idx = np.where(np.abs(Diff4lim)==min(np.abs(Diff4lim)))[0][0]
        Diff4lim = stress-stress2[idx]
        idx = np.where(np.abs(Diff4lim)==min(np.abs(Diff4lim)))[0][0]
        yieldstress = stress[idx]
        strainatyieldstress = strain[idx]
        yieldidx = idx
        
       


        idx = true_dSdE2.argmax()
        idx = idx + true_dSdE[idx:-1].argmax()
        Diff4lim = abs(true_strain[idx:-1]-true_strain[idx])-self.dE*0
        idx2 = idx+np.where(np.abs(Diff4lim)==min(np.abs(Diff4lim)))[0][0]

        true_yieldstress = true_stress[idx2]
        true_strainatyieldstress = true_strain[idx2]
        true_yieldidx = idx2



        idx = true_dSdE2.argmax()
        idx = idx + true_dSdE[idx:-1].argmax()
        
        true_inflectionidx = idx
        true_stressLin = true_dSdE[idx]*(true_strain-true_strain[idx])+true_stress[idx]
        true_strainLin=true_strain+self.dE
        idxs = np.where((true_stressLin>true_stress.min())*(true_stressLin<true_stress.max()))
        true_stressLin=true_stressLin[idxs]
        true_strainLin=true_strainLin[idxs]
        
        stress2=np.interp(true_strainLin,true_strain,true_stress)
        Diff4lim = true_stressLin-stress2
        idx = np.where(np.abs(Diff4lim)==min(np.abs(Diff4lim)))[0][0]
        Diff4lim = true_stress-stress2[idx]
        idx = np.where(np.abs(Diff4lim)==min(np.abs(Diff4lim)))[0][0]
        true_yieldstress = true_stress[idx]
        true_strainatyieldstress = true_strain[idx]
        true_yieldidx = idx


        #plateau identification for loading - only engineering strain and stress...
        if True:            
            dSdEUp=dSdE[0:yieldidx]
            idxmin = np.argmin(dSdEUp)
            self.dSdELimUp = dSdEUp[dSdEUp>0].min()+self.dSdELim
            FMTidxs=[]
            FMT=[]
            for Sel,shift in zip([range(0,idxmin),range(idxmin,dSdEUp.shape[0])],[0,idxmin]):
                Diff4lim = dSdEUp[Sel]-self.dSdELimUp
                if (Diff4lim<0).all():
                    FMTidxs.append(dSdEUp.shape[0]-1)
                else:
                    FMTidxs.append(np.where(np.abs(Diff4lim)==min(np.abs(Diff4lim)))[0][0]+shift)
                FMT.append([strain[FMTidxs[-1]],stress[FMTidxs[-1]]])
            midstrain = np.mean([fmt[0] for fmt in FMT])
            Diff4lim = strain-midstrain
            idx = np.where(np.abs(Diff4lim)==min(np.abs(Diff4lim)))[0][0]
            length=strain[FMTidxs[1]]-strain[FMTidxs[0]]
            FMTplateau = {'length':length,'midstrain':strain[idx],'midstress':stress[idx],'meandSdE':dSdE[FMTidxs[0]:FMTidxs[1]].mean(), 
                         'Idx Plateau start':FMTidxs[0],'Idx Plateau end':FMTidxs[1],'Stress Plateau start':stress[FMTidxs[0]],'Stress Plateau end':stress[FMTidxs[1]],
                         'Strain Plateau start':strain[FMTidxs[0]],'Strain Plateau end':strain[FMTidxs[1]]}
            dSdEUp=true_dSdE[0:true_yieldidx]
            idxmin = np.argmin(dSdEUp)
            self.dSdELimUp = dSdEUp[dSdEUp>0].min()+self.dSdELim
            FMTidxs=[]
            FMT=[]
            for Sel,shift in zip([range(0,idxmin),range(idxmin,dSdEUp.shape[0])],[0,idxmin]):
                Diff4lim = dSdEUp[Sel]-self.dSdELimUp
                if (Diff4lim<0).all():
                    FMTidxs.append(dSdEUp.shape[0]-1)
                else:
                    FMTidxs.append(np.where(np.abs(Diff4lim)==min(np.abs(Diff4lim)))[0][0]+shift)
                FMT.append([true_strain[FMTidxs[-1]],true_stress[FMTidxs[-1]]])
            midstrain = np.mean([fmt[0] for fmt in FMT])
            Diff4lim = true_strain-midstrain
            idx = np.where(np.abs(Diff4lim)==min(np.abs(Diff4lim)))[0][0]
            length=true_strain[FMTidxs[1]]-true_strain[FMTidxs[0]]
            true_FMTplateau = {'length':length,'midstrain':true_strain[idx],'midstress':true_stress[idx],'meandSdE':true_dSdE[FMTidxs[0]:FMTidxs[1]].mean(), 
                         'Idx Plateau start':FMTidxs[0],'Idx Plateau end':FMTidxs[1],'Stress Plateau start':true_stress[FMTidxs[0]],'Stress Plateau end':true_stress[FMTidxs[1]],
                         'Strain Plateau start':true_strain[FMTidxs[0]],'Strain Plateau end':true_strain[FMTidxs[1]]}

        titles='strength strengthidx ductility ductilityidx dSdE dSdE2 dSdE dSdE2 strain stress strain stress yieldstress strainatyieldstress yieldidx inflectionidx stressLin strainLin'.split()
        rupdata={}
        rupdata['eng']={}
        rupdata['true']={}
        for title in titles:
            exec(f"rupdata['eng']['{title}'] = {title}")
            exec(f"rupdata['true']['{title}'] = {f'true_{title}'}")
        rupdata['eng']['FMTplateau']=FMTplateau
        rupdata['true']['FMTplateau']=true_FMTplateau
        return rupdata
    def getAllCycles(self):
        cdata = []
        for cn in list(range(self.first,self.ncycles)):
            cdata.append(self.getCycle(cn))

        valdict = {}
        valdict['complete'] = 'unrecoveredStrain unrecoveredStrainInc'.split()
        valdict['MT'] = 'hystwidth hyst'.split()
        cycdata={}
        cycdata['cycle #'] = list(range(self.first,self.ncycles))
        titles='unrecoveredStrain unrecoveredStrainInc hystwidth hyst length midstrain midstress meandSdE'
        for title in titles.split():
            rec=False
            for key in valdict.keys():
                if title in valdict[key]:
                    cycdata[title]=[]
                    rec=True
            if not rec:
                for key1,ext in zip(['FMT','RMT'],['Up','Down']):
                    cycdata[f'{title}{ext}'] = []
                    cycdata[f'{title}'] = []
        for cdat in cdata:
            for title in titles.split():
                rec=False
                for key in valdict.keys():
                    if title in valdict[key]:
                        cycdata[title].append(float(cdat[key][title]))
                        rec=True
                if not rec:
                    for key1,ext in zip(['FMT','RMT'],['Up','Down']):
                        cycdata[f'{title}{ext}'].append(cdat[key1]['plateau'][title])
                    cycdata[f'{title}'].append(np.mean([cycdata[f'{title}Up'][-1],cycdata[f'{title}Down'][-1]]))
        for key in cycdata.keys():
            cycdata[key]=np.array(cycdata[key])
        return cdata,cycdata
    def applyTreshold(self,cycdata,dSdEAbsLim=None):
        if dSdEAbsLim is None:
            dSdEAbsLim=self.dSdEAbsLim
        idxs = (cycdata['meandSdEUp']>dSdEAbsLim)+(cycdata['meandSdEDown']>dSdEAbsLim)
        cycdata2 = copy.deepcopy(cycdata)
        for key in cycdata.keys():    
            if key !='cycle #':
                cycdata2[key][idxs]=np.nan
        return cycdata2
    def getCycle(self,cn):
        cycle = self.cycles.get_group(cn)
        strain = cycle[[self.strain_colname]].to_numpy()[:,0]
        stress = cycle[[self.stress_colname]].to_numpy()[:,0]
        strainUp = cycle[cycle["Step"]==self.stepUp][[self.strain_colname]].to_numpy()[:,0]
        stressUp = cycle[cycle["Step"]==self.stepUp][[self.stress_colname]].to_numpy()[:,0] 
        strainDown = cycle[cycle["Step"]==self.stepDown][[self.strain_colname]].to_numpy()[:,0]
        stressDown = cycle[cycle["Step"]==self.stepDown][[self.stress_colname]].to_numpy()[:,0]

        #urecovered strains taken for both loading and unloading: unloaded-loaded= increment of the unrecovered strain
        if self.minstress is None:
            unrecoveredStrainUp = strainUp.min()
            unrecoveredStrainDown = strainDown.min()
        else:
            Diff4lim = stressUp-self.minstress
            idx = np.where(np.abs(Diff4lim)==min(np.abs(Diff4lim)))[0][0]
            unrecoveredStrainUp = strainUp[idx]
            Diff4lim = stressDown-self.minstress
            idx = np.where(np.abs(Diff4lim)==min(np.abs(Diff4lim)))[0][0]
            unrecoveredStrainDown = strainDown[idx]
        unrecoveredStrain = unrecoveredStrainDown
        #plateau identification for loading
        dSdEUp = self.deriv(stressUp,strainUp)
        idxmin = np.argmin(dSdEUp)
        self.dSdELimUp = dSdEUp[dSdEUp>0].min()+self.dSdELim
        FMTidxs=[]
        FMT=[]
        for Sel,shift in zip([range(0,idxmin),range(idxmin,dSdEUp.shape[0])],[0,idxmin]):
            Diff4lim = dSdEUp[Sel]-self.dSdELimUp
            #FMTidxs.append(np.where(np.abs(Diff4lim)==min(np.abs(Diff4lim)))[0][0]+shift)
            #if (Diff4lim<0).all():
            #    FMTidxs.append(dSdEUp.shape[0]-1)
            #Claude
            if len(Diff4lim)==0 or (Diff4lim<0).all():
                FMTidxs.append(0 if shift==0 else dSdEUp.shape[0]-1)
            else:
                FMTidxs.append(np.where(np.abs(Diff4lim)==min(np.abs(Diff4lim)))[0][0]+shift)
            FMT.append([strainUp[FMTidxs[-1]],stressUp[FMTidxs[-1]]])
            
        midstrain = np.mean([fmt[0] for fmt in FMT])
        Diff4lim = strainUp-midstrain
        idx = np.where(np.abs(Diff4lim)==min(np.abs(Diff4lim)))[0][0]
        lengthUp=strainUp[FMTidxs[1]]-strainUp[FMTidxs[0]]
        FMTplateau = {'length':lengthUp,'midstrain':strainUp[idx],'midstress':stressUp[idx],'meandSdE':dSdEUp[FMTidxs[0]:FMTidxs[1]].mean()}


        #plateau identification for unloading
        dSdEDown = self.deriv(stressDown,strainDown)
        dSdE = np.hstack((dSdEUp,dSdEDown))
        idxmin = np.argmin(dSdEDown)
        self.dSdELimDown = dSdEDown[dSdEDown>0].min()+self.dSdELim
        RMTidxs=[]
        RMT=[]

        #for Sel,shift in zip([range(0,idxmin),range(idxmin,dSdEDown.shape[0])],[0,idxmin]):
        #    Diff4lim = dSdEDown[Sel]-self.dSdELimDown
        #    RMTidxs.append(np.where(np.abs(Diff4lim)==min(np.abs(Diff4lim)))[0][0]+shift)     
        #    RMT.append([strainDown[RMTidxs[-1]],stressDown[RMTidxs[-1]]])  
        #Claude
        for Sel,shift in zip([range(0,idxmin),range(idxmin,dSdEDown.shape[0])],[0,idxmin]):
            Diff4lim = dSdEDown[Sel]-self.dSdELimDown
            if len(Diff4lim)==0 or (Diff4lim<0).all():
                # slope never reaches the threshold on this side -> plateau runs to the branch end
                RMTidxs.append(0 if shift==0 else dSdEDown.shape[0]-1)
            else:
                RMTidxs.append(np.where(np.abs(Diff4lim)==min(np.abs(Diff4lim)))[0][0]+shift)
            RMT.append([strainDown[RMTidxs[-1]],stressDown[RMTidxs[-1]]])

        midstrain = np.mean([rmt[0] for rmt in RMT])
        Diff4lim = strainDown-midstrain
        idx = np.where(np.abs(Diff4lim)==min(np.abs(Diff4lim)))[0][0]
        lengthDown=strainDown[RMTidxs[0]]-strainDown[RMTidxs[1]]
        RMTplateau = {'length':lengthDown,'midstrain':strainDown[idx],'midstress':stressDown[idx],'meandSdE':dSdEDown[RMTidxs[0]:RMTidxs[1]].mean()}

       
        titles= 'cycle stress strain stressUp strainUp stressDown strainDown dSdEUp dSdEUpLim dSdEDown FMTidxs FMT RMTidxs RMT'.split()
        cdata = {}
        cdata['load']={}
        cdata['unload']={}
        cdata['FMT']={}
        cdata['RMT']={}
        for key3 in ['FMT','RMT']:
            for key1, key2 in zip(['start','end'],[0,1]):
                if key3=='FMT':
                    data=[FMT,FMTidxs]
                else:
                    data=[RMT,RMTidxs]
                cdata[key3][key1]=data[0][key2]
                cdata[key3][f'{key1}idx']=data[1][key2]
        cdata['FMT']['plateau'] = FMTplateau
        cdata['RMT']['plateau'] = RMTplateau
        cdata['MT']={}
        cdata['MT']['hystwidth'] = FMTplateau['midstress']-RMTplateau['midstress']
        cdata['MT']['hyst'] = np.trapezoid(stress,strain/100)

        cdata['complete']={}
        cdata['cycle'] = cycle
        for key1 in ['strain','stress','dSdE','unrecoveredStrain']:
            exec(f"cdata['complete']['{key1}'] = {key1}")
            for key2 in ['Up','Down']:
                if key2=='Up':
                    exec(f"cdata['load']['{key1}'] = {key1}{key2}")
                    #exec(f"cdata['load']['MT'] = {key1}{key2}")
                else:
                    exec(f"cdata['unload']['{key1}'] = {key1}{key2}")
        cdata['complete']['unrecoveredStrainInc'] = unrecoveredStrainDown-unrecoveredStrainUp

        return cdata
    def deriv(self,stress,strain):
        if strain[0]>strain[-1]:
            dSdE = dxdt(stress[::-1], strain[::-1], kind=self.kind, left=self.left, right=self.right, order=self.order,iwindow=self.iwindow)[::-1]
        else:
            dSdE = dxdt(stress, strain, kind=self.kind, left=self.left, right=self.right, order=self.order,iwindow=self.iwindow)
        return dSdE
