from DAPI_Segmentation import DAPI_Segmentation
from ChAT_Segmentation import ChAT_Segmentation
from Overlapping import Overlapping

#-----------------------------------------------
#Initiallizing experiment variables
exec(open("infoMakerWin.py").read())


#-----------------------------------------------
#Define which modules should be executed (0 = no, 1 = yes)
run_flags = {
"DAPI": 1,
"ChAT": 1,
"Overlap": 1}
    
#-----------------------------------------------
#Executing the modules
if (run_flags["DAPI"]):    
    print("\n--------------------------");
    print("\n-----DAPI Segmentation----");
    print("\n--------------------------");
    resultDAPI = DAPI_Segmentation(exp);

if (run_flags["ChAT"]):
    print("\n--------------------------");
    print("\n----ChAT Segmentation-----");
    print("\n--------------------------");
    resultChAT = ChAT_Segmentation(exp);

if (run_flags["Overlap"]):
    print("\n--------------------------");
    print("\n-------Overlapping--------");
    print("\n--------------------------");
    resultOverlapping = Overlapping(exp);