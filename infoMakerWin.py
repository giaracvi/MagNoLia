import os as os
os.chdir(os.path.abspath(""))

#Initiallizing variables
exec(open("infoMaker.py").read())

#Deleting existing experiment variables
if ('exp' in locals()) or ('exp' in globals()):
    del exp  
    
#Initiallizing experiment variables
exp = Experiment()