import pandas as pd
Data={
    "Name":["ghayth","Anass","Talel","Raslen","Ahmed","Dali"]
    ,"age":[20,15,21,19,28,36]
    ,"city":["Jendouba","Kef","BUCAREST","HEIDELBERG","Nabeul","Sousse"]
    }
Data=pd.DataFrame(Data,index=["First_Person","Second_Person","Third_Person","Fourth_Person","Fifth_Person","Sixth_Person"])
print(Data)