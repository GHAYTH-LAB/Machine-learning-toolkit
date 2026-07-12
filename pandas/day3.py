import pandas as pd
Data={
    "Name":["ghayth","Anass","Talel","Raslen","Ahmed","Dali"]
    ,"age":[20,15,21,19,28,36]
    ,"city":["Jendouba","Kef","BUCAREST","HEIDELBERG","Nabeul","Sousse"]
    }
Data=pd.DataFrame(Data,index=["First_Person","Second_Person","Third_Person","Fourth_Person","Fifth_Person","Sixth_Person"])
print(Data)
print(Data.loc["First_Person"])
print(Data["age"])
Data["married"]=[1,0,1,1,0,0]
New_row=pd.DataFrame([{
    "Name":"Hamza"
    ,"age":22
    ,"city":"Sfax"
    ,"married":0
}],index=["SEVENTH PERSON"])
Data=pd.concat([Data,New_row])
Data.loc["Ninth_person"]={
    "Name":"hazem"
    ,"age":45
    ,"city":"MONASTIR"
    ,"married":1
}
print(Data)