"""import pandas as pd
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
print(Data)"""
"""import pandas as pd
arr={
    "Name":["Anass","Ghayth","Adem","Yassine"]
    ,"Age":[15,19,20,25]
    ,"Favourite_team":["psg","Barca","Barca","Real"]
}

new_arr=pd.DataFrame(arr,index=["First_Person","Second-Person","Third_Person","Fourth_Person"])
print(new_arr)
print(new_arr.loc["First_Person"])
print(new_arr["Name"])
print(new_arr.loc["First_Person","Favourite_team"])
added_row=pd.DataFrame({
    "Name":["ALA","raslen"]
    ,"Age":[37,19]
    ,"city":["Barcelone","London"]
    ,"married":[1,0]
})
new_arr=pd.concat([new_arr,added_row])"""
"""import pandas as pd
Data=[17,18,25,30]
print(Data)
Data=pd.Series(Data,index=["Ghayth","Abidli","Anass","Raslen"])
print("Data as series are")
print(Data)
print(Data.loc["Ghayth"])
print(Data.iloc[1])
print(Data[Data>=18])
"""
"""
import pandas as pd
Dict={
    "Name":["Hamza","Talel","Raslen","Mouhib"]
    ,"Nationality":["Tunisia","Romania","Tunisia","Algeria"]
    ,"Age":[27,18,55,34]
}
Dict=pd.DataFrame(Dict)
print("Initial DataFrame")
print(Dict)
Dict.index=["G","h",3,4]
print(Dict.loc["G"])
print(Dict["Name"])
print(Dict.loc["G","Name"])
new_dicti={
    "Name":["Aziz","Aniss"]
    ,"Nationality":["Mar","Nigeria"]
    ,"Age":[31,26]
}
new_dicti=pd.DataFrame(new_dicti,index=[14,"True"])
Dict=pd.concat([Dict,new_dicti])
print(Dict)
Dict["prenames"]=["Ab",17,30,77,78,33]
print(Dict)
"""
import pandas as pd
Data={
    "Name":["Ghayth","Aymen","Tarek"]
    ,"Age":[27,35,22]
    ,"Score":[50,70,10]
}
New_Data=pd.DataFrame(Data,index=["Student_1","Student_2","Student_3"])
print("Data is")
print(New_Data)
New_Data.index=["Student4","Student3","Student2"]
print(New_Data.loc["Student4"])
print(New_Data.iloc[0])
print("----------------")
print(New_Data["Age"])
New_Row=pd.DataFrame({
    "Name":["Anass","Amine"]
    ,"Age":[25,35]
    ,"Score":[65,25]
})
New_Data=pd.concat([New_Data,New_Row])
print("----------------")
print(New_Data)
New_Data["Married"]=[1,0,0,0,1]
print("----------------")
print(New_Data)






