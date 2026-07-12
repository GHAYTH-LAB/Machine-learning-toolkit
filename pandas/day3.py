import pandas as pd
#creating a Dictionnary named Data
Data={
    "Name":["Ghayth","ALBERT","Yassine"],
    "age":[19,30,22],
    "job":["swe engeneer","teacher","police"]
}
#convert a dictionnary into a DataFrame
df=pd.DataFrame(Data)
print(df)
#changing the row indexes 
df.index=["First_employee","Second_employee","Third_employee"]
print(df)
#adding New Column named married
df["married"]=[0,1,0]
print(df)
print("---------------------")
#printing the row that contains Second employee label
print(df.loc["Second_employee"])
print("---------------------")
#adding a new column named Score
df["score"]=[15,20,25]
print(df)
#adding a new row with an index
New_Row=pd.DataFrame([{"Name":"houssem"
         ,"age":27
         ,"job":"instructor"
         ,"married":1
         ,"score":37
         }], index=["Fourth_employee"])
df=pd.concat([df,New_Row])
print("---------------------")
print("final DataFrame is : ")
print(df)
